/*-
 * Copyright 2020-2022 QuPath developers, University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.biop.cellpose;

import com.google.gson.Gson;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.analysis.features.ObjectMeasurements.Compartments;
import qupath.lib.analysis.features.ObjectMeasurements.Measurements;
import qupath.lib.images.servers.ColorTransforms;
import qupath.lib.images.servers.ColorTransforms.ColorTransform;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.io.GsonTools;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.scripting.QP;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps;

import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.function.Function;

import static qupath.ext.biop.cellpose.OpCreators.TileOpCreator;

/**
 * Cell detection based on the following method:
 * <pre>
 *   Uwe Schmidt, Martin Weigert, Coleman Broaddus, and Gene Myers.
 *     "Cell Detection with Star-convex Polygons."
 *   <i>International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)</i>, Granada, Spain, September 2018.
 * </pre>
 * See the main repo at <a href="https://github.com/mpicbg-csbd/stardist">...</a>
 * <p>
 * Very much inspired by stardist-imagej at <a href="https://github.com/mpicbg-csbd/stardist-imagej">...</a> but re-written from scratch to use OpenCV and
 * adapt the method of converting predictions to contours (very slightly) to be more QuPath-friendly.
 * <p>
 * Models are expected in the same format as required by the Fiji plugin, or converted to a frozen .pb file for use with OpenCV.
 *
 * @author Pete Bankhead (this implementation, but based on the others)
 */
public class CellposeBuilder {

    private static final Logger logger = LoggerFactory.getLogger(CellposeBuilder.class);
    private final transient CellposeSetup cellposeSetup;

    private final List<ImageOp> preprocessing = new ArrayList<>();
    private final LinkedHashMap<String, String> cellposeParameters = new LinkedHashMap<>();
    // Cellpose Related Options
    private String modelNameOrPath;
    // Cellpose Training options
    private transient File modelDirectory = null;
    // QuPath Object handling options
    private ColorTransform[] channels = new ColorTransform[0];
    private Double cellExpansion = Double.NaN;
    private Double cellConstrainScale = Double.NaN;
    private Boolean ignoreCellOverlaps = Boolean.FALSE;
    private Double pixelSize = Double.NaN;
    private Integer tileWidth = 1024;
    private Integer tileHeight = 1024;
    private PathClass globalPathClass = PathClass.getNullClass();
    private Boolean measureShape = Boolean.FALSE;
    private Boolean constrainToParent = Boolean.TRUE;
    private Function<ROI, PathObject> creatorFun;
    private Collection<Compartments> compartments = Arrays.asList(Compartments.values());
    private Collection<Measurements> measurements;
    private transient boolean saveBuilder;
    private transient String builderName;
    private Integer overlap = null;
    private double simplifyDistance = 1.4;
    private TileOpCreator globalPreprocessing;
    private int nThreads = -1;

    private transient File tempDirectory = null;
    private transient File groundTruthDirectory = null;

    private ImageOp extendChannelOp = null;

    private boolean doReadResultsAsynchronously = false;
    private boolean useGPU = true;
    private boolean useTestDir = true;
    private boolean saveTrainingImages = true;
    private boolean useCellposeSAM = false;
    private String outputModelName;

    /**
     * can create a cellpose builder from a serialized JSON version of this builder.
     *
     * @param builderFile the path to a serialized JSON builder made with {@link #saveBuilder(String)}
     */
    protected CellposeBuilder(File builderFile) {

        // Need to know setup options, which are transient
        this.cellposeSetup = CellposeSetup.getInstance();

        Gson gson = GsonTools.getInstance();
        try {
            gson.fromJson(new FileReader(builderFile), CellposeBuilder.class);
            logger.info("Builder parameters loaded from {}", builderFile);
        } catch (FileNotFoundException e) {
            logger.error("Could not load builder from " + builderFile.getAbsolutePath(), e);
        }
    }

    public CellposeBuilder extendChannelOp(ImageOp extendChannelOp) {
        this.extendChannelOp = extendChannelOp;
        return this;
    }
    /**
     * Build a cellpose model by providing a string which can be the name of a pretrained model or a path to a custom model
     *
     * @param modelPath the model name or path
     */
    protected CellposeBuilder(String modelPath) {

        // Initialize all CellposeBuilderOld
        this.modelNameOrPath = modelPath;

        // Need to know setup options in order to guide the user in case of version inconsistency
        this.cellposeSetup = CellposeSetup.getInstance();

    }

    /**
     * overwrite use GPU
     * @param useGPU add or remove the option
     * @return this builder
     */
    public CellposeBuilder useGPU( boolean useGPU ) {
        this.useGPU = useGPU;

        return this;
    }

    /**
     * overwrite useTestDir
     * @param useTestDir add or remove the option
     * @return this builder
     */
    public CellposeBuilder useTestDir( boolean useTestDir ) {
        this.useTestDir = useTestDir;

        return this;
    }

    /**
     * overwrite saveTrainingImages
     * @param saveTrainingImages false to not resave training images
     * @return this builder
     */
    public CellposeBuilder saveTrainingImages( boolean saveTrainingImages ) {
        this.saveTrainingImages = saveTrainingImages;

        return this;
    }

    /**
     * Specify the training directory
     *
     */
    public CellposeBuilder groundTruthDirectory(File groundTruthDirectory) {
        this.groundTruthDirectory = groundTruthDirectory;
        return this;
    }

    /**
     * Specify the temporary directory
     */
    public CellposeBuilder tempDirectory(File trainingDirectory) {
        this.tempDirectory = trainingDirectory;
        return this;
    }

    /**
     * Specify the number of threads to use for processing.
     * If you encounter problems, setting this to 1 may help to resolve them by preventing
     * multithreading.
     *
     * @param nThreads the number of threads to use
     * @return this builder
     */
    public CellposeBuilder nThreads(int nThreads) {
        this.nThreads = nThreads;
        return this;
    }

    /**
     * Use an asynchronous method to read the results from the cellpose as it writes files
     * Can result in faster processing. !!EXPERIMENTAL!!
     */
    public CellposeBuilder readResultsAsynchronously() {
        this.doReadResultsAsynchronously = true;
        return this;
    }

    /**
     * Resolution at which the cell detection should be run.
     * The units depend upon the {@link PixelCalibration} of the input image.
     * <p>
     * The default is to use the full resolution of the input image.
     * <p>
     * For an image calibrated in microns, the recommended default is approximately 0.5.
     *
     * @param pixelSize Pixel size in microns for the analysis
     * @return this builder
     */
    public CellposeBuilder pixelSize(double pixelSize) {
        this.pixelSize = pixelSize;
        return this;
    }

    /**
     * Add preprocessing operations, if required.
     *
     * @param ops a series of ImageOps to apply to the input image
     * @return this builder
     */
    public CellposeBuilder preprocess(ImageOp... ops) {
        Collections.addAll(this.preprocessing, ops);
        return this;
    }

    /**
     * Add an {@link TileOpCreator} to generate preprocessing operations based upon the
     * entire image, rather than per tile.
     * <p>
     * Note that only a single such operation is permitted, which is applied after
     * channel extraction but <i>before</i> any other preprocessing.
     * <p>
     * The intended use is with {@link OpCreators#imageNormalizationBuilder()} to perform
     * normalization based upon percentiles computed across the image, rather than per tile.
     *
     * @param global preprocessing operation
     * @return this builder
     */
    public CellposeBuilder preprocessGlobal(TileOpCreator global) {
        this.globalPreprocessing = global;
        return this;
    }

    /**
     * Customize the extent to which contours are simplified.
     * Simplification reduces the number of vertices, which in turn can reduce memory requirements and
     * improve performance.
     * <p>
     * Implementation note: this currently uses the Visvalingam-Whyatt algorithm.
     *
     * @param distance simplify distance threshold; set &le; 0 to turn off additional simplification
     * @return this builder
     */
    public CellposeBuilder simplify(double distance) {
        this.simplifyDistance = distance;
        return this;
    }

    /**
     * Specify channels. Useful for detecting nuclei for one channel
     * within a multi-channel image, or potentially for trained models that
     * support multi-channel input.
     *
     * @param channels 0-based indices of the channels to use
     * @return this builder
     */
    public CellposeBuilder channels(int... channels) {
        return channels(Arrays.stream(channels)
                .mapToObj(c -> ColorTransforms.createChannelExtractor(c))
                .toArray(ColorTransform[]::new));
    }

    /**
     * Specify channels by name. Useful for detecting nuclei for one channel
     * within a multi-channel image, or potentially for trained models that
     * support multi-channel input.
     *
     * @param channels 0-based indices of the channels to use
     * @return this builder
     */
    public CellposeBuilder channels(String... channels) {
        return channels(Arrays.stream(channels)
                .map(c -> ColorTransforms.createChannelExtractor(c))
                .toArray(ColorTransform[]::new));
    }

    /**
     * Define the channels (or color transformers) to apply to the input image.
     * <p>
     * This makes it possible to supply color deconvolved channels, for example.
     *
     * @param channels the channels to use
     * @return this builder
     */
    public CellposeBuilder channels(ColorTransform... channels) {
        this.channels = channels.clone();
        return this;
    }

    /**
     * Amount by which to expand detected nuclei to approximate the cell area.
     * Units are the same as for the {@link PixelCalibration} of the input image.
     * <p>
     * Warning! This is rather experimental, relying heavily on JTS and a convoluted method of
     * resolving overlaps using a Voronoi tessellation.
     * <p>
     * In short, be wary.
     *
     * @param distance expansion distance in microns
     * @return this builder
     */
    public CellposeBuilder cellExpansion(double distance) {
        this.cellExpansion = distance;
        return this;
    }

    /**
     * Constrain any cell expansion defined using {@link #cellExpansion(double)} based upon
     * the nucleus size. Only meaningful for values &gt; 1; the nucleus is expanded according
     * to the scale factor, and used to define the maximum permitted cell expansion.
     *
     * @param scale
     * @return this builder
     */
    public CellposeBuilder cellConstrainScale(double scale) {
        this.cellConstrainScale = scale;
        return this;
    }

    /**
     * Create annotations rather than detections (the default).
     * If cell expansion is not zero, the nucleus will be included as a child object.
     *
     * @return this builder
     */
    public CellposeBuilder createAnnotations() {
        this.creatorFun = r -> PathObjects.createAnnotationObject(r);
        return this;
    }

    /**
     * Request that a classification is applied to all created objects.
     *
     * @param pathClass the classification to give to all detected PathObjects
     * @return this builder
     */
    public CellposeBuilder classify(PathClass pathClass) {
        this.globalPathClass = pathClass;
        return this;
    }

    /**
     * Request that a classification is applied to all created objects.
     * This is a convenience method that get a {@link PathClass} from a String representation.
     *
     * @param pathClassName the classification to give to all detected PathObjects as a String
     * @return this builder
     */
    public CellposeBuilder classify(String pathClassName) {
        return classify(PathClass.fromString(pathClassName, null));
    }

    /**
     * If true, ignore overlaps when computing cell expansion.
     *
     * @param ignore
     * @return this builder
     */
    public CellposeBuilder ignoreCellOverlaps(boolean ignore) {
        this.ignoreCellOverlaps = ignore;
        return this;
    }

    /**
     * If true, constrain nuclei and cells to any parent annotation (default is true).
     *
     * @param constrainToParent
     * @return this builder
     */
    public CellposeBuilder constrainToParent(boolean constrainToParent) {
        this.constrainToParent = constrainToParent;
        return this;
    }

    /**
     * Request default intensity measurements are made for all available cell compartments.
     *
     * @return this builder
     */
    public CellposeBuilder measureIntensity() {
        this.measurements = Arrays.asList(
                Measurements.MEAN,
                Measurements.MEDIAN,
                Measurements.MIN,
                Measurements.MAX,
                Measurements.STD_DEV);
        return this;
    }

    /**
     * Request specified intensity measurements are made for all available cell compartments.
     *
     * @param measurements the measurements to make
     * @return this builder
     */
    public CellposeBuilder measureIntensity(Collection<Measurements> measurements) {
        this.measurements = new ArrayList<>(measurements);
        return this;
    }

    /**
     * Request shape measurements are made for the detected cell or nucleus.
     *
     * @return this builder
     */
    public CellposeBuilder measureShape() {
        this.measureShape = true;
        return this;
    }

    /**
     * Specify the compartments within which intensity measurements are made.
     * Only effective if {@link #measureIntensity()} and {@link #cellExpansion(double)} have been selected.
     *
     * @param compartments cell compartments for intensity measurements
     * @return this builder
     */
    public CellposeBuilder compartments(Compartments... compartments) {
        this.compartments = Arrays.asList(compartments);
        return this;
    }

    /**
     * Size in pixels of a tile used for detection.
     * Note that tiles are independently normalized, and therefore tiling can impact
     * the results. Default is 1024.
     *
     * @param tileSize the width and height of each tile for exporting images
     * @return this builder
     */
    public CellposeBuilder tileSize(int tileSize) {
        return tileSize(tileSize, tileSize);
    }

    /**
     * Size in pixels of a tile used for detection.
     * Note that tiles are independently normalized, and therefore tiling can impact
     * the results. Default is 1024.
     *
     * @param tileWidth the width of each tile for exporting images
     * @param tileHeight the height of each tile for exporting images
     * @return this builder
     */
    public CellposeBuilder tileSize(int tileWidth, int tileHeight) {
        this.tileWidth = tileWidth;
        this.tileHeight = tileHeight;
        return this;
    }

    /**
     * Apply percentile normalization separately to the input image channels.
     * <p>
     * Note that this can be used in combination with {@link #preprocess(ImageOp...)},
     * in which case the order in which the operations are applied depends upon the order
     * in which the methods of the builder are called.
     * <p>
     * Warning! This is applied on a per-tile basis. This can result in artifacts and false detections
     * without background/constant regions.
     * Consider using {@link #inputAdd(double...)} and {@link #inputScale(double...)} as alternative
     * normalization strategies, if appropriate constants can be determined to apply globally.
     *
     * @param min minimum percentile
     * @param max maximum percentile
     * @return this builder
     * @see #normalizePercentiles(double, double, boolean, double)
     */
    public CellposeBuilder normalizePercentiles(double min, double max) {
        return normalizePercentiles(min, max, true, 0.0);
    }

    /**
     * Apply percentile normalization to the input image channels, or across all channels jointly.
     * <p>
     * Note that this can be used in combination with {@link #preprocess(ImageOp...)},
     * in which case the order in which the operations are applied depends upon the order
     * in which the methods of the builder are called.
     * <p>
     * Warning! This is applied on a per-tile basis. This can result in artifacts and false detections
     * without background/constant regions.
     * Consider using {@link #inputAdd(double...)} and {@link #inputScale(double...)} as alternative
     * normalization strategies, if appropriate constants can be determined to apply globally.
     *
     * @param min        minimum percentile
     * @param max        maximum percentile
     * @param perChannel if true, normalize each channel separately; if false, normalize channels jointly
     * @param eps        small constant to apply
     * @return this builder
     * @since v0.4.0
     */
    public CellposeBuilder normalizePercentiles(double min, double max, boolean perChannel, double eps) {
        this.preprocessing.add(ImageOps.Normalize.percentile(min, max, perChannel, eps));
        return this;
    }


    /**
     * Add an offset as a preprocessing step.
     * Usually the value will be negative. Along with {@link #inputScale(double...)} this can be used as an alternative (global) normalization.
     * <p>
     * Note that this can be used in combination with {@link #preprocess(ImageOp...)},
     * in which case the order in which the operations are applied depends upon the order
     * in which the methods of the builder are called.
     *
     * @param values either a single value to add to all channels, or an array of values equal to the number of channels
     * @return this builder
     * @see #inputSubtract(double...)
     * @see #inputScale(double...)
     */
    public CellposeBuilder inputAdd(double... values) {
        this.preprocessing.add(ImageOps.Core.add(values));
        return this;
    }

    /**
     * Subtract an offset as a preprocessing step.
     * <p>
     * Note that this can be used in combination with {@link #preprocess(ImageOp...)},
     * in which case the order in which the operations are applied depends upon the order
     * in which the methods of the builder are called.
     *
     * @param values either a single value to subtract from all channels, or an array of values equal to the number of channels
     * @return this builder
     * @see #inputAdd(double...)
     * @see #inputScale(double...)
     * @since v0.4.0
     */
    public CellposeBuilder inputSubtract(double... values) {
        this.preprocessing.add(ImageOps.Core.subtract(values));
        return this;
    }

    /**
     * Multiply by a scale factor as a preprocessing step.
     * Along with {@link #inputAdd(double...)} this can be used as an alternative (global) normalization.
     * <p>
     * Note that this can be used in combination with {@link #preprocess(ImageOp...)},
     * in which case the order in which the operations are applied depends upon the order
     * in which the methods of the builder are called.
     *
     * @param values either a single value to add to all channels, or an array of values equal to the number of channels
     * @return this builder
     * @see #inputAdd(double...)
     * @see #inputSubtract(double...)
     */
    public CellposeBuilder inputScale(double... values) {
        this.preprocessing.add(ImageOps.Core.multiply(values));
        return this;
    }

    //  CELLPOSE OPTIONS
    // ------------------

    /**
     * Generic means of adding a cellpose parameter
     *
     * @param flagName  the name of the flag, eg. "save_every"
     * @param flagValue the value that is linked to the flag, eg. "20". Can be an empty string or null if it is not needed
     * @return this builder
     * @see <a href="https://cellpose.readthedocs.io/en/latest/command.html#input-settings">the cellpose documentation</a> for a list of available flags
     */
    public CellposeBuilder addParameter(String flagName, String flagValue) {
        this.cellposeParameters.put(flagName, flagValue);
        return this;

    }

    /**
     * Generic means of adding a cellpose parameter
     *
     * @param flagName the name of the flag, eg. "save_every"	 * @param flagName the name of the flag, eg. "save_every"
     * @return
     * @see <a href="https://cellpose.readthedocs.io/en/latest/command.html#input-settings">the cellpose documentation</a> for a list of available flags
     */
    public CellposeBuilder addParameter(String flagName) {
        addParameter(flagName, null);
        return this;
    }

    /**
     * Use Omnipose implementation: Adds --omni flag to command
     *
     * @return this builder
     */
    public CellposeBuilder useOmnipose() {
        if (this.cellposeSetup.getOmniposePythonPath().isEmpty())
            logger.warn("Omnipose environment path not set. Using cellpose path instead.");
        addParameter("omni");
        return this;
    }

    /**
     * Use Omnipose implementation: Adds --omni flag to command
     *
     * @return this builder
     */
    public CellposeBuilder useCellposeSAM() {
        if (this.cellposeSetup.getCellposeSAMPythonPath().isEmpty())
            logger.warn("Cellpose SAM environment path not set. Using cellpose path instead.");
        this.useCellposeSAM = true;
        return this;
    }

    /**
     * Exclude on edges. Adds --exclude_on_edges flag to cellpose command
     *
     * @return this builder
     */
    public CellposeBuilder excludeEdges() {
        addParameter("exclude_on_edges");
        return this;
    }

    /**
     * Explicitly set the cellpose channels manually. This corresponds to --chan and --chan2
     * @param channel1 --chan value passed to cellpose/omnipose
     * @param channel2 --chan2 value passed to cellpose/omnipose
     * @return this builder
     */
    public CellposeBuilder cellposeChannels(Integer channel1, Integer channel2) {
        addParameter("chan", channel1.toString());
        addParameter("chan2", channel2.toString());
        return this;
    }


    /**
     * cellprob threshold, default is 0, decrease to find more and larger masks
     *
     * @param threshold cell/nuclei masks threshold, between -6 and +6
     * @return this builder
     * @deprecated use {@link CellposeBuilder#cellprobThreshold(Double)}
     */
    @Deprecated
    public CellposeBuilder maskThreshold(Double threshold) {
        logger.warn("'maskThreshold() is deprecated. Replace with cellprobThreshold() in the builder.");
        addParameter("cellprob_threshold", threshold.toString());
        return this;
    }

    /**
     * /**
     * cellprob threshold, default is 0, decrease to find more and larger masks
     *
     * @param threshold cell/nuclei masks threshold, between -6 and +6
     * @return this builder
     */
    public CellposeBuilder cellprobThreshold(Double threshold) {
        addParameter("cellprob_threshold", threshold.toString());
        return this;
    }

    /**
     * Flow error threshold, 0 turns off this optional QC step. Default: 0.
     *
     * @param threshold flow threshold (default 0.0)
     * @return this builder
     */
    public CellposeBuilder flowThreshold(Double threshold) {
        addParameter("flow_threshold", threshold.toString());
        return this;
    }

    /**
     * The estimated diameter of the objects to detect. Cellpose will further downsample (or upsample) the images in order to match
     * the diamteter corresponding to the model being used
     *
     * @param diameter in pixels
     * @return this builder
     */
    public CellposeBuilder diameter(Double diameter) {
        addParameter("diameter", diameter.toString());
        return this;
    }

    // CELLPOSE TRAINING OPTIONS

    /**
     * Define the directory where the newly trained model should be saved
     *
     * @param modelDir a directory (does not need to exist yet) where to save the cellpose model
     * @return this Builder
     */
    public CellposeBuilder modelDirectory(File modelDir) {
        this.modelDirectory = modelDir;
        return this;
    }

    /**
     * Defines the number of epochs for training
     *
     * @param nEpochs number of epochs for training
     * @return this Builder
     */
    public CellposeBuilder epochs(Integer nEpochs) {
        addParameter("n_epochs", nEpochs.toString());
        return this;
    }

    /**
     * Defines the learning rate
     *
     * @param learningRate learning rate per epoch
     * @return this Builder
     */
    public CellposeBuilder learningRate(Double learningRate) {
        addParameter("learning_rate", learningRate.toString());
        return this;
    }

    /**
     * Defines the batch size for training
     *
     * @param batchSize batch size for training
     * @return this Builder
     */
    public CellposeBuilder batchSize(Integer batchSize) {
        addParameter("batch_size", batchSize.toString());
        return this;
    }

    /**
     * Excludes training data with less than n training masks (n labels)
     *
     * @param n minimum number of labels per training image
     * @return this builder
     */
    public CellposeBuilder minTrainMasks(Integer n) {
        addParameter("min_train_masks", n.toString());
        return this;
    }

    /**
     * Save this builder as a JSON file in order to be able to reuse it in place
     *
     * @param name // A name to append to the JSON file. Keep it meaningful for your needs
     * @return this builder
     */
    public CellposeBuilder saveBuilder(String name) {
        this.saveBuilder = true;
        this.builderName = name;
        return this;
    }

    /**
     * Set the overlap (in pixels) between tiles. This overlap should be larger than 2x the largest object you are
     * trying to segment
     *
     * @param overlap the overlap, in pixels
     * @return this builder
     */
    public CellposeBuilder setOverlap(int overlap) {
        this.overlap = overlap;
        return this;
    }

    /**
     * Convenience method to call global normalization for the dataset
     *
     * @param percentileMin  the min percentile 0-100
     * @param percentileMax  the max percentile 0-100
     * @param normDownsample a large downsample for the computation to be efficient over the whole image
     * @return this builder
     */
    public CellposeBuilder normalizePercentilesGlobal(double percentileMin, double percentileMax, double normDownsample) {

        TileOpCreator normOp = new OpCreators.ImageNormalizationBuilder().percentiles(percentileMin, percentileMax)
                .perChannel(true)
                .downsample(normDownsample)
                .useMask(true)
                .build();

        // Deactivate cellpose normalization
        this.noCellposeNormalization();

        // Add this operation to the preprocessing
        return this.preprocessGlobal(normOp);
    }

    /**
     * convenience method? to deactivate cellpose normalization.
     *
     * @return this builder
     */
    public CellposeBuilder noCellposeNormalization() {
        return this.addParameter("no_norm");
    }

    /**
     * Set the final model name. Setting this to "My Model" would lead to the final cellpose file being called
     * "My Model_yyyy-MM-dd_HH_mm.cpm" with the current timestamp and .cpm meaning Cellpose Model
     * @param outputName the prefix os the cellpose model name
     * @return this builder
     */
    public CellposeBuilder setOutputModelName(String outputName) {
        this.outputModelName = outputName;
        return this;
    }
    /**
     * Create a {@link Cellpose2D}, all ready for detection.
     *
     * @return a new {@link Cellpose2D} instance
     */
    public Cellpose2D build() {
        Cellpose2D cellpose = new Cellpose2D();

        // Give it the number of threads to use
        cellpose.nThreads = this.nThreads;

        cellpose.useGPU = this.useGPU;
        cellpose.useTestDir = this.useTestDir;
        cellpose.saveTrainingImages = this.saveTrainingImages;

        // Check the model. If it is a file, then it is a custom model
        File file = new File(this.modelNameOrPath);
        if (file.exists()) {
            logger.info("Provided model '{}' is a file. Assuming custom model", file);
        }

        // Assign model
        cellpose.model = this.modelNameOrPath;

        // Assign current cellpose extension settings
        cellpose.cellposeSetup = this.cellposeSetup;

        // Pick up info on project location and where the data will be stored for training and inference
        File quPathProjectDir = QP.getProject().getPath().getParent().toFile();

        // Prepare temp directory in case it was not set
        if (this.tempDirectory == null) {
            this.tempDirectory = new File(quPathProjectDir, "cellpose-temp");
        }

        // Prepare training directory in case it was not set
        if (this.groundTruthDirectory == null) {
            this.groundTruthDirectory = new File(quPathProjectDir, "cellpose-training");
        }

        if (this.modelDirectory == null) {
            this.modelDirectory = new File(quPathProjectDir, "models");
            this.modelDirectory.mkdirs();
        }

        cellpose.outputModelName = this.outputModelName;
        cellpose.modelDirectory = this.modelDirectory;
        cellpose.groundTruthDirectory = this.groundTruthDirectory;
        cellpose.tempDirectory = this.tempDirectory;
        cellpose.doReadResultsAsynchronously = this.doReadResultsAsynchronously;
        cellpose.useCellposeSAM = this.useCellposeSAM;

        cellpose.extendChannelOp = this.extendChannelOp;

        // TODO make compatible with --all_channels
        if (this.channels.length > 2) {
            logger.warn("You supplied {} channels, but Cellpose needs two channels at most. Keeping the first two",this.channels.length);
            this.channels = Arrays.copyOf(this.channels, 2);
        }

        cellpose.op = ImageOps.buildImageDataOp(this.channels);

        // these are all the cellpose parameters we wish to send to the command line.
        cellpose.parameters = this.cellposeParameters;

        cellpose.globalPreprocess = this.globalPreprocessing;
        cellpose.preprocess = new ArrayList<>(this.preprocessing);

        cellpose.pixelSize = this.pixelSize;
        cellpose.cellConstrainScale = this.cellConstrainScale;
        cellpose.cellExpansion = this.cellExpansion;
        cellpose.tileWidth = this.tileWidth;
        cellpose.tileHeight = this.tileHeight;
        cellpose.ignoreCellOverlaps = this.ignoreCellOverlaps;
        cellpose.measureShape = this.measureShape;
        cellpose.simplifyDistance = this.simplifyDistance;
        cellpose.constrainToParent = this.constrainToParent;
        cellpose.creatorFun = this.creatorFun;
        cellpose.globalPathClass = this.globalPathClass;
        cellpose.outputModelName = this.outputModelName;

        cellpose.compartments = new LinkedHashSet<>(this.compartments);

        if (this.measurements != null)
            cellpose.measurements = new LinkedHashSet<>(this.measurements);
        else
            cellpose.measurements = Collections.emptyList();

        // If overlap is set, then it takes precedence
        if (this.overlap != null) {
            cellpose.overlap = this.overlap;
        } else {
            if (this.cellposeParameters.containsKey("diameter")) { // Diameter was set
                double diameter = Double.parseDouble(this.cellposeParameters.get("diameter"));
                if (diameter == 0.0) {
                    cellpose.overlap = 30 * 2;
                    logger.info("Tile overlap was not set and diameter was set to {}. Will default to {} pixels overlap. Use `.setOverlap( int )` to modify overlap", diameter, cellpose.overlap);
                } else {
                    cellpose.overlap = (int) (diameter * 2);
                    logger.info("Tile overlap was not set, but diameter exists. Using provided diameter {} x 2: {} pixels overlap", diameter, cellpose.overlap);
                }
            } else { // Nothing was set, let's get lucky
                cellpose.overlap = 30 * 2;
                logger.info("Neither diameter nor overlap provided. Overlap defaulting to {} pixels. Use `.setOverlap( int )` to modify overlap", cellpose.overlap);
            }
        }
        logger.info("If tiling is necessary, {} pixels overlap will be taken between tiles", cellpose.overlap);

        // If we would like to save the builder we can do it here thanks to Serialization and lots of magic by Pete
        if (this.saveBuilder) {
            Gson gson = GsonTools.getInstance(true);

            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH'h'mm");
            LocalDateTime now = LocalDateTime.now();
            File savePath = new File(this.modelDirectory, this.builderName + "_" + dtf.format(now) + ".json");

            try {
                FileWriter fw = new FileWriter(savePath);
                gson.toJson(this, CellposeBuilder.class, fw);
                fw.flush();
                fw.close();
                logger.info("Cellpose Builder serialized and saved to {}", savePath);

            } catch (IOException e) {
                logger.error("Could not save builder to JSON file {}", savePath.getAbsolutePath(), e);
            }
        }

        return cellpose;

    }

}
