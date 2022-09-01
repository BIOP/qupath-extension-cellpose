package qupath.ext.biop.cellpose;

import com.google.gson.Gson;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.analysis.features.ObjectMeasurements;
import qupath.lib.gui.scripting.QPEx;
import qupath.lib.images.servers.ColorTransforms;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.images.servers.PixelType;
import qupath.lib.io.GsonTools;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.projects.Projects;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.scripting.QP;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps;

import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.function.Function;

/**
 * This class will contain all parameters that are usable by cellpose
 */
public class CellposeBuilder {

    private final transient static Logger logger = LoggerFactory.getLogger(CellposeBuilder.class);

    private String version;
    // Cellpose Related Options
    private String model;
    private transient CellposeSetup cellposeSetup;

    private Integer channel1 = 0; //GRAY
    private Integer channel2 = 0; // NONE

    private Boolean isInvert = Boolean.FALSE;

    private Boolean useOmnipose = Boolean.FALSE;
    private Boolean excludeEdges = Boolean.FALSE;
    private Boolean doCluster = Boolean.FALSE;
    private Boolean useGPU = Boolean.FALSE;


    private Double maskThreshold = Double.NaN;
    private Double flowThreshold = Double.NaN;
    private Double diameter = Double.NaN;

    // Cellpose Training options
    private File modelDirectory = null;
    private File trainDirectory = null;
    private File valDirectory = null;
    private Integer nEpochs = null;
    private Integer batchSize = null;
    private Double learningRate= Double.NaN;

    // QuPath Object handling options
    private ColorTransforms.ColorTransform[] channels = new ColorTransforms.ColorTransform[0];
    private Double cellExpansion = Double.NaN;
    private Double cellConstrainScale = Double.NaN;
    private Boolean ignoreCellOverlaps = Boolean.FALSE;
    private Double pixelSize = Double.NaN;
    private Integer tileWidth = 1024;
    private Integer tileHeight = 1024;
    private PathClass globalPathClass = PathClassFactory.getPathClassUnclassified();
    private Boolean measureShape = Boolean.FALSE;
    private Boolean constrainToParent = Boolean.TRUE;
    private Double iouThreshold = 0.1;

    // Check if these need to be here
    private Function<ROI, PathObject> creatorFun;
    private Collection<ObjectMeasurements.Compartments> compartments = Arrays.asList(ObjectMeasurements.Compartments.values());
    private Collection<ObjectMeasurements.Measurements> measurements;
    private List<ImageOp> ops = new ArrayList<>();

    private transient boolean saveBuilder;
    private transient String builderName;
    private double simplifyDistance = 0.0;
    private double normPercentileMin = -1.0;
    private double normPercentileMax = -1.0;
    private Integer overlap = null;

    /**
     * can create a cellpose builder from a serialized JSON version of this builder.
     *
     * @param builderFile the path to a serialized JSON builder made with {@link #saveBuilder(String)}
     */

    CellposeBuilder(File builderFile) {

        // Need to know setup options, which are transient
        this.cellposeSetup = CellposeSetup.getInstance();

        Gson gson = GsonTools.getInstance();
        try {
            gson.fromJson(new FileReader(builderFile), CellposeBuilder.class);
            logger.info("Builder parameters loaded from {}", builderFile);
        } catch (FileNotFoundException e) {
            logger.error("Could not load builder from "+builderFile.getAbsolutePath(), e);
        }
    }

    CellposeBuilder(String modelPath) {

        this.version = new CellposeExtension().getVersion().toString();

        // Initialize all CellposeBuilder
        this.model = modelPath;

        this.ops.add(ImageOps.Core.ensureType(PixelType.FLOAT32));

        // Need to know setup options in order to guide the user in case of version inconsistency
        this.cellposeSetup = CellposeSetup.getInstance();

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
     * Apply percentile normalization to the input image channels.
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
     */
    public CellposeBuilder normalizePercentiles(double min, double max) {
        this.normPercentileMin = min;
        this.normPercentileMax = max;

        this.ops.add(ImageOps.Normalize.percentile(min, max));
        this.ops.add(ImageOps.Core.clip(0.0,1.0));
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
                .mapToObj(ColorTransforms::createChannelExtractor)
                .toArray(ColorTransforms.ColorTransform[]::new));
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
                .map(ColorTransforms::createChannelExtractor)
                .toArray(ColorTransforms.ColorTransform[]::new));
    }

    /**
     * Define the channels (or color transformers) to apply to the input image.
     * <p>
     * This makes it possible to supply color deconvolved channels, for example.
     *
     * @param channels ColorTransform channels to use, typically only used internally
     * @return this builder
     */
    public CellposeBuilder channels(ColorTransforms.ColorTransform... channels) {
        this.channels = channels.clone();
        if (this.channels.length >= 2) {
            this.channel1 = 1;
            this.channel2 = 2;
        }
        return this;
    }

    /**
     * Add preprocessing operations, if required.
     *
     * @param ops series of ImageOps to apply to this server before saving the images
     * @return this builder
     */
    public CellposeBuilder preprocess(ImageOp... ops) {
        Collections.addAll(this.ops, ops);
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
     */
    public CellposeBuilder inputAdd(double... values) {
        this.ops.add(ImageOps.Core.add(values));
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
     */
    public CellposeBuilder inputScale(double... values) {
        this.ops.add(ImageOps.Core.subtract(values));
        return this;
    }

    /**
     * Size in pixels of a tile used for detection.
     * Note that tiles are independently normalized, and therefore tiling can impact
     * the results. Default is 1024.
     *
     * @param tileSize if the regions must be broken down, how large should the tiles be, in pixels (width and height)
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
     * @param tileWidth if the regions must be broken down, how large should the tiles be (width), in pixels
     * @param tileHeight if the regions must be broken down, how large should the tiles be (height), in pixels
     * @return this builder
     */
    public CellposeBuilder tileSize(int tileWidth, int tileHeight) {
        this.tileWidth = tileWidth;
        this.tileHeight = tileHeight;
        return this;
    }

        /**
         * Sets the channels to use by cellpose, in case there is an issue with the order or the number of exported channels
         * @param channel1 the main channel
         * @param channel2 the second channel (typically nuclei)
         * @return this builder
         */
    public CellposeBuilder cellposeChannels(int channel1, int channel2) {
        this.channel1 = channel1;
        this.channel2 = channel2;
        return this;
    }

    /**
     * Probability threshold to apply for detection, between 0 and 1.
     *
     * @param threshold probability threshold between 0 and 1 (default 0.5)
     * @return this builder
     */
    public CellposeBuilder maskThreshold(double threshold) {
        this.maskThreshold = threshold;
        return this;
    }

    /**
     * Flow threshold to apply for detection, between 0 and 1.
     *
     * @param threshold flow threshold (default 0.0)
     * @return this builder
     */
    public CellposeBuilder flowThreshold(double threshold) {
        this.flowThreshold = threshold;
        return this;
    }

    /**
     * The extimated diameter of the objects to detect. Cellpose will further downsample the images in order to match
     * their expected diameter
     *
     * @param diameter in pixels
     * @return this builder
     */
    public CellposeBuilder diameter(double diameter) {
        this.diameter = diameter;
        return this;
    }

    /**
     * Inverts the image channels within cellpose. Adds the --invert flag to the command
     * @return this builder
     */
    public CellposeBuilder invert() {
        this.isInvert = true;
        return this;
    }

    public CellposeBuilder simplify(double distance) {
        this.simplifyDistance = distance;
        return this;
    }

    /**
     * Use Omnipose implementation: Adds --omni flag to command
     *
     * @return this builder
     */
    public CellposeBuilder useOmnipose() {
        if( this.cellposeSetup.getVersion().equals(CellposeSetup.CellposeVersion.OMNIPOSE) ||
                this.cellposeSetup.getVersion().equals(CellposeSetup.CellposeVersion.CELLPOSE_1) ||
                this.cellposeSetup.getVersion().equals(CellposeSetup.CellposeVersion.CELLPOSE_2)) {
            this.useOmnipose = true;
        } else {
            logger.warn("--omni flag not available in {}", CellposeSetup.CellposeVersion.CELLPOSE);
        }
        return this;
    }

    /**
     * Exclude on edges. Adds --exclude_on_edges flag to command
     *
     * @return this builder
     */
    public CellposeBuilder excludeEdges() {
        if( this.cellposeSetup.getVersion().equals(CellposeSetup.CellposeVersion.OMNIPOSE) ) {
            this.excludeEdges = true;
        } else {
            logger.warn("--exclude_edges flag not available in {}", CellposeSetup.CellposeVersion.CELLPOSE);
        }
        return this;
    }

    /**
     * DBSCAN clustering. Reduces oversegmentation of thin features, adds --cluster flag to command.
     *
     * @return this builder
     */
    public CellposeBuilder clusterDBSCAN() {
        if( this.cellposeSetup.getVersion().equals(CellposeSetup.CellposeVersion.OMNIPOSE) ) {
            this.doCluster = true;
        } else {
            logger.warn("--cluster flag not available in {}", CellposeSetup.CellposeVersion.CELLPOSE);
        }
        return this;
    }

    /**
     * Use a GPU for prediction or training. Whether this works or nt will depend on your cellpose environment
     *
     * @return this Builder
     */
    public CellposeBuilder useGPU() {
        this.useGPU = Boolean.TRUE;
        return this;
    }

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
    public CellposeBuilder epochs(int nEpochs) {
        this.nEpochs = nEpochs;
        return this;
    }

    /**
     * Defines the learning rate
     *
     * @param learningRate learning rate per epoch
     * @return this Builder
     */
    public CellposeBuilder learningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    /**
     * Defines the batch size for training
     *
     * @param batchSize batch size for training
     * @return this Builder
     */
    public CellposeBuilder batchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    /**
     * Select the Intersection over Union (IoU) cutoff for excluding overlapping detections.
     * Default of 0.1 is good enough
     * <p>
     *
     * @param iouThreshold distance threshold
     * @return this builder
     */
    public CellposeBuilder iou(double iouThreshold) {
        this.iouThreshold = iouThreshold;
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
     * @param distance cell expansion distance in microns
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
     * @param scale a number to multiply each pixel in the image by
     * @return this builder
     */
    public CellposeBuilder cellConstrainScale(double scale) {
        this.cellConstrainScale = scale;
        return this;
    }

    /**
     * Request that a classification is applied to all created objects.
     *
     * @param pathClass the PathClass of all detections resulting from this run
     * @return this builder
     */
    public CellposeBuilder classify(PathClass pathClass) {
        this.globalPathClass = pathClass;
        return this;
    }

    /**
     * Request that a classification is applied to all created objects.
     * This is a convenience method that get a {@link PathClass} from  {@link PathClassFactory}.
     *
     * @param pathClassName the name of the PathClass for all detections
     * @return this builder
     */
    public CellposeBuilder classify(String pathClassName) {
        return classify(PathClassFactory.getPathClass(pathClassName, (Integer) null));
    }

    /**
     * If true, ignore overlaps when computing cell expansion.
     *
     * @param ignore ignore overlaps when computing cell expansion.
     * @return this builder
     */
    public CellposeBuilder ignoreCellOverlaps(boolean ignore) {
        this.ignoreCellOverlaps = ignore;
        return this;
    }

    /**
     * If true, constrain nuclei and cells to any parent annotation (default is true).
     *
     * @param constrainToParent constrain nuclei and cells to any parent annotation
     * @return this builder
     */
    public CellposeBuilder constrainToParent(boolean constrainToParent) {
        this.constrainToParent = constrainToParent;
        return this;
    }

    /**
     * Create annotations rather than detections (the default).
     * If cell expansion is not zero, the nucleus will be included as a child object.
     *
     * @return this builder
     */
    public CellposeBuilder createAnnotations() {
        this.creatorFun = PathObjects::createAnnotationObject;
        return this;
    }

    /**
     * Request default intensity measurements are made for all available cell compartments.
     *
     * @return this builder
     */
    public CellposeBuilder measureIntensity() {
        this.measurements = Arrays.asList(
                ObjectMeasurements.Measurements.MEAN,
                ObjectMeasurements.Measurements.MEDIAN,
                ObjectMeasurements.Measurements.MIN,
                ObjectMeasurements.Measurements.MAX,
                ObjectMeasurements.Measurements.STD_DEV);
        return this;
    }

    /**
     * Request specified intensity measurements are made for all available cell compartments.
     *
     * @param measurements the measurements to make
     * @return this builder
     */
    public CellposeBuilder measureIntensity(Collection<ObjectMeasurements.Measurements> measurements) {
        this.measurements = new ArrayList<>(measurements);
        return this;
    }

    /**
     * Request shape measurements are made for the detected cell or nucleus.
     *
     * @return this builder
     */
    public CellposeBuilder measureShape() {
        measureShape = true;
        return this;
    }

    /**
     * Specify the compartments within which intensity measurements are made.
     * Only effective if {@link #measureIntensity()} and {@link #cellExpansion(double)} have been selected.
     *
     * @param compartments cell compartments for intensity measurements
     * @return this builder
     */
    public CellposeBuilder compartments(ObjectMeasurements.Compartments... compartments) {
        this.compartments = Arrays.asList(compartments);
        return this;
    }

    /**
     * Save this builder as a JSON file in order to be able to reuse it in place
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
     * @param overlap the overlap, in pixels
     * @return this builder
     */
    public CellposeBuilder setOverlap( int overlap) {
        this.overlap = overlap;
        return this;
    }

    /**
     * Create a {@link Cellpose2D}, all ready for detection.
     *
     * @return a Cellpose2D object, ready to be run
     */
    public Cellpose2D build() {

        // The cellpose class that will run the detections and training
        Cellpose2D cellpose = new Cellpose2D();

        ArrayList<ImageOp> mergedOps = new ArrayList<>(ops);

        // Check the model. If it is a file, then it is a custom model
        File file = new File(model);
        if (file.isFile()) {
            logger.info("Provided model {} is a file. Assuming custom model", file);
        }

        // Pass all cellpose options in one go...
        cellpose.model = model;

        cellpose.cellposeSetup = cellposeSetup;

        // Add all operations (preprocessing, channel extraction and normalization)
        mergedOps.add(ImageOps.Core.ensureType(PixelType.FLOAT32));

        // Ensure there are only 2 channels at most
        if (channels.length > 2) {
            logger.warn("You supplied {} channels, but Cellpose needs two channels at most. Keeping the first two", channels.length);
            channels = Arrays.copyOf(channels, 2);
        }
        cellpose.op = ImageOps.buildImageDataOp(channels).appendOps(mergedOps.toArray(ImageOp[]::new));

        // CellPose accepts either one or two channels. This will help the final command
        cellpose.channel1 = channel1;
        cellpose.channel2 = channel2;

        if (maskThreshold.isNaN()) cellpose.maskThreshold = 0.0;
        else cellpose.maskThreshold = maskThreshold;

        if (flowThreshold.isNaN()) cellpose.flowThreshold = 0.4;
        else cellpose.flowThreshold = flowThreshold;

        cellpose.pixelSize = pixelSize;

        if (diameter.isNaN()) cellpose.diameter = 0.0;
        else cellpose.diameter = diameter;

        cellpose.simplifyDistance = simplifyDistance;

        cellpose.invert = isInvert;

        cellpose.doCluster = doCluster;
        cellpose.excludeEdges = excludeEdges;
        cellpose.useOmnipose = useOmnipose;
        cellpose.useGPU = useGPU;

        // Pick up info on project location
        File quPathProjectDir = QP.getProject().getPath().getParent().toFile();
        if (modelDirectory == null) {
            modelDirectory = new File(quPathProjectDir, "models");
        }

        modelDirectory.mkdirs();

        // Define training and validation directories inside the QuPath Project
        File groundTruthDirectory = new File(quPathProjectDir, "cellpose-training");

        File trainDirectory = new File(groundTruthDirectory, "train");
        File valDirectory = new File(groundTruthDirectory, "test");
        trainDirectory.mkdirs();
        valDirectory.mkdirs();

        // Cleanup a previous run
        try {
            FileUtils.cleanDirectory(trainDirectory);
            FileUtils.cleanDirectory(valDirectory);
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
        }
        cellpose.modelDirectory = modelDirectory;
        cellpose.trainDirectory = trainDirectory;
        cellpose.valDirectory = valDirectory;
        cellpose.nEpochs = nEpochs;
        cellpose.learningRate = learningRate;
        cellpose.batchSize = batchSize;

        // Overlap for segmentation of tiles. Should be large enough that any object must be "complete"
        // in at least one tile for resolving overlaps
        if (this.overlap != null) {
            cellpose.overlap = this.overlap;

        } else { // The overlap was not set by the user
            if (diameter.isNaN()) {
                cellpose.overlap = 30 * 2; // 30 pixels (largest median object size of Cellpose models times 2.0 to be sure.
                logger.warn("Diameter was not set. Default overlap used");
            } else {
                cellpose.overlap = (int) Math.round(2 * diameter);
            }
        }
        logger.info("If tiling is necessary, {} pixels overlap will be taken between tiles", cellpose.overlap);

        if (this.normPercentileMax > -1.0 && this.normPercentileMin > -1.0 )
            logger.warn("You called the builder with normalization, values below 0 and above after normalization will be clipped");

        // Intersection over union threshold to deal with duplicates
        cellpose.iouThreshold = iouThreshold;
        cellpose.cellConstrainScale = cellConstrainScale;
        cellpose.cellExpansion = cellExpansion;
        cellpose.tileWidth = tileWidth;
        cellpose.tileHeight = tileHeight;
        cellpose.ignoreCellOverlaps = ignoreCellOverlaps;
        cellpose.measureShape = measureShape;
        cellpose.constrainToParent = constrainToParent;
        cellpose.creatorFun = creatorFun;
        cellpose.globalPathClass = globalPathClass;

        cellpose.compartments = new LinkedHashSet<>(compartments);

        if (measurements != null) cellpose.measurements = new LinkedHashSet<>(measurements);
        else cellpose.measurements = Collections.emptyList();

        // If we would like to save the builder we can do it here thanks to Serialization and lots of magic by Pete
        if (saveBuilder) {
            Gson gson = GsonTools.getInstance(true);

            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH'h'mm");
            LocalDateTime now = LocalDateTime.now();
            File savePath = new File( modelDirectory, builderName+"_"+dtf.format(now)+".json");

            try {
                FileWriter fw = new FileWriter(savePath);
                gson.toJson(this, CellposeBuilder.class, fw);
                fw.flush();
                fw.close();
                logger.info("Cellpose Builder serialized and saved to {}", savePath);

            } catch (IOException e) {
                logger.error("Could not save builder to JSON file "+savePath.getAbsolutePath(), e);
            }
        }

        return cellpose;
    }
}