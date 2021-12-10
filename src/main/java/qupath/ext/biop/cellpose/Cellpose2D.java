/*-
 * Copyright 2020-2021 BioImaging & Optics Platform BIOP, Ecole Polytechnique Fédérale de Lausanne
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

import ij.IJ;
import ij.ImagePlus;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.index.strtree.STRtree;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.biop.cmd.VirtualEnvironmentRunner;
import qupath.lib.analysis.features.ObjectMeasurements;
import qupath.lib.analysis.features.ObjectMeasurements.Compartments;
import qupath.lib.analysis.features.ObjectMeasurements.Measurements;
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.common.GeneralTools;
import qupath.lib.geom.ImmutableDimension;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.*;
import qupath.lib.images.servers.ColorTransforms.ColorTransform;
import qupath.lib.objects.CellTools;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.RoiTools;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.scripting.QP;
import qupath.opencv.ops.ImageDataOp;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps;
import qupath.opencv.tools.OpenCVTools;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Dense object detection based on the following publication and code
 * <pre>
 * Stringer, C., Wang, T., Michaelos, M. et al.
 *     "Cellpose: a generalist algorithm for cellular segmentation"
 *     <i>Nat Methods 18, 100–106 (2021). https://doi.org/10.1038/s41592-020-01018-x</i>
 * </pre>
 * See the main repo at https://github.com/mouseland/cellpose
 *
 * <p>
 * The structure of this extension was adapted from the qupath-stardist-extension at https://github.com/qupath/qupath-extension-stardist
 * This way the Cellpose builder mirrors the StarDist2D builder, which should allow users familiar with the StarDist extension to use this one.
 * <p>
 *
 * @author Olivier Burri
 */
public class Cellpose2D {

    private final static Logger logger = LoggerFactory.getLogger(Cellpose2D.class);
    private int channel1;
    private int channel2;
    private double iouThreshold = 0.1;
    private double maskThreshold;
    private double flowThreshold;
    private File cellposeTempFolder;
    private String model;
    private double diameter;
    private ImageDataOp op;
    private double pixelSize;
    private double cellExpansion;
    private double cellConstrainScale;
    private boolean ignoreCellOverlaps;
    private Function<ROI, PathObject> creatorFun;
    private PathClass globalPathClass;
    private boolean constrainToParent = true;
    private int tileWidth = 1024;
    private int tileHeight = 1024;
    private int overlap;
    private boolean measureShape = false;
    private Collection<ObjectMeasurements.Compartments> compartments;
    private Collection<ObjectMeasurements.Measurements> measurements;
    private boolean invert;
    private boolean useOmnipose=false;
    private boolean excludeEdges = false;
    private boolean doCluster = false;
    private CellposeOptions cellposeOptions;

    /**
     * Create a builder to customize detection parameters.
     * This accepts either Text describing the built-in models from cellpose (cyto, cyto2, nuc)
     * or a path to a custom model (as a String)
     *
     * @param modelPath name or path to model to use for prediction.
     * @return this builder
     */
    public static Builder builder(String modelPath) {
        return new Builder(modelPath);
    }

    // Convenience method to convert a PathObject to cells, taken verbatim from StarDist2D
    private static PathObject objectToCell(PathObject pathObject) {
        ROI roiNucleus = null;
        var children = pathObject.getChildObjects();
        if (children.size() == 1)
            roiNucleus = children.iterator().next().getROI();
        else if (children.size() > 1)
            throw new IllegalArgumentException("Cannot convert object with multiple child objects to a cell!");
        return PathObjects.createCellObject(pathObject.getROI(), roiNucleus, pathObject.getPathClass(), pathObject.getMeasurementList());
    }

    private static PathObject cellToObject(PathObject cell, Function<ROI, PathObject> creator) {
        var parent = creator.apply(cell.getROI());
        var nucleusROI = cell instanceof PathCellObject ? ((PathCellObject) cell).getNucleusROI() : null;
        if (nucleusROI != null) {
            var nucleus = creator.apply(nucleusROI);
            nucleus.setPathClass(cell.getPathClass());
            parent.addPathObject(nucleus);
        }
        parent.setPathClass(cell.getPathClass());
        var cellMeasurements = cell.getMeasurementList();
        if (!cellMeasurements.isEmpty()) {
            try (var ml = parent.getMeasurementList()) {
                for (int i = 0; i < cellMeasurements.size(); i++)
                    ml.addMeasurement(cellMeasurements.getMeasurementName(i), cellMeasurements.getMeasurementValue(i));
            }
        }
        return parent;
    }

    /**
     * Detect cells within one or more parent objects, firing update events upon completion.
     *
     * @param imageData the image data containing the object
     * @param parents   the parent objects; existing child objects will be removed, and replaced by the detected cells
     */
    public void detectObjects(ImageData<BufferedImage> imageData, Collection<? extends PathObject> parents) {
        //runInPool(() -> detectObjectsImpl(imageData, parents));
        // Multi step process
        // 1. Extract all images and save to temp folder
        // 2. Run Cellpose on folder
        // 3. Pick up Label images and convert to PathObjects

        // Define temporary folder to work in
        cellposeTempFolder = new File(QP.buildFilePath(QP.PROJECT_BASE_DIR, "cellpose-temp"));
        cellposeTempFolder.mkdirs();

        try {
            FileUtils.cleanDirectory(cellposeTempFolder);
        } catch (IOException e) {
            logger.error("Could not clean temp directory {}", cellposeTempFolder);
            logger.error("Message: ", e);
        }

        // Get downsample factor
        int downsample = 1;
        if (Double.isFinite(pixelSize) && pixelSize > 0) {
            downsample = (int) Math.round(pixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize().doubleValue());
        }

        ImageServer<BufferedImage> server = imageData.getServer();
        PixelCalibration cal = server.getPixelCalibration();

        double expansion = cellExpansion / cal.getAveragedPixelSize().doubleValue();

        final int finalDownsample = downsample;

        List<PathTile> allTiles = parents.parallelStream().map(parent -> {

            // Get for each annotation the individual overlapping tiles
            Collection<? extends ROI> rois = RoiTools.computeTiledROIs(parent.getROI(), ImmutableDimension.getInstance(tileWidth * finalDownsample, tileWidth * finalDownsample), ImmutableDimension.getInstance(tileWidth * finalDownsample, tileHeight * finalDownsample), true, overlap);

            // Keep a reference to the images here while they are being saved
            logger.info("Saving images for {} tiles", rois.size());

            // Save each tile to an image and keep a reference to it
            var individualTiles = rois.parallelStream()
                    .map(t -> {
                        // Make a new RegionRequest
                        var region = RegionRequest.createInstance(server.getPath(), finalDownsample, t);
                        try {
                            return saveTileImage(op, imageData, region);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        return null;
                    })
                    .collect(Collectors.toList());
            return new PathTile(parent, individualTiles);
        }).collect(Collectors.toList());

        // Here the files are saved, and we can run cellpose.
        try {
            runCellPose();
        } catch (IOException | InterruptedException e) {
            logger.error("Failed to Run Cellpose", e);
        }

        // Recover all the images from CellPose to get the masks
        allTiles.parallelStream().forEach(tileMap -> {
            PathObject parent = tileMap.getObject();
            // Read each image
            List<PathObject> allDetections = Collections.synchronizedList(new ArrayList<>());
            tileMap.getTileFiles().parallelStream().forEach(tilefile -> {
                File ori = tilefile.getFile();
                File maskFile = new File(ori.getParent(), FilenameUtils.removeExtension(ori.getName()) + "_cp_masks.tif");
                if (maskFile.exists()) {
                    try {
                        logger.info("Getting objects for {}", maskFile);

                        // thank you Pete for the ContourTracing Class
                        List<PathObject> detections = ContourTracing.labelsToDetections(maskFile.toPath(), tilefile.getTile());

                        allDetections.addAll(detections);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });

            // Filter Detections: Remove overlaps
            List<PathObject> filteredDetections = filterDetections(allDetections);

            // Remove the detections that are not contained within the parent
            Geometry mask = parent.getROI().getGeometry();
            filteredDetections = filteredDetections.stream().filter(t -> mask.covers(t.getROI().getGeometry())).collect(Collectors.toList());

            // Convert to detections, dilating to approximate cells if necessary
            // Drop cells if they fail (rather than catastrophically give up)
            filteredDetections = filteredDetections.parallelStream()
                    .map(n -> {
                        try {
                            return convertToObject(n, n.getROI().getImagePlane(), expansion, constrainToParent ? mask : null);
                        } catch (Exception e) {
                            logger.warn("Error converting to object: " + e.getLocalizedMessage(), e);
                            return null;
                        }
                    }).filter(Objects::nonNull)
                    .collect(Collectors.toList());

            // Resolve cell overlaps, if needed
            if (expansion > 0 && !ignoreCellOverlaps) {
                logger.info("Resolving cell overlaps for {}", parent);
                if (creatorFun != null) {
                    // It's awkward, but we need to temporarily convert to cells and back
                    var cells = filteredDetections.stream().map(Cellpose2D::objectToCell).collect(Collectors.toList());
                    cells = CellTools.constrainCellOverlaps(cells);
                    filteredDetections = cells.stream().map(c -> cellToObject(c, creatorFun)).collect(Collectors.toList());
                } else
                    filteredDetections = CellTools.constrainCellOverlaps(filteredDetections);
            }

            // Add measurements
            if (measureShape)
                filteredDetections.parallelStream().forEach(c -> ObjectMeasurements.addShapeMeasurements(c, cal));

            // Add intensity measurements, if needed
            if (!filteredDetections.isEmpty() && !measurements.isEmpty()) {
                logger.info("Making measurements for {}", parent);
                var stains = imageData.getColorDeconvolutionStains();
                var builder = new TransformedServerBuilder(server);
                if (stains != null) {
                    List<Integer> stainNumbers = new ArrayList<>();
                    for (int s = 1; s <= 3; s++) {
                        if (!stains.getStain(s).isResidual())
                            stainNumbers.add(s);
                    }
                    builder.deconvolveStains(stains, stainNumbers.stream().mapToInt(i -> i).toArray());
                }

                var server2 = builder.build();

                filteredDetections.parallelStream().forEach(cell -> {
                    try {
                        ObjectMeasurements.addIntensityMeasurements(server2, cell, finalDownsample, measurements, compartments);
                    } catch (IOException e) {
                        logger.error(e.getLocalizedMessage(), e);
                    }
                });
            }

            // Assign the objects to the parent object
            parent.setLocked(true);
            parent.clearPathObjects();
            tileMap.getObject().addPathObjects(filteredDetections);
        });

        // Update the hierarchy
        imageData.getHierarchy().fireHierarchyChangedEvent(this);

    }

    private PathObject convertToObject(PathObject object, ImagePlane plane, double cellExpansion, Geometry mask) {
        var geomNucleus = object.getROI().getGeometry();
        PathObject pathObject;
        if (cellExpansion > 0) {
            var geomCell = CellTools.estimateCellBoundary(geomNucleus, cellExpansion, cellConstrainScale);
            if (mask != null)
                geomCell = GeometryTools.attemptOperation(geomCell, g -> g.intersection(mask));

            if (geomCell.isEmpty()) {
                logger.warn("Empty cell boundary at {} will be skipped", object.getROI().getGeometry().getCentroid());
                return null;
            }

            var roiCell = GeometryTools.geometryToROI(geomCell, plane);
            var roiNucleus = GeometryTools.geometryToROI(geomNucleus, plane);
            if (creatorFun == null)
                pathObject = PathObjects.createCellObject(roiCell, roiNucleus, null, null);
            else {
                pathObject = creatorFun.apply(roiCell);
                if (roiNucleus != null) {
                    pathObject.addPathObject(creatorFun.apply(roiNucleus));
                }
            }
        } else {
            if (mask != null) {
                geomNucleus = GeometryTools.attemptOperation(geomNucleus, g -> g.intersection(mask));
            }
            var roiNucleus = GeometryTools.geometryToROI(geomNucleus, plane);
            if (creatorFun == null)
                pathObject = PathObjects.createDetectionObject(roiNucleus);
            else
                pathObject = creatorFun.apply(roiNucleus);
        }

        // Set classification, if available
        PathClass pathClass = globalPathClass;

        if (pathClass != null && pathClass.isValid())
            pathObject.setPathClass(pathClass);
        return pathObject;

    }

    /**
     * Filters the overlapping detections based on their size, a bit like CellDetection and applying the iou threshold
     *
     * @param rawDetections the list of detections to filter
     * @return a list with the filtered results
     */
    private List<PathObject> filterDetections(List<PathObject> rawDetections) {

        // Sort by size
        rawDetections.sort(Comparator.comparingDouble(o -> -1 * o.getROI().getArea()));

        // Create array of detections to keep & to skip
        var detections = new LinkedHashSet<PathObject>();
        var skippedDetections = new HashSet<PathObject>();
        int skipErrorCount = 0;

        // Create a spatial cache to find overlaps more quickly
        // (Because of later tests, we don't need to update envelopes even though geometries may be modified)
        Map<PathObject, Envelope> envelopes = new HashMap<>();
        var tree = new STRtree();
        for (var det : rawDetections) {
            var env = det.getROI().getGeometry().getEnvelopeInternal();
            envelopes.put(det, env);
            tree.insert(env, det);
        }

        for (var detection : rawDetections) {
            if (skippedDetections.contains(detection))
                continue;

            detections.add(detection);
            var envelope = envelopes.get(detection);

            @SuppressWarnings("unchecked")
            var overlaps = (List<PathObject>) tree.query(envelope);
            for (var nuc2 : overlaps) {
                if (nuc2 == detection || skippedDetections.contains(nuc2) || detections.contains(nuc2))
                    continue;

                // If we have an overlap, retain the larger object only
                try {
                    var env = envelopes.get(nuc2);
                    //iou
                    Geometry intersection = detection.getROI().getGeometry().intersection(nuc2.getROI().getGeometry());
                    Geometry union = detection.getROI().getGeometry().union(nuc2.getROI().getGeometry());
                    double iou = intersection.getArea() / union.getArea();

                    // If the intersection area is close to that of nuc2, then nuc2 is almost contained inside the first object
                    // CAREFUL, as this uses already simplified shapes, the result will be different with and without simplifications
                    if (envelope.intersects(env) && detection.getROI().getGeometry().intersects(nuc2.getROI().getGeometry())) {
                        if( iou > this.iouThreshold || intersection.getArea() / nuc2.getROI().getGeometry().getArea() > 0.9 ) {
                             skippedDetections.add(nuc2);
                         }
                    }
                } catch (Exception e) {
                    skippedDetections.add(nuc2);
                    skipErrorCount++;
                }

            }
        }
        if (skipErrorCount > 0) {
            int skipCount = skippedDetections.size();
            logger.warn("Skipped {} detection(s) due to error in resolving overlaps ({}% of all skipped)",
                    skipErrorCount, GeneralTools.formatNumber(skipErrorCount * 100.0 / skipCount, 1));
        }
        return new ArrayList<>(detections);
    }

    /**
     * Saves a region request as an image. We use the ImageJ API because ImageWriter did not work for me
     *
     * @param op        the operations to apply on the image before saving it (32-bit, channel extraction, preprocessing)
     * @param imageData the current ImageData
     * @param request   the region we want to save
     * @return a simple object that contains the request and the associated file in the temp folder
     * @throws IOException an error in case of read/write issue
     */
    private TileFile saveTileImage(ImageDataOp op, ImageData<BufferedImage> imageData, RegionRequest request) throws IOException {

        // This applies all ops to the current tile
        Mat mat;

        mat = op.apply(imageData, request);

        // Convert to something we can save
        ImagePlus imp = OpenCVTools.matToImagePlus("Temp", mat);

        //BufferedImage image = OpenCVTools.matToBufferedImage(mat);

        File tempFile = new File(cellposeTempFolder, "Temp_" + request.getX() + "_" + request.getY() + ".tif");
        logger.info("Saving to {}", tempFile);
        IJ.save(imp, tempFile.getAbsolutePath());

        return new TileFile(request, tempFile);
    }


    /**
     * This class actually runs Cellpose by calling the virtual environment
     *
     * @throws IOException          Exception in case files could not be read
     * @throws InterruptedException Exception in case of command thread has some failing
     */
    private void runCellPose() throws IOException, InterruptedException {

        // Create command to run
        VirtualEnvironmentRunner veRunner = new VirtualEnvironmentRunner(cellposeOptions.getEnvironmentNameorPath(), cellposeOptions.getEnvironmentType());

        // This is the list of commands after the 'python' call
        List<String> cellposeArguments = new ArrayList<>(Arrays.asList("-W", "ignore", "-m", "cellpose"));

        cellposeArguments.add("--dir");
        cellposeArguments.add("" + this.cellposeTempFolder);

        cellposeArguments.add("--pretrained_model");
        cellposeArguments.add("" + this.model);

        cellposeArguments.add("--chan");
        cellposeArguments.add("" + channel1);

        cellposeArguments.add("--chan2");
        cellposeArguments.add("" + channel2);

        cellposeArguments.add("--diameter");
        cellposeArguments.add("" + diameter);


        cellposeArguments.add("--flow_threshold");
        cellposeArguments.add("" + flowThreshold);

        if( cellposeOptions.getVersion().equals(CellposeOptions.CellposeVersion.OMNIPOSE)) cellposeArguments.add("--mask_threshold");
        else cellposeArguments.add("--cellprob_threshold");
        cellposeArguments.add("" + maskThreshold);

        if(useOmnipose) cellposeArguments.add("--omni");
        if(doCluster) cellposeArguments.add("--cluster");
        if(excludeEdges) cellposeArguments.add("--exclude_on_edges");
        
        if (invert) cellposeArguments.add("--invert");

        cellposeArguments.add("--save_tif");

        cellposeArguments.add("--no_npy");

        cellposeArguments.add("--resample");

        if (cellposeOptions.useGPU()) cellposeArguments.add("--use_gpu");

        veRunner.setArguments(cellposeArguments);

        // Finally, we can run Cellpose
        veRunner.runCommand();

        logger.info("Cellpose command finished running");
    }

    /**
     * Builder to help create a {@link Cellpose2D} with custom parameters.
     */
    public static class Builder {

        private final String modelPath;
        private final CellposeOptions cellposeOptions;
        private ColorTransform[] channels = new ColorTransform[0];

        private double maskThreshold = 0.0;
        private double flowThreshold = 0.4;
        private double diameter = 30;

        private double cellExpansion = Double.NaN;
        private double cellConstrainScale = Double.NaN;
        private boolean ignoreCellOverlaps = false;

        private double pixelSize = Double.NaN;

        private int tileWidth = 1024;
        private int tileHeight = 1024;

        private Function<ROI, PathObject> creatorFun;

        private PathClass globalPathClass;

        private boolean measureShape = false;
        private Collection<Compartments> compartments = Arrays.asList(Compartments.values());
        private Collection<Measurements> measurements;

        private boolean constrainToParent = true;

        private final List<ImageOp> ops = new ArrayList<>();
        private double iouThreshold = 0.1;

        private int channel1 = 0; //GRAY
        private int channel2 = 0; // NONE
        private boolean isInvert = false;

        private boolean useOmnipose = false;
        private boolean excludeEdges = false;
        private boolean doCluster = false;

        private Builder(String modelPath) {
            this.modelPath = modelPath;
            this.ops.add(ImageOps.Core.ensureType(PixelType.FLOAT32));
            this.cellposeOptions = CellposeOptions.getInstance();

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
        public Builder pixelSize(double pixelSize) {
            this.pixelSize = pixelSize;
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
        public Builder channels(int... channels) {
            return channels(Arrays.stream(channels)
                    .mapToObj(ColorTransforms::createChannelExtractor)
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
        public Builder channels(String... channels) {
            return channels(Arrays.stream(channels)
                    .map(ColorTransforms::createChannelExtractor)
                    .toArray(ColorTransform[]::new));
        }

        /**
         * Define the channels (or color transformers) to apply to the input image.
         * <p>
         * This makes it possible to supply color deconvolved channels, for example.
         *
         * @param channels ColorTransform channels to use, typically only used internally
         * @return this builder
         */
        public Builder channels(ColorTransform... channels) {
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
        public Builder preprocess(ImageOp... ops) {
            Collections.addAll(this.ops, ops);
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
        public Builder normalizePercentiles(double min, double max) {
            this.ops.add(ImageOps.Normalize.percentile(min, max));
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
        public Builder inputAdd(double... values) {
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
        public Builder inputScale(double... values) {
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
        public Builder tileSize(int tileSize) {
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
        public Builder tileSize(int tileWidth, int tileHeight) {
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
        public Builder cellposeChannels(int channel1, int channel2) {
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
        public Builder maskThreshold(double threshold) {
            this.maskThreshold = threshold;
            return this;
        }

        /**
         * Flow threshold to apply for detection, between 0 and 1.
         *
         * @param threshold flow threshold (default 0.0)
         * @return this builder
         */
        public Builder flowThreshold(double threshold) {
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
        public Builder diameter(double diameter) {
            this.diameter = diameter;
            return this;
        }

        /**
         * Inverts the image channels within cellpose. Adds the --invert flag to the command
         * @return this builder
         */
        public Builder invert() {
            this.isInvert = true;
            return this;
        }

        /**
         * Use Omnipose implementation: Adds --omni flag to command
         *
         * @return this builder
         */
        public Builder useOmnipose() {
            if( this.cellposeOptions.getVersion().equals(CellposeOptions.CellposeVersion.OMNIPOSE) ) {
                this.useOmnipose = true;
            } else {
                logger.warn("--omni flag not available in {}", CellposeOptions.CellposeVersion.CELLPOSE);
            }
                return this;
        }

        /**
         * Exclude on edges. Adds --exclude_on_edges flag to command
         *
         * @return this builder
         */
        public Builder excludeEdges() {
            if( this.cellposeOptions.getVersion().equals(CellposeOptions.CellposeVersion.OMNIPOSE) ) {
                this.excludeEdges = true;
            } else {
                logger.warn("--exclude_edges flag not available in {}", CellposeOptions.CellposeVersion.CELLPOSE);
            }
            return this;
        }

        /**
         * DBSCAN clustering. Reduces oversegmentation of thin features, adds --cluster flag to command.
         *
         * @return this builder
         */
        public Builder clusterDBSCAN() {
            if( this.cellposeOptions.getVersion().equals(CellposeOptions.CellposeVersion.OMNIPOSE) ) {
                this.doCluster = true;
            } else {
                logger.warn("--cluster flag not available in {}", CellposeOptions.CellposeVersion.CELLPOSE);
            }
            return this;
        }

        /**
         * Select the Intersection over Union (IoU) cutoff for excluding overlapping detections.
         * Default of 0.1 is good enough
         * <p>
         * Implementation note: this currently uses the Visvalingam-Whyatt algorithm.
         *
         * @param iouThreshold distance threshold; set &le; 0 to turn off additional simplification
         * @return this builder
         */
        public Builder iou(double iouThreshold) {
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
        public Builder cellExpansion(double distance) {
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
        public Builder cellConstrainScale(double scale) {
            this.cellConstrainScale = scale;
            return this;
        }

        /**
         * Request that a classification is applied to all created objects.
         *
         * @param pathClass the PathClass of all detections resulting from this run
         * @return this builder
         */
        public Builder classify(PathClass pathClass) {
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
        public Builder classify(String pathClassName) {
            return classify(PathClassFactory.getPathClass(pathClassName, (Integer) null));
        }

        /**
         * If true, ignore overlaps when computing cell expansion.
         *
         * @param ignore ignore overlaps when computing cell expansion.
         * @return this builder
         */
        public Builder ignoreCellOverlaps(boolean ignore) {
            this.ignoreCellOverlaps = ignore;
            return this;
        }

        /**
         * If true, constrain nuclei and cells to any parent annotation (default is true).
         *
         * @param constrainToParent constrain nuclei and cells to any parent annotation
         * @return this builder
         */
        public Builder constrainToParent(boolean constrainToParent) {
            this.constrainToParent = constrainToParent;
            return this;
        }

        /**
         * Create annotations rather than detections (the default).
         * If cell expansion is not zero, the nucleus will be included as a child object.
         *
         * @return this builder
         */
        public Builder createAnnotations() {
            this.creatorFun = PathObjects::createAnnotationObject;
            return this;
        }

        /**
         * Request default intensity measurements are made for all available cell compartments.
         *
         * @return this builder
         */
        public Builder measureIntensity() {
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
        public Builder measureIntensity(Collection<Measurements> measurements) {
            this.measurements = new ArrayList<>(measurements);
            return this;
        }

        /**
         * Request shape measurements are made for the detected cell or nucleus.
         *
         * @return this builder
         */
        public Builder measureShape() {
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
        public Builder compartments(Compartments... compartments) {
            this.compartments = Arrays.asList(compartments);
            return this;
        }

        /**
         * Create a {@link Cellpose2D}, all ready for detection.
         *
         * @return a CellPose2D object, ready to be run
         */
        public Cellpose2D build() {

            // The cellpose class that will run the detections
            Cellpose2D cellpose = new Cellpose2D();

            ArrayList<ImageOp> mergedOps = new ArrayList<>(ops);

            // Check the model. If it is a file, then it is a custom model
            File file = new File(modelPath);
            if (file.isFile()) {
                logger.info("Provided model {} is a file. Assuming custom model", file);
            }

            cellpose.model = modelPath;
            cellpose.cellposeOptions = cellposeOptions;

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
            cellpose.maskThreshold = maskThreshold;
            cellpose.flowThreshold = flowThreshold;
            cellpose.pixelSize = pixelSize;
            cellpose.diameter = diameter;
            cellpose.invert = isInvert;
            cellpose.doCluster = doCluster;
            cellpose.excludeEdges = excludeEdges;
            cellpose.useOmnipose = useOmnipose;

            // Overlap for segmentation of tiles. Should be large enough that any object must be "complete"
            // in at least one tile for resolving overlaps
            cellpose.overlap = (int) Math.round(2 * diameter);

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

            if (measurements != null)
                cellpose.measurements = new LinkedHashSet<>(measurements);
            else
                cellpose.measurements = Collections.emptyList();

            return cellpose;
        }

    }

    private static class TileFile {
        private final RegionRequest request;
        private final File file;

        TileFile(RegionRequest request, File tempFile) {
            this.request = request;
            this.file = tempFile;
        }

        public File getFile() {
            return file;
        }

        public RegionRequest getTile() {
            return request;
        }
    }

    private static class PathTile {
        private final PathObject object;
        private final List<TileFile> tile;

        PathTile(PathObject object, List<TileFile> tile) {
            this.object = object;
            this.tile = tile;
        }

        public PathObject getObject() {
            return this.object;
        }

        public List<TileFile> getTileFiles() {
            return tile;
        }
    }

}
