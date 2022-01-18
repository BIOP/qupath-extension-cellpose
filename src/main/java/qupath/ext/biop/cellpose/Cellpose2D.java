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
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.common.ColorTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.geom.ImmutableDimension;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.LabeledImageServer;
import qupath.lib.images.servers.PixelCalibration;
import qupath.lib.images.servers.TransformedServerBuilder;
import qupath.lib.images.writers.ImageWriterTools;
import qupath.lib.objects.CellTools;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.projects.Project;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.RoiTools;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.scripting.QP;
import qupath.opencv.ops.ImageDataOp;
import qupath.opencv.ops.ImageOps;
import qupath.opencv.tools.OpenCVTools;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
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

    protected Integer channel1 = 0;
    protected Integer channel2 = 0;
    protected Double iouThreshold = 0.1;
    protected Double maskThreshold = 0.0;
    protected Double flowThreshold = 0.0;
    protected String model = null;
    protected Double diameter = 0.0;
    protected ImageDataOp op = null;
    protected Double pixelSize = null;
    protected Double cellExpansion = 0.0;
    protected Double cellConstrainScale = 1.5;
    protected Boolean ignoreCellOverlaps = Boolean.FALSE;
    protected Function<ROI, PathObject> creatorFun;
    protected PathClass globalPathClass;
    protected Boolean constrainToParent = Boolean.TRUE;
    protected Integer tileWidth = 1024;
    protected Integer tileHeight = 1024;
    protected Integer overlap = null;
    protected Boolean measureShape = Boolean.FALSE;
    protected Collection<ObjectMeasurements.Compartments> compartments;
    protected Collection<ObjectMeasurements.Measurements> measurements;
    protected Boolean invert = Boolean.FALSE;
    protected Boolean useOmnipose = Boolean.FALSE;
    protected Boolean excludeEdges = Boolean.FALSE;
    protected Boolean doCluster = Boolean.FALSE;
    // Training parameters
    protected File modelDirectory = null;
    protected File trainDirectory = null;
    protected File valDirectory = null;
    protected Integer nEpochs = null;
    public Double learningRate = null;
    public Integer batchSize = null;
    protected CellposeSetup cellposeSetup = CellposeSetup.getInstance();
    protected Boolean useGPU = Boolean.FALSE;
    private File cellposeTempFolder = null;

    /**
     * Create a builder to customize detection parameters.
     * This accepts either Text describing the built-in models from cellpose (cyto, cyto2, nuc)
     * or a path to a custom model (as a String)
     *
     * @param modelPath name or path to model to use for prediction.
     * @return this builder
     */
    public static CellposeBuilder builder(String modelPath) {
        return new CellposeBuilder(modelPath);
    }

    /**
     * Load a previouslt serialized builder.
     * See {@link CellposeBuilder#CellposeBuilder(File)} and {@link CellposeBuilder#saveBuilder(String)}
     *
     * @param builderPath path to the builder JSON file.
     * @return this builder
     */
    public static CellposeBuilder builder(File builderPath) {
        return new CellposeBuilder(builderPath);
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
            runCellpose();
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
                        if (iou > this.iouThreshold || intersection.getArea() / nuc2.getROI().getGeometry().getArea() > 0.9) {
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
    private void runCellpose() throws IOException, InterruptedException {

        // Create command to run
        VirtualEnvironmentRunner veRunner = new VirtualEnvironmentRunner(cellposeSetup.getEnvironmentNameOrPath(), cellposeSetup.getEnvironmentType(), this.getClass().getSimpleName());

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

        if(!diameter.isNaN()) {
            cellposeArguments.add("--diameter");
            cellposeArguments.add("" + diameter);
        }

        if(!flowThreshold.isNaN()) {
            cellposeArguments.add("--flow_threshold");
            cellposeArguments.add("" + flowThreshold);
        }

        if(!maskThreshold.isNaN()) {
            if (cellposeSetup.getVersion().equals(CellposeSetup.CellposeVersion.OMNIPOSE))
                cellposeArguments.add("--mask_threshold");
            else cellposeArguments.add("--cellprob_threshold");

            cellposeArguments.add("" + maskThreshold);
        }

        if (useOmnipose) cellposeArguments.add("--omni");
        if (doCluster) cellposeArguments.add("--cluster");
        if (excludeEdges) cellposeArguments.add("--exclude_on_edges");

        if (invert) cellposeArguments.add("--invert");

        cellposeArguments.add("--save_tif");

        cellposeArguments.add("--no_npy");

        cellposeArguments.add("--resample");

        if (useGPU) cellposeArguments.add("--use_gpu");

        veRunner.setArguments(cellposeArguments);

        // Finally, we can run Cellpose
        veRunner.runCommand();

        logger.info("Cellpose command finished running");
    }

    public File train() {

        try {
            saveTrainingImages();
            runCellposeTraining();
            return moveAndReturnModelFile();

        } catch (IOException | InterruptedException e) {
            logger.error(e.getMessage(), e);
        }
        return null;
    }

    private void runCellposeTraining() throws IOException, InterruptedException {

        //python -m cellpose --train --dir ~/images_cyto/train/ --test_dir ~/images_cyto/test/ --pretrained_model cyto --chan 2 --chan2 1

        // Create command to run
        VirtualEnvironmentRunner veRunner = new VirtualEnvironmentRunner(cellposeSetup.getEnvironmentNameOrPath(), cellposeSetup.getEnvironmentType(), this.getClass().getSimpleName()+"-train");

        // This is the list of commands after the 'python' call

        List<String> cellposeArguments = new ArrayList<>(Arrays.asList("-W", "ignore", "-m", "cellpose"));

        cellposeArguments.add("--train");

        cellposeArguments.add("--dir");
        cellposeArguments.add("" + trainDirectory.getAbsolutePath());
        cellposeArguments.add("--test_dir");
        cellposeArguments.add("" + valDirectory.getAbsolutePath());

        cellposeArguments.add("--pretrained_model");
        if (model != null) {
            cellposeArguments.add("" + model);
        } else {
            cellposeArguments.add("None");
        }

        // The channel order will always be 1 and 2, in the order defined by channels(...) in the builder
        cellposeArguments.add("--chan");
        cellposeArguments.add("" + channel1);
        cellposeArguments.add("--chan2");
        cellposeArguments.add("" + channel2);

        cellposeArguments.add("--diameter");
        cellposeArguments.add("" + diameter);

        if( nEpochs !=null ) {
            cellposeArguments.add("--n_epochs");
            cellposeArguments.add("" + nEpochs);
        }

        if( !learningRate.isNaN() ) {
            cellposeArguments.add("--learning_rate");
            cellposeArguments.add(""+learningRate);
        }

        if ( batchSize != null ) {
            cellposeArguments.add("--batch_size");
            cellposeArguments.add(""+batchSize);
        }

        if (invert) cellposeArguments.add("--invert");
        if (useOmnipose) cellposeArguments.add("--omni");

        if (useGPU) cellposeArguments.add("--use_gpu");

        veRunner.setArguments(cellposeArguments);

        // Finally, we can run Cellpose
        veRunner.runCommand();

        logger.info("Cellpose command finished running");
    }

    /**
     * Saves the the images from two servers (typically a server with the original data and another with labels)
     * to the right directories as image/mask pairs, ready for cellpose
     *
     * @param annotations    the annotations in which to create RegionRequests to save
     * @param imageName      thge desired name of the images (the position of the request will be appended to make them unique)
     * @param originalServer the server that will contain the images
     * @param labelServer    the server that contains the labels
     * @param saveDirectory  the location where to save the pair of images
     */
    private void saveImagePairs(List<PathObject> annotations, String imageName, ImageServer<BufferedImage> originalServer, ImageServer<BufferedImage> labelServer, File saveDirectory) {

        if (annotations.isEmpty()) {
            return;
        }
        int downsample = 1;
        if (Double.isFinite(pixelSize) && pixelSize > 0) {
            downsample = (int) Math.round(pixelSize / originalServer.getPixelCalibration().getAveragedPixelSize().doubleValue());
        }

        AtomicInteger idx = new AtomicInteger();
        int finalDownsample = downsample;

        logger.info("Saving Images...");
        annotations.forEach(a -> {
            int i = idx.getAndIncrement();

            RegionRequest request = RegionRequest.createInstance(originalServer.getPath(), finalDownsample, a.getROI());
            File imageFile = new File(saveDirectory, imageName + "_region_" + i + ".tif");
            File maskFile = new File(saveDirectory, imageName + "_region_" + i + "_masks.tif");
            try {

                ImageWriterTools.writeImageRegion(originalServer, request, imageFile.getAbsolutePath());
                ImageWriterTools.writeImageRegion(labelServer, request, maskFile.getAbsolutePath());

            } catch (IOException ex) {
                logger.error(ex.getMessage());
            }
        });
    }

    /**
     * Save training images for the project
     *
     * @throws IOException an error in case the images cannot be saved
     */
    private void saveTrainingImages() throws IOException {

        Project<BufferedImage> project = QP.getProject();
        // Prepare location to save images

        project.getImageList().parallelStream().forEach(e -> {

            ImageData<BufferedImage> imageData;
            try {
                imageData = e.readImageData();
                String imageName = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName());

                // Make the server using the required ops
                ImageServer<BufferedImage> processed = ImageOps.buildServer(imageData, op, imageData.getServer().getPixelCalibration(), 2048, 2048);

                Collection<PathObject> allAnnotations = imageData.getHierarchy().getAnnotationObjects();
                // Get Squares for Training
                List<PathObject> trainingAnnotations = allAnnotations.stream().filter(a -> a.getPathClass() == PathClassFactory.getPathClass("Training")).collect(Collectors.toList());
                List<PathObject> validationAnnotations = allAnnotations.stream().filter(a -> a.getPathClass() == PathClassFactory.getPathClass("Validation")).collect(Collectors.toList());

                logger.info("Found {} Training objects and {} Validation Objects in image {}", trainingAnnotations.size(), validationAnnotations.size(), imageName);

                LabeledImageServer labelServer = new LabeledImageServer.Builder(imageData)
                        .backgroundLabel(0, ColorTools.BLACK)
                        .multichannelOutput(false)
                        .useInstanceLabels()
                        .useFilter(o -> o.getPathClass() == null) // Keep only objects with no PathClass
                        .build();

                saveImagePairs(trainingAnnotations, imageName, processed, labelServer, trainDirectory);
                saveImagePairs(validationAnnotations, imageName, processed, labelServer, valDirectory);
            } catch (Exception ex) {
                logger.error(ex.getMessage());
            }
        });

    }

    /**
     * Checks the default folder where cellpose drops a trained model (../train/models/)
     * and moves it to the defined modelDirectory using {@link #modelDirectory}
     *
     * @return the File of the moved model
     * @throws IOException in case there was a problem moving the file
     */
    private File moveAndReturnModelFile() throws IOException {
        File cellPoseModelFolder = new File(trainDirectory, "models");
        // Find the first file in there
        File[] all = cellPoseModelFolder.listFiles();
        Optional<File> cellPoseModel = Arrays.stream(all).filter(f -> f.getName().contains("cellpose")).findFirst();
        if (cellPoseModel.isPresent()) {
            logger.info("Found model file at {} ", cellPoseModel);
            File model = cellPoseModel.get();
            File newModel = new File(modelDirectory, model.getName());
            FileUtils.copyFile(model, newModel);
            return newModel;
        }
        return null;
    }

    /**
     * Static class to hold the correspondence between a RegionRequest and a saved file, so that we can place the detected ROIs in the right place.
     */
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

    /**
     * Static class that contains all the tiles as {@link TileFile}s for each object
     * This will allow us to make sure that each object has the right child objects assigned to it after prediction
     */
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
