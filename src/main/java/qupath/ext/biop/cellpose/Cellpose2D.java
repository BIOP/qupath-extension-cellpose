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
import ij.gui.PolygonRoi;
import ij.gui.Roi;
import ij.gui.Wand;
import ij.measure.ResultsTable;
import ij.process.ImageProcessor;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.ButtonType;
import javafx.scene.control.Dialog;
import javafx.scene.image.WritableImage;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.GeometryCollection;
import org.locationtech.jts.index.strtree.STRtree;
import org.locationtech.jts.simplify.VWSimplifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.biop.cmd.VirtualEnvironmentRunner;
import qupath.fx.dialogs.Dialogs;
import qupath.fx.utils.FXUtils;
import qupath.imagej.tools.IJTools;
import qupath.lib.analysis.features.ObjectMeasurements;
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.analysis.images.SimpleImage;
import qupath.lib.analysis.images.SimpleImages;
import qupath.lib.common.ColorTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.geom.ImmutableDimension;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.scripting.QPEx;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.*;
import qupath.lib.images.writers.ImageWriterTools;
import qupath.lib.objects.CellTools;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.projects.Project;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.RoiTools;
import qupath.lib.roi.interfaces.ROI;
import qupath.opencv.ops.ImageDataOp;
import qupath.opencv.ops.ImageDataServer;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps;
import qupath.opencv.tools.OpenCVTools;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeMap;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Dense object detection based on the cellpose and omnipose publications
 * <pre>
 * Stringer, C., Wang, T., Michaelos, M. et al.
 *     "Cellpose: a generalist algorithm for cellular segmentation"
 *     <i>Nat Methods 18, 100–106 (2021). <a href="https://doi.org/10.1038/s41592-020-01018-x">doi.org/10.1038/s41592-020-01018-x</a></i>
 * </pre>
 * And
 * <pre>
 * Cutler, K.J., Stringer, C., Lo, T.W. et al.
 *     "Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation"
 *     <i>Nat Methods 19, 1438–1448 (2022). https://doi.org/10.1038/s41592-022-01639-4</i>
 * </pre>
 * See the main repos at <a href="https://github.com/mouseland/cellpose">https://github.com/mouseland/cellpose</a> and <a href="https://github.com/kevinjohncutler/omnipose">https://github.com/kevinjohncutler/omnipose</a>
 *
 * <p>
 * The structure of this extension was adapted from the qupath-stardist-extension at <a href="https://github.com/qupath/qupath-extension-stardist">https://github.com/qupath/qupath-extension-stardist</a>
 * This way the Cellpose builder mirrors the StarDist2D builder, which should allow users familiar with the StarDist extension to use this one.
 * <p>
 *
 * @author Olivier Burri
 * @author Pete Bankhead
 */
public class Cellpose2D {

    private final static Logger logger = LoggerFactory.getLogger(Cellpose2D.class);

    public ImageOp extendChannelOp;

    protected double simplifyDistance = 1.4;
    protected ImageDataOp op;
    protected OpCreators.TileOpCreator globalPreprocess;
    protected List<ImageOp> preprocess;
    protected double pixelSize;
    protected double cellExpansion;
    protected double cellConstrainScale;
    protected boolean ignoreCellOverlaps;
    protected Function<ROI, PathObject> creatorFun;
    protected PathClass globalPathClass;
    protected boolean constrainToParent = true;
    protected int tileWidth;
    protected int tileHeight;
    protected boolean measureShape = false;
    protected Collection<ObjectMeasurements.Compartments> compartments;
    protected Collection<ObjectMeasurements.Measurements> measurements;
    protected int nThreads = -1;
    public boolean saveTrainingImages;

    // CELLPOSE PARAMETERS
    public boolean useGPU;
    public boolean useTestDir;
    public String outputModelName;
    public File groundTruthDirectory;
    protected CellposeSetup cellposeSetup = CellposeSetup.getInstance();
    // Parameters and parameter values that will be passed to the cellpose command
    protected LinkedHashMap<String, String> parameters;
    // No defaults. All should be handled by the builder
    protected String model;
    protected Integer overlap;
    protected File modelDirectory;
    protected boolean doReadResultsAsynchronously;
    protected boolean useCellposeSAM;
    File tempDirectory;
    private List<String> theLog;
    private ResultsTable trainingResults;
    private ResultsTable qcResults;
    private File modelFile;

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
     * Load a previously serialized builder.
     * See {@link CellposeBuilder#CellposeBuilder(File)} and {@link CellposeBuilder#saveBuilder(String)}
     *
     * @param builderPath path to the builder JSON file.
     * @return this builder
     */
    public static CellposeBuilder builder(File builderPath) {
        return new CellposeBuilder(builderPath);
    }

    /**
     * Build a normalization op that can be based upon the entire (2D) image, rather than only local tiles.
     * <p>
     * Example:
     * <pre>
     * <code>
     *   var builder = Cellpose2D.builder()
     *   	.preprocess(
     *   		Cellpose2D.imageNormalizationBuilder()
     *   			.percentiles(0, 99.8)
     *   			.perChannel(false)
     *   			.downsample(10)
     *   			.build()
     *   	).pixelSize(0.5) // Any other options to customize StarDist2D
     *   	.build()
     * </code>
     * </pre>
     * <p>
     * Note that currently this requires downsampling the image to a manageable size.
     *
     * @return a builder for a normalization op
     */
    public static OpCreators.ImageNormalizationBuilder imageNormalizationBuilder() {
        return new OpCreators.ImageNormalizationBuilder();
    }

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
            parent.addChildObject(nucleus);
        }
        parent.setPathClass(cell.getPathClass());
        var cellMeasurements = cell.getMeasurementList();
        if (!cellMeasurements.isEmpty()) {
            try (var ml = parent.getMeasurementList()) {
                ml.putAll(cellMeasurements);
            }
        }
        return parent;
    }

    /**
     * Optionally submit runnable to a thread pool. This limits the parallelization used by parallel streams.
     *
     * @param runnable The runnable to submit
     */
    private void runInPool(Runnable runnable) {
        if (nThreads > 0) {
            if (nThreads == 1)
                logger.info("Processing with {} thread", nThreads);
            else
                logger.info("Processing with {} threads", nThreads);
            // Using an outer thread poll impacts any parallel streams created inside
            var pool = new ForkJoinPool(nThreads);
            try {
                pool.submit(runnable);
            } finally {
                pool.shutdown();
                try {
                    pool.awaitTermination(2, TimeUnit.DAYS);
                } catch (InterruptedException e) {
                    logger.warn("Process was interrupted! {}", e.getLocalizedMessage(), e);
                }
            }
        } else {
            runnable.run();
        }
    }

    /**
     * The directory that was used for saving the training images
     *
     * @return the directory
     */
    public File getTrainingDirectory() {
        return new File(groundTruthDirectory, "train");
    }

    /**
     * The directory that was used for saving the validation images
     *
     * @return the directory
     */
    public File getValidationDirectory() {
        return new File(groundTruthDirectory, "test");
    }

    private Geometry simplify(Geometry geom) {
        if (simplifyDistance <= 0)
            return geom;
        try {
            return VWSimplifier.simplify(geom, simplifyDistance);
        } catch (Exception e) {
            return geom;
        }
    }

    /**
     * Detect cells within one or more parent objects, firing update events upon completion.
     *
     * @param imageData the image data containing the object
     * @param parents   the parent objects; existing child objects will be removed, and replaced by the detected cells
     */
    public void detectObjects(ImageData<BufferedImage> imageData, Collection<? extends PathObject> parents) {
        runInPool(() -> detectObjectsImpl(imageData, parents));
    }

    /**
     * Detect cells within one or more parent objects, firing update events upon completion.
     *
     * @param imageData the image data containing the object
     * @param parents   the parent objects; existing child objects will be removed, and replaced by the detected cells
     */
    public void detectObjectsImpl(ImageData<BufferedImage> imageData, Collection<? extends PathObject> parents) {

        // Multistep process
        // 1. Extract all images and save to temp folder
        // 2. Run Cellpose on folder
        // 3. Pick up Label images and convert to PathObjects
        // 4. Resolve overlaps
        // 5. Make measurements if requested

        Objects.requireNonNull(parents);

        // Make the temp directory
        cleanDirectory(tempDirectory);

        PixelCalibration resolution = imageData.getServer().getPixelCalibration();

        if (Double.isFinite(pixelSize) && pixelSize > 0) {
            double downsample = pixelSize / resolution.getAveragedPixelSize().doubleValue();
            resolution = resolution.createScaledInstance(downsample, downsample);
        }

        // The opServer is needed only to get tile requests, or calculate global normalization percentiles
        ImageDataServer<BufferedImage> opServer = ImageOps.buildServer(imageData, op, resolution, tileWidth, tileHeight);


        // Get downsample factor
        double downsample = 1;
        if (Double.isFinite(pixelSize) && pixelSize > 0) {
            downsample = pixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize().doubleValue();
        }

        ImageServer<BufferedImage> server = imageData.getServer();
        PixelCalibration cal = server.getPixelCalibration();

        double expansion = cellExpansion / cal.getAveragedPixelSize().doubleValue();

        final double finalDownsample = downsample;

        List<TileFile> allTiles = parents.parallelStream().map(parent -> {
            RegionRequest request = RegionRequest.createInstance(
                    opServer.getPath(),
                    opServer.getDownsampleForResolution(0),
                    parent.getROI());

            Collection<? extends ROI> rois = RoiTools.computeTiledROIs(parent.getROI(), ImmutableDimension.getInstance((int) (tileWidth * finalDownsample), (int) (tileWidth * finalDownsample)), ImmutableDimension.getInstance((int) (tileWidth * finalDownsample * 1.5), (int) (tileHeight * finalDownsample * 1.5)), true, (int) (overlap * finalDownsample));

            List<RegionRequest> tiles = rois.stream().map(r -> RegionRequest.createInstance(opServer.getPath(), opServer.getDownsampleForResolution(0), r)).toList();

            // Compute op with preprocessing
            ArrayList<ImageOp> fullPreprocess = new ArrayList<>();
            fullPreprocess.add(ImageOps.Core.ensureType(PixelType.FLOAT32));

            // Do global preprocessing calculations, if required
            if (globalPreprocess != null) {
                try {
                    var normalizeOps = globalPreprocess.createOps(op, imageData, parent.getROI(), request.getImagePlane());
                    fullPreprocess.addAll(normalizeOps);

                    // If this has happened, then we need to disable cellpose normalization
                    this.parameters.put("no_norm", null);

                } catch (IOException e) {
                    throw new RuntimeException("Exception computing global normalization", e);
                }
            }

            if (!preprocess.isEmpty()) {
                fullPreprocess.addAll(preprocess);
            }

            if (fullPreprocess.size() > 1) {
                fullPreprocess.add(ImageOps.Core.ensureType(PixelType.FLOAT32));
            }

            ImageDataOp opWithPreprocessing = op.appendOps(fullPreprocess.toArray(ImageOp[]::new));


            // Keep a reference to the images here while they are being saved
            logger.info("Saving images for {} tiles", tiles.size());

            // Save each tile to an image and keep a reference to it
            List<TileFile> individualTiles = tiles.parallelStream()
                    .map(tile -> {
                        try {
                            return saveTileImage(opWithPreprocessing, imageData, tile, parent);
                        } catch (IOException e) {
                            logger.warn("Could not save tile image", e);
                        }
                        return null;
                    })
                    .collect(Collectors.toList());
            return individualTiles;
        }).flatMap(List::stream).collect(Collectors.toList());

        // Here the files are saved, and we can run cellpose to recover the masks

        try {
            runCellpose(allTiles);
        } catch (IOException | InterruptedException e) {
            logger.error("Failed to Run Cellpose", e);
            return;
        }

        // Group the candidates per parent object, as this is needed to optimize when checking for overlap
        Map<PathObject, List<CandidateObject>> candidatesPerParent = allTiles.stream()
                .flatMap(t -> t.getCandidates().stream())
                .collect(Collectors.groupingBy(c -> c.parent));

        candidatesPerParent.entrySet().parallelStream().forEach(e -> {
            PathObject parent = e.getKey();
            List<CandidateObject> parentCandidates = e.getValue();

            // Filter the detections
            List<CandidateObject> filteredDetections = filterDetections(parentCandidates);

            // Remove the detections that are not contained within the parent
            Geometry mask = parent.getROI().getGeometry();

            // Convert to detections, dilating to approximate cells if necessary
            // Drop cells if they fail (rather than catastrophically give up)
            List<PathObject> finalObjects = filteredDetections.parallelStream()
                    .map(n -> {
                        try {
                            return convertToObject(n, parent.getROI().getImagePlane(), expansion, constrainToParent ? mask : null);
                        } catch (Exception oe) {
                            logger.warn("Error converting to object: {}", oe.getLocalizedMessage(), oe);
                            return null;
                        }
                    }).filter(Objects::nonNull)
                    .collect(Collectors.toList());

            // Resolve cell overlaps, if needed
            if (expansion > 0 && !ignoreCellOverlaps) {
                logger.info("Resolving cell overlaps for {}", parent);
                if (creatorFun != null) {
                    // It's awkward, but we need to temporarily convert to cells and back
                    var cells = finalObjects.stream().map(Cellpose2D::objectToCell).collect(Collectors.toList());
                    cells = CellTools.constrainCellOverlaps(cells);
                    finalObjects = cells.stream().map(c -> cellToObject(c, creatorFun)).collect(Collectors.toList());
                } else
                    finalObjects = CellTools.constrainCellOverlaps(finalObjects);
            }

            // Add measurements
            if (measureShape)
                finalObjects.parallelStream().forEach(c -> ObjectMeasurements.addShapeMeasurements(c, cal));

            // Add intensity measurements, if needed
            if (!finalObjects.isEmpty() && !measurements.isEmpty()) {
                logger.info("Making measurements");
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

                finalObjects.parallelStream().forEach(cell -> {
                    try {
                        ObjectMeasurements.addIntensityMeasurements(server2, cell, finalDownsample, measurements, compartments);
                    } catch (IOException ie) {
                        logger.info("Error adding intensity measurement: {}", ie.getLocalizedMessage(), ie);
                    }
                });

            }

            // Assign the objects to the parent object
            parent.setLocked(true);
            parent.clearChildObjects();
            parent.addChildObjects(finalObjects);
        });

        // Update the hierarchy
        imageData.getHierarchy().fireHierarchyChangedEvent(this);

    }

    private void cleanDirectory(File directory) {
        // Delete the existing directory
        try {
            FileUtils.deleteDirectory(directory);
        } catch (IOException e) {
            logger.error("Failed to delete temp directory", e);
        }

        // Recreate the directory
        try {
            FileUtils.forceMkdir(directory);
        } catch (IOException e) {
            logger.error("Failed to create temp directory", e);
        }
    }

    private PathObject convertToObject(CandidateObject object, ImagePlane plane, double cellExpansion, Geometry mask) {
        var geomNucleus = simplify(object.geometry);
        PathObject pathObject;
        if (cellExpansion > 0) {
//			cellExpansion = geomNucleus.getPrecisionModel().makePrecise(cellExpansion);
//			cellExpansion = Math.round(cellExpansion);
            // Note that prior to QuPath v0.4.0 an extra fix was needed here
            var geomCell = CellTools.estimateCellBoundary(geomNucleus, cellExpansion, cellConstrainScale);
            if (mask != null) {
                geomCell = GeometryTools.attemptOperation(geomCell, g -> g.intersection(mask));
                // Fix nucleus overlaps (added v0.4.0)
                var geomCell2 = geomCell;
                geomNucleus = GeometryTools.attemptOperation(geomNucleus, g -> g.intersection(geomCell2));
                geomNucleus = GeometryTools.ensurePolygonal(geomNucleus);
            }
            geomCell = simplify(geomCell);

            // Intersection with complex mask could give rise to linestring(s) - which we want to remove
            geomCell = GeometryTools.ensurePolygonal(geomCell);

            if (geomCell.isEmpty()) {
                logger.warn("Empty cell boundary at {} will be skipped", object.geometry.getCentroid());
                return null;
            }
            if (geomNucleus.isEmpty()) {
                logger.warn("Empty nucleus at {} will be skipped", object.geometry.getCentroid());
                return null;
            }
            var roiCell = GeometryTools.geometryToROI(geomCell, plane);
            var roiNucleus = GeometryTools.geometryToROI(geomNucleus, plane);
            if (creatorFun == null)
                pathObject = PathObjects.createCellObject(roiCell, roiNucleus, null, null);
            else {
                pathObject = creatorFun.apply(roiCell);
                if (roiNucleus != null) {
                    pathObject.addChildObject(creatorFun.apply(roiNucleus));
                }
            }
        } else {
            if (mask != null) {
                geomNucleus = GeometryTools.attemptOperation(geomNucleus, g -> g.intersection(mask));
                geomNucleus = GeometryTools.ensurePolygonal(geomNucleus);
                if (geomNucleus.isEmpty()) {
                    return null;
                }
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
     * @param rawCandidates the list of detections to filter
     * @return a list with the filtered results
     */
    private List<CandidateObject> filterDetections(Collection<CandidateObject> rawCandidates) {

        // Sort by size
        List<CandidateObject> candidateList = new ArrayList<>(rawCandidates);
        candidateList.sort(Comparator.comparingDouble(o -> -1 * o.area));

        // Create array of detections to keep & to skip
        var retainedObjects = new LinkedHashSet<CandidateObject>();
        var skippedObjects = new HashSet<CandidateObject>();
        int skipErrorCount = 0;

        // Create a spatial cache to find overlaps more quickly
        // (Because of later tests, we don't need to update envelopes even though geometries may be modified)
        Map<CandidateObject, Envelope> envelopes = new HashMap<>();
        var tree = new STRtree();
        for (var det : candidateList) {
            var env = det.geometry.getEnvelopeInternal();
            envelopes.put(det, env);
            tree.insert(env, det);
        }

        for (CandidateObject currentCandidate : candidateList) {
            if (skippedObjects.contains(currentCandidate))
                continue;

            retainedObjects.add(currentCandidate);
            var envelope = envelopes.get(currentCandidate);

            List<CandidateObject> overlaps = (List<CandidateObject>) tree.query(envelope);
            for (CandidateObject overlappingCandidate : overlaps) {
                if (overlappingCandidate == currentCandidate || skippedObjects.contains(overlappingCandidate) || retainedObjects.contains(overlappingCandidate))
                    continue;

                // If we have an overlap, try to keep the largest object
                try {

                    var env = envelopes.get(overlappingCandidate);
                    if (envelope.intersects(env) && currentCandidate.geometry.intersects(overlappingCandidate.geometry)) {
                        // Retain the nucleus only if it is not fragmented, or less than half its original area
                        var difference = overlappingCandidate.geometry.difference(currentCandidate.geometry);

                        if (difference instanceof GeometryCollection) {
                            difference = GeometryTools.ensurePolygonal(difference);

                            // Keep only largest polygon?
                            double maxArea = -1;
                            int index = -1;

                            for (int i = 0; i < difference.getNumGeometries(); i++) {
                                double area = difference.getGeometryN(i).getArea();
                                if (area > maxArea) {
                                    maxArea = area;
                                    index = i;
                                }
                            }
                            difference = difference.getGeometryN(index);
                        }
// difference instanceof Polygon &&
                        if (difference.getArea() > overlappingCandidate.area / 2.0)
                            overlappingCandidate.geometry = difference;
                        else {
                            skippedObjects.add(overlappingCandidate);
                        }
                    }

                } catch (Exception e) {
                    skipErrorCount++;
                    skippedObjects.add(overlappingCandidate);

                }

            }
        }
        if (skipErrorCount > 0) {
            int skipCount = skippedObjects.size();
            logger.warn("Skipped {} objects(s) due to error in resolving overlaps ({}% of all skipped)",
                    skipErrorCount, GeneralTools.formatNumber(skipErrorCount * 100.0 / skipCount, 1));
        }
        return new ArrayList<>(retainedObjects);
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
    private TileFile saveTileImage(ImageDataOp op, ImageData<BufferedImage> imageData, RegionRequest request, PathObject parent) throws IOException {

        // This applies all ops to the current tile
        Mat mat;

        mat = op.apply(imageData, request);

        // Convert to something we can save
        ImagePlus imp = OpenCVTools.matToImagePlus("Temp", mat);

        //BufferedImage image = OpenCVTools.matToBufferedImage(mat);

        File tempFile = new File(tempDirectory,
                "Temp_" +
                        request.getX() + "_" +
                        request.getY() +
                        "_z" + request.getZ() +
                        "_t" + request.getT() + ".tif");
        logger.info("Saving to {}", tempFile);

        // Add check if image is too small, do not process it!
        if (imp.getWidth() < 10 || imp.getHeight() < 10) {
            logger.warn("Image {} will not be saved as it is too small: {}", tempFile, imp);
        } else {
            IJ.save(imp, tempFile.getAbsolutePath());
        }

        return new TileFile(request, tempFile, parent);
    }

    /**
     * Selects the right folder to run from, based on whether it's cellpose or omnipose.
     * Hopefully this will become deprecated soon
     *
     * @return the virtual environment runner that can run the desired command
     */
    private VirtualEnvironmentRunner getVirtualEnvironmentRunner() {

        // Make sure that cellposeSetup.getCellposePythonPath() is not empty
        if (cellposeSetup.getCellposePythonPath().isEmpty() && !this.useCellposeSAM) {
            throw new IllegalStateException("Cellpose python path is empty. Please set it in Edit > Preferences");
        }

        // Make sure that cellposeSetup.getCellposeSAMPythonPath() is not empty
        if (cellposeSetup.getCellposeSAMPythonPath().isEmpty() && this.useCellposeSAM) {
            throw new IllegalStateException("CellposeSAM python path is empty. Please set it in Edit > Preferences");
        }

        // Change the envType based on the setup options
        VirtualEnvironmentRunner.EnvType type = VirtualEnvironmentRunner.EnvType.EXE;
        String condaPath = null;
        if (!cellposeSetup.getCondaPath().isEmpty()) {
            type = VirtualEnvironmentRunner.EnvType.CONDA;
            condaPath = cellposeSetup.getCondaPath();
        }

        // Set python executable to switch between Omnipose and Cellpose
        String pythonPath = this.useCellposeSAM ? cellposeSetup.getCellposeSAMPythonPath() : cellposeSetup.getCellposePythonPath();
        if (this.parameters.containsKey("omni") && !cellposeSetup.getOmniposePythonPath().isEmpty())
            pythonPath = cellposeSetup.getOmniposePythonPath();


        return new VirtualEnvironmentRunner(pythonPath, type, condaPath, this.getClass().getSimpleName());

    }

    /**
     * This class actually runs Cellpose by calling the virtual environment
     *
     * @throws IOException          Exception in case files could not be read
     * @throws InterruptedException Exception in case of command thread has some failing
     */
    private void runCellpose(List<TileFile> allTiles) throws InterruptedException, IOException {

        // Need to define the name of the command we are running. We used to be able to use 'cellpose' for both but not since Cellpose v2
        String runCommand = this.parameters.containsKey("omni") ? "omnipose" : "cellpose";
        VirtualEnvironmentRunner veRunner = getVirtualEnvironmentRunner();

        // This is the list of commands after the 'python' call
        // We want to ignore all warnings to make sure the log is clean (-W ignore)
        // We want to be able to call the module by name (-m)
        // We want to make sure UTF8 mode is by default (-X utf8)
        List<String> cellposeArguments = new ArrayList<>(Arrays.asList("-Xutf8", "-W", "ignore", "-m", runCommand));

        cellposeArguments.add("--dir");
        cellposeArguments.add("" + this.tempDirectory);

        cellposeArguments.add("--pretrained_model");
        cellposeArguments.add(this.model);

        this.parameters.forEach((parameter, value) -> {
            cellposeArguments.add("--" + parameter);
            if (value != null) {
                cellposeArguments.add(value);
            }
        });

        // These all work for cellpose v2
        cellposeArguments.add("--save_tif");

        cellposeArguments.add("--no_npy");

        if (this.useGPU) cellposeArguments.add("--use_gpu");

        cellposeArguments.add("--verbose");

        veRunner.setArguments(cellposeArguments);

        // Finally, we can run Cellpose
        veRunner.runCommand(false);

        processCellposeFiles(veRunner, allTiles);
    }

    private void processCellposeFiles(VirtualEnvironmentRunner veRunner, List<TileFile> allTiles) throws CancellationException, InterruptedException, IOException {

        // Make sure that allTiles is not null, if it is, just return null
        // as we are likely just running validation and thus do not need to give any results back
        if (allTiles == null) {
            veRunner.getProcess().waitFor();
            return;
        }

        // Build a thread pool to process reading the images in parallel
        ExecutorService executor = Executors.newFixedThreadPool(5);

        if (!this.doReadResultsAsynchronously) {
            // We need to wait for the process to finish
            veRunner.getProcess().waitFor();
            allTiles.forEach(entry -> {
                executor.execute(() -> {
                    // Read the objects from the file
                    entry.setCandidates(readObjectsFromTileFile(entry));
                });

            });
        } else { // Experimental file listening and running

            //Make a map of the original names and the expected names
            LinkedHashMap<File, TileFile> remainingFiles = allTiles.stream().map(entry -> {
                File expectedFile = entry.getLabelFile();
                return new AbstractMap.SimpleEntry<>(expectedFile, entry);
            }).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (a, b) -> b, LinkedHashMap::new));

            try {
                // We need to listen for changes in the temp folder
                veRunner.startWatchService(this.tempDirectory.toPath());

                // The command above will run in a separate thread, now we can start listening for the files changing
                while (!remainingFiles.isEmpty() && veRunner.getProcess().isAlive()) {
                    if (!veRunner.getProcess().isAlive()) {
                        // It's no longer running so check the exit code
                        int exitValue = veRunner.getProcess().exitValue();
                        if (exitValue != 0) {
                            throw new IOException("Cellpose process exited with value " + exitValue + ". Please check output above for indications of the problem.\nWill attempt to continue");
                        }
                    }

                    // Get the files that have changes
                    List<String> changedFiles = veRunner.getChangedFiles();

                    if (changedFiles.isEmpty()) {
                        continue;
                    }

                    // Find the tiles that corresponds to the changed files
                    LinkedHashMap<File, TileFile> finishedFiles = remainingFiles.entrySet().stream().filter(set -> {
                        // Create a file that matches the mask name
                        return changedFiles.contains(set.getKey().getName());
                    }).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (a, b) -> b, LinkedHashMap::new));

                    // Announce that these files are done
                    finishedFiles.forEach((key, tile) -> executor.execute(() -> {
                        // Read the objects from the file
                        tile.setCandidates(readObjectsFromTileFile(tile));
                    }));

                    // Remove from the queue
                    finishedFiles.forEach((k, v) -> {
                        remainingFiles.remove(k);
                    });
                }
            } catch (IOException e) {
                logger.error(e.getMessage(), e);

            } finally {
                // No matter what, try and check if there are tiles left

                // Get the files that have changes
                List<String> changedFiles = veRunner.getChangedFiles();

                // Find the tiles that corresponds to the changed files
                LinkedHashMap<File, TileFile> finishedFiles = remainingFiles.entrySet().stream().filter(set -> {
                    // Create a file that matches the mask name
                    return changedFiles.contains(set.getKey().getName());
                }).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (a, b) -> b, LinkedHashMap::new));

                // Announce that these files are done
                finishedFiles.forEach((key, tile) -> {
                    executor.execute(() -> {
                        // Read the objects from the file
                        tile.setCandidates(readObjectsFromTileFile(tile));
                    });
                });
                // Remove them from the list of remaining files

                veRunner.closeWatchService();

            }
        }

        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.MINUTES);
    }

    /**
     * Executes the cellpose training by
     * 1. Saving the images
     * 2. running cellpose
     * 3. moving the resulting model file to the desired directory
     *
     * @return a link to the model file, which can be displayed
     */
    public File train() {

        try {

            if(this.saveTrainingImages) {
                // Clear a previous run
                cleanDirectory(this.groundTruthDirectory);

                saveTrainingImages();
            }
            runTraining();

            this.modelFile = moveRenameAndReturnModelFile();

            // Get the training results before overwriting the log with a new run
            this.trainingResults = parseTrainingResults();

            // Get cellpose masks from the validation
            runCellposeOnValidationImages();

            this.qcResults = runCellposeQC();

            return modelFile;

        } catch (IOException | InterruptedException e) {
            logger.error("Error while running cellpose training: {}", e.getMessage(), e);
        }
        return null;
    }

    /**
     * Configures and runs the {@link VirtualEnvironmentRunner} that will ultimately run cellpose training
     *
     * @throws IOException          Exception in case files could not be read
     * @throws InterruptedException Exception in case of command thread has some failing
     */
    private void runTraining() throws IOException, InterruptedException {
        String runCommand = this.parameters.containsKey("omni") ? "omnipose" : "cellpose";
        VirtualEnvironmentRunner veRunner = getVirtualEnvironmentRunner();

        // This is the list of commands after the 'python' call
        List<String> cellposeArguments = new ArrayList<>(Arrays.asList("-Xutf8", "-W", "ignore", "-m", runCommand));

        cellposeArguments.add("--train");

        cellposeArguments.add("--dir");
        cellposeArguments.add(getTrainingDirectory().getAbsolutePath());

        if(this.useTestDir) {
            cellposeArguments.add("--test_dir");
            cellposeArguments.add(getValidationDirectory().getAbsolutePath());
        }

        cellposeArguments.add("--pretrained_model");
        cellposeArguments.add(Objects.requireNonNullElse(model, "None"));

        this.parameters.forEach((parameter, value) -> {
            cellposeArguments.add("--" + parameter);
            if (value != null) {
                cellposeArguments.add(value);
            }
        });

        // Some people may deactivate this...
        if (this.useGPU) cellposeArguments.add("--use_gpu");

        cellposeArguments.add("--verbose");

        veRunner.setArguments(cellposeArguments);

        // Finally, we can run Cellpose
        veRunner.runCommand(true);

        // Get the log
        this.theLog = veRunner.getProcessLog();
    }

    /**
     * Make a cellpose run on the validation images (test) folder
     * This will create 'cp_masks' images, which can be read when running QC
     */
    private void runCellposeOnValidationImages() {

        // Assume that cellpose training was already run and run cellpose again on the /test folder
        logger.info("Running the new model {} on the validation images to obtain labels for QC", this.modelFile.getName());

        File tmp = this.tempDirectory;
        this.tempDirectory = getValidationDirectory();

        String tmpModel = this.model;
        this.model = this.modelFile.getAbsolutePath();

        try {
            runCellpose(null);
        } catch (InterruptedException | IOException e) {
            logger.error(e.getMessage(), e);
        }
        // Make sure things are back the way they were
        this.tempDirectory = tmp;
        this.model = tmpModel;
    }

    /**
     * Runs the python script "run-cellpose-qc.py", which should be in the QuPath Extensions folder
     *
     * @return the results table with the QC metrics or null
     * @throws IOException          if the python script is not found
     * @throws InterruptedException if the running the QC fails for some reason
     */
    private ResultsTable runCellposeQC() throws IOException, InterruptedException {

        File qcFolder = getQCFolder();

        qcFolder.mkdirs();

        // Let's check if the QC notebook is available in the 'extensions' folder
        String cellposeVersion = CellposeExtension.getExtensionVersion();
        List<File> extensionDirList = QuPathGUI.getExtensionCatalogManager()
                .getCatalogManagedInstalledJars()
                .parallelStream()
                .filter(e->e.toString().contains("qupath-extension-cellpose-"+cellposeVersion))
                .map(Path::getParent)
                .map(Path::toString)
                .map(File::new)
                .collect(Collectors.toList());

        if(extensionDirList.isEmpty()){
            logger.warn("Cellpose extension not installed ; cannot find QC script");
            return null;
        }

        File qcPythonFile = new File(extensionDirList.getFirst(), "run-cellpose-qc.py");
        if (!qcPythonFile.exists()) {
            logger.warn("File {} was not found in {}.\nPlease download it from {}", qcPythonFile.getName(),
                    extensionDirList.getFirst().getAbsolutePath(),
                    new CellposeExtension().getRepository().getUrlString());
            return null;
        }

        // Start the Virtual Environment Runner
        VirtualEnvironmentRunner qcRunner = getVirtualEnvironmentRunner();
        List<String> qcArguments = new ArrayList<>(Arrays.asList(qcPythonFile.getAbsolutePath(), getValidationDirectory().getAbsolutePath(), this.modelFile.getName()));

        qcRunner.setArguments(qcArguments);

        qcRunner.runCommand(true);


        // The results are stored in the validation directory, open them as a results table
        File qcResults = new File(getValidationDirectory(), "QC-Results" + File.separator + "Quality_Control for " + this.modelFile.getName() + ".csv");

        if (!qcResults.exists()) {
            logger.warn("No QC results file name {} found in {}\nCheck in the logger for a potential reason", qcResults.getName(), qcResults.getParent());
            logger.warn("In case you are missing the 'skimage' module, simply run 'pip install scikit-image' in your cellpose environment");
            return null;
        }

        // Move this csv file into the QC folder in the main QuPath project
        File finalQCResultFile = new File(qcFolder, qcResults.getName());
        FileUtils.moveFile(qcResults, finalQCResultFile);

        return ResultsTable.open(finalQCResultFile.getAbsolutePath());
    }

    /**
     * Returns the log from running the cellpose command, with any error messages and status updates of the cellpose process
     * You can use this during training or prediction
     *
     * @return the entire dump of the cellpose log, each line is one element of the String array.
     */
    public List<String> getOutputLog() {
        return theLog;
    }

    /**
     * Returns a parsed version of the cellpose log as a ResultsTable with columns
     * Epoch, Time, Loss, Loss Test and LR
     *
     * @return an ImageJ ResultsTable that can be displayed with {@link ResultsTable#show(String)}
     */
    public ResultsTable getTrainingResults() {
        return this.trainingResults;
    }

    /**
     * Get the results table associated with the Quality Control run
     *
     * @return the results table with the QC metrics
     */
    public ResultsTable getQCResults() {
        return this.qcResults;
    }

    /**
     * Returns a parsed version of the cellpose log as a ResultsTable with columns
     * Epoch, Time, Loss, Loss Test and LR
     *
     * @return a parsed results table
     */
    private ResultsTable parseTrainingResults() {
        ResultsTable trainingResults = new ResultsTable();

        if (this.theLog != null) {
            // Try to parse the output of Cellpose to give meaningful information to the user. This is very old school
            for (String line : this.theLog) {
                Matcher m;
                for (LogParser parser : LogParser.values()) {
                    m = parser.getPattern().matcher(line);
                    if (m.find()) {
                        trainingResults.incrementCounter();
                        trainingResults.addValue("Epoch", Double.parseDouble(m.group("epoch")));
                        trainingResults.addValue("Time", Double.parseDouble(m.group("time")));
                        trainingResults.addValue("Loss", Double.parseDouble(m.group("loss")));
                        if (parser != LogParser.OMNI) { // Omnipose does not provide validation loss
                            trainingResults.addValue("Validation Loss", Double.parseDouble(m.group("val")));
                            trainingResults.addValue("LR", Double.parseDouble(m.group("lr")));

                        } else {
                            trainingResults.addValue("Validation Loss", Double.NaN);
                            trainingResults.addValue("LR", Double.NaN);

                        }
                    }
                }
            }
        }

        //Save results to QC
        File qcTrainingResults = new File(getQCFolder(), "Training Result - " + modelFile.getName());

        logger.info("Saving Training Results to {}", qcTrainingResults.getParent());

        trainingResults.save(qcTrainingResults.getAbsolutePath());
        return trainingResults;
    }

    /**
     * Displays a JavaFX graph as a dialog, so you can inspect the Losses per epoch
     * also saves the graph to the QC folder if requested
     *
     * @param save boolean deciding whether plot should be saved
     */
    public void showTrainingGraph(boolean show, boolean save) {
        ResultsTable output = this.trainingResults;
        File qcFolder = getQCFolder();

        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Epochs");
        yAxis.setForceZeroInRange(true);
        yAxis.setAutoRanging(false);
        yAxis.setLowerBound(0);
        yAxis.setUpperBound(3.0);

        //creating the chart
        final LineChart<Number, Number> lineChart = new LineChart<>(xAxis, yAxis);

        lineChart.setTitle("Cellpose Training");
        //defining a series
        XYChart.Series<Number, Number> loss = new XYChart.Series<>();
        XYChart.Series<Number, Number> lossTest = new XYChart.Series<>();
        loss.setName("Loss");
        lossTest.setName("Loss Test");
        //populating the series with data
        for (int i = 0; i < output.getCounter(); i++) {
            loss.getData().add(new XYChart.Data<>(output.getValue("Epoch", i), output.getValue("Loss", i)));
            lossTest.getData().add(new XYChart.Data<>(output.getValue("Epoch", i), output.getValue("Validation Loss", i)));

        }
        lineChart.getData().add(loss);
        lineChart.getData().add(lossTest);
        FXUtils.runOnApplicationThread(() -> {
            Dialog<ButtonType> dialog = Dialogs.builder().content(lineChart).title("Cellpose Training").buttons("Close").buttons(ButtonType.CLOSE).build();

            if (show) dialog.show();

            if (save) {
                // Save a copy as well in the QC folder
                String trainingGraph = "Training Result - " + this.modelFile.getName() + ".png";

                File trainingGraphFile = new File(qcFolder, trainingGraph);
                WritableImage writableImage = new WritableImage((int) dialog.getWidth(), (int) dialog.getHeight());
                dialog.getDialogPane().snapshot(null, writableImage);
                RenderedImage renderedImage = SwingFXUtils.fromFXImage(writableImage, null);

                logger.info("Saving Training Graph to {}", trainingGraphFile.getName());

                try {
                    ImageIO.write(renderedImage, "png", trainingGraphFile);
                } catch (IOException e) {
                    logger.error("Could not write Training Graph image {} in {}.", trainingGraphFile.getName(), trainingGraphFile.getParent());
                    logger.error("Error Message", e);
                }
            }
        });
    }

    /**
     * Overloaded method for backwards compatibility.
     */
    public void showTrainingGraph() {
        showTrainingGraph(true, true);
    }

    /**
     * Get the location of the QC folder
     *
     * @return the Quality Control folder
     */
    private File getQCFolder() {
        return new File(this.modelDirectory, "QC");
    }

    /**
     * Saves the images from two servers (typically a server with the original data and another with labels)
     * to the right directories as image/mask pairs, ready for cellpose
     *
     * @param annotations    the annotations in which to create RegionRequests to save
     * @param imageName      the desired name of the images (the position of the request will be appended to make them unique)
     * @param originalServer the server that will contain the images
     * @param labelServer    the server that contains the labels
     * @param saveDirectory  the location where to save the pair of images
     */
    private void saveImagePairs(List<PathObject> annotations, String imageName, ImageServer<BufferedImage> originalServer, ImageServer<BufferedImage> labelServer, File saveDirectory) {

        if (annotations.isEmpty()) {
            return;
        }
        double downsample;
        if (Double.isFinite(pixelSize) && pixelSize > 0) {
            downsample = pixelSize / originalServer.getPixelCalibration().getAveragedPixelSize().doubleValue();
        } else {
            downsample = 1.0;
        }
        AtomicInteger idx = new AtomicInteger();

        annotations.forEach(a -> {
            int i = idx.getAndIncrement();

            RegionRequest request = RegionRequest.createInstance(originalServer.getPath(), downsample, a.getROI());
            File imageFile = new File(saveDirectory, imageName + "_region_" + i + ".tif");
            File maskFile = new File(saveDirectory, imageName + "_region_" + i + "_masks.tif");

            try {
                // Ignore the tile if it is too small
                if (request.getWidth() < 10 || request.getHeight() < 10) {
                    throw new Exception("Tile size too small, ignoring");
                }
                ImageWriterTools.writeImageRegion(originalServer, request, imageFile.getAbsolutePath());
                ImageWriterTools.writeImageRegion(labelServer, request, maskFile.getAbsolutePath());
                logger.info("Saved image pair: \n\t{}\n\t{}", imageFile.getName(), maskFile.getName());

            } catch (IOException ex) {
                logger.error(ex.getMessage());
                logger.error("Troubleshooting:\n - Check that the channel names are correct in the builder.");
            } catch (Exception e) {
                logger.warn(e.getMessage());
                logger.warn("Tile {} too small", request);
            }
        });
    }

    /**
     * Goes through the current project and saves the images and masks to the training and validation directories
     */
    public void saveTrainingImages() {

        // Create the required directories if they don't exist
        File trainDirectory = getTrainingDirectory();
        trainDirectory.mkdirs();

        File valDirectory = getValidationDirectory();
        valDirectory.mkdirs();


        Project<BufferedImage> project = QPEx.getQuPath().getProject();

        project.getImageList().forEach(e -> {

            ImageData<BufferedImage> imageData;
            try {

                imageData = e.readImageData();

                // If there is an op for the channels, apply it and add it as an extra channel and make the new ImageData
                if (this.extendChannelOp != null) {
                    // Create an average channels server
                    ImageServer<BufferedImage> avgServer = new TransformedServerBuilder(imageData.getServer()).averageChannelProject().build();
                    ImageData<BufferedImage> avgImageData = new ImageData<>(avgServer, imageData.getHierarchy(), ImageData.ImageType.OTHER);
                    // Create a filtered server channel
                    ImageDataOp op2 = ImageOps.buildImageDataOp(ColorTransforms.createMeanChannelTransform());

                    op2 = op2.appendOps(extendChannelOp);
                    ImageServer<BufferedImage> opServer = ImageOps.buildServer(avgImageData, op2, imageData.getServer().getPixelCalibration());

                    // Combine both into a new server
                    ImageServer<BufferedImage> combinedServer = new TransformedServerBuilder(avgServer).concatChannels(opServer).build();

                    imageData = new ImageData<>(combinedServer, imageData.getHierarchy(), ImageData.ImageType.OTHER);
                }

                String imageName = GeneralTools.stripExtension(imageData.getServer().getMetadata().getName());

                Collection<PathObject> allAnnotations = imageData.getHierarchy().getAnnotationObjects();
                // Get Squares for Training, Validation and Testing
                List<PathObject> trainingAnnotations = allAnnotations.stream().filter(a -> a.getPathClass() == PathClass.getInstance("Training")).collect(Collectors.toList());
                List<PathObject> validationAnnotations = allAnnotations.stream().filter(a -> a.getPathClass() == PathClass.getInstance("Validation")).collect(Collectors.toList());

                // TODO add test annotations too
                //List<PathObject> testingAnnotations = allAnnotations.stream().filter(a -> a.getPathClass() == PathClass.getInstance("Test")).collect(Collectors.toList());

                PixelCalibration resolution = imageData.getServer().getPixelCalibration();
                if (Double.isFinite(pixelSize) && pixelSize > 0) {
                    double downsample = pixelSize / resolution.getAveragedPixelSize().doubleValue();
                    resolution = resolution.createScaledInstance(downsample, downsample);
                }

                logger.info("Found {} Training objects and {} Validation objects in image {}", trainingAnnotations.size(), validationAnnotations.size(), imageName);

                if (!trainingAnnotations.isEmpty() || !validationAnnotations.isEmpty()) {
                    // Make the server using the required ops
                    // Do global preprocessing calculations, if required
                    ArrayList<ImageOp> fullPreprocess = new ArrayList<>();
                    fullPreprocess.add(ImageOps.Core.ensureType(PixelType.FLOAT32));
                    if (globalPreprocess != null) {
                        try {
                            var normalizeOps = globalPreprocess.createOps(op, imageData, null, null);
                            fullPreprocess.addAll(normalizeOps);

                            // If this has happened, then we should expect to not use the cellpose normalization?
                            this.parameters.put("no_norm", null);

                        } catch (IOException ex) {
                            throw new RuntimeException("Exception computing global normalization", ex);
                        }
                    }

                    if (!preprocess.isEmpty()) {
                        fullPreprocess.addAll(preprocess);
                    }
                    if (fullPreprocess.size() > 1)
                        fullPreprocess.add(ImageOps.Core.ensureType(PixelType.FLOAT32));

                    var opWithPreprocessing = op.appendOps(fullPreprocess.toArray(ImageOp[]::new));

                    ImageServer<BufferedImage> processed = ImageOps.buildServer(imageData, opWithPreprocessing, resolution, tileWidth, tileHeight);

                    LabeledImageServer labelServer = new LabeledImageServer.Builder(imageData)
                            .backgroundLabel(0, ColorTools.BLACK)
                            .multichannelOutput(false)
                            .useInstanceLabels()
                            .useFilter(o -> o.getPathClass() == null) // Keep only objects with no PathClass
                            .build();

                    saveImagePairs(trainingAnnotations, imageName, processed, labelServer, trainDirectory);
                    saveImagePairs(validationAnnotations, imageName, processed, labelServer, valDirectory);
                }
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
    private File moveRenameAndReturnModelFile() throws IOException {

        if (this.outputModelName == null) {
            this.outputModelName = "Custom_model";
        }
        // Append timestamp
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH_mm");
        this.outputModelName += "_" + formatter.format(LocalDateTime.now());

        File cellPoseModelFolder = new File(getTrainingDirectory(), "models");
        // Find the first file in there
        File[] all = cellPoseModelFolder.listFiles();
        Optional<File> cellPoseModel = Arrays.stream(Objects.requireNonNull(all)).filter(f -> f.getName().contains("cellpose")).findFirst();
        if (cellPoseModel.isPresent()) {
            File model = cellPoseModel.get();
            logger.info("Found model file at {} ", model);
            File newModel = new File(modelDirectory, this.outputModelName + ".cpm");
            FileUtils.copyFile(model, newModel);
            logger.info("Model file {} moved to {}", newModel.getName(), newModel.getParent().replace('\\', '/'));
            return newModel;
        }
        return null;
    }

    private Collection<CandidateObject> readObjectsFromTileFile(TileFile tileFile) {
        RegionRequest request = tileFile.getTile();

        logger.info("Reading {}", tileFile.getLabelFile().getName());
        // Open the image
        ImagePlus label_imp = IJ.openImage(tileFile.getLabelFile().getAbsolutePath());
        ImageProcessor ip = label_imp.getProcessor();

        Wand wand = new Wand(ip);

        // create range list
        int width = ip.getWidth();
        int height = ip.getHeight();

        int[] pixel_width = new int[width];
        int[] pixel_height = new int[height];

        IntStream.range(0, width - 1).forEach(val -> pixel_width[val] = val);
        IntStream.range(0, height - 1).forEach(val -> pixel_height[val] = val);

        /*
         * Will iterate through pixels, when getPixel > 0 ,
         * then use the magic wand to create a roi
         * finally set value to 0 and add to the roiManager
         */

        // will "erase" found ROI by setting them to 0
        ip.setColor(0);
        List<CandidateObject> rois = new ArrayList<>();

        for (int yCoordinate : pixel_height) {
            for (int xCoordinate : pixel_width) {
                float val = ip.getf(xCoordinate, yCoordinate);
                if (val > 0.0) {
                    // use the magic wand at this coordinate
                    wand.autoOutline(xCoordinate, yCoordinate, val, val);
                    // if there is a region, then the wand has points
                    if (wand.npoints > 0) {
                        // get the Polygon, fill with 0 and add to the manager
                        Roi roi = new PolygonRoi(wand.xpoints, wand.ypoints, wand.npoints, Roi.FREEROI);
                        // Name the Roi with the position in the stack followed by the label ID
                        // ip.fill should use roi, otherwise make a rectangle that erases surrounding pixels

                        CandidateObject o = new CandidateObject(IJTools.convertToROI(roi, -1 * request.getX() / request.getDownsample(), -1 * request.getY() / request.getDownsample(), request.getDownsample(), request.getImagePlane()).getGeometry(), tileFile.getParent());

                        rois.add(o);
                        ip.fill(roi);
                    }
                }
            }
        }
        label_imp.close();
        return rois;
    }

    /**
     * convert a label image to a collection of Geometry objects
     *
     * @param tileFile the current tileFile we are processing
     * @return a collection of CandidateObject that will be added to the total objects
     */
    private Collection<CandidateObject> readObjectsFromFileOld(TileFile tileFile) {

        logger.info("Reading objects from file {}", tileFile.getLabelFile().getName());
        try {
            BufferedImage bfImage = ImageIO.read(tileFile.getLabelFile());
            SimpleImage image = ContourTracing.extractBand(bfImage.getRaster(), 0);
            float[] pixels = SimpleImages.getPixels(image, true);

            float maxValue = 1;
            for (float p : pixels) {
                if (p > maxValue)
                    maxValue = p;
            }
            int maxLabel = (int) maxValue;

            Map<Number, CandidateObject> candidates = new TreeMap<>();
            float lastLabel = Float.NaN;
            for (float p : pixels) {
                if (p >= 1 && p <= maxLabel && p != lastLabel && !candidates.containsKey(p)) {
                    Geometry geometry = ContourTracing.createTracedGeometry(image, p, p, tileFile.getTile());
                    if (geometry != null && !geometry.isEmpty())
                        candidates.put(p, new CandidateObject(geometry, tileFile.getParent()));
                    lastLabel = p;
                }
            }
            bfImage.flush();
            // Ignore the IDs, because they will be the same across different images, and we don't really need them
            if (candidates.isEmpty()) return Collections.emptyList();
            return candidates.values();

        } catch (IOException e) {
            logger.warn("Image {} could not be read for some reason: \n{}", tileFile.getLabelFile(), e.getLocalizedMessage());
        }
        return Collections.emptyList();
    }


    public enum LogParser {

        // Cellpose 2 pattern when training : "Look for "Epoch 0, Time  2.3s, Loss 1.0758, Loss Test 0.6007, LR 0.2000"
        // Cellpose 3 pattern when training : "5, train_loss=2.6546, test_loss=2.0054, LR=0.1111, time 2.56s"
        // Omnipose pattern when training   : "Train epoch: 10 | Time: 0.22min | last epoch: 0.74s | <sec/epoch>: 0.73s | <sec/batch>: 0.33s | <Batch Loss>: 5.076259 | <Epoch Loss>: 4.429341"
        // WARNING: Currently Omnipose does not provide any output to the validation loss (Test loss in Cellpose)
        CP2("Cellpose v2", ".*Epoch\\s*(?<epoch>\\d+),\\s*Time\\s*(?<time>\\d+\\.\\d)s,\\s*Loss\\s*(?<loss>\\d+\\.\\d+),\\s*Loss Test\\s*(?<val>\\d+\\.\\d+),\\s*LR\\s*(?<lr>\\d+\\.\\d+).*"),
        CP3("Cellpose v3", ".* (?<epoch>\\d+), train_loss=(?<loss>\\d+\\.\\d+), test_loss=(?<val>\\d+\\.\\d+), LR=(?<lr>\\d+\\.\\d+), time (?<time>\\d+\\.\\d+)s.*"),
        OMNI("Omnipose", ".*Train epoch: (?<epoch>\\d+) \\| Time: (?<time>\\d+\\.\\d+)min .*\\<Epoch Loss\\>: (?<loss>\\d+\\.\\d+).*");

        private final String name;
        private final Pattern pattern;

        LogParser(String name, String regex) {
            this.name = name;
            this.pattern = Pattern.compile(regex);
        }

        public String getName() {
            return this.name;
        }

        public Pattern getPattern() {
            return this.pattern;
        }
    }

    /**
     * Static class to hold the correspondence between a
     * RegionRequest and a saved file.
     * This also contains a way to infer the resulting image mask file name
     */
    private static class TileFile {
        private final RegionRequest request;
        private final File imageFile;
        private final PathObject parent;

        private Collection<CandidateObject> candidates = Collections.emptyList();


        TileFile(RegionRequest request, File imageFile, PathObject parent) {
            this.request = request;
            this.parent = parent;
            this.imageFile = imageFile;
        }

        public File getImageFile() {
            return imageFile;
        }

        public File getLabelFile() {
            return new File(FilenameUtils.removeExtension(imageFile.getAbsolutePath()) + "_cp_masks.tif");
        }

        public RegionRequest getTile() {
            return request;
        }

        public PathObject getParent() {
            return parent;
        }

        public Collection<CandidateObject> getCandidates() {
            return this.candidates;
        }

        public void setCandidates(Collection<CandidateObject> candidates) {
            this.candidates = candidates;
        }
    }

    /**
     * Static class that holds each geometry in order to quickly check overlaps
     */
    private static class CandidateObject {
        private final double area;
        private Geometry geometry;
        private final PathObject parent; // Perhaps this duplicated things a bit, but we need it to sort the data

        CandidateObject(Geometry geom, PathObject parent) {
            this.geometry = geom;
            this.area = geom.getArea();
            this.parent = parent;

            // Clean up the geometry already
            geometry = GeometryTools.ensurePolygonal(geometry);

            // Keep only largest polygon?
            double maxArea = -1;
            int index = -1;

            for (int i = 0; i < geometry.getNumGeometries(); i++) {
                double area = geometry.getGeometryN(i).getArea();
                if (area > maxArea) {
                    maxArea = area;
                    index = i;
                }
            }
            geometry = geometry.getGeometryN(index);
        }
    }
}