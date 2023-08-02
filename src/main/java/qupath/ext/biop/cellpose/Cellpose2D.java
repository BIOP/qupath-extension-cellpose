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
import ij.measure.ResultsTable;
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
import org.locationtech.jts.geom.prep.PreparedGeometry;
import org.locationtech.jts.geom.prep.PreparedGeometryFactory;
import org.locationtech.jts.index.strtree.STRtree;
import org.locationtech.jts.simplify.VWSimplifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.biop.cmd.VirtualEnvironmentRunner;
import qupath.lib.analysis.features.ObjectMeasurements;
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.analysis.images.SimpleImage;
import qupath.lib.analysis.images.SimpleImages;
import qupath.lib.common.ColorTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.geom.ImmutableDimension;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.dialogs.Dialogs;
import qupath.lib.gui.scripting.QPEx;
import qupath.lib.gui.tools.GuiTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.*;
import qupath.lib.images.writers.ImageWriterTools;
import qupath.lib.objects.CellTools;
import qupath.lib.objects.PathCellObject;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.projects.Project;
import qupath.lib.projects.Projects;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.RoiTools;
import qupath.lib.roi.interfaces.ROI;
import qupath.lib.scripting.QP;
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
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Dense object detection based on the cellpose and omnipose publications
 * <pre>
 * Stringer, C., Wang, T., Michaelos, M. et al.
 *     "Cellpose: a generalist algorithm for cellular segmentation"
 *     <i>Nat Methods 18, 100–106 (2021). https://doi.org/10.1038/s41592-020-01018-x</i>
 * </pre>
 * And
 * <pre>
 * Cutler, K.J., Stringer, C., Lo, T.W. et al.
 *     "Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation"
 *     <i>Nat Methods 19, 1438–1448 (2022). https://doi.org/10.1038/s41592-022-01639-4</i>
 * </pre>
 * See the main repos at https://github.com/mouseland/cellpose and https://github.com/kevinjohncutler/omnipose
 *
 * <p>
 * The structure of this extension was adapted from the qupath-stardist-extension at https://github.com/qupath/qupath-extension-stardist
 * This way the Cellpose builder mirrors the StarDist2D builder, which should allow users familiar with the StarDist extension to use this one.
 * <p>
 *
 * @author Olivier Burri
 * @author Pete Bankhead
 */
public class Cellpose2D {

    private final static Logger logger = LoggerFactory.getLogger(Cellpose2D.class);

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


    // CELLPOSE PARAMETERS

    protected CellposeSetup cellposeSetup = CellposeSetup.getInstance();

    // Parameters and parameter values that will be passed to the cellpose command
    protected LinkedHashMap<String, String> parameters;

    // No defaults. All should be handled by the builder
    protected String model;
    protected Integer overlap;

    protected File modelDirectory;
    protected File trainDirectory;
    protected File valDirectory;
    protected int nThreads = -1;
    File tempDirectory;
    private String[] theLog;
    // Results table from the training
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
     * @return
     */
    public static OpCreators.ImageNormalizationBuilder imageNormalizationBuilder() {
        return new OpCreators.ImageNormalizationBuilder();
    }

    private static List<Geometry> filterObjects(List<Geometry> objectGeometry) {

        // Sort in descending order of size
        Collections.sort(objectGeometry, Comparator.comparingDouble((Geometry n) -> n.getArea()).reversed());

        // Create array of nuclei to keep & to skip
        var objects = new LinkedHashSet<Geometry>();
        var skippedObjects = new HashSet<Geometry>();
        int skipErrorCount = 0;

        // Create a spatial cache to find overlaps more quickly
        // (Because of later tests, we don't need to update envelopes even though geometries may be modified)
        Map<Geometry, Envelope> envelopes = new HashMap<>();
        var tree = new STRtree();
        for (var cell : objectGeometry) {
            var env = cell.getEnvelopeInternal();
            envelopes.put(cell, env);
            tree.insert(env, cell);
        }

        var preparingFactory = new PreparedGeometryFactory();

        for (var cell : objectGeometry) {
            if (skippedObjects.contains(cell))
                continue;

            objects.add(cell);
            var envelope = envelopes.computeIfAbsent(cell, g -> g.getEnvelopeInternal());

            @SuppressWarnings("unchecked")
            var overlaps = (List<Geometry>) tree.query(envelope);

            // Remove the overlaps that we can be sure don't apply using quick tests, to avoid expensive ones
            var iter = overlaps.iterator();
            while (iter.hasNext()) {
                var cell2 = iter.next();
                if (cell2 == cell || skippedObjects.contains(cell2) || objects.contains(cell2))
                    iter.remove();
                else {
                    // Envelope text needed because nuclei can have been modified
                    var env = envelopes.computeIfAbsent(cell2, g -> g.getEnvelopeInternal());
                    if (!envelope.intersects(env))
                        iter.remove();
                }
            }

            // If we need to compare a lot of intersections, preparing the geometry can speed things up
            PreparedGeometry prepared = null;
            if (overlaps.size() > 5) {
                prepared = preparingFactory.create(cell);
            }
            for (var cell2 : overlaps) {
                // If we have an overlap, retain the largest object (i.e. the one we met first)
                try {
                    boolean checkDifference = true;
                    if (prepared == null) {
                        // We could check for intersection, but it seems faster to just compute difference
                        // (this would warrant some more systematic checking though)
                        checkDifference = true;//nucleus.geometry.intersects(nucleus2.geometry);
                    } else
                        checkDifference = prepared.intersects(cell2);
                    if (checkDifference) {
                        // Retain the nucleus only if it is not fragmented, or less than half its original area
                        var difference = cell2.difference(cell);

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
                        if (difference.getArea() > cell2.getArea() / 2.0)
                            cell2 = difference;
                        else {
                            skippedObjects.add(cell2);
                        }
                    }
                } catch (Exception e) {
                    skippedObjects.add(cell2);
                    skipErrorCount++;
                }
            }
        }
        if (skipErrorCount > 0) {
            int skipCount = skippedObjects.size();
            String s = skipErrorCount == 1 ? "1 nucleus" : skipErrorCount + " nuclei";
            logger.warn("Skipped {} due to error in resolving overlaps ({}% of all skipped)",
                    s, GeneralTools.formatNumber(skipErrorCount * 100.0 / skipCount, 1));
        }
        return new ArrayList<>(objects);
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
     * @param runnable
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
                pool.submit(() -> runnable.run());
            } finally {
                pool.shutdown();
                try {
                    pool.awaitTermination(24, TimeUnit.HOURS);
                } catch (InterruptedException e) {
                    logger.warn("Process was interrupted! " + e.getLocalizedMessage(), e);
                }
            }
        } else {
            runnable.run();
        }
    }

    /**
     * The directory that was used for saving the training images
     *
     * @return
     */
    public File getTrainingDirectory() {
        return trainDirectory;
    }

    /**
     * The directory that was used for saving the validation images
     *
     * @return
     */
    public File getValidationDirectory() {
        return valDirectory;
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
        //runInPool(() -> detectObjectsImpl(imageData, parents));
        // Multi step process
        // 1. Extract all images and save to temp folder
        // 2. Run Cellpose on folder
        // 3. Pick up Label images and convert to PathObjects

        Objects.requireNonNull(parents);

        try {
            FileUtils.cleanDirectory(tempDirectory);
        } catch (IOException e) {
            logger.error("Could not clean temp directory {}", tempDirectory);
            logger.error("Message: ", e);
        }

        PixelCalibration resolution = imageData.getServer().getPixelCalibration();
        if (Double.isFinite(pixelSize) && pixelSize > 0) {
            double downsample = pixelSize / resolution.getAveragedPixelSize().doubleValue();
            resolution = resolution.createScaledInstance(downsample, downsample);
        }

        // The opServer is needed only to get tile requests, or calculate global normalization percentiles
        ImageDataServer<BufferedImage> opServer = ImageOps.buildServer(imageData, op, resolution, tileWidth, tileHeight);
//		var opServer = ImageOps.buildServer(imageData, op, resolution, tileWidth-pad*2, tileHeight-pad*2);


        // Get downsample factor
        double downsample = 1;
        if (Double.isFinite(pixelSize) && pixelSize > 0) {
            downsample = Math.round(pixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize().doubleValue());
        }

        ImageServer<BufferedImage> server = imageData.getServer();
        PixelCalibration cal = server.getPixelCalibration();

        double expansion = cellExpansion / cal.getAveragedPixelSize().doubleValue();

        final double finalDownsample = downsample;

        List<PathTile> allTiles = parents.parallelStream().map(parent -> {
            RegionRequest request = RegionRequest.createInstance(
                    opServer.getPath(),
                    opServer.getDownsampleForResolution(0),
                    parent.getROI());

            // Get all the required tiles that intersect with the mask ROI
            Geometry mask = parent.getROI().getGeometry();

            /*
            // This is the StarDist way, but it makes overly large tiles
            List<TileRequest> tiles = opServer.getTileRequestManager().getTileRequests(request)
                    .stream()
                    .filter(t -> mask == null || mask.intersects(GeometryTools.createRectangle(t.getImageX(), t.getImageY(), t.getImageWidth(), t.getImageHeight())))
                    .collect(Collectors.toList());
            */

            Collection<? extends ROI> rois = RoiTools.computeTiledROIs(parent.getROI(), ImmutableDimension.getInstance((int) (tileWidth * finalDownsample), (int) (tileWidth * finalDownsample)), ImmutableDimension.getInstance((int) (tileWidth * finalDownsample), (int) (tileHeight * finalDownsample)), true, (int) (overlap * finalDownsample));

            List<RegionRequest> tiles = rois.stream().map(r -> {
                return RegionRequest.createInstance(opServer.getPath(), opServer.getDownsampleForResolution(0), r);
            }).collect(Collectors.toList());

            // Detect all potential nuclei
            ImagePlane plane = request.getImagePlane();

            // Compute op with preprocessing
            ArrayList<ImageOp> fullPreprocess = new ArrayList<ImageOp>();
            fullPreprocess.add(ImageOps.Core.ensureType(PixelType.FLOAT32));

            // Do global preprocessing calculations, if required
            if (globalPreprocess != null) {
                try {
                    var normalizeOps = globalPreprocess.createOps(op, imageData, parent.getROI(), request.getImagePlane());
                    fullPreprocess.addAll(normalizeOps);

                    // If this has happened, then we should expect to not use the cellpose normalization?
                    this.parameters.put("no_norm", null);

                } catch (IOException e) {
                    throw new RuntimeException("Exception computing global normalization", e);
                }
            }

            if (!preprocess.isEmpty()) {
                fullPreprocess.addAll(preprocess);
            }
            if (fullPreprocess.size() > 1)
                fullPreprocess.add(ImageOps.Core.ensureType(PixelType.FLOAT32));

            var opWithPreprocessing = op.appendOps(fullPreprocess.toArray(ImageOp[]::new));


            // Keep a reference to the images here while they are being saved
            logger.info("Saving images for {} tiles", tiles.size());

            // Save each tile to an image and keep a reference to it
            var individualTiles = tiles.parallelStream()
                    .map(t -> {
                        // Make a new RegionRequest with the requested padding
                        // Create a padded request, if we need one
                        /*
                        // This is the StarDist way
                        RegionRequest requestPadded = t.getRegionRequest();
                            int x1 = Math.max(0, Math.round(t.getRegionRequest().getX() - finalDownsample * overlap));
                            int y1 = Math.max(0, Math.round(t.getRegionRequest().getY() - finalDownsample * overlap));
                            int x2 = (int)Math.min(server.getWidth(), Math.round(t.getRegionRequest().getMaxX() + finalDownsample * overlap));
                            int y2 = (int)Math.min(server.getHeight(), Math.round(t.getRegionRequest().getMaxY() + finalDownsample * overlap));
                            requestPadded = RegionRequest.createInstance(server.getPath(), finalDownsample, x1, y1, x2-x1, y2-y1, t.getRegionRequest().getZ(), t.getRegionRequest().getT());
                        */
                        try {
                            RegionRequest requestPadded = t;
                            return saveTileImage(opWithPreprocessing, imageData, requestPadded);
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
            runDetection();
        } catch (IOException | InterruptedException e) {
            logger.error("Failed to Run Cellpose", e);
        }

        // Recover all the images from CellPose to get the masks
        double finalDownsample1 = downsample;
        allTiles.parallelStream().forEach(tileMap -> {
            PathObject parent = tileMap.getObject();
            // Read each image
            List<CandidateObject> allCandidates = Collections.synchronizedList(new ArrayList<>());
            tileMap.getTileFiles().parallelStream().forEach(tilefile -> {
                File ori = tilefile.getFile();
                File maskFile = new File(ori.getParent(), FilenameUtils.removeExtension(ori.getName()) + "_cp_masks.tif");
                if (maskFile.exists()) {
                    logger.info("Getting objects for {}", maskFile);

                    // thank you, Pete for the ContourTracing Class
                    Collection<CandidateObject> candidates = null;
                    try {

                        candidates = readObjectsFromFile(maskFile, tilefile.getTile()).stream().filter(c -> parent.getROI().getGeometry().intersects(c.geometry)).collect(Collectors.toList());

                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    logger.info("Getting objects for {} Done", maskFile);

                    allCandidates.addAll(candidates);
                }
            });

            //Remove candidates that are not within the parent objecs
            // Filter Detections: Remove overlaps
            List<CandidateObject> filteredDetections = filterDetections(allCandidates);


            // Remove the detections that are not contained within the parent
            Geometry mask = parent.getROI().getGeometry();
            //filteredDetections = filteredDetections.stream().filter(t -> mask.covers(t.getROI().getGeometry())).collect(Collectors.toList());

            // Convert to detections, dilating to approximate cells if necessary
            // Drop cells if they fail (rather than catastrophically give up)
            List<PathObject> finalObjects = filteredDetections.parallelStream()
                    .map(n -> {
                        try {
                            return convertToObject(n, tileMap.getObject().getROI().getImagePlane(), expansion, constrainToParent ? mask : null);
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
                        ObjectMeasurements.addIntensityMeasurements(server2, cell, finalDownsample1, measurements, compartments);
                    } catch (IOException e) {
                        logger.info(e.getLocalizedMessage(), e);
                    }
                });

            }

            // Assign the objects to the parent object
            parent.setLocked(true);
            parent.clearChildObjects();
            tileMap.getObject().addChildObjects(finalObjects);
        });

        // Update the hierarchy
        imageData.getHierarchy().fireHierarchyChangedEvent(this);

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

            // Intersection with complex mask could give linestrings - which we want to remove
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
        List<CandidateObject> candidateList = rawCandidates.stream().collect(Collectors.toList());
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

            var overlaps = (List<CandidateObject>) tree.query(envelope);
            for (var overlappingCandidate : overlaps) {
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
                    skippedObjects.add(overlappingCandidate);
                    skipErrorCount++;
                }
/*
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

*/

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
    private TileFile saveTileImage(ImageDataOp op, ImageData<BufferedImage> imageData, RegionRequest request) throws IOException {

        // Create a mask around pixels we can use
        var regionMask = GeometryTools.createRectangle(request.getX(), request.getY(), request.getWidth(), request.getHeight());


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
        IJ.save(imp, tempFile.getAbsolutePath());

        return new TileFile(request, tempFile);
    }

    /**
     * Selects the right folder to run from, based on whether it's cellpose or omnipose.
     * Hopefully this will become deprecated soon
     * @return
     */
    private VirtualEnvironmentRunner getVirtualEnvironmentRunner() {
        // Need to decide whether to call cellpose or omnipose
        VirtualEnvironmentRunner veRunner;

        if (this.parameters.containsKey("omni") && !cellposeSetup.getOmniposePytonPath().isEmpty()) {
            veRunner = new VirtualEnvironmentRunner(cellposeSetup.getOmniposePytonPath(), VirtualEnvironmentRunner.EnvType.EXE, this.getClass().getSimpleName());
        } else {
            veRunner = new VirtualEnvironmentRunner(cellposeSetup.getCellposePytonPath(), VirtualEnvironmentRunner.EnvType.EXE, this.getClass().getSimpleName());
        }
        return veRunner;
    }

    /**
     * This class actually runs Cellpose by calling the virtual environment
     *
     * @throws IOException          Exception in case files could not be read
     * @throws InterruptedException Exception in case of command thread has some failing
     */
    private void runDetection() throws IOException, InterruptedException {
        String runCommand = this.parameters.containsKey("omni") ? "omnipose" : "cellpose";
        VirtualEnvironmentRunner veRunner = getVirtualEnvironmentRunner();

        // This is the list of commands after the 'python' call
        List<String> cellposeArguments = new ArrayList<>(Arrays.asList("-W", "ignore", "-m", runCommand));

        cellposeArguments.add("--dir");
        cellposeArguments.add("" + this.tempDirectory);

        cellposeArguments.add("--pretrained_model");
        cellposeArguments.add("" + this.model);

        this.parameters.forEach((parameter, value) -> {
            cellposeArguments.add("--" + parameter);
            if (value != null) {
                cellposeArguments.add(value);
            }
        });

        // These all work for cellpose v2

        cellposeArguments.add("--save_tif");

        cellposeArguments.add("--no_npy");

        cellposeArguments.add("--use_gpu");

        cellposeArguments.add("--verbose");

        veRunner.setArguments(cellposeArguments);

        // Finally, we can run Cellpose
        theLog = veRunner.runCommand();

        logger.info("Cellpose command finished running");
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

            // Clear a previous run
            FileUtils.cleanDirectory(this.trainDirectory);

            saveTrainingImages();

            runTraining();

            this.modelFile = moveAndReturnModelFile();

            // Get the training results before overwriting the log with a new run
            this.trainingResults = parseTrainingResults();

            // Get cellpose masks from the validation
            runCellposeOnValidationImages();

            this.qcResults = runCellposeQC();

            return modelFile;

        } catch (IOException | InterruptedException e) {
            logger.error(e.getMessage(), e);
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
        List<String> cellposeArguments = new ArrayList<>(Arrays.asList("-W", "ignore", "-m", runCommand));

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

        this.parameters.forEach((parameter, value) -> {
            cellposeArguments.add("--" + parameter);
            if (value != null) {
                cellposeArguments.add(value);
            }
        });


        cellposeArguments.add("--use_gpu");

        cellposeArguments.add("--verbose");

        veRunner.setArguments(cellposeArguments);

        // Finally, we can run Cellpose
        theLog = veRunner.runCommand();

        logger.info("Cellpose command finished running");
    }

    /**
     * Make a cellpose run on the validation images (test) folder
     * This will create 'cp_masks' images, which can be read when running QC
     */
    private void runCellposeOnValidationImages() {

        // Assume that cellpose training was already run and run cellpose again on the /test folder
        logger.info("Running the new model {} on the validation images to obtain labels for QC", this.modelFile.getName());

        File tmp = this.tempDirectory;
        this.tempDirectory = this.valDirectory;

        String tmpModel = this.model;
        this.model = modelFile.getAbsolutePath();

        try {
            runDetection();
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
     * @return a results table with the QC statistics
     * @return the results table with the QC metrics or null
     * @throws IOException
     * @throws InterruptedException
     */
    private ResultsTable runCellposeQC() throws IOException, InterruptedException {

        File qcFolder = getQCFolder();

        qcFolder.mkdirs();

        // Let's check if the QC notebook is available in the 'extensions' folder
        File extensionsDir = QuPathGUI.getExtensionDirectory();
        File qcPythonFile = new File(extensionsDir, "run-cellpose-qc.py");

        if (!qcPythonFile.exists()) {
            logger.warn("File {} was not found in {}.\nPlease download it from {}", qcPythonFile.getName(),
                    extensionsDir.getAbsolutePath(),
                    new CellposeExtension().getRepository().getUrlString());
            return null;
        }

        // Start the Virtual Environment Runner
        VirtualEnvironmentRunner qcRunner = getVirtualEnvironmentRunner();
        List<String> qcArguments = new ArrayList<>(Arrays.asList(qcPythonFile.getAbsolutePath(), this.valDirectory.getAbsolutePath(), this.modelFile.getName()));

        qcRunner.setArguments(qcArguments);

        qcRunner.runCommand();

        // The results are stored in the validation directory, open them as a results table
        File qcResults = new File(this.valDirectory, "QC-Results" + File.separator + "Quality_Control for " + this.modelFile.getName() + ".csv");

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
    public String[] getOutputLog() {
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
            // Look for "Epoch 0, Time  2.3s, Loss 1.0758, Loss Test 0.6007, LR 0.2000"
            String epochPattern = ".*Epoch\\s*(\\d+),\\s*Time\\s*(\\d+\\.\\d)s,\\s*Loss\\s*(\\d+\\.\\d+),\\s*Loss Test\\s*(\\d+\\.\\d+),\\s*LR\\s*(\\d+\\.\\d+).*";
            // Build Matcher
            Pattern pattern = Pattern.compile(epochPattern);
            Matcher m;
            for (String line : this.theLog) {
                m = pattern.matcher(line);
                if (m.find()) {
                    trainingResults.incrementCounter();
                    trainingResults.addValue("Epoch", Double.parseDouble(m.group(1)));
                    trainingResults.addValue("Time[s]", Double.parseDouble(m.group(2)));
                    trainingResults.addValue("Loss", Double.parseDouble(m.group(3)));
                    trainingResults.addValue("Loss Test", Double.parseDouble(m.group(4)));
                    trainingResults.addValue("LR", Double.parseDouble(m.group(5)));
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
            lossTest.getData().add(new XYChart.Data<>(output.getValue("Epoch", i), output.getValue("Loss Test", i)));

        }
        lineChart.getData().add(loss);
        lineChart.getData().add(lossTest);
        GuiTools.runOnApplicationThread(() -> {
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
     * @return
     */
    private File getQCFolder() {
        File projectFolder = QP.getProject().getPath().getParent().toFile();
        File qcFolder = new File(projectFolder, "QC");
        return qcFolder;
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
        int downsample = 1;
        if (Double.isFinite(pixelSize) && pixelSize > 0) {
            downsample = (int) Math.round(pixelSize / originalServer.getPixelCalibration().getAveragedPixelSize().doubleValue());
        }

        AtomicInteger idx = new AtomicInteger();
        int finalDownsample = downsample;

        annotations.forEach(a -> {
            int i = idx.getAndIncrement();

            RegionRequest request = RegionRequest.createInstance(originalServer.getPath(), finalDownsample, a.getROI());
            File imageFile = new File(saveDirectory, imageName + "_region_" + i + ".tif");
            File maskFile = new File(saveDirectory, imageName + "_region_" + i + "_masks.tif");
            try {

                ImageWriterTools.writeImageRegion(originalServer, request, imageFile.getAbsolutePath());
                ImageWriterTools.writeImageRegion(labelServer, request, maskFile.getAbsolutePath());
                logger.info("Saved image pair: \n\t{}\n\t{}", imageFile.getName(), maskFile.getName());

            } catch (IOException ex) {
                logger.error(ex.getMessage());
            }
        });
    }

    /**
     * Save training images for the project
     */
    public void saveTrainingImages() {

        Project<BufferedImage> project = QPEx.getQuPath().getProject();

        project.getImageList().forEach(e -> {

            ImageData<BufferedImage> imageData;
            try {
                imageData = e.readImageData();
                String imageName = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName());

                Collection<PathObject> allAnnotations = imageData.getHierarchy().getAnnotationObjects();
                // Get Squares for Training, Validation and Testing
                List<PathObject> trainingAnnotations = allAnnotations.stream().filter(a -> a.getPathClass() == PathClassFactory.getPathClass("Training")).collect(Collectors.toList());
                List<PathObject> validationAnnotations = allAnnotations.stream().filter(a -> a.getPathClass() == PathClassFactory.getPathClass("Validation")).collect(Collectors.toList());

                List<PathObject> testingAnnotations = allAnnotations.stream().filter(a -> a.getPathClass() == PathClassFactory.getPathClass("Test")).collect(Collectors.toList());

                PixelCalibration resolution = imageData.getServer().getPixelCalibration();
                if (Double.isFinite(pixelSize) && pixelSize > 0) {
                    double downsample = pixelSize / resolution.getAveragedPixelSize().doubleValue();
                    resolution = resolution.createScaledInstance(downsample, downsample);
                }

                logger.info("Found {} Training objects and {} Validation objects in image {}", trainingAnnotations.size(), validationAnnotations.size(), imageName);

                if (!trainingAnnotations.isEmpty() || !validationAnnotations.isEmpty()) {
                    // Make the server using the required ops
                    // Do global preprocessing calculations, if required
                    ArrayList<ImageOp> fullPreprocess = new ArrayList<ImageOp>();
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

                    op = op.appendOps(fullPreprocess.toArray(ImageOp[]::new));

                    ImageServer<BufferedImage> processed = ImageOps.buildServer(imageData, op, resolution, tileWidth, tileHeight);

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
    private File moveAndReturnModelFile() throws IOException {
        File cellPoseModelFolder = new File(trainDirectory, "models");
        // Find the first file in there
        File[] all = cellPoseModelFolder.listFiles();
        Optional<File> cellPoseModel = Arrays.stream(Objects.requireNonNull(all)).filter(f -> f.getName().contains("cellpose")).findFirst();
        if (cellPoseModel.isPresent()) {
            File model = cellPoseModel.get();
            logger.info("Found model file at {} ", model);
            File newModel = new File(modelDirectory, model.getName());
            FileUtils.copyFile(model, newModel);
            logger.info("Model file {} moved to {}", newModel.getName(), newModel.getParent());
            return newModel;
        }
        return null;
    }

    private Collection<CandidateObject> readObjectsFromFile(File maskFile, RegionRequest region) throws IOException {
        BufferedImage bfImage = ImageIO.read(maskFile);
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
                Geometry geometry = ContourTracing.createTracedGeometry(image, p, p, region);
                if (geometry != null && !geometry.isEmpty())
                    candidates.put(p, new CandidateObject(geometry));
                lastLabel = p;
            }
        }
        // Ignore the IDs, because they will be the same across different images, and we don't really need them
        return candidates.values();
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

    private static class CandidateObject {
        private final double area;
        private Geometry geometry;

        CandidateObject(Geometry geom) {
            this.geometry = geom;
            this.area = geom.getArea();

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