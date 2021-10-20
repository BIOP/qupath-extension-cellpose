package qupath.ext.biop.cellpose;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.biop.cmd.VirtualEnvironmentRunner;
import qupath.lib.common.ColorTools;
import qupath.lib.common.GeneralTools;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.*;
import qupath.lib.images.writers.ImageWriterTools;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.classes.PathClassFactory;
import qupath.lib.projects.Project;
import qupath.lib.regions.RegionRequest;
import qupath.lib.scripting.QP;
import qupath.opencv.ops.ImageDataOp;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class Cellpose2DTraining {

    private final static Logger logger = LoggerFactory.getLogger(Cellpose2DTraining.class);
    private String pretrainedModel = "cyto";
    private File modelDirectory;
    private File trainDirectory;
    private File valDirectory;
    private double diameter;
    private int nChannels;
    private int nEpochs;
    private ImageDataOp op;
    private double pixelSize;

    private void saveImagePairs(List<PathObject> annotations, String imageName, ImageServer<BufferedImage> originalServer, ImageServer<BufferedImage> labelServer, File saveDirectory) {

        if(annotations.isEmpty()) { return; }
        int downsample = 1;
        if (Double.isFinite(pixelSize) && pixelSize > 0) {
            downsample = (int) Math.round(pixelSize / originalServer.getPixelCalibration().getAveragedPixelSize().doubleValue());
        }

        AtomicInteger idx = new AtomicInteger();
        int finalDownsample = downsample;

        logger.info("Saving Images...");
        annotations.parallelStream().forEach(a -> {
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

    private void saveTrainingImages() throws IOException {

        Project<BufferedImage> project = QP.getProject();
        // Prepare location to save images

        project.getImageList().stream().forEach(e -> {

            ImageData<BufferedImage> imageData = null;
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
                logger.error( ex.getMessage());
            }
        });

    }

    private File moveAndReturnModelFile() throws IOException {
        File cellPoseModelFolder = new File(trainDirectory, "models");
        // Find the first file in there
        File[] all = cellPoseModelFolder.listFiles();
        Optional<File> cellPoseModel = Arrays.stream(all).filter(f -> f.getName().contains("cellpose")).findFirst();
        if(cellPoseModel.isPresent()) {
            logger.info("Found model file at {} ", cellPoseModel);
            File model = cellPoseModel.get();
            File newModel = new File(modelDirectory, model.getName());
            FileUtils.copyFile(model, newModel);
            return newModel;
        }
        return null;
    }
    /**
     * Builder to help create a {@link Cellpose2D} with custom parameters.
     */
    public static class Builder {

        private File modelDirectory;
        private ColorTransforms.ColorTransform[] channels = new ColorTransforms.ColorTransform[0];
        private String pretrainedModel = "cyto";

        private double diameter = 100;
        private int nEpochs;

        private double pixelSize = Double.NaN;

        private List<ImageOp> ops = new ArrayList<>();

        private Builder(String pretrainedModel) {
            this.pretrainedModel = pretrainedModel;
            this.ops.add(ImageOps.Core.ensureType(PixelType.FLOAT32));
        }

        Builder modelDirectory( File modelDirectory) {
            this.modelDirectory = modelDirectory;
            return this;
        }

        Builder epochs(int nEpochs) {
            this.nEpochs = nEpochs;
            return this;
        }

        /**
         * The extimated diameter of the objects to detect. Cellpose will further downsample the images in order to match
         * their expected diameter
         *
         * @param diameter in pixels
         * @return
         */
        public Builder diameter(double diameter) {
            this.diameter = diameter;
            return this;
        }

        /**
         * Add preprocessing operations, if required.
         *
         * @param ops
         * @return this builder
         */
        public Builder preprocess(ImageOp... ops) {
            for (var op : ops)
                this.ops.add(op);
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
                    .mapToObj(c -> ColorTransforms.createChannelExtractor(c))
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
        public Builder channels(String... channels) {
            return channels(Arrays.stream(channels)
                    .map(c -> ColorTransforms.createChannelExtractor(c))
                    .toArray(ColorTransforms.ColorTransform[]::new));
        }

        /**
         * Define the channels (or color transformers) to apply to the input image.
         * <p>
         * This makes it possible to supply color deconvolved channels, for example.
         *
         * @param channels
         * @return this builder
         */
        public Builder channels(ColorTransforms.ColorTransform... channels) {
            this.channels = channels.clone();
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
         * @param pixelSize
         * @return this builder
         */
        public Builder pixelSize(double pixelSize) {
            this.pixelSize = pixelSize;
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
         * Create a {@link Cellpose2D}, all ready for detection.
         *
         * @return
         */
        public Cellpose2DTraining build() {

            // Directory to move trained models.
            if( modelDirectory == null) {
                modelDirectory = new File(QP.getProject().getPath().getParent().toFile(), "models");
            }

            // Define training and validation directories inside the QuPath Project\
            File groundTruthDirectory = new File(QP.getProject().getPath().getParent().toFile(), "cellpose-training");

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
            // The cellpose class that will run the detections
            Cellpose2DTraining cellpose = new Cellpose2DTraining();

            ArrayList<ImageOp> mergedOps = new ArrayList<>(ops);

            cellpose.pretrainedModel = pretrainedModel;

            // Add all operations (preprocessing, channel extraction and normalization)

            // Ensure there are only 2 channels at most
            if (channels.length > 2) {
                logger.warn("You supplied {} channels, but Cellpose needs two channels at most. Keeping the first two", channels.length);
                channels = Arrays.copyOf(channels, 2);
            }
            mergedOps.add(ImageOps.Core.ensureType(PixelType.FLOAT32));

            cellpose.op =  ImageOps.buildImageDataOp(channels)
                    .appendOps(mergedOps.toArray(ImageOp[]::new));

            // CellPose accepts either one or two channels. This will help the final command
            cellpose.nChannels = channels.length;
            cellpose.pixelSize = pixelSize;
            cellpose.diameter = diameter;
            cellpose.nEpochs = nEpochs;
            cellpose.trainDirectory = trainDirectory;
            cellpose.valDirectory = valDirectory;
            cellpose.modelDirectory = modelDirectory;

            return cellpose;
        }

    }

    public File train() {

        try {
            saveTrainingImages();
            runCellPose();
            File model = moveAndReturnModelFile();

            return model;
        } catch (IOException | InterruptedException e) {
            logger.error(e.getMessage(), e);
        }
        return null;

    }
    public void runCellPose() throws IOException, InterruptedException {

        //python -m cellpose --train --dir ~/images_cyto/train/ --test_dir ~/images_cyto/test/ --pretrained_model cyto --chan 2 --chan2 1
        // Get options
        CellposeOptions cellposeOptions = CellposeOptions.getInstance();

        // Create command to run
        VirtualEnvironmentRunner veRunner = new VirtualEnvironmentRunner(cellposeOptions.getEnvironmentNameorPath(), cellposeOptions.getEnvironmentType());

        // This is the list of commands after the 'python' call
        List<String> cellposeArguments = new ArrayList<>();

        cellposeArguments.addAll(Arrays.asList("-W", "ignore", "-m", "cellpose"));

        cellposeArguments.add("--train");

        cellposeArguments.add("--dir");
        cellposeArguments.add("" + trainDirectory.getAbsolutePath());
        cellposeArguments.add("--test_dir");
        cellposeArguments.add("" + valDirectory.getAbsolutePath());

        cellposeArguments.add("--pretrained_model");
        if(pretrainedModel != "") {
            cellposeArguments.add("" + pretrainedModel);
        } else {
            cellposeArguments.add("None");
        }

        // The channel order will always be 1 and 2, in the order defined by channels(...) in the builder
        cellposeArguments.add("--chan");
        cellposeArguments.add("1");

        if (nChannels > 1) {
            cellposeArguments.add("--chan2");
            cellposeArguments.add("2");
        }

        cellposeArguments.add("--diameter");
        cellposeArguments.add("" + diameter);

        cellposeArguments.add("--n_epochs");
        cellposeArguments.add("" + nEpochs);

        if (cellposeOptions.useGPU()) cellposeArguments.add("--use_gpu");

        veRunner.setArguments(cellposeArguments);

        // Finally, we can run Cellpose
        veRunner.runCommand();

        logger.info("Cellpose command finished running");
    }
}