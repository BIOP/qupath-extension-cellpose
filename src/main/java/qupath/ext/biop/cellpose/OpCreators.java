/*-
 * Copyright 2022 QuPath developers, University of Edinburgh
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

import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.awt.common.BufferedImageTools;
import qupath.lib.images.ImageData;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.interfaces.ROI;
import qupath.opencv.ops.ImageDataOp;
import qupath.opencv.ops.ImageOp;
import qupath.opencv.ops.ImageOps;
import qupath.opencv.ops.ImageOps.Normalize;
import qupath.opencv.tools.OpenCVTools;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Helper class for creating new {@linkplain ImageOp ImageOps} based upon other image properties.
 * <p>
 * This addresses that problem that every {@link ImageOp} only knows about the image tile that it
 * 'sees' at runtime.
 * This means that all processing needs to be local.
 * <p>
 * Often, we want ops to use information from across the entire image - particularly for
 * normalization as a step in preprocessing, such as when normalizing to zero mean and unit variance
 * across the entire image.
 * <p>
 * Before this class, this was problematic because either the parameters needed to be calculated
 * elsewhere (which was awkward), or else normalization would always treat each image tile independent -
 * which could result in tiles within the same image being normalized in very different ways.
 *
 * @author Pete Bankhead
 * @implNote This is currently in development. If it proves useful enough, it is likely to be
 * refined and moved to the core QuPath software.
 * @since v0.4.0
 */
public class OpCreators {

    /**
     * Build a normalization op that can be based upon the entire (2D) image, rather than only local tiles.
     * <p>
     * Note that currently this requires downsampling the image to a manageable size.
     *
     * @return
     */
    public static ImageNormalizationBuilder imageNormalizationBuilder() {
        return new ImageNormalizationBuilder();
    }

    /**
     * Helper class for creating (tile-based) ImageOps with parameters that are derived from an entire image or ROI.
     * <p>
     * This is most useful for normalization, where statistics may need to be calculated across the image
     * even if they are then applied locally (e.g. an offset and scaling factor).
     *
     * @author Pete Bankhead
     */
    public interface TileOpCreator {

        /**
         * Compute the (tile-based) ops from the image.
         *
         * @param op        the data op, which determines how to extract channels from the image data
         * @param imageData the image data to process
         * @param mask      ROI mask that may be used to restrict the region being considered (optional)
         * @param plane     the 2D image plane to use; if not provided, the plane from any ROI will be used, or otherwise the default plane
         * @return
         * @throws IOException
         */
        List<ImageOp> createOps(ImageDataOp op, ImageData<BufferedImage> imageData, ROI mask, ImagePlane plane) throws IOException;

    }

    abstract static class DownsampledOpCreator implements TileOpCreator {

        private static final Logger logger = LoggerFactory.getLogger(DownsampledOpCreator.class);

        private boolean useMask = false;

        private double downsample = Double.NaN;
        private int maxDimension = 2048;

        DownsampledOpCreator(int maxDimension, double downsample, boolean useMask) {
            this.maxDimension = maxDimension;
            this.downsample = downsample;
            this.useMask = useMask;
        }

        DownsampledOpCreator() {
            this(2048, Double.NaN, false);
        }

        @Override
        public List<ImageOp> createOps(ImageDataOp op, ImageData<BufferedImage> imageData, ROI mask, ImagePlane plane) throws IOException {
            var server = imageData.getServer();
            double downsample = this.downsample;

            int x = 0, y = 0, width = server.getWidth(), height = server.getHeight();
            if (useMask && mask != null) {
                x = (int) Math.round(mask.getBoundsX());
                y = (int) Math.round(mask.getBoundsY());
                width = (int) Math.round(mask.getBoundsWidth());
                height = (int) Math.round(mask.getBoundsHeight());
            }
            if (plane == null) {
                if (mask == null) {
                    logger.warn("Plane not specified - will use the default plane");
                    plane = ImagePlane.getDefaultPlane();
                } else {
                    logger.debug("Plane not specified - will use the ROI mask plane");
                    plane = mask.getImagePlane();
                }
            }


            if (Double.isNaN(downsample)) {
                downsample = Math.max(width, height) / (double) maxDimension;
                downsample = Math.max(downsample, 1.0);
                logger.info("Computed downsample: {}", downsample);
            }

            var request = RegionRequest.createInstance(server.getPath(), downsample,
                    x, y, width, height, plane.getZ(), plane.getT());

            try (var scope = new PointerScope()) {
                var mat = op.apply(imageData, request);

                if (useMask && mask != null) {
                    var img = BufferedImageTools.createROIMask(mat.cols(), mat.rows(), mask, request);
                    var matMask = OpenCVTools.imageToMat(img);
                    opencv_core.bitwise_not(matMask, matMask);
                    if (mat.depth() != opencv_core.CV_32F && mat.depth() != opencv_core.CV_64F)
                        mat.convertTo(mat, opencv_core.CV_32F);
                    mat.setTo(OpenCVTools.scalarMat(Double.NaN, mat.depth()), matMask);
                    matMask.close();
                    // Show image for debugging
//					OpenCVTools.matToImagePlus("Masked input", mat).show();
                }

                return compute(mat);
            }
        }

        protected abstract List<ImageOp> compute(Mat mat);

    }

    /**
     * Tile op creator that computes offset and scale values across the full image
     * to normalize using min and max percentiles.
     */
    public static class PercentileTileOpCreator extends DownsampledOpCreator {

        private static final Logger logger = LoggerFactory.getLogger(PercentileTileOpCreator.class);

        private double percentileMin = 0;
        private double percentileMax = 99.8;
        private boolean perChannel = false;

        private double eps = 1e-6;

        private PercentileTileOpCreator(int maxSize, double downsample, boolean useMask, double percentileMin, double percentileMax, boolean perChannel, double eps) {
            super(maxSize, downsample, useMask);
            this.percentileMin = percentileMin;
            this.percentileMax = percentileMax;
            this.perChannel = perChannel;
            this.eps = eps;
        }

        @Override
        protected List<ImageOp> compute(Mat mat) {
            if (perChannel) {
                int nChannels = mat.channels();
                double[] toSubtract = new double[nChannels];
                double[] toScale = new double[nChannels];
                int c = 0;
                try (var scope = new PointerScope()) {
                    for (var matChannel : OpenCVTools.splitChannels(mat)) {
                        double[] percentiles = OpenCVTools.percentiles(matChannel, percentileMin, percentileMax);
                        toSubtract[c] = percentiles[0];
                        toScale[c] = 1.0 / Math.max(percentiles[1] - percentiles[0], eps);
                        c++;
                    }
                }
                logger.info("Computed percentile normalization offsets={}, scales={}", Arrays.toString(toSubtract), Arrays.toString(toScale));
                return List.of(
                        ImageOps.Core.subtract(toSubtract),
                        ImageOps.Core.multiply(toScale));
            } else {
                double[] percentiles = OpenCVTools.percentiles(mat, percentileMin, percentileMax);
                logger.info("Computed percentiles {}, {}", percentiles[0], percentiles[1]);
                return List.of(
                        ImageOps.Core.subtract(percentiles[0]),
                        ImageOps.Core.multiply(1.0 / Math.max(percentiles[1] - percentiles[0], 1e-6)));
            }
        }

    }

    /**
     * Tile op creator that computes offset and scale values across the full image
     * to normalize to zero mean and unit variance.
     */
    public static class ZeroMeanVarianceTileOpCreator extends DownsampledOpCreator {

        private static final Logger logger = LoggerFactory.getLogger(ZeroMeanVarianceTileOpCreator.class);

        private boolean perChannel = false;
        private double eps = 1e-6;

        private ZeroMeanVarianceTileOpCreator(int maxSize, double downsample, boolean useMask, boolean perChannel, double eps) {
            super(maxSize, downsample, useMask);
            this.perChannel = perChannel;
            this.eps = eps;
        }

        @Override
        protected List<ImageOp> compute(Mat mat) {
            if (perChannel) {
                int nChannels = mat.channels();
                double[] toSubtract = new double[nChannels];
                double[] toScale = new double[nChannels];
                int c = 0;
                try (var scope = new PointerScope()) {
                    for (var matChannel : OpenCVTools.splitChannels(mat)) {
                        toSubtract[c] = OpenCVTools.mean(matChannel);
                        toScale[c] = 1.0 / (OpenCVTools.stdDev(matChannel) + eps);
                        c++;
                    }
                }
                logger.info("Computed mean/variance normalization offsets={}, scales={}", Arrays.toString(toSubtract), Arrays.toString(toScale));
                return List.of(
                        ImageOps.Core.subtract(toSubtract),
                        ImageOps.Core.multiply(toScale)
                );
            } else {
                double toSubtract = OpenCVTools.mean(mat);
                double toScale = 1.0 / (OpenCVTools.stdDev(mat) + eps);
                logger.info("Computed mean/variance normalization offset={}, scale={}", toSubtract, toScale);
                return List.of(
                        ImageOps.Core.subtract(toSubtract),
                        ImageOps.Core.multiply(toScale)
                );
            }
        }

    }

    /**
     * Builder for a {@link TileOpCreator} that can be used for image preprocessing
     * using min/max percentiles or zero-mean-unit-variance normalization.
     */
    public static class ImageNormalizationBuilder {

        private static final Logger logger = LoggerFactory.getLogger(ImageNormalizationBuilder.class);

        private boolean zeroMeanUnitVariance = false;

        private double minPercentile = 0;
        private double maxPercentile = 100;

        private boolean perChannel = false;
        private double eps = 1e-6; // 1e-6 - update javadoc if this changes

        private double downsample = Double.NaN;
        private int maxDimension = 2048;    // 2048 - update javadoc if this changes
        private boolean useMask = false;

        /**
         * Specify min and max percentiles to calculate normalization values.
         * See {@link Normalize#percentile(double, double)}.
         *
         * @param minPercentile
         * @param maxPercentile
         * @return this builder
         */
        public ImageNormalizationBuilder percentiles(double minPercentile, double maxPercentile) {
            this.minPercentile = minPercentile;
            this.maxPercentile = maxPercentile;
            if (zeroMeanUnitVariance) {
                logger.warn("Specifying percentiles overrides previous zero-mean-unit-variance request");
                zeroMeanUnitVariance = false;
            }
            return this;
        }

        /**
         * Error constant used for numerical stability and avoid dividing by zero.
         * Default is 1e-6;
         *
         * @param eps
         * @return this builder
         */
        public ImageNormalizationBuilder eps(double eps) {
            this.eps = eps;
            return this;
        }

        /**
         * Compute the normalization values separately per channel; if false, values are computed
         * jointly across channels.
         *
         * @param perChannel
         * @return this builder
         */
        public ImageNormalizationBuilder perChannel(boolean perChannel) {
            this.perChannel = perChannel;
            return this;
        }

        /**
         * Specify the downsample factor to use when calculating the normalization.
         * If this is not provided, then {@link #maxDimension(int)} will be used to calculate
         * a downsample value automatically.
         * <p>
         * The downsample should be &geq; 1.0 and high enough to ensure that the entire image
         * can be fit in memory. A downsample of 1.0 for a whole slide image will probably
         * fail due to memory or array size limits.
         *
         * @param downsample
         * @return this builder
         * see {@link #maxDimension(int)}
         */
        public ImageNormalizationBuilder downsample(double downsample) {
            this.downsample = downsample;
            return this;
        }

        /**
         * The maximum width or height, which is used to calculate a downsample factor for
         * the image if {@link #downsample(double)} is not specified.
         * <p>
         * The current default value is 2048;
         *
         * @param maxDimension
         * @return this builder
         */
        public ImageNormalizationBuilder maxDimension(int maxDimension) {
            this.maxDimension = maxDimension;
            return this;
        }

        /**
         * Optionally use any ROI mask provided for the calculation.
         * This can restrict the region that is considered.
         *
         * @param useMask
         * @return this builder
         */
        public ImageNormalizationBuilder useMask(boolean useMask) {
            this.useMask = useMask;
            return this;
        }

        /**
         * Normalize for zero mean and unit variance.
         * This is an alternative to using {@link #percentiles(double, double)}.
         *
         * @return this builder
         */
        public ImageNormalizationBuilder zeroMeanUnitVariance() {
            return zeroMeanUnitVariance(true);
        }

        /**
         * Optionally normalize for zero mean and unit variance.
         * This is an alternative to using {@link #percentiles(double, double)}.
         *
         * @param doZeroMeanUnitVariance
         * @return this builder
         */
        public ImageNormalizationBuilder zeroMeanUnitVariance(boolean doZeroMeanUnitVariance) {
            this.zeroMeanUnitVariance = doZeroMeanUnitVariance;
            if (zeroMeanUnitVariance && (minPercentile != 0 || maxPercentile != 100))
                logger.warn("Setting zero-mean-unit-variance will override previous percentiles that were set");
            return this;
        }

        /**
         * Build a {@link TileOpCreator} according to the builder's parameters.
         *
         * @return this builder
         */
        public TileOpCreator build() {
            if (zeroMeanUnitVariance) {
                logger.debug("Creating zero-mean-unit-variance normalization op");
                return new ZeroMeanVarianceTileOpCreator(maxDimension, downsample, useMask, perChannel, eps);
            } else {
                logger.debug("Creating percentile normalization op");
                return new PercentileTileOpCreator(maxDimension, downsample, useMask, minPercentile, maxPercentile, perChannel, eps);
            }
        }

    }

}
