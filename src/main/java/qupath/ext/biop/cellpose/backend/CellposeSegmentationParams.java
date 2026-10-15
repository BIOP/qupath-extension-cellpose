/*-
 * Copyright 2026 BioImaging & Optics Platform (BIOP), Ecole Polytechnique Federale de Lausanne
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

package qupath.ext.biop.cellpose.backend;

/**
 * Immutable value object describing the Cellpose parameters the {@link ApposeBackend} needs,
 * decoupled from the {@code CellposeBuilder}/{@code Cellpose2D} internals. It carries only the
 * subset of Cellpose options the in-process transport supports.
 */
public final class CellposeSegmentationParams {

    private final String model;
    private final boolean customModel;
    private final boolean cellposeSam;
    private final Integer channel1;
    private final Integer channel2;
    private final double diameter;
    private final double flowThreshold;
    private final double cellprobThreshold;
    private final boolean do3D;
    private final CellposeDevice device;

    private CellposeSegmentationParams(Builder builder) {
        this.model = builder.model;
        this.customModel = builder.customModel;
        this.cellposeSam = builder.cellposeSam;
        this.channel1 = builder.channel1;
        this.channel2 = builder.channel2;
        this.diameter = builder.diameter;
        this.flowThreshold = builder.flowThreshold;
        this.cellprobThreshold = builder.cellprobThreshold;
        this.do3D = builder.do3D;
        this.device = builder.device;
    }

    /**
     * @return the model name (for a built-in model) or path (for a custom model)
     */
    public String getModel() {
        return model;
    }

    /**
     * @return true if {@link #getModel()} is a path to a custom model, false if it is a built-in model name
     */
    public boolean isCustomModel() {
        return customModel;
    }

    /**
     * @return true to use the Cellpose-SAM (Cellpose 4) family, false for the Cellpose 3 family
     */
    public boolean isCellposeSam() {
        return cellposeSam;
    }

    /**
     * @return cellpose's {@code --chan} value (1-based, 0 for grayscale), or null if unset
     */
    public Integer getChannel1() {
        return channel1;
    }

    /**
     * @return cellpose's {@code --chan2} value (1-based, 0 for grayscale), or null if unset
     */
    public Integer getChannel2() {
        return channel2;
    }

    /** @return the expected object diameter in PIXELS at the analysis resolution, or 0 to let Cellpose estimate it */
    public double getDiameter() {
        return diameter;
    }

    public double getFlowThreshold() {
        return flowThreshold;
    }

    public double getCellprobThreshold() {
        return cellprobThreshold;
    }


    public boolean isDo3D() {
        return do3D;
    }

    /**
     * @return the requested compute device (never null; defaults to {@link CellposeDevice#AUTO})
     */
    public CellposeDevice getDevice() {
        return device;
    }

    /**
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for {@link CellposeSegmentationParams}.
     */
    public static final class Builder {
        private String model = "cyto3";
        private boolean customModel = false;
        private boolean cellposeSam = false;
        private Integer channel1 = 0;
        private Integer channel2 = null;
        private double diameter = 30.0;
        private double flowThreshold = 0.4;
        private double cellprobThreshold = 0.0;
        private boolean do3D = false;
        private CellposeDevice device = CellposeDevice.AUTO;

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder customModel(boolean customModel) {
            this.customModel = customModel;
            return this;
        }

        public Builder cellposeSam(boolean cellposeSam) {
            this.cellposeSam = cellposeSam;
            return this;
        }

        public Builder channel1(Integer channel1) {
            this.channel1 = channel1;
            return this;
        }

        public Builder channel2(Integer channel2) {
            this.channel2 = channel2;
            return this;
        }

        public Builder diameter(double diameter) {
            this.diameter = diameter;
            return this;
        }

        public Builder flowThreshold(double flowThreshold) {
            this.flowThreshold = flowThreshold;
            return this;
        }

        public Builder cellprobThreshold(double cellprobThreshold) {
            this.cellprobThreshold = cellprobThreshold;
            return this;
        }


        public Builder do3D(boolean do3D) {
            this.do3D = do3D;
            return this;
        }

        /**
         * @param device the compute device to request; {@code null} is treated as
         *               {@link CellposeDevice#AUTO}
         * @return this builder
         */
        public Builder device(CellposeDevice device) {
            this.device = device == null ? CellposeDevice.AUTO : device;
            return this;
        }

        public CellposeSegmentationParams build() {
            return new CellposeSegmentationParams(this);
        }
    }
}
