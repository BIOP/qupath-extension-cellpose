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
 * The compute device the in-process {@link ApposeBackend} should run Cellpose on. It controls both
 * which pixi sub-environment is activated (a CUDA-enabled {@code cuNNN} variant vs the {@code cpu}
 * variant) and the {@code use_gpu} flag handed to the Cellpose scripts.
 */
public enum CellposeDevice {

    /** Detect a usable GPU automatically (the default), falling back to CPU when none is found. */
    AUTO("Auto - use a GPU if one is available"),

    /** Activate the CUDA sub-environment and request GPU execution regardless of detection. */
    GPU("GPU - always use the GPU"),

    /** Activate the CPU sub-environment and request CPU execution. */
    CPU("CPU - never use the GPU");

    private final String label;

    CellposeDevice(String label) {
        this.label = label;
    }

    /**
     * @return the label shown in the preferences, rather than the constant name
     */
    @Override
    public String toString() {
        return label;
    }
}
