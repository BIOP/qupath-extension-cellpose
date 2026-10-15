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

/** The transport used to run Cellpose from within QuPath. */
public enum CellposeTransport {

    /**
     * Run Cellpose as an external Python process, exchanging images through temporary TIFF files.
     * This is the default.
     */
    SUBPROCESS("User's Python installation - standard"),

    /** Run Cellpose in-process through Appose, exchanging images through shared memory. */
    APPOSE("Appose - runs faster, self-setup");

    private final String label;

    CellposeTransport(String label) {
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
