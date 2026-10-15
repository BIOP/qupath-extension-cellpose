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

/**
 * Two interchangeable ways to run Cellpose, behind {@link qupath.ext.biop.cellpose.backend.CellposeBackend}.
 * <p>
 * {@link qupath.ext.biop.cellpose.backend.SubprocessBackend} launches a user-installed Python and
 * exchanges TIFFs on disk. {@link qupath.ext.biop.cellpose.backend.ApposeBackend} runs Cellpose in a
 * Python worker that Appose keeps alive, exchanging pixels through shared memory: build the pixi
 * environment ({@link qupath.ext.biop.cellpose.backend.ApposeEnvironments}), start a worker per
 * model-family and device sub-environment, load the model once with {@code cpX_init.py}, then run
 * {@code cpX.py} per tile, marshalling pixels in and labels out through
 * {@link qupath.ext.biop.cellpose.backend.NDArrays}.
 * <p>
 * Detection only: training and QC always take the subprocess path.
 *
 * @see <a href="https://github.com/BIOP/qupath-extension-cellpose#in-process-appose-backend">README: In-process (Appose) backend</a>
 */
package qupath.ext.biop.cellpose.backend;
