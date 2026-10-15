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

import qupath.ext.biop.cellpose.TileFile;

import java.io.IOException;
import java.util.List;

/**
 * A pluggable backend that runs Cellpose segmentation on a set of input tiles and makes the
 * resulting detections available on each tile.
 * <p>
 * Two implementations are provided: {@link SubprocessBackend}, the default, which exchanges images
 * with an external Python process through temporary TIFF files, and {@link ApposeBackend}, an
 * opt-in in-process transport based on Appose. Whichever is used, the contract is the same: after
 * {@link #run(List, CellposeSegmentationParams)} returns, every tile has had its candidate
 * detections populated through the tile reader the owning {@code Cellpose2D} supplied.
 */
public interface CellposeBackend extends AutoCloseable {

    /**
     * Run segmentation on the given tiles, populating each tile's candidate detections.
     *
     * @param tiles  the tiles to process; a {@code null} value is permitted for the
     *               "run-cellpose-only" case used by training/validation, in which no detections
     *               are read back
     * @param params the segmentation parameters (used by the Appose backend; the subprocess
     *               backend builds its command from the raw Cellpose flag map instead and ignores
     *               this argument)
     * @throws IOException          if reading/writing images or communicating with Python fails
     * @throws InterruptedException if the running thread is interrupted
     */
    void run(List<TileFile> tiles, CellposeSegmentationParams params) throws IOException, InterruptedException;

    /** Release any resources held by this backend. */
    @Override
    void close();

    /**
     * Resolve which transport to use: the builder-level override if set, then the extension-wide
     * preference, then {@link CellposeTransport#SUBPROCESS}.
     *
     * @param builderTransport the transport set on the builder, or null if unset
     * @param preference       the extension-wide preference, or null
     * @return the resolved transport (never null)
     */
    static CellposeTransport resolveTransport(CellposeTransport builderTransport, CellposeTransport preference) {
        if (builderTransport != null)
            return builderTransport;
        if (preference != null)
            return preference;
        return CellposeTransport.SUBPROCESS;
    }

    /**
     * Resolve which compute device the in-process backend should target: the builder-level override
     * if set, then the extension-wide preference, then {@link CellposeDevice#AUTO}.
     *
     * @param builderDevice the device set on the builder, or null if unset
     * @param preference    the extension-wide preference, or null
     * @return the resolved device (never null)
     */
    static CellposeDevice resolveDevice(CellposeDevice builderDevice, CellposeDevice preference) {
        if (builderDevice != null)
            return builderDevice;
        if (preference != null)
            return preference;
        return CellposeDevice.AUTO;
    }
}
