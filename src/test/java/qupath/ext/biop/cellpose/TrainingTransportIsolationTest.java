/*-
 * Copyright 2026 QuPath developers
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

import org.junit.jupiter.api.Test;
import qupath.ext.biop.cellpose.backend.CellposeBackend;
import qupath.ext.biop.cellpose.backend.CellposeTransport;
import qupath.ext.biop.cellpose.backend.SubprocessBackend;

import java.util.LinkedHashMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

/**
 * Guards the invariant that the training and QC paths always run through the external subprocess
 * transport, whatever detection transport is selected, because the in-process backend implements
 * inference only. Training itself is not exercised here.
 */
class TrainingTransportIsolationTest {

    /** Minimal instance carrying just the fields the training-support backend reads. */
    private static Cellpose2D instanceWithTransport(CellposeTransport transport) {
        Cellpose2D cellpose = new Cellpose2D();
        cellpose.transport = transport;
        cellpose.parameters = new LinkedHashMap<>();
        cellpose.model = "cyto3";
        cellpose.disableGPU = false;
        cellpose.doReadResultsAsynchronously = false;
        return cellpose;
    }

    @Test
    void trainingSupportStaysSubprocessWhenApposeSelected() {
        Cellpose2D cellpose = instanceWithTransport(CellposeTransport.APPOSE);
        assertEquals(CellposeTransport.APPOSE, cellpose.transport);
        CellposeBackend backend = cellpose.createTrainingSupportBackend();
        assertInstanceOf(SubprocessBackend.class, backend,
                "training/QC must always run through the subprocess transport, never Appose");
    }

    @Test
    void trainingSupportIsSubprocessForEveryTransportSetting() {
        CellposeTransport[] settings = {CellposeTransport.SUBPROCESS, CellposeTransport.APPOSE, null};
        for (CellposeTransport t : settings) {
            CellposeBackend backend = instanceWithTransport(t).createTrainingSupportBackend();
            assertInstanceOf(SubprocessBackend.class, backend,
                    "training-support transport must not depend on the detection transport (" + t + ")");
        }
    }
}
