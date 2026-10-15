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

package qupath.ext.biop.cellpose.backend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for the backend-selection logic: a builder-level transport overrides the extension
 * preference, and the preference is honored when the builder is silent.
 */
class BackendSelectionTest {

    @Test
    void builderFlagOverridesPreference() {
        assertEquals(CellposeTransport.APPOSE,
                CellposeBackend.resolveTransport(CellposeTransport.APPOSE, CellposeTransport.SUBPROCESS));
        assertEquals(CellposeTransport.SUBPROCESS,
                CellposeBackend.resolveTransport(CellposeTransport.SUBPROCESS, CellposeTransport.APPOSE));
    }

    @Test
    void preferenceHonoredWhenBuilderSilent() {
        assertEquals(CellposeTransport.APPOSE,
                CellposeBackend.resolveTransport(null, CellposeTransport.APPOSE));
        assertEquals(CellposeTransport.SUBPROCESS,
                CellposeBackend.resolveTransport(null, CellposeTransport.SUBPROCESS));
    }

    @Test
    void defaultsToSubprocessWhenNothingSet() {
        assertEquals(CellposeTransport.SUBPROCESS,
                CellposeBackend.resolveTransport(null, null));
    }
}
