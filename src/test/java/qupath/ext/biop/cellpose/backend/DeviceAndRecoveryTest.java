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

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the device-resolution logic and the Windows recovery / lock-format matchers.
 */
class DeviceAndRecoveryTest {

    @Test
    void resolveDeviceBuilderWinsThenPreferenceThenAuto() {
        assertEquals(CellposeDevice.GPU,
                CellposeBackend.resolveDevice(CellposeDevice.GPU, CellposeDevice.CPU));
        assertEquals(CellposeDevice.CPU,
                CellposeBackend.resolveDevice(null, CellposeDevice.CPU));
        assertEquals(CellposeDevice.AUTO,
                CellposeBackend.resolveDevice(null, null));
    }

    @Test
    void cpuIsAlwaysHonoured() {
        // GPU and AUTO depend on the host and are not asserted here.
        assertNull(ApposeEnvironments.cudaVariant(CellposeDevice.CPU));
    }

    @Test
    void aGpuIsRequestedUnlessCpuWasAskedFor() {
        // macOS gets the CPU environment but still has MPS, and cp_utils only looks for an
        // accelerator when it is asked for one.
        assertFalse(ApposeEnvironments.requestGpu(CellposeDevice.CPU));
        assertTrue(ApposeEnvironments.requestGpu(CellposeDevice.AUTO));
        assertTrue(ApposeEnvironments.requestGpu(CellposeDevice.GPU));
    }

    @Test
    void envNameCombinesFamilyAndCudaBuild() {
        assertEquals("cp3-cpu", ApposeEnvironments.envName(false, null));
        assertEquals("cp3-cu126", ApposeEnvironments.envName(false, "cu126"));
        assertEquals("cp4-cpu", ApposeEnvironments.envName(true, null));
        assertEquals("cp4-cu130", ApposeEnvironments.envName(true, "cu130"));
    }

    @Test
    void oldCardsTakeTheOnlyBuildThatHasThem() {
        // cu130 starts at sm_75, so Maxwell, Pascal and Volta have nowhere else to go.
        assertEquals("cu126", ApposeEnvironments.cudaBuildFor("5.0"));
        assertEquals("cu126", ApposeEnvironments.cudaBuildFor("6.1"));
        assertEquals("cu126", ApposeEnvironments.cudaBuildFor("7.0"));
    }

    @Test
    void sharedRangePrefersTheOlderBuild() {
        // Both builds cover sm_75..sm_90; staying on cu126 avoids a second multi-GB download
        // for machines that already have it.
        assertEquals("cu126", ApposeEnvironments.cudaBuildFor("7.5"));
        assertEquals("cu126", ApposeEnvironments.cudaBuildFor("8.6"));
        assertEquals("cu126", ApposeEnvironments.cudaBuildFor("9.0"));
    }

    @Test
    void newCardsTakeTheBuildThatHasThem() {
        // cu126 stops at sm_90 and ships no PTX, so Blackwell cannot run it at all.
        assertEquals("cu130", ApposeEnvironments.cudaBuildFor("10.0"));
        assertEquals("cu130", ApposeEnvironments.cudaBuildFor("12.0"));
        // cu130 carries compute_120 PTX, so anything newer can still JIT.
        assertEquals("cu130", ApposeEnvironments.cudaBuildFor("13.0"));
    }

    @Test
    void cardsBelowEveryBuildGetNoCuda() {
        assertNull(ApposeEnvironments.cudaBuildFor("3.7"));
        assertNull(ApposeEnvironments.cudaBuildFor("2.0"));
    }

    @Test
    void anUnreadableCapabilityFallsBackToTheOlderBuild() {
        // A driver too old to report the capability predates every card that needs cu130.
        assertEquals("cu126", ApposeEnvironments.cudaBuildFor(null));
        assertEquals("cu126", ApposeEnvironments.cudaBuildFor(""));
        assertEquals("cu126", ApposeEnvironments.cudaBuildFor("N/A"));
    }

    @Test
    void windowsFileLockDetectedThroughFullCauseChain() {
        Throwable nested = new IllegalStateException(
                "failed to link cellpose.conda ... The process cannot access the file because it is "
                        + "being used by another process. (os error 32)");
        Throwable top = new IOException("pixi build failed", nested);
        String chain = ApposeEnvironments.collectCauseMessages(top);
        assertTrue(ApposeEnvironments.looksLikeWindowsFileLock(chain),
                "the file-lock signature should be found in the nested cause");
        assertFalse(ApposeEnvironments.looksLikeWindowsFileLock("pixi build failed"),
                "the generic top-level message alone must not match");
    }

    @Test
    void lockFormatSkewMatcher() {
        assertTrue(ApposeEnvironments.looksLikeLockFormatSkew("unsupported lock file version 7"));
        assertTrue(ApposeEnvironments.looksLikeLockFormatSkew("failed to parse pixi.lock"));
        assertFalse(ApposeEnvironments.looksLikeLockFormatSkew("some unrelated build error"));
    }

    @Test
    void collectCauseMessagesToleratesSelfReference() {
        // Java forbids a true cause cycle, so this only exercises a nested chain.
        Throwable a = new RuntimeException("a");
        Throwable b = new RuntimeException("b", a);
        String chain = ApposeEnvironments.collectCauseMessages(b);
        assertTrue(chain.contains("a") && chain.contains("b"));
    }
}
