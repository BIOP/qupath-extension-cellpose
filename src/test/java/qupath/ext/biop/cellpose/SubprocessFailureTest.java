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

package qupath.ext.biop.cellpose;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;
import qupath.ext.biop.cellpose.backend.SubprocessBackend;
import qupath.ext.biop.cmd.VirtualEnvironmentRunner;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A subprocess run that fails must fail the call. It used to return zero objects: the exit code
 * was never checked, and each missing mask became a NullPointerException on a reader thread.
 * A shell script stands in for Python, so these run without Cellpose installed.
 */
@DisabledOnOs(OS.WINDOWS)
class SubprocessFailureTest {

    @TempDir
    Path dir;

    private List<TileFile> tiles() throws IOException {
        TileFile a = new TileFile(null, dir.resolve("Temp_0_0.tif").toFile(), null);
        TileFile b = new TileFile(null, dir.resolve("Temp_0_640.tif").toFile(), null);
        return List.of(a, b);
    }

    private SubprocessBackend backend(String script, boolean async, AtomicInteger reads) throws IOException {
        Path python = dir.resolve("fake-python");
        Files.writeString(python, "#!/bin/sh\n" + script + "\n");
        assertTrue(python.toFile().setExecutable(true));
        return new SubprocessBackend(dir.toFile(), new LinkedHashMap<>(), "nuclei", true, async,
                () -> new VirtualEnvironmentRunner(python.toString(), VirtualEnvironmentRunner.EnvType.EXE, "test"),
                tile -> reads.incrementAndGet());
    }

    private static void touch(File f) throws IOException {
        Files.write(f.toPath(), new byte[] {0});
    }

    @Test
    void aCrashAtImportFailsTheRunAndQuotesTheTraceback() throws IOException {
        for (boolean async : new boolean[] {false, true}) {
            AtomicInteger reads = new AtomicInteger();
            var backend = backend("echo 'Traceback (most recent call last):'\n"
                    + "echo \"ModuleNotFoundError: No module named 'packaging'\"\nexit 1", async, reads);
            IOException e = assertThrows(IOException.class, () -> backend.run(tiles(), null));
            assertTrue(e.getMessage().contains("exited with code 1"), e.getMessage());
            assertTrue(e.getMessage().contains("No module named 'packaging'"), e.getMessage());
            assertEquals(0, reads.get());
        }
    }

    @Test
    void aMissingMaskAfterACleanExitFailsTheRun() throws IOException {
        List<TileFile> tiles = tiles();
        touch(tiles.get(0).getLabelFile());
        AtomicInteger reads = new AtomicInteger();
        var backend = backend("exit 0", false, reads);
        IOException e = assertThrows(IOException.class, () -> backend.run(tiles, null));
        assertTrue(e.getMessage().contains("no mask for 1 of 2"), e.getMessage());
        assertTrue(e.getMessage().contains("Temp_0_640_cp_masks.tif"), e.getMessage());
        assertEquals(1, reads.get(), "the mask that exists is still read");
    }

    @Test
    void everyMaskPresentReadsEachTileOnce() throws Exception {
        for (boolean async : new boolean[] {false, true}) {
            List<TileFile> tiles = tiles();
            for (TileFile t : tiles)
                touch(t.getLabelFile());
            AtomicInteger reads = new AtomicInteger();
            backend("exit 0", async, reads).run(tiles, null);
            assertEquals(2, reads.get(), "async=" + async);
        }
    }
}
