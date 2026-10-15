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

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Tests that {@link TileFile} can carry both the tile image and its labels in memory for the
 * in-process backend, while a file-backed tile still reports itself as such.
 */
public class TileFileInMemoryTest {

    private static final File TILE = new File("cellpose-temp", "Temp_0_0_z0_t0.tif");

    @Test
    void subprocessTileReportsItselfAsFileBacked() {
        TileFile tile = new TileFile(null, TILE, null);
        Assertions.assertFalse(tile.isInMemory());
        Assertions.assertEquals(TILE, tile.getImageFile());
        Assertions.assertTrue(tile.getLabelFile().getName().endsWith("_cp_masks.tif"));
        Assertions.assertNull(tile.getLabels(), "labels come from disk for the subprocess backend");
    }

    @Test
    void inMemoryTileProducesItsImageOnDemandAndOnlyWhenAsked() {
        AtomicInteger materialisations = new AtomicInteger();
        ImagePlus expected = new ImagePlus("tile", new ByteProcessor(8, 8));
        TileFile tile = new TileFile(null, TILE, null, () -> {
            materialisations.incrementAndGet();
            return expected;
        });

        Assertions.assertTrue(tile.isInMemory());
        Assertions.assertEquals(0, materialisations.get(),
                "constructing a tile must not extract pixels; memory is bounded by how many are open at once");

        Assertions.assertSame(expected, tile.openImage());
        Assertions.assertEquals(1, materialisations.get());
    }

    @Test
    void inMemoryTileCarriesLabelsAndReleasesThem() {
        TileFile tile = new TileFile(null, TILE, null, () -> null);
        ImageProcessor labels = new ByteProcessor(4, 4);

        tile.setLabels(labels);
        Assertions.assertSame(labels, tile.getLabels());

        tile.clearLabels();
        Assertions.assertNull(tile.getLabels());
    }

    @Test
    void aTileThatCannotBeMaterialisedReportsNullRatherThanThrowing() {
        TileFile tile = new TileFile(null, TILE, null, () -> null);
        Assertions.assertNull(tile.openImage(), "the backend skips the tile and logs, rather than failing the run");
    }
}
