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

import ij.IJ;
import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import qupath.ext.biop.cellpose.backend.ApposeBackend;
import qupath.ext.biop.cellpose.backend.CellposeSegmentationParams;

import java.io.File;
import java.nio.file.Files;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Live end-to-end smoke test for the in-process {@link ApposeBackend}, gated behind the
 * {@code CELLPOSE_LIVE=true} environment variable because it builds a multi-GB pixi environment and
 * downloads model weights. It runs Cellpose 3 on a synthetic grayscale image of bright disks and
 * asserts that the mask file is written with the input dimensions and holds at least one label.
 *
 * Run with:
 *   CELLPOSE_LIVE=true ./gradlew test --tests qupath.ext.biop.cellpose.ApposeLiveSmokeTest
 */
@Tag("live")
@EnabledIfEnvironmentVariable(named = "CELLPOSE_LIVE", matches = "true")
public class ApposeLiveSmokeTest {

    // Deliberately non-square to catch a (H,W) vs (W,H) transpose.
    private static final int WIDTH = 224;
    private static final int HEIGHT = 320;

    @Test
    void cellpose3RoundTripThroughApposeBackend() throws Exception {
        File dir = Files.createTempDirectory("cellpose-appose-smoke").toFile();
        File tileImage = new File(dir, "Temp_0_0_z0_t0.tif");
        writeSyntheticCells(tileImage);

        // RegionRequest and parent are unused by ApposeBackend.run.
        TileFile tile = new TileFile(null, tileImage, null);

        CellposeSegmentationParams params = CellposeSegmentationParams.builder()
                .model("cyto3")
                .customModel(false)
                .cellposeSam(false)
                .channel1(0)          // grayscale: cell channel 0
                .channel2(0)          // grayscale: nuclei channel 0 -> channels [0,0]
                .diameter(30)
                .flowThreshold(0.4)
                .cellprobThreshold(0.0)
                .do3D(false)
                .build();

        AtomicInteger readerCalls = new AtomicInteger();
        try (ApposeBackend backend = new ApposeBackend(t -> readerCalls.incrementAndGet())) {
            backend.run(List.of(tile), params);
        }

        Assertions.assertEquals(1, readerCalls.get(), "tile reader should be called once");

        File maskFile = tile.getLabelFile();
        Assertions.assertTrue(maskFile.exists(), "mask file should be written: " + maskFile);

        ImagePlus maskImp = IJ.openImage(maskFile.getAbsolutePath());
        Assertions.assertNotNull(maskImp, "mask image should be readable");
        Assertions.assertEquals(WIDTH, maskImp.getWidth(), "mask width must match input (no transpose)");
        Assertions.assertEquals(HEIGHT, maskImp.getHeight(), "mask height must match input (no transpose)");

        ImageProcessor mp = maskImp.getProcessor();
        Set<Integer> labels = new HashSet<>();
        for (int y = 0; y < mp.getHeight(); y++)
            for (int x = 0; x < mp.getWidth(); x++)
                labels.add(mp.get(x, y));
        labels.remove(0); // background
        maskImp.close();

        System.out.println("[smoke] Cellpose returned " + labels.size() + " label(s) for a "
                + WIDTH + "x" + HEIGHT + " image");
        Assertions.assertFalse(labels.isEmpty(),
                "Cellpose should segment at least one of the synthetic disks");
    }

    /** Write a grayscale 8-bit image with a handful of bright ~30px disks on a dark background. */
    private static void writeSyntheticCells(File file) {
        ByteProcessor bp = new ByteProcessor(WIDTH, HEIGHT);
        bp.setColor(20);
        bp.fill();
        bp.setColor(220);
        int r = 16;
        int[][] centers = {{60, 70}, {150, 90}, {90, 200}, {170, 260}, {60, 280}};
        for (int[] c : centers)
            bp.fillOval(c[0] - r, c[1] - r, 2 * r, 2 * r);
        // light blur to give Cellpose a gradient to work with
        bp.smooth();
        IJ.save(new ImagePlus(file.getName(), (ImageProcessor) bp), file.getAbsolutePath());
    }
}
