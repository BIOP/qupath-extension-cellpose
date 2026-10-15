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
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import qupath.ext.biop.cellpose.backend.ApposeBackend;
import qupath.ext.biop.cellpose.backend.CellposeSegmentationParams;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Live diagnostic that drives the real {@link ApposeBackend} across the channel configurations the
 * shipped example scripts produce: two exported channels with {@code cellposeChannels(...)} unset,
 * and the {@code cellposeChannels(1, 2)} form the detection template documents. Every case runs
 * independently and reports its own outcome, so one failure does not hide the rest.
 *
 * Run with:
 *   CELLPOSE_LIVE=true ./gradlew test --tests qupath.ext.biop.cellpose.ApposeChannelMatrixLiveTest
 */
@Tag("live")
@EnabledIfEnvironmentVariable(named = "CELLPOSE_LIVE", matches = "true")
public class ApposeChannelMatrixLiveTest {

    private static final int WIDTH = 224;
    private static final int HEIGHT = 320;

    private record Case(String name, int nChannels, Integer chan1, Integer chan2, boolean sam, int nTiles) {
        Case(String name, int nChannels, Integer chan1, Integer chan2, boolean sam) {
            this(name, nChannels, chan1, chan2, sam, 1);
        }
    }

    @Test
    void channelConfigurationsUsedByTheShippedScripts() throws Exception {
        List<Case> cases = List.of(
                new Case("CP3 1ch  defaults          ", 1, 0, null, false),
                new Case("CP3 2ch  defaults          ", 2, 0, null, false),
                new Case("CP3 2ch  cellposeChannels(1,2)", 2, 1, 2, false),
                new Case("CP3 2ch  cellposeChannels(2,1)", 2, 2, 1, false),
                new Case("CP3 3ch  defaults          ", 3, 0, null, false),
                new Case("CP3 7ch  defaults          ", 7, 0, null, false),
                // Several tiles in one run(), which is how Cellpose2D calls the backend.
                new Case("CP3 2ch  3 tiles, one run  ", 2, 0, null, false, 3)
        );

        boolean includeSam = "true".equals(System.getenv("CELLPOSE_LIVE_SAM"));
        if (includeSam) {
            cases = new ArrayList<>(cases);
            cases.addAll(List.of(
                    new Case("CP4 1ch  defaults          ", 1, 0, null, true),
                    new Case("CP4 2ch  defaults          ", 2, 0, null, true),
                    new Case("CP4 2ch  cellposeChannels(1,2)", 2, 1, 2, true),
                    new Case("CP4 2ch  cellposeChannels(2,1)", 2, 2, 1, true),
                    new Case("CP4 7ch  defaults          ", 7, 0, null, true)
            ));
        } else {
            System.out.println("[matrix] Cellpose-SAM cases skipped; set CELLPOSE_LIVE_SAM=true to build the cp4 environment and include them.");
        }

        List<String> failures = new ArrayList<>();
        StringBuilder report = new StringBuilder("\n=== Appose channel matrix ===\n");

        for (Case c : cases) {
            String outcome;
            try {
                int labels = runOne(c);
                outcome = labels > 0 ? ("OK   labels=" + labels) : "EMPTY (ran, zero labels)";
                if (labels == 0)
                    failures.add(c.name() + " -> produced NO labels");
            } catch (Throwable t) {
                Throwable root = t;
                while (root.getCause() != null)
                    root = root.getCause();
                outcome = "FAIL " + root.getClass().getSimpleName() + ": " + firstLine(root.getMessage());
                failures.add(c.name() + " -> " + outcome);
            }
            report.append(String.format("  %-32s %s%n", c.name().trim(), outcome));
        }

        System.out.println(report);
        Assertions.assertTrue(failures.isEmpty(), report + "\nFailures:\n - " + String.join("\n - ", failures));
    }

    private int runOne(Case c) throws Exception {
        File dir = Files.createTempDirectory("cellpose-appose-matrix").toFile();
        List<TileFile> tiles = new ArrayList<>();
        for (int i = 0; i < c.nTiles(); i++) {
            File tileImage = new File(dir, "Temp_" + i + "_0_z0_t0.tif");
            writeSyntheticTile(tileImage, c.nChannels());
            tiles.add(new TileFile(null, tileImage, null));
        }

        CellposeSegmentationParams.Builder builder = CellposeSegmentationParams.builder()
                .model(c.sam() ? "cpsam" : "cyto3")
                .customModel(false)
                .cellposeSam(c.sam())
                .diameter(30)
                .flowThreshold(0.4)
                .cellprobThreshold(0.0)
                .do3D(false);
        builder.channel1(c.chan1());
        builder.channel2(c.chan2());

        AtomicInteger readerCalls = new AtomicInteger();
        try (ApposeBackend backend = new ApposeBackend(t -> readerCalls.incrementAndGet())) {
            backend.run(tiles, builder.build());
        }
        if (readerCalls.get() != c.nTiles())
            throw new IllegalStateException("tile reader called " + readerCalls.get() + " time(s), expected " + c.nTiles());

        File maskFile = tiles.get(tiles.size() - 1).getLabelFile();
        if (!maskFile.exists())
            throw new IllegalStateException("no mask written");
        ImagePlus maskImp = IJ.openImage(maskFile.getAbsolutePath());
        ImageProcessor mp = maskImp.getProcessor();
        Set<Integer> labels = new HashSet<>();
        for (int y = 0; y < mp.getHeight(); y++)
            for (int x = 0; x < mp.getWidth(); x++)
                labels.add(mp.get(x, y));
        labels.remove(0);
        int w = maskImp.getWidth();
        int h = maskImp.getHeight();
        maskImp.close();
        if (w != WIDTH || h != HEIGHT)
            throw new IllegalStateException("mask dimensions " + w + "x" + h + " != " + WIDTH + "x" + HEIGHT);
        return labels.size();
    }

    /**
     * 32-bit float multi-channel tile, mimicking what {@code Cellpose2D.saveTileImage} produces
     * after the op chain. Channel 0 carries bright disks, channel 1 smaller nuclei-like disks, and
     * any further channels a faint copy of channel 0 -- flat filler channels would dilute the
     * grayscale average below threshold and the case would report zero labels for that reason alone.
     */
    private static void writeSyntheticTile(File file, int nChannels) {
        ImageStack stack = new ImageStack(WIDTH, HEIGHT);
        int[][] centers = {{60, 70}, {150, 90}, {90, 200}, {170, 260}, {60, 280}};
        for (int c = 0; c < nChannels; c++) {
            FloatProcessor fp = new FloatProcessor(WIDTH, HEIGHT);
            fp.setColor(20.0);
            fp.fill();
            int r = c == 1 ? 8 : 16;
            fp.setColor(c <= 1 ? 220.0 : 120.0);
            for (int[] centre : centers)
                fp.fillOval(centre[0] - r, centre[1] - r, 2 * r, 2 * r);
            fp.smooth();
            stack.addSlice("C" + (c + 1), fp);
        }
        ImagePlus imp = new ImagePlus(file.getName(), stack);
        IJ.save(imp, file.getAbsolutePath());
    }

    private static String firstLine(String message) {
        if (message == null)
            return "(no message)";
        int nl = message.indexOf('\n');
        return nl < 0 ? message : message.substring(0, nl);
    }
}
