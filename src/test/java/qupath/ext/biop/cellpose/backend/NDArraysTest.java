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

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import org.apposed.appose.NDArray;
import org.apposed.appose.NDArray.DType;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Round-trip tests for {@link NDArrays}: ImageJ processors to Appose {@link NDArray} and back, for
 * 8/16/32-bit data and for single- and multi-channel images.
 */
class NDArraysTest {

    private static final int WIDTH = 7;
    private static final int HEIGHT = 5;

    @Test
    void byteProcessorRoundTrip() {
        ByteProcessor ip = new ByteProcessor(WIDTH, HEIGHT);
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                ip.set(x, y, (y * WIDTH + x) % 256);

        try (NDArray nd = NDArrays.fromProcessor(ip)) {
            assertEquals(DType.UINT8, nd.dType());
            assertEquals(2, nd.shape().length());
            assertEquals(HEIGHT, nd.shape().get(0));
            assertEquals(WIDTH, nd.shape().get(1));

            ImageProcessor back = NDArrays.toProcessor(nd);
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++)
                    assertEquals(ip.get(x, y), back.get(x, y), "byte pixel (" + x + "," + y + ")");
        }
    }

    @Test
    void shortProcessorRoundTrip() {
        ShortProcessor ip = new ShortProcessor(WIDTH, HEIGHT);
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                ip.set(x, y, (y * WIDTH + x) * 1000);

        try (NDArray nd = NDArrays.fromProcessor(ip)) {
            assertEquals(DType.UINT16, nd.dType());

            ImageProcessor back = NDArrays.toProcessor(nd);
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++)
                    assertEquals(ip.get(x, y), back.get(x, y), "short pixel (" + x + "," + y + ")");
        }
    }

    @Test
    void floatProcessorRoundTripAndCOrder() {
        FloatProcessor ip = new FloatProcessor(WIDTH, HEIGHT);
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                ip.setf(x, y, (y * WIDTH + x) + 0.5f);

        try (NDArray nd = NDArrays.fromProcessor(ip)) {
            assertEquals(DType.FLOAT32, nd.dType());

            // Verify C-order (row-major): element (y, x) lives at offset y*WIDTH + x.
            ByteBuffer buffer = nd.buffer().order(ByteOrder.nativeOrder());
            buffer.rewind();
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++) {
                    float raw = buffer.getFloat((y * WIDTH + x) * Float.BYTES);
                    assertEquals(ip.getf(x, y), raw, 0f, "C-order offset (" + x + "," + y + ")");
                }

            ImageProcessor back = NDArrays.toProcessor(nd);
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++)
                    assertEquals(ip.getf(x, y), back.getf(x, y), 0f, "float pixel (" + x + "," + y + ")");
        }
    }

    @Test
    void multiChannelImagePlusIsCOrderedChannelsFirst() {
        int nChannels = 3;
        ImageStack stack = new ImageStack(WIDTH, HEIGHT);
        for (int c = 0; c < nChannels; c++) {
            FloatProcessor ip = new FloatProcessor(WIDTH, HEIGHT);
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++)
                    ip.setf(x, y, c * 1000 + y * WIDTH + x);
            stack.addSlice("c" + c, ip);
        }
        ImagePlus imp = new ImagePlus("multi", stack);

        try (NDArray nd = NDArrays.fromImagePlus(imp)) {
            assertEquals(DType.FLOAT32, nd.dType());
            assertEquals(3, nd.shape().length());
            assertEquals(nChannels, nd.shape().get(0));
            assertEquals(HEIGHT, nd.shape().get(1));
            assertEquals(WIDTH, nd.shape().get(2));

            // C-order (channels, height, width): offset = c*H*W + y*W + x.
            ByteBuffer buffer = nd.buffer().order(ByteOrder.nativeOrder());
            for (int c = 0; c < nChannels; c++)
                for (int y = 0; y < HEIGHT; y++)
                    for (int x = 0; x < WIDTH; x++) {
                        int idx = (c * HEIGHT * WIDTH + y * WIDTH + x) * Float.BYTES;
                        assertEquals(c * 1000 + y * WIDTH + x, buffer.getFloat(idx), 0f,
                                "channel " + c + " pixel (" + x + "," + y + ")");
                    }
        }
    }

    @Test
    void singleChannelImagePlusIsTwoDimensional() {
        FloatProcessor ip = new FloatProcessor(WIDTH, HEIGHT);
        ImagePlus imp = new ImagePlus("single", ip);
        try (NDArray nd = NDArrays.fromImagePlus(imp)) {
            assertEquals(2, nd.shape().length());
            assertEquals(HEIGHT, nd.shape().get(0));
            assertEquals(WIDTH, nd.shape().get(1));
        }
    }

    @Test
    void labelsRoundTripThroughShortProcessor() throws Exception {
        NDArray labels = NDArrays.allocateLabels(WIDTH, HEIGHT);
        try {
            assertEquals(DType.UINT32, labels.dType());
            ByteBuffer buffer = labels.buffer().order(ByteOrder.nativeOrder());
            buffer.rewind();
            for (int i = 0; i < WIDTH * HEIGHT; i++)
                buffer.putInt(i % 500);

            ShortProcessor sp = NDArrays.labelsToShortProcessor(labels, WIDTH, HEIGHT);
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++)
                    assertEquals((y * WIDTH + x) % 500, sp.get(x, y), "label pixel (" + x + "," + y + ")");
        } finally {
            labels.close();
        }
    }

    @Test
    void tooManyLabelsIsReportedRatherThanTruncated() {
        // Cellpose writes its result with `output_labels[:] = masks`, which casts silently, so a
        // narrower buffer would wrap high labels into other objects with no error.
        NDArray labels = NDArrays.allocateLabels(WIDTH, HEIGHT);
        try {
            ByteBuffer buffer = labels.buffer().order(ByteOrder.nativeOrder());
            buffer.rewind();
            for (int i = 0; i < WIDTH * HEIGHT; i++)
                buffer.putInt(0);
            buffer.rewind();
            buffer.putInt(70000);

            IOException e = assertThrows(IOException.class,
                    () -> NDArrays.labelsToShortProcessor(labels, WIDTH, HEIGHT));
            assertTrue(e.getMessage().contains("70000"), e.getMessage());
            assertTrue(e.getMessage().contains("tileSize"), e.getMessage());
        } finally {
            labels.close();
        }
    }

    @Test
    void shmIsUsableThenClosedCleanly() {
        NDArray nd = NDArrays.allocateLabels(WIDTH, HEIGHT);
        assertNotNull(nd.shm(), "shared memory should be allocated");
        // Closing must be idempotent.
        assertDoesNotThrow(nd::close);
        assertDoesNotThrow(nd::close);
        try (NDArray again = NDArrays.allocateLabels(WIDTH, HEIGHT)) {
            assertNotNull(again.shm());
        }
    }
}
