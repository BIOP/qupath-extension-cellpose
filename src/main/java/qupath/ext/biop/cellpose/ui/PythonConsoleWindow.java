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

package qupath.ext.biop.cellpose.ui;

import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.ToolBar;
import javafx.scene.layout.BorderPane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.GraphicsEnvironment;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A console window that surfaces the Python diagnostic output of the in-process (Appose) Cellpose
 * backend to the user.
 * <p>
 * {@link #appendMessage(String)} is callable from any thread and before the JavaFX {@link Stage}
 * exists, since the Appose service starts emitting on a daemon thread. Messages are enqueued and
 * drained on the FX thread by a single coalesced {@link Platform#runLater}; history is bounded, and
 * closing the window hides it rather than destroying it. Every UI operation is guarded so a headless
 * or scripted run never throws.
 */
public final class PythonConsoleWindow {

    private static final Logger logger = LoggerFactory.getLogger(PythonConsoleWindow.class);

    /** Upper bound on retained lines; when exceeded we trim back to {@link #TRIM_TO}. */
    private static final int MAX_LINES = 10_000;
    private static final int TRIM_TO = 8_000;

    /** Cross-thread inbox: any thread may add; only the FX thread drains. */
    private static final ConcurrentLinkedQueue<String> PENDING = new ConcurrentLinkedQueue<>();
    /** Retained history; only ever touched on the FX thread. */
    private static final Deque<String> HISTORY = new ArrayDeque<>();
    /** Guards against scheduling more than one pending FX flush at a time. */
    private static final AtomicBoolean FLUSH_SCHEDULED = new AtomicBoolean(false);

    /** The single window instance (created lazily on the FX thread). */
    private static volatile PythonConsoleWindow instance;

    private final Stage stage;
    private final TextArea textArea;
    private final ToggleButton autoScroll;

    private PythonConsoleWindow() {
        textArea = new TextArea();
        textArea.setEditable(false);
        textArea.setWrapText(false);
        textArea.setStyle("-fx-font-family: 'monospaced';");

        autoScroll = new ToggleButton("Auto-scroll");
        autoScroll.setSelected(true);

        Button clear = new Button("Clear");
        clear.setOnAction(e -> {
            HISTORY.clear();
            textArea.clear();
        });

        Button save = new Button("Save to file...");
        save.setOnAction(e -> saveToFile());

        ToolBar toolBar = new ToolBar(clear, autoScroll, save);

        BorderPane root = new BorderPane();
        root.setTop(toolBar);
        root.setCenter(textArea);
        BorderPane.setMargin(textArea, new Insets(0));

        stage = new Stage();
        stage.setTitle("Cellpose Python console");
        stage.setScene(new Scene(root, 800, 480));
        // Hide rather than destroy, so reopening shows the full history.
        stage.setOnCloseRequest(e -> {
            e.consume();
            stage.hide();
        });

        // Seed with whatever history accumulated before the window existed.
        if (!HISTORY.isEmpty())
            textArea.setText(String.join("\n", HISTORY) + "\n");
    }

    /**
     * Append one message to the console. Safe to call from any thread and before the window exists;
     * with no JavaFX available the message stays in the buffer and is never rendered.
     *
     * @param message the line to append; {@code null} is ignored
     */
    public static void appendMessage(String message) {
        if (message == null)
            return;
        PENDING.add(message);
        // Bound the inbox, which never drains when headless.
        while (PENDING.size() > MAX_LINES)
            PENDING.poll();
        scheduleFlush();
    }

    private static void scheduleFlush() {
        if (!isFxAvailable())
            return;
        if (FLUSH_SCHEDULED.compareAndSet(false, true)) {
            try {
                Platform.runLater(PythonConsoleWindow::flush);
            } catch (IllegalStateException e) {
                // FX toolkit not started; allow a later message to retry.
                FLUSH_SCHEDULED.set(false);
            }
        }
    }

    /** Drain the inbox into history (and the text area, if the window exists). FX thread only. */
    private static void flush() {
        FLUSH_SCHEDULED.set(false);
        StringBuilder appended = new StringBuilder();
        String line;
        while ((line = PENDING.poll()) != null) {
            HISTORY.addLast(line);
            appended.append(line).append('\n');
        }
        boolean trimmed = false;
        if (HISTORY.size() > MAX_LINES) {
            while (HISTORY.size() > TRIM_TO)
                HISTORY.pollFirst();
            trimmed = true;
        }
        PythonConsoleWindow window = instance;
        if (window == null || appended.length() == 0)
            return;
        if (trimmed) {
            window.textArea.setText(String.join("\n", HISTORY) + "\n");
        } else {
            window.textArea.appendText(appended.toString());
        }
        if (window.autoScroll.isSelected())
            window.textArea.setScrollTop(Double.MAX_VALUE);
    }

    /**
     * Show (creating if necessary) the console window. Safe to call from any thread; a no-op with a
     * logged warning in a headless environment.
     */
    public static void show() {
        if (!isFxAvailable()) {
            logger.warn("Cannot show the Cellpose Python console: no graphics environment available");
            return;
        }
        try {
            Platform.runLater(PythonConsoleWindow::showOnFx);
        } catch (IllegalStateException e) {
            logger.warn("Cannot show the Cellpose Python console: JavaFX is not running");
        }
    }

    private static void showOnFx() {
        PythonConsoleWindow window = instance;
        if (window == null) {
            window = new PythonConsoleWindow();
            instance = window;
        }
        // Drain any buffered messages into the freshly created text area.
        flush();
        window.stage.show();
        window.stage.toFront();
    }

    private void saveToFile() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Save Cellpose Python console log");
        chooser.setInitialFileName("cellpose-python-console.log");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Log files", "*.log", "*.txt"));
        var file = chooser.showSaveDialog(stage);
        if (file == null)
            return;
        try {
            Files.writeString(file.toPath(), textArea.getText(), StandardCharsets.UTF_8);
            logger.info("Saved Cellpose Python console log to {}", file);
        } catch (IOException e) {
            logger.error("Failed to save Cellpose Python console log to {}: {}", file, e.getMessage(), e);
            throw new UncheckedIOException(e);
        }
    }

    private static boolean isFxAvailable() {
        return !GraphicsEnvironment.isHeadless();
    }
}
