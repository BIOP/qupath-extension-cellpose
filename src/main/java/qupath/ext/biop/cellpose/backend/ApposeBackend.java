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

import static qupath.ext.biop.cellpose.backend.ApposeEnvironments.elapsedMs;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ShortProcessor;
import org.apposed.appose.Environment;
import org.apposed.appose.NDArray;
import org.apposed.appose.Service;
import org.apposed.appose.Service.Task;
import org.apposed.appose.Service.TaskStatus;
import org.apposed.appose.TaskEvent;
import org.apposed.appose.TaskException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.common.ThreadTools;
import qupath.ext.biop.cellpose.TileFile;
import qupath.ext.biop.cellpose.ui.PythonConsoleWindow;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * In-process Cellpose segmentation backend based on Appose.
 * <p>
 * Tile pixels are passed to Python as shared-memory {@link NDArray}s and the labels are read back
 * the same way; a tile that has a file behind it is still read from and written to disk. Tiles are
 * processed by a bounded pool, but the Cellpose call itself is serialized (see {@link #pythonLock}).
 * <p>
 * The Appose service lifecycle is adapted, with attribution, from the BSD-3-Clause imglib2-cellpose
 * project; see NOTICE.
 */
public class ApposeBackend implements CellposeBackend {

    private static final Logger logger = LoggerFactory.getLogger(ApposeBackend.class);

    /** Attempts for the transient Appose worker "thread death" (see {@link #submit}). */
    private static final int THREAD_DEATH_ATTEMPTS = 3;
    private static final long THREAD_DEATH_RETRY_DELAY_MS = 500L;

    /** Workers this backend has replaced after a death; reported with each run's timing summary. */
    private final AtomicInteger workerReplacements = new AtomicInteger();

    /** Cellpose-SAM (Cellpose 4) accepts at most three channels. */
    private static final int CELLPOSE_SAM_MAX_CHANNELS = 3;

    /** Live Appose services, closed by a JVM shutdown hook so no Python subprocess is orphaned. */
    private static final Set<Service> LIVE_SERVICES = ConcurrentHashMap.newKeySet();
    private static final AtomicBoolean SHUTDOWN_HOOK_INSTALLED = new AtomicBoolean(false);

    private final Consumer<TileFile> tileReader;

    /** Requested worker count; <= 0 means follow QuPath's own parallelism setting. */
    private final int requestedThreads;

    /**
     * Lock serializing the Cellpose call: the vendored scripts cache the model in a single Python
     * global, so concurrent {@code model.eval} calls would race on it.
     *
     * <p>Scoped to the environment, not to the {@link Worker}, so that replacing a dead worker does
     * not swap the monitor out from under a caller that is holding it.
     */
    private Object pythonLock() {
        return pythonLockFor(envName);
    }

    private static Object pythonLockFor(String envName) {
        return PYTHON_LOCKS.computeIfAbsent(envName, k -> new Object());
    }

    private Worker worker;
    private String envName;
    private CellposeSegmentationParams workerParams;
    private CellposeDevice workerDevice;
    /** True while recovering from a dead worker, so a failure there cannot recurse. */
    private boolean recovering;
    private String runScript;
    private String initScript;
    private boolean useGpu;

    /**
     * @param tileReader callback turning a finished tile's masks into candidate objects
     */
    public ApposeBackend(Consumer<TileFile> tileReader) {
        this(tileReader, -1);
    }

    /**
     * @param tileReader callback turning a finished tile's masks into candidate objects
     * @param nThreads   worker count for tile extraction and mask tracing; <= 0 follows
     *                   QuPath's parallelism setting
     */
    public ApposeBackend(Consumer<TileFile> tileReader, int nThreads) {
        this.tileReader = tileReader;
        installShutdownHook();
        this.requestedThreads = nThreads;
    }

    @Override
    public void run(List<TileFile> tiles, CellposeSegmentationParams params) throws IOException, InterruptedException {
        if (tiles == null || tiles.isEmpty()) {
            return;
        }

        // Produce one tile before the environment build and the model load. Tiles are lazy on this
        // path, so without this a bad channel name surfaces only after several minutes of setup,
        // where the subprocess backend rejects it immediately.
        probeFirstTile(tiles.get(0));

        long setupStart = System.nanoTime();
        ensureService(params);
        initializeModel(params);
        logger.info("Cellpose Python worker and model ready in {} ms", elapsedMs(setupStart));

        int workers = Math.max(1, Math.min(
                requestedThreads > 0 ? requestedThreads : ThreadTools.getParallelism(),
                tiles.size()));
        logger.info("Segmenting {} tile(s) with {} worker(s)", tiles.size(), workers);
        long segmentStart = System.nanoTime();
        int replacementsBefore = workerReplacements.get();

        // Bounded, so at most `workers` tiles hold pixels at once.
        ExecutorService pool = Executors.newFixedThreadPool(workers,
                ThreadTools.createThreadFactory("cellpose-appose-", true));
        List<Future<?>> futures = new ArrayList<>(tiles.size());
        try {
            for (TileFile tile : tiles)
                futures.add(pool.submit(() -> { processTile(tile, params); return null; }));

            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (ExecutionException e) {
                    futures.forEach(f -> f.cancel(true));
                    Throwable cause = e.getCause();
                    if (cause instanceof IOException)
                        throw (IOException) cause;
                    if (cause instanceof InterruptedException)
                        throw (InterruptedException) cause;
                    throw new IOException("Cellpose segmentation failed: " + cause.getMessage(), cause);
                } catch (CancellationException e) {
                    throw new InterruptedException("Cellpose segmentation cancelled");
                }
            }
        } catch (InterruptedException e) {
            futures.forEach(f -> f.cancel(true));
            Thread.currentThread().interrupt();
            throw e;
        } finally {
            pool.shutdownNow();
        }

        // The numbers a slow run needs: per-tile cost, and how often the worker had to be rebuilt.
        long elapsed = elapsedMs(segmentStart);
        logger.info("Segmented {} tile(s) in {} ms ({} ms per tile); the Python worker was replaced {} time(s) during this run",
                tiles.size(), elapsed, elapsed / tiles.size(), workerReplacements.get() - replacementsBefore);
    }

    /** Load the model into the worker, unless it already holds the same one. */
    private void initializeModel(CellposeSegmentationParams params) throws IOException, InterruptedException {
        String signature = (params.isCustomModel() ? "custom:" : "named:") + params.getModel();
        synchronized (pythonLock()) {
            if (signature.equals(worker.loadedModel)) {
                logger.info("Reusing the Cellpose model already loaded in the Python worker ({})", params.getModel());
                return;
            }
            Map<String, Object> inputs = new HashMap<>();
            inputs.put("use_gpu", useGpu);
            inputs.put("custom_model", params.isCustomModel() ? params.getModel() : null);
            inputs.put("model_name", params.isCustomModel() ? null : params.getModel());
            // Clear first, so a failed load cannot leave the worker claiming a model it lacks.
            worker.loadedModel = null;
            long loadStart = System.nanoTime();
            submit(initScript, inputs, "Initializing Cellpose model");
            worker.loadedModel = signature;
            logger.info("Loaded the Cellpose model {} in {} ms", params.getModel(), elapsedMs(loadStart));
        }
    }

    /** Extract one tile, segment it, and hand the labels to the tile reader. */
    /**
     * Check that a tile image can actually be produced, reporting failure through the same channel
     * the segmentation loop uses.
     *
     * @param tile the tile to materialise and discard
     * @throws IOException if the tile image cannot be produced
     */
    private static void probeFirstTile(TileFile tile) throws IOException {
        try {
            if (tile.openImage() == null)
                throw new IOException("Could not produce the tile image for " + tile.getImageFile().getName());
        } catch (RuntimeException e) {
            throw new IOException("Could not produce tile images: " + e.getMessage(), e);
        }
    }

    private void processTile(TileFile tile, CellposeSegmentationParams params) throws IOException, InterruptedException {
        // model.eval cannot be interrupted mid-tile, so cancellation is honored between tiles.
        if (Thread.currentThread().isInterrupted())
            throw new InterruptedException("Cellpose segmentation cancelled before tile "
                    + tile.getImageFile().getName());

        ImagePlus imp = tile.openImage();
        if (imp == null) {
            logger.warn("Could not obtain tile image {}; skipping", tile.getImageFile());
            return;
        }
        int width = imp.getWidth();
        int height = imp.getHeight();

        // Cellpose 3 gets only the channels it will use (see Cp3Layout); cp4.py slices for itself.
        Cp3Layout layout = params.isCellposeSam()
                ? null
                : Cp3Layout.resolve(params.getChannel1(), params.getChannel2(), imp.getStackSize());

        // Both allocations live inside the try: allocating the labels outside it would leak the
        // input segment if that second allocation threw.
        NDArray input = null;
        NDArray labels = null;
        ShortProcessor labelProcessor;
        try {
            input = allocate(() -> layout == null ? NDArrays.fromImagePlus(imp) : layout.pack(imp));
            labels = allocate(() -> NDArrays.allocateLabels(width, height));
            Map<String, Object> inputs = buildInputs(params, imp.getStackSize(), layout, input, labels);
            long tileStart = System.nanoTime();
            synchronized (pythonLock()) {
                submit(runScript, inputs, "Segmenting tile " + tile.getImageFile().getName());
            }
            logger.debug("Segmented tile {} ({}x{}, {} channel(s)) in {} ms", tile.getImageFile().getName(),
                    width, height, imp.getStackSize(), elapsedMs(tileStart));
            labelProcessor = NDArrays.labelsToShortProcessor(labels, width, height);
        } finally {
            closeQuietly(input);
            closeQuietly(labels);
            imp.close();
        }

        if (tile.isInMemory()) {
            tile.setLabels(labelProcessor);
        } else {
            ImagePlus maskImp = new ImagePlus(tile.getLabelFile().getName(), labelProcessor);
            IJ.save(maskImp, tile.getLabelFile().getAbsolutePath());
            maskImp.close();
        }

        // Outside the Python lock, so tracing overlaps another tile's segmentation.
        tileReader.accept(tile);
    }

    /**
     * A Python worker kept alive between runs, with the lock serializing access to it and a note of
     * which model it currently holds loaded.
     */
    private static final class Worker {
        final Service service;
        /** Model currently loaded in this worker, or null if none. */
        String loadedModel;

        Worker(Service service) {
            this.service = service;
        }
    }

    /** Live workers, keyed by pixi sub-environment (model family + device). */
    private static final Map<String, Worker> WORKERS = new HashMap<>();

    /** Cellpose-call monitors, keyed by the same environment name; outlive any one worker. */
    private static final Map<String, Object> PYTHON_LOCKS = new ConcurrentHashMap<>();

    /** Shut down every cached Python worker, releasing the GPU memory they hold. */
    public static void shutdownWorkers() {
        Map<String, Worker> snapshot;
        // Snapshot under the class monitor and release it before taking any environment monitor:
        // recovery takes them the other way round, and holding both here would deadlock against it.
        synchronized (ApposeBackend.class) {
            if (WORKERS.isEmpty()) {
                logger.info("No Cellpose Python worker is running");
                return;
            }
            snapshot = new LinkedHashMap<>(WORKERS);
            WORKERS.clear();
        }
        logger.info("Shutting down {} Cellpose Python worker(s)", snapshot.size());
        snapshot.forEach((envName, w) -> {
            // Wait for the tile in flight: this method is reachable from a menu item while a run is
            // in progress, and closing the service under a running model.eval kills it mid-tile.
            synchronized (pythonLockFor(envName)) {
                closeWorker(w);
            }
        });
    }

    /** Close one worker's service, ending its Python process and any task still running in it. */
    private static void closeWorker(Worker w) {
        try {
            ApposeEnvironments.withExtensionClassLoader(() -> {
                w.service.close();
                return null;
            });
        } catch (Exception e) {
            logger.warn("Error closing Appose Cellpose service: {}", e.getMessage(), e);
        } finally {
            LIVE_SERVICES.remove(w.service);
        }
    }

    private synchronized void ensureService(CellposeSegmentationParams params) throws IOException {
        if (worker != null)
            return;

        // Asked before the device is resolved, because the answer decides which environment to build.
        CellposeDevice device = ApposeEnvironments.chooseDeviceOnFirstBuild(params.getDevice());
        String cudaVariant = ApposeEnvironments.cudaVariant(device);
        this.useGpu = ApposeEnvironments.requestGpu(device);
        String envName = ApposeEnvironments.envName(params.isCellposeSam(), cudaVariant);
        String scriptName = params.isCellposeSam() ? "cp4.py" : "cp3.py";
        String initName = params.isCellposeSam() ? "cp4_init.py" : "cp3_init.py";

        this.runScript = ApposeEnvironments.readResource(scriptName);
        this.initScript = ApposeEnvironments.readResource(initName);

        this.envName = envName;
        this.workerParams = params;
        this.workerDevice = device;
        qupath.ext.biop.cellpose.CellposeExtension.setLastApposeFamilyWasSam(params.isCellposeSam());

        boolean reused;
        synchronized (ApposeBackend.class) {
            reused = WORKERS.containsKey(envName);
        }
        this.worker = acquireWorker(envName, params, device);

        // A worker idle since the last run can die on its first task -- before any Python runs --
        // and Appose relaunches that task as a zombie we cannot see. Spend the death on a task with
        // nothing at stake, rather than on a tile holding shared memory.
        if (reused)
            warmUpWorker();
    }

    /**
     * Send a trivial task so that a stale worker fails here rather than on real work.
     *
     * <p>Best-effort: a warm-up that cannot be recovered is left for the real task to report.
     */
    private void warmUpWorker() {
        synchronized (pythonLock()) {
            try {
                submit("task.update(message='Cellpose worker ready')", new HashMap<>(), "Worker warm-up");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } catch (IOException e) {
                logger.warn("Cellpose worker warm-up failed: {}", e.getMessage());
            }
        }
    }

    /**
     * Run in the worker by {@link #prewarm}: the init script has already imported torch and
     * Cellpose by the time this runs, so what is left is the CUDA context and cuDNN, which PyTorch
     * loads on the first convolution and which otherwise lands on the first tile (about 1 s).
     */
    private static final String PREWARM_SCRIPT = String.join("\n",
            "use_gpu, device = get_torch_device(use_gpu)",
            "if device.type == 'cuda':",
            "    x = torch.zeros(1, 2, 64, 64, device=device)",
            "    torch.nn.functional.conv2d(x, torch.zeros(32, 2, 3, 3, device=device), padding=1)",
            "    torch.cuda.synchronize()",
            "task.update(message=f'Worker ready (device={describe_device(device)}; {runtime_versions()})')");

    /**
     * Start the Python worker before the first detection: launch it, import torch and Cellpose,
     * and create the CUDA context. Those take about 5 s on a desktop GPU and are the same for
     * every model, while loading a model takes about 0.2 s, so the first detection after this
     * costs about what every later one does. Called at QuPath start-up when the "start the
     * Python worker with QuPath" preference is on.
     *
     * <p>Best-effort, and never builds anything: when the environment for this family and device
     * has not been installed yet it does nothing, so starting QuPath can never begin a multi-GB
     * download. A failure is logged and the worker then starts on the first detection, as it would
     * have without the preference.
     *
     * @param cellposeSam true to start the Cellpose-SAM (cp4) worker, false for Cellpose 3
     * @param device      the compute device preference
     */
    public static void prewarm(boolean cellposeSam, CellposeDevice device) {
        long start = System.nanoTime();
        ApposeBackend backend = new ApposeBackend(tile -> { });
        try {
            String envName = ApposeEnvironments.envName(cellposeSam, ApposeEnvironments.cudaVariant(device));
            if (!ApposeEnvironments.isInstalled(envName)) {
                logger.info("Not starting the Cellpose Python worker with QuPath: the {} environment is not "
                        + "installed yet; it is built on the first detection", envName);
                return;
            }
            backend.ensureService(CellposeSegmentationParams.builder().cellposeSam(cellposeSam).device(device).build());
            synchronized (backend.pythonLock()) {
                Map<String, Object> inputs = new HashMap<>();
                inputs.put("use_gpu", backend.useGpu);
                backend.submit(PREWARM_SCRIPT, inputs, "Starting the Cellpose Python worker with QuPath");
            }
            logger.info("Started the Cellpose Python worker ({}) with QuPath in {} ms; the first detection "
                    + "only has to load its model", backend.envName, elapsedMs(start));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (IOException | RuntimeException e) {
            logger.warn("Could not start the Cellpose Python worker with QuPath ({}); it will start on the "
                    + "first detection instead", e.getMessage());
        } finally {
            backend.close();
        }
    }

    /** Reuse the worker for this environment if one is already running, otherwise start one. */
    private static synchronized Worker acquireWorker(String envName, CellposeSegmentationParams params,
                                                     CellposeDevice device) throws IOException {
        // A worker holds a Python process rooted in the environment it was started from, so a moved
        // environment directory means every cached worker is pointing at the old one. Reentrant on
        // this monitor; done here rather than in ApposeEnvironments because the lock order is
        // ApposeBackend -> ApposeEnvironments and must not be taken the other way round.
        if (ApposeEnvironments.environmentDirectoryChanged()) {
            logger.info("The Cellpose Appose environment directory changed; stopping the running worker(s)");
            shutdownWorkers();
        }

        Worker existing = WORKERS.get(envName);
        if (existing != null) {
            // A cached worker can be gone: shut down from the menu, or dead on its own.
            if (existing.service.isAlive()) {
                logger.info("Reusing the running Cellpose Python worker for environment {}", envName);
                return existing;
            }
            logger.info("The cached Cellpose Python worker for environment {} is no longer running; starting a new one", envName);
            LIVE_SERVICES.remove(existing.service);
            WORKERS.remove(envName);
        }

        String cpUtils = ApposeEnvironments.readResource("cp_utils.py");
        // numpy must be imported before the stdin reader starts, or the worker deadlocks on
        // Windows. init() replaces rather than appends, so this is one combined string.
        String init = "import numpy\n" + parentWatcherSnippet() + cpUtils;

        logger.info("Starting Appose Cellpose service (device={} -> environment {})", device.name(), envName);
        long startedAt = System.nanoTime();
        try {
            Service created = ApposeEnvironments.withExtensionClassLoader(() -> {
                Environment environment = ApposeEnvironments.getEnvironment();
                // Appose activates a pixi sub-environment by running `pixi install --environment`,
                // which checks the installed packages against the lock and then reports the
                // environment as installed even when it changed nothing. Expected on every start.
                logger.info("Verifying the {} environment against the lock (pixi install --environment {})",
                        envName, envName);
                long activateStart = System.nanoTime();
                Environment activated = environment.activate(envName);
                logger.info("The {} environment was verified in {} ms", envName,
                        elapsedMs(activateStart));
                Service svc = ApposeEnvironments.pinToSelectedGpu(activated).python();
                // stdout is the Appose IPC channel, so the debug callback is the only route by
                // which Python diagnostics reach the user.
                svc.debug(msg -> {
                    logger.debug("[Cellpose Python] {}", msg);
                    if (isConsoleWorthy(msg))
                        PythonConsoleWindow.appendMessage(msg);
                });
                svc.init(init);
                return svc;
            });
            LIVE_SERVICES.add(created);
            logger.info("Cellpose is running in the {} environment ({}); the Python worker started in {} ms",
                    envName, device.name(), elapsedMs(startedAt));
            Worker w = new Worker(created);
            WORKERS.put(envName, w);
            return w;
        } catch (org.apposed.appose.BuildException e) {
            throw new IOException("Failed to activate the Appose environment '" + envName + "'", e);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Failed to start the Appose Cellpose service for environment '" + envName + "'", e);
        }
    }

    /** Build the input globals read by {@code cp3.py}/{@code cp4.py}. */
    private Map<String, Object> buildInputs(CellposeSegmentationParams params, int nChannels,
                                            Cp3Layout layout, NDArray input, NDArray labels) {
        Map<String, Object> inputs = new HashMap<>();

        // Images (shared memory).
        inputs.put("input", input);
        inputs.put("output_labels", labels);
        inputs.put("output_flows", null);

        // 2D tiles, so no Z or T. The channel axis is 0 when a (C, Y, X) stack is sent.
        inputs.put("t_axis", null);
        inputs.put("z_axis", null);
        int sentChannels = layout == null ? nChannels : layout.packedChannels();
        inputs.put("channel_axis", sentChannels > 1 ? Integer.valueOf(0) : null);

        // Model selection.
        boolean custom = params.isCustomModel();
        inputs.put("custom_model", custom ? params.getModel() : null);
        inputs.put("model_name", custom ? null : params.getModel());

        // Core parameters.
        inputs.put("diameter", params.getDiameter());
        inputs.put("use_3D", params.isDo3D());
        inputs.put("flow_threshold", params.getFlowThreshold());
        inputs.put("cellprob_threshold", params.getCellprobThreshold());
        // The resolved device, not the raw builder flag, so this matches the activated environment.
        inputs.put("use_gpu", useGpu);

        // Fixed defaults for the 2D-tile case (mirroring the script defaults).
        inputs.put("stitch_threshold", 0.0);
        inputs.put("anisotropy", 1.0);
        inputs.put("compute_flows", false);
        inputs.put("resample", true);
        inputs.put("normalize", true);
        inputs.put("min_size", 15.0);
        inputs.put("tile_overlap", 0.1);
        inputs.put("flow3D_smooth", 0);
        inputs.put("niter", null);

        if (params.isCellposeSam()) {
            // cp4.py's chan0/chan1/chan2 are 0-based array indices, not --chan/--chan2 values.
            inputs.put("n_channels", nChannels);
            int[] indices = resolveCp4Channels(params.getChannel1(), params.getChannel2(), nChannels);
            inputs.put("chan0", indices.length > 0 ? Integer.valueOf(indices[0]) : null);
            inputs.put("chan1", indices.length > 1 ? Integer.valueOf(indices[1]) : null);
            inputs.put("chan2", indices.length > 2 ? Integer.valueOf(indices[2]) : null);
        } else {
            // cp3.py passes these to model.eval as `channels`, describing the packed array.
            inputs.put("cell_channel", layout.cellChannel());
            inputs.put("nuclei_channel", layout.nucleiChannel());
        }

        return inputs;
    }

    /**
     * Translate cellpose's {@code --chan}/{@code --chan2} numbers (1-based, {@code 0} meaning
     * grayscale) into the 0-based channel indices {@code cp4.py} uses to slice the tile. With
     * neither set, all exported channels are used, truncated to three with a warning.
     *
     * @param chan      the {@code --chan} value, or null if unset
     * @param chan2     the {@code --chan2} value, or null if unset
     * @param nChannels the number of channels in the exported tile
     * @return the 0-based channel indices to send as {@code chan0}/{@code chan1}/{@code chan2}
     * @throws IllegalArgumentException if a requested channel does not exist in the tile
     */
    static int[] resolveCp4Channels(Integer chan, Integer chan2, int nChannels) {
        List<Integer> requested = new ArrayList<>();
        for (Integer value : new Integer[] {chan, chan2}) {
            if (value == null || value == 0)
                continue;
            int index = value - 1;
            if (index >= nChannels)
                throw new IllegalArgumentException(channelOutOfRangeMessage(chan, chan2, value, nChannels));
            if (!requested.contains(index))
                requested.add(index);
        }

        if (requested.isEmpty()) {
            int used = Math.min(nChannels, CELLPOSE_SAM_MAX_CHANNELS);
            if (used < nChannels)
                logger.warn("Cellpose-SAM accepts at most {} channels; using the first {} of the {} exported channels. "
                        + "Use .channels(...) to export fewer channels, or cellposeChannels(...) to choose explicitly.",
                        CELLPOSE_SAM_MAX_CHANNELS, used, nChannels);
            int[] all = new int[used];
            for (int i = 0; i < used; i++)
                all[i] = i;
            return all;
        }

        int[] indices = new int[requested.size()];
        for (int i = 0; i < indices.length; i++)
            indices[i] = requested.get(i);
        return indices;
    }

    /**
     * How a tile's channels are packed for Cellpose 3, together with the cellpose channel spec
     * describing the packed array.
     * <p>
     * Only the channels in use are sent: given a stack of more than three channels Cellpose treats
     * the channel axis as Z and returns an empty mask with no error, even when {@code channel_axis}
     * is passed.
     *
     * @param bands         0-based slice indices to send, in order, or null to average all channels
     * @param cellChannel   the cellpose {@code --chan} value describing the packed array
     * @param nucleiChannel the cellpose {@code --chan2} value describing the packed array, or null
     */
    record Cp3Layout(int[] bands, Integer cellChannel, Integer nucleiChannel) {

        /** Number of channels actually sent to Python (1 means a plain 2D plane). */
        int packedChannels() {
            return bands == null ? 1 : bands.length;
        }

        /** Build the input NDArray for this layout. The caller must close it. */
        NDArray pack(ImagePlus imp) {
            return bands == null ? NDArrays.meanOfChannels(imp) : NDArrays.fromImagePlusChannels(imp, bands);
        }

        /**
         * Work out the packing for a tile from the cellpose channel numbers.
         *
         * @param chan      the {@code --chan} value, or null if unset
         * @param chan2     the {@code --chan2} value, or null if unset
         * @param nChannels the number of channels in the exported tile
         * @return the layout to use
         * @throws IllegalArgumentException if a requested channel does not exist in the tile
         */
        static Cp3Layout resolve(Integer chan, Integer chan2, int nChannels) {
            List<Integer> requested = new ArrayList<>();
            for (Integer value : new Integer[] {chan, chan2}) {
                if (value == null || value == 0)
                    continue;
                if (value > nChannels)
                    throw new IllegalArgumentException(channelOutOfRangeMessage(chan, chan2, value, nChannels));
                if (!requested.contains(value - 1))
                    requested.add(value - 1);
            }

            if (requested.isEmpty()) {
                // Grayscale: a single-channel tile needs no averaging, so send it untouched.
                return nChannels == 1
                        ? new Cp3Layout(new int[] {0}, 0, null)
                        : new Cp3Layout(null, 0, null);
            }
            if (requested.size() == 1)
                return new Cp3Layout(new int[] {requested.get(0)}, 0, null);
            return new Cp3Layout(new int[] {requested.get(0), requested.get(1)}, 1, 2);
        }
    }

    private static String channelOutOfRangeMessage(Integer chan, Integer chan2, int value, int nChannels) {
        return String.format(
                "cellposeChannels(%s, %s) asks for channel %d, but the exported tile only has %d channel(s). "
                        + "These values are cellpose's --chan/--chan2 (1-based, 0 = grayscale) and must refer to "
                        + "channels selected by .channels(...) in the builder.",
                chan, chan2, value, nChannels);
    }

    /**
     * Submit one script to the worker, recovering from the Appose "thread death" failure.
     *
     * <p>Retrying that failure on its own is not safe. When a worker dies before running any Python,
     * Appose drops the task and relaunches it inside the same worker, where we can neither observe
     * nor cancel it; a plain re-submit would then run the same work twice at once. So a retry
     * happens only when no Python ran -- an UPDATE from the script is the observable gate -- and
     * only after the worker has been replaced, which ends the relaunched task with its process.
     * A failure after Python started is reported as-is.
     */
    private void submit(String script, Map<String, Object> inputs, String description) throws IOException, InterruptedException {
        IOException last = null;
        for (int attempt = 1; attempt <= THREAD_DEATH_ATTEMPTS; attempt++) {
            AtomicBoolean pythonStarted = new AtomicBoolean();
            try {
                submitOnce(script, inputs, description, pythonStarted);
                return;
            } catch (IOException e) {
                if (!isThreadDeath(e))
                    throw e;
                if (pythonStarted.get()) {
                    logger.warn("{}: the Python worker reported 'thread death' after the script had started"
                            + " (attempt {} of {}); not retried, because the task may still be running in the"
                            + " worker. Check the appose version in the worker's start-up line: this failure"
                            + " is spurious on appose-python older than 0.12.",
                            description, attempt, THREAD_DEATH_ATTEMPTS);
                    throw new IOException(description + " failed after Python had started, so it was not"
                            + " retried: the work may still be running in the Python worker", e);
                }
                if (recovering)
                    throw e;
                last = e;
                if (attempt < THREAD_DEATH_ATTEMPTS) {
                    logger.warn("{}: the Python worker died before running anything (attempt {} of {});"
                            + " replacing it and retrying", description, attempt, THREAD_DEATH_ATTEMPTS);
                    Thread.sleep(THREAD_DEATH_RETRY_DELAY_MS * attempt);
                    long recoverStart = System.nanoTime();
                    recoverWorker();
                    logger.info("{}: replaced the Python worker in {} ms", description,
                            elapsedMs(recoverStart));
                }
            }
        }
        throw new IOException(description + " failed after " + THREAD_DEATH_ATTEMPTS
                + " attempts: the Python worker died each time", last);
    }

    /**
     * Replace a dead worker and restore the state the new one needs.
     *
     * <p>Called while holding {@link #pythonLock()}, which is why that monitor belongs to the
     * environment rather than to the worker being replaced.
     */
    private void recoverWorker() throws IOException, InterruptedException {
        Worker dead = this.worker;
        synchronized (ApposeBackend.class) {
            if (WORKERS.get(envName) == dead)
                WORKERS.remove(envName);
        }
        if (dead != null)
            closeWorker(dead);
        recovering = true;
        try {
            this.worker = acquireWorker(envName, workerParams, workerDevice);
            workerReplacements.incrementAndGet();
            // The replacement holds no model, and a tile submitted without one segments nothing.
            if (workerParams != null)
                initializeModel(workerParams);
        } finally {
            recovering = false;
        }
    }

    /**
     * True if this failure is Appose reporting {@code Task failed: thread death} -- a worker task
     * thread that died before reporting completion, most often on the first task after a service
     * starts -- rather than an error raised by Python.
     */
    private static boolean isThreadDeath(Throwable t) {
        for (Throwable cause = t; cause != null; cause = cause.getCause()) {
            String message = cause.getMessage();
            if (message != null && message.toLowerCase().contains("thread death"))
                return true;
        }
        return false;
    }

    private void submitOnce(String script, Map<String, Object> inputs, String description,
                           AtomicBoolean pythonStarted) throws IOException, InterruptedException {
        Task task;
        try {
            task = ApposeEnvironments.withExtensionClassLoader(() -> {
                Task t = worker.service.task(script, inputs);
                t.listen(event -> {
                    // Every vendored script reports progress as soon as its imports succeed, so an
                    // UPDATE is the observable proof that Python ran.
                    if (event.responseType == Service.ResponseType.UPDATE)
                        pythonStarted.set(true);
                    relay(event);
                });
                t.start();
                return t;
            });
        } catch (Exception e) {
            throw new IOException(description + " failed to start: " + e.getMessage(), e);
        }

        try {
            ApposeEnvironments.withExtensionClassLoader(() -> {
                task.waitFor();
                return null;
            });
        } catch (InterruptedException e) {
            cancelQuietly(task);
            Thread.currentThread().interrupt();
            throw e;
        } catch (TaskException e) {
            throw new IOException(description + " failed: " + e.getMessage(), e);
        } catch (Exception e) {
            throw new IOException(description + " failed: " + e.getMessage(), e);
        }

        if (task.status != TaskStatus.COMPLETE) {
            throw new IOException(description + " failed with status " + task.status + ": " + task.error);
        }
    }

    private void cancelQuietly(Task task) {
        try {
            ApposeEnvironments.withExtensionClassLoader(() -> {
                task.cancel();
                return null;
            });
        } catch (Exception e) {
            logger.debug("Error cancelling Cellpose task: {}", e.getMessage());
        }
    }

    /** Allocate an NDArray under the extension classloader (ShmFactory ServiceLoader needs it). */
    private static NDArray allocate(Callable<NDArray> allocation) throws IOException {
        try {
            return ApposeEnvironments.withExtensionClassLoader(allocation);
        } catch (IOException e) {
            throw e;
        } catch (Exception e) {
            throw new IOException("Failed to allocate a shared-memory array for Cellpose: " + e.getMessage(), e);
        }
    }

    private static void closeQuietly(NDArray ndArray) {
        if (ndArray == null)
            return;
        try {
            ApposeEnvironments.withExtensionClassLoader(() -> {
                ndArray.close();
                return null;
            });
        } catch (Exception e) {
            logger.warn("Error closing shared-memory array: {}", e.getMessage(), e);
        }
    }

    private static void relay(TaskEvent event) {
        if (event.message != null && !event.message.isEmpty()) {
            String line;
            if (event.maximum > 0)
                line = "Cellpose: " + event.message + " (" + event.current + "/" + event.maximum + ")";
            else
                line = "Cellpose: " + event.message;
            logger.info(line);
            PythonConsoleWindow.appendMessage(line);
        }
    }

    /**
     * Decide whether a raw Appose debug line belongs in the user-facing Python console: everything
     * human-readable, but not the routine IPC request/response JSON unless it reports a failure.
     */
    private static boolean isConsoleWorthy(String msg) {
        if (msg == null || msg.isBlank())
            return false;
        String t = msg.strip();
        boolean protocolJson = t.indexOf('{') >= 0
                && (t.contains("\"requestType\"") || t.contains("\"responseType\"") || t.contains("\"script\""));
        if (!protocolJson)
            return true;
        String lower = t.toLowerCase();
        return lower.contains("failure") || lower.contains("\"error\"")
                || lower.contains("traceback") || lower.contains("exception");
    }

    /**
     * A Python daemon, injected into the init script, that exits the worker if QuPath dies. On
     * Windows it must use {@code OpenProcess}, not {@code os.kill(pid, 0)}, which crashes the
     * target. Internal names take a leading underscore so the worker does not export them to task
     * scripts.
     */
    private static String parentWatcherSnippet() {
        return String.join("\n",
                "import os as _os, sys as _sys, threading as _thr, time as _time",
                "def _watch_parent():",
                "    _ppid = _os.getppid()",
                "    _is_win = _sys.platform.startswith('win')",
                "    _k = None",
                "    if _is_win:",
                "        import ctypes as _ct",
                "        _k = _ct.windll.kernel32",
                "    while True:",
                "        _time.sleep(2.0)",
                "        _alive = True",
                "        try:",
                "            if _is_win:",
                "                _h = _k.OpenProcess(0x1000, False, _ppid)",
                "                if not _h:",
                "                    _alive = False",
                "                else:",
                "                    _code = _ct.c_ulong(0)",
                "                    if _k.GetExitCodeProcess(_h, _ct.byref(_code)) and _code.value != 259:",
                "                        _alive = False",
                "                    _k.CloseHandle(_h)",
                "            else:",
                "                _os.kill(_ppid, 0)",
                "        except Exception:",
                "            _alive = False",
                "        if not _alive:",
                "            _os._exit(1)",
                "_thr.Thread(target=_watch_parent, daemon=True).start()",
                "");
    }

    private static void installShutdownHook() {
        if (SHUTDOWN_HOOK_INSTALLED.compareAndSet(false, true)) {
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                for (Service svc : LIVE_SERVICES) {
                    try {
                        svc.close();
                    } catch (Exception e) {
                        try {
                            svc.kill();
                        } catch (Exception ignore) {
                            // best effort at JVM shutdown
                        }
                    }
                }
            }, "cellpose-appose-shutdown"));
        }
    }

    @Override
    public void close() {
        // The worker outlives this backend; it is closed at JVM shutdown or from Extensions >
        // Cellpose. Only the reference is dropped here.
        worker = null;
    }
}
