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

import org.apposed.appose.Appose;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.biop.cellpose.CellposeExtension;
import qupath.ext.biop.cellpose.ui.PythonConsoleWindow;

import java.awt.GraphicsEnvironment;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import javafx.scene.control.ButtonType;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;

/**
 * Builds and caches the Appose (pixi) environment used by the in-process {@link ApposeBackend}.
 * <p>
 * The vendored {@code pixi.toml} declares the {@code cp3-*}/{@code cp4-*} sub-environments; the two
 * Cellpose major versions cannot share one environment, so callers activate the sub-environment
 * matching the model family.
 * <p>
 * The environment-build call and the CUDA-suffix naming scheme are adapted, with attribution, from
 * the BSD-3-Clause imglib2-cellpose project; see NOTICE.
 */
final class ApposeEnvironments {

    private static final Logger logger = LoggerFactory.getLogger(ApposeEnvironments.class);

    /** Root of the vendored Appose resources (scripts and pixi manifest). */
    static final String RESOURCE_ROOT = "/qupath/ext/biop/cellpose/appose/";

    /**
     * Directory (under the Appose base) holding this extension's Python environment. Deliberately
     * not the manifest's own workspace name, {@code cellpose-appose}, which the Fiji cellpose-appose
     * plugin also uses: sharing a directory would have the two tools overwrite each other's manifest
     * and lock.
     */
    private static final String ENV_DIR_NAME = "qupath-cellpose-appose";

    private static final Object LOCK = new Object();
    private static volatile Environment environment;

    /**
     * Directory {@link #environment} was built in, or null if nothing is built. Compared against the
     * current preference so a changed location is noticed without a restart.
     */
    private static volatile Path builtAt;

    private static volatile boolean firstRunWarned = false;

    private ApposeEnvironments() {}

    /**
     * Build (once) and return the shared Appose environment from the vendored pixi manifest.
     *
     * @return the shared Appose environment
     * @throws IOException if the pixi manifest cannot be read or the environment cannot be built
     */
    static Environment getEnvironment() throws IOException {
        Environment local = environment;
        if (local != null && getEnvironmentPath().equals(builtAt))
            return local;
        synchronized (LOCK) {
            if (environment != null && !getEnvironmentPath().equals(builtAt)) {
                logger.info("The Cellpose Appose environment directory changed from {} to {}; "
                        + "rebuilding at the new location", builtAt, getEnvironmentPath());
                environment = null;
                builtAt = null;
                // The new location gets its own multi-GB download, so the warning applies again.
                firstRunWarned = false;
            }
            if (environment == null) {
                String pixiToml = readResource("pixi.toml");
                String pixiLock = readResource("pixi.lock");

                Path envDir = getEnvironmentPath();
                boolean firstBuild = !Files.exists(envDir.resolve(".pixi"));
                warnIfUnsuitable(getEnvironmentBase());

                // The committed pixi.lock must stay at format v6: the pixi Appose bundles rejects
                // v7 and every install then fails.
                syncManifest(envDir, pixiToml, pixiLock);

                if (firstBuild)
                    warnFirstRun();

                environment = buildEnvironment(pixiToml, envDir, firstBuild);
                builtAt = envDir;
            }
            return environment;
        }
    }

    /**
     * Whether a built environment exists somewhere other than the currently configured location.
     *
     * <p>Called from {@code ApposeBackend} while it holds its own monitor, so it takes no lock here:
     * {@link #getEnvironment()} makes the same comparison authoritatively before rebuilding.
     *
     * @return true if the environment directory preference has moved since the environment was built
     */
    static boolean environmentDirectoryChanged() {
        return environment != null && directoryMovedFrom(builtAt);
    }

    /**
     * Whether the configured environment directory now resolves somewhere other than {@code built}.
     *
     * @param built the directory an environment was built in, or null if none has been
     * @return true if the two differ
     */
    static boolean directoryMovedFrom(Path built) {
        return built != null && !getEnvironmentPath().equals(built);
    }

    /**
     * Build the environment, routing pixi progress and output to the log and the Python console. On
     * a lock-format mismatch the staged lock is deleted and the build retried once, resolving fresh;
     * a Windows file lock is surfaced with manual recovery steps and never auto-wiped.
     */
    private static Environment buildEnvironment(String pixiToml, Path envDir, boolean firstBuild) throws IOException {
        if (firstBuild) {
            String bar = "*".repeat(78);
            logger.info(bar);
            logger.info("**  BUILDING THE CELLPOSE PYTHON ENVIRONMENT (first run only)");
            logger.info("**  Downloading a multi-GB PyTorch/Cellpose environment. This can take");
            logger.info("**  SEVERAL MINUTES. QuPath is NOT frozen -- watch progress in");
            logger.info("**  Extensions > Cellpose > Python console.");
            logger.info(bar);
        } else {
            logger.info("Preparing the Cellpose Appose environment at {}...", envDir);
        }
        long buildStart = System.nanoTime();
        try {
            Environment built = runBuild(pixiToml);
            logger.info("The Cellpose Appose environment at {} is ready ({} ms)", envDir,
                    elapsedMs(buildStart));
            return built;
        } catch (BuildException e) {
            String causes = collectCauseMessages(e);
            if (looksLikeWindowsFileLock(causes)) {
                surfaceWindowsFileLock(envDir);
                throw new IOException("Cellpose environment build blocked by a Windows file lock; "
                        + "see the recovery steps in the log/notification", e);
            }
            if (looksLikeLockFormatSkew(causes)) {
                logger.warn("The staged pixi.lock could not be read by this pixi (format skew); "
                        + "deleting it and re-resolving once. Cause: {}", causes);
                try {
                    Files.deleteIfExists(envDir.resolve("pixi.lock"));
                } catch (IOException ignore) {
                    // best effort; the retry below will surface any real problem
                }
                try {
                    return runBuild(pixiToml);
                } catch (BuildException e2) {
                    String causes2 = collectCauseMessages(e2);
                    if (looksLikeWindowsFileLock(causes2)) {
                        surfaceWindowsFileLock(envDir);
                        throw new IOException("Cellpose environment build blocked by a Windows file lock; "
                                + "see the recovery steps in the log/notification", e2);
                    }
                    throw new IOException("Failed to build the Appose (pixi) environment for Cellpose "
                            + "(after re-resolving without the staged lock)", e2);
                }
            }
            throw new IOException("Failed to build the Appose (pixi) environment for Cellpose", e);
        }
    }

    private static Environment runBuild(String pixiToml) throws BuildException {
        try {
            return withExtensionClassLoader(() -> Appose.pixi()
                    .content(pixiToml)
                    // base() sets the environment directory itself; it does not compose with
                    // name(), so this takes the full path and name() must not also be set.
                    .base(getEnvironmentPath().toFile())
                    .subscribeProgress((title, current, maximum) -> {
                        String line = "Cellpose env build: " + title + " (" + current + "/" + maximum + ")";
                        logger.info(line);
                        PythonConsoleWindow.appendMessage(line);
                    })
                    .subscribeOutput(line -> {
                        logger.info("[pixi] {}", line);
                        PythonConsoleWindow.appendMessage("[pixi] " + line);
                    })
                    .subscribeError(line -> {
                        logger.warn("[pixi] {}", line);
                        PythonConsoleWindow.appendMessage("[pixi] " + line);
                    })
                    .build());
        } catch (BuildException e) {
            throw e;
        } catch (Exception e) {
            throw new BuildException("Unexpected error building the Cellpose Appose environment", e);
        }
    }

    /**
     * Stage the manifest and the committed lock into the environment directory before building. A
     * changed manifest wipes the existing {@code .pixi} env to force a clean reinstall, and a staged
     * lock the bundled pixi cannot read is cleared first.
     *
     * @param envDir       the environment directory
     * @param expectedToml the bundled manifest content
     * @param expectedLock the bundled lock content
     */
    private static void syncManifest(Path envDir, String expectedToml, String expectedLock) throws IOException {
        Path tomlFile = envDir.resolve("pixi.toml");
        Path lockFile = envDir.resolve("pixi.lock");

        clearIncompatibleLock(lockFile);
        Files.createDirectories(envDir);

        boolean changed = !Files.exists(tomlFile)
                || !normalize(Files.readString(tomlFile, StandardCharsets.UTF_8)).equals(normalize(expectedToml));
        if (changed && Files.exists(tomlFile))
            deletePixiDir(envDir.resolve(".pixi"));

        Files.writeString(tomlFile, expectedToml, StandardCharsets.UTF_8);
        Files.writeString(lockFile, expectedLock, StandardCharsets.UTF_8);
    }

    /**
     * Delete an on-disk {@code pixi.lock} whose format is too new for the bundled pixi (version 7+).
     * Best-effort: a malformed lock is removed too, so the build can proceed.
     */
    private static void clearIncompatibleLock(Path lockFile) {
        try {
            if (!Files.exists(lockFile))
                return;
            for (String line : Files.readAllLines(lockFile, StandardCharsets.UTF_8)) {
                String t = line.trim();
                if (t.startsWith("version:")) {
                    int v = Integer.parseInt(t.substring("version:".length()).trim());
                    if (v >= 7)
                        Files.deleteIfExists(lockFile);
                    return;
                }
            }
        } catch (Exception e) {
            try {
                Files.deleteIfExists(lockFile);
            } catch (IOException ignore) {
                // best effort
            }
        }
    }

    private static String normalize(String s) {
        return s.replace("\r\n", "\n").strip();
    }

    /**
     * Best-effort recursive delete of the {@code .pixi} directory, falling back to rename-then-delete
     * when a locked file blocks it on Windows.
     */
    private static void deletePixiDir(Path pixiDir) throws IOException {
        if (!Files.exists(pixiDir))
            return;
        try {
            deleteRecursively(pixiDir);
        } catch (IOException e) {
            Path renamed = pixiDir.resolveSibling(".pixi-old-" + System.currentTimeMillis());
            try {
                Files.move(pixiDir, renamed);
                deleteRecursively(renamed);
            } catch (IOException e2) {
                throw new IOException("Could not remove the existing Cellpose environment at " + pixiDir, e2);
            }
        }
    }

    private static void deleteRecursively(Path path) throws IOException {
        if (!Files.exists(path))
            return;
        try (var walk = Files.walk(path)) {
            walk.sorted((a, b) -> b.getNameCount() - a.getNameCount())
                    .forEach(p -> {
                        try {
                            Files.deleteIfExists(p);
                        } catch (IOException e) {
                            throw new RuntimeException(e);
                        }
                    });
        } catch (RuntimeException e) {
            if (e.getCause() instanceof IOException io)
                throw io;
            throw e;
        }
    }

    /**
     * Whether a dialog or notification can be shown at all.
     *
     * <p>{@code GraphicsEnvironment.isHeadless()} is false when QuPath runs a script from the
     * command line, but the JavaFX toolkit is never started there, so every dialog throws
     * {@code ExceptionInInitializerError} -- an {@code Error}, which a catch for
     * {@code RuntimeException} does not cover.
     *
     * @return true if there is a running QuPath window to show it in
     */
    private static boolean canPrompt() {
        if (GraphicsEnvironment.isHeadless())
            return false;
        try {
            return qupath.lib.gui.QuPathGUI.getInstance() != null;
        } catch (RuntimeException | LinkageError e) {
            return false;
        }
    }

    /**
     * Before the very first build, let the user pick CPU or GPU, and where it goes.
     *
     * <p>Returns the requested device unchanged when an environment already exists or when no
     * dialog can be shown, so scripted and headless runs are unaffected.
     *
     * @param requested the device the preference or the builder asked for
     * @return the device to install and use
     * @throws IOException if the user cancels
     */
    static CellposeDevice chooseDeviceOnFirstBuild(CellposeDevice requested) throws IOException {
        if (Files.exists(getEnvironmentPath().resolve(".pixi")))
            return requested; // already built; nothing to decide
        if (!canPrompt())
            return requested;

        GpuOutlook outlook = gpuOutlook();
        String gpuChoice = "GPU -- " + outlook.description();
        String cpuChoice = "CPU -- always works, but segmentation is slower";
        String elsewhere = "Choose a different location first...";

        while (true) {
            String preferred = requested == CellposeDevice.CPU ? cpuChoice
                    : (outlook.likely() ? gpuChoice : cpuChoice);
            String answer;
            try {
                answer = qupath.fx.dialogs.Dialogs.showChoiceDialog("Cellpose",
                        "Cellpose needs to build a Python environment before it can run.\n\n"
                                + "Location:  " + getEnvironmentPath() + "\n"
                                + "Size:      several GB, which can take a few minutes\n\n"
                                + "This sets the Cellpose Appose device preference, which can be "
                                + "changed later in Edit > Preferences > Cellpose. Cancel stops "
                                + "before anything is downloaded.",
                        new String[] {gpuChoice, cpuChoice, elsewhere}, preferred);
            } catch (RuntimeException | LinkageError e) {
                // A prompt that cannot be shown must not stop a run that would otherwise work.
                logger.debug("Could not ask which Cellpose environment to build: {}", e.getMessage());
                return requested;
            }

            if (answer == null)
                throw new IOException("Building the Cellpose Python environment was cancelled");
            if (elsewhere.equals(answer)) {
                File chosen = qupath.fx.dialogs.FileChoosers.promptForDirectory(
                        "Where should the Cellpose Python environment go?", getEnvironmentBase().toFile());
                if (chosen != null) {
                    CellposeExtension.setApposeEnvDirPreference(chosen.getAbsolutePath());
                    logger.info("Cellpose Appose environment directory set to {}", chosen.getAbsolutePath());
                    if (Files.exists(getEnvironmentPath().resolve(".pixi")))
                        return requested; // an environment already exists there
                }
                continue; // round again, so the chosen path is on screen before committing to it
            }

            CellposeDevice device = gpuChoice.equals(answer) ? CellposeDevice.GPU : CellposeDevice.CPU;
            CellposeExtension.setDevicePreference(device);
            logger.info("Cellpose Appose device set to {} for the first environment build", device.name());
            return device;
        }
    }

    /** Whether the GPU is likely to work here, and a line saying why. */
    private record GpuOutlook(boolean likely, String description) {}

    /**
     * Assess whether this machine can actually use a GPU, for the first-build prompt.
     *
     * @return the outlook and a description naming the hardware where there is any
     */
    private static GpuOutlook gpuOutlook() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("mac") || os.contains("darwin"))
            return new GpuOutlook(true, "Apple Metal (MPS), used when available on Apple Silicon");

        String nvidiaSmi = nvidiaSmi();
        if (nvidiaSmi == null)
            return new GpuOutlook(false, "no NVIDIA GPU detected on this system");

        String details = runQuietly(List.of(nvidiaSmi, "--query-gpu=name,compute_cap", "--format=csv,noheader"));
        String first = details == null ? null : details.lines()
                .map(String::strip).filter(line -> !line.isEmpty()).findFirst().orElse(null);
        String capability = computeCapability();
        String build = cudaBuildFor(capability);
        if (build == null)
            return new GpuOutlook(false, (first == null ? "the GPU" : first)
                    + " is older than the bundled CUDA builds support");
        return new GpuOutlook(true, (first == null ? "NVIDIA GPU detected" : first) + ", using " + build);
    }


    /**
     * @return the directory containing the environment: the user's chosen location, or the Appose
     * default when the preference is empty
     */
    static Path getEnvironmentBase() {
        String configured = CellposeExtension.getApposeEnvDirPreference();
        if (!configured.isEmpty())
            return Paths.get(configured);
        return Paths.get(System.getProperty("user.home"), ".local", "share", "appose");
    }

    /** @return the directory Appose installs the Cellpose environment into. */
    static Path getEnvironmentPath() {
        return getEnvironmentBase().resolve(ENV_DIR_NAME);
    }

    /**
     * Warn about a base directory pixi cannot build in, namely one whose path contains a space.
     *
     * @param base the chosen containing directory
     */
    private static void warnIfUnsuitable(Path base) {
        if (base.toString().contains(" ")) {
            logger.warn("The Cellpose Appose environment directory contains a space: {}. "
                    + "pixi cannot build in such a path; choose a directory without spaces in "
                    + "Edit > Preferences > Cellpose.", base);
        }
    }

    /**
     * Resolve which CUDA build the in-process backend should use, if any.
     *
     * @param device the requested device (never null)
     * @return {@code cu126} or {@code cu130}, or null to use the CPU build
     */
    static String cudaVariant(CellposeDevice device) {
        if (device == CellposeDevice.CPU)
            return null;
        String nvidiaSmi = nvidiaSmi();
        if (nvidiaSmi == null) {
            if (device == CellposeDevice.GPU)
                logger.warn("The Cellpose Appose device is set to GPU, but no NVIDIA GPU was found; using the CPU environment");
            return null;
        }
        String capability = computeCapability();
        String build = cudaBuildFor(capability);
        if (build == null) {
            logger.warn("The GPU reports compute capability {}, which none of the bundled CUDA builds support; "
                    + "using the CPU environment. The CUDA builds cover 5.0 and above.", capability);
        } else {
            logger.info("NVIDIA GPU compute capability {} -> {} environment",
                    capability == null ? "unknown" : capability, build);
        }
        return build;
    }

    /**
     * Whether Python should be asked for a GPU.
     *
     * <p>Deliberately not the same question as {@link #cudaVariant(CellposeDevice)}. macOS has no
     * CUDA build, so it always gets the CPU environment, but its PyTorch carries MPS -- and
     * {@code cp_utils.get_torch_device} resolves CUDA, then MPS, then CPU only when it is asked for
     * a GPU at all. Tying this to the CUDA variant would silently disable MPS on Apple Silicon.
     * Where no accelerator exists the Python side falls back to CPU on its own.
     *
     * @param device the requested device (never null)
     * @return false only when the user asked for CPU
     */
    static boolean requestGpu(CellposeDevice device) {
        return device != CellposeDevice.CPU;
    }

    /**
     * The CUDA build whose PyTorch supports a given GPU compute capability.
     *
     * <p>Measured architecture lists: {@code cu126} builds {@code sm_50..sm_90} and ships no PTX, so
     * it cannot run anything newer; {@code cu130} builds {@code sm_75..sm_120} plus {@code
     * compute_120} PTX, so it covers newer cards and can JIT for later ones. Where both apply,
     * {@code cu126} wins so that machines with an environment already built do not have to download
     * another one.
     *
     * @param computeCapability the capability as reported by nvidia-smi, e.g. {@code 8.6}, or null
     *                          if it could not be read
     * @return the sub-environment suffix, or null if no bundled CUDA build supports the card
     */
    static String cudaBuildFor(String computeCapability) {
        // A driver too old to report the capability predates every card that needs cu130.
        if (computeCapability == null || computeCapability.isBlank())
            return "cu126";
        int capability;
        try {
            String[] parts = computeCapability.strip().split("\\.");
            capability = Integer.parseInt(parts[0]) * 10
                    + (parts.length > 1 ? Integer.parseInt(parts[1]) : 0);
        } catch (RuntimeException e) {
            logger.warn("Could not read the GPU compute capability from '{}'; assuming cu126", computeCapability);
            return "cu126";
        }
        if (capability < 50)
            return null;
        if (capability <= 90)
            return "cu126";
        return "cu130";
    }

    /**
     * Name of the pixi sub-environment to activate for a given model family and CUDA build.
     *
     * @param cellposeSam true for the Cellpose-SAM (Cellpose 4) family, false for Cellpose 3
     * @param cudaVariant the CUDA build, or null for the CPU build (see {@link #cudaVariant})
     * @return the sub-environment name, e.g. {@code cp3-cpu} or {@code cp4-cu130}
     */
    static String envName(boolean cellposeSam, String cudaVariant) {
        String family = cellposeSam ? "cp4" : "cp3";
        return family + "-" + (cudaVariant == null ? "cpu" : cudaVariant);
    }

    /**
     * Every card nvidia-smi reports, with compute capability where the driver can report it.
     *
     * @param nvidiaSmi the nvidia-smi executable to run
     * @return the cards, possibly empty
     */
    private static List<CellposeGpu.Gpu> listGpus(String nvidiaSmi) {
        String output = runQuietly(List.of(nvidiaSmi, "--query-gpu=index,uuid,name,compute_cap,memory.total,memory.free",
                "--format=csv,noheader,nounits"));
        if (output != null)
            return CellposeGpu.parse(output, true);
        // Drivers older than about 510 have no compute_cap field and reject the whole query.
        return CellposeGpu.parse(runQuietly(List.of(nvidiaSmi, "--query-gpu=index,uuid,name,memory.total,memory.free",
                "--format=csv,noheader,nounits")), false);
    }

    /**
     * Run a command and return its output, or null if it fails or times out.
     *
     * @param command the command and its arguments
     * @return the combined output, or null
     */
    private static String runQuietly(List<String> command) {
        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process process = pb.start();
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (!process.waitFor(15, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                return null;
            }
            return process.exitValue() == 0 ? output : null;
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException)
                Thread.currentThread().interrupt();
            return null;
        }
    }

    /**
     * Read one of the vendored Appose resources as a UTF-8 string.
     *
     * @param name the resource file name, relative to {@link #RESOURCE_ROOT}
     * @return the resource content
     * @throws IOException if the resource is missing or cannot be read
     */
    static String readResource(String name) throws IOException {
        String path = RESOURCE_ROOT + name;
        try (InputStream stream = ApposeEnvironments.class.getResourceAsStream(path)) {
            if (stream == null)
                throw new IOException("Missing bundled Appose resource: " + path);
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    /**
     * Milliseconds since a {@link System#nanoTime()} reading. Durations use the monotonic clock: the
     * wall clock can step backwards (an NTP or WSL time correction), which logged a tile batch as
     * taking -687 ms.
     *
     * @param startNanos the {@code System.nanoTime()} reading at the start
     * @return the elapsed milliseconds
     */
    static long elapsedMs(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000L;
    }

    /**
     * Run the given body with the extension's classloader as the thread context classloader, which
     * Appose's {@code ServiceLoader} lookups need to find the registrations in the shaded JAR.
     *
     * @param body the work to run
     * @param <T>  the result type
     * @return the body's result
     * @throws Exception whatever the body throws
     */
    static <T> T withExtensionClassLoader(Callable<T> body) throws Exception {
        Thread thread = Thread.currentThread();
        ClassLoader previous = thread.getContextClassLoader();
        thread.setContextClassLoader(ApposeEnvironments.class.getClassLoader());
        try {
            return body.call();
        } finally {
            thread.setContextClassLoader(previous);
        }
    }

    /**
     * Locate a working {@code nvidia-smi}: the PATH first and, on Windows, {@code System32} and
     * whatever {@code where} resolves. Always null on macOS, which has no NVIDIA support.
     *
     * @return the executable to run, or null if no NVIDIA GPU appears to be available
     */
    /**
     * Detection results, cached for the session: the hardware does not change while QuPath runs,
     * and every detection call spawns nvidia-smi, which a project batch would do once per image.
     */
    private static final Object GPU_LOCK = new Object();
    private static boolean gpuProbed;
    private static String nvidiaSmiPath;
    private static List<CellposeGpu.Gpu> gpus = List.of();

    private static String nvidiaSmi() {
        probeGpu();
        return nvidiaSmiPath;
    }

    /**
     * Compute capability of the card the worker will run on (see {@link CellposeGpu}), e.g.
     * {@code 8.6}, or null if there is none to ask.
     */
    private static String computeCapability() {
        return workerGpu().map(CellposeGpu.Gpu::computeCapability).orElse(null);
    }

    /** The card the worker will run on, if any. */
    static java.util.Optional<CellposeGpu.Gpu> workerGpu() {
        probeGpu();
        return CellposeGpu.effective(gpus, CellposeGpu.preferredDevice(), System.getenv("CUDA_VISIBLE_DEVICES"));
    }

    /**
     * {@code environment} with its workers pinned to the selected card. Logs the cards and the
     * choice so a multi-GPU report shows which card was used.
     */
    static Environment pinToSelectedGpu(Environment environment) {
        probeGpu();
        String inherited = System.getenv("CUDA_VISIBLE_DEVICES");
        java.util.Optional<CellposeGpu.Gpu> selected =
                CellposeGpu.select(gpus, CellposeGpu.preferredDevice(), inherited);
        if (gpus.size() > 1 || selected.isEmpty() && !gpus.isEmpty()) {
            if (selected.isPresent())
                logger.info("Cellpose runs on GPU {} (preference '{}'). Cards:\n{}", selected.get().describe(),
                        CellposeGpu.preferredDevice(), CellposeGpu.describe(gpus));
            else
                logger.info("QuPath was started with CUDA_VISIBLE_DEVICES={}; keeping that allocation. Cards:\n{}",
                        inherited, CellposeGpu.describe(gpus));
        }
        return CellposeGpu.pinned(environment, selected);
    }

    private static void probeGpu() {
        if (gpuProbed)
            return;
        // Deliberately not the environment-build lock, which is held for the length of a build.
        synchronized (GPU_LOCK) {
            if (gpuProbed)
                return;
            nvidiaSmiPath = locateNvidiaSmi();
            gpus = nvidiaSmiPath == null ? List.of() : listGpus(nvidiaSmiPath);
            gpuProbed = true;
        }
    }

    private static String locateNvidiaSmi() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("mac") || os.contains("darwin"))
            return null;
        boolean windows = os.contains("win");

        if (detectCuda(List.of("nvidia-smi")))
            return "nvidia-smi";
        if (windows) {
            String system32 = "C:\\Windows\\System32\\nvidia-smi.exe";
            if (detectCuda(List.of(system32)))
                return system32;
            String resolved = whereWindows("nvidia-smi");
            if (resolved != null && detectCuda(List.of(resolved)))
                return resolved;
        }
        return null;
    }

    /** Run a candidate nvidia-smi command; true if it exits 0 or prints a recognizable driver line. */
    private static boolean detectCuda(List<String> command) {
        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            Process process = pb.start();
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (!process.waitFor(15, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                return false;
            }
            boolean driverLine = output.contains("Driver Version")
                    || output.contains("CUDA Version")
                    || output.contains("NVIDIA-SMI");
            return process.exitValue() == 0 || driverLine;
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException)
                Thread.currentThread().interrupt();
            return false;
        }
    }

    /** Resolve a command's path via {@code where} on Windows; null if not found. */
    private static String whereWindows(String command) {
        try {
            ProcessBuilder pb = new ProcessBuilder("where", command);
            pb.redirectErrorStream(true);
            Process process = pb.start();
            String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (!process.waitFor(15, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                return null;
            }
            if (process.exitValue() != 0)
                return null;
            for (String line : output.split("\\R")) {
                String trimmed = line.strip();
                if (!trimmed.isEmpty())
                    return trimmed;
            }
            return null;
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException)
                Thread.currentThread().interrupt();
            return null;
        }
    }

    /** Concatenate the full cause chain's messages so signature checks see nested causes. */
    static String collectCauseMessages(Throwable t) {
        StringBuilder sb = new StringBuilder();
        Throwable current = t;
        int guard = 0;
        while (current != null && guard++ < 50) {
            if (current.getMessage() != null)
                sb.append(current.getMessage()).append('\n');
            if (current.getCause() == current)
                break;
            current = current.getCause();
        }
        return sb.toString();
    }

    /** True if the message chain looks like a Windows conda-link file lock during (re)install. */
    static boolean looksLikeWindowsFileLock(String messages) {
        if (messages == null)
            return false;
        String m = messages.toLowerCase(Locale.ROOT);
        return m.contains("being used by another process")
                || m.contains("os error 32")
                || m.contains("failed to link");
    }

    /** True if the message chain looks like pixi failing to parse the staged lock's format. */
    static boolean looksLikeLockFormatSkew(String messages) {
        if (messages == null)
            return false;
        String m = messages.toLowerCase(Locale.ROOT);
        return m.contains("lock file version")
                || m.contains("lockfile version")
                || m.contains("failed to parse") && m.contains("lock")
                || m.contains("unexpected") && m.contains("lock")
                || m.contains("could not parse") && m.contains("lock");
    }

    /** Warn the user, once, that the first run downloads a large Python environment. */
    private static void warnFirstRun() {
        if (firstRunWarned)
            return;
        firstRunWarned = true;
        String message = "Building the Cellpose Python environment on first run. This downloads a "
                + "multi-GB Python/PyTorch/Cellpose environment and may take several minutes. "
                + "Progress is shown in Extensions > Cellpose > Python console.";
        logger.info(message);
        if (canPrompt()) {
            try {
                qupath.fx.dialogs.Dialogs.showInfoNotification("Cellpose", message);
            } catch (RuntimeException | LinkageError e) {
                logger.debug("Could not show first-run notification: {}", e.getMessage());
            }
        }
    }

    /** Surface manual recovery steps for a Windows file lock; does not auto-wipe the environment. */
    private static void surfaceWindowsFileLock(Path envDir) {
        Path pixiDir = envDir.resolve(".pixi");
        String message = "The Cellpose environment build was blocked because a file inside\n"
                + pixiDir + "\nis in use by another process. To recover:\n"
                + "1. Fully quit QuPath.\n"
                + "2. In Task Manager, end any stray python.exe or pixi.exe processes.\n"
                + "3. Delete the folder " + pixiDir + "\n"
                + "4. Relaunch QuPath and run Cellpose again.";
        logger.warn(message);
        if (canPrompt()) {
            try {
                qupath.fx.dialogs.Dialogs.showWarningNotification("Cellpose environment locked", message);
            } catch (RuntimeException | LinkageError e) {
                logger.debug("Could not show file-lock notification: {}", e.getMessage());
            }
        }
    }
}
