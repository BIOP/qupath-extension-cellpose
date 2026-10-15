package qupath.ext.biop.cellpose;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.StringProperty;
import org.controlsfx.control.PropertySheet;
import org.controlsfx.control.action.Action;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.fx.prefs.controlsfx.PropertyItemBuilder;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.gui.tools.MenuTools;
import qupath.ext.biop.cellpose.backend.CellposeDevice;
import qupath.ext.biop.cellpose.backend.CellposeTransport;
import qupath.ext.biop.cellpose.backend.ApposeBackend;
import qupath.ext.biop.cellpose.backend.CellposeGpu;
import qupath.ext.biop.cellpose.ui.PythonConsoleWindow;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.jar.Attributes;
import java.util.jar.Manifest;
import java.util.stream.Collectors;

/**
 * Install Cellpose as an extension.
 * <p>
 * Installs Cellpose into QuPath, adding some metadata and adds the necessary global variables to QuPath's Preferences
 *
 * @author Olivier Burri
 */
public class CellposeExtension implements QuPathExtension, GitHubProject {

    private static final Logger logger = LoggerFactory.getLogger(CellposeExtension.class);
    private boolean isInstalled = false;

    private static final ObjectProperty<CellposeTransport> cellposeTransport =
            PathPrefs.createPersistentPreference("cellposeTransport", CellposeTransport.SUBPROCESS, CellposeTransport.class);

    /**
     * @return the currently selected detection transport preference (defaults to
     * {@link CellposeTransport#SUBPROCESS}). A builder-level transport, when set, overrides this.
     */
    public static CellposeTransport getTransportPreference() {
        return cellposeTransport.get();
    }

    /**
     * Directory containing the in-process backend's Python environment, or empty for the Appose
     * default ({@code ~/.local/share/appose}).
     */
    private static final StringProperty cellposeApposeEnvDir =
            PathPrefs.createPersistentPreference("cellposeApposeEnvDir", "");

    /**
     * @return the directory chosen to hold the Appose environment, or an empty string to use the
     * Appose default location
     */
    public static String getApposeEnvDirPreference() {
        String value = cellposeApposeEnvDir.get();
        return value == null ? "" : value.strip();
    }

    /**
     * Record where the Appose environment should live.
     *
     * @param dir the containing directory, or empty/null for the Appose default
     */
    public static void setApposeEnvDirPreference(String dir) {
        cellposeApposeEnvDir.set(dir == null ? "" : dir.strip());
    }

    private static final ObjectProperty<CellposeDevice> cellposeApposeDevice =
            PathPrefs.createPersistentPreference("cellposeApposeDevice", CellposeDevice.AUTO, CellposeDevice.class);

    /**
     * @return the currently selected Appose compute-device preference (defaults to
     * {@link CellposeDevice#AUTO}). A builder-level device, when set, overrides this.
     */
    public static CellposeDevice getDevicePreference() {
        return cellposeApposeDevice.get();
    }

    /**
     * Set the compute device used by the in-process (Appose) transport.
     *
     * @param device the device to use; null is treated as {@link CellposeDevice#AUTO}
     */
    public static void setDevicePreference(CellposeDevice device) {
        cellposeApposeDevice.set(device == null ? CellposeDevice.AUTO : device);
    }

    /** Start the Appose Python worker with QuPath rather than on the first detection. */
    private static final BooleanProperty cellposeApposeStartWithQuPath =
            PathPrefs.createPersistentPreference("cellposeApposeStartWithQuPath", false);

    /** Whether the last Appose detection used Cellpose-SAM, so start-up starts that worker. */
    private static final BooleanProperty cellposeApposeLastWasSam =
            PathPrefs.createPersistentPreference("cellposeApposeLastWasSam", false);

    /**
     * Record which model family the last Appose detection used.
     *
     * @param cellposeSam true for Cellpose-SAM (cp4), false for Cellpose 3
     */
    public static void setLastApposeFamilyWasSam(boolean cellposeSam) {
        if (cellposeApposeLastWasSam.get() != cellposeSam)
            cellposeApposeLastWasSam.set(cellposeSam);
    }

    private static final AtomicBoolean workerStarting = new AtomicBoolean(false);

    /**
     * Start the Appose Python worker in the background, if the transport is Appose and the
     * preference asks for it. At most one start at a time.
     */
    private static void startWorkerIfWanted() {
        if (cellposeTransport.get() != CellposeTransport.APPOSE || !cellposeApposeStartWithQuPath.get())
            return;
        if (!workerStarting.compareAndSet(false, true))
            return;
        Thread t = new Thread(() -> {
            try {
                ApposeBackend.prewarm(cellposeApposeLastWasSam.get(), getDevicePreference());
            } finally {
                workerStarting.set(false);
            }
        }, "Cellpose-worker-start");
        t.setDaemon(true);
        t.start();
    }

    private static final LinkedHashMap<String, String> SCRIPTS = new LinkedHashMap<>() {{
        put("Cellpose training script template", "scripts/Cellpose_training_template.groovy");
        put("Cellpose detection script template", "scripts/Cellpose_detection_template.groovy");
        put("Detect nuclei and cells using Cellpose.groovy", "scripts/Detect_nuclei_and_cells_using_Cellpose.groovy");
        put("Create Cellpose training and validation images", "scripts/Create_Cellpose_training_and_validation_images.groovy");
    }};

    @Override
    public GitHubRepo getRepository() {
        return GitHubRepo.create("Cellpose 2D QuPath Extension", "biop", "qupath-extension-cellpose");
    }

    @Override
    public void installExtension(QuPathGUI qupath) {
        if (isInstalled)
            return;

        SCRIPTS.entrySet().forEach(entry -> {
            String name = entry.getValue();
            String command = entry.getKey();
            try (InputStream stream = CellposeExtension.class.getClassLoader().getResourceAsStream(name)) {
                String script = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
                if (script != null) {
                    MenuTools.addMenuItems(
                            qupath.getMenu("Extensions>Cellpose", true),
                            new Action(command, e -> openScript(qupath, script)));
                }
            } catch (Exception e) {
                logger.error(e.getLocalizedMessage(), e);
            }
        });
        // Get a copy of the cellpose options
        CellposeSetup options = CellposeSetup.getInstance();

        // Create the options we need
        StringProperty cellposePath = PathPrefs.createPersistentPreference("cellposePythonPath", "");
        StringProperty cellposeSAMPath = PathPrefs.createPersistentPreference("cellposeSAMPythonPath", "");
        StringProperty omniposePath = PathPrefs.createPersistentPreference("omniposePythonPath", "");
        StringProperty condaPath = PathPrefs.createPersistentPreference("condaPath", "");

        //Set options to current values
        options.setCellposePythonPath(cellposePath.get());
        options.setCellposeSAMPythonPath(cellposeSAMPath.get());
        options.setOmniposePythonPath(omniposePath.get());
        options.setCondaPath(condaPath.get());

        // Listen for property changes
        cellposePath.addListener((v, o, n) -> options.setCellposePythonPath(n));
        cellposeSAMPath.addListener((v, o, n) -> options.setCellposeSAMPythonPath(n));
        omniposePath.addListener((v, o, n) -> options.setOmniposePythonPath(n));
        condaPath.addListener((v, o, n) -> options.setCondaPath(n));

        PropertySheet.Item cellposePathItem = new PropertyItemBuilder<>(cellposePath, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.GENERAL)
                .name("Cellpose 'python.exe' location")
                .category("Cellpose/Omnipose")
                .description("Enter the full path to your cellpose environment, including 'python.exe'\nDo not include quotes (\') or double quotes (\") around the path.")
                .build();

        PropertySheet.Item cellposeSAMPathItem = new PropertyItemBuilder<>(cellposeSAMPath, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.GENERAL)
                .name("Cellpose SAM 'python.exe' location")
                .category("Cellpose/Omnipose")
                .description("Enter the full path to your cellposeSAM environment, including 'python.exe'\nDo not include quotes (\') or double quotes (\") around the path.")
                .build();

        PropertySheet.Item omniposePathItem = new PropertyItemBuilder<>(omniposePath, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.GENERAL)
                .name("Omnipose 'python.exe' location")
                .category("Cellpose/Omnipose")
                .description("Enter the full path to your omnipose environment, including 'python.exe'\nDo not include quotes (\') or double quotes (\") around the path.")
                .build();

        PropertySheet.Item condaPathItem = new PropertyItemBuilder<>(condaPath, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.GENERAL)
                .name("'Conda/Mamba' script location (optional)")
                .category("Cellpose/Omnipose")
                .description("The full path to you conda/mamba command, in case you want the extension to use the 'conda activate' command.\ne.g 'C:\\ProgramData\\Miniconda3\\condabin\\mamba.bat'\nDo not include quotes (\') or double quotes (\") around the path.")
                .build();

        PropertySheet.Item transportItem = new PropertyItemBuilder<>(cellposeTransport, CellposeTransport.class)
                .propertyType(PropertyItemBuilder.PropertyType.GENERAL)
                .name("Cellpose processing type")
                .category("Cellpose/Omnipose")
                .description("How Cellpose runs when detecting objects.\n\n"
                        + "User's Python installation - standard (default): runs Cellpose from a Python "
                        + "installation you set up yourself, using the paths above, writing each "
                        + "image tile to a temporary file.\n\n"
                        + "Appose - runs faster, self-setup: runs Cellpose inside QuPath and "
                        + "builds its own Python environment the first time it is used, which is a "
                        + "multi-GB download taking several minutes. After that, tiles stay in memory "
                        + "and the Python process is reused between runs.\n\n"
                        + "Training and the QC notebook always use the Python installation, "
                        + "whichever is chosen here.")
                .build();

        PropertySheet.Item apposeEnvDirItem = new PropertyItemBuilder<>(cellposeApposeEnvDir, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.DIRECTORY)
                .name("Cellpose Appose environment directory")
                .category("Cellpose/Omnipose")
                .description("Where the in-process (Appose) transport keeps its Python environment.\n"
                        + "Leave empty for the default location (~/.local/share/appose).\n"
                        + "The environment is several GB, so on a shared workstation it is worth putting it "
                        + "somewhere other than the system drive.\n"
                        + "Takes effect on the next detection: any running Python worker is stopped "
                        + "and the environment is built at the new location.\n"
                        + "A new location means a new environment. The old one is left where it is, so "
                        + "delete it yourself to reclaim the space.\n"
                        + "Avoid paths containing spaces -- pixi cannot build in them.")
                .build();

        PropertySheet.Item deviceItem = new PropertyItemBuilder<>(cellposeApposeDevice, CellposeDevice.class)
                .propertyType(PropertyItemBuilder.PropertyType.GENERAL)
                .name("Cellpose Appose device")
                .category("Cellpose/Omnipose")
                .description("Compute device for the in-process (Appose) transport:\nAUTO (default): detect an NVIDIA GPU and use it if present, otherwise CPU.\nGPU: force the CUDA environment.\nCPU: force the CPU environment.")
                .build();

        PropertySheet.Item gpuItem = new PropertyItemBuilder<>(CellposeGpu.preferenceProperty(), String.class)
                .propertyType(PropertyItemBuilder.PropertyType.CHOICE)
                .choices(CellposeGpu.preferenceChoices())
                .name("Cellpose Appose GPU (multi-GPU machines)")
                .category("Cellpose/Omnipose")
                .description("Which NVIDIA GPU the in-process (Appose) transport uses when there is more than one.\n"
                        + "auto (default): the card with the most free memory when Cellpose first runs; on a desktop "
                        + "with a display card and a compute card that is the compute card, on a shared server the "
                        + "least-loaded card. A number is the index nvidia-smi shows.\n"
                        + "The CUDA build is chosen for that card. The cards found and the one used are written "
                        + "to the log. When QuPath is started by a cluster scheduler that sets "
                        + "CUDA_VISIBLE_DEVICES, auto keeps that allocation. Takes effect when the Python worker "
                        + "next starts (Extensions > Cellpose > Shut down Python worker).")
                .build();

        PropertySheet.Item startWithQuPathItem = new PropertyItemBuilder<>(cellposeApposeStartWithQuPath, Boolean.class)
                .propertyType(PropertyItemBuilder.PropertyType.GENERAL)
                .name("Cellpose Appose: start the Python worker with QuPath")
                .category("Cellpose/Omnipose")
                .description("Start the Appose Python worker in the background when QuPath starts, instead of on "
                        + "the first detection.\n"
                        + "Off (default): the first detection in a session waits about 5-8 s while Python starts, "
                        + "imports Cellpose and PyTorch and prepares the GPU; later detections reuse the worker.\n"
                        + "On: that wait happens at start-up instead, so the first detection is as fast as the "
                        + "rest. The worker then holds some GPU memory from the start of the session (Extensions "
                        + "> Cellpose > Shut down Python worker releases it).\n"
                        + "It starts the worker for the model family you used last (Cellpose 3 or Cellpose-SAM), "
                        + "and never installs anything: until the environment has been built by a first "
                        + "detection, it does nothing. Turning it on starts the worker straight away.")
                .build();

        // Add Permanent Preferences and Populate Preferences
        var items = QuPathGUI.getInstance().getPreferencePane().getPropertySheet().getItems();
        items.addAll(cellposePathItem, cellposeSAMPathItem, omniposePathItem, condaPathItem, transportItem, deviceItem, gpuItem, apposeEnvDirItem);

        // The start-up option means nothing for the subprocess transport, so it is only listed
        // while Appose is selected, directly under the processing type.
        Runnable showStartOption = () -> {
            boolean wanted = cellposeTransport.get() == CellposeTransport.APPOSE;
            if (wanted && !items.contains(startWithQuPathItem))
                items.add(Math.min(items.indexOf(transportItem) + 1, items.size()), startWithQuPathItem);
            else if (!wanted)
                items.remove(startWithQuPathItem);
        };
        showStartOption.run();
        cellposeTransport.addListener((obs, old, now) -> {
            showStartOption.run();
            startWorkerIfWanted();
        });
        cellposeApposeStartWithQuPath.addListener((obs, old, now) -> {
            if (Boolean.TRUE.equals(now))
                startWorkerIfWanted();
        });
        startWorkerIfWanted();

        // The in-process backend keeps its Python worker alive between runs; shutting it down is
        // how a user reclaims the GPU memory it holds without restarting QuPath.
        MenuTools.addMenuItems(
                qupath.getMenu("Extensions>Cellpose", true),
                new Action("Python console", e -> PythonConsoleWindow.show()),
                new Action("Shut down Python worker", e -> ApposeBackend.shutdownWorkers()));

    }

    @Override
    public String getName() {
        return "BIOP Cellpose extension";
    }

    @Override
    public String getDescription() {
        return "An extension that allows running a Cellpose/Omnipose Virtual Environment within QuPath";
    }

    @Override
    public Version getQuPathVersion() {
        return QuPathExtension.super.getQuPathVersion();
    }

    private static void openScript(QuPathGUI qupath, String script) {
        var editor = qupath.getScriptEditor();
        if (editor == null) {
            logger.error("No script editor is available!");
            return;
        }
        qupath.getScriptEditor().showScript("Cellpose detection", script);
    }

    protected static String getExtensionVersion(){
        String versionString = null;
        try {
            List<URL> manifestList = Collections.list(CellposeExtension.class.getClassLoader().getResources("META-INF/MANIFEST.MF"))
                    .parallelStream()
                    .filter(e -> e.toString().contains("qupath-extension-cellpose") && !e.toString().contains("javadoc"))
                    .sorted(Comparator.comparing(URL::toString))
                    .collect(Collectors.toList());
            Collections.reverse(manifestList);

        for (URL url : manifestList) {
                if (url == null)
                    continue;
                try (InputStream ignored = url.openStream()) {
                    Manifest manifest = new Manifest(url.openStream());
                    Attributes attributes = manifest.getMainAttributes();
                    versionString = attributes.getValue("Implementation-Version");
                    break;
                } catch (IOException e) {
                    logger.error("Error reading manifest", e);
                } catch (IllegalArgumentException e) {
                    logger.error("Error determining version: {}", e.getLocalizedMessage(), e);
                }
            }
        } catch (Exception e) {
            logger.error("Error searching for build string", e);
        }
        return versionString;
    }
}