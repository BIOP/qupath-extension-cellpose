package qupath.ext.biop.cellpose;

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

import java.io.InputStream;
import java.util.LinkedHashMap;

/**
 * Install Cellpose as an extension.
 * <p>
 * Ibnstalls Cellpose into QuPath, adding some metadata and adds the necessary global variables to QuPath's Preferences
 *
 * @author Olivier Burri
 */
public class CellposeExtension implements QuPathExtension, GitHubProject {

    private static final Logger logger = LoggerFactory.getLogger(CellposeExtension.class);

    private boolean isInstalled = false;

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
                String script = new String(stream.readAllBytes(), "UTF-8");
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
        StringProperty omniposePath = PathPrefs.createPersistentPreference("omniposePythonPath", "");


        //Set options to current values
        options.setCellposePytonPath(cellposePath.get());
        options.setOmniposePytonPath(omniposePath.get());

        // Listen for property changes
        cellposePath.addListener((v, o, n) -> options.setCellposePytonPath(n));
        omniposePath.addListener((v, o, n) -> options.setOmniposePytonPath(n));

        PropertySheet.Item cellposePathItem = new PropertyItemBuilder<>(cellposePath, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.GENERAL)
                .name("Cellpose 'python.exe' location")
                .category("Cellpose/Omnipose")
                .description("Enter the full path to your cellpose environment, including 'python.exe'")
                .build();

        PropertySheet.Item omniposePathItem = new PropertyItemBuilder<>(omniposePath, String.class)
                .propertyType(PropertyItemBuilder.PropertyType.GENERAL)
                .name("Omnipose 'python.exe' location")
                .category("Cellpose/Omnipose")
                .description("Enter the full path to your omnipose environment, including 'python.exe'")
                .build();

        // Add Permanent Preferences and Populate Preferences
        QuPathGUI.getInstance().getPreferencePane().getPropertySheet().getItems().addAll(cellposePathItem, omniposePathItem);

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
}