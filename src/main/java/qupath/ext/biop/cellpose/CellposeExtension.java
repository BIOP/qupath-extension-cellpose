package qupath.ext.biop.cellpose;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.StringProperty;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.panes.PreferencePane;
import qupath.lib.gui.prefs.PathPrefs;

/**
 * Install Cellpose as an extension.
 * <p>
 * Ibnstalls Cellpose into QuPath, adding some metadata and adds the necessary global variables to QuPath's Preferences
 *
 * @author Olivier Burri
 */
public class CellposeExtension implements QuPathExtension, GitHubProject {
    @Override
    public GitHubRepo getRepository() {
        return GitHubRepo.create("Cellpose 2D QuPath Extension", "biop", "qupath-extension-cellpose");
    }

    @Override
    public void installExtension(QuPathGUI qupath) {
        // Add Permanent Preferences and Populate Preferences
        PreferencePane prefs = QuPathGUI.getInstance().getPreferencePane();
        StringProperty envPath = PathPrefs.createPersistentPreference("cellposeEnvPath", "");
        StringProperty envType = PathPrefs.createPersistentPreference("cellposeEnvType", "conda");
        BooleanProperty useGPU = PathPrefs.createPersistentPreference("cellposeUseGPU", false);
        prefs.addDirectoryPropertyPreference(envPath, "Cellpose Environment Directory", "Cellpose",
                "Choose directory where your chosen Cellpose virtual environment (conda or venv) is located.");
        prefs.addPropertyPreference(envType, String.class, "Cellpose Environment Type", "Cellpose",
                "Write either 'conda' or 'venv' (without quotes). This changes how the environment is started.");
        prefs.addPropertyPreference(useGPU, Boolean.class,"Use GPU", "Cellpose",
                "Use the GPU when calling Cellpose");
    }

    @Override
    public String getName() {
        return "BIOP Cellpose extension";
    }

    @Override
    public String getDescription() {
        return "An extension for QuPath that allows running Cellpose by calling a Python virtual Environment";
    }

    @Override
    public Version getQuPathVersion() {
        return QuPathExtension.super.getQuPathVersion();
    }

    @Override
    public Version getVersion() {
        return Version.parse("0.1.0");
    }
}