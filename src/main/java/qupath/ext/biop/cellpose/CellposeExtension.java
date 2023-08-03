package qupath.ext.biop.cellpose;

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

        // Add Permanent Preferences and Populate Preferences
        PreferencePane prefs = QuPathGUI.getInstance().getPreferencePane();

        prefs.addPropertyPreference(cellposePath, String.class, "Cellpose 'python.exe' location", "Cellpose/Omnipose",
                "Enter the full path to your cellpose environment, including 'python.exe'");
        prefs.addPropertyPreference(omniposePath, String.class, "Omnipose 'python.exe location (Optional)", "Cellpose/Omnipose",
                "Enter the full path to your omnipose environment, including 'python.exe'");
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

}