package qupath.ext.biop.cellpose;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.StringProperty;
import javafx.collections.FXCollections;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.biop.cmd.VirtualEnvironmentRunner;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.panes.PreferencePane;
import qupath.lib.gui.prefs.PathPrefs;

import qupath.ext.biop.cmd.VirtualEnvironmentRunner.EnvType;
import qupath.ext.biop.cellpose.CellposeSetup.CellposeVersion;

/**
 * Install Cellpose as an extension.
 * <p>
 * Ibnstalls Cellpose into QuPath, adding some metadata and adds the necessary global variables to QuPath's Preferences
 *
 * @author Olivier Burri
 */
public class CellposeExtension implements QuPathExtension, GitHubProject {
    private final static Logger logger = LoggerFactory.getLogger(CellposeExtension.class);


    @Override
    public GitHubRepo getRepository() {
        return GitHubRepo.create("Cellpose 2D QuPath Extension", "biop", "qupath-extension-cellpose");
    }

    @Override
    public void installExtension(QuPathGUI qupath) {

        // Get a copy of the cellpose options
        CellposeSetup options = CellposeSetup.getInstance();

        // Create the options we need
        ObjectProperty<EnvType> envType = PathPrefs.createPersistentPreference("cellposeEnvType", EnvType.CONDA, EnvType.class);
        ObjectProperty<CellposeVersion> version = PathPrefs.createPersistentPreference("cellposeVersion",CellposeVersion.CELLPOSE_1, CellposeVersion.class);
        StringProperty envPath = PathPrefs.createPersistentPreference("cellposeEnvPath", "");

        //Set options to current values
        options.setEnvironmentType(envType.get());
        options.setVersion(version.get());
        options.setEnvironmentNameOrPath(envPath.get());

        // Listen for property changes
        envType.addListener((v,o,n) -> options.setEnvironmentType(n));
        version.addListener((v,o,n) -> options.setVersion(n));
        envPath.addListener((v,o,n) -> options.setEnvironmentNameOrPath(n));

        // Add Permanent Preferences and Populate Preferences
        PreferencePane prefs = QuPathGUI.getInstance().getPreferencePane();

        prefs.addPropertyPreference(envPath, String.class, "Cellpose Environment name or directory", "Cellpose",
                "Enter either the directory where your chosen Cellpose virtual environment (conda or venv) is located. Or the name of the conda environment you created.");
        prefs.addChoicePropertyPreference(version,
                FXCollections.observableArrayList(CellposeVersion.values()),
                CellposeVersion.class,"Cellpose Version", "Cellpose",
                "This will have an effect on the flags that can be sent to the Cellpose command.");
        prefs.addChoicePropertyPreference(envType,
                FXCollections.observableArrayList(VirtualEnvironmentRunner.EnvType.values()),
                VirtualEnvironmentRunner.EnvType.class,"Cellpose Environment Type", "Cellpose",
                "This changes how the environment is started.");
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
        return Version.parse("0.3.4");
    }
}