package qupath.ext.biop.cellpose;

import qupath.fx.dialogs.Dialogs;

import java.io.File;

public class CellposeSetup {
    private static final CellposeSetup instance = new CellposeSetup();
    private String cellposePythonPath = null;
    private String omniposePythonPath = null;
    private String condaPath = null;

    public static CellposeSetup getInstance() {
        return instance;
    }

    public String getCellposePythonPath() {
        return cellposePythonPath;
    }

    public void setCellposePythonPath(String path) {
        checkPath( path );
        this.cellposePythonPath = path;
    }

    public String getOmniposePythonPath() {
        return omniposePythonPath;
    }

    public void setOmniposePythonPath(String path) {
        checkPath( path );
        this.omniposePythonPath = path;
    }

    public void setCondaPath(String condaPath) {
        checkPath( condaPath );
        this.condaPath = condaPath; }

    public String getCondaPath() { return condaPath; }

    private void checkPath(String path) {
        // It should be a file and it should exist
        if(!path.trim().isEmpty()) {
            File toCheck = new File(path);
            if (!toCheck.exists())
                Dialogs.showWarningNotification("Cellpose/Omnipose extension: Path not found", "The path to \"" + path + "\" does not exist or does not point to a valid file.");
        }
    }
}
