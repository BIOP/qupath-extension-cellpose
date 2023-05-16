package qupath.ext.biop.cellpose;

import qupath.ext.biop.cmd.VirtualEnvironmentRunner.EnvType;

public class CellposeSetup {
    private static final CellposeSetup instance = new CellposeSetup();
    private String cellposePythonPath = null;
    private String omniposePythonPath = null;

    public static CellposeSetup getInstance() {
        return instance;
    }

    public String getCellposePytonPath() {
        return cellposePythonPath;
    }

    public void setCellposePytonPath(String path) {
        this.cellposePythonPath = path;
    }

    public String getOmniposePytonPath() {
        return omniposePythonPath;
    }

    public void setOmniposePytonPath(String path) {
        this.omniposePythonPath = path;
    }
}
