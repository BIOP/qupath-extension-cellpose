package qupath.ext.biop.cellpose;

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
        this.cellposePythonPath = path;
    }

    public String getOmniposePythonPath() {
        return omniposePythonPath;
    }

    public void setOmniposePythonPath(String path) {
        this.omniposePythonPath = path;
    }

    public void setCondaPath(String condaPath) { this.condaPath = condaPath; }

    public String getCondaPath() { return condaPath; }
}
