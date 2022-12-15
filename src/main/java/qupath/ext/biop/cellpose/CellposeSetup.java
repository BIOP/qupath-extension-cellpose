package qupath.ext.biop.cellpose;

import qupath.ext.biop.cmd.VirtualEnvironmentRunner.EnvType;

public class CellposeSetup {
    private EnvType envType;
    private String environmentNameOrPath;


    private static CellposeSetup instance = new CellposeSetup();

    public EnvType getEnvironmentType() {
        return envType;
    }

    public void setEnvironmentType(EnvType envType) {
        this.envType = envType;
    }

    public String getEnvironmentNameOrPath() {
        return environmentNameOrPath;
    }

    public void setEnvironmentNameOrPath(String environmentNameOrPath) {
        this.environmentNameOrPath = environmentNameOrPath;
    }

    public static CellposeSetup getInstance() {
        return instance;
    }
}
