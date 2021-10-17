package qupath.ext.biop.cellpose;

import qupath.ext.biop.cmd.VirtualEnvironmentRunner.EnvType;

public class CellposeOptions {
    private boolean useGPU;
    private EnvType envType;
    private String environmentNameorPath;

    private static CellposeOptions instance = new CellposeOptions();

    public EnvType getEnvironmentType() {
        return envType;
    }

    public void setEnvironmentType(EnvType envType) {
        this.envType = envType;
    }

    public String getEnvironmentNameorPath() {
        return environmentNameorPath;
    }

    public void setEnvironmentNameorPath(String environmentNameorPath) {
        this.environmentNameorPath = environmentNameorPath;
    }

    public boolean useGPU() {
        return useGPU;
    }

    public void useGPU(boolean useGPU) {
        this.useGPU = useGPU;
    }

    public static CellposeOptions getInstance() {
        return instance;
    }
}
