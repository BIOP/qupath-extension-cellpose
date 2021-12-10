package qupath.ext.biop.cellpose;

import qupath.ext.biop.cmd.VirtualEnvironmentRunner.EnvType;

public class CellposeOptions {
    private boolean useGPU;
    private EnvType envType;
    private String environmentNameorPath;
    private CellposeVersion version;
    public enum CellposeVersion {
        CELLPOSE("Cellpose before v0.7.0"),
        OMNIPOSE("Omnipose after v0.7.2");

        private final String description;

        CellposeVersion(String description) {
            this.description = description;
        }

        public String getDescription() {return this.description;}

        @Override
        public String toString() {
            return this.description;
        }
    }

    private static CellposeOptions instance = new CellposeOptions();

    public EnvType getEnvironmentType() {
        return envType;
    }

    public void setVersion( CellposeVersion version) { this.version = version; }
    public CellposeVersion getVersion() { return this.version; }

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
