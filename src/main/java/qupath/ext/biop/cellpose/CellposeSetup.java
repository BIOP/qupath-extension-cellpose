package qupath.ext.biop.cellpose;

import qupath.ext.biop.cmd.VirtualEnvironmentRunner.EnvType;

public class CellposeSetup {
    private EnvType envType;
    private String environmentNameOrPath;
    private CellposeVersion version;
    public enum CellposeVersion {
        CELLPOSE("Cellpose before v0.7.0"),
        OMNIPOSE("Omnipose after v0.7.2"),
        CELLPOSE_1("Cellpose Version 1.0");

        private final String description;

        CellposeVersion(String description) {
            this.description = description;
        }

        public String getDescription() {return this.description;}

        @Override
        public String toString() {
            return  getDescription();
        }
    }

    private static CellposeSetup instance = new CellposeSetup();

    public EnvType getEnvironmentType() {
        return envType;
    }

    public void setVersion( CellposeVersion version) { this.version = version; }
    public CellposeVersion getVersion() { return this.version; }

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
