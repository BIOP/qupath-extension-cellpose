package qupath.ext.biop.cmd;

import org.controlsfx.tools.Platform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A wrapper to run python virtualenvs, that tries to figure out the commands to run based on the environment type
 *
 * @author Olivier Burri
 * @author Romain Guiet
 * @author Nicolas Chiaruttini
 */
public class VirtualEnvironmentRunner {
    private final static Logger logger = LoggerFactory.getLogger(VirtualEnvironmentRunner.class);
    private final EnvType envType;

    private String environmentNameOrPath;

    private List<String> arguments;

    /**
     * This enum helps us figure out the type of virtualenv. We need to change {@link #getStartCommand()} as well.
     */
    public enum EnvType {
        CONDA("Anaconda or Miniconda", "If you need to start your virtual environment with 'conda activate' then this is the type for you"),
        VENV( "Python venv", "If you use 'myenv/Scripts/activate' to call your virtual environment, then use this environment type"),
        OTHER("Other (Unsupported)", "Currently only conda and venv are supported.");

        private final String description;
        private final String help;

        EnvType(String description, String help) {
            this.description = description;
            this.help = help;

        }

        public String getDescription() {return this.description;}

        @Override
        public String toString() {
            return this.description;
        }
    }

    public VirtualEnvironmentRunner(String environmentNameOrPath, EnvType type) {
        this.environmentNameOrPath = environmentNameOrPath;
        this.envType = type;
    }

    /**
     * This methods returns the command that will be needed by the {@link ProcessBuilder}, to start Python in the
     * desired virtual environment type
     * @return a list of Strings up to the start of the 'python' command. Use {@link #setArguments(List)} to set the actual command to run.
     */
    private List<String> getStartCommand() {

        Platform platform = Platform.getCurrent();
        List<String> cmd = new ArrayList<>();

        switch (envType) {
            case CONDA:
                switch (platform) {
                    case WINDOWS:
                        cmd.addAll(Arrays.asList("cmd.exe", "/C", "conda", "activate", environmentNameOrPath, "&", "python"));
                        break;
                    case UNIX:
                    case OSX:
                        // https://docs.conda.io/projects/conda/en/4.6.1/user-guide/tasks/manage-environments.html#id2
                        cmd.addAll(Arrays.asList("bash", "-c", "conda", "activate", environmentNameOrPath, "&", "python"));
                        break;
                }
                break;

            case VENV:
                switch (platform) {
                    case WINDOWS:
                        cmd.addAll(Arrays.asList("cmd.exe", "/C", new File(environmentNameOrPath, "Scripts/python.exe").getAbsolutePath()));
                        break;
                    case UNIX:
                    case OSX:
                        cmd.addAll(Arrays.asList("bash", "-c", "conda", new File(environmentNameOrPath, "bin/python").getAbsolutePath()));
                        break;
                }
                break;
            case OTHER:
                logger.error("Environment is unknown, please set the environment type to something different than 'Other'");
                return null;
        }
        // Because Arrays.asList returns an unmodifiable list, we change it here
        return cmd;
    }

    /**
     * This is the code you actually want to run after 'python'. For example adding {@code Arrays.asList("--version")}
     * should return the version of python that is being run.
     * @param arguments
     */
    public void setArguments(List<String> arguments) {
        this.arguments = arguments;
    }

    /**
     * This builds, runs the command and outputs it to the logger as it is being run
     * @throws IOException // In case there is an issue starting the process
     * @throws InterruptedException // In case there is an issue after the process is started
     */
    public void runCommand() throws IOException, InterruptedException {
        // Get how to start the command, based on the VENV Type
        List<String> cmd = getStartCommand();

        // Get the arguments specific to the command we want to run
        cmd.addAll(arguments);

        logger.info("Executing command: {}", cmd.toString().replace(",", ""));

        // Now the cmd line is ready
        ProcessBuilder pb = new ProcessBuilder(cmd);

        // Start the process and follow it throughout
        Process p = pb.start();

        Thread t = new Thread(Thread.currentThread().getName() + "-" + p.hashCode()) {
            @Override
            public void run() {
                BufferedReader stdIn = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String console = "cellpose";
                try {
                    for (String line = stdIn.readLine(); line != null; ) {
                        logger.info("{}: {}", console, line);
                        line = stdIn.readLine();
                    }
                } catch (IOException e) {
                    logger.warn(e.getMessage());
                }
            }
        };
        t.setDaemon(true);
        t.start();

        p.waitFor();

        logger.info("Virtual Enviroment Runner Finished");
    }
}
