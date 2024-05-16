package qupath.ext.biop.cmd;

import org.controlsfx.tools.Platform;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

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

    private final String condaPath;

    private WatchService watchService;
    private String name;
    private String pythonPath;

    private List<String> arguments;

    private List<String> logResults;
    private Process process;

    /**
     * This enum helps us figure out the type of virtualenv. We need to change {@link #getActivationCommand()} as well.
     */
    public enum EnvType {
        CONDA("Anaconda or Miniconda", "If you need to start your virtual environment with 'conda activate' then this is the type for you"),
        VENV( "Python venv", "If you use 'myenv/Scripts/activate' to call your virtual environment, then use this environment type"),
        EXE("Python Executable", "Use this if you'd like to call the python executable directly. Can be useful in case you have issues with conda."),
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

    public VirtualEnvironmentRunner(String environmentNameOrPath, EnvType type, String name) {
        this(environmentNameOrPath, type, null, name );
    }

    public VirtualEnvironmentRunner(String environmentNameOrPath, EnvType type, String condaPath, String name) {
        this.pythonPath = environmentNameOrPath;
        this.envType = type;
        this.name = name;
        this.condaPath = condaPath;
        if (envType.equals(EnvType.OTHER))
            logger.error("Environment is unknown, please set the environment type to something different than 'Other'");
    }

    /**
     * This methods returns the command that will be needed by the {@link ProcessBuilder}, to start Python in the
     * desired virtual environment type.
     * Issue is that under windows you can just pile a bunch of Strings together, and it runs
     * In Mac or UNIX, the bash -c command must be followed by the full command enclosed in quotes
     * @return a list of Strings up to the start of the 'python' command. Use {@link #setArguments(List)} to set the actual command to run.
     */
    private List<String> getActivationCommand() {

        Platform platform = Platform.getCurrent();
        List<String> cmd = new ArrayList<>();
        String condaCommand = this.condaPath;

        switch (envType) {
            case CONDA:
                switch (platform) {
                    case WINDOWS:
                        if( condaCommand == null ) {
                            condaCommand = "conda.bat";
                        }
                        // Adjust path to the folder with the env name based on the python location. On Windows it's at the root of the environment
                        cmd.addAll(Arrays.asList("CALL", condaCommand, "activate", new File(pythonPath).getParent(), "&", "python"));
                        break;
                    case UNIX:
                    case OSX:
                        if( condaCommand == null ) {
                            condaCommand = "conda";
                        }
                        // Adjust path to the folder with the env name based on the python location. In Linux/MacOS it's in the 'bin' sub folder
                        cmd.addAll(Arrays.asList(condaCommand, "activate", new File(pythonPath).getParentFile().getParent(), ";", "python"));
                        break;
                }
                break;
            case VENV:
                switch (platform) {
                    case WINDOWS:
                        cmd.add(new File(pythonPath, "Scripts/python").getAbsolutePath());
                        break;
                    case UNIX:
                    case OSX:
                        cmd.add(new File(pythonPath, "bin/python").getAbsolutePath());
                        break;
                }
                break;
            case EXE:
                cmd.add(pythonPath);
                break;
            case OTHER:
                return null;
        }
        return cmd;
    }

    /**
     * This is the code you actually want to run after 'python'. For example adding {@code Arrays.asList("--version")}
     * should return the version of python that is being run.
     * @param arguments any cellpose or omnipose command line argument
     */
    public void setArguments(List<String> arguments) {
        this.arguments = arguments;
    }

    /**
     * This builds, runs the command and outputs it to the logger as it is being run
     * @param waitUntilDone whether to wait for the process to end or not before exiting this method
     * @throws IOException  in case there is an issue with the process
     */
    public void runCommand(boolean waitUntilDone) throws IOException {

        // Get how to start the command, based on the VENV Type
        List<String> command = getActivationCommand();

        // Get the arguments specific to the command we want to run
        command.addAll(arguments);

        // OK so here we need to either just continue appending the commands in the case of windows
        // or making a big string for NIX systems
        List<String> shell = new ArrayList<>();

        switch (Platform.getCurrent()) {

            case UNIX:
            case OSX:
                shell.addAll(Arrays.asList("bash", "-c"));

                // If there are spaces, then we should encapsulate the command with quotes
                command = command.stream().map(s -> {
                            if (s.trim().contains(" "))
                                return "\"" + s.trim() + "\"";
                            return s;
                        }).collect(Collectors.toList());

                // The last part needs to be sent as a single string, otherwise it does not run
                String cmdString = command.toString().replace(",","");

                shell.add(cmdString.substring(1, cmdString.length()-1));
                break;

            case WINDOWS:
            default:
                shell.addAll(Arrays.asList("cmd.exe", "/C"));
                shell.addAll(command);
                break;
        }


        // Try to make a command that is fully readable and that can be copy pasted
        List<String> printable = shell.stream().map(s -> {
            // add quotes if there are spaces in the paths
            if (s.contains(" "))
                return "\"" + s + "\"";
            else
                return s;
        }).collect(Collectors.toList());
        String executionString = printable.toString().replace(",", "");

        logger.info("Executing command:\n{}", executionString.substring(1, executionString.length()-1));
        logger.info("This command should run directly if copy-pasted into your shell");

        // Now the cmd line is ready
        ProcessBuilder pb = new ProcessBuilder(shell).redirectErrorStream(true);

        // Start the process and follow it throughout
        this.process = pb.start();

        // Keep the log of the process
        logResults = new ArrayList<>();

        Thread t = new Thread(Thread.currentThread().getName() + "-" + this.process.hashCode()) {
            @Override
            public void run() {
                BufferedReader stdIn = new BufferedReader(new InputStreamReader(process.getInputStream()));
                try {
                    for (String line = stdIn.readLine(); line != null; ) {
                        logger.info("{}: {}", name, line);
                        logResults.add(line);
                        line = stdIn.readLine();
                    }
                } catch (IOException e) {
                    logger.warn(e.getMessage());
                }
            }
        };
        t.setDaemon(true);
        t.start();


        logger.info("Virtual Environment Runner Started");

        // If we ask to wait, let's wait directly here rather than handle it outside
        if(waitUntilDone) {
            try {
                this.process.waitFor();
            } catch (InterruptedException e) {
                logger.error(e.getMessage());
            }
        }
    }

    public Process getProcess() {
        return this.process;
    }
    public List<String> getProcessLog() {
        return this.logResults;
    }
    public void startWatchService(Path folderToListen) throws IOException {
        this.watchService = FileSystems.getDefault().newWatchService();

        folderToListen.register(watchService, StandardWatchEventKinds.ENTRY_MODIFY);
    }

    public List<String> getChangedFiles() throws InterruptedException {
        WatchKey key = watchService.poll(100, TimeUnit.MICROSECONDS);
        if (key == null)
            return Collections.emptyList();
        List<WatchEvent<?>> events = key.pollEvents();
        List<String>files = events.stream()
                .map(e -> ((Path) e.context()).toString())
                .collect(Collectors.toList());
        key.reset();
        return files;
    }

    public void closeWatchService() throws IOException {
        if (watchService != null ) watchService.close();
    }
}