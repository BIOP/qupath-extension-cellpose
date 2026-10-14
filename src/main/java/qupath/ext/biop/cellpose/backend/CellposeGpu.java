package qupath.ext.biop.cellpose.backend;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.prefs.Preferences;
import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import org.apposed.appose.Builder;
import org.apposed.appose.BuildException;
import org.apposed.appose.Environment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Which NVIDIA GPU the in-process (Appose) Cellpose worker runs on, on machines with more than one.
 *
 * <p>The cards are listed with {@code nvidia-smi}. The one used is the card named by the
 * {@value #PREF_KEY} preference (an index or a {@code GPU-} UUID) or, on {@code auto}, the card
 * with the most free memory (in whole GiB, so a few MiB between idle cards is a tie), then the most
 * total memory, then the highest compute capability, then the lowest index. On a desktop with a
 * display card beside a compute card that is the compute card; on a shared server with identical
 * cards it is the least-loaded one rather than always card 0. Free memory is read once per session,
 * before the environment is built, so the CUDA build and the pinned card cannot disagree. The worker is pinned to it
 * with {@code CUDA_DEVICE_ORDER=PCI_BUS_ID} and {@code CUDA_VISIBLE_DEVICES} set to the card's
 * UUID, so inside Python it is {@code cuda:0} whatever its position, and the CUDA build is chosen
 * for that card's compute capability rather than for whichever card is listed first.
 *
 * <p>When QuPath was started with {@code CUDA_VISIBLE_DEVICES} already set, as cluster schedulers
 * do, {@code auto} keeps that allocation and the build is chosen for the first allocated card.
 *
 * <p>The preference lives in a Java preferences node that is not specific to this extension
 * ({@value #NODE}), so other Appose-based QuPath extensions that read the same key share one
 * setting. The system property {@value #PREF_KEY} overrides it for a single run.
 */
public final class CellposeGpu {

    private static final Logger logger = LoggerFactory.getLogger(CellposeGpu.class);

    /** Preference key and system property: {@code auto}, an index, or a {@code GPU-} UUID. */
    public static final String PREF_KEY = "appose.gpuDevice";

    /** Preferences node shared with other Appose-based QuPath extensions. */
    public static final String NODE = "/io/github/apposed/qupath";

    /** The automatic choice. */
    public static final String AUTO = "auto";

    /** One NVIDIA card as nvidia-smi reports it. */
    public record Gpu(int index, String uuid, String name, String computeCapability, int memoryMiB, int freeMiB) {

        int capabilityValue() {
            if (computeCapability == null || computeCapability.isBlank())
                return -1;
            try {
                String[] p = computeCapability.strip().split("\\.");
                return Integer.parseInt(p[0]) * 10 + (p.length > 1 ? Integer.parseInt(p[1]) : 0);
            } catch (RuntimeException e) {
                return -1;
            }
        }

        /** E.g. {@code 1: NVIDIA GeForce RTX 3090 (compute 8.6, 24576 MiB)}. */
        public String describe() {
            return index + ": " + name + " (compute "
                    + (computeCapability == null || computeCapability.isBlank() ? "unknown" : computeCapability)
                    + ", " + memoryMiB + " MiB" + (freeMiB >= 0 ? ", " + freeMiB + " MiB free" : "") + ")";
        }

        /** Free memory in whole GiB for ranking; total memory when free is unknown. */
        int freeGiB() {
            return (freeMiB >= 0 ? freeMiB : memoryMiB) / 1024;
        }
    }

    private static final Object LOCK = new Object();
    private static StringProperty preferenceProperty;

    private CellposeGpu() {}

    /**
     * Parses {@code nvidia-smi --query-gpu=index,uuid,name[,compute_cap],memory.total,memory.free
     * --format=csv,noheader,nounits}.
     */
    static List<Gpu> parse(String csv, boolean withCapability) {
        List<Gpu> gpus = new ArrayList<>();
        if (csv == null)
            return gpus;
        for (String line : csv.split("\\R")) {
            String t = line.strip();
            if (t.isEmpty() || t.startsWith("["))
                continue;
            String[] f = t.split("\\s*,\\s*");
            int m = withCapability ? 4 : 3;
            if (f.length < m + 2)
                continue;
            try {
                // memory reads [N/A] on some cards; that leaves the card listed, ranked by what is known
                gpus.add(new Gpu(Integer.parseInt(f[0].strip()), f[1].strip(), f[2].strip(),
                        withCapability ? f[3].strip() : "", mebibytes(f[m]), mebibytes(f[m + 1])));
            } catch (NumberFormatException e) {
                // not a line describing a card
            }
        }
        return gpus;
    }

    private static int mebibytes(String field) {
        try {
            return (int) Math.round(Double.parseDouble(field.strip()));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    /** The preference: system property, else the shared node, else {@code auto}. */
    public static String preferredDevice() {
        String sys = System.getProperty(PREF_KEY);
        if (sys != null && !sys.isBlank())
            return sys.strip();
        try {
            String v = node().get(PREF_KEY, AUTO);
            return v == null || v.isBlank() ? AUTO : v.strip();
        } catch (RuntimeException e) {
            return AUTO;
        }
    }

    /** Sets the shared preference. */
    public static void setPreferredDevice(String value) {
        node().put(PREF_KEY, value == null || value.isBlank() ? AUTO : value.strip());
    }

    private static Preferences node() {
        return Preferences.userRoot().node(NODE);
    }

    /** The preference as a property for the preference pane, kept in step with the shared node. */
    public static StringProperty preferenceProperty() {
        synchronized (LOCK) {
            if (preferenceProperty == null) {
                StringProperty p = new SimpleStringProperty(preferredDevice());
                p.addListener((obs, old, now) -> {
                    String value = now == null || now.isBlank() ? AUTO : now.strip();
                    if (!value.equals(node().get(PREF_KEY, AUTO)))
                        setPreferredDevice(value);
                });
                try {
                    node().addPreferenceChangeListener(evt -> {
                        if (!PREF_KEY.equals(evt.getKey()))
                            return;
                        String v = evt.getNewValue() == null ? AUTO : evt.getNewValue();
                        Runnable update = () -> {
                            if (!v.equals(p.get()))
                                p.set(v);
                        };
                        try {
                            if (Platform.isFxApplicationThread())
                                update.run();
                            else
                                Platform.runLater(update);
                        } catch (IllegalStateException e) {
                            update.run();
                        }
                    });
                } catch (RuntimeException e) {
                    // the pane may show a stale value until restart
                }
                preferenceProperty = p;
            }
            return preferenceProperty;
        }
    }

    /** Values the preference pane offers; fixed, so building the pane never waits on nvidia-smi. */
    public static List<String> preferenceChoices() {
        return List.of(AUTO, "0", "1", "2", "3", "4", "5", "6", "7");
    }

    /**
     * The card to pin the worker to, or empty to pin nothing.
     *
     * @param inheritedMask {@code CUDA_VISIBLE_DEVICES} as QuPath was started with, or null
     */
    static Optional<Gpu> select(List<Gpu> gpus, String preference, String inheritedMask) {
        if (gpus.isEmpty())
            return Optional.empty();
        String pref = preference == null || preference.isBlank() ? AUTO : preference.strip();
        if (!AUTO.equalsIgnoreCase(pref)) {
            for (Gpu g : gpus) {
                if (pref.equalsIgnoreCase(g.uuid()) || pref.equals(String.valueOf(g.index())))
                    return Optional.of(g);
            }
            logger.warn("The GPU preference '{}' matches none of the cards present; choosing automatically", pref);
        }
        if (inheritedMask != null && !inheritedMask.isBlank())
            return Optional.empty();
        return gpus.stream()
                .sorted(Comparator.comparingInt(Gpu::freeGiB).reversed()
                        .thenComparing(Comparator.comparingInt(Gpu::memoryMiB).reversed())
                        .thenComparing(Comparator.comparingInt(Gpu::capabilityValue).reversed())
                        .thenComparingInt(Gpu::index))
                .findFirst();
    }

    /**
     * The card the worker will actually run on: the selected one, else the first card of an
     * inherited {@code CUDA_VISIBLE_DEVICES} allocation, else the first card listed.
     */
    static Optional<Gpu> effective(List<Gpu> gpus, String preference, String inheritedMask) {
        Optional<Gpu> selected = select(gpus, preference, inheritedMask);
        if (selected.isPresent() || gpus.isEmpty())
            return selected;
        if (inheritedMask != null && !inheritedMask.isBlank()) {
            String first = inheritedMask.split(",")[0].strip();
            for (Gpu g : gpus) {
                if (first.equalsIgnoreCase(g.uuid()) || first.equals(String.valueOf(g.index())))
                    return Optional.of(g);
            }
        }
        return Optional.of(gpus.get(0));
    }

    /** The variables that pin a worker to {@code gpu}. */
    static Map<String, String> pinVariables(Gpu gpu) {
        Map<String, String> vars = new LinkedHashMap<>();
        vars.put("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        vars.put("CUDA_VISIBLE_DEVICES",
                gpu.uuid() == null || gpu.uuid().isBlank() ? String.valueOf(gpu.index()) : gpu.uuid());
        return vars;
    }

    /** {@code env} with its workers pinned to {@code gpu}, or {@code env} itself when there is none. */
    static Environment pinned(Environment env, Optional<Gpu> gpu) {
        return env == null || gpu.isEmpty() ? env : new PinnedEnvironment(env, pinVariables(gpu.get()));
    }

    /** One line per card, for the log. */
    static String describe(List<Gpu> gpus) {
        StringBuilder sb = new StringBuilder();
        for (Gpu g : gpus)
            sb.append("  ").append(g.describe()).append('\n');
        return sb.toString().stripTrailing();
    }

    /** An environment that adds variables to whatever the wrapped one passes its workers. */
    static final class PinnedEnvironment implements Environment {
        private final Environment delegate;
        private final Map<String, String> extra;

        PinnedEnvironment(Environment delegate, Map<String, String> extra) {
            this.delegate = delegate;
            this.extra = extra;
        }

        @Override
        public String base() {
            return delegate.base();
        }

        @Override
        public List<String> binPaths() {
            return delegate.binPaths();
        }

        @Override
        public List<String> launchArgs() {
            return delegate.launchArgs();
        }

        @Override
        public Map<String, String> envVars() {
            Map<String, String> merged = new LinkedHashMap<>(delegate.envVars());
            merged.putAll(extra);
            return merged;
        }

        @Override
        public Builder<?> builder() {
            return delegate.builder();
        }

        @Override
        public String type() {
            return delegate.type();
        }

        @Override
        public Environment activate(String name) throws BuildException {
            return new PinnedEnvironment(delegate.activate(name), extra);
        }

        @Override
        public Environment rebuild() throws BuildException {
            return new PinnedEnvironment(delegate.rebuild(), extra);
        }

        @Override
        public Environment delete() throws BuildException {
            return delegate.delete();
        }
    }
}
