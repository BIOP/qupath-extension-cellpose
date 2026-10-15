package qupath.ext.biop.cellpose.backend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.apposed.appose.Builder;
import org.apposed.appose.Environment;
import org.junit.jupiter.api.Test;

class CellposeGpuTest {

    private static final String DISPLAY_AND_COMPUTE = String.join("\n",
            "0, GPU-aaaa, NVIDIA T400, 7.5, 2048, 1500",
            "1, GPU-bbbb, NVIDIA GeForce RTX 5090, 12.0, 32607, 31800");

    @Test
    void autoPicksTheComputeCardAndItsBuild() {
        var gpus = CellposeGpu.parse(DISPLAY_AND_COMPUTE, true);
        var chosen = CellposeGpu.select(gpus, "auto", null).orElseThrow();
        assertEquals("GPU-bbbb", chosen.uuid());
        // the build follows the card the worker runs on, not the first card listed
        assertEquals("cu130", ApposeEnvironments.cudaBuildFor(chosen.computeCapability()));
        assertEquals("cu126", ApposeEnvironments.cudaBuildFor(gpus.get(0).computeCapability()));
    }

    @Test
    void explicitChoiceAndSchedulerAllocation() {
        var gpus = CellposeGpu.parse(DISPLAY_AND_COMPUTE, true);
        assertEquals("GPU-aaaa", CellposeGpu.select(gpus, "0", "1").orElseThrow().uuid());
        assertTrue(CellposeGpu.select(gpus, "auto", "1").isEmpty());
        // with a scheduler mask the build is chosen for the first allocated card
        assertEquals("GPU-bbbb", CellposeGpu.effective(gpus, "auto", "1").orElseThrow().uuid());
        assertEquals("GPU-aaaa", CellposeGpu.effective(gpus, "auto", "GPU-aaaa,GPU-bbbb").orElseThrow().uuid());
        assertTrue(CellposeGpu.effective(List.of(), "auto", null).isEmpty());
    }

    @Test
    void oldDriversWithoutComputeCapabilityStillParse() {
        var gpus = CellposeGpu.parse("0, GPU-c, Quadro K2200, 4096, [N/A]\n[N/A]\n", false);
        assertEquals(1, gpus.size());
        assertEquals("", gpus.get(0).computeCapability());
        assertEquals(-1, gpus.get(0).freeMiB());
        assertTrue(CellposeGpu.parse(null, true).isEmpty());
    }

    @Test
    void onASharedServerAutoPicksTheLeastLoadedCard() {
        StringBuilder csv = new StringBuilder();
        for (int i = 0; i < 8; i++)
            csv.append(i).append(", GPU-").append(i).append(", NVIDIA A100, 8.0, 81920, ")
                    .append(i == 5 ? 79000 : 9000).append('\n');
        assertEquals("GPU-5", CellposeGpu.select(CellposeGpu.parse(csv.toString(), true), "auto", null)
                .orElseThrow().uuid());
        // idle identical cards differing by a few MiB: lowest index, stable from run to run
        var idle = CellposeGpu.parse("0, GPU-0, A, 8.0, 81920, 81000\n1, GPU-1, A, 8.0, 81920, 81003", true);
        assertEquals("GPU-0", CellposeGpu.select(idle, "auto", null).orElseThrow().uuid());
    }

    @Test
    void pinningKeepsTheSubEnvironmentsVariables() {
        var gpu = CellposeGpu.parse(DISPLAY_AND_COMPUTE, true).get(1);
        Environment base = new Environment() {
            public String base() { return "/env"; }
            public List<String> binPaths() { return List.of("/env/bin"); }
            public List<String> launchArgs() { return List.of(); }
            public Map<String, String> envVars() { return Map.of("PIXI_ENVIRONMENT_NAME", "cp3-cu130"); }
            public Builder<?> builder() { return null; }
        };
        Environment pinned = CellposeGpu.pinned(base, java.util.Optional.of(gpu));
        assertEquals("cp3-cu130", pinned.envVars().get("PIXI_ENVIRONMENT_NAME"));
        assertEquals("GPU-bbbb", pinned.envVars().get("CUDA_VISIBLE_DEVICES"));
        assertEquals("PCI_BUS_ID", pinned.envVars().get("CUDA_DEVICE_ORDER"));
        assertTrue(CellposeGpu.pinned(base, java.util.Optional.empty()) == base);
    }
}
