---
layout: distill
title: "Profiler and HLO Cookbook"
description: "Commands and search patterns for HLO, XProf, rocprofv3, and rocprof-compute on the pinned MI355X stack."
date: 2026-09-13

section_label: "Appendix D"

previous_section_url: "/pages/c-appendix-config"
previous_section_name: "Appendix C: Configuration"

next_section_url: "/pages/e-appendix-compatibility"
next_section_name: "Appendix E: Compatibility"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: HLO Dumps
  - name: XProf
  - name: rocprofv3
  - name: rocprof-compute
  - name: Counter Passes
  - name: Search Patterns
  - name: Triage Order
---

Run timing and each profiler in separate processes. A profile explains a timing run; it does
not supply the published timing.

## HLO Dumps

Dump text HLO and selected pass boundaries:

```bash
XLA_FLAGS="$XLA_FLAGS \
  --xla_dump_to=/tmp/hlo \
  --xla_dump_hlo_as_text \
  --xla_dump_hlo_pass_re=.*" \
python3 workload.py
```

For a smaller output, replace `.*` with the pass or stage under investigation. Retain:

```text
before_optimizations
after_optimizations
after layout assignment
after autotuning
scheduled GPU module
buffer-assignment or compiled-memory report
```

Useful searches:

```bash
rg -n "custom-call|custom_call_target" /tmp/hlo
rg -n "all-gather|all-reduce|reduce-scatter|all-to-all|collective-permute" /tmp/hlo
rg -n "replica_groups|device_list|sharding=" /tmp/hlo
rg -n "f8|bf16|f16|f32" /tmp/hlo
rg -n "fusion|dot\\(" /tmp/hlo
```

Record the exact dump flags because exhaustive pass dumps can change compile time and disk
use.

## XProf

Capture after warmup and synchronize inside the range:

```python
for _ in range(10):
    state, metrics = train_step(state, batch)
    jax.block_until_ready((state, metrics))

with jax.profiler.trace("/tmp/xprof/run-id"):
    for _ in range(5):
        state, metrics = train_step(state, batch)
        jax.block_until_ready((state, metrics))
```

Open the capture:

```bash
xprof --logdir /tmp/xprof/run-id
```

Use:

| View | Question |
|---|---|
| Trace Viewer | What ran, on which stream, and what overlapped? |
| Op Profile / Framework Stats | Which JAX or HLO operations own time? |
| Kernel Stats | Which kernels launched, with what duration and geometry? |
| Memory Viewer / Profile | What contributes to the high-water mark? |
| Graph Viewer | Which HLO dependencies and fusions were compiled? |

An XSpace contains host and device XPlanes, lanes, events, and event stats. Preserve the
native `.xplane.pb`; exported timeline JSON is convenient but loses structure.

## rocprofv3

Use the profiler shipped in the v26.6 container:

```bash
ROCPROF=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3
```

Kernel trace:

```bash
NVTE_FRAMEWORK=jax "$ROCPROF" \
  --kernel-trace -f rocpd pftrace -d /tmp/prof -o kernels \
  -- python3 workload.py
```

Runtime, marker, memory, kernel, and RCCL trace:

```bash
NVTE_FRAMEWORK=jax "$ROCPROF" \
  -r --kernel-trace --rccl-trace -f rocpd pftrace \
  -d /tmp/prof -o runtime \
  -- python3 workload.py
```

Aggregated kernel statistics:

```bash
"$ROCPROF" --kernel-trace --stats \
  -d /tmp/prof -o stats -- python3 workload.py
```

Prefer `rocpd` for retained evidence and `pftrace` for quick Perfetto inspection. Keep ROCTx
ranges around the measured train step when the workload supports them.

## rocprof-compute

```bash
PROFILE_DIR=/tmp/rocprof-compute/workload
rocprof-compute profile --output-directory "$PROFILE_DIR" -- python3 workload.py
rocprof-compute analyze --path "$PROFILE_DIR"
```

Use `analyze -b <section>` for focused reports. The standalone GUI syntax is
version-dependent; current releases require `--experimental --gui`, while older
releases accepted `--gui`. Check `rocprof-compute analyze --help` in the recorded
container and save the exact command. Retain `pmc_perf.csv`, `sysinfo.csv`, raw pass
outputs, tool version, and the analyze command.

`rocprof-compute` replays work to collect incompatible counter sets. Its timeline and
duration are not application timing.

## Counter Passes

The Llama 7B repository groups counters into separate passes:

```text
SQ_WAVES GRBM_GUI_ACTIVE SQ_BUSY_CYCLES SQ_INSTS_VALU_MFMA_MOPS_BF16
FETCH_SIZE
WRITE_SIZE
TCC_HIT_sum TCC_MISS_sum
```

Example:

```bash
"$ROCPROF" --kernel-trace \
  --pmc "SQ_WAVES GRBM_GUI_ACTIVE SQ_BUSY_CYCLES SQ_INSTS_VALU_MFMA_MOPS_BF16" \
  -- python3 workload.py
```

Counter collection serializes dispatches on the current workflow. Label the output
**counter-only** and do not use overlap or duration from that run.

## Search Patterns

| Target | HLO or trace pattern | Interpretation |
|---|---|---|
| Fused AITER attention | `aiter::fmha_fwd` and `aiter::fmha_bwd` | Expected TE/AITER route in current Llama 7B work |
| XLA attention fallback | `fusion` and `gemm_fusion_dot` with no fused-attention custom call | XLA-generated attention |
| Dense GEMM | `gemm`, `rocblas`, `hipblaslt` | Confirm dtype and shapes separately |
| Grouped expert GEMM | `grouped`, `ragged_dot`, hipBLASLt custom call | Prove against the matched dense-padded arm |
| Collectives | `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all`, `rccl` | Check participants and overlap |
| Recompilation | Repeated compile regions or new module fingerprints | Timing window is invalid until explained |
| Host/input stall | Gap before the device step with host/input activity incomplete | Real-data pipeline did not keep up |
| Memory pressure | Allocation peak plus spill/copy activity | Reconcile with compiled memory |

Kernel names change across ROCm and library versions. Store the raw name and use patterns
only for discovery.

For a mesh claim, triangulate:

1. config axis factors;
2. optimized HLO sharding and replica groups;
3. trace collective type, size, and participants; and
4. physical rank/topology map.

## Triage Order

1. Confirm run identity, device count, partition mode, batch, and dtype.
2. Exclude compilation, autotuning, and input wait from the measured window.
3. Compare the actual mesh and HLO replica groups with the intended mesh.
4. Prove the kernel route and check for a silent or slow fallback.
5. Reconcile peak memory and unexpected copies.
6. Measure collective exposure and placement.
7. Use counters for the remaining kernel-level gap.
8. Recheck loss, gradients, and convergence whenever numerics changed.

Record the first failed check in [Appendix E]({{ '/pages/e-appendix-compatibility' |
relative_url }}) before applying a workaround.
