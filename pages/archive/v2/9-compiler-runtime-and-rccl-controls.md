---
layout: distill
title: "Tuning the Compiler, Runtime, and RCCL"
description: "After the workload, mesh, and kernel routes are fixed, change XLA, JAX, HIP, and RCCL controls only when a profile identifies the mechanism."
date: 2026-09-13

section_number: 9

previous_section_url: "/pages/8-mixture-of-experts-on-mi355x"
previous_section_name: "Chapter 8: Training Mixture-of-Experts on MI355X"

next_section_url: "/pages/10-llama-7b-exposing-the-complete-stack"
next_section_name: "Chapter 10: Llama 7B"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "The Decision"
  - name: "Initialization Order"
  - name: "The Case-Study Baseline"
  - name: "Autotuning and Caches"
  - name: "Latency Hiding, Streams, and Memory"
    subsections:
      - name: "Read LHS in Scheduled HLO"
  - name: "Collective Combining, Pipelining, and Reordering"
    subsections:
      - name: "Combining"
      - name: "Pipelining"
      - name: "Reordering and Code Motion"
  - name: "Command Buffers"
  - name: "Triton GEMM Selection"
  - name: "RCCL Controls"
  - name: "MI355X Hazards"
  - name: "Deprecated, Removed, and No-Op Controls"
  - name: "Controlled Sweep Method"
  - name: "Versioned Status"
  - name: "Primary References"
---

## The Decision

Start from the frozen case-study configuration. Change one control only when a
profile identifies the mechanism it can affect. Keep the change only if the same
workload passes correctness checks and improves the target metric.

Earlier chapters identify the memory constraint, collective payloads, overlap
opportunities, and eligible kernel routes. This chapter is their single control
surface: scheduler flags, collective combining and pipelining, command buffers,
route-selection flags, hardware queues, and RCCL overrides are not tuned elsewhere.

This chapter uses:

- **[source]** for checked-in configuration or inspected source;
- **[analytical]** for conclusions from graph, memory, or scheduling structure;
- **[measured]** for observations backed by a complete Appendix F bundle;
- **[cited]** for behavior defined by named external documentation.

No flag sweep results are available for the three v26.6 case repositories. This
chapter defines the experiments and records the controls already present; it does
not assign speedups to them.

The headline metric is tokens/s/GPU. Peak HBM, exposed collective time, compile
time, and validation loss explain the result. A lower step time is not accepted
if the global batch, token count, model math, precision contract, or loss behavior
changed.

## Initialization Order

XLA flags, HIP controls, Transformer Engine selectors, JAX-AITER controls, and RCCL
variables are process initialization state. Set them before the component that
reads them.

Use this order:

1. Select devices and backend: `HIP_VISIBLE_DEVICES` and `JAX_PLATFORMS=rocm`.
2. Set HIP and allocator controls, including `GPU_MAX_HW_QUEUES` and
   `XLA_PYTHON_CLIENT_MEM_FRACTION`.
3. Set RCCL safety and diagnostic variables before any communicator is created.
4. Set Transformer Engine or JAX-AITER environment variables before importing
   those packages.
5. Construct one complete `XLA_FLAGS` string.
6. Set `PYTHONPATH` and library paths for the pinned MaxText, Transformer Engine,
   and JAX-AITER trees.
7. Import JAX.
8. Configure the JAX persistent compilation cache, if the experiment permits one.
9. Discover devices, initialize distributed JAX, construct the mesh, and compile.

**[cited]** JAX warns that backend initialization time is not a stable boundary.
Changing `XLA_FLAGS` after importing or using JAX may have no effect. RCCL also
caches environment values after first use. A flag arm therefore requires a fresh
process.

A safe Python launcher sets the environment before importing the target program:

```python
import os

env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
env.update(
    {
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "JAX_PLATFORMS": "rocm",
        "XLA_FLAGS": " ".join(flags),
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.97",
        "RCCL_WARP_SPEED_AUTO": "0",
    }
)
# Start a new Python process with env.
```

Removing an inherited `XLA_FLAGS` value before setting the frozen string avoids
duplicate keys with last-one-wins behavior. Apply explicit experiment overrides to
the parsed list, then serialize it once. Do the same for TE and JAX-AITER selector
families so an unrelated shell does not leak a kernel route into the run.

### Prove what reached XLA

Every HLO dump contains a `.debug_options` file. Save it and check the effective
values rather than quoting the launcher. Also save:

```bash
python3 - <<'PY'
import jax
print(jax.__version__)
print(jax.devices()[0].client.platform_version)
print(jax.devices())
PY
```

The effective environment, package versions, optimized HLO, and kernel trace are
one evidence unit. A shell command alone is not proof.

## The Case-Study Baseline

All three case repositories currently read the same XLA flag file:

```bash
--xla_gpu_autotune_level=4
--xla_gpu_enable_triton_gemm=true
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_all_gather_combine_threshold_bytes=8589934592
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
--xla_gpu_all_reduce_combine_threshold_bytes=8589934592
--xla_gpu_enable_command_buffer=
```

The last line is important: an empty repeated-enum value clears the command-buffer
command list. It disables command-buffer capture; it does not enable all commands.

Other frozen values differ by case:

- Llama 7B uses an allocator fraction of 0.9.
- Llama 70B and Mixtral normally use 0.97.
- The patched Llama 70B MXFP8 arm uses 0.94.
- Mixtral has one explicit LHS-off arm. Its sparse expert arms add the grouped-GEMM
  and ragged-AllToAll flags discussed below.

The repositories do not explicitly set async stream priority, collective
pipelining, queue count, or a forced RCCL algorithm/protocol. Their values therefore
come from the pinned runtime defaults or the outer launch environment. Capture them
before treating any prior run as a baseline.

## Autotuning and Caches

### XLA autotuning

**[cited]** `--xla_gpu_autotune_level` controls GEMM and convolution autotuning:

- `0` disables it;
- `1` enables timing without correctness checks;
- `2` also initializes output buffers with random values;
- `3` also resets output buffers after each candidate;
- `4` and above also compare outputs and check out-of-bounds reads and writes.

The upstream default is 4, and the case repositories set 4 explicitly. Level 4
can require extra compile-time memory for a reference output. If autotuning fails
with an allocation error, record that failure before testing level 3. Lowering the
level removes checks; it is not a memory optimization for the compiled train step.

Autotuning is keyed to a concrete fusion or library problem. A cache generated for
another GPU type, local shape, dtype, layout, XLA build, or fusion graph is not
portable evidence.

### JAX executable cache

JAX's persistent compilation cache stores compiled programs:

```python
import jax

jax.config.update("jax_compilation_cache_dir", "/path/to/cache")
```

**[cited]** Its key includes the non-optimized HLO, `jaxlib` version, relevant XLA
compilation flags, and device configuration. The cache does not make changes to
TE, JAX-AITER, RCCL, HIP, or untracked shared libraries safe. Use a separate cache
namespace that includes the full environment manifest, or run flag sweeps cold.

The raw Llama 7B implementation intentionally leaves `jax_cache_dir` empty unless
the caller opts in. Preserve that rule for publication timing: cold-compile cost
and warmed-step timing must be reported separately.

### XLA kernel and autotune caches

Current JAX can enable XLA caches alongside the persistent cache:

```python
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)
```

The current documented values are `none`,
`xla_gpu_per_fusion_autotune_cache_dir`, `xla_gpu_kernel_cache_file`, and `all`.
XLA also exposes monolithic dump/load flags:

```bash
--xla_gpu_dump_autotune_results_to=/path/cache.textproto
--xla_gpu_load_autotune_results_from=/path/cache.textproto
```

The three case repositories do not use a persisted autotune file. Platform support
for these mechanisms has changed across XLA releases. Before using one on the
pinned ROCm build, prove a dump/load round trip, inspect cache-hit logging, and
confirm that optimized HLO and kernel selection match. Otherwise report it as
unsupported for that pin.

### Cache policy for comparisons

Choose one policy before the sweep:

- **Cold:** new empty cache directory per arm. Report compile time separately.
- **Warm:** populate each arm's own cache once, then time a fresh process using
  that same arm-specific cache.

Never let the baseline populate a shared cache during the candidate measurement.
Do not combine a cache-policy change with an autotune-level change.

## Latency Hiding, Streams, and Memory

### Latency-hiding scheduler

```bash
--xla_gpu_enable_latency_hiding_scheduler=true
```

Do not attribute every collective or every start/done pair to LHS. The stages are
separate:

1. Shardy or another partitioner inserts the semantic AllGather, AllReduce,
   ReduceScatter, or other communication required by the per-device program.
2. A later asynchronous-collective conversion may split an eligible collective
   into start and done instructions.
3. The latency-hiding scheduler chooses a legal order and can move independent
   compute between start and done. It does not necessarily insert either the
   semantic collective or its asynchronous form.

This ordering follows the compiler path in
[Chapter 2]({{ '/pages/2-lowering-jax-jit-on-rocm' | relative_url }}#reading-a-compiler-delta).
**[cited]**
LHS can reduce exposed communication, but the resulting longer live ranges can
increase peak memory. A start/done pair proves that an asynchronous representation
is present, not that LHS created it; only the selected schedule can show whether
LHS opened an overlap window.

The Mixtral experiment contains the required first A/B:

```text
baseline: LHS=true
comparison: same BF16 FSDP-1/EP-8 recipe with LHS=false
```

No v26.6 result exists yet. The acceptance evidence is:

- identical post-partition semantic HLO, including local shapes, collective count,
  payloads, and replica groups;
- the same asynchronous start/done set at the pre-schedule pass boundary;
- a changed order, or an explicitly explained rematerialization, in scheduled HLO;
- overlap visible on separate trace lanes;
- unchanged loss and finite gradients;
- tokens/s/GPU and peak HBM from unprofiled runs.

If the HLO has no asynchronous collective or no independent compute window, the
flag has no mechanism to help.

### Read LHS in Scheduled HLO

Read the dump for the scheduled GPU module, not only an optimized-HLO DAG. The
following SVGs contain literal top-level operation lines, in file order, extracted
from `is_scheduled=true` HLO produced by the matched eight-MI355X fixture in
`bench/hlo_feature_fixtures.py`. The complete HLO and XLA DOT graph remain under
`artifacts/hlo-fixtures/lhs/`; the SVG removes parameter declarations and nested
fusion bodies only so the selected schedule is readable.

{% include figure.liquid path="pages/archive/img/hlo-lhs-off.svg" class="img-fluid" zoomable=true alt="Literal scheduled HLO operation sequence with latency hiding disabled" caption="Captured LHS-off schedule: `all-gather-start` is immediately followed by `all-gather-done`; the dependent GEMM and then the independent GEMM follow. Raw artifact: `artifacts/hlo-fixtures/lhs/off/`." %}

{% include figure.liquid path="pages/archive/img/hlo-lhs-on.svg" class="img-fluid" zoomable=true alt="Literal scheduled HLO operation sequence with latency hiding enabled" caption="Captured LHS-on schedule: the independent GEMM is placed after `all-gather-start` and before `all-gather-done`; the dependent GEMM follows. Raw artifact: `artifacts/hlo-fixtures/lhs/on/`." %}

A DAG alone cannot prove this schedule. It expresses a partial order and may admit
many topological orders; it can show that the GEMM is independent of the gather,
but not which legal order XLA selected, which allocation slices remained live, or
whether the runtime actually overlapped streams. Require all three artifacts:

1. **Scheduled HLO** to prove the chosen instruction order and the operations
   placed between collective start and done.
2. **Buffer assignment** to prove the corresponding live ranges, aliases, reused
   slices, and compiled peak.
3. **A device trace** to prove the GEMM and RCCL work executed concurrently on the
   expected lanes and shortened exposed collective time.

The HLO dump and trace commands are in
[Appendix D]({{ '/pages/d-profiler-and-hlo-cookbook' | relative_url }}#feature-comparison-bundles).
In the
LHS-on example, `%shard` and the asynchronous gather state or result must remain
live from `%ag.start` through `%ag.done`, while GEMM operands and then `%gemm` are
also live. Those overlapping intervals can prevent buffer reuse and raise the
static peak or runtime high-water mark. Report the changed live intervals and
bytes, not only total allocation, and reject the schedule if that longer lifetime
exceeds the memory budget.

Interpret this A/B only after checking collective combining. Several layer-local
gathers can become one combined gather: that may reduce launch overhead, but it can
also collapse several start/done intervals into one and remove the overlap windows
where independent layer GEMMs would have fit. Before interpreting an LHS result,
count collectives and record each payload size and replica group in optimized HLO,
then match those operations to the trace. One large combined gather with no usable
compute window is a combining outcome, not proof that LHS is ineffective in
general.

### Async Stream Priority

```bash
--xla_gpu_enable_highest_priority_async_stream=true
```

**[cited]** This selects highest priority for XLA's asynchronous communication
stream. Current upstream XLA defaults it to true, but the case repositories do not
set it. Read `.debug_options` before adding a redundant flag.

Priority can move time between communication and compute without changing the
amount of either. Sweep `true` versus `false` only when the trace shows an async
communication stream and either collectives are delayed or compute is being
starved. Keep LHS, queue count, and RCCL settings fixed.

### HIP Hardware Queues

```bash
GPU_MAX_HW_QUEUES=2
```

**[cited]** HIP defines this as the maximum reusable hardware queues per process
and device; extra streams share them round-robin. AMD's current MI355X JAX/MaxText
recipe sets it to 2. None of the three case launchers sets it explicitly.

Record the inherited value first. If queue count becomes an experiment, compare
only values supported by the runtime and use a trace to show whether stream
concurrency changed. More queues are not automatically more overlap.

### Scheduler Memory Slop

```bash
--xla_gpu_memory_limit_slop_factor=95
```

**[cited]** The scheduler multiplies its available-memory budget by this percentage
after accounting for long-lived inputs and outputs. LHS uses that budget when it
trades memory for overlap. A lower value can force a more memory-conservative
schedule; a higher value can leave less reserve for runtime allocations.

This is distinct from:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.97
```

The environment variable controls allocator reservation. The slop factor controls
the compiler's scheduling budget. Changing both in one arm makes an OOM or timing
change uninterpretable.

The current upstream default and AMD recipes use 95. The case repositories omit
the XLA flag, so verify the effective value before sweeping it.

## Collective Combining, Pipelining, and Reordering

### Combining

The case baseline sets 8 GiB thresholds:

```bash
--xla_gpu_all_gather_combine_threshold_bytes=8589934592
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
--xla_gpu_all_reduce_combine_threshold_bytes=8589934592
```

**[cited]** A combiner can merge compatible collectives up to its byte threshold.
The threshold is a cap, not a promised message size. Combining can reduce launch
overhead and expose a larger transfer, but it can also delay an early collective,
extend buffer lifetimes, or remove fine-grained overlap.

Do not sweep all three thresholds together. Use the HLO and trace to identify the
dominant collective, then sweep that operation's threshold around observed layer
payload boundaries. For every arm, record:

- number and size of collectives after optimization;
- replica groups;
- peak HBM;
- exposed versus overlapped duration;
- tokens/s/GPU.

**Required before/after artifacts:** retain optimized HLO immediately before and
after the collective-combiner pass, with operation count, operand/result bytes,
dimensions, channel IDs, and replica groups; then retain scheduled HLO, buffer
assignment, and a trace for each experiment arm. **Representation selection:** use
optimized HLO to prove that combining changed the semantic collective set,
scheduled HLO to read the chosen order and overlap windows, and the trace to measure
the actual RCCL launches and overlap.

The optional controls
`xla_gpu_enable_all_gather_combine_by_dim` and
`xla_gpu_enable_reduce_scatter_combine_by_dim` decide whether dimension matching
constrains combination. They are not explicit in the case repositories. Leave them
at the pin's default unless the HLO shows missed compatible combines attributable
to dimension grouping.

### Pipelining

The pinned MaxText examples use the boolean family:

```bash
--xla_gpu_enable_pipelined_all_gather=true
--xla_gpu_enable_pipelined_all_reduce=true
--xla_gpu_enable_pipelined_reduce_scatter=true
--xla_gpu_enable_while_loop_double_buffering=true
```

The three case repositories do not set these flags. They rely on XLA defaults.
Current upstream XLA also has enum controls:

```bash
--xla_gpu_pipeline_all_gather=default|off|on|explicit
--xla_gpu_pipeline_all_reduce=default|off|on|explicit
--xla_gpu_pipeline_reduce_scatter=default|off|on|explicit
```

`default` follows the compiler optimization effort. `explicit` considers only HLO
marked pipelineable. These enum controls and the older booleans coexist in some
releases. Do not mix the families until the pinned XLA source establishes their
precedence.

Pipelining is relevant only when the collective is in a suitable loop. Prove the
transformation in HLO: a new flag with an unchanged graph is an observed no-effect
for that workload.

**Required before/after artifacts:** retain HLO immediately before and after the
named pipelining pass, including the full while-loop state and body, and identify
the newly staged collective or changed start/done placement; retain scheduled HLO,
buffer assignment, and a trace for both arms. **Representation selection:** use
loop HLO to prove the pipeline rewrite, scheduled HLO and buffer assignment to
prove order and live-range cost, and the trace to prove iteration-to-iteration
overlap.

### Reordering and Code Motion

Two controls appear in upstream recommendations but are not general training
defaults:

```bash
--xla_disable_hlo_passes=collective-permute-motion
--xla_gpu_enable_while_loop_reduce_scatter_code_motion=true
```

The first disables a named HLO pass and is recommended in a specific pipeline
parallelism recipe. The second hoists eligible reduce-scatter work out of a while
loop. None of the current one-node cases uses pipeline parallelism. Do not add
either flag to their baseline.

If a future HLO contains the matching pattern, compare before/after loop bodies,
buffer lifetimes, and collective order. Collective reordering can deadlock when
ranks disagree about launch order, so the compiler's control dependencies are part
of correctness. Do not hand-force an order based only on a single-rank trace.

**Required before/after artifacts:** retain HLO immediately before and after the
named motion pass, preserving loop boundaries, channel IDs, replica groups, and
control dependencies; retain each arm's scheduled HLO and a multi-rank trace of
collective launch order. **Representation selection:** use loop HLO to prove code
motion across a loop boundary, scheduled HLO to prove reordering within a
computation, and the multi-rank trace to prove that every rank executed a compatible
runtime order.

## Command Buffers

**[cited]** XLA command buffers record a supported thunk sequence into a HIP graph
and replay it, reducing repeated host launch work. They are most relevant when
many small kernels or custom calls leave the host on the critical path.

The case baseline explicitly disables capture:

```bash
--xla_gpu_enable_command_buffer=
```

Current XLA treats this as a repeated enum of command categories. Supported
categories and ROCm coverage are version-dependent; ROCm does not support every
control-flow category available elsewhere. Query the pin before enabling a list.
For a controlled sweep:

1. prove host launch gaps in a no-buffer trace;
2. enable only the required categories;
3. confirm graph construction and replay after the first execution;
4. check that FFI, collectives, and dynamic addresses remain correct;
5. time steady-state runs outside a profiler.

Profiling can disable command buffers so that events remain attributable. Current
XLA has `xla_enable_command_buffers_during_profiling`, disabled by default. A
profiled run can therefore use different runtime behavior from an unprofiled
timing run. Record both effective modes.

Do not use the first captured execution as steady-state timing. Do not compare an
enabled arm's replay iterations with a disabled arm's compile or capture iteration.

## Triton GEMM Selection

```bash
--xla_gpu_enable_triton_gemm=true
```

**[cited]** This enables Triton-based matrix multiplication as a candidate. It
does not force every GEMM through Triton, and the autotuner can choose the
rocBLAS/hipBLASLt fallback.

The controlled A/B is `true` versus `false`, with autotune level fixed. Prove:

- `__triton_gemm` or generic Triton fusion configuration in optimized HLO;
- library custom calls in the non-Triton route;
- the actual kernels in `rocprofv3`;
- identical outputs and gradients;
- compile time, tokens/s/GPU, and peak HBM.

`xla_gpu_triton_gemm_any` broadens Triton-fusion formation for supported GEMMs,
while `xla_gpu_cublas_fallback=false` can remove the library candidate. Those are
debugging or forced-route controls, not case baselines. They can reduce the
candidate set and make a workload slower or unsupported.

The name `xla_gpu_enable_cublaslt` survives on ROCm because XLA shares interfaces.
Current upstream marks it deprecated, but the pinned Mixtral grouped-GEMM rewrite
requires it together with:

```bash
--xla_gpu_experimental_use_ragged_dot_grouped_gemm=true
```

Keep that pair local to the FP16 ragged-dot arm. Do not add it to every model.

## RCCL Controls

RCCL intentionally uses many `NCCL_*` environment names. They are RCCL controls
when the process links `librccl`.

### Diagnostics first

```bash
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING
```

**[cited]** These logs show communicator setup, topology/graph selection, and the
algorithm/protocol cost model. Enable them for a diagnostic run, save the log, then
remove verbose logging from timing runs.

### Algorithm and protocol

```bash
NCCL_ALGO=...
NCCL_PROTO=...
```

These override RCCL's automatic selection. Do not set them because a value worked
for another message size, participant count, or GPU generation. First collect the
automatic decision and a collective microbenchmark over the model's actual payload
sizes. Force one control at a time, then recheck every collective size represented
in the train step.

### Channels

```bash
NCCL_MIN_NCHANNELS=...
```

AMD documents this as a possible tuning control for partial participation on an
eight-GPU system. More channels consume more GPU resources. The current jobs use
all eight MI355X GPUs, but mixed FSDP/EP meshes can form two- or four-rank replica
groups. No forced minimum belongs in the baseline; consider it only when a trace
shows that a partial-participant collective is the bottleneck.

Transport-forcing controls are excluded here. The one-node cases have no evidence
that overriding automatic peer transport helps. Multi-node affinity is outside
the current scope.

## MI355X Hazards

### RCCL WarpSpeed and NaN loss

**[cited]** AMD's current MI355X JAX/MaxText guide says the gfx950 WarpSpeed auto
path can cause NaN training loss and requires:

```bash
RCCL_WARP_SPEED_AUTO=0
```

Primus sets it automatically for MI355X; manual launchers must set it themselves.
The variable is documented as a no-op on MI300X.

The three inspected case launchers do not set it. Before publishing new MI355X
results, either prove the outer environment supplied it or add it to the frozen
launcher and rerun. Do not use `RCCL_WARP_SPEED_ENABLE`, CU-count, unroll, or
threads-per-block tuning in a training result until the numerical issue is cleared
for the pinned RCCL build.

### Ragged AllToAll barrier

The Mixtral sparse arms set:

```bash
--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=true
--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false
```

The repository records that the pinned XLA/RCCL communicator does not support the
NCCL device barrier used by that path. Both flags are version-sensitive, and one
is explicitly `unsupported`. Keep them local to the two sparse arms. Newer XLA
enabled the NCCL barrier by default after adding collective-memory support; that
does not retroactively make it valid for the pin.

### gfx950 GroupedGEMM dtype

The pinned XLA hipBLASLt grouped-GEMM rewrite accepts FP16 on gfx950, not BF16.
This is why the Mixtral ragged and dense-padded pair both use FP16. A BF16 grouped
arm is unsupported for that pin, not a failed performance experiment.

### MXFP8 workspace

The Llama 70B MXFP8 arm requires the Transformer Engine gfx950 workspace patch
described in Chapter 7. A flag cannot repair an undersized FFI workspace. The
runner probes the installed source and fails if the patch is absent. Preserve that
probe.

### Memory reserve and autotune checks

High allocator reservation, LHS, large collective combines, command buffers, and
level-4 autotune checks can all need temporary memory at different phases. Change
one layer at a time:

1. compile/autotune;
2. command-buffer capture;
3. warmed steady-state execution.

An OOM at one phase does not justify lowering every memory-related control.

## Deprecated, Removed, and No-Op Controls

Flag status is tied to XLA commit, not to a copied command line.

- `xla_gpu_graph_level` is the legacy command-buffer control and has been removed
  from current upstream XLA. Use `xla_gpu_enable_command_buffer`.
- `xla_gpu_enable_pipelined_collectives` is deprecated. The operation-specific
  pipeline controls are separate; the older all-gather/all-reduce/reduce-scatter
  booleans are not the same deprecated flag.
- `xla_gpu_enable_nccl_clique_optimization` is a declared no-op in current
  upstream XLA.
- `xla_gpu_unsupported_force_triton_gemm` was removed in 2025.
- `xla_gpu_enable_cublaslt` is marked deprecated in current upstream XLA, but is
  still required by the pinned experimental ROCm grouped-GEMM path.
- `cublas`, `cudnn`, and `nccl` inside option or custom-call names can be
  compatibility vocabulary on ROCm. The name alone is neither a CUDA dependency
  nor proof of the selected ROCm kernel.

An unknown removed flag may terminate startup. A deprecated no-op may still parse.
For each release:

1. inspect `xla/debug_options_flags.cc` and `xla/xla.proto`;
2. compile one small module;
3. save `.debug_options`;
4. verify that HLO or runtime behavior changed in the expected place.

If the effective value changes but the relevant HLO and trace do not, record
“no effect for this workload and version,” not “the flag works.”

## Controlled Sweep Method

### 1. Freeze the manifest

Record:

- MI355X/gfx950 count and partition mode;
- container digest, ROCm/RCCL, JAX/`jaxlib`, plugin/PJRT, MaxText, TE, JAX-AITER,
  and Tokamax commits;
- model, sequence length, per-device microbatch, global batch, gradient
  accumulation, dtype roles, sharding, attention, remat, and data mode;
- complete environment and XLA flags;
- cache policy and output directories.

### 2. State one hypothesis

Tie the control to observed evidence:

```text
Trace: AllGather is exposed before each layer.
Hypothesis: LHS can place independent layer compute between start and done.
Control: xla_gpu_enable_latency_hiding_scheduler.
Reject if: HLO has no async pair, peak HBM exceeds budget, correctness changes,
or tokens/s/GPU does not improve outside run variance.
```

Do not begin with a list of recommended flags. Begin with the profile mechanism.

### 3. Build one baseline and one candidate

Use separate fresh processes and artifact directories. The only intended
difference is the tested control. Diff the resolved manifest before running.

For numeric thresholds, use a small staged sweep rather than many simultaneous
values:

1. current baseline;
2. one lower value tied to an observed payload boundary;
3. one higher value tied to the next boundary.

Stop if the HLO no longer changes or correctness fails.

### 4. Separate compile, profile, and timing

- Compile once and record compile/autotune/capture time.
- Warm up until compilation and graph capture are complete.
- Time synchronized steps without XProf, `rocprofv3`, PMCs, or verbose RCCL logs.
- Capture one separate warmed step for HLO-to-kernel and overlap evidence.
- Never use PMC duration as step time; counter collection can serialize dispatch.

Report median and spread over the frozen measurement window. Report tokens/s/GPU
before MFU.

### 5. Validate correctness

At minimum:

- compare first-step loss from identical state and data;
- require finite loss and gradients;
- compare a short loss trajectory;
- for kernel or low-precision changes, compare outputs and gradients against the
  reference route;
- use the Chapter 4 convergence guardrail before accepting a new precision or
  numerical mode.

The MI355X WarpSpeed safety variable is a prerequisite, not a performance arm.

### 6. Explain the result

Save and compare:

- `.debug_options`;
- optimized and scheduled HLO;
- buffer assignment or `memory_analysis()`;
- XProf or Perfetto overlap view;
- `rocprofv3` kernel and RCCL traces;
- kernel counts, collective payloads, and stream placement;
- tokens/s/GPU, peak HBM, compile time, and loss checks.

A timing delta without a mechanism is not enough to retain a flag.

### 7. Sweep in dependency order

Use this order so later controls do not hide earlier causes:

1. safety environment and exact software pins;
2. kernel eligibility and autotuning;
3. LHS and scheduler memory slop;
4. one collective combine threshold;
5. pipeline or code motion for a proven loop pattern;
6. async stream priority and HIP queue count;
7. command-buffer categories;
8. forced RCCL algorithm, protocol, or channels.

Rebase each accepted change into a new baseline. Do not keep a stack of individually
faster arms without testing the combined configuration.

## Versioned Status

Status on 2026-09-13:

- **Configured by all three repositories:** autotune level 4, Triton GEMM eligible,
  LHS enabled, 8 GiB thresholds for AllGather/ReduceScatter/AllReduce, command
  buffers disabled.
- **Defined as an A/B:** LHS true versus false in the Mixtral plan. No v26.6
  results are present.
- **Configured only by the Mixtral FP16 sparse pair:** hipBLASLt ragged
  GroupedGEMM selection, one-shot ragged AllToAll, and the NCCL barrier disabled.
- **Required by current AMD MI355X guidance:** `RCCL_WARP_SPEED_AUTO=0` and, in
  AMD's Primus recipe, `GPU_MAX_HW_QUEUES=2`. The inspected case launchers do not
  currently set either.
- **Inherited, not controlled in the cases:** highest-priority async stream,
  operation-specific collective pipelining, combine-by-dimension, and collective
  code motion.
- **Not measured in the cases:** command-buffer enablement, persistent cache
  effects, Triton-versus-library GEMM timing, forced RCCL algorithm/protocol, and
  channel-count sweeps.

Revalidate every default and deprecation after changing the container or any JAX,
XLA, ROCm, RCCL, MaxText, TE, or JAX-AITER pin.

**Recommendation status: BLOCKED.** The checked-in flag files define the baseline
to reproduce, not an optimized setting. No individual flag has an accepted
before/after bundle. Preserve the baseline, add the MI355X safety environment,
and change one control only after a profile identifies its mechanism.

## Primary References

- [JAX XLA flags](https://docs.jax.dev/en/latest/xla_flags.html)
- [JAX GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
- [JAX persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
- [OpenXLA flag guidance](https://openxla.org/xla/flags_guidance)
- [OpenXLA effort levels](https://openxla.org/xla/effort_levels)
- [OpenXLA persisted autotuning](https://openxla.org/xla/persisted_autotuning)
- [From HLO to thunks and command buffers](https://openxla.org/xla/hlo_to_thunks)
- [ROCm JAX/MaxText training guidance](https://rocm.docs.amd.com/projects/primus/en/latest/02-user-guide/jax-maxtext-training.html)
- [HIP environment variables](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html)
- [RCCL environment variables](https://rocm.docs.amd.com/projects/rccl/en/latest/api-reference/env-variables.html)
- [RCCL usage tips](https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/rccl-usage-tips.html)
- [Current XLA debug-option definitions](https://github.com/openxla/xla/blob/main/xla/xla.proto)

<h3 markdown=1 class="next-section">Next: [Llama 7B]({{ '/pages/10-llama-7b-exposing-the-complete-stack' | relative_url }}).</h3>
