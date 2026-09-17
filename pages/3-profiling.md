---
layout: distill
title: "Profiling and Analysis of a Training Step"
description: "How to measure a JAX training step on MI355X, attribute GPU work through XSpace and HLO, and validate kernels with ROCprofiler-SDK counters."
date: 2026-09-16

section_number: 3

previous_section_url: "/pages/2-the-jax-software-stack-on-rocm"
previous_section_name: "Chapter 2: The JAX Software Stack on ROCm"

next_section_url: "/pages/4-mixed-precision"
next_section_name: "Chapter 4: Training in Mixed Precision"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "Profiler metrics"
    subsections:
      - name: "End-to-end metrics"
      - name: "Roofline analysis"
  - name: "A two-capture workflow"
    subsections:
      - name: "Warmup and asynchronous execution"
      - name: "Names that survive compilation"
  - name: "The JAX profiler"
    subsections:
      - name: "XLA profiler backend"
      - name: "XSpace"
  - name: "XProf"
    subsections:
      - name: "GPU Kernel Stats"
      - name: "GPU roofline analysis"
  - name: "rocprofv3"
    subsections:
      - name: "Trace collection"
      - name: "Perfetto"
      - name: "ROCTx annotations"
      - name: "PMC collection"
  - name: "TraceLens"
  - name: "rocprof-compute"
  - name: "Worked example: profiling a Mixtral 8x22B training step"
    subsections:
      - name: "Decomposing a train step"
      - name: "Forward pass"
      - name: "Backward pass"
      - name: "Time attribution in JAX"
      - name: "Time attribution in MaxText"
      - name: "Captured end-to-end result"
      - name: "Component roofline worksheet"
  - name: "From profile to precision"
---

A useful training profile connects four levels of evidence:

1. the duration and throughput of one optimizer update;
2. the JAX or MaxText component that requested the work;
3. the optimized HLO that XLA compiled; and
4. the HIP kernel or RCCL operation that ran on the GPU.

No single profiler exposes all four levels reliably. This chapter uses an
XPlane capture for framework and compiler attribution, then a separate
`rocprofv3` capture for ROCm runtime evidence. Hardware counters are collected
in another filtered run because dispatch-level counter collection changes the
execution schedule.

## Profiler metrics

Define the unit of measurement before collecting a trace. In this chapter, one
**train step** is one optimizer update. If gradient accumulation is two, the
step contains two forward/backward microsteps and one optimizer update.
Compilation, checkpointing, evaluation, and data-loader stalls are reported
separately unless the experiment is explicitly measuring end-to-end job time.

### End-to-end metrics

Let $t_{\mathrm{step}}$ be synchronized wall time for one update,
$N_{\mathrm{tok}}$ the global input tokens consumed by that update, $G$ the
number of accelerators participating in it, $F_{\mathrm{model}}$ the declared
useful model FLOPs per update, and $C_{\mathrm{peak}}$ the matching per-device
compute peak. Then

$$
R_{\mathrm{node}}=\frac{N_{\mathrm{tok}}}{t_{\mathrm{step}}},
\qquad
R_{\mathrm{device}}=\frac{N_{\mathrm{tok}}}{G\,t_{\mathrm{step}}},
$$

and

$$
\mathrm{MFU}
=\frac{F_{\mathrm{model}}}
       {G\,t_{\mathrm{step}}\,C_{\mathrm{peak}}}
=\frac{R_{\mathrm{node}}\,(F_{\mathrm{model}}/N_{\mathrm{tok}})}
       {G\,C_{\mathrm{peak}}}.
$$

[MaxText defines MFU](https://maxtext.readthedocs.io/en/latest/reference/performance_metrics.html)
from theoretical model FLOPs and measured step time. That convention is
deliberately different from counting every instruction executed by the GPU.
Rematerialization, padded expert slots, routing, optimizer fusions, and
communication can consume time without increasing useful model FLOPs.

| Metric | Required denominator | Report with it |
|---|---|---|
| Step time | one synchronized optimizer update | warmup window, statistic, spread |
| Tokens/s | global input tokens in that update | global or per-device; padded or non-padding |
| Useful MFU | declared useful model FLOPs | dtype peak, device count, FLOP ledger |
| Executed FLOP rate | operations actually executed | kernel/component boundary and evidence source |
| Memory footprint | bytes on one physical GPU | static compiled peak, observed high-water mark, worst device |
| Component time | union of attributed device intervals | device/rank, step, overlap rule |

The denominators prevent several common errors:

- Do not multiply a resolved global batch by the device count again.
- Do not count top-$k$ expert assignments as input tokens. Assignment
  throughput is a separate metric equal to $kN_{\mathrm{tok}}/t$ before drops.
- Use the dense BF16 peak for dense BF16 MFU. A structured-sparse or FP8 peak
  is valid only when the measured kernels use that path.
- Report the maximum per-device memory footprint. Summing HBM across devices
  does not show whether one shard fits.
- Do not call one observed duration a median. A stable timing result needs
  several post-warmup updates and a dispersion statistic.

Memory also has two useful views. XProf's
[Memory Viewer](https://openxla.org/xprof/memory_viewer) uses compiler data to
show the static buffer assignment and its peak in program order. The dynamic
Memory Profile and device telemetry show allocator behavior at runtime. Record
arguments, outputs, aliases, temporaries, and runtime high-water marks rather
than collapsing unlike values into one unexplained number.

### Roofline analysis

The [JAX Scaling Book roofline chapter](https://jax-ml.github.io/scaling-book/roofline/)
derives the general model. For a boundary with $F$ floating-point operations,
$Q$ bytes transferred at the selected memory level, peak compute $C$, and
bandwidth $\beta$,

$$
I=\frac{F}{Q},
\qquad
P=\frac{F}{t},
\qquad
P_{\mathrm{roof}}=\min(C,I\beta).
$$

The roofline efficiency is

$$
\eta_{\mathrm{roof}}=\frac{P}{\min(C,I\beta)}.
$$

Every term must describe the same work. An HBM roofline uses HBM bytes; an
L1/shared-memory roofline uses traffic at that level. A network roofline uses
collective payload bytes and link bandwidth, not HBM bandwidth.

For $A[M,K]B[K,N]$, a useful first prediction is

$$
F_{\mathrm{GEMM}}=2MKN,
$$

$$
Q_{\mathrm{algorithmic}}
=s_A MK+s_B KN+s_C MN,
\qquad
I_{\mathrm{algorithmic}}
=\frac{2MKN}{Q_{\mathrm{algorithmic}}},
$$

where $s_A,s_B,s_C$ are element sizes. This byte expression assumes one read
of each input and one output write. It excludes cache misses, workspace,
padding, split-$K$ reductions, and rereads, so it is a prediction. Counter-
derived $Q_{\mathrm{HBM}}$ is the measurement used for a kernel's empirical
HBM roofline.

For a component containing kernels $k$,

$$
I_c=\frac{\sum_k F_k}{\sum_k Q_k},
\qquad
P_c=\frac{\sum_k F_k}{t_{\mathrm{union},c}}.
$$

$t_{\mathrm{union},c}$ is the union of that component's intervals on one
selected GPU, not a sum that double-counts overlapping streams. Keep useful
and executed FLOPs in separate columns. A rematerialized projection has one
useful contribution to the model ledger but can execute more than once.

{% include figure.liquid path="pages/img/ch3-roofline-mi355x.png" class="img-fluid" alt="The theoretical MI355X BF16 roofline with an 8 TB per second HBM slope, a 2.5166 PFLOP per second compute ceiling, and a ridge near 315 FLOP per byte" caption="The theoretical dense-BF16/HBM roofline for one full MI355X in SPX mode. The line uses published peaks, not measured kernel points. Empirical points must use measured traffic and a matching dtype-specific compute ceiling." %}

## A two-capture workflow

The workflow has two profiler evidence families, plus an unprofiled timing run:

| Execution | Purpose | Keep |
|---|---|---|
| Unprofiled | headline step time, tokens/s, MFU | resolved config, all step records, environment |
| XPlane | JAX/MaxText and HLO attribution | `.xplane.pb`, HLO protos, optimized HLO, memory analysis |
| `rocprofv3` trace | HIP, kernel, memory-copy, and RCCL timing | rocpd database and converted Perfetto trace |
| Filtered PMC | per-dispatch hardware evidence | exact kernel/dispatch selector, counters, launch dimensions |

The PMC execution belongs to the ROCm evidence family but remains a separate
process from the timing trace. Counter collection serializes dispatches, so its
durations cannot replace the unprofiled or trace timings.

{% include figure.liquid path="pages/img/ch3-two-capture-workflow.png" class="img-fluid" alt="A workflow that runs the same resolved MaxText configuration under XPlane and rocprofv3, then joins JAX scopes, HLO operations, kernels, and counters" caption="Use one resolved configuration and independent processes. XPlane plus optimized HLO identifies the model component; rocprofv3 identifies the runtime operation and hardware behavior. The correlation ledger is the shared output." %}

For MaxText v26.6, a compact XPlane capture can be requested with:

```bash
python -m maxtext.trainers.pre_train.train "$CONFIG" \
  run_name=profile-xplane steps=14 \
  profiler=xplane \
  skip_first_n_steps_for_profiler=10 \
  profiler_steps=3 \
  profile_cleanly=true \
  dump_hlo=true dump_step=10 \
  dump_hlo_delete_local_after=false
```

The exact output directory comes from the resolved MaxText configuration.
Retain that configuration beside the trace. MaxText's
[`Profiler`](https://github.com/ROCm/maxtext/blob/b47d74bf4ef860c6cbd0fe5e4705c362ba360dbe/src/maxtext/common/profiler.py)
calls `jax.profiler.start_trace()` and `stop_trace()` at the requested steps;
[`profile_cleanly`](https://github.com/ROCm/maxtext/blob/b47d74bf4ef860c6cbd0fe5e4705c362ba360dbe/src/maxtext/configs/base.yml)
adds synchronization at the capture boundaries.

### Warmup and asynchronous execution

JAX dispatch is asynchronous. Measuring only the Python call usually measures
enqueue latency:

```python
import time

compiled = jax.jit(train_step)

# Compile, autotune, allocate, and populate caches outside the timing window.
state, metrics = compiled(state, batch)
jax.block_until_ready((state, metrics))

start = time.perf_counter()
state, metrics = compiled(state, batch)
jax.block_until_ready((state, metrics))
elapsed = time.perf_counter() - start
```

One warmup call is a minimum, not a universal rule. Continue until compilation
and autotuning have finished and step times have reached a stable regime. Then
declare the retained window before comparing configurations. Capture complete
steps only; partial first or last steps distort XProf aggregates.

Profiled runs are diagnostic. Trace buffers, callbacks, file output, and
dispatch-level counters add overhead. Headline timing comes from the matched
unprofiled process.

### Names that survive compilation

Use [`jax.profiler.StepTraceAnnotation`](https://docs.jax.dev/en/latest/profiling.html)
for a host-visible step boundary. Use
[`jax.named_scope`](https://docs.jax.dev/en/latest/_autosummary/jax.named_scope.html)
when a model component is ambiguous:

```python
with jax.named_scope("moe/router"):
    gate_logits = router(hidden_states)
```

`named_scope` extends JAX's name stack, so the name reaches JAXPR and HLO
metadata. A host trace annotation creates a timeline event but does not rename
HLO operations. Add scopes sparingly: Flax module paths and MaxText's existing
MoE scopes already identify many operations, and per-layer or per-token names
can make the trace and HLO harder to group.

Compiler optimization can fuse, clone, or remove named operations. Preserve
the optimized HLO and read the relationship as many-to-many:

```text
JAX/Flax scope → HLO operation(s) → thunk/custom call → kernel dispatch(es)
```

Do not split one `jax.jit` into several compiled calls merely to obtain ranges.
That changes fusion and scheduling boundaries and therefore changes the
program being profiled.

## The JAX profiler

[`jax.profiler.start_trace`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.start_trace.html)
is the Python entry point to OpenXLA's profiler. The implementation combines
host instrumentation, backend-specific device events, and compiler sideband
data. XProf then derives higher-level views from those sources.

### XLA profiler backend

The control path is:

1. `start_trace()` constructs a
   [`ProfilerSession`](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/lib/profiler_session.cc).
   The session creates registered `ProfilerInterface` implementations and
   starts them. Only one session profiles a process at a time.
2. The host tracer collects TraceMe events, including JAX profiler
   annotations.
3. The ROCm
   [`GpuTracer`](https://github.com/openxla/xla/blob/main/xla/backends/profiler/gpu/device_tracer_rocm.cc)
   enables XLA's annotation stack, starts `RocmTracer`, and owns a
   `RocmTraceCollector`.
4. [`RocmTracer`](https://github.com/openxla/xla/blob/main/xla/backends/profiler/gpu/rocm_tracer.cc)
   configures ROCprofiler-SDK services for HIP runtime APIs, kernel dispatches,
   and memory copies. HIP correlation IDs associate API callbacks with
   asynchronous activity records. The active JAX/XLA annotation stack and
   direct application ROCTx range, when present, are recorded at the API
   boundary. Each accepted SDK record is normalized into a `RocmTracerEvent`.
5. [`RocmTraceCollector`](https://github.com/openxla/xla/blob/main/xla/backends/profiler/gpu/rocm_collector.cc)
   turns those tracer events into host and per-GPU XPlanes, attaches correlation,
   kernel, stream, annotation, and memory-copy fields, and normalizes device
   timestamps to the session's wall-clock origin.
6. `ProfilerSession::CollectData(XSpace*)` stops all profilers, merges their
   planes, post-processes the single-host space, and exports the TensorBoard
   profile directory.

XLA supplies more than a label. It owns the profiler session and factories,
emits runtime annotations, and exports HLO proto, cost, source, and buffer
metadata used by XProf. ROCprofiler-SDK supplies the ROCm runtime event
records. Attribution is their join through correlation IDs, name-stack
metadata, and HLO sidecars; it is not a device-side HLO timer.

{% include figure.liquid path="pages/img/ch3-jax-profiler-pipeline.png" class="img-fluid" alt="OpenXLA profiler pipeline from JAX host annotations and ROCprofiler-SDK records through host and ROCm tracers into XSpace and XProf" caption="How a JAX ROCm capture is assembled. ROCprofiler-SDK records timed HIP, dispatch, and copy activity. OpenXLA coordinates the session, carries annotation and compiler metadata, builds XPlanes, and exports XSpace for XProf." %}

### XSpace

The authoritative
[`xplane.proto`](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/xplane.proto)
defines a small hierarchy:

- `XSpace` contains repeated `XPlane` objects plus hostnames, warnings, and
  errors.
- `XPlane` contains parallel `XLine` timelines, plane-level stats, and maps
  that deduplicate event and stat metadata.
- `XLine` has an ID, name, start timestamp in nanoseconds, duration in
  picoseconds, and repeated events.
- `XEvent` refers to event metadata by ID and stores an offset, duration, and
  repeated `XStat` values.
- `XStat` stores a typed value whose name and description are in the plane's
  stat-metadata map.

An `.xplane.pb` file is a serialized XSpace. A GPU plane normally contains
lines for streams or queues; the host plane contains thread timelines.
Additional planes carry task environment and scope-range relationships. HLO
protos are retained alongside the XSpace in the profile directory.

{% include figure.liquid path="pages/img/ch3-xspace-structure.png" class="img-fluid" alt="The XSpace protobuf hierarchy from XSpace through XPlane and XLine to XEvent and XStat, with one GPU kernel event expanded" caption="XSpace separates parallel timelines from shared metadata. A kernel interval is an XEvent on a stream line; correlation IDs, framework/HLO names, resource details, and ROCTx labels are XStats or derived sideband information." %}

XProf's GPU lines must be read with this distinction in mind. The
[Trace Viewer documentation](https://openxla.org/xprof/trace_viewer) states
that raw GPU stream data is directly grounded in the collected profile, while
GPU XLA-op and framework lines are derived from stream data and optional
compiler metadata. GPU execution can have an $N{:}M$ relationship between HLO
operations and kernels.

## XProf

The [JAX Scaling Book profiling chapter](https://jax-ml.github.io/scaling-book/profiling/)
already explains the generic XProf workflow. The official
[XProf documentation](https://openxla.org/xprof) covers Overview, Trace Viewer,
HLO Op Stats, Graph Viewer, Memory Viewer, and exports. Start a local instance
against the retained log directory:

```bash
python -m pip install xprof
xprof --logdir="$PROFILE_DIR" --port=6006
```

On MI355X, first confirm that GPU stream lines are present, the device is
identified as `gfx950`, and the selected interval contains complete post-
warmup steps. A host-only trace cannot support kernel or GPU roofline claims.

### GPU Kernel Stats

[GPU Kernel Stats](https://openxla.org/xprof/gpu_kernel_stats) groups each
unique kernel and originating framework-operation pair. Its most useful
columns are kernel name, op name, occurrences, total/average/minimum/maximum
duration, register count, shared-memory use, block and grid dimensions, and
occupancy when available.

Use it in this order:

1. Sort by total duration to find the kernel/op pairs that dominate the
   capture.
2. Filter by an existing MaxText scope such as `wi_0`, `attention`, or
   `combine`.
3. Check occurrences against layers, gradient-accumulation microsteps, and
   forward/backward expectations.
4. Follow the op name into HLO. Record every emitted kernel, including
   transposes, reductions, and epilogues around the main GEMM.
5. Match the full or stable kernel-name prefix in the `rocprofv3` trace.

Do not infer that one row equals one model operation. One HLO fusion can emit
several kernels, and the same library kernel can serve several HLO operations.

{% comment %}
AUTHOR SCREENSHOT PLACEHOLDER — DO NOT PUBLISH AS A FIGURE.
Proposed asset: pages/img/ch3-xprof-kernel-stats-mixtral.png
Tool/page: XProf 2.23.x, GPU Kernel Stats.
Capture configuration: MaxText v26.6; 8× MI355X in SPX/NPS1; Mixtral 8x22B;
BF16; FSDP=4, EP=2; sequence length 4096; per-device batch 4; gradient
accumulation 2; synthetic reused batch; attention=cudnn_flash_te;
scan_layers=true; remat_policy=save_dot_with_context_except_mlp; XPlane
capture of complete steps 10–12.
Visible selection: filter Op Name for MoeBlock_0/wi_0; select the largest
total-duration row. Keep Kernel Name, Op Name, Occurrences, Total/Average/Min/
Max Duration, Registers, Shared Memory, Block, Grid, and Occupancy columns
visible. Show the run/device selector and all three complete steps.
Caption to teach: one logical expert up-projection can map to more than one
kernel/op pair; occurrence count and total duration must be reconciled with
56 scanned layers and two accumulation microsteps before assigning time.
{% endcomment %}

### GPU roofline analysis

XProf's [Roofline Model](https://openxla.org/xprof/roofline_model) supports GPU
profiles in beta. It can show program-level and operation-level points for HBM
and L1/shared memory. The page may expose both XLA cost-model FLOP rates and
rates derived from hardware counters. Keep those labels in exported data; they
are different evidence.

Before quoting efficiency:

- verify the Device Information peaks against the captured MI355X partition,
  dtype, and clock convention;
- identify whether FLOPs and bytes are compiler estimates or counter-derived;
- filter to one component and inspect the operations omitted by that filter;
- compare operation points with GPU Kernel Stats because HLO-to-kernel
  attribution is many-to-many; and
- recompute the point from retained rows if the device peak or byte level is
  wrong.

An attractive point on an incorrect roofline is not evidence. In particular,
do not apply one BF16 peak to FP32 router reductions, FP8 GEMMs, and RCCL
kernels.

{% comment %}
AUTHOR SCREENSHOT PLACEHOLDER — DO NOT PUBLISH AS A FIGURE.
Proposed asset: pages/img/ch3-xprof-roofline-mixtral-wi0.png
Tool/page: XProf 2.23.x, Roofline Model, Operation-Level Analysis.
Capture configuration: exactly the Mixtral XPlane capture specified in the
GPU Kernel Stats placeholder above.
Visible selection: GPU 0; HBM memory level; named-op filter MoeBlock_0/wi_0;
Device Information panel with gfx950 peak FLOP/s and HBM bandwidth; chart axes
with the ridge; one selected wi_0 point and its details; statistics table with
operation, occurrences, self time, model/measured FLOP rate, HBM bandwidth,
operational intensity, bottleneck, and roofline efficiency.
Caption to teach: XProf provides the first component roofline estimate, but
the reader must retain whether each point uses XLA model data or hardware
counters and must validate the selected MI355X roof.
{% endcomment %}

## rocprofv3

[`rocprofv3`](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
is the command-line client built on ROCprofiler-SDK. Use it after XProf has
identified the step and candidate HLO operations.

### Trace collection

Run the same resolved MaxText configuration in a fresh process with its JAX
profiler disabled:

```bash
rocprofv3 \
  --runtime-trace --rccl-trace \
  --output-directory "$ROCPROF_DIR" \
  --output-format rocpd -- \
  python -m maxtext.trainers.pre_train.train "$CONFIG" \
    run_name=profile-rocprof profiler="" steps=14
```

`--runtime-trace` collects HIP runtime, marker, kernel-dispatch, and memory
activity; `--rccl-trace` makes the collective requirement explicit. Keep the
default rocpd database. AMD recommends
[converting rocpd after collection](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocpd-output-format.html)
because direct CSV and PFTrace output can omit newer record types:

```bash
rocpd convert -i "$ROCPROF_DIR/<pid>_results.db" \
  --output-format pftrace
```

The trace answers runtime questions that HLO cannot: which HIP calls launched
work, which queues carried kernels, whether kernels overlapped, when memory
copies occurred, and where RCCL activity was exposed. Correlation IDs connect
host APIs to asynchronous dispatch records. Kernel start/end timestamps, not
HIP API duration, define device execution time.

Trace collection still has overhead. Use it to explain the matched unprofiled
timing, not to replace it.

Without profiler-control instrumentation, this command also records startup
and warmup; select complete steps 10–12 during analysis. If the application
has a thin ROCTx control hook, add `--selected-regions`, resume immediately
before step 10, synchronize after step 12, and then pause. That reduces the
trace without changing the compiled train-step body.

### Perfetto

Open the converted trace in [Perfetto UI](https://perfetto.dev/docs/visualization/perfetto-ui).
For one complete step:

1. pin the host step or ROCTx track;
2. pin the selected GPU's compute and copy queues;
3. show RCCL API and RCCL kernel tracks;
4. zoom to the exact step boundaries;
5. select the candidate dispatch and record its kernel name, queue, start,
   duration, correlation ID, grid, workgroup, LDS, scratch, and register data;
6. measure interval unions for components and intersection intervals for
   overlap.

Do not sum all eight GPUs' kernel durations and compare that total with wall
time. Choose the critical rank/device, or report per-device values and their
spread.

{% comment %}
AUTHOR SCREENSHOT PLACEHOLDER — DO NOT PUBLISH AS A FIGURE.
Proposed asset: pages/img/ch3-perfetto-mixtral-step.png
Tool/page: Perfetto UI loaded with a PFTrace converted from rocprofv3 rocpd.
Capture configuration: same MaxText v26.6 Mixtral configuration as the XProf
placeholders, but profiler=""; rocprofv3 runtime and RCCL tracing; host ROCTx
range/control hook around complete step 10 after warmup; one process using
eight GPUs.
Visible lanes: host train-step/ROCTx range, HIP runtime launch thread, the
critical GPU's compute queues/streams, memory-copy queue, RCCL API, and RCCL
kernels. Show the GPU ID and zoom to one whole optimizer update. Select one
expert wi_0 dispatch so its kernel name,
correlation ID, queue/stream, start, duration, grid/workgroup, LDS, scratch,
VGPR, AccumVGPR, and SGPR fields are visible.
Caption to teach: host launch time is not GPU execution time; the selected
dispatch and RCCL lanes show the device interval and any real compute/
communication overlap inside one synchronized update.
{% endcomment %}

### ROCTx annotations

[ROCTx](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html)
provides host-side markers, push/pop ranges, and profiler pause/resume control.
With `rocprofv3 --selected-regions`, collection starts disabled and occurs only
between `roctxProfilerResume(0)` and `roctxProfilerPause(0)`. Place the pause
after a device synchronization so queued work is not cut off.

ROCTx does not create ranges inside a compiled GPU executable. Python inside a
`jax.jit` function runs while JAX traces the function, not for every execution.
A ROCTx call around the outer Python invocation can bracket the whole
synchronized step, but it cannot divide that executable into attention,
router, and expert device ranges. `jax.named_scope` plus optimized HLO supplies
that internal attribution.

For non-jitted code or a program that intentionally launches several compiled
executables, host ROCTx ranges can bracket each launch. Synchronize within the
range when the range is meant to cover device completion. Do not refactor one
compiled train step into many calls solely to make a prettier ROCTx lane.

When a JAX/XLA build exports direct application ROCTx events into XSpace, they
appear as host marker lines. They remain host ranges; their presence does not
turn them into device-side annotations.

### PMC collection

ROCprofiler-SDK supports dispatch-level performance monitoring counters (PMCs).
List and validate the counters provided for the installed gfx950 stack:

```bash
rocprofv3-avail list --pmc
rocprofv3-avail pmc-check SQ_WAVES GRBM_GUI_ACTIVE
```

Then filter to a stable kernel name and occurrence:

```bash
rocprofv3 \
  --pmc SQ_WAVES GRBM_GUI_ACTIVE \
  --kernel-include-regex '<escaped-kernel-prefix>' \
  --kernel-iteration-range 10-10 -- \
  python kernel_replay.py
```

The counters above illustrate the command; they are not a complete roofline
counter set. Select architecture-specific FLOP, HBM-traffic, cache, occupancy,
or stall metrics from `rocprofv3-avail`, and record the formulas used to turn
raw counters into bytes or operations.

The
[ROCprofiler-SDK counter service](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/counter_collection_services.html)
requires serialized kernel execution for per-dispatch counting. Streams and
queues retain their identities, but kernels on the target GPU no longer run
concurrently. Thus:

- PMC durations do not measure the production schedule or overlap;
- co-dependent kernels can deadlock under dispatch serialization;
- multiple `--pmc` groups run separate application passes; and
- counters from different passes are comparable only when the selected
  dispatch, shape, route, and launch configuration repeat.

Use the normal trace for timing and overlap. Use the PMC run for isolated
hardware evidence.

## TraceLens

[AMD TraceLens](https://github.com/AMD-AGI/TraceLens) can read JAX XPlane,
`rocprofv3` JSON, and PFTrace data and generate structured performance reports.
Its official
[JAX report guide](https://rocm.docs.amd.com/projects/tracelens/en/latest/how-to/generate-perf-report-jax.html)
provides:

```bash
TraceLens_generate_perf_report_jax \
  --profile_path path/to/host.xplane.pb
```

For this book, TraceLens is an optional analysis layer. The current
[compatibility matrix](https://rocm.docs.amd.com/projects/tracelens/en/latest/reference/compatibility.html)
pins `xprof==2.20.1` and `protobuf>=6.31.1,<7`, while the inspected JAX 0.11
MaxText environment contains XProf 2.23.1. Run TraceLens in a separate
environment, retain the generated tables, and validate its categories against
XProf, HLO, and raw runtime records. It does not replace the two-capture
evidence chain.

## rocprof-compute

[ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/use.html)
collects predefined counter groups and derives kernel-level analyses. Start
from a kernel name and launch configuration found in the normal trace, then
prefer a deterministic single-kernel reproducer:

```bash
rocprof-compute profile \
  --output-directory ./profiles/mixtral-expert-wi \
  --kernel 'stable_kernel_name_substring' \
  --dispatch 1 -- \
  python expert_replay.py

rocprof-compute analyze \
  --path ./profiles/mixtral-expert-wi \
  --experimental --gui
```

Profile mode can replay the application many times to collect incompatible
counter groups. Kernel and dispatch filters reduce that cost. Replaying a full
distributed MaxText job also repeats initialization and can make dispatch IDs
unstable, so extraction is preferable. Iteration multiplexing avoids
application replay by spreading counter groups over repeated dispatches, but
it needs enough identical occurrences and trades accuracy for collection
speed.

The analysis combines top kernel statistics, speed-of-light metrics, memory
and cache behavior, and an empirical roofline. Its timing is replay timing,
not the train-step timing used for throughput.

{% comment %}
AUTHOR SCREENSHOT PLACEHOLDER — DO NOT PUBLISH AS A FIGURE.
Proposed asset: pages/img/ch3-rocprof-compute-expert-wi.png
Tool/page: ROCm Compute Profiler standalone GUI, analyze mode.
Capture configuration: single-GPU deterministic replay extracted from the
MaxText v26.6 Mixtral FSDP=4/EP=2 expert wi_0 operation; preserve local M/N/K,
BF16 dtype, layouts, selected backend/kernel, fixed-capacity shape,
grid/workgroup, and gfx950; filter to one dispatch; standard replay mode, no iteration
multiplexing.
Visible selection: chosen kernel and dispatch in the dropdowns; Top Stats,
System Speed-of-Light, Empirical Roofline, Roofline AI Data Metrics, memory
chart, and relevant instruction/cache panels. Keep the workload directory and
gfx950 system information visible.
Caption to teach: replayed counters explain one kernel's compute and memory
limits; they do not reproduce train-step overlap or provide an end-to-end MFU.
{% endcomment %}

## Worked example: profiling a Mixtral 8x22B training step

The worked configuration comes from the local
`mixtral8-22b/configs/mixtral8-22b.yml` experiment at source commit
`a32b51d6` and MaxText's
[`mixtral-8x22b.yml`](https://github.com/ROCm/maxtext/blob/b47d74bf4ef860c6cbd0fe5e4705c362ba360dbe/src/maxtext/configs/models/mixtral-8x22b.yml)
at the inspected ROCm MaxText v26.6 commit `b47d74bf`.
It matches the main architecture facts published by
[Mistral AI](https://mistral.ai/news/mixtral-8x22b/).

| Quantity | Value |
|---|---:|
| Decoder layers | 56 |
| Model width $D$ | 6,144 |
| Query heads / KV heads | 48 / 8 |
| Head dimension | 128 |
| Expert width $F$ | 16,384 |
| Experts / selected experts | 8 / 2 |
| Vocabulary | 32,768 |
| Sequence length | 4,096 |
| Per-device batch / accumulation | 4 / 2 |
| Devices / mesh | 8 MI355X / FSDP=4, EP=2 |
| Global tokens per update | 262,144 |
| Dtype | BF16 |

Here $4\times8\times2=64$ sequences contribute to an update, so
$N=64\times4096=262{,}144$ input tokens.

The experiment requests Transformer Engine fused attention, scanned layers,
and `save_dot_with_context_except_mlp` rematerialization. It uses a synthetic
batch and performs two forward/backward microsteps before each AdamW update.

### Decomposing a train step

Use components that remain meaningful across compiler versions:

```text
optimizer update
└── accumulation microstep × 2
    ├── forward
    │   ├── embedding and 56 decoder layers
    │   └── final norm, LM head, loss
    └── backward
        ├── loss/LM-head gradients
        ├── decoder-layer VJPs in reverse order
        └── gradient collectives and accumulation
```

Within each decoder layer, separate attention, routing, token movement, expert
compute, normalization/residual work, and their backward counterparts. This
gives each component a FLOP rule, byte boundary, HLO anchor, and runtime
kernel set.

### Forward pass

MaxText's
[`MixtralDecoderLayer`](https://github.com/ROCm/maxtext/blob/b47d74bf4ef860c6cbd0fe5e4705c362ba360dbe/src/maxtext/models/mixtral.py)
applies pre-attention RMSNorm, grouped-query self-attention, an attention
residual, post-attention RMSNorm, and a routed MoE block. The router selects
two of eight experts for each token. The selected routes become dispatch and
combine masks in this fixed-capacity configuration. Tokens enter two expert
input projections (`wi_0`, `wi_1`), combine through SiLU gating, pass through
`wo`, and return to token order under the route weights. The existing scope
names come from the same release's
[`RoutedMoE`](https://github.com/ROCm/maxtext/blob/b47d74bf4ef860c6cbd0fe5e4705c362ba360dbe/src/maxtext/layers/moe.py).

{% include figure.liquid path="pages/img/ch3-mixtral-forward.png" class="img-fluid" alt="Mixtral 8x22B forward pass with attention, router, dispatch, top-two expert projections, combine, residuals, final norm, LM head, and loss" caption="Forward attribution map for one Mixtral decoder layer. The labels on the right are present in the current MaxText source and should be reused before adding new scopes." %}

### Backward pass

The backward pass is not one reversed forward kernel. Automatic
differentiation produces input-gradient and weight-gradient GEMMs, reduction
and transpose fusions, the fused-attention backward implementation, reverse
token movement, router gradients, and gradient collectives. Custom VJPs can
replace the default transpose rules.

Rematerialized forward work executes inside the backward interval. Attribute
that device time to backward/rematerialization while keeping the useful model
FLOP ledger unchanged.

{% include figure.liquid path="pages/img/ch3-mixtral-backward.png" class="img-fluid" alt="Mixtral 8x22B backward pass with expert and attention VJPs, router and reverse-dispatch gradients, gradient collectives, and AdamW" caption="Backward attribution map. Parameter-gradient leaves feed sharding-dependent collectives before AdamW. Dashed arrows group those leaves; they do not assert a device schedule." %}

### Time attribution in JAX

For a standalone JAX model, place `named_scope` around stable architecture
components before applying `value_and_grad` and `jit`. Keep one
`StepTraceAnnotation` around the synchronized optimizer update. After capture:

1. list optimized HLO operations by name stack;
2. identify custom calls, fusions, collectives, and loop bodies;
3. map each HLO operation to all corresponding Kernel Stats rows;
4. split forward, backward, rematerialized, and optimizer occurrences; and
5. reconcile the interval union with the selected step boundary.

A scope name is an attribution hint, not a timing boundary. Confirm every
large bucket against HLO shapes and runtime kernels.

### Time attribution in MaxText

Use MaxText's existing names first:

| Component | Source/HLO anchors | Runtime evidence |
|---|---|---|
| Attention projections | `self_attention`, query/key/value/out dots | GEMM kernel names and local shapes |
| Attention core | fused-attention custom call and forward/backward targets | backend kernels inside each call |
| Router | `MoeBlock_0/gate`, `top_k`, softmax/reduction fusions | vector/reduction kernels and HBM traffic |
| Dispatch | `MoeBlock_0/dispatch`, mask materialization, AllToAll HLO | dispatch kernels, RCCL calls, payload bytes |
| Expert up | `wi_0`, `wi_1` | selected GEMM backend, capacity shape or group sizes |
| Activation | `ffn_act` | fused SiLU/multiply kernels |
| Expert down | `wo` | dense or grouped GEMMs |
| Combine | `combine`, `weight_sum`, unpermute | reverse movement, reduction/fusion kernels |
| Backward | VJP names, gradient dots, fused-attention backward | transpose GEMMs, custom VJP kernels |
| Optimizer | AdamW update fusions | elementwise/reduction kernels after accumulation |

With `scan_layers=true`, expect a compiled loop body and repeated occurrences,
not 56 independently named Python calls. With accumulation two, most
forward/backward operations occur for both microsteps while the optimizer runs
once. These counts are useful checks on attribution.

The following is a real optimized-HLO fixture retained under
`artifacts/hlo-fixtures/moe/ragged-grouped/`. The fixture is small
(`tokens[64,128]`, four expert matrices) so the graph stays readable. It is not
a Mixtral timing result or the fixed-capacity path measured below. It shows the
correlation signature to use if a sparse ragged expert path is selected:
`ragged_dot_general` lowers to the compatibility target
`__cublas$lt$groupedMatmul`, which reaches the BLASLt implementation on ROCm.

{% include figure.liquid path="pages/img/ch3-hlo-ragged-grouped.svg" class="img-fluid" zoomable=true alt="Graphviz rendering of a real optimized gfx950 HLO fixture in which tokens, expert matrices, and group sizes enter a grouped matrix multiplication custom call" caption="Literal gfx950 optimized HLO rendered with Graphviz from the retained JAX/ROCm fixture. The custom-call target is the bridge between the JAX ragged-dot name and the grouped GEMM kernel sought in rocprofv3. This is compiler evidence, not a performance measurement." %}

The attention path has the same pattern. In the retained Transformer Engine
fixture, Q, K, V, and metadata enter
`custom_call_target="te_fused_attn_forward_ffi"`. The call identifies the
framework-to-runtime handoff; the `rocprofv3` trace identifies the kernels
that implemented it.

{% include figure.liquid path="pages/img/hlo-attention-te.svg" class="img-fluid" zoomable=true alt="Graphviz-rendered real HLO fixture for Transformer Engine fused-attention forward with Q, K, V, and metadata entering an FFI custom call" caption="Real pre-optimization HLO fixture for the Transformer Engine attention route. Use its custom-call and source names for attribution, then use the ROCm trace for the actual backend kernels and durations." %}

### Captured end-to-end result

The local source `/home/clchong/work/crusoe-cluster-results.md` contains one
successful v26.6 run for the FSDP=4/EP=2 fixed-capacity one-hot configuration:

| Field | Captured or derived value |
|---|---:|
| Run | `20260907T100351Z-...-fsdp4-ep2-...-rccl-warp-off` |
| Hardware | one node, 8× MI355X |
| Workload | synthetic, sequence 4,096, global batch 64 |
| Recorded step samples | 1 |
| Step time [measured] | 20.599 s |
| Tokens/s/device [derived from measured step] | 1,590.7 |
| Useful TFLOP/s/device [derived from measured step and model ledger] | 385.3 |
| BF16 MFU [derived] | 15.31% |

The MFU uses the 2.5166 PFLOP/s dense-BF16 peak from Chapter 1:

$$
\mathrm{MFU}=\frac{385.3}{2516.6}=0.1531.
$$

The run completed, but one recorded step provides no variance estimate. The
summary does not retain the immutable container digest or full effective
configuration, and the local result bundle does not contain a matching
XPlane, optimized full-model HLO, `rocprofv3` trace, router ledger, or PMC set.
Therefore the table is an end-to-end anchor only. No component time
percentages are claimed.

### Component roofline worksheet

MaxText's useful training-FLOP convention counts forward matrix work and twice
that work for backward. For this exact configuration, the analytical ledger
uses

$$
\begin{aligned}
F_{\mathrm{proj}}&=3(2NLD[(H_q+2H_{kv})d_h+D]),\\
F_{\mathrm{attn}}&=3(2BLS^2H_qd_h),\\
F_{\mathrm{router}}&=3(2NLDE),\\
F_{\mathrm{experts}}&=3(2NLk(3DF)),\\
F_{\mathrm{head}}&=3(2NDV).
\end{aligned}
$$

Here $B=64$ sequences, $S=4096$, $N=BS$, $L=56$, $E=8$, $k=2$,
and $V=32{,}768$. The factor three is one forward plus two backward
matrix-multiplication equivalents; it does not count the optimizer. Substitution
gives:

| Component | Formula basis | Useful PFLOPs/update [analytical] | Share |
|---|---|---:|---:|
| Attention projections | Q, K, V, O across 56 layers | 7.758 | 12.22% |
| Causal attention core | QK and probability-V products | 2.217 | 3.49% |
| Router | $D\times8$ gate in 56 layers | 0.004 | 0.007% |
| Two selected experts | two SwiGLU experts, three matrices each | 53.199 | 83.78% |
| Output vocabulary head | $D\times32{,}768$ | 0.317 | 0.50% |
| **Total** | 242.212 GFLOPs/token × 262,144 tokens | **63.495** | **100%** |

This is a useful-work prediction. It excludes norms, sorting, token movement,
collectives, optimizer arithmetic, padding, dropped assignments, and
rematerialized repeats. Those operations remain in the time and byte ledgers.

Fill one row per component from matched captures:

| Worksheet field | XPlane/HLO source | ROCm source |
|---|---|---|
| Scope and HLO IDs | optimized HLO name stack, custom-call target | correlated kernel prefix |
| Local shapes and dtype | HLO operands/results and shardings | grid/workgroup and kernel metadata |
| Occurrences | Kernel Stats and loop body | dispatch IDs in one complete step |
| Device time | union of GPU events on critical device | normal trace start/end timestamps |
| Useful FLOPs | analytical model ledger | carried unchanged |
| Executed FLOPs | optimized shapes, padding, remat, backend contract | validated instruction/counter metrics |
| HBM bytes | XLA estimate, clearly labeled | counter-derived read/write bytes |
| Communication bytes | HLO collective shape and replica groups | RCCL payload/trace records |
| Arithmetic intensity | executed FLOPs / measured HBM bytes | derived from retained counters |
| Roofline efficiency | matching dtype peak and memory level | recomputed point and counter provenance |

The missing Mixtral capture should use the exact screenshot configurations
specified above and retain machine-readable exports. The minimum completion
bundle is:

```text
manifest.yaml
config/effective.yaml
timing/steps.jsonl
hlo/gfx950_gpu_after_optimizations.txt
hlo/memory-analysis.txt
profiles/xprof/<host>.xplane.pb
profiles/rocprof/<rank>_results.db
ledgers/component-kernels.csv
ledgers/router-loads.csv
ledgers/component-roofline.csv
```

`component-kernels.csv` should contain step, rank, device, component, scope,
HLO ID, custom-call target, full kernel name, dispatch ID, occurrence, shape,
dtype, start, duration, useful FLOPs, executed FLOPs, HBM bytes, and counter
pass. This is enough to reproduce both the time attribution and the component
roofline without reading values from screenshots.

## From profile to precision

The output of this chapter is a denominator-checked baseline: synchronized
step time, useful FLOP ledger, per-device memory peak, component-to-kernel map,
and counter-backed roofline points for the components worth changing.

[Chapter 4]({{ '/pages/4-mixed-precision' | relative_url }}) changes the
numeric recipe. Keep token count, shapes, mesh, routing, and capture windows
fixed; update the dtype-specific compute ceiling; then repeat the same
attribution. That comparison shows whether lower precision accelerated the
intended GEMMs, reduced HBM traffic, introduced conversion work, or moved the
bottleneck elsewhere.

<h3 markdown=1 class="next-section">Next: [training in mixed precision]({{ '/pages/4-mixed-precision' | relative_url }}).</h3>

