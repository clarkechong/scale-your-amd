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

Global tokens/s is the input token count divided by synchronized step time;
per-device tokens/s divides that value by the number of accelerators. MFU uses
the declared useful model FLOPs and the matching per-device compute peak:

$$
\mathrm{MFU}
=\frac{F_{\mathrm{model}}}
       {G\,t_{\mathrm{step}}\,C_{\mathrm{peak}}}.
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

JAX dispatch is asynchronous. Compile and autotune before the retained timing
window, then call `jax.block_until_ready()` at its boundaries. Profiler runs
are diagnostic; headline timing comes from an otherwise matched unprofiled
process.

Memory also has two useful views. XProf's
[Memory Viewer](https://openxla.org/xprof/memory_viewer) uses compiler data to
show the static buffer assignment and its peak in program order. The dynamic
Memory Profile and device telemetry show allocator behavior at runtime. Record
arguments, outputs, aliases, temporaries, and runtime high-water marks rather
than collapsing unlike values into one unexplained number.

### Roofline analysis

The [JAX Scaling Book roofline chapter](https://jax-ml.github.io/scaling-book/roofline/)
introduces the plot below. Arithmetic intensity is the work performed per byte
moved, $I=F/Q$. For peak compute $C$ and bandwidth $\beta$, the roofline is

$$
P_{\mathrm{roof}}=\min(C,I\beta).
$$

{% include figure.liquid path="pages/img/scaling-book-roofline.png" class="img-fluid" alt="Generic roofline plot with two bandwidth ceilings, two algorithms, and bandwidth-bound and compute-bound regions" caption="The generic roofline from the MIT-licensed <a href='https://jax-ml.github.io/scaling-book/roofline/'>JAX Scaling Book</a>. Its BW1 and BW2 lines illustrate two memory systems or two achieved bandwidths. The MI355X values are substituted in the text rather than drawn into the source figure." %}

The source chapter applies this diagram to TPU v5e. For one full MI355X in
SPX mode, the dense BF16 matrix ceiling is 2.5166 PFLOP/s and HBM bandwidth is
8 TB/s, so the BF16/HBM ridge is about 315 FLOP/byte. An operation to the left
of that point is HBM-bound even with an ideal kernel. An operation to the right
can be compute-bound.

The byte boundary must match the line being used. An HBM roofline counts HBM
traffic; an L1/LDS roofline counts traffic at that level; a network roofline
uses collective payload and link bandwidth. For a component, sum the executed
FLOPs and measured bytes of its kernels, then divide by the union of its device
intervals so overlapping streams are not counted twice.

## The JAX profiler

[`jax.profiler.start_trace`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.start_trace.html)
is the Python entry point to OpenXLA's profiler. The implementation combines
host instrumentation, backend-specific device events, and compiler sideband
data. XProf then derives higher-level views from those sources.

### XLA profiler backend

Calling `start_trace()` opens an OpenXLA
[`ProfilerSession`](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/lib/profiler_session.cc).
The ROCm path then has two important objects:

- [`RocmTracer`](https://github.com/openxla/xla/blob/main/xla/backends/profiler/gpu/rocm_tracer.cc)
  configures ROCprofiler-SDK and receives HIP API, kernel-dispatch, and
  memory-copy records. Correlation IDs and the active JAX/XLA name stack travel
  with those records.
- [`RocmTraceCollector`](https://github.com/openxla/xla/blob/main/xla/backends/profiler/gpu/rocm_collector.cc)
  normalizes timestamps and writes the events and metadata into host and GPU
  XPlanes.

The profiler session merges those planes with host annotations and compiler
metadata into one XSpace. XProf reads that XSpace.

{% include figure.liquid path="pages/img/ch3-jax-profiler-pipeline.png" class="img-fluid" alt="Minimal ROCm profiler pipeline from RocmTracer through RocmTraceCollector to XSpace and XProf" caption="The ROCm profiler path reduced to its core objects. The tracer receives runtime records; the collector organizes them into XPlanes; the profiler session stores those planes in XSpace for XProf." %}

### XSpace

The authoritative
[`xplane.proto`](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/xplane.proto)
defines the hierarchy. An XSpace contains XPlanes for hosts, devices, and
sideband data. Each plane contains parallel XLines, each timed interval is an
XEvent, and XStats hold fields such as correlation ID, HLO name, kernel
geometry, and source metadata.

{% include figure.liquid path="pages/img/what-is-an-xspace.png" class="img-fluid" alt="Annotated XProf Trace Viewer identifying an XPlane, a timeline lane, XEvents, and the XStats details pane" caption="An XSpace as displayed by XProf. The screenshot labels a displayed timeline as an XLane; the protobuf message is named `XLine`. XEvents occupy intervals on a line, and the selected event's XStats appear in the details pane." %}

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
Proposed filename: ch3-xprof-kernel-stats-mixtral.png
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
Proposed filename: ch3-xprof-roofline-mixtral-wi0.png
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
Proposed filename: ch3-perfetto-mixtral-step.png
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
  --profile_path "$XPLANE"
```

For this book, TraceLens is an optional analysis layer. The current
[compatibility matrix](https://rocm.docs.amd.com/projects/tracelens/en/latest/reference/compatibility.html)
pins `xprof==2.20.1` and `protobuf>=6.31.1,<7`, while the inspected JAX 0.11
MaxText environment contains XProf 2.23.1. Run TraceLens in a separate
environment, retain the generated tables, and validate its categories against
XProf, HLO, and raw runtime records. It does not replace those primary
sources.

## rocprof-compute

[ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/use.html)
collects predefined counter groups and derives kernel-level analyses. Start
from a kernel name and launch configuration found in the normal trace, then
prefer a deterministic single-kernel reproducer:

```bash
rocprof-compute profile \
  --output-directory "$PROFILE_OUTPUT" \
  --kernel 'stable_kernel_name_substring' \
  --dispatch 1 -- \
  python expert_replay.py

rocprof-compute analyze \
  --path "$PROFILE_OUTPUT" \
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
Proposed filename: ch3-rocprof-compute-expert-wi.png
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

The worked configuration uses case-study source commit `a32b51d6` and
MaxText's
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

One optimizer update contains two accumulation microsteps. Each microstep has
a forward pass through the embedding, 56 decoder layers, final norm, language
model head, and loss, followed by the corresponding reverse-mode work. The
optimizer runs once after gradients from both microsteps have accumulated.

Within a decoder layer, attribute attention, routing, token movement, expert
compute, normalization, and residual work separately. The next two diagrams
give those forward and backward boundaries.

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

{% include figure.liquid path="pages/img/ch3-mixtral-forward.png" class="img-fluid" alt="Vertical Mixtral 8x22B forward flow through attention, routing, dispatch, expert MLP, combine, and residual operations" caption="One decoder layer expanded into a single top-to-bottom flow. The highlighted blocks include the MaxText scope names used to correlate model regions with HLO and XProf." %}

### Backward pass

The backward pass is not one reversed forward kernel. Automatic
differentiation produces input-gradient and weight-gradient GEMMs, reduction
and transpose fusions, the fused-attention backward implementation, reverse
token movement, router gradients, and gradient collectives. Custom VJPs can
replace the default transpose rules.

Rematerialized forward work executes inside the backward interval. Attribute
that device time to backward/rematerialization while keeping the useful model
FLOP ledger unchanged.

{% include figure.liquid path="pages/img/ch3-mixtral-backward.png" class="img-fluid" alt="Mixtral 8x22B backward flow with a main activation-gradient path and a separate parameter-gradient rail" caption="Activation gradients follow the vertical decoder-layer VJP. Parameter gradients from the MoE and attention branches feed the side rail for collectives, microstep accumulation, and the optimizer update." %}

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

The following real optimized-HLO fixture uses `tokens[64,128]` and four expert
matrices, which keeps the graph readable. It is not
a Mixtral timing result or the fixed-capacity path measured below. It shows the
correlation signature to use if a sparse ragged expert path is selected:
`ragged_dot_general` lowers to the compatibility target
`__cublas$lt$groupedMatmul`, which reaches the BLASLt implementation on ROCm.

{% include figure.liquid path="pages/img/ch3-hlo-ragged-grouped.svg" class="img-fluid" zoomable=true alt="Graphviz rendering of a real optimized gfx950 HLO fixture in which tokens, expert matrices, and group sizes enter a grouped matrix multiplication custom call" caption="Literal gfx950 optimized HLO rendered with Graphviz. The custom-call target is the bridge between the JAX ragged-dot name and the grouped GEMM kernel sought in rocprofv3. This is compiler evidence, not a performance measurement." %}

The attention path has the same pattern. In the retained Transformer Engine
fixture, Q, K, V, and metadata enter
`custom_call_target="te_fused_attn_forward_ffi"`. The call identifies the
framework-to-runtime handoff; the `rocprofv3` trace identifies the kernels
that implemented it.

{% include figure.liquid path="pages/img/hlo-attention-te.svg" class="img-fluid" zoomable=true alt="Graphviz-rendered real HLO fixture for Transformer Engine fused-attention forward with Q, K, V, and metadata entering an FFI custom call" caption="Real pre-optimization HLO fixture for the Transformer Engine attention route. Use its custom-call and source names for attribution, then use the ROCm trace for the actual backend kernels and durations." %}

### Captured end-to-end result

The retained cluster summary contains one successful v26.6 run for the
FSDP=4/EP=2 fixed-capacity one-hot configuration:

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

One recorded step provides no variance or component attribution, so the table
is an end-to-end anchor only.

### Component roofline worksheet

MaxText's useful training-FLOP convention counts forward matrix work and twice
that work for backward. The table applies that factor of three to the Q/K/V/O
projections, causal attention, router, selected expert matrices, and language
model head. It does not count the optimizer.

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

The missing Mixtral capture should retain the resolved configuration, step
records, optimized HLO, memory analysis, XSpace, ROCprof database, router
loads, and a machine-readable component ledger. Each component row needs its
scope and HLO ID, kernel name, dispatch occurrence, shape, dtype, interval,
useful and executed FLOPs, HBM bytes, and counter pass. That is sufficient to
reproduce the attribution and roofline without reading values from screenshots.

<h3 markdown=1 class="next-section">Next: [training in mixed precision]({{ '/pages/4-mixed-precision' | relative_url }}).</h3>

