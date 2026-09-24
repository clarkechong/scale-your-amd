---
layout: distill
title: "Profiling and Analysis of a Training Step"
description: "Use roofline estimates, XProf, rocprofv3, and rocprof-compute to explain one complete JAX training step on MI355X."
date: 2026-09-14

section_number: 3

previous_section_url: "/pages/2-lowering-jax-jit-on-rocm"
previous_section_name: "Chapter 2: Lowering jax.jit on ROCm"

next_section_url: "/pages/4-training-in-mixed-precision"
next_section_name: "Chapter 4: Training in Mixed Precision"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "Recommended Reading"
  - name: "Roofline Analysis"
    subsections:
      - name: "Arithmetic intensity"
      - name: "The MI355X BF16 roofline"
      - name: "From a bound to a profile"
  - name: "JAX and ROCm Profiling Tool Stack"
    subsections:
      - name: "XProf"
      - name: "rocprofv3"
      - name: "What ROCTx can and cannot attribute"
      - name: "Hardware counters"
      - name: "rocprof-compute"
  - name: "Worked Example: A Complete Llama 7B Training Step"
    subsections:
      - name: "Workload and predictions"
      - name: "Clean timing"
      - name: "Locate the cost with XProf"
      - name: "Inspect the dispatches with rocprofv3"
      - name: "Add hardware counters"
      - name: "Profile one GEMM with rocprof-compute"
      - name: "Reconcile the evidence"
  - name: "End-to-End Training Performance"
  - name: "References"
---

Performance analysis is performed across multiple abstraction layers, and no
single profiler currently provides complete visibility from JAX programs down to
GPU hardware. Framework-level profilers are therefore used first to identify
expensive phases, execution patterns, and overall utilization. While tools such
as XProf aim to bridge multiple layers of the stack, low-level details such as
kernel occupancy, memory behavior, hardware counters, and instruction-level
characteristics remain the domain of dedicated GPU profilers.

The typical workflow is therefore top-down: begin with JAX-level traces and
roofline analysis, identify hotspots, then progressively move to XLA, library,
stream, and kernel-level tools to understand the root cause of observed behavior.
This chapter applies that workflow to one complete BF16 Llama 7B optimizer update:

1. estimate its compute and memory bounds;
2. time it without instrumentation;
3. use XProf to identify expensive model regions and execution patterns;
4. use optimized HLO and `rocprofv3` to identify the underlying operations and
   dispatches;
5. collect hardware counters only for the remaining kernel-level questions; and
6. reconcile the evidence from all layers.

Timing and profiling use separate processes. Counter collection changes execution
and is not a source of application timing. ROCTx is useful for delimiting the
selected compiled update, but it cannot recover attribution inside a compiled
`jax.jit(train_step)` module.

The working results in this chapter were collected using one
MI355X in SPX/NPS1 mode, ROCm 7.14, JAX and jaxlib 0.11.0, the ROCm PJRT/plugin
0.11.0, XProf 2.23.1, and ROCm Compute Profiler 3.7.0.

## Recommended Reading

- The JAX Scaling Book's
  [roofline chapter](https://jax-ml.github.io/scaling-book/roofline/) derives
  compute, memory, and communication bounds.
- Its
  [Transformer chapter](https://jax-ml.github.io/scaling-book/transformers/)
  derives parameter, projection, attention, and training FLOP counts.
- [Chapter 1]({{ '/pages/1-mi355x-as-a-training-machine' | relative_url }})
  supplies the MI355X compute and HBM constants.
- [Chapter 2]({{ '/pages/2-lowering-jax-jit-on-rocm' | relative_url }})
  explains how JAX scopes become HLO and GPU dispatches.
- [Appendix B]({{ '/pages/b-measurement-and-convergence-protocol' | relative_url }})
  defines the timing protocol.
- [Appendix D]({{ '/pages/d-profiler-and-hlo-cookbook' | relative_url }})
  is the command reference for HLO dumps and profilers.

The roofline concepts needed to interpret the worked example are summarized
below.

## Roofline Analysis

Roofline analysis is a useful top-level analysis method for high-performance
workloads. It measures achieved performance against the theoretical ceiling for a
workload with a particular arithmetic intensity.

The analysis can be applied hierarchically. Consider an XLA module
`train_step` as the top-level workload. If the complete update operates far below
its roofline, decompose it into model regions, HLO operations, and finally kernels,
then repeat the analysis at the level where a specific optimization can be made.
The purpose is not simply to label the complete step as "slow"; it is to identify
which component is limited by which resource.

For a workload with a known FLOP count and data movement, first estimate:

$$
\begin{aligned}
t_{\mathrm{compute}}
  &= \frac{\mathrm{FLOPs}}{P_{\mathrm{compute}}},\\
t_{\mathrm{HBM}}
  &= \frac{\mathrm{HBM\ bytes}}{\beta_{\mathrm{HBM}}},\\
t_{\mathrm{comms}}
  &= \frac{\mathrm{link\ bytes}}{\beta_{\mathrm{link}}}.
\end{aligned}
$$

Perfect overlap gives the largest term. No overlap gives their sum:

$$
\max(t_{\mathrm{compute}},t_{\mathrm{HBM}},t_{\mathrm{comms}})
\leq t_{\mathrm{resource}}
\leq
t_{\mathrm{compute}}+t_{\mathrm{HBM}}+t_{\mathrm{comms}}.
$$

Kernel-launch latency, idle gaps, poor tiling, cache misses, synchronization, and
inefficient implementations can push measured time above this resource model.

### Arithmetic intensity

Workloads tend to be limited by compute, memory movement, communication, or some
combination of them. Arithmetic intensity quantifies the balance between the first
two: it is the number of FLOPs performed per byte transferred.

$$
I=\frac{\mathrm{FLOPs}}{\mathrm{bytes}}.
$$

A memory-copy kernel performs no arithmetic and therefore has zero FLOP/byte. It
can saturate HBM bandwidth while achieving zero FLOP/s, because FLOP/s is simply
the wrong measure of useful work for a copy.

A large square GEMM is the opposite case. Its arithmetic grows as $O(n^3)$ while
its input and output data grow as $O(n^2)$. Its arithmetic intensity therefore
increases with matrix size. At sufficiently large dimensions, the Matrix Cores
rather than HBM become the limiting resource.

The HBM roofline is:

$$
P_{\mathrm{roofline}}
=
\min\left(P_{\mathrm{compute}},I_{\mathrm{HBM}}\beta_{\mathrm{HBM}}\right).
$$

For a projection $X[B_{\mathrm{tok}},D]W[D,F]$, when reading the weight dominates
traffic and each element uses $w$ bytes:

$$
I_{\mathrm{HBM}}\approx\frac{2B_{\mathrm{tok}}}{w}.
$$

In BF16, $w=2$, so the approximate intensity equals the local token count. This
estimate assumes one weight read and omits cache effects, workspaces, casts, and
output traffic. It is a quick classification, not a measured byte count.

The ridge point is where the memory and compute ceilings meet:

- to its left, attainable FLOP/s rises with arithmetic intensity and HBM
  bandwidth;
- to its right, attainable FLOP/s is capped by the Matrix Core peak; and
- near the ridge, both resources matter and imperfect overlap is most visible.

Two different optimization questions follow. First, is the implementation
efficient relative to the roofline that its arithmetic intensity permits? Cache
misses, bank conflicts, spills, poor memory access, insufficient parallel work, or
an unsuitable MFMA tile can leave a kernel below that bound. Second, can the
algorithm itself increase reuse? Fusion, tiling, and retaining intermediate values
on chip can raise arithmetic intensity by avoiding full HBM round trips. Some
algorithms, such as a pure memory copy, cannot be transformed into compute-bound
work by tuning.

### The MI355X BF16 roofline

One MI355X has a published dense BF16 matrix peak of 2.5166 PFLOP/s and 8 TB/s
of HBM bandwidth. Its BF16 ridge point is:

$$
I_{\mathrm{ridge}}
=
\frac{2.5166\times10^{15}}{8\times10^{12}}
=314.6\ \mathrm{FLOP/byte}.
$$

An operation below approximately 315 FLOP/byte cannot reach the dense BF16
compute ceiling when HBM supplies every counted byte. Above that point it may be
compute-bound, but only if the kernel exposes enough parallel work and keeps the
Matrix Cores supplied.

Changing dtype changes both the compute ceiling and the number of bytes moved.
[Chapter 4]({{ '/pages/4-training-in-mixed-precision' | relative_url }}) treats that
as a training decision. This chapter keeps BF16 fixed.

### From a bound to a profile

Theoretical and empirical rooflines answer different questions. The theoretical
line uses data-sheet peaks. An empirical line uses sustained compute and bandwidth
measured on the same machine. A cache-level roofline also needs a different byte
count for each boundary; HBM traffic cannot be reused as an estimate of LDS or L2
traffic.

A roofline classifies a possible limit, but it does not identify the model
component, selected backend, kernel, launch geometry, or cause of idle time. It can
show where an optimization opportunity exists without explaining why the
opportunity exists. That requires the profiling stack.

## JAX and ROCm Profiling Tool Stack

Performance analysis is performed across multiple abstraction layers. At the
identification stage, the framework-level profiler is usually the right place to
start because it retains the model and HLO context. Once an expensive region has
been identified, lower-level tools expose details that the framework trace does
not preserve.

The tools therefore answer different questions:

| Tool | Main question | Useful output | Main limitation |
|---|---|---|---|
| Clean timer | How long is one update? | step time, tokens/s/GPU | no attribution |
| XProf | Which JAX or HLO region owns time? | scopes, HLO ops, timeline, memory | some AMD hardware fields are missing |
| `rocprofv3` | What did ROCm dispatch? | kernels, HIP/HSA, copies, RCCL, ROCTx | no automatic JAX source attribution |
| `rocprofv3 --pmc` | What did selected dispatches count? | MFMA, waves, cache counters | perturbs and can serialize execution |
| `rocprof-compute` | Why is one kernel below its roofline? | derived compute, cache, occupancy, resource metrics | replay-based and expensive |

Use the first tool that can answer the current question. A complete run under every
profiler produces more data but does not necessarily produce a better explanation.

### XProf

XProf reads a JAX XPlane capture. Its main advantage is that
`jax.named_scope` names survive into HLO metadata and profiler views, which lets a
kernel be associated with model regions such as `attention` or `mlp`.

Capture only warmed steps:

```python
for _ in range(10):
    state, loss = train_step(state, batch)
    jax.block_until_ready((state, loss))

with jax.profiler.trace("/tmp/xprof"):
    for _ in range(5):
        state, loss = train_step(state, batch)
        jax.block_until_ready((state, loss))
```

Open the capture with:

```bash
xprof --logdir /tmp/xprof
```

The native output is an `.xplane.pb`. Its structure is useful when interpreting
what XProf can correlate:

- **XSpace** is the complete capture.
- **XPlane** represents one device or host component, such as `/device:GPU:0` or
  `/host:CPU`. An eight-GPU process normally contributes eight GPU planes and a
  host plane.
- **XLane** is a timeline within a plane. A GPU plane contains stream lanes for
  device execution and framework-related lanes used for attribution.
- **XEvent** is one event on a lane: a kernel dispatch, operation, API call, or
  user annotation.
- **XStats** are the key/value fields attached to an event, including correlation
  IDs, HLO metadata, and any retained kernel details.

Preserve the `.xplane.pb`. The exported JSON timeline is convenient for Perfetto
but is a flattened, lossy view that does not retain all XProf relationships.

For a first pass, use:

1. Trace Viewer for the step shape, idle gaps, and overlap;
2. HLO or Framework Op Stats for model attribution;
3. Kernel Stats for kernel names and launch geometry; and
4. Memory Viewer for the allocation peak.

<!-- SCREENSHOT NEEDED:
Open the retained capture at
/tmp/llama7b/profile-xprof/trace
and save a Trace Viewer image as pages/img/xprof-llama7b-trace.png.
Show five repeated updates and the Framework Name Scope, Framework Ops, XLA
Modules, XLA Ops, and device stream rows.
-->

### rocprofv3

`rocprofv3` records the ROCm runtime and device execution below XLA. It can capture
kernel dispatches, HIP/HSA calls, memory operations, RCCL, markers, and hardware
counters. It does not know which Python expression caused a dispatch unless the
application or compiler supplied metadata.

The basic form is:

```bash
rocprofv3 <collection-modes> -- <command-to-profile>
```

Useful individual trace modes include:

- `--kernel-trace` for GPU kernel dispatches;
- `--memory-copy-trace` and `--memory-allocation-trace` for data movement and
  allocation;
- `--marker-trace` for ROCTx marks and ranges;
- `--hip-runtime-trace` for HIP runtime calls;
- `--hsa-core-trace` for the lower-level HSA API;
- `--rccl-trace` for RCCL collectives; and
- `--att` or `--advanced-thread-trace` for heavyweight instruction-level thread
  tracing.

Aggregate modes are convenient when the question is still broad:

- `--hip-trace` captures HIP API activity, but not kernel or memory-copy traces;
- `-r` or `--runtime-trace` combines runtime, marker, RCCL, memory, and kernel
  activity; and
- `-s` or `--sys-trace` additionally includes HSA activity.

Output format is selected independently with `-f`:

- `rocpd` is the lossless relational database and should be retained;
- `csv` or `json` is convenient for scripts and quick inspection;
- `pftrace` is a flattened Perfetto timeline; and
- `otf2` is intended for HPC trace viewers.

```bash
rocprofv3 \
  --kernel-trace \
  --marker-trace \
  --stats \
  -f rocpd csv \
  -d /tmp/rocprof \
  -o transformer-block \
  -- python3 workload.py
```

The kernel trace supplies fields that are absent from the current XProf capture,
including VGPR, accumulation-VGPR, SGPR, LDS, scratch, workgroup, and grid sizes.

`--stats` also creates an aggregated `top_kernels` view. For example:

```bash
rocprofv3 --kernel-trace --stats \
  -f rocpd \
  -d /tmp/rocprof \
  -o train-step \
  -- python3 workload.py
```

The resulting database can be queried directly:

```python
import sqlite3

database = sqlite3.connect("/tmp/rocprof/train-step_results.db")
for duration, calls, percentage, name in database.execute(
    "SELECT total_duration,total_calls,percentage,name "
    "FROM top_kernels ORDER BY total_duration DESC LIMIT 10"
):
    print(duration, calls, percentage, name)
```

### What ROCTx can and cannot attribute

ROCTx provides two related but distinct facilities. Marks and ranges add
application-defined timestamps, which `rocprofv3 --marker-trace` records.
Profiler-control calls such as `roctxProfilerPause` and
`roctxProfilerResume` restrict collection to a selected interval.

For opt-in collection, launch `rocprofv3` with `--selected-regions`. Profiling is
then disabled when the process starts and enabled only between
`roctxProfilerResume(0)` and `roctxProfilerPause(0)`. Without
`--selected-regions`, collection starts immediately; pause and resume can hide
intervals only after the application reaches the first control call. This
distinction matters for JAX because initialization and compilation can launch tens
of thousands of kernels before the first training step.

ROCTx and `jax.named_scope` are independent. A named JAX scope flows into HLO and
XProf, but it does not automatically become an ROCTx range.

For a monolithic `jax.jit(train_step)`, Python executes only the call into the
compiled executable. It can push a range before that call, block until the returned
arrays are ready, and pop the range afterwards. This accurately brackets one
complete update, but every kernel launched by that executable lies inside the same
outer range. Python cannot insert separate ROCTx ranges around attention, MLP,
backward, or optimizer operations that execute inside the JIT.

`--kernel-rename` does not recover that missing attribution. It simply replaces
each enclosed kernel's name with the label of the outer range. With one
`train_step` range, all kernels acquire the same name and their original identities
are lost. Do not enable it in the primary kernel trace.

The most useful ROCTx operation in this workflow is therefore coarse collection
control: pause through initialization and warmup, resume for one synchronized
update, then pause again. Use `jax.named_scope`, XProf, and HLO metadata for
operation-level attribution within the compiled module. Deeper ROCTx attribution
would require compiler or custom-call instrumentation and would no longer be a
transparent profile of the same program.

### Hardware counters

List and validate counters on the installed profiler before collecting them:

```bash
rocprofv3-avail list --pmc
rocprofv3-avail -d 0 pmc-check \
  SQ_WAVES \
  GRBM_GUI_ACTIVE \
  SQ_BUSY_CYCLES \
  SQ_INSTS_VALU_MFMA_MOPS_BF16
```

Then collect one compatible group:

```bash
rocprofv3 \
  --kernel-trace \
  --pmc SQ_WAVES GRBM_GUI_ACTIVE SQ_BUSY_CYCLES \
        SQ_INSTS_VALU_MFMA_MOPS_BF16 \
  -- python3 workload.py
```

Counter groups and names depend on architecture and profiler version. In the
controlled one-block counter run below, collection increased the update from
4.48 ms to 7.67 ms. The latter is not an application timing result.

### rocprof-compute

ROCm Compute Profiler collects counter sets and derives metrics for a selected
kernel. Use it only after a trace has identified a kernel and a question.

The workflow has two stages:

1. `profile` replays the selected dispatch as needed and records raw counter sets;
2. `analyze` derives occupancy, cache, compute, memory, and speed-of-light metrics
   from the retained workload directory.

```bash
rocprof-compute profile \
  --output-directory /tmp/rocprof-compute \
  -k <kernel-name-substring> \
  -d 1 \
  --set compute_thruput_flops \
  --no-roof \
  -- python3 workload.py

rocprof-compute analyze \
  --path /tmp/rocprof-compute \
  -b 2.1.2
```

The kernel filter is a substring and the dispatch index is one-based in the current
release. The profiler replays the workload for incompatible counter sets. Its
reported kernel duration and application runtime are not replacements for clean
timing. Use `rocprof-compute analyze --experimental --gui` when the installed
version supports the graphical report, and record the exact syntax because it has
changed between releases.

## Worked Example: A Complete Llama 7B Training Step

A single transformer block is useful for isolating one operation, but it hides
several properties of a real training update: repeated layers, scanned execution,
the vocabulary projection, the full optimizer state, rematerialized backward work,
and competition between many kernel families. The running example is therefore
the complete raw-JAX Llama 2 7B update from the `llama7b` case-study repository.

The workload uses one MI355X and:

```text
decoder layers             32
model width D             4096
SwiGLU width F           11008
query heads                 32
KV heads                    32
head dimension             128
vocabulary              32,000
sequence length           4,096
local sequences               4
tokens per update        16,384
compute dtype              BF16
parameter dtype             FP32
optimizer                  AdamW
attention                    XLA
remat        minimal_with_context
```

There is no inter-device communication in this run. That keeps the first
walkthrough focused on compute, HBM traffic, framework attribution, and kernel
behavior. Chapter 6 introduces distributed arrays and collectives.

For the exploratory XProf capture used below, a disposable copy of the repository
added only `jax.named_scope` boundaries around the model, attention, MLP, loss, and
optimizer. The tensor operations, dtypes, shapes, optimizer, and compiled step were
unchanged. These scopes affect profiler metadata rather than the numerical
computation.

### Workload and predictions

The model contains 6,738,415,616 parameters. Its analytical work at batch four and
sequence length 4,096 is:

```text
learnable-weight work     649.50 TFLOPs
causal-attention work      52.78 TFLOPs
total training work       702.28 TFLOPs
```

The training convention counts one forward contraction and two backward
contractions. It does not count extra work introduced by rematerialization.

At the MI355X dense BF16 ceiling:

$$
t_{\mathrm{compute,min}}
=
\frac{702.28\ \mathrm{TFLOPs}}
     {2{,}516.6\ \mathrm{TFLOP/s}}
=279.1\ \mathrm{ms}.
$$

The local token count is 16,384, far above the approximate 315-token BF16 ridge
point for the large projection GEMMs. Those contractions should be compute-bound
if their local shapes create enough parallel work. The prediction does not apply
unchanged to softmax, normalization, masking, optimizer fusions, or the smaller
projections.

Persistent training state already occupies:

| Item | Dtype | Size |
|---|---:|---:|
| Parameters | FP32 | 25.10 GiB |
| Adam first moment | FP32 | 25.10 GiB |
| Adam second moment | FP32 | 25.10 GiB |
| Persistent total |  | 75.31 GiB |
| Gradient tree, while live | FP32 | 25.10 GiB |

The loss also produces a `[4,4096,32000]` FP32 logits tensor of approximately
1.95 GiB. A fully materialized attention-score tensor is 4 GiB in BF16, or 8 GiB
after promotion to FP32. The XLA attention path can materialize score-related
state that a fused Flash Attention kernel would avoid.

XLA's compiled-memory analysis predicts:

```text
arguments        75.3 GiB
outputs          75.3 GiB
temporaries      92.6 GiB
alias-adjusted  167.9 GiB
```

The prediction is specific: attention and the MLP should dominate the device
timeline; backward should exceed twice the measured forward kernel time because
selective rematerialization recomputes part of the forward pass; no RCCL kernel
should appear; and measured time should exceed the 279.1 ms compute-only floor.

### Clean timing

The raw training loop compiles before entering the loop and synchronizes every
step. Run all 30 updates, discard steps 0–9, and summarize steps 10–29:

```bash
cd /home/clchong/work/llama7b
python3 scripts/precision/jax_bf16.py
```

The working result is:

```text
median step          1.511 s
minimum              1.508 s
maximum              1.516 s
tokens/s/GPU          10,843
useful throughput      464.7 TFLOP/s
MFU                     18.5%
allocator peak         167.94 GiB
```

The runtime peak and compiled-memory estimate agree to the displayed precision.
The update is 5.42 times slower than the compute-only floor. Timing establishes
the size of that gap, but it does not identify whether attention, GEMMs,
rematerialization, optimizer work, memory traffic, or launch overhead is
responsible.

### Locate the cost with XProf

Capture five warmed updates:

```bash
cd /home/clchong/work/llama7b
python3 scripts/profile/xprof.py steps=16

xprof \
  --logdir=/tmp/llama7b/profile-xprof/trace \
  --port=6006
```

The device kernels account for 7.546 seconds across five updates, or 1.509 seconds
per update, matching the synchronized loop timing. The Memory Profile reports a
167.94 GiB peak, independently matching both the allocator and compiled-memory
figures.

The phase attribution from retained HLO operation names is:

| Phase | Kernel-time share |
|---|---:|
| Forward | 27.4% |
| Backward and remat | 66.6% |
| Optimizer | 2.6% |
| Unattributed | 3.4% |

Backward is 2.43 times forward rather than exactly twice forward. That is the
expected signature of a rematerialized training step: model FLOP accounting
charges two backward contractions, while the executable additionally recomputes
selected forward operations.

Kernel Stats makes the next target clear. Across the five updates, the three
largest families are an attention-backward fusion, another attention fusion, and
an XLA-generated attention `dot_general` fusion. Together they account for more
time than any one MLP GEMM family. The named operation paths contain
`attention/self_attn.attend`, while MLP projection kernels retain
`mlp/{gate_proj,up_proj,down_proj}`.

This capture also exposes a limitation rather than hiding it: XProf's Overview page
reports no step time because this raw loop does not emit a framework step marker.
The Trace Viewer, Kernel Stats, HLO metadata, and Memory Profile remain populated.
A missing Overview result is therefore not evidence that the device was idle.

### Inspect the dispatches with rocprofv3

The Llama runner supports `ROCPROF_SELECTED_STEP`. It calls
`roctxProfilerResume(0)` before the selected update and
`roctxProfilerPause(0)` after synchronization. Combine that code with
`--selected-regions`; the flag is what makes collection disabled by default:

```bash
cd /home/clchong/work/llama7b
ROCPROF=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3

ROCPROF_SELECTED_STEP=10 \
$ROCPROF \
  --selected-regions \
  --kernel-trace \
  --stats \
  -f rocpd csv \
  -d /tmp/llama7b-rocprof \
  -o llama7b \
  -- python3 scripts/precision/jax_bf16.py steps=12
```

The selected update contains 3,517 dispatches totalling 1.504 seconds of device
duration. The corresponding synchronized call took 1.505 seconds. No RCCL kernel
appears, as expected for one GPU.

The leading kernel families are:

| Kernel family | Calls | Device-time share | Average dispatch |
|---|---:|---:|---:|
| `fusion_168` | 32 | 17.3% | 8.14 ms |
| `fusion_169` | 32 | 9.7% | 4.54 ms |
| `gemm_fusion_dot_52` | 32 | 8.7% | 4.07 ms |
| `Cijk_...MT256x256x64...` | 192 | 8.4% | 0.66 ms |
| second `Cijk_...MT256x256x64...` | 128 | 6.8% | 0.80 ms |
| `fusion_71` | 32 | 4.9% | 2.30 ms |

The counts of 32 align with the scanned decoder depth and make repeated
per-layer work immediately visible. The `Cijk_...ISA950...` names are
Tensile/hipBLASLt GEMMs and carry their selected macro-tile and matrix-instruction
configuration. For one representative MLP GEMM, `rocprofv3` reports:

```text
grid                  65,536 work-items
workgroup                 256 work-items
workgroups                 256
VGPRs                       112
accumulation VGPRs          384
SGPRs                       112
LDS                     133,120 bytes
scratch                       0 bytes/work-item
```

There are exactly 256 workgroups for 256 CUs. The kernel can place one workgroup
per CU, but its 130 KiB LDS allocation prevents a second equally sized workgroup
from residing on a CU. This is a useful launch-level observation, but it is not yet
a bottleneck diagnosis.

The agreement between XProf's 1.509 seconds of kernels per update and
`rocprofv3`'s independent 1.504 seconds confirms that both tools observed the same
device work. XProf supplies model and HLO attribution; `rocprofv3` supplies
unaltered kernel identities and launch resources.

Running the same command without `--selected-regions` collected approximately
80,000 dispatches and 50 seconds of GPU activity from initialization,
autotuning, compilation-related probes, and all twelve updates. That trace cannot
be interpreted as one training step even though the application calls pause before
the loop.

### Add hardware counters

The next question is whether the selected GEMMs issue BF16 MFMA work and what
resources their launches consume. First validate a compatible counter group:

```bash
rocprofv3-avail -d 0 pmc-check \
  SQ_WAVES GRBM_GUI_ACTIVE SQ_BUSY_CYCLES \
  SQ_INSTS_VALU_MFMA_MOPS_BF16
```

The corresponding full-model command is:

```bash
ROCPROF_SELECTED_STEP=10 \
rocprofv3 \
  --selected-regions \
  --kernel-trace \
  --pmc SQ_WAVES GRBM_GUI_ACTIVE SQ_BUSY_CYCLES \
        SQ_INSTS_VALU_MFMA_MOPS_BF16 \
  -f rocpd csv \
  -d /tmp/llama7b-pmc \
  -o llama7b \
  -- python3 scripts/precision/jax_bf16.py steps=12
```

On the pinned stack this particular combination aborted during profiler
initialization with:

```text
aqlprofile API table load failed: HSA_STATUS_ERROR
```

No model code executed and no full-Llama PMC result was produced. This is a
tool-compatibility failure, not a property of the training step. It also shows why
the trace result and counter result must remain separate evidence: a valid
`rocprofv3` kernel trace does not prove that PMC collection works in the same
container.

To demonstrate the counter workflow without pretending this failure did not occur,
the next subsection uses the existing one-block reproduction. It exercises the same
MI355X, BF16 GEMM path and `Cijk_...MI16x16...ISA950` kernel family while reducing
the number of dispatches. Its result explains that selected kernel only; it is not
substituted back into the complete Llama step as if the workloads were identical.

### Profile one GEMM with rocprof-compute

The complete Llama step is the right workload for locating time and the wrong
workload for an introductory replay-based counter experiment: it launches 3,517
kernels, while `rocprof-compute` is intended to answer a question about one
identified dispatch.

Use `bench/transformer_block.py` as a controlled reproduction of the selected
kernel-level question. It is a one-layer, Llama-3-8B-shaped BF16 block rather than
the Llama 2 7B model above, and it must not be used to claim a full-model speedup.
Its XProf trace identifies a repeated MLP-backward `dot_general`; the corresponding
`rocprofv3` kernel begins:

```text
Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_
MT256x256x64_MI16x16x1_CMS_..._ISA950_...
```

Filter that exact family and its first dispatch:

```bash
KERNEL="Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64"

ROCPROF=rocprofv3 \
ROCPROF_SELECTED_STEP=10 \
rocprof-compute profile \
  --output-directory runs/ch3-rpc-one-gemm \
  -k "$KERNEL" \
  -d 1 \
  --set compute_thruput_flops \
  --no-roof \
  --no-native-tool \
  -- python3 -m bench.transformer_block \
       --strategy dp --devices 1 --tokens 2048 --layers 1 \
       --warmup 10 --repeats 1 --tag ch3-rpc-one-gemm

rocprof-compute analyze \
  --path runs/ch3-rpc-one-gemm \
  -b 2.1.2
```

The kernel takes 252 microseconds and achieves 955 TFLOP/s of BF16 MFMA work,
38.0% of the 2.5166 PFLOP/s peak.

Two additional focused sets explain the launch:

```bash
rocprof-compute profile ... --set launch_stats ...
rocprof-compute profile ... --set mem_thruput ...
```

The derived results are:

| Metric | Result |
|---|---:|
| Grid | 32,768 work-items |
| Workgroup | 256 work-items |
| Total wavefronts | 512 |
| VGPRs | 120 |
| Accumulation VGPRs | 384 |
| SGPRs | 96 |
| LDS | 135,168 bytes |
| Scratch | 0 bytes/work-item |
| BF16 MFMA rate | 955 TFLOP/s |
| BF16 peak fraction | 38.0% |
| LDS bandwidth fraction | 9.5% |
| LDS conflicts/access | approximately 0 |
| L1 utilization | 47.9% |
| L2 utilization | 94.7% |

The grid contains 128 workgroups. The GPU has 256 CUs, so this dispatch cannot place
even one workgroup on every CU. Each workgroup also reserves 135 KiB of the 160 KiB
LDS and 504 combined vector and accumulation registers per lane, preventing another
large resident workgroup on the same CU. There is no scratch spill and no meaningful
LDS bank-conflict signal.

The low MFMA fraction is therefore consistent with limited grid parallelism and a
large per-workgroup resource footprint. High L2 utilization suggests that data
delivery also matters, but the selected metrics do not by themselves prove that L2
is the sole bottleneck.

The native `rocprof-compute` collector failed during HSA initialization in this
container. Re-running with `ROCPROF=rocprofv3 --no-native-tool` succeeded. Treat that
as a version-specific tool workaround, not part of the kernel result.

### Reconcile the evidence

The initial estimate correctly predicted that Llama's large projection GEMMs have
enough arithmetic intensity to sit on the compute side of the HBM roofline. It
could not predict the cost of XLA attention, rematerialized operations, optimizer
fusions, or the launch efficiency of each selected GEMM.

The complete explanation is:

1. the clean Llama 7B update takes 1.511 seconds, reaches 18.5% MFU, and peaks at
   167.94 GiB;
2. XProf accounts for 1.509 seconds of kernels per update, attributes most device
   time to attention and backward/rematerialized work, and independently reports
   the same memory peak;
3. `rocprofv3 --selected-regions` records one update as 3,517 dispatches totalling
   1.504 seconds, exposes repeated 32-layer kernel families, and supplies launch
   resources that XProf does not;
4. ROCTx provides the outer collection boundary only; it does not recreate
   attention, MLP, loss, or optimizer attribution inside `jax.jit(train_step)`;
5. selected-region PMC collection for the full model fails during profiler
   initialization on this pinned stack and therefore supports no counter claim;
   and
6. the controlled one-block reproduction demonstrates the final escalation:
   `rocprof-compute` shows one MLP GEMM at 38.0% of BF16 peak, with insufficient
   grid parallelism for all CUs and a large per-workgroup resource footprint.

The one-block counter result explains its selected kernel, not the entire Llama
gap. The complete model is slower than the compute-only floor because several
large attention fusions, many GEMM families, rematerialized backward work,
optimizer operations, data movement, and dispatch overhead all contribute. The
value of the stack is not that one profiler produces a final answer. It is that
each part of the explanation comes from the abstraction layer that can support it.

## End-to-End Training Performance

The purpose of profiling is ultimately to identify improvement opportunities in
relation to the objective being optimized. For LLM training, the headline is
usually useful tokens per second, constrained by the memory required to fit the
model and by whether the faster configuration still learns correctly.

Report:

- **tokens/s/GPU** and aggregate tokens/s, with non-padding tokens and the batch
  definition stated;
- **step time** from an unprofiled, synchronized run after compilation and
  autotuning;
- **MFU** with the model-FLOP convention and dtype-specific hardware ceiling
  stated;
- **peak HBM** split conceptually into parameters, optimizer state, gradients,
  activations, and executable temporaries;
- **achieved bandwidth or FLOP rate** only at the layer where the relevant bytes
  or operations were actually measured;
- **collective time**, separating total duration, exposed duration, and overlap;
- **compile and autotuning cost**, amortized over the intended run length;
- **loss and convergence evidence** whenever precision, kernels, or mathematical
  execution changed; and
- **run identity**: hardware, partition mode, ROCm/JAX versions, flags, model
  configuration, batch, and measurement method.

The complete publication protocol is
[Appendix B]({{ '/pages/b-measurement-and-convergence-protocol' | relative_url }}).
The key rule is simple: a profile explains a clean timing run; it does not replace
one.

## References

- [JAX Scaling Book: Rooflines](https://jax-ml.github.io/scaling-book/roofline/).
- [JAX Scaling Book: Transformer Math](https://jax-ml.github.io/scaling-book/transformers/).
- [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html).
- [`jax.named_scope`](https://docs.jax.dev/en/latest/_autosummary/jax.named_scope.html).
- [XProf](https://github.com/openxla/xprof).
- [Using `rocprofv3`](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html).
- [Using ROCTx](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html).
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/).
- [ROCm Compute Profiler kernel filtering](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html).
