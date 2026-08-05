---
layout: distill
title: "How to Profile AMD GPU Programs"
description: "Your model is slower than the arithmetic says it should be. Where did the time go? Capturing a trace with the ROCm JAX stack, reading it in XProf, following a slow kernel back to the Python line that emitted it, and an honest account of which parts of the tooling do not work yet."
date: 2026-08-04

section_number: 3

previous_section_url: "/pages/2-amd-gpus"
previous_section_name: "Chapter 2: AMD GPUs"

next_section_url: "/pages/4-sharding"
next_section_name: "Chapter 4: Sharding"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: A Thousand-Foot View of the Stack
  - name: Setup
  - name: Your First Trace
  - name: The XProf Tools
    subsections:
      - name: Trace Viewer
      - name: Graph Viewer
      - name: Op Profile
      - name: Kernel Stats
      - name: Memory Profile
      - name: Roofline
  - name: What Works Today
    subsections:
      - name: How a Profile Gets Made
  - name: From an HLO Op Back to a Python Line
  - name: The Matmul, Revisited
  - name: A Training Step
  - name: Worked Problems
  - name: References
---

> **Draft.** The sections that need a screenshot or a captured number are not written
> yet and are empty rather than wrong: the XProf tool tour, the HLO traversal, the
> matmul measurement and the training step. What is here is the part that stands
> without them, which includes the limitations table most readers arrive for.

**Depends on:** [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) for the MI300X
constants and for the matmul whose predicted time we are about to check. Nothing
later.

[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) ended on a cliffhanger. It
predicted 105 microseconds for a `4096^3` bf16 matmul and then had no way whatsoever
to find out whether that was true. **This chapter is the instrument, and the reason it
sits here rather than after the parallelism chapters is that from
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}) onward every claim in this
book is checked against a profile.** You cannot run the experiments without the
apparatus.

**If you are impatient, here is the minimum.** [Setup](#setup) and
[Your First Trace](#your-first-trace) are what
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}) onward actually depends on, and
[What Works Today](#what-works-today) is the table you will come back to. The tool
tour can be skimmed and returned to when you need a specific view. Nothing after that
is load-bearing until
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}), which is where the
escalation path to hardware counters and ISA lives, because that is where you finally
have a reason to care about them.

**The tone of this chapter is candid rather than apologetic.** Several XProf views
report zeros on AMD. Most readers who arrive here have already seen one of those zeros
and assumed they misconfigured something, and being told plainly that the field is
never written is worth more than a workaround.

## A Thousand-Foot View of the Stack

**Between the Python you wrote and the kernel that runs there are five
representations, and knowing their names is most of what it takes to read a
profile.** The profiler shows you the last two; your code is the first; the middle
ones are where a surprising amount of the performance is decided.

- **JAX**, the Python you wrote. `jax.jit` traces your function into a graph rather
  than running it.
- **StableHLO**, the portable serialisation of that graph. This is the interchange
  format, and the level at which the program is still recognisably your code.
- **HLO** (High Level Optimizer IR), XLA's internal representation, and the one that
  matters most for profiling. **This is where the optimisations you can actually
  observe happen**: operator fusion, which merges elementwise work into its
  neighbouring matmul; layout assignment, which chooses physical memory layouts for
  each buffer; and scheduling, which decides what overlaps what. Every op name you
  see in a profile is an HLO op name.
- **LLVM IR and ROCDL**, where the fused HLO becomes device code.
- **GCN ISA**, the actual instructions, including the `v_mfma_*` matrix instructions
  that produce every FLOP in this book.

**The AMD-specific part of the picture is where the plugin sits, and why ROCm ships
two wheels rather than one.** JAX talks to hardware through PJRT, a C API that
separates the framework from the backend. On ROCm that backend arrives as a pair:

- **`jax-rocm7-pjrt`** is the PJRT plugin proper, the shared library implementing the
  device API against HIP.
- **`jax-rocm7-plugin`** is the Python-side registration package that tells JAX the
  plugin exists and how to load it.

They are versioned together and they must match `jaxlib`. Installing three of the four
and forgetting the fourth is the single most common way to end up with
`jax.default_backend() == "cpu"` on a machine with eight GPUs in it, which is why the
verification snippet below asserts on the backend rather than trusting it.

<!-- BLOCKED: the two XLA pipeline diagrams (fw101 `xla-gpu-pipeline` and
     `xla-hlo-to-thunk`) drop in here almost unchanged, per the roadmap. They are
     figures, not prose, and need exporting from /root/work/fw101 into
     assets/img/ before they can be referenced. Nothing else in the chapter depends
     on them, so the section reads without them for now. -->

## Setup

**Pull the container. That is the whole recommendation.** AMD publishes prebuilt JAX
images with ROCm, JAX, jaxlib and both plugin wheels already pinned to versions that
work together, which is one string you can quote instead of four versions you have to
reconcile.

```bash
docker pull rocm/jax:latest

docker run -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --ipc=host --shm-size 64G \
  rocm/jax:latest /bin/bash
```

Confirm you are on the GPU you think you are on:

```bash
rocm-smi --showproductname
```

**Map the architecture string, because it is the thing every version-dependent claim
in this book is keyed to.** `gfx942` is MI300-class (MI300X, MI325X) and `gfx950` is
MI350-class (MI355X). This matters immediately: the fp8 format differs between them,
per [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}).

Then verify the whole chain end to end, in Python, rather than trusting the image:

```python
import jax, jaxlib, jax.numpy as jnp

print("jax   ", jax.__version__)
print("jaxlib", jaxlib.__version__)

devs = jax.devices()
print("backend", jax.default_backend())
print("devices", len(devs), devs[:2])
assert jax.default_backend() == "rocm", "ROCm plugin NOT active"
assert devs and devs[0].platform == "rocm"

# Compile and run something real, which exercises XLA and PJRT rather than imports.
x = jnp.arange(8.0)
y = jax.jit(lambda a: (a * a).sum())(x)
y.block_until_ready()
assert float(y) == 140.0
print("OK: full chain works")
```

**The `block_until_ready()` is not decoration.** JAX dispatches asynchronously, so
without it you are timing the enqueue rather than the work, and a broken backend can
look fine right up until you ask for the value. Every measurement in this book blocks
before stopping the clock.

**Everything else about installation is in
[Appendix A]({{ '/pages/a-appendix-install' | relative_url }})**: the four wheels in
the right order, the ROCm version matrix, building from source, and the
known-broken combinations. It is necessary, it is nobody's reason for reading this
book, and it rots faster than any other page here.

## Your First Trace

**Wrap the computation in `jax.profiler.trace` and you have a profile.** There is no
launcher to configure and no separate profiler binary to run.

```python
import jax, jax.numpy as jnp

A = jnp.ones((4096, 4096), dtype=jnp.bfloat16)
B = jnp.ones((4096, 4096), dtype=jnp.bfloat16)
matmul = jax.jit(lambda a, b: a @ b)

# Warm up: the first call compiles, and may autotune. Never profile it.
for _ in range(3):
    matmul(A, B).block_until_ready()

with jax.profiler.trace("/tmp/traces/jax_matmul_trace"):
    for _ in range(10):
        matmul(A, B).block_until_ready()
```

**Three iterations of warmup and ten of measurement, and we report the median.** The
first call compiles and may run kernel autotuning, and the next few are still ramping
clocks, so a mean over everything is a number about startup rather than about steady
state. This is the protocol for the whole book and it is written out in
[Appendix B]({{ '/pages/b-appendix-protocol' | relative_url }}).

**Write every trace under `/tmp/traces/<workload>/`**, because a single `--logdir`
then picks all of them up as selectable runs and you can compare two configurations
without moving files around. The layout the profiler writes:

```
/tmp/traces/                              <- pass this to --logdir
└── jax_matmul_trace/                     <- one directory per workload
    └── plugins/
        └── profile/
            └── 2026_07_14_17_23_15/      <- one directory per capture
                ├── hostname.trace.json.gz    <- Chrome trace, for Perfetto
                └── hostname.xplane.pb        <- XSpace proto, for XProf
```

**Two files, two audiences.** The `.trace.json.gz` opens in Perfetto or
`chrome://tracing` and is useful when you want a raw timeline and nothing else. The
`.xplane.pb` is the XSpace protocol buffer that carries the structured event data, and
it is what XProf reads to build every view in the next section. When a view in XProf
shows a zero, the question is always whether the field was written into this file.

Launch the viewer:

```bash
xprof --logdir=/tmp/traces --port=6006
```

Then open the forwarded port. If you are working over VS Code Remote or a dev
container, port 6006 is forwarded automatically through the PORTS panel and you do not
need an `ssh -L` tunnel.

## The XProf Tools

<!-- BLOCKED ON SCREENSHOTS. Every subsection below needs a real screenshot of the
     view, against the pinned stack in "What Works Today", before it can be written:
     the style guide requires a concrete artifact on screen before each explanation,
     and a tool tour without artifacts is exactly the feature-list prose the guide
     forbids.

     What each subsection has to deliver, from the roadmap:
     - Trace Viewer: most space. The XLA Ops row as the real hardware timeline and
       everything below it as the approximate trace built from jax.named_scope and
       the Python stack. Include the video-game navigation keys (w/a/s/d).
     - Graph Viewer: reading the HLO graph around a fusion, and the op-name search.
     - Op Profile: the tree by self-time, and the IDLE row, which on our 8-GPU AMD
       traces reads 73.9% against 38.7% on 4x H100 and is misleading without step
       markers (see limitations rows 1 and 5).
     - Kernel Stats: the columns that work (name, duration, occurrences, grid/block)
       and the four that read 0 on AMD (occupancy, registers, shared memory,
       TensorCore utilization).
     - Memory Profile: peak allocation over time and the largest buffers, which is
       the view that answers "why did this OOM at batch 9".
     - Roofline: currently unusable on AMD until the peak-FLOP fix lands; explain
       that here and point at the limitations table rather than teaching the view.

     Decision needed before writing (recorded in docs/writing-notes.md): are the two
     dozen screenshots captured by hand or scripted against XProf's data endpoints?
     This is the largest unplanned cost in the chapter and the most likely thing to
     stall it. -->

### Trace Viewer

### Graph Viewer

### Op Profile

### Kernel Stats

### Memory Profile

### Roofline

## What Works Today

**Several XProf views report zeros on AMD, and none of it is your configuration.**
This is the section people will be linked to directly, so it is written to be read
cold.

> **Verified against:** `jax` 0.11.0, `jaxlib` 0.11.0, `jax-rocm7-pjrt` 0.11.0,
> `jax-rocm7-plugin` 0.11.0, ROCm 7.2.4, XProf 2.22.3, on 8x MI300X (gfx942) in SPX
> partitioning mode. Observations made **14 July 2026**. Software behaviour changes
> every few months and three of these rows have fixes in progress, so treat any row
> whose status column is not empty as likely to have moved.

| Symptom you see | Cause | Workaround | Fix status |
|---|---|---|---|
| Overview and Input Pipeline read "No step time measured", and there is no step timeline | Nothing in the trace marks a step boundary, and the GPU path does not infer them | **You fix this**: wrap your step in `jax.profiler.StepTraceAnnotation` | Yours to fix, not a bug |
| Roofline reports a peak compute ceiling of 0 GFLOP/s, and labels the axes "per TensorCore" | XProf's peak-FLOP table has NVIDIA entries only, so AMD falls through to zero, and the frontend decides "is this a GPU" by string-matching "Nvidia GPU" | Compute the ceiling by hand from [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) | Fix implemented locally, verified on 8x gfx942, upstream submission pending |
| Kernel Stats shows occupancy, registers per thread and shared memory as 0 for every kernel | The ROCm collector never emits them: the occupancy path is dead code and the register and shared-memory fields are dropped when API and activity records are joined | Escalate to `rocprofv3`, and read [Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }})'s occupancy section before you act on the number | Register and shared-memory fix documented; occupancy still zero |
| "GPU TensorCore utilization" reads 0 on every row | The kernel-name classifier recognises NVIDIA kernel-name patterns only, so MFMA kernels are never counted as matrix work | Ignore the column | Proposed: add the `Cijk_` and `mfma` patterns |
| Op times on a multi-GPU node are implausibly large: our 8-GPU traces show roughly 115 s of device op time against roughly 15 s of wall clock | Times are summed across devices without normalising | Divide by the device count. **And never quote an op time without saying how many devices it covers** | Proposed |
| Device Compute Precisions reads 0% / 0% | Depends on step markers, so it follows from the first row; the classifier also does not recognise `bf16` | Fix the step markers first | Proposed |
| HBM bandwidth in the device plane reads 2479.6 GB/s, less than half the data sheet | Two stacked bugs: a 2x undercount inherited from a DDR assumption in the CUDA path, and a decimal/binary unit mismatch in the display | Use 5.3 TB/s. The arithmetic is in [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) | Not yet filed |

**This is a curated subset, not an exhaustive list.** Our internal audit of the same
stack documents twenty issues; these are the seven a reader hits in their first week.
The ones left out are either cosmetic, or invisible unless you are working on the
collector itself. Notably absent: a row about NVIDIA-shaped capability structs, which
is real and which nobody outside the profiler team will ever notice.

**And we are on both sides of this table**, which is the strongest argument for
pinning it to a version. Three of these rows have fixes in progress in our own tree,
so an undated table of broken things would be quoted back at us after the bugs were
fixed.

### How a Profile Gets Made

**The reason to know the pipeline is that it makes the zeros above legible.** Once you
know a field has to be written by the collector before XProf can read it, a blank
column stops looking arbitrary and starts looking like a specific missing write.

Four stages, from the hardware up:

1. **`rocprofiler-sdk`** delivers callbacks as HIP API calls are made and as kernels
   actually execute on the device. Two separate streams: what the host asked for, and
   what the device did.
2. **The ROCm tracer** turns those callbacks into neutral `RocmTracerEvent` records,
   and joins the API-side record to the activity-side record for the same dispatch by
   correlation ID. **This join is where several of the zeros above happen**: if a
   field lives on the API record and the join keeps only the activity record, the
   field is gone before anyone tries to display it.
3. **The ROCm collector** converts events into XPlane structures: one plane per
   device, lines for kernel and memory activity, and XStats attached to each event for
   things like grid dimensions and occupancy. Device-level properties, including the
   core count, clock rate and the HBM bandwidth figure discussed in
   [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}), are attached here as plane
   metadata. It then serialises the whole XSpace to `*.xplane.pb`.
4. **XProf** reads that file and derives every view from it: op statistics, the kernel
   table, the roofline. It computes nothing from the hardware directly, which is why a
   missing XStat is unrecoverable at this stage and why the fix for a broken view is
   almost always in the collector rather than in the UI.

**The one-line diagnostic that follows from this:** if a view is blank, dump the
XPlane and look for the stat. If the stat is absent, no amount of clicking in XProf
will help, and the escalation is `rocprofv3`, which talks to `rocprofiler-sdk`
directly and skips stages 2 through 4 entirely.
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) covers that path.

<!-- BLOCKED: the pipeline diagram from gpu_profiling/docs/xla-rocm-profiler-backend.md
     belongs here as a figure. Same asset-export dependency as the XLA pipeline
     diagrams above. The prose stands without it. -->

## From an HLO Op Back to a Python Line

<!-- BLOCKED: needs a real HLO op and a real kernel name pulled from a captured
     trace, and the arc only works in the direction the reader travels, so it cannot
     be written from memory.

     What it has to deliver, from the roadmap: one continuous arc from a slow row in
     Kernel Stats, to the HLO op that emitted it, to the jax.named_scope annotation,
     to the source line, plus what to do when the chain breaks. Teach the anatomy of
     the op along the way: name, output shape and dtype, the plain major-to-minor
     layout, operands. Then the AMD specifics: fusion kinds (kLoop, kInput, kCustom),
     custom calls into hipBLASLt and rocBLAS and how they surface as `Cijk_*` kernels
     in Kernel Stats, and async collective pairs.

     Do NOT port the source book's worked example: its op carries TPU-specific
     `T(8,128)(2,1)` tiling and an `S(1)` memory-space annotation, neither of which
     exists on AMD. Write it fresh against an op from the jax_matmul trace, following
     the toy-case-first move.

     Unblock by: capturing jax_matmul.py and transformer_block.py on the pinned stack
     and pasting the real op text and the real Cijk_ kernel name. fw101's
     "tracing upwards from kernel to python" material feeds the traversal. -->

## The Matmul, Revisited

<!-- BLOCKED on the measurement. Chapter 2 predicts 105 microseconds for
     2 * 4096^3 / 1307.4e12 on one MI300X, and this section closes that loop:
     expected against measured, stated plainly whether the model held, and if it did
     not, why. The candidates to check in order are warmup and autotuning (the first
     call is not comparable), clock throttling on a 750 W part, and whether the
     kernel that ran was a hipBLASLt Cijk_ kernel or something slower.

     Unblock by: running scripts/jax_matmul.py (already the right shape, bf16
     4096x4096, 3 warmup + 10 profiled iterations) on 1 MI300X, reading the median
     kernel duration from Kernel Stats, and dividing by nothing at all, since this is
     a single-device capture and the 8x summing bug in the limitations table does not
     apply.

     Note when writing: 105 microseconds is short enough that launch overhead and
     clock ramp are comparable to the work, so quote the median of the ten profiled
     iterations and say so. -->

## A Training Step

<!-- BLOCKED on the capture, and it needs to be captured twice.

     What it has to deliver: forward, backward and optimizer in the Trace Viewer, and
     the introduction of jax.profiler.StepTraceAnnotation. Show it in the honest
     order: capture without the annotation first, hit the "No step time measured"
     wall from the limitations table, then add the annotation and recapture so the
     Overview page comes alive. Most readers have already seen the broken version, so
     leading with it is the correct ordering rather than a rhetorical device.

     Unblock by: running scripts/basic_training.py and scripts/transformer_block.py,
     which already profile steps 10-20, once with and once without the annotation.

     Watch out for: transformer_block.py runs on 8 GPUs, so every op time in that
     capture is summed across devices. Either capture on one device for this section
     or divide, and say which. -->

## Worked Problems

<!-- BLOCKED: all three problems are read-a-trace exercises, so they need the trace
     to exist and be published alongside the chapter before they can be set.

     From the roadmap: given a trace, identify which of two matmuls is memory-bound
     and say why; find the kernel with the largest self-time and trace it back to its
     Python line; explain a device-op-time figure that is 8x the wall clock. The
     third one is answerable from the limitations table alone and could be written
     first if the other two stay blocked.

     Also unresolved: whether the example traces can be published at all, since the
     committed ones carry internal hostnames. See docs/writing-notes.md. -->

## References

**Tooling.**

- [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html).
  `jax.profiler.trace`, `StepTraceAnnotation` and `named_scope`, which are the three
  APIs this chapter uses.
- [XProf](https://github.com/openxla/xprof). The profiler UI that reads the XSpace
  protos, formerly the TensorBoard profiler plugin.
- [ROCm Compute Profiler and rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/index.html).
  The layer underneath, and the escalation path when a field XProf wants was never
  written. Used in earnest in
  [Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}).

**Installation and versions.**

- [Install JAX for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html)
  (AMD). The container tags and the four-package wheel install.
- [JAX on ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/jax-compatibility.html)
  (AMD). Which plugin version goes with which ROCm version, which is the mapping the
  limitations table above is pinned against.
- [ROCm/rocm-jax](https://github.com/ROCm/rocm-jax). Sources for the ROCm JAX plugin
  and the Dockerfiles behind the `rocm/jax` images.

**Background.**

- [XLA architecture](https://openxla.org/xla/architecture) and
  [HLO operation semantics](https://openxla.org/xla/operation_semantics). Reference
  for the op names and fusion kinds that appear in every profile.
