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
    subsections:
      - name: "Step 1: The Kernel Stats Row"
      - name: "Step 2: The HLO Op"
      - name: "Step 3: The Source Line"
      - name: When the Chain Breaks
  - name: The Matmul, Revisited
    subsections:
      - name: Working the Gap
      - name: The Size Sweep
      - name: What to Take From This
  - name: A Training Step
    subsections:
      - name: What Is In A Step
      - name: The Step Markers, And Why We Are Not Going To Fix The Overview Page
      - name: One GPU Versus Eight
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

> **Verified against:** `rocm/jax-training:maxtext-v26.5`, so `jax` 0.10.0, `jaxlib`
> 0.10.0, `jax-rocm7-pjrt` and `jax-rocm7-plugin` 0.10.0+rocm7.14.0, ROCm 7.14.0,
> XProf 2.23.0, on 8x MI300X (gfx942) in SPX/NPS1. Observations made **5 August 2026**,
> and **every row below was re-measured on this stack rather than carried over** from
> the ROCm 7.2.4 table it replaces. Two rows changed; they are called out underneath.

| Symptom you see | Cause | Workaround | Fix status |
|---|---|---|---|
| Overview and Input Pipeline read "No step time measured", and there is no step timeline | The device plane's events carry no `group_id`, so nothing associates a kernel with a step even when the host recorded one | **None that works.** See the note below: adding `StepTraceAnnotation` does not fix this on ROCm, though you should add it anyway for the other views | Open, and worse than previously documented |
| Roofline reports a peak compute ceiling of 0 GFLOP/s, and labels the axes "per TensorCore" | XProf's peak-FLOP table has NVIDIA entries only, so AMD falls through to zero. The collector says so out loud: `hardware_type_utils.cc: Unsupported device vendor AMD` | Compute the ceiling by hand from [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}), and read the sustained-clock caveat in [The Matmul, Revisited](#the-matmul-revisited) before you do | Fix implemented locally, verified on 8x gfx942, upstream submission pending |
| Kernel Stats shows occupancy, registers per thread and shared memory as 0 for every kernel | The ROCm collector never emits them: the occupancy path is dead code and the register and shared-memory fields are dropped when API and activity records are joined | Escalate to `rocprofv3`, and read [Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }})'s occupancy section before you act on the number | Register and shared-memory fix documented; occupancy still zero |
| "GPU TensorCore utilization" reads 0 on every row, and so does "TensorCore eligibility" | The kernel-name classifier recognises NVIDIA kernel-name patterns only, so MFMA kernels are never counted as matrix work | Ignore both columns | Proposed: add the `Cijk_` and `mfma` patterns |
| Op times on a multi-GPU node are implausibly large | Times are summed across devices without normalising. The same workload measures **0.95x** wall clock on one GPU and **7.07x** on eight | Divide by the device count. **And never quote an op time without saying how many devices it covers** | Proposed |
| Device Compute Precisions reads 0% / 0% | Depends on step markers, so it follows from the first row; the classifier also does not recognise `bf16` | Fix the step markers first | Proposed |
| HBM bandwidth in the device plane reads 2662.4 GB/s, half the data sheet | A 2x undercount inherited from a DDR assumption in the CUDA path. `2662.4 * 2 = 5324.8 GB/s`, which is the 5.3 TB/s spec to three figures | Double it, or just use 5.3 TB/s. The arithmetic is in [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) | Not yet filed |

**Two rows moved since the ROCm 7.2.4 version of this table, and one of them is a
correction to advice we previously gave.**

**The step-marker workaround does not work, and we were wrong to present it as one.**
The previous table said to wrap your step in `jax.profiler.StepTraceAnnotation` and
called the problem yours to fix. It is not. Annotating works exactly as documented on the
host side: a run with the annotation puts ten events named `train` on the host plane, each
carrying a `step_num` stat, and a run without it puts none. **Both runs still report "No
step time measured."** The reason is visible one level down: device-plane events carry
`correlation_id`, `hlo_module`, `hlo_op`, `kernel_details`, `program_id`, `scope_range_id`
and `tf_op`, and **no `group_id`**. That is the field XProf uses to attach a kernel to a
step, the ROCm collector never writes it, and so the annotation cannot reach the device
timeline no matter how correctly you place it.

**Annotate anyway.** Step markers are not what makes the Op Name column useful, and
[From an HLO Op Back to a Python Line](#from-an-hlo-op-back-to-a-python-line) depends on
`jax.named_scope`, which does work. You are giving up the Overview page, not the trace.

**The HBM figure changed from 2479.6 to 2662.4 GB/s, which is a partial fix.** The
previous table attributed the gap to two stacked bugs, a 2x undercount and a
decimal-versus-binary unit mismatch. On this stack the unit mismatch is gone and the
factor is now cleanly 2. Worth knowing if you are comparing against an older capture,
because the number moved without the underlying bug being fixed.

**This is a curated subset, not an exhaustive list.** Our internal audit of the same
stack documents twenty issues; these are the seven a reader hits in their first week.
The ones left out are either cosmetic, or invisible unless you are working on the
collector itself. The one that is worth a sentence, since you can see it yourself:
the device plane describes an MI300X in NVIDIA's vocabulary, reporting
`compute_cap_major: 9, compute_cap_minor: 4` for what is actually gfx942. Nothing
downstream depends on it being wrong, but it is a good illustration of why the rows
above exist. The GPU path was built around one vendor's capability struct and AMD is
being fitted into it.

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

**The profiler tells you a kernel with an unpronounceable name took 551 microseconds. The
question you actually have is which line of your code that was.** This section is the walk
between those two things, in the direction you will really travel it.

The example is a three-line MLP, annotated the way
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}) will annotate everything:

```python
def mlp(x, w1, w2):
    with jax.named_scope("up_proj"):
        h = x @ w1
    with jax.named_scope("activation"):
        h = jax.nn.gelu(h)
    with jax.named_scope("down_proj"):
        return h @ w2
```

### Step 1: The Kernel Stats Row

```
10x  551.37 us  Cijk_Ailk_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT512x128x64_MI16x16x1_...
               op_name = jit(mlp)/up_proj/dot_general
```

**The Op Name column is the whole bridge, and it is the reason to annotate.** Without the
`named_scope` that column would read `jit(mlp)/dot_general` and both of this function's two
GEMMs would be indistinguishable. With it, the row names the block.

**`Cijk_` is a Tensile kernel name, and Tensile backs both rocBLAS and hipBLASLt, so the
prefix alone does not tell you which library you are in.** On this stack the answer is
hipBLASLt: the loaded code object is
`/opt/rocm/lib/hipblaslt/library/gfx942/TensileLibrary_..._Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx942.co`,
which is also where the `Ailk_Bljk_Cijk` in the kernel name comes from. If you need to know
for certain rather than by inference, grep `/proc/<pid>/maps` for `hipblaslt` while the
process runs.

The rest of the name is a tile specification: `MT512x128x64` is the macro tile, `MI16x16x1`
is the MFMA instruction shape it issues.
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) uses these to reason about
occupancy.

### Step 2: The HLO Op

Dump the optimised HLO and find the op whose metadata matches:

```bash
XLA_FLAGS="--xla_dump_to=hlo" python your_script.py
```

```
%cublas-lt-matmul.2 = (bf16[4096,14336]{1,0}, s8[79691776]{0})
    custom-call(%x.1, %w1.1),
    custom_call_target="__cublas$lt$matmul",
    metadata={op_name="jit(mlp)/up_proj/dot_general" stack_frame_id=2},
    backend_config={"gemm_backend_config":{ ... "epilogue":"GELU" ... }}
```

**The anatomy, left to right, because every field earns its place.**

**`%cublas-lt-matmul.2`** is the op's name within the module, assigned by the compiler and
not by you.

**`(bf16[4096,14336]{1,0}, s8[79691776]{0})`** is the output, and it is a tuple because
the real output comes with a scratch buffer: 76 MiB of `s8` workspace the GEMM library
wants. **`{1,0}` is the layout**, a permutation listing dimensions minor-to-major, so
`{1,0}` is ordinary row-major. On a TPU you would see tiling here, something like
`T(8,128)`, and a memory-space annotation. **On AMD you will essentially always see
`{1,0}` or `{0,1}`** and nothing else, which makes the field boring and worth knowing is
boring.

**`custom-call(%x.1, %w1.1)`** gives the operands, which is how you walk backwards through
the graph.

**`custom_call_target="__cublas$lt$matmul"` says cuBLAS on an AMD GPU, and that is not a
mistake.** XLA:ROCm reuses the CUDA path's custom-call target names and routes them to the
ROCm libraries underneath, so a string containing `cublas` means hipBLASLt here. It is
confusing exactly once.

**`stack_frame_id=2`** is the link back to Python.

### Step 3: The Source Line

**The HLO dump carries its own symbol table, at the top of the file:**

```
FileNames
1 "/tmp/hlodemo/demo.py"
FunctionNames
1 "<module>"
2 "mlp"
FileLocations
2 {file_name_id=1 function_name_id=2 line=5 end_line=5 column=12 end_column=18}
StackFrames
2 {file_location_id=2 parent_frame_id=2}
```

Follow `stack_frame_id=2` to `StackFrames 2`, to `FileLocations 2`, to file 1, function
`mlp`, **line 5, columns 12 to 18**. That is `x @ w1`, and the arc is closed: a kernel name
you cannot read, to the expression that caused it.

### When the Chain Breaks

**Three ways it breaks, all of them common.**

**The scope you annotated has no op.** Search the HLO above for `activation` and it is not
there. There is no GELU kernel in Kernel Stats either: two kernels for a three-scope
function. The GELU was folded into the preceding GEMM as `"epilogue":"GELU"`, which is
hipBLASLt applying the activation while the result tiles are still in registers. **This is
the compiler doing its job, and the annotation disappearing is the evidence.** Do not go
looking for the missing 5% of runtime; there isn't any.

**The op is a fusion, and its metadata names only one of the things inside it.** A separate
example, `jnp.tanh(x @ w) * 2.0 + 1.0` then a sum, produces:

```
ROOT %input_reduce_fusion = bf16[4096]{0} fusion(%get-tuple-element.1),
    kind=kInput, calls=%fused_reduce,
    metadata={op_name="jit(f)/convert_element_type" ...}
```

**Four Python operations went in and the metadata names a fifth thing that barely
matters.** `convert_element_type` is a dtype cast the compiler inserted; the tanh, the
multiply, the add and the reduction are all in there too. Follow `calls=%fused_reduce` to
the fusion's own computation to see the whole list. **The rule is that a fusion's metadata
names one contributing op, not the expensive one**, so treat it as a hint rather than an
answer.

**The three fusion kinds tell you what the fusion is shaped like.** `kLoop` is elementwise
work fused into a single pass. `kInput` is what you see above: a reduction with its
producers pulled in. `kCustom` wraps a library call. The kind is the fastest way to guess
what a fusion costs before reading it.

**Nothing named it at all.** If the Op Name column is empty or reads only `jit(f)`, you did
not annotate, and no amount of profiler skill will recover the mapping. Go back and add
`jax.named_scope`. It costs nothing at runtime and it is the difference between this walk
taking two minutes and taking an afternoon.

## The Matmul, Revisited

**[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) predicted 105 microseconds for a
`bf16[4096, 4096]` matmul on one MI300X. It takes 150.** **[measured]**

That is the whole result, and the rest of this section is about the 45 microseconds in
between, because the gap is more instructive than the prediction would have been if it had
landed.

```bash
python -m bench.jax_matmul --trace --sizes 4096
```

**One GPU, deliberately.** An 8-GPU capture sums op times across devices, per the
limitations table above, and this number has to be readable without anyone dividing
anything. Three warmup iterations discarded, ten measured, median of the ten quoted, which
is [Appendix B]({{ '/pages/b-appendix-protocol' | relative_url }})'s protocol.

**Read the kernel duration from Kernel Stats, not the wall clock.** The same run measures
317 microseconds end to end in Python and 150 microseconds on the device. The difference is
dispatch, and at this size it is larger than the arithmetic. This is the first thing the
profiler buys you that a timing loop cannot: a timing loop around a 150 microsecond kernel
is more than half measurement apparatus.

### Working the Gap

**Three candidates, in the order worth checking.**

**Warmup and autotuning: ruled out, and worth about 4%.** The first call compiles and, if
autotuning is on, tries several GEMM implementations; all three warmup iterations are
discarded before the clock starts. Running the same shape at
`--xla_gpu_autotune_level=4` instead of the container's `0` moves the kernel from 157 to
151 microseconds. Real, and far too small to explain the gap. (At 4096 it is 4%. Hold that
thought, because at smaller sizes it is 31%.)

**Clocks: this is most of it.** MI300X's 1307.4 TFLOP/s comes from
`304 CUs * 2048 FLOP/clock * 2.10 GHz`, and 2.10 GHz is a peak boost clock. Watch
`rocm-smi` during 30 seconds of back-to-back matmuls and the device sits at **1590 MHz
drawing 672 W**, against a 750 W cap. It is not throttling in the thermal sense, at 47 °C
junction it has plenty of headroom; it is simply power-limited, which is the designed
behaviour of a 750 W part running dense MFMA.

**Rescale the roofline to the clock the device actually holds and the prediction moves from
105 to 138 microseconds:**

```
304 * 2048 * 1.590e9 = 990 TFLOP/s      # not 1307.4
2 * 4096^3 / 990e12  = 138 microseconds # not 105
```

**Against that number, 150 microseconds is 93% of achievable, and there is very little left
to explain.** **[measured]**

**The kernel that ran: a hipBLASLt Tensile kernel, and the grid is the tell.**

```
Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT256x224x64_MI16x16x1_SN_LDSB1_...
grid 304,1,1   block 256,1,1
```

`MT256x224x64` is the macro tile. A 4096-by-4096 output covers `16 x 19 = 304` tiles, and
the device has exactly 304 CUs, so this launch is **precisely one workgroup per compute
unit and precisely one wave**. That is the best case for tail effects and the worst case
for latency hiding: with one workgroup resident per CU there is no second workgroup to run
while the first waits on memory.

### The Size Sweep

**The same measurement at four sizes, each in a fresh process, autotuning off as the
container ships it:** **[measured]**

| n | Predicted | Measured kernel | Achieved | Fraction of 1307 TFLOP/s | Workgroups on 304 CUs |
|---|---|---|---|---|---|
| 1024 | 1.6 us | 15.0 us | 143 TFLOP/s | 11% | 64 |
| 2048 | 13.1 us | 31.3 us | 549 TFLOP/s | 42% | 256 |
| 4096 | 105.1 us | 157.1 us | 875 TFLOP/s | 67% | 304 |
| 8192 | 841.0 us | 1271.3 us | 865 TFLOP/s | 66% | 1184 |

{% include figure.liquid path="assets/img/matmul-size-sweep.png" class="img-fluid" caption="Achieved bf16 throughput on one MI300X against matrix size, each point measured in its own process. The upper line is the data sheet's boost-clock roofline; the lower one is the same arithmetic at the 1590 MHz the device actually sustains." %}

**The left-hand end of that table is not a bandwidth story, it is an occupancy story.** At
n=1024 the tile size gives 64 workgroups for 304 CUs, so **79% of the machine is idle** and
no amount of arithmetic intensity will help. This is the practical meaning of the ridge
point being about the *problem* rather than about the *hardware*: a matmul with an
arithmetic intensity of 341 FLOPs per byte is nominally compute-bound by a wide margin, and
still reaches 11% of peak, because it never fills the device.

**Autotuning matters most exactly where the kernel is worst.** At n=1024 it finds a
`MT64x96x128` tile instead of `MT128x128x64`, taking 15.0 microseconds down to 10.4, a 31%
improvement. At n=4096 it is worth 4%. The container ships with autotuning off, so **every
GEMM figure in this book is the untuned one**, and small awkward shapes are where you
should expect to pay for that.

### What to Take From This

**The model held.** A 105 microsecond prediction against a 150 microsecond measurement
sounds like a miss until you notice that the entire discrepancy is one number in the
prediction being wrong, and it is not one of the ones we derived. `2 * 4096^3` was right.
The FLOPs-per-clock figure was right. The clock was wrong, because the data sheet quotes a
boost clock and the device runs a sustained one.

**So the correction is not to the method, it is to the constant.** For anything
compute-bound and sustained on MI300X, `990 TFLOP/s` is a better bf16 roofline than
`1307.4 TFLOP/s`, and Chapter 2's number should be read as a ceiling that a real workload
approaches from about 25% below.

**And the honest caveat, which is why the protocol says what it says**: the same kernel
measures 148 to 157 microseconds across five fresh processes, and 168 to 172 microseconds
if the same process measured 1024 and 2048 first, because by then the device has already
dropped its clock. **A number this size is a statement about the device's power state as
much as about the code.** Quote the median, say how many iterations, and say what ran
before it.

## A Training Step

**A matmul is one kernel. A training step is a few hundred, and the skill this chapter
is really teaching is reading the shape of the whole thing rather than any one row.**

The workload is four Llama-3-8B-shaped transformer blocks, 2048 tokens per device, bf16,
with a real Adam update so the step has all three of its phases:

```bash
python -m bench.transformer_block --strategy fsdp --tokens 2048 --trace
python -m bench.transformer_block --strategy dp --devices 1 --tokens 2048 --trace
```

**Start with the single-device capture.** Everything in the limitations table about op
times being summed across devices applies to the eight-GPU one, and there is no reason to
fight that while learning to read a step.

### What Is In A Step

**Group the kernels by which phase of the step emitted them.** Autodiff makes this easy to
do mechanically: JAX names the reverse pass `transpose(jvp(scope))`, so any op whose name
contains `transpose(` is backward work.

| Phase | Share of device time | One GPU, 10 steps |
|---|---|---|
| Forward | 26.1% | 78.5 ms |
| Backward | 50.0% | 150.3 ms |
| No op name at all | 23.9% | 71.7 ms |

**[measured]**

**Backward costs 1.92x forward**, which is the number to remember. The textbook figure is
2x, on the argument that the reverse pass computes both an input gradient and a weight
gradient where the forward computed one output, and the measurement agrees to within 4%.
The same ratio comes out at 1.88 and 1.91 on the other two captures we took, so it is
stable across batch size.

**That third row is not a rounding error and you should not let it pass.** Nearly a
quarter of the device time belongs to kernels XProf shows with an empty Op Name. Look at
what they are and the missing quarter resolves into two different things:

```
22.64 ms   80x  loop_add_convert_fusion_6
11.31 ms   40x  loop_add_convert_fusion_11
10.24 ms   40x  Cijk_Ailk_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT512x144x32_...
 7.12 ms   40x  Cijk_Alik_Bljk_BSS_BH_Bias_HA_S_SAV_UserArgs_MT128x224x64_...
```

**The `loop_add_convert_fusion` kernels are the Adam update**, which is exactly the
elementwise add-and-cast you would write by hand, fused. **The `Cijk_..._BSS_...` kernels
are backward-pass GEMMs** that lost their metadata somewhere in the pipeline; note the
`BSS` where the forward kernels said `BBS`, which is the fp32 accumulation the weight
gradient needs.

**So the honest reading of that table is that forward is about 26%, backward is
*at least* 50% and probably nearer 58%, and the optimizer is a few percent.** Not
"23.9% unaccounted for". The kernel names recover what the op names lost, and this is the
routine version of the walk in
[From an HLO Op Back to a Python Line](#from-an-hlo-op-back-to-a-python-line).

### The Step Markers, And Why We Are Not Going To Fix The Overview Page

**The conventional advice at this point is to add step markers**, and the conventional
demonstration is to show the Overview page reading "No step time measured", add
`jax.profiler.StepTraceAnnotation`, and show it come alive:

```python
with jax.profiler.StepTraceAnnotation("train", step_num=step):
    loss, params, opt_state = train_step(params, opt_state, batch)
```

**On this stack it does not come alive, and we captured it both ways to be sure.** The
annotated run puts ten events named `train` on the host plane, each carrying its
`step_num`; the unannotated run puts none. **Both report "No step time measured."** The
[limitations table](#what-works-today) has the diagnosis: the annotation never reaches
the device timeline because ROCm device events carry no `group_id`.

**Add the annotation anyway**, for the reason given there: it costs nothing, it is correct,
and it will start working when the collector does. What you should not do is spend an
afternoon convinced you have placed it wrong.

### One GPU Versus Eight

**The same four blocks, the same 2048 tokens per device, sharded eight ways with FSDP:**

| | One GPU | Eight GPUs (FSDP) |
|---|---|---|
| Step time | 31.6 ms | 42.4 ms |
| MFU | 26.9% | 20.1% |
| Collectives, share of device kernel time | 0% | 27.7% |
| Time on the wire per step | 0 | 4.8 ms |
| Summed device op time over measured wall clock | 0.95x | 7.07x |

**[measured]**

**The last row is the limitations table made concrete.** On one GPU, summed op time is
0.95 times wall clock, which is just the statement that the device was busy 95% of the
time and is what you would expect the profiler to tell you. On eight GPUs the same
arithmetic gives 7.07, because the tool added up eight devices' worth of work and compared
it against one device's worth of clock. **Neither number is wrong; the second one is
answering a question you did not ask.**

**And the eight-GPU step is slower per device than the one-GPU step**, which is the
correct and slightly deflating result: at 2048 tokens per device this configuration is
below the point where FSDP's communication hides behind its compute.
[Chapter 6]({{ '/pages/6-training' | relative_url }}) is where that threshold gets derived
and tested; for now it is enough to notice that the profiler told us, and that it told us
by putting 27.7% of the device time into kernels with `nccl` in the name.

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
