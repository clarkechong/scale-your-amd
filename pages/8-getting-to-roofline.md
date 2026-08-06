---
layout: distill
title: "Getting to Roofline"
description: "The prediction says 40% MFU. You measured 22%. A triage order for closing that gap, cheapest checks first: host starvation, overlap, kernel selection, fusion, attention implementation, and what to do when the profiler runs out of answers."
date: 2026-08-04

section_number: 8

previous_section_url: "/pages/7-moe"
previous_section_name: "Chapter 7: Mixture-of-Experts"

next_section_url: "/pages/9-llama"
next_section_name: "Chapter 9: Llama 3"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: A Triage Order
  - name: Kernel Selection and Tuning
    subsections:
      - name: What Autotuning Buys, Measured
  - name: Fusion
    subsections:
      - name: Reading Fusion Decisions
  - name: Attention Kernels
    subsections:
      - name: How To Tell Which One You Got
  - name: When XProf Is Not Enough
    subsections:
      - name: The Handoff, Keyed To A Kernel Name
      - name: What Comes Back
      - name: Counters, For When The Static Fields Are Not Enough
  - name: What Occupancy Does and Does Not Tell You
    subsections:
      - name: The Same Argument, On Our Own Kernel
  - name: When to Write Your Own Kernel
    subsections:
      - name: Pallas Works, But Not By Default
  - name: Worked Problems
  - name: References
---

> **Draft.** Every section is written and measured on the pinned stack. What is still
> owed is the two worked problems at the end, which need a profile a reader can load
> rather than a number we can quote.

**Depends on:** [Chapter 3]({{ '/pages/3-profiling' | relative_url }}) for capturing and
reading a trace, and [Chapter 6]({{ '/pages/6-training' | relative_url }}) for the
prediction the measurement is failing to meet. The MoE-kernel material also leans on
[Chapter 7]({{ '/pages/7-moe' | relative_url }}).

**Everything in this chapter except the MoE-kernel material is readable straight after
[Chapter 6]({{ '/pages/6-training' | relative_url }}).** It sits after
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) because it needs the sparse vocabulary to
be *complete*, not to be useful, so if you are training a dense model right now and
sitting at 22% MFU you have not skipped a prerequisite.

You did the arithmetic. You picked a strategy, checked it against the inequalities,
laid out the mesh so the frequent collectives stay on the baseboard, and predicted 40%
MFU. **You measured 22%, and this is the chapter about the factor of two.**

**This is not a list of optimizations. It is an order to check things in.** The order
matters because the cheap checks find most of the problems and the expensive checks are
the ones people start with. Almost everybody opens Kernel Stats first, and the most
common cause of a step-time gap is not in Kernel Stats at all.

## A Triage Order

**Work down this list. Stop when the gap is explained.** Each check is cheaper than the
one after it, and each one has a specific artifact in the profile that answers it.

**1. Is the device even busy?** Look at the device rows in the Trace Viewer across a few
steps. Gaps between kernels, with host rows still active, means the GPU is waiting for
the host, and nothing about your kernels is the problem.

**Put host starvation first, because it is common and nobody looks for it.** A step-time
gap caused by the input pipeline looks nothing like a kernel problem: every kernel in
Kernel Stats is fast, the collectives are overlapped, the sharding is right, and the
step is slow anyway. It also has a nasty interaction with our own tooling, because the
XProf Input Pipeline page is one of the broken views from
[Chapter 3]({{ '/pages/3-profiling' | relative_url }}): it depends on step markers, and
without them it reports "No step time measured" rather than telling you your host is the
bottleneck.

So the obvious instrument is unavailable and you need the trace-viewer symptom instead:
**device rows going quiet while host rows stay busy.** That combination, a frequent cause
plus a broken detector, is exactly what this chapter exists for.

The usual fixes are on the data side rather than the model side: prefetch further ahead,
move preprocessing off the critical path, and check you are not synchronising on the
device every step by calling something that forces a device-to-host copy.
[Chapter 9]({{ '/pages/9-llama' | relative_url }}) covers the production version of this.

**2. Are the collectives overlapping?** One line, because
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}) owns this question: check that
`--xla_gpu_enable_latency_hiding_scheduler=true` and
`--xla_gpu_enable_highest_priority_async_stream=true` are set, then look at whether
compute kernels run concurrently with the RCCL kernel in the Trace Viewer. If a
collective's duration appears as a gap in the compute rows, it is on the critical path
and it should not be.

**3. Does the collective cost what it should?** Take the buffer size out of the HLO,
divide by `β_g = 320 GB/s` per
[Chapter 4]({{ '/pages/4-sharding' | relative_url }})'s cost table, and compare. A
collective within 2x of prediction is fine. A collective 30x over prediction is not a
tuning problem, it is a bug, and
[Chapter 6]({{ '/pages/6-training' | relative_url }})'s second worked problem is exactly
that case.

**4. Is the step doing more FLOPs than you counted?** Three usual suspects, all of which
inflate issued FLOPs without changing the model: rematerialization, which is intended and
which [Chapter 5]({{ '/pages/5-transformers' | relative_url }}) tells you costs 33%;
padding, from a batch or sequence length that does not divide the mesh; and for a sparse
model, the implementation factor from
[Chapter 7]({{ '/pages/7-moe' | relative_url }}), which can be 32x. **Check this before
touching any kernel**, because if you are issuing 4x the FLOPs you thought, your kernels
are fine and your configuration is not.

**5. Did the matmuls get the right kernel?** Now, finally, Kernel Stats. Sort by total
duration, take the top few, compute each one's achieved FLOP rate from its shape and
duration, and compare against
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }})'s peak. See
[Kernel Selection and Tuning](#kernel-selection-and-tuning).

**6. Is anything falling back to a slow path?** Custom calls that did not lower to a
library kernel, an attention implementation that is not the flash-style one you assumed,
an operation running in fp32 because a cast got inserted. See
[Attention Kernels](#attention-kernels).

**7. Is there a bubble?** Only if you are running pipeline parallelism, and
[Chapter 6]({{ '/pages/6-training' | relative_url }}) gives you the expected bubble
fraction to compare against.

**The reason to follow the order rather than your instincts** is that steps 1 through 4
are configuration problems with cheap fixes, and steps 5 onward are kernel problems that
mostly are not yours to fix. Most of the gap, most of the time, is in the first four.

## Kernel Selection and Tuning

**Your bf16 matmul goes to hipBLASLt, and the kernel name will not tell you that.**
[Chapter 3]({{ '/pages/3-profiling' | relative_url }}) walks a GEMM from Kernel Stats back
to source and finds a kernel called
`Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT256x224x64_MI16x16x1_...`. **`Cijk_` is a Tensile name,
and Tensile generates kernels for both rocBLAS and hipBLASLt, so the prefix settles
nothing.**

**The way to settle it is to look at which code object got loaded:**

```bash
grep -iE "blas|tensile" /proc/<pid>/maps
```

On this stack that returns, among others:

```
/opt/rocm/lib/hipblaslt/library/gfx942/TensileLibrary_BB_BB_UA_Type_BB_HPA_
    Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx942.co
```

**hipBLASLt.** `librocblas.so` is mapped too, because it is linked, but its kernel library
is never loaded. The `Ailk_Bljk_Cijk` in that filename is the same contraction descriptor
that appears in the kernel name, which is where the naming comes from and why the two
libraries look alike from the outside. The container sets
`--xla_gpu_enable_cublaslt=True`, which on ROCm means hipBLASLt rather than anything from
NVIDIA.

### What Autotuning Buys, Measured

**The container ships `--xla_gpu_autotune_level=0`, so every GEMM in this book is
untuned unless it says otherwise.** That is worth knowing before you compare any number
here against your own. Running the same shapes at level 4, one process per shape, one
MI300X: **[measured]**

| Matmul | Autotuning off | Autotuning on | Gain |
|---|---|---|---|
| 1024 cubed | 15.0 us | 10.4 us | **1.45x** |
| 2048 cubed | 31.3 us | 30.3 us | 1.03x |
| 4096 cubed | 157.1 us | 151.3 us | 1.04x |
| 8192 cubed | 1271.3 us | 1184.7 us | 1.07x |

**Autotuning is worth 45% on the small awkward shape and 3 to 7% on the large clean
ones**, which is the shape of result you should expect: the heuristic picks a reasonable
tile when the problem is big enough for many tiles to be reasonable, and picks badly when
it is not.

**You can see it picking differently.** At 4096 the untuned choice is a `MT256x224x64`
macro tile and the tuned one is `MT512x112x64`; at 1024 it moves from `MT128x128x64` to
`MT64x96x128`, which is the change that buys the 45%. **The tuned kernels also come from a
different Tensile family**, carrying `Bias_HA_S_SAV` in their names where the untuned ones
do not.

**Two caveats before you turn it on everywhere.** Autotuning is not free: it runs the
candidate kernels at compile time, and in a profile that lands as
`stream_executor::gpu::RedzoneAllocatorKernel` work that will pollute your capture if you
trace the first iteration. And **it is not deterministic across runs**: we saw the same
shape select different kernels on different days at the same autotune level. If you need
reproducible numbers, pin the level and say which.

**On Composable Kernel, briefly, because it is easy to aim at the wrong target.** CK is
the template library underneath much of AITER today, and AMD is moving off CK templates
towards a Python DSL over MLIR. **Treat CK as the current implementation detail rather
than the thing to learn**, and see
[Chapter 7]({{ '/pages/7-moe' | relative_url }})'s survey of what AITER exposes to JAX,
which is less than you would hope.

> **Verified against:** `rocm/jax-training:maxtext-v26.5`, ROCm 7.14.0, `jax` 0.10.0, on
> MI300X (gfx942), **5 August 2026**.

## Fusion

**The cleanest way to see what XLA will and will not fuse is to write the same
elementwise work twice, once with a matmul in the middle.** **[measured]**

```python
def fusible(a):                       # tanh, multiply, add
    return jnp.tanh(a) * 2.0 + 1.0

def unfusible(a, b):                  # the same, around a GEMM
    return jnp.tanh(a @ b) * 2.0 + 1.0
```

| | Kernels launched | Time |
|---|---|---|
| `fusible` | **1**, `loop_add_fusion` | 0.28 ms |
| `unfusible` | **3**, `Cijk_...` plus `wrapped_convert` plus `loop_add_fusion` | 1.67 ms |

**Three elementwise operations became one kernel.** No intermediate `tanh` result is
written to HBM; the chain runs in registers in a single pass over the data. That is the
whole value of fusion, and it is why an elementwise chain costs about what one pass over
its input costs rather than three.

**Putting a GEMM in the middle splits it into three.** The library call is opaque: XLA
cannot see inside hipBLASLt's kernel, so it cannot fuse into it, and the elementwise work
on either side becomes its own launch. **Library calls are fusion barriers**, and that is
the general rule to carry around.

**The exception is the epilogue, and it matters because it hides work.** The example in
[Chapter 3]({{ '/pages/3-profiling' | relative_url }}) puts a GELU immediately after a
matmul and it does not become a separate kernel at all:

```
%cublas-lt-matmul.2 = ... custom-call(%x.1, %w1.1),
    custom_call_target="__cublas$lt$matmul",
    backend_config={"gemm_backend_config":{ ... "epilogue":"GELU" ... }}
```

**hipBLASLt applied the activation while the output tiles were still in registers**, so
the fusion happened inside the library rather than in XLA. Check the `epilogue` field
before concluding an activation was not fused. A missing kernel is the evidence that it
was.

### Reading Fusion Decisions

**Dump the optimised HLO and the fusions name themselves:**

```bash
XLA_FLAGS="--xla_dump_to=hlo --xla_dump_hlo_as_dot" python your_script.py
```

```
ROOT %input_reduce_fusion = bf16[4096]{0} fusion(%get-tuple-element.1),
    kind=kInput, calls=%fused_reduce, metadata={op_name="jit(f)/convert_element_type"}
```

**`kind` is the field to read first.** `kLoop` is elementwise work fused into one pass.
`kInput` is a reduction with its producers pulled in, which is what this one is.
`kCustom` wraps a library call. **The kind tells you the shape of the fusion before you
read a line of it**, and `calls=` points at the computation holding the actual contents.

**Do not trust a fusion's `metadata`.** It names one contributing operation and not
necessarily the expensive one: the fusion above names a dtype conversion the compiler
inserted, while the tanh, the multiply, the add and the reduction are all inside it.

## Attention Kernels

**Three attention implementations, one that runs.** MaxText exposes the choice as a
config value, so this is a clean A/B: same model, same batch, only `attention` changing.
Llama 3 8B, 8x MI300X, 8192-token sequences, per-device batch 4. **[measured]**

| `attention` | Outcome |
|---|---|
| `dot_product` | **Out of memory**, trying to allocate 181.97 GiB |
| `flash` (Pallas) | **Fails to compile**: `Shared memory size limit exceeded: requested 98304, available: 65536` |
| `cudnn_flash_te` | **Runs.** 3.90 s per step, 432.8 TFLOP/s per device |

**Start with the first row, because it is the one that matters for
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}).** That chapter's
activation-memory count assumes the score matrix is never materialised, and it flags the
assumption as load-bearing. **It is load-bearing and it is correct**: ask for the
implementation that materialises scores and the run does not start. A 182 GiB allocation
request against a 192 GiB device is what `B * N * S * S` looks like at
`4 * 32 * 8192 * 8192` in fp32. **You cannot accidentally train at 8k context with a
materialised score matrix, because it does not fit.**

**The second row is an AMD-specific trap and the error message is unusually good.** The
`flash` path is a Pallas kernel whose tiling asks for 96 KB of LDS. **MI300X has 64 KB per
workgroup**, so it fails at compile time with the numbers in the message. NVIDIA parts
from Hopper onward have 228 KB, which is why the kernel was written that way and why it
has not been a problem elsewhere. **This is the single most likely reason a working
MaxText config from an NVIDIA cluster will not start on MI300X.**

**The third row works and is what you should use.** `cudnn_flash_te` routes to
`transformer_engine_rocm_jax`, which despite the config value's name has nothing to do
with cuDNN on this platform; it is AMD's Transformer Engine port with a fused attention
backend. **It needs one extra config key or it refuses to configure**, and the error names
pydantic rather than the cause:

```
Value error, max_segments_per_seq must be set when using TransformerEngine attention
```

Set `max_segments_per_seq=1` if you are not packing sequences.

### How To Tell Which One You Got

**Look for the kernel in the trace, not the config in the log.** With
`cudnn_flash_te` the scope breakdown from
[Chapter 3]({{ '/pages/3-profiling' | relative_url }})'s tooling shows a single
`_FusedDotProductAttention_0` scope taking **15.4% of device kernel time** in the bf16 run.
If you instead see a chain of `dot_general`, `softmax` and `dot_general` under an
`attention` scope, you got the unfused path and your activation memory is not what
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}) says it is.

**One number worth carrying forward.** That 15.4% rises to 19.4% under fp8, not because
attention got slower but because everything else got faster;
[Chapter 6]({{ '/pages/6-training' | relative_url }})'s fp8 section works through why that
caps the achievable speedup near 1.3x.

> **Verified against:** `rocm/jax-training:maxtext-v26.5`, MaxText `release/v26.5` at
> `a7c6c7e5`, `transformer_engine_rocm_jax` 2.15.0.dev0+rocm7.15.0, on 8x MI300X,
> **5 August 2026**. `ROCm/jax-aiter` also exposes flash attention over XLA FFI and would
> be the way to reach AITER's kernels directly; it is not on PyPI and we did not build it,
> so we make no claim about its performance. See
> [Chapter 7]({{ '/pages/7-moe' | relative_url }}) for what its operator list does and
> does not contain.

## When XProf Is Not Enough

**XProf tells you a kernel took 165 microseconds and reports zero for its occupancy, its
registers and its shared memory.** Per
[Chapter 3]({{ '/pages/3-profiling' | relative_url }})'s limitations table, those zeros are
the ROCm collector never writing the fields, so no amount of clicking recovers them.
**`rocprofv3` talks to `rocprofiler-sdk` directly and has them all.**

### The Handoff, Keyed To A Kernel Name

**Take the kernel name from Kernel Stats and filter on it.** The whole point is to profile
that kernel and not the process:

```bash
rocprofv3 --kernel-include-regex "Cijk_" --kernel-trace --stats \
    -d prof -o mm -- python your_script.py
```

**The output is a SQLite database**, which is more useful than it sounds because it means
the follow-up is a query rather than a spreadsheet:

```python
import sqlite3
db = sqlite3.connect("prof/mm_results.db")
db.execute("""
  select s.sgpr_count, s.arch_vgpr_count, s.accum_vgpr_count,
         d.group_segment_size, d.private_segment_size,
         d.workgroup_size_x, d.grid_size_x, count(*), avg(d.end - d.start) / 1000.0
  from rocpd_kernel_dispatch_<guid> d
  join rocpd_info_kernel_symbol_<guid> s on d.kernel_id = s.id
  where s.display_name like 'Cijk%' group by s.id order by count(*) desc
""").fetchone()
```

**Table names carry a per-session GUID suffix**, so list them from
`sqlite_master` rather than hard-coding. And **join dispatches to symbols**: the symbol
table holds every kernel in the loaded code object, which for a Tensile library is
thousands, and only a handful were dispatched.

### What Comes Back

For the 4096-cubed bf16 matmul from
[Chapter 3]({{ '/pages/3-profiling' | relative_url }}): **[measured]**

| Field | Value | XProf shows |
|---|---|---|
| SGPRs | 112 | 0 |
| Architected VGPRs | 128 | 0 |
| Accumulation VGPRs | 352 | 0 |
| LDS per workgroup | **63744 B of 65536** | 0 |
| Scratch | 0 B | 0 |
| Workgroup | 256 threads | correct already |
| Grid | 77824 threads, so **304 workgroups on 304 CUs** | correct already |
| Average duration | 165.2 us | correct already |

**That LDS figure is the answer to a question Chapter 3 left open.** The 4096 matmul runs
exactly one workgroup per compute unit, and the reason is right there: **the kernel uses
97% of the LDS a workgroup can have**, so a second one cannot be resident no matter how
many registers are free. The 480 total VGPRs per thread point the same way. **This kernel
was designed to occupy a CU alone**, trading occupancy for a large tile, which is a
deliberate choice rather than a defect, and the next section is about why that can be the
right one.

**It also explains the failure in [Attention Kernels](#attention-kernels) above.** The
Pallas flash kernel asked for 98304 bytes of LDS. The limit is 65536. A GEMM at 63744 fits
with 1792 bytes to spare, which is how close the working kernels run to the edge.

### Counters, For When The Static Fields Are Not Enough

**`rocprofv3` 1.3.2 on this stack exposes the counters the occupancy discussion needs**,
confirmed present by `rocprofv3 --list-avail`: `OccupancyPercent`, `MfmaUtil`,
`SQ_INSTS_MFMA`, `TCC_HIT_sum`, `TCC_MISS_sum`, `VALUBusy` and `MemUnitStalled`.
**[measured]** Collect them with `--pmc`, and expect the run to be serialised and slow,
because counter collection replays dispatches.

**And the ISA, when you need to see the instruction mix:**

```bash
GPU_DUMP_CODE_OBJECT=1 python your_script.py
llvm-objdump --disassemble-symbols=<kernel> <code-object>
```

**Do this last.** Reading MFMA instruction scheduling is a real skill and it is almost
never where the answer is; the triage order at the top of this chapter exists so that you
arrive here having already ruled out the six cheaper explanations.

> **Verified against:** `rocprofv3` 1.3.2, ROCm 7.14.0, on MI300X (gfx942), **5 August
> 2026**. The recipe above was run end to end and the numbers in the table are its output.
> A separate route, getting counters into XProf itself via `XLA_ROCM_PM_SAMPLE_COUNTERS`,
> exists on a feature branch and is not something a reader can use today.

## What Occupancy Does and Does Not Tell You

**This section earns its place defensively: occupancy is the number you are most likely
to misread, and the misreading costs a week.**
[Chapter 3]({{ '/pages/3-profiling' | relative_url }}) already told you that XProf reports
occupancy as 0 on AMD and that the escalation is `rocprofv3`. This is where you arrive
holding a real occupancy figure, and the most useful thing to say about it is that **low
occupancy is usually not the bug.**

**What the number is.** Occupancy is resident wavefronts per SIMD divided by the maximum
of 8, or equivalently per CU divided by 32. Resident means their registers are live and
reserved. It is set by whichever of four resources runs out first:

```
VGPR limit:      floor(512 / VGPRs-per-lane)        -> waves per SIMD
SGPR limit:      floor(~800 / SGPRs-per-wave)       -> waves per SIMD
LDS limit:       floor(LDS-per-CU / LDS-per-group)  -> workgroups per CU
Workgroup limit: fixed pool of workgroup and barrier slots
```

**The unit mismatch in that list is why hand-computed occupancy disagrees with the
profiler.** The register files are per-SIMD and the LDS is per-CU, so the first two rows
give waves per SIMD and the third gives workgroups per CU. You cannot take a minimum
across different units: convert first, using the workgroup's wave count and the fact
that the hardware spreads a 4-wave workgroup one wave per SIMD.

**Note the two capacities that differ across the family**, per
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}): LDS is 64 KB per CU on MI300X and
160 KB on MI355X, while the 512-entry-per-lane vector register file is the same on both.
AMD's own worked example is that the *same* kernel is LDS-bound at 25% occupancy on
CDNA 3 and register-bound at 50% on CDNA 4, purely because of that LDS change. **That
relocation of the bottleneck teaches more than either number does**, and it is the reason
to compute occupancy rather than look it up.

**Now the result that matters, and it is measured rather than argued.** AMD ran an MXFP8
MFMA microbenchmark on MI355X where each wave runs `K` independent accumulator chains,
with occupancy throttled separately by reserving LDS, so that instruction-level
parallelism and occupancy move on independent axes:

| Independent chains per wave | Throughput at 12% occupancy | Throughput at max occupancy | `MfmaUtil` |
|---|---|---|---|
| 1 | 3.47 PFLOP/s | 4.55 PFLOP/s | 70-95% |
| 2 | 4.65 | 4.67 | 95-96% |
| 4 | 4.46 | 4.69 | 90-96% |
| 8 | **4.82** | **4.83** | ~97% |

> **Source:** AMD's occupancy-math post, MI355X (gfx950), ROCm 7.0.1. **These are AMD's
> measurements, not ours.** The throughput figures are an *issue ceiling*: the
> microbenchmark keeps its operands register-resident and moves no memory, so a real
> HBM-fed GEMM lands lower. What the sweep isolates is the matrix engine itself.

**Eight independent chains hold about 97% of the matrix peak at 12% occupancy, and beat
two chains running at 96% occupancy.** Same chip, same 512-entry register file, two ways
to spend it: on independent accumulator chains, or on more resident waves. The
low-occupancy route wins, at one-eighth the occupancy.

**That is the same roofline argument as
[Chapter 1]({{ '/pages/1-rooflines' | relative_url }}), one level down the stack.**
Occupancy and per-wave tile size compete for one register file, so buying waves costs
arithmetic intensity per wave. The matrix core wants a fixed number of independent
operations in flight; you can supply them with eight waves of one chain each or one wave
of eight chains, and the second route gets there with fewer resources.

**So what should you read instead?** `MfmaUtil` and `VALUBusy`, from `rocprofv3`. They
tell you whether the engines are being fed, which is the actual question. In the sweep
above, `MfmaUtil` reads 70% for one chain and 98% for eight at the same wave count: the
mechanism, not just the outcome.

### The Same Argument, On Our Own Kernel

**AMD's sweep is a microbenchmark. Here is the effect on the plain 4096-cubed bf16 matmul
from [Chapter 3]({{ '/pages/3-profiling' | relative_url }})**, with the resource figures
pulled out of `rocprofv3` by the recipe in
[The Handoff](#the-handoff-keyed-to-a-kernel-name). **[measured]**

```
arch VGPRs 128 + accum VGPRs 352 = 480 per lane
  VGPR limit:      floor(512 / 480)     = 1 wave per SIMD
  SGPR limit:      floor(~800 / 112)    = 7 waves per SIMD
  LDS limit:       floor(65536 / 63744) = 1 workgroup per CU
  workgroup is 256 threads = 4 waves    = 1 wave per SIMD

occupancy = 1 of 8 waves per SIMD = 12.5%
```

**Two separate resources bind at exactly one**, the register file and the LDS, which is
what a deliberately-tuned kernel looks like: hipBLASLt sized the tile so that one
workgroup fills a CU and nothing is left over. The grid confirms it, at **304 workgroups
for 304 compute units, precisely one wave of work**.

**That kernel runs at 70% of the data sheet roofline and about 93% of the roofline at the
clock the device actually sustains.** At 12.5% occupancy.

**This is the section's whole thesis arriving unprompted in an ordinary measurement.** If
you had profiled this matmul, read 12.5% occupancy, and gone looking for a way to raise
it, you would have been optimising a kernel that was already at the roofline. **The number
was low because the kernel was good.**

**And here is the move this section deliberately does not make: it does not teach you to
tune registers and tiles.** That is kernel authoring, and it is a stated non-goal of this
book. **The whole job of this section is to stop a JAX user from chasing a number they
cannot move**, because the tile shapes come from hipBLASLt, AITER or XLA, not from you.
Frame it as triage: **if the matrix core is already saturated, occupancy is a
distraction. Stop here.**

## When to Write Your Own Kernel

**Mostly, do not.** The gap between a competent hand-written kernel and what hipBLASLt
or XLA produces for a standard shape is small, and the gap between a first attempt and
either of them is enormous in the other direction. Every hour spent on a kernel is an
hour not spent on the configuration problems in
[A Triage Order](#a-triage-order), which is where most of the missing throughput is.

**Three situations where the answer changes**, and note that only the third is common:

**When the operation does not exist.** [Chapter 7]({{ '/pages/7-moe' | relative_url }})'s
ragged grouped GEMM is the canonical example: no library on your platform offers it, XLA
generates something generic, and the alternative implementations cost `E / E_a` times the
FLOPs. That is a case where writing a kernel is the only path to the roofline rather than
an optimisation of it.

**When the fusion boundary is wrong and you cannot move it.** If a profile shows an
elementwise chain reading and writing HBM between two matmuls that should have been
fused, and no amount of restructuring the JAX makes XLA fuse them, a hand-written kernel
recovers memory traffic that the compiler is spending.

**When you are AMD, or otherwise in the business of shipping the library.** This is the
common case in practice and it is out of scope here.

### Pallas Works, But Not By Default

**Pallas runs on ROCm, and the default configuration does not.** A trivial kernel:

```python
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

def add_one_kernel(x_ref, o_ref):
    o_ref[...] = x_ref[...] + 1.0

@jax.jit
def add_one(x):
    return pl.pallas_call(
        add_one_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        compiler_params=pltriton.CompilerParams(),   # required on ROCm
    )(x)
```

**Without that `compiler_params` line it fails, and the error tells you exactly what to
do:** **[measured]**

```
ValueError: Mosaic GPU does not yet support AMD ROCm devices.
Use ``compiler_params=pltriton.CompilerParams()`` for ROCm.
```

**With it, the kernel compiles and produces correct results.** Two things follow that are
easy to get wrong. **The Triton backend is bundled inside `jaxlib`**, so you do not need
the standalone `triton` or `jax-triton` packages, and neither is installed in this
container. And **Mosaic is the TPU-and-NVIDIA path**, which is the same wall
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) hits when MaxText's megablox kernels
refuse to lower: those are Mosaic kernels, and no `compiler_params` rescues them because
they are written against a different backend, not merely configured for one.

**The XLA FFI route is present and is what the vendor path uses.** `jax.ffi` exposes
`register_ffi_target` and `ffi_call` on this stack, and
[`ROCm/jax-aiter`](https://github.com/ROCm/jax-aiter) is a working existence proof: it
brings AITER's kernels into JAX over FFI, with `custom_vjp` so gradients flow, and no
PyTorch at runtime. **If you are writing HIP anyway, this is the better-supported of the
two routes**; Pallas is the one to reach for when you want to stay in Python.

> **Verified against:** `jax` 0.10.0, `jaxlib` 0.10.0, ROCm 7.14.0 on MI300X (gfx942),
> **5 August 2026**. `triton` and `jax-triton` are absent from the container and Pallas
> works regardless. Both statements are the kind that rot fastest in this book.

## Worked Problems

<!-- BLOCKED: both problems from the roadmap are read-a-profile exercises and need a
     published trace.

     1. Given a profile at 22% MFU, produce a ranked list of suspects. This one is
        nearly writable from the triage order alone and should be written as soon as
        there is any real profile to hang it on; the answer is the triage order applied
        to a specific artifact.
     2. Identify from Kernel Stats whether a matmul got the tuned kernel. Blocked on the
        Kernel Selection section, which is itself blocked on the library survey.

     A third problem worth adding, which is writable now and would pair well with the
     occupancy section: given an occupancy figure and an MfmaUtil figure, decide whether
     occupancy is worth pursuing. The answer is the table above. Not written yet because
     the section it tests is the only written one, and a chapter whose only problem tests
     its only section reads oddly. -->

## References

**Occupancy and the matrix core.**

- [Occupancy Math on the AMD MI355X GPU (CDNA4)](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html)
  (AMD, 2026). The four limiters, the per-SIMD versus per-CU unit conversion, the
  CDNA 3 to CDNA 4 bottleneck-relocation example, and the ILP-versus-occupancy sweep
  quoted above. The right thing to read rather than have reproduced.
- [AMD CDNA 4 architecture whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
  (AMD). The register file, LDS and matrix-core changes that the occupancy math depends
  on.
- [AMD Instinct MI300/CDNA3 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
  (AMD). Where the 512-total-VGPR rule and the MFMA instruction encodings are specified.

**Profiling tools, for the escalation path.**

- [ROCm Systems Profiler and rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/index.html)
  (AMD). Kernel traces, PMC counters, and the derived metrics including
  `OccupancyPercent`, `MfmaUtil` and `VALUBusy`.
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/)
  (AMD). The roofline and memory-hierarchy analysis layer above rocprofv3.

**Kernel libraries, for the sections still to be written.**

- [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/) (AMD). The GEMM
  library XLA dispatches to, and where offline tuning lives.
- [AITER](https://github.com/ROCm/aiter) (AMD). AMD's fast kernels, including flash
  attention and grouped GEMM.
- [ROCm/jax-aiter](https://github.com/ROCm/jax-aiter) (AMD). The XLA FFI bridge that
  makes some of AITER reachable from JAX.
- [Pallas on ROCm](https://docs.jax.dev/en/latest/pallas/index.html) (JAX). The
  kernel-authoring path from JAX, via the Triton backend on AMD.
