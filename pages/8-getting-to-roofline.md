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
  - name: Fusion
  - name: Attention Kernels
  - name: When XProf Is Not Enough
  - name: What Occupancy Does and Does Not Tell You
  - name: When to Write Your Own Kernel
  - name: Worked Problems
  - name: References
---

> **Draft.** The triage order and the occupancy section are written; the four sections
> that need before-and-after measurements or a verified library survey are not, and are
> marked where they belong. The occupancy material is complete because AMD published the
> measurements it rests on.

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

<!-- BLOCKED: needs a verified library survey plus real before-and-after numbers, and the
     library situation is the fastest-rotting content in the book.

     What it has to deliver, from the roadmap: hipBLASLt heuristics and offline tuning,
     where rocBLAS still gets used, Triton on ROCm, and AITER. What to do when the
     autotuner picks badly. Real before-and-after numbers.

     Specific things to check and pin to a version before writing:
     - Which library XLA:ROCm actually dispatches a bf16 matmul to at the shapes in this
       book, and how that appears in Kernel Stats (Cijk_* names indicate rocBLAS/Tensile
       kernels; hipBLASLt has its own naming).
     - Whether offline GEMM tuning is reachable from JAX on ROCm and what it buys, with
       a measured before-and-after on one shape.
     - Composable Kernel should be named as the legacy path rather than the future, since
       AMD is moving AITER off CK templates onto a Python DSL over MLIR. Check the state
       of that before writing and do not present CK as the destination.

     Everything here needs a "verified against" line with wheel and ROCm versions, the
     same discipline as Chapter 3's limitations table. -->

## Fusion

<!-- BLOCKED on artifacts. Needs real HLO before and after a fusion decision, plus the
     measured effect, and the style guide requires the artifact on screen before the
     explanation.

     What it has to deliver: what XLA fuses and what it does not, how to read fusion
     decisions out of HLO (fusion kinds kLoop, kInput, kCustom, and the operand list),
     and the cases worth forcing. Dump with
     XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_dot".

     Depends on Chapter 3's "From an HLO Op Back to a Python Line" section, which is
     itself blocked: that section teaches how to read an op, and this one assumes it.
     Do not write this before that. -->

## Attention Kernels

<!-- BLOCKED: needs both a verified statement of what is available and a measured
     before-and-after, and it is the section most likely to be out of date by
     publication.

     What it has to deliver: flash-style attention on AMD, what the kernel looks like in
     a trace, and how to tell which implementation you actually got. This surprises
     people constantly, which is the reason the section exists.

     The concrete JAX-on-ROCm answer to check is ROCm/jax-aiter, which bridges AITER's
     flash attention into JAX over XLA FFI with a custom_vjp so gradients still flow.
     That is the one place a JAX user on AMD reaches vendor kernels without going
     through PyTorch, and it deserves a measured before-and-after against XLA's own
     attention lowering. It is also alpha, so the claim must be dated and version-pinned.

     Cross-reference when written: Chapter 5's activation-memory count assumes the score
     matrix is never materialised, which is only true if you got a flash-style kernel.
     That assumption is load-bearing and this section is where it gets checked. -->

## When XProf Is Not Enough

<!-- BLOCKED: needs the escalation recipes verified end to end on the pinned stack.

     What it has to deliver, from the roadmap: rocprofv3 and rocprof-compute for cache
     hit rates, MFMA utilization, LDS bank conflicts and memory coalescing. TraceLens for
     large multi-node timelines. Concrete handoff recipes: given a kernel name from
     Kernel Stats, the exact invocation that profiles just it, and how to extract its ISA.

     Material that feeds it, all local:
     - fw101 has working rocprofv3 invocations (--kernel-trace --stats, --pmc with
       counter names) and rocprof-compute profile/analyze recipes, with committed
       artifacts under fw101/gpu_kernel/prof/.
     - fw101 also has the ISA dump route: GPU_DUMP_CODE_OBJECT=1 to get code objects,
       then llvm-objdump --disassemble-symbols.
     - gpu_profiling/docs/writeup/rocm-pm-sampler-wiring.md documents getting counters
       into XProf itself via XLA_ROCM_PM_SAMPLE_COUNTERS, on a feature branch. That is
       a fix in progress rather than something a reader can use, so it should be
       mentioned as forthcoming rather than recommended.

     Why this is blocked rather than transcribed: the roadmap asks for a recipe keyed to
     a kernel name from Kernel Stats, and the existing invocations profile whole
     processes. The specific handoff has not been done, and a recipe that has not been
     run is exactly the kind of thing that wastes a reader's afternoon.

     Also unverified: whether TraceLens is publicly available and appropriate to
     recommend. Check before naming it. -->

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

**If you do write one, the options on ROCm from JAX are Pallas, which routes through the
Triton backend, or an XLA FFI custom call into HIP.** Pallas on ROCm is labelled
experimental and Mosaic GPU is NVIDIA-only, so the Pallas route is less mature than the
TPU equivalent you may have read about; the FFI route is more work and fewer surprises.

<!-- BLOCKED: the version-pinned status of Pallas on ROCm and of the XLA FFI custom-call
     path. Both statements above are correct as far as we know and neither has been
     tested on the pinned stack in this book. Before publication: write a trivial Pallas
     kernel and a trivial FFI custom call on ROCm 7.2.4 with jax 0.11.0, confirm both
     run, and date the claim. If Pallas on ROCm does not work at all, that is a more
     useful sentence than the hedge above. -->

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
