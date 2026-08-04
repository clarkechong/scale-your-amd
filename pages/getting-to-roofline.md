---
layout: distill
title: "Getting to Roofline"
description: "The prediction says 40% MFU. You measured 22%. A triage order that starts with the cheapest checks, kernel selection and tuning, fusion, which attention implementation you actually got, the escalation path to hardware counters, and why low occupancy is usually not the bug."
date: 2026-08-04

section_number: 8

previous_section_url: "/pages/moe"
previous_section_name: "Chapter 7: Mixture-of-Experts"

next_section_url: "/pages/llama"
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
---

> **Skeleton.** Section structure only; the prose, the before-and-after numbers and the
> worked answers are still to be written. The brief for this chapter is the Chapter 8
> section of `docs/structure.md`.

**Depends on:** [Chapter 3]({{ '/pages/profiling' | relative_url }}) for capturing and
reading a trace, and [Chapter 6]({{ '/pages/training' | relative_url }}) for the
prediction the measurement is failing to meet. The MoE-kernel material also leans on
[Chapter 7]({{ '/pages/moe' | relative_url }}).

> **To write.** Open on the gap in the subtitle, because it is the most common reason
> anyone opens a chapter like this. Then say what the chapter is: not a list of
> optimizations, but **an order to check things in**, cheapest first, because the expensive
> checks are the ones people start with.
>
> **Everything except the MoE-kernel material is readable straight after
> [Chapter 6]({{ '/pages/training' | relative_url }}).** Say so in one line at the top,
> matching the pointer at the end of Chapter 6, so a reader who is training a dense model
> right now and sitting at 22% MFU does not have to get through sparsity first. The chapter
> sits after Chapter 7 because it needs the sparse vocabulary to be *complete*, not to be
> useful.
>
> This chapter has no equivalent in the source book at all, and it is where the deep tooling
> material belongs, because here the reader has a specific question the tool answers.

## A Triage Order

> **To write.** Given a gap between predicted and achieved, what to check and in what
> sequence. Cheapest checks first:
>
> 1. Is the device even busy?
> 2. Are the collectives overlapping? One line only, cross-referencing
>    [Chapter 4]({{ '/pages/sharding' | relative_url }}), which owns that question.
> 3. Is the kernel selection right?
> 4. Is anything falling back to a slow path?
> 5. Is there a bubble?
>
> **Put host starvation first, because it is common and nobody looks for it.** A step-time
> gap caused by the input pipeline looks nothing like a kernel problem and is invisible if
> you only read Kernel Stats, which is where everyone starts.
>
> It also has a nasty interaction with our own tooling: the XProf Input Pipeline page is
> one of the broken views from
> [Chapter 3]({{ '/pages/profiling' | relative_url }}), so the reader cannot use the
> obvious instrument and needs the trace-viewer symptom instead, which is device rows
> going quiet while host rows stay busy. That combination, a frequent cause plus a broken
> detector, is exactly what this chapter exists for.
> [Chapter 9]({{ '/pages/llama' | relative_url }}) shows the fix at production scale.

## Kernel Selection and Tuning

> **To write.** hipBLASLt heuristics and offline tuning, where rocBLAS still gets used,
> Triton on ROCm, and AITER. What to do when the autotuner picks badly, with real
> before-and-after numbers rather than advice.
>
> Composable Kernel gets named as the legacy path rather than the future, since AMD is
> moving AITER off CK templates onto a Python DSL over MLIR. Check the state of that before
> writing and do not present CK as the destination.

## Fusion

> **To write.** What XLA fuses and what it does not, how to read fusion decisions from HLO,
> and the cases worth forcing. Show a fused and an unfused version of the same computation
> and the step-time difference, so the reader can recognise the pattern rather than
> memorise a rule.

## Attention Kernels

> **To write.** Flash-style attention on AMD, what the kernel looks like in a trace, and
> how to tell which implementation you actually got. **This surprises people constantly**,
> which is the reason the section exists.
>
> The concrete JAX-on-ROCm answer to check is `ROCm/jax-aiter`, which bridges AITER's flash
> attention into JAX over XLA FFI with a `custom_vjp` so gradients still flow. That is the
> one place where a JAX user on AMD can reach vendor kernels without going through PyTorch,
> and it is worth a measured before-and-after. It is also alpha, so date the claim.

## When XProf Is Not Enough

> **To write.** The escalation path. `rocprofv3` and rocprof-compute for cache hit rates,
> MFMA utilization, LDS bank conflicts and memory coalescing. TraceLens for large
> multi-node timelines.
>
> **Give concrete handoff recipes rather than tool descriptions:** given a kernel name from
> Kernel Stats, the exact invocation that profiles just that kernel, and how to extract its
> ISA. The `fw101` material feeds this directly.

## What Occupancy Does and Does Not Tell You

> **To write.** This section earns its place defensively: occupancy is the number the reader
> is most likely to misread, and the misreading costs a week.
> [Chapter 3]({{ '/pages/profiling' | relative_url }}) already told them XProf reports
> occupancy as 0 on AMD and to escalate to `rocprofv3`. This is where they arrive holding a
> real occupancy figure, and the useful thing to say is **that low occupancy is usually not
> the bug.**
>
> Keep it to three moves and resist the fourth.
>
> 1. **What the number is.** Resident waves per SIMD over a maximum of 8, set by whichever
>    of four resources runs out first: VGPRs, SGPRs, LDS, or workgroup and barrier slots.
>    Registers are per-SIMD and LDS is per-CU, which is the unit mismatch that makes
>    hand-computed occupancy disagree with the profiler.
> 2. **The result that matters.** On a measured MXFP8 MFMA sweep, eight independent
>    accumulator chains per wave hold roughly 97% of the matrix peak at **12% occupancy**,
>    and beat a two-chain kernel running at 96% occupancy. Occupancy and per-wave tile size
>    compete for one 512-register file, so buying waves costs arithmetic intensity: **the
>    same roofline argument as [Chapter 1]({{ '/pages/rooflines' | relative_url }}), one
>    level down the stack.**
> 3. **What to read instead.** `MfmaUtil` and `VALUBusy` tell you whether the engine is fed,
>    which is the actual question.
>
> **The fourth move, which we do not make, is teaching the reader to tune registers and
> tiles.** That is kernel authoring and it is a stated non-goal. The section's whole job is
> to stop a JAX user from chasing a number they cannot move, since the tile shapes come from
> hipBLASLt or AITER or XLA, not from them. Frame it as triage: if the matrix core is
> already saturated, occupancy is a distraction, stop here.
>
> AMD published a good from-first-principles treatment of this in July 2026,
> [Occupancy Math on the AMD MI355X GPU (CDNA4)](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html),
> which is the right thing to link rather than reproduce. Its CDNA3-against-CDNA4 worked
> example is worth citing directly: the *same* kernel is LDS-bound at 25% occupancy on
> MI300X and register-bound at 50% on MI355X, purely because LDS went from 64 KB to 160 KB
> per CU. That relocation of the bottleneck teaches more than either number does.
>
> Two cautions when using it: its constants are gfx950 throughout, so an MI300X reader needs
> the 64 KB LDS figure substituted, and its headline throughput is an issue ceiling measured
> with register-resident operands and no memory traffic, which the post says plainly and we
> should repeat.

## When to Write Your Own Kernel

> **To write.** And when not to. Mostly not to. Keep this short and make the default answer
> visible from the heading, because the reader who has got this far is in exactly the mood
> to do something expensive.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number. Ship the profile alongside the chapter.

**Question 1:** Given a profile at 22% MFU, produce a ranked list of suspects.

> **To write.** The answer is the triage order applied to a specific artifact, and the
> grading criterion is the *order*, not the list.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 2:** Identify from Kernel Stats whether a matmul got the tuned kernel.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 3:** A kernel reports 18% occupancy and 94% `MfmaUtil`. What should you do?

> **To write.** Nothing. That is the point of the question, and it is the cheapest possible
> way to check the reader took the occupancy section seriously.

{% details Click here for the answer. %}

To write.

{% enddetails %}
