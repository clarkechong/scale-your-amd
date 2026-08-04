---
layout: distill
title: "How to Profile AMD GPU Programs"
description: "Your model is slower than the arithmetic says it should be. Where did the time go? Capturing a trace with the ROCm JAX stack, reading it in XProf, following a slow kernel back to the Python line that emitted it, and an honest account of which parts of the tooling do not work yet."
date: 2026-08-04

section_number: 3

previous_section_url: "/pages/amd-gpus"
previous_section_name: "Chapter 2: AMD GPUs"

next_section_url: "/pages/sharding"
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
---

> **Skeleton.** Section structure only; the prose, the screenshots and the worked
> answers are still to be written. The brief for this chapter is the Chapter 3 section
> of `docs/structure.md`.

**Depends on:** [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) for the MI300X
constants and for the matmul whose predicted time we are about to check. Nothing
later.

> **To write.** Open on the cliffhanger, because that is what makes this chapter feel
> like the next step rather than a detour.
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) predicted a time for a 4096
> cubed matmul and had no way to find out whether it was right. This chapter is the
> payoff. Say that explicitly in the first paragraph, and note that from
> [Chapter 4]({{ '/pages/sharding' | relative_url }}) onward every claim in the book is
> checked against a profile, so the instrument has to be in hand before the experiments
> start.
>
> **Tell the impatient reader what the minimum is.** This is the one chapter a scaling
> reader might resent, because it is tooling and it sits between them and any
> parallelism content. So name the short path here: *Your First Trace* and
> *What Works Today* are what Chapter 4 onward actually depends on, the tool tour can
> be skimmed and returned to, and nothing after that is load-bearing until
> [Chapter 8]({{ '/pages/getting-to-roofline' | relative_url }}).
>
> Scope is "capture a trace and read it". The escalation path to hardware counters and
> ISA lives in [Chapter 8]({{ '/pages/getting-to-roofline' | relative_url }}), where the
> reader has a reason to care.

> **This chapter is the biggest length risk in the book and should be actively
> resisted while writing.** We have more material here than anywhere else, which is
> exactly why it will bloat. Three economies are already built into the structure
> below: setup shrinks to a container pull, the profiler's internals collapse into the
> limitations section that needs them, and the two "read the compiler's output"
> sections merge into one. If a section does not help the reader answer "where did my
> time go", it belongs in an appendix or in Chapter 8.

## A Thousand-Foot View of the Stack

> **To write.** JAX to StableHLO to HLO to LLVM and ROCDL to GCN ISA, with the passes
> that matter called out: fusion and layout assignment. Then where the PJRT plugin sits
> and why ROCm ships two wheels rather than one.
>
> The `fw101` PJRT section and the two XLA pipeline diagrams
> (`xla-gpu-pipeline`, `xla-hlo-to-thunk`) drop in here almost unchanged. Copy the
> images into `assets/img/` first, then uncomment.

{% comment %}
{% include figure.liquid path="assets/img/xla-gpu-pipeline.png" class="img-fluid" caption="<b>Figure:</b> the XLA GPU compilation pipeline." %}
{% endcomment %}

## Setup

> **To write.** Fifteen lines, no more. Pull AMD's prebuilt JAX image, confirm the GPU
> with `rocm-smi --showproductname`, map gfx942 to MI300-class and gfx950 to
> MI350-class, then a verification snippet that asserts the backend and runs a real
> computation end to end.
>
> Everything else goes to
> [Appendix A]({{ '/pages/appendix-install' | relative_url }}): installing the four
> wheels in the right order, the ROCm version matrix, and building from source. Those
> are necessary, they are nobody's reason for reading the book, and they rot fastest.

```python
import jax

assert jax.default_backend() == "rocm", jax.default_backend()
print(jax.devices())
```

> **To write.** Replace the snippet above with the real one, which should also run a
> small computation rather than only listing devices: a backend that initialises but
> cannot execute is a failure mode people hit, and printing `jax.devices()` does not
> catch it.

## Your First Trace

> **To write.** Purely mechanical, and it should stay that way. Wrap
> `jax.profiler.trace` around the Chapter 2 matmul, show the trace directory layout
> down to the `.xplane.pb` and `.trace.json.gz`, launch `xprof --logdir`, and forward
> the port.
>
> Establish the convention of writing every trace under `/tmp/traces/<workload>/` so
> that one `--logdir` picks them all up. The book uses that convention from here on.

## The XProf Tools

> **To write.** One at a time, each with a real screenshot: what it is for, and what to
> look at first. Trace Viewer gets the most space, including the video-game navigation
> keys, because it is where the reader will spend their time.
>
> **Screenshots are the largest unplanned cost in this chapter.** Two dozen or more,
> and they all invalidate when the XProf UI or the build changes. Decide before writing
> whether they are captured by hand or scripted against XProf's data endpoints; that
> decision is the most likely thing to stall the chapter.

### Trace Viewer

> **To write.** The chronological timeline, the device rows against the host rows, and
> what the XLA Ops row actually is compared with everything below it, which is an
> approximate trace built from `jax.named_scope` and the Python
> stack.<d-footnote>Sidenotes go in the margin on wide screens and collapse into the
> appendix on narrow ones. Use them for tangents so the main line of reasoning stays
> clean.</d-footnote>

### Graph Viewer

> **To write.** Reading the HLO graph, and when it beats reading the timeline: mostly
> when you want to know what the compiler decided rather than how long it took.

### Op Profile

> **To write.** Self time against total time, and the trap of reading a fusion's total
> time as though it were a single kernel.

### Kernel Stats

> **To write.** The per-kernel table, which is where most readers start and which is
> also where several of the broken columns below live. Name that tension here and point
> forward to *What Works Today* rather than letting the reader discover it.

### Memory Profile

> **To write.** Peak allocation, the allocation timeline, and how to tell an activation
> problem from a parameter problem. This is the view that answers "why did this OOM at
> a batch size the arithmetic says should fit".

### Roofline

> **To write.** What the view is meant to show, and why on AMD the compute ceiling
> reads zero. Do the roofline by hand with
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }})'s peak instead, and treat this
> subsection as the motivation for the table below rather than as a tool tour.

<a id="limitations"></a>

## What Works Today

> **To write.** This is the section people will be linked to directly, so it needs a
> stable anchor (`#limitations`, aliased above the heading) and it needs to be findable
> from the landing page.
>
> **Pin it to a version and date it, exactly as the spec tables are pinned.** Hardware
> numbers change once a generation; this table describes software behaviour, which
> changes every few months, and several rows have fixes in flight in our own repository.
> An undated table of broken things is a liability: it will be quoted back at us after
> the bug is fixed. So the heading carries the exact wheel and ROCm versions the
> observations were made against, and any row with a known fix says so.
>
> **Tone matters.** This is a candid "here is the state of the tooling" section, not an
> apology. Most readers arriving here have already hit one of these and assumed they
> had misconfigured something.
>
> Seven rows, curated down from the twenty in the internal audit. The selection rule is
> "a reader will hit this in their first week", which is why the row about
> NVIDIA-shaped capability structs does not appear. Say that the table is a curated
> subset rather than implying it is exhaustive.

### How a Profile Gets Made

> **To write.** `rocprofiler-sdk` to the ROCm collector to XPlane and XSpace protos to
> XProf, as a diagram and about four paragraphs.
>
> **It sits here, immediately before the table, rather than in its own top-level
> section.** It exists to make the limitations legible: once you know the collector
> writes XStats that XProf later reads, the broken fields stop looking arbitrary and
> start looking like a specific missing write. Ahead of any symptom it is just stack
> tourism.
>
> Source is `gpu_profiling/docs/xla-rocm-profiler-backend.md`.

> **To write.** Fill in the cause, workaround and status columns properly, and state
> the versions in a caption above the table.

| Symptom you will observe | Cause | What to do | Fix status |
|---|---|---|---|
| Overview and Input Pipeline read "No step time measured" | No step markers in the trace | Add `jax.profiler.StepTraceAnnotation` yourself | Reader-side, works today |
| Roofline compute ceiling reads 0 GFLOP/s, labels say "per TensorCore" | Peak FLOPs is never computed for AMD | Do it by hand, per Chapter 2 | To write |
| Kernel Stats occupancy, registers per thread and shared memory all read 0 | The collector does not emit them | Use `rocprofv3`, but read Chapter 8 on what occupancy does *not* tell you before acting on it | To write |
| "GPU TensorCore utilization" reads 0 on every row | The kernel-name classifier does not recognise MFMA kernels | Ignore the column | To write |
| Multi-GPU op times are roughly 8x the wall-clock figure | Op times are summed across devices | Divide by the device count | To write |
| Device Compute Precisions reads 0% / 0% | Depends on step markers, so it follows from row 1 | Fix row 1 | To write |
| HBM bandwidth in the device plane is understated by 2x and shown in binary units while labelled decimal | Two stacked bugs in the collector | Use 5.3 TB/s; the arithmetic is in Chapter 2 | To write |

> **Note for the writer.** Keep Liquid out of table cells. Liquid renders before
> kramdown, so `{% raw %}{{ '/x' | relative_url }}{% endraw %}` does survive inside a
> row, but the pipe in the filter reads like a cell delimiter and the next person to
> edit the table will assume it is broken. Link in the prose around the table instead.

> **To write.** Note in prose that three of these have fixes in progress by our own
> team, which is the strongest possible argument for the version-pinning discipline: we
> are on both sides of this table.

## From an HLO Op Back to a Python Line

> **To write.** These were two sections in an earlier outline and they are one skill:
> following the compiler's output back to the code you wrote. Split, they make the
> reader learn to parse an op, put the book down, and then learn separately what to do
> with it. Run it as one arc instead, in the direction the reader actually travels: a
> slow kernel row in Kernel Stats, to the HLO op that emitted it, to the
> `jax.named_scope` annotation, to the source line, plus what to do when the chain
> breaks.
>
> Along the way, teach the anatomy of the op. **Do not port the source book's version:**
> its worked example is TPU-specific, and neither the `T(8,128)(2,1)` tiling nor the
> `S(1)` memory-space annotation exists on AMD. Write it fresh against a real op from
> the matmul trace: op name, output shape and dtype, the plain major-to-minor layout,
> operands.
>
> Then the AMD specifics that matter for the traversal: fusion kinds (`kLoop`, `kInput`,
> `kCustom`), custom calls into hipBLASLt and rocBLAS and how they surface as `Cijk_*`
> kernels in Kernel Stats, and async collective pairs. Follow the toy-case-first move:
> a small op fully decomposed, then the real one.
>
> The `fw101` "tracing upwards from kernel to python" material feeds this.

## The Matmul, Revisited

> **To write.** Close the loop opened in
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}). Expected time against measured
> time, and say plainly whether the model held.
>
> If it did not, this is the best possible place to explain why, because the reader now
> has the instrument in hand: warmup, autotuning, clock throttling. Tag the measured
> number **[measured]** and link
> [Appendix B]({{ '/pages/appendix-protocol' | relative_url }}).

## A Training Step

> **To write.** Forward, backward, optimizer. This is where
> `jax.profiler.StepTraceAnnotation` is introduced and where the Overview page comes
> alive.
>
> **Show it both ways.** Capture without the annotation first, hit the "No step marker
> observed" wall, then add it and recapture. Most readers have already seen the broken
> version, so leading with it is the honest ordering and it makes row 1 of the table
> above land.
>
> `scripts/transformer_block.py` and `scripts/basic_training.py` are the workloads.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number. Ship a trace alongside the chapter so these are actually
> attemptable rather than hypothetical.

**Question 1:** Given the trace, decide which of two matmuls is memory-bound and say
why.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 2:** Find the kernel with the largest self time and trace it back to its
Python line.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 3:** A colleague reports a device op time of 3.2 seconds for a step that
takes 400 ms of wall clock. Explain the discrepancy.

{% details Click here for the answer. %}

To write. The answer is row 5 of the table above, and the exercise exists to make the
reader use the table rather than read it.

{% enddetails %}
