---
layout: distill
title: "How to Parallelize a Transformer for Training"
description: "You added seven more GPUs and got four times the throughput. Where did the rest go? Data, fully-sharded, tensor, pipeline and context parallelism, each with its roofline and its measured step time, then how they compose and how to choose."
date: 2026-08-04

section_number: 6

previous_section_url: "/pages/transformers"
previous_section_name: "Chapter 5: Transformer Math"

next_section_url: "/pages/moe"
next_section_name: "Chapter 7: Mixture-of-Experts"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: How to Read This Chapter
  - name: Data Parallelism
  - name: Fully Sharded Data Parallelism
  - name: Tensor Parallelism
  - name: Pipeline Parallelism
  - name: Context Parallelism
  - name: How They Compose
  - name: The Optimal Split
  - name: Memory, Not Just Time
  - name: Low Precision as a Parallelism Decision
  - name: A Decision Procedure
  - name: Worked Problems
---

> **Skeleton.** Section structure only; the prose, the five rooflines and the measured
> step times are still to be written. The brief for this chapter is the Chapter 6 section
> of `docs/structure.md`.

**Depends on:** [Chapter 4]({{ '/pages/sharding' | relative_url }}) for the collective
cost model and the sharding notation, and
[Chapter 5]({{ '/pages/transformers' | relative_url }}) for the FLOP and byte counts
being sharded. Dense models only: expert parallelism is
[Chapter 7]({{ '/pages/moe' | relative_url }}).

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

> **To write.** Open on the observation in the subtitle, which is the experience the
> reader has actually had. Then say what the chapter's product is: **not five facts, but
> one decision procedure.**

## How to Read This Chapter

> **To write.** State the template up front, because the chapter is deliberately
> repetitive and a reader who does not know that will think it is padded. Each of the
> five strategies gets the same five parts, so that the reader learns the *move* rather
> than five separate results:
>
> 1. **What it shards**, as a one-line sharding of In, `W_in`, `W_out` and Out.
> 2. **Why do this, why not do this.** Qualitative motivation before any algebra.
> 3. **The algorithm**, as a numbered listing with every collective annotated either
>    *(on critical path)* or *(not on critical path, can be overlapped)*. That
>    annotation is how the reader learns that data parallelism's all-reduce is forgiving
>    and tensor parallelism's is not, and it is nearly free to write.
> 4. **The roofline.** Set `t_math > t_comms`, solve for a clean inequality, and
>    substitute the MI300X and xGMI constants from
>    [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) to get a real number.
> 5. **Predict, then measure.** Run it, show the trace, say whether the bound held.
>
> Every measured step time in this chapter is single-node and therefore **[measured]**;
> every inter-node claim is **[analytical]**. Say that once here.

## Data Parallelism

> **To write.** Replicate the weights, shard the batch, all-reduce the gradients. The
> five parts as above.
>
> The two results worth landing: the gradient all-reduce is off the critical path and can
> be overlapped with the backward pass, which is why data parallelism scales so
> forgivingly; and the ceiling is memory, not time, since every device holds the full
> parameter and optimizer state. Point forward to *Memory, Not Just Time* for where that
> ceiling actually is on 192 GiB.

## Fully Sharded Data Parallelism

> **To write.** Shard the parameters and optimizer state as well as the batch, then
> all-gather the weights layer by layer on the way forward and reduce-scatter the
> gradients on the way back. The ZeRO stages are the same idea at three depths; name the
> correspondence once and then use one vocabulary.
>
> The roofline here is the one readers most often get wrong, because the comms volume
> depends on the parameter count rather than on the batch, so it does *not* improve as
> you grow the batch. That asymmetry against tensor parallelism is what makes
> *The Optimal Split* below have an answer at all.

## Tensor Parallelism

> **To write.** Shard the feed-forward dimension and the heads, all-reduce or
> reduce-scatter the activations inside each layer. The five parts.
>
> The result to land: these collectives are **on the critical path**, twice per layer,
> so the inequality is much tighter than data parallelism's and it is a function of `D`
> rather than of the batch. Substitute the xGMI numbers and get the device count past
> which tensor parallelism stops paying. **On AMD that number matters more than on TPU,
> because the answer is usually smaller than the eight GPUs on the baseboard**, and
> going past the baseboard is the cliff from
> [Chapter 4]({{ '/pages/sharding' | relative_url }}).

## Pipeline Parallelism

> **To write.** Split the layers across devices, send activations between stages, and
> manage the bubble with microbatching.
>
> **This gets full treatment including a roofline**, unlike the source book, which
> declines to derive one on the grounds that pipelining matters less on TPU. On a
> scale-out Ethernet fabric it is a first-class strategy: the point-to-point transfer
> between stages is small compared with an all-reduce, which is exactly what you want on
> the slow axis. Skipping it would be a real hole.
>
> Derive the bubble fraction, then the inequality, then be honest that the interesting
> regime is inter-node and therefore **[analytical]** until we have a cluster.

## Context Parallelism

> **To write.** Shard the sequence, exchange keys and values ring-style so that every
> query eventually sees every key. The five parts, and the memory motivation, since this
> is usually reached for because activations do not fit rather than because time does not
> work out.
>
> **Settle the sequence-parallelism confusion here, in one sentence**, because readers
> arrive with the two words fused and MaxText exposes both as separate axes. Sequence
> parallelism in the Megatron sense is a companion to tensor parallelism that shards
> norms and residual activations along the sequence axis to save activation memory.
> Context parallelism shards attention itself and needs the ring exchange. Different
> collectives, different reasons, and a reader who conflates them writes a config that is
> silently wrong rather than loudly broken.

## How They Compose

> **To write.** Which combinations put which collective on which mesh axis, and the
> interaction with node topology: what has to stay inside a node and what can cross. On
> AMD this is sharper than on TPU, because intra-node and inter-node bandwidth differ by
> a large factor rather than a small one.
>
> [Chapter 4]({{ '/pages/sharding' | relative_url }})'s multi-process section is the
> mechanism; this is where it gets spent. A worked mesh layout for 64 GPUs, with the
> reasoning for which axis went where, is worth more here than a general principle.

## The Optimal Split

> **To write.** FSDP comms grow with the data axis while tensor-parallel comms shrink
> with it, so the worst case is minimized where the two meet. Derive it, then sanity-check
> the answer against the measured step times from the sections above.

## Memory, Not Just Time

> **To write.** Optimizer state, activations, remat, and the parameter ceiling for pure
> data parallelism at 192 GiB per device. **Several strategies are chosen for memory
> reasons and the time roofline never explains that**, which is a real gap in how this
> material is usually taught.
>
> Gradient accumulation belongs here as the lever that decouples the two: it buys a large
> global batch without the memory of one, at the cost of more steps. It is also where the
> two meanings of "critical batch size" from
> [Chapter 1]({{ '/pages/rooflines' | relative_url }}) finally meet, because the global
> batch is bounded below by the hardware ridge point and above by convergence, and the
> data-parallel degree has to fit between them.

## Low Precision as a Parallelism Decision

> **To write.** This is a scaling result, not a numerics footnote, and it belongs next to
> the inequalities it perturbs.
>
> On MI300X fp8 is exactly 2x bf16, so training in fp8 halves `t_math` and therefore
> **moves every inequality in this chapter**: it doubles the critical batch size, makes
> tensor parallelism go communication-bound sooner, and changes which strategy wins at a
> given scale.
>
> Keep it tight: what fp8 does to each roofline above, the practical scaling recipes, and
> the gfx942-against-gfx950 format split from
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}). It is also the best-supported
> thing in the book on the software side, since AMD's ROCm MaxText fork ships
> `nanoo_fp8` for MI300X and `fp8` for MI355X as documented benchmark configurations, so
> this section can be **[measured]** rather than derived.
>
> [Chapter 11]({{ '/pages/inference' | relative_url }}) handles the inference side:
> weight-only quantization, KV cache quantization, fp4 and fp6. Do not pre-empt it here.

## A Decision Procedure

> **To write.** Explicit, as a flowchart or a short ordered checklist. The source book
> leaves this implicit and it is the single most common thing readers want from a chapter
> like this.
>
> Cover the small-model, large-model and large-batch regimes separately, since they have
> different answers, and make precision an input to the procedure rather than an
> afterthought.

> **End the chapter by pointing at Chapter 8.** A reader who has just chosen a strategy
> is about to run it and discover they are at 22% MFU.
> [Chapter 8]({{ '/pages/getting-to-roofline' | relative_url }}) is readable from here:
> everything in it except the MoE-kernel section stands on this chapter and
> [Chapter 3]({{ '/pages/profiling' | relative_url }}). One line, so a dense-model reader
> does not have to get through sparsity first to find the triage list.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number.

**Question 1:** From a trace alone, determine the parallelism strategy and the
per-device batch size.

{% details Click here for the answer. %}

To write. The replica-groups trick from
[Chapter 4]({{ '/pages/sharding' | relative_url }}) does most of the work.

{% enddetails %}

**Question 2:** Compute what fraction of step time *should* be the gradient all-reduce
for a given model and device count, then compare with the profile.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 3:** For a given model on 64 GPUs, pick a parallelism strategy and justify it
against the inequalities in this chapter.

> **To write.** The answer should be a defence rather than a single configuration, since
> more than one split is defensible at 64 GPUs. Say which constraint each choice is
> respecting.

{% details Click here for the answer. %}

To write.

{% enddetails %}
