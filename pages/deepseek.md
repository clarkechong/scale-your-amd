---
layout: distill
title: "Training DeepSeek-V2-Lite on MI300X"
description: "The sparse capstone, and the hardest thing in the book we can run end to end. Same method as the dense chapter, harder model: routing and imbalance on a real run, expert parallelism against the eight-GPU mesh ceiling, and what latent attention does to the memory profile."
date: 2026-08-04

section_number: 10

previous_section_url: "/pages/llama"
previous_section_name: "Chapter 9: Llama 3"

next_section_url: "/pages/inference"
next_section_name: "Chapter 11: Inference"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: The Configuration, and Why Each Degree
  - name: Routing and Imbalance in a Real Run
  - name: Expert Parallelism and All-to-All Placement
  - name: What Latent Attention Does to the Memory Profile
  - name: MFU Against the Roofline
  - name: Fallbacks If This Model Disappoints
  - name: Worked Problems
---

> **Skeleton.** Section structure only; the configuration, the profile and the numbers are
> still to be written. The brief for this chapter is the Chapter 10 section of
> `docs/structure.md`.

**Depends on:** [Chapter 5]({{ '/pages/transformers' | relative_url }}) for MoE and latent
attention accounting, [Chapter 7]({{ '/pages/moe' | relative_url }}) for everything about
routing and expert parallelism, [Chapter 8]({{ '/pages/getting-to-roofline' | relative_url }})
for triage, and [Chapter 9]({{ '/pages/llama' | relative_url }}) for the method this chapter
repeats.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

> **To write.** Same method as [Chapter 9]({{ '/pages/llama' | relative_url }}), harder
> model: predict from Chapters [5]({{ '/pages/transformers' | relative_url }}) and
> [7]({{ '/pages/moe' | relative_url }}), measure, explain the gap. Lean on the fact that the
> reader has now seen the method once, and spend the saved space on the sparsity.
>
> **The four numbers from [Chapter 7]({{ '/pages/moe' | relative_url }}) are the
> instrumentation for this chapter**, so this is where they earn their keep. If they turn out
> not to be sufficient to diagnose this run, that is a finding and Chapter 7's list should
> change.
>
> **This chapter was converted from a serving capstone to a training capstone, which retired
> the book's largest risk.** The serving version needed `decode.py` working on ROCm plus a JAX
> serving path, neither of which exists, and
> [Chapter 12]({{ '/pages/serving' | relative_url }}) now explains why chasing them was the
> wrong call anyway. The training version needs a model that AMD's own ROCm MaxText fork
> already lists as pre-optimised, which DeepSeek-V2-Lite is.

## The Configuration, and Why Each Degree

> **To write.** As in [Chapter 9]({{ '/pages/llama' | relative_url }}), but with the expert
> axis in play. Justify the expert-parallel degree against
> [Chapter 7]({{ '/pages/moe' | relative_url }})'s inequalities, and say explicitly which
> implementation of the expert layer this configuration selects, since that is the single
> decision with the largest effect on the FLOP bill.

## Routing and Imbalance in a Real Run

> **To write.** The tokens-per-expert histogram from a real run rather than a toy one, how it
> moves over training, and what it costs in step time. Show the artifact first, then the
> arithmetic.
>
> This is the section where [Chapter 7]({{ '/pages/moe' | relative_url }})'s claims either
> hold on a real model or do not, so be willing to report that a predicted effect was smaller
> than expected. That is a result too.

## Expert Parallelism and All-to-All Placement

> **To write.** The placement question made concrete: where the expert axis sits in the mesh,
> what the all-to-all costs **[measured]** inside the baseboard, and what it would cost
> crossing it, which stays **[analytical]** until we have a cluster.
>
> This is also where [Chapter 7]({{ '/pages/moe' | relative_url }})'s prediction that a
> switchless full mesh is unusually good at all-to-all gets tested on a real workload rather
> than a microbenchmark. Compare against
> [Chapter 4]({{ '/pages/sharding' | relative_url }})'s sweep and explain any difference.

## What Latent Attention Does to the Memory Profile

> **To write.** Multi-head latent attention is accounted for in
> [Chapter 5]({{ '/pages/transformers' | relative_url }}); this section is only about what it
> does to *this run*: the activation and cache footprint in the Memory Profile view, and
> whether the predicted saving showed up.
>
> Keep it to training. The serving consequences are
> [Chapter 11]({{ '/pages/inference' | relative_url }})'s.

## MFU Against the Roofline

> **To write.** Against the sparse roofline from
> [Chapter 5]({{ '/pages/transformers' | relative_url }}), not the dense one, and say which
> denominator is being used, because for an MoE that choice changes the number a lot and
> published figures are inconsistent about it.
>
> Tagged **[measured]**, with
> [Appendix B]({{ '/pages/appendix-protocol' | relative_url }}) behind it.

## Fallbacks If This Model Disappoints

> **To write.** Qwen3 30B-A3B is the alternative, and Mixtral 8x7B is the simpler fallback
> below that. All three are in the same pre-optimised list, so switching costs a re-run rather
> than a rewrite. State that here so the chapter does not read as though it depended on one
> model working.

> **Close by pointing forward at Part IV** for what happens to the checkpoint next. **Make no
> serving claims here at all:** this is a training chapter, and
> [Chapter 11]({{ '/pages/inference' | relative_url }}) and
> [Chapter 12]({{ '/pages/serving' | relative_url }}) own that ground.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number.

**Question 1:** From this run's tokens-per-expert histogram, estimate the step time that
perfect balance would have achieved.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 2:** Given this configuration on 32 GPUs across four baseboards, decide where to
put the expert axis.

{% details Click here for the answer. %}

To write. **[analytical]**, and the answer should say what measurement would settle it.

{% enddetails %}

**Question 3:** Recompute this run's MFU against the dense parameter count instead of the
activated one, and say which figure you would publish.

{% details Click here for the answer. %}

To write. Both numbers, and an argument for one of them.

{% enddetails %}
