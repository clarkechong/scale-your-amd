---
layout: distill
title: "Training Llama 3 on MI300X"
description: "The dense capstone. A parallelism strategy justified against the inequalities rather than copied, MaxText on ROCm, capturing a profile from a run that lasts days, a per-layer breakdown, and the two operational things that decide whether a real training run finishes."
date: 2026-08-04

section_number: 9

previous_section_url: "/pages/getting-to-roofline"
previous_section_name: "Chapter 8: Getting to Roofline"

next_section_url: "/pages/deepseek"
next_section_name: "Chapter 10: DeepSeek-V2-Lite"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: The Configuration, and Why Each Degree
  - name: Running MaxText on ROCm
  - name: Capturing a Profile at Production Scale
  - name: A Per-Layer Breakdown
  - name: MFU Against the Roofline
  - name: Checkpointing
  - name: The Input Pipeline at Scale
  - name: The Failure That Is Not a Performance Problem
  - name: Worked Problems
---

> **Skeleton.** Section structure only; the configuration, the profile and the numbers are
> still to be written. The brief for this chapter is the Chapter 9 section of
> `docs/structure.md`.

**Depends on:** Chapters [1]({{ '/pages/rooflines' | relative_url }}) through
[6]({{ '/pages/training' | relative_url }}) for the predictions this chapter checks, and
[Chapter 8]({{ '/pages/getting-to-roofline' | relative_url }}) for the triage that explains
the gap. Nothing from [Chapter 7]({{ '/pages/moe' | relative_url }}): this model is dense.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

> **To write.** This chapter establishes the method that
> [Chapter 10]({{ '/pages/deepseek' | relative_url }}) reuses on a harder model, so the
> structure matters more than usual: predict from the earlier chapters, measure, explain the
> gap, then deal with the things that no roofline predicts.
>
> **This chapter is in better shape than it looks.** AMD's ROCm MaxText fork ships Llama 3
> 8B and 70B as pre-optimised configurations, with Llama 3.1 405B documented as a multi-node
> benchmark, so the config and the launch path are given rather than invented. Use their
> configuration as the starting point and spend the chapter explaining **why** each degree
> was chosen, which is the part their documentation does not do and the part the reader
> needs.

## The Configuration, and Why Each Degree

> **To write.** Take AMD's configuration and justify it line by line against
> [Chapter 6]({{ '/pages/training' | relative_url }})'s inequalities: why this data-parallel
> degree, why this tensor-parallel degree, why remat here and not there, why this precision.
>
> Then the more useful exercise, which is to name a degree that is *not* obviously right and
> work out what it would take to change it. A capstone that only ratifies the given config
> teaches less than one that pushes on it.

## Running MaxText on ROCm

> **To write.** The container, the config file, the launch command, and multi-node launch via
> Primus. Keep it short and point at
> [Appendix A]({{ '/pages/appendix-install' | relative_url }}) for anything that is really
> installation rather than training.

## Capturing a Profile at Production Scale

> **To write.** Everything that changes when the run is real rather than a benchmark: taking
> a few steps out of thousands, how large the trace files get, multi-host capture and where
> the per-host traces land, and how to avoid capturing during warmup or during a checkpoint.
>
> This is the section that makes
> [Chapter 3]({{ '/pages/profiling' | relative_url }})'s mechanics survive contact with a
> real job, and it is genuinely undocumented elsewhere.

## A Per-Layer Breakdown

> **To write.** Where the step time actually goes, layer by layer and block by block, against
> the [Chapter 5]({{ '/pages/transformers' | relative_url }}) accounting. Attention against
> MLP against norms against the vocabulary projection, and whether the ratios match what the
> arithmetic said they would be.

## MFU Against the Roofline

> **To write.** The number the whole book has been building towards, tagged **[measured]**,
> with [Appendix B]({{ '/pages/appendix-protocol' | relative_url }}) behind it. Say plainly
> whether the prediction held, and if it did not, use
> [Chapter 8]({{ '/pages/getting-to-roofline' | relative_url }})'s triage order to say why
> rather than reaching for an explanation.
>
> State whether the figure is MFU or HFU, per
> [Chapter 5]({{ '/pages/transformers' | relative_url }}). This is the first place in the book
> where the distinction has real money attached to it.

## Checkpointing

> **To write.** What a checkpoint costs in time and bytes at this scale, how often to take
> one, and how long a restart takes. At 405B on a shared cluster this is a first-order
> throughput term and no other chapter owns it: a run that checkpoints too often is slow, and
> one that checkpoints too rarely loses a day when a node fails.

## The Input Pipeline at Scale

> **To write.** Checkpointing's sibling, and the other thing that stalls real runs. How a
> sharded dataset gets fed to a multi-process mesh without every host reading the same shard,
> and what a deterministic resume costs.
>
> [Chapter 8]({{ '/pages/getting-to-roofline' | relative_url }}) taught the reader to
> *recognise* host starvation in a trace. This is where they see it prevented at production
> scale rather than diagnosed after the fact, which is the more useful order for something
> this common.

## The Failure That Is Not a Performance Problem

> **To write.** AMD's own benchmark scripts disable an RCCL feature to avoid NaN losses on
> MI355X. That is a perfect, real, and slightly uncomfortable example of the kind of thing no
> roofline predicts, and one footnote honestly told is worth more than a page of generalities
> about robustness.
>
> Resist expanding this into a section on reliability. Its value is entirely in being
> specific.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number.

**Question 1:** Given this configuration and a different device count, work out the new
parallelism split and predict the step time.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 2:** From the per-layer breakdown, decide whether remat is currently costing more
than it saves.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 3:** Given a checkpoint size and a mean time between node failures, choose a
checkpoint interval.

{% details Click here for the answer. %}

To write. The answer is a short optimization and it is the only place in the book where
reliability enters an arithmetic argument.

{% enddetails %}
