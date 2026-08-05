---
layout: distill
title: "Training Llama 3 on MI300X"
description: "The dense capstone. A parallelism strategy justified against the inequalities rather than copied, MaxText on ROCm, capturing a profile from a run that lasts days, a per-layer breakdown, and the two operational things that decide whether a real training run finishes."
date: 2026-08-04

section_number: 9

previous_section_url: "/pages/8-getting-to-roofline"
previous_section_name: "Chapter 8: Getting to Roofline"

next_section_url: "/pages/10-deepseek"
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
  - name: References
---

> **Not started. This chapter is entirely blocked on a training run.** Everything in it
> is a measurement of Llama 3 on hardware, so there is nothing here that can honestly be
> written in advance: a capstone whose numbers were predicted rather than observed would
> defeat the purpose of the chapter and of the book's central claim. The section briefs
> are in the source of this page, and the roadmap for the chapter is the Chapter 9 section
> of `docs/structure.md`.
>
> The predictions this chapter is meant to check are already written, and you can make
> them yourself from [Chapter 5]({{ '/pages/5-transformers' | relative_url }})'s accounting
> and [Chapter 6]({{ '/pages/6-training' | relative_url }})'s decision procedure. The first
> worked problem in
> [Chapter 6]({{ '/pages/6-training' | relative_url }}) is this chapter's configuration
> question, answered analytically.

**Depends on:** Chapters [1]({{ '/pages/1-rooflines' | relative_url }}) through
[6]({{ '/pages/6-training' | relative_url }}) for the predictions this chapter checks, and
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) for the triage that explains
the gap. Nothing from [Chapter 7]({{ '/pages/7-moe' | relative_url }}): this model is dense.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

<!-- WHY THIS WHOLE CHAPTER IS BLOCKED, and what unblocks it.

     Hard dependency: a Llama 3 training run on MI300X under MaxText, captured. Nothing
     in the chapter is writable without it. Specifically needed:
       - One 8x MI300X node for Llama 3 8B. This is the minimum viable version of the
         chapter and it is not gated on a cluster.
       - Multi-node for 70B and for anything Llama 3.1 405B related. We have no cluster,
         so the multi-node half of this chapter is blocked twice over.
       - Capture wrappers around the MaxText configs (listed as a needed script in
         docs/structure.md).

     Verification owed before writing, because the roadmap makes claims we have not
     checked in this pass:
       - The roadmap says AMD maintains a ROCm MaxText fork shipping Llama 3 8B and 70B
         as pre-optimised configurations with Llama 3.1 405B documented as a multi-node
         benchmark. We have only inspected upstream AI-Hypercomputer/maxtext at commit
         9f9ac05, which carries in-tree ROCm support (run_rocm.py, nanoo_fp8) but is not
         AMD's fork. Confirm which repository and branch to target, and pin it.
       - Whether the pre-optimised configs run as documented on the stack this book pins
         (ROCm 7.2.4, jax 0.11.0). See docs/writing-notes.md.

     SECTION BRIEFS, from the roadmap:

     The Configuration, and Why Each Degree.
       Start from the published configuration rather than inventing one, then spend the
       section explaining *why* each parallelism degree was chosen, against Chapter 6's
       inequalities. That explanation is the part AMD's documentation does not do and the
       part the reader needs. Every degree should trace to a specific inequality: the
       FSDP degree to the memory ledger, the tensor-parallel degree to the F threshold,
       the per-device batch to the ridge point.

     Running MaxText on ROCm.
       The launch path, container tag, and the config file, kept short. This is the
       section most likely to rot, so it should point at Appendix A for anything that is
       really installation.

     Capturing a Profile at Production Scale.
       A few steps out of thousands: which steps (not the first, and after any autotuning
       has settled), trace file sizes, multi-host capture and where the per-host traces
       land, and how to avoid capturing 40 GB of trace by accident. Chapter 3 taught the
       mechanics on a toy; this is the same thing when the run costs money.

     A Per-Layer Breakdown.
       Where the step time goes, layer by layer and op by op, against Chapter 5's
       accounting. This is where the "stack of MLPs" approximation gets checked against
       a real model: Chapter 5 predicts attention is roughly 40% of layer FLOPs at 8k
       context, and this is the section that finds out.

     MFU Against the Roofline.
       The book's central promise cashed out for a dense model: predicted MFU from
       Chapter 6, measured MFU, and the gap explained using Chapter 8's triage order. Say
       plainly whether the model held. If it did not, the explanation is the most
       valuable paragraph in the chapter.

     Checkpointing.
       What a checkpoint costs in time and bytes at this scale, how often to take one,
       and how long a restart takes. At 70B this is a first-order throughput term and no
       other chapter owns it. Orbax is the mechanism; the numbers are what matter.

     The Input Pipeline at Scale.
       How a sharded dataset gets fed to a multi-process mesh without every host reading
       the same shard, and what a deterministic resume costs. This is checkpointing's
       sibling and the other thing that stalls real runs. Chapter 8 taught the reader to
       *recognise* host starvation in a trace; this is where they see it prevented at
       production scale rather than diagnosed after the fact.

     The Failure That Is Not a Performance Problem.
       AMD's own benchmark scripts disable an RCCL feature to avoid NaN losses on MI355X.
       That is a real, slightly uncomfortable example of the kind of thing no roofline
       predicts, and one honestly told footnote is worth more than a page of generalities
       about robustness. VERIFY THE SPECIFIC FLAG AND THE MODELS AFFECTED before writing;
       we have not checked this claim in this pass, and getting it wrong would be worse
       than omitting it.

     Worked Problems.
       From the roadmap's general pattern: given this chapter's trace, re-derive the
       parallelism strategy from replica groups alone (tests Chapter 4); compute what
       fraction of step time should be the gradient all-reduce and compare (tests
       Chapter 6); estimate the checkpoint overhead as a fraction of throughput at a
       given interval (tests this chapter). -->

## The Configuration, and Why Each Degree

## Running MaxText on ROCm

## Capturing a Profile at Production Scale

## A Per-Layer Breakdown

## MFU Against the Roofline

## Checkpointing

## The Input Pipeline at Scale

## The Failure That Is Not a Performance Problem

## Worked Problems

## References

**Provisional, and to be replaced with what the chapter actually uses.** These are the
sources the chapter will be written against; none of them is cited in prose yet because
there is no prose.

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Meta, 2024). The model
  being trained, and the architecture numbers
  [Chapter 5]({{ '/pages/5-transformers' | relative_url }}) already uses.
- [MaxText](https://github.com/AI-Hypercomputer/maxtext) (AI-Hypercomputer). The training
  framework, its Llama 3 configurations, and its ROCm support.
- [Training a model with ROCm MaxText](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/)
  (AMD). AMD's own documentation for JAX and MaxText training on Instinct hardware. The
  exact page to cite depends on which repository this chapter ends up targeting; see the
  verification note in the source of this page.
- [Orbax](https://orbax.readthedocs.io/) (Google). The checkpointing library, for the
  checkpointing section.
