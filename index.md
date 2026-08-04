---
layout: distill
title: "How To Scale Your Model with AMD"
subtitle: "A Systems View of LLMs on AMD GPUs"
description: "Given a model and some number of MI300X-class GPUs, how do I run it in JAX so that adding GPUs adds throughput? Rooflines, sharding, parallelism and Mixture-of-Experts training, predicted from the hardware and then checked against a profile."
date: 2026-08-04

section_number: 0

previous_section_url: ""
previous_section_name: "Chapter 0: Intro"

next_section_url: "/pages/rooflines"
next_section_name: "Chapter 1: Rooflines"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Why Should You Care?
  - name: Expected Background
  - name: What This Book Adds
  - name: How to Read a Number in This Book
  - name: Reading Paths
  - name: Notation
  - name: Links to Chapters
---

> **Skeleton.** Structure and navigation only. The prose, the arithmetic and the
> figures are still to be written. `docs/structure.md` is the brief: it says what
> each section has to deliver and why it exists, and it is the document to argue
> with before writing rather than after.

**One sentence: given a model and some number of MI300X-class GPUs, how do I run it
in JAX so that adding GPUs adds throughput?** Everything in this book serves that
question.

> **To write.** Open on the strong-scaling framing: adding GPUs cuts compute time
> but adds communication, and the whole book is about where those two cross. Then
> the payoff, concretely, in the terms a reader recognises: ballpark whether a
> training run is close to its hardware limit, choose between data, tensor,
> pipeline, context and expert parallelism at a given scale, and know what an
> all-reduce costs before paying for it.
>
> Also, in the first hundred words rather than the tenth chapter: **this book is
> about training.** A reader looking for a serving deployment guide should be sent
> to vLLM on ROCm immediately, with
> [Chapter 12]({{ '/pages/serving' | relative_url }}) as the explanation of why that
> is the right answer rather than a shortfall.

## Why Should You Care?

> **To write.** Why this is not niche knowledge any more: even small models now run
> close enough to hardware limits that a 20% modelling win is worth nothing if it
> costs 20% of roofline. Keep it to a few paragraphs, and make one of them about AMD
> specifically, since the reader's implicit question is whether the CUDA-shaped
> intuitions they arrived with transfer. Mostly they do; the places they do not are
> the interesting part of
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}).

## Expected Background

> **To write.** Stated plainly: comfortable with the Transformer architecture, some
> JAX, no assumed knowledge of GPUs, ROCm, collectives, or the TPU book. Say what is
> *not* assumed as loudly as what is, because "you do not need to have read the TPU
> scaling book" is a selling point for exactly the readers most likely to worry about
> it. Every concept this book leans on is taught in it.

## What This Book Adds

> **To write.** Three lines, then one more. Measurements on real AMD hardware rather
> than pure analysis. Mixture-of-Experts as a first-class subject rather than a
> subsection, covering the routing, imbalance, capacity and dispatch decisions that
> dense models never have to make. An honest account of what happens to the
> checkpoint after training.
>
> Then the deliberate omission, said once here so nobody hunts for it: there is no
> standalone "programming AMD GPUs in JAX" chapter. The API material is distributed
> instead, sharding and `shard_map` in
> [Chapter 4]({{ '/pages/sharding' | relative_url }}) alongside the notation they
> implement, and the parallelism implementations in
> [Chapter 6]({{ '/pages/training' | relative_url }}) and
> [Chapter 7]({{ '/pages/moe' | relative_url }}) next to the strategies they
> realise. Being JAX-only throughout means every chapter is already a JAX chapter.

## How to Read a Number in This Book

> **To write.** Define the two tags here, once, because every performance claim in
> the book carries one of them.
>
> - **[measured]** is a number read off a captured profile on hardware we ran, with
>   the protocol in
>   [Appendix B]({{ '/pages/appendix-protocol' | relative_url }}) behind it.
> - **[analytical]** is derived from published specifications and not checked.
>
> Then the scope, stated without hedging: every single-node training roofline in this
> book is checked against a profile. Inter-node claims are **[analytical]** until we
> have a cluster, and inference claims are **[analytical]** by design, because we do
> not run a serving stack. A book that measures what it can and labels the rest is
> trusted; one that quietly derives while implying it measured is not, and the reader
> always finds out.

## Reading Paths

> **To write.** One short paragraph: fifteen pages is enough that "read it in order"
> is not the only useful instruction. Reading in order works, and these are the
> shortcuts.

| Route | Chapters |
|---|---|
| I want to train a dense model efficiently | 1, 2, 3, 4, 5, 6, 8, 9 |
| I want to train a Mixture-of-Experts model | 1, 2, 3, 4, 5, 7, 8, 10 |
| I need to size and ship what I trained | 1, 2, 5, 11, 12 |
| My profile looks wrong and I just need the tooling | 2, 3, 8 |

> **To write.** Note that Chapters 1, 2 and 5 appear in all four routes, which makes
> them the irreducible core. Note also why the training routes carry Chapter 3 even
> though it is a tooling chapter: Chapter 8 is unreadable without it, and a book
> whose spine is predict-then-measure cannot honestly let you skip the measuring. The
> last row is a reference route rather than a reading route, for readers who arrive
> from a search result with a broken profile and no interest in the rest of the book.
> There will be a lot of them.

## Notation

Fixed for the whole book, introduced in
[Chapter 5]({{ '/pages/transformers' | relative_url }}), and repeated at the top of
every chapter that uses it.

{% include notation.liquid %}

## Links to Chapters

**Part I: Preliminaries.** Everything needed before the word "parallelism" means
anything. None of it depends on a large model or on more than one node.

* [**Chapter 1: All About Rooflines**]({{ '/pages/rooflines' | relative_url }}).
  What actually limits how fast this runs? Compute, memory bandwidth and
  communication, and the arithmetic that tells you which one you are up against.
* [**Chapter 2: How to Think About AMD GPUs**]({{ '/pages/amd-gpus' | relative_url }}).
  How fast should this run on an MI300X, and what in the hardware decides that?
  Every constant the rest of the book substitutes into.
* [**Chapter 3: How to Profile AMD GPU Programs**]({{ '/pages/profiling' | relative_url }}).
  Your model is slower than the arithmetic says it should be. Where did the time go?
* [**Chapter 4: Sharded Matrices and How to Multiply Them**]({{ '/pages/sharding' | relative_url }}).
  You split the matrix across eight GPUs. What does that cost?

**Part II: Training Transformers.** Part I taught the hardware and the cost model;
this part spends them.

* [**Chapter 5: All the Transformer Math You Need**]({{ '/pages/transformers' | relative_url }}).
  How many parameters, FLOPs and bytes, exactly? Dense and sparse, side by side.
* [**Chapter 6: How to Parallelize a Transformer for Training**]({{ '/pages/training' | relative_url }}).
  You added seven more GPUs and got four times the throughput. Where did the rest
  go?
* [**Chapter 7: Mixture-of-Experts at Scale**]({{ '/pages/moe' | relative_url }}).
  Only a fraction of the parameters run per token, so why isn't it a fraction of the
  time?

**Part III: Training in Practice.** The gap between the performance you should get
and the performance you get, then closing it twice on real models.

* [**Chapter 8: Getting to Roofline**]({{ '/pages/getting-to-roofline' | relative_url }}).
  The prediction says 40% MFU. You measured 22%. A triage order, cheapest checks
  first.
* [**Chapter 9: Training Llama 3 on MI300X**]({{ '/pages/llama' | relative_url }}).
  The dense capstone: predict, measure, explain the gap, and survive a run that
  lasts days.
* [**Chapter 10: Training DeepSeek-V2-Lite on MI300X**]({{ '/pages/deepseek' | relative_url }}).
  The sparse capstone. Same method, harder model.

**Part IV: After Training.** What the model costs to serve, how it leaves JAX, and
where to read next.

* [**Chapter 11: How to Think About Inference**]({{ '/pages/inference' | relative_url }}).
  Serving is not training with `no_grad`. How much memory, at what batch size, and
  what does a token cost?
* [**Chapter 12: Getting Your Model Into Production**]({{ '/pages/serving' | relative_url }}).
  You trained it in JAX. Now it has to serve traffic, and that probably isn't JAX.
* [**Chapter 13: Conclusions and Further Reading**]({{ '/pages/conclusion' | relative_url }}).
  What we got wrong, and where to go next.

**Appendices.** The two things that rot fastest, kept where they can be revised
without touching a chapter.

* [**Appendix A: Installing JAX on ROCm**]({{ '/pages/appendix-install' | relative_url }}).
  Containers, wheels, the version matrix, and the combinations known to be broken.
* [**Appendix B: How We Measure**]({{ '/pages/appendix-protocol' | relative_url }}).
  The protocol behind every **[measured]** number in the book.

<h3 markdown=1 class="next-section">Without further ado, [here is Chapter 1 on rooflines]({{ '/pages/rooflines' | relative_url }}).</h3>
