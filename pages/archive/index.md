---
layout: distill
title: "How To Scale Your Model with AMD"
subtitle: "A Systems View of LLMs on AMD GPUs"
description: "Given a model and some number of MI300X-class GPUs, how do I run it in JAX so that adding GPUs adds throughput? Rooflines, sharding, parallelism and Mixture-of-Experts training, predicted from the hardware and then checked against a profile."
date: 2026-08-04

section_number: 0

previous_section_url: ""
previous_section_name: "Chapter 0: Intro"

next_section_url: "/pages/1-rooflines"
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

**One sentence: given a model and some number of MI300X-class GPUs, how do I run it
in JAX so that adding GPUs adds throughput?** Everything in this book serves that
question.

**The reason it is a hard question is that adding GPUs does two things at once.** It
cuts the compute each device has to do, and it adds communication that did not exist
before. Below some scale the first effect wins and you get the speedup you paid for;
above it the second wins and you are buying hardware to move bytes around. **The whole
book is about where those two cross**, and about the fact that the crossing point moves
depending on which parallelism strategy you chose, which model you are running, and
which generation of hardware you are on.

By the end you should be able to do three things you probably cannot do now: **ballpark
whether a training run is close to its hardware limit** before you tune anything, **choose
between data, tensor, pipeline, context and expert parallelism** at a given scale from an
inequality rather than from four experiments, and **know what an all-reduce costs** before
you pay for it.

**One thing to say in the first hundred words rather than the tenth chapter: this book is
about training.** If you arrived looking for a serving deployment guide, the honest answer
is that production serving on AMD means exporting your weights to
[vLLM](https://docs.vllm.ai/) or [SGLang](https://docs.sglang.ai/), and you should go
straight there. [Chapter 12]({{ '/pages/12-serving' | relative_url }}) explains why that is
the correct engineering decision rather than a shortfall, and
[Chapter 11]({{ '/pages/11-inference' | relative_url }}) gives you the arithmetic to size what
you hand off. Everything between is about getting the model trained in the first place.

## Why Should You Care?

**Because the gap between a model that runs and a model that runs well is now a factor of
two or three, and it is entirely yours to close.** Not long ago you could treat the
hardware as a black box and spend your attention on the model. That stopped being true
when models got large enough that a single training run costs real money and a single
serving deployment costs real money every hour.

**The concrete version: a 20% modelling win is worth nothing if it costs 20% of your
roofline.** Those two numbers are the same size, they trade against each other constantly,
and only one of them is usually being measured. A team that knows its MFU knows which
trades are worth making.

**And there is an AMD-specific reason, which is the implicit question most readers arrive
with: do the CUDA-shaped intuitions I have transfer?** Mostly, yes. A matmul is a matmul,
arithmetic intensity does not care who made the chip, and every parallelism strategy in
this book exists on both vendors. **The places where the intuitions do not transfer are
specific and worth knowing**, and they are mostly in
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}): a wavefront is 64 threads rather than
32, the scale-up domain stops hard at eight GPUs rather than growing with a switch, fp8
means a different set of bit patterns on MI300X than on anything else, and the profiler
reports several fields as zero. None of those is a reason not to use the hardware. All of
them are reasons to read a chapter before you benchmark.

## Expected Background

**Assumed:** you are comfortable with the Transformer architecture, you have written some
JAX, and you can read arithmetic with units in it.
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}) counts a Transformer rather than
explaining one, so if attention and a feed-forward block are unfamiliar, start elsewhere.

**Not assumed, and this list is as important as the one above:** nothing about GPUs,
nothing about ROCm, nothing about collectives, and **nothing about the TPU scaling book.**

**You do not need to have read
[How To Scale Your Model](https://jax-ml.github.io/scaling-book/)**, the excellent Google
DeepMind book that this one is a response to. That matters because it is the thing readers
most often worry about, and because it constrains us usefully: this book pays full price
for rooflines, sharding notation and Transformer math rather than borrowing them. Where we
cite the source book it is as further reading on a tangent, never for something
load-bearing.

**One deliberate omission, said once here so nobody hunts for it.** There is no standalone
"programming AMD GPUs in JAX" chapter. The API material is distributed instead: sharding,
`Mesh`, `NamedSharding` and `shard_map` are in
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}) alongside the notation they implement,
and the parallelism implementations are in
[Chapter 6]({{ '/pages/6-training' | relative_url }}) and
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) next to the strategies they realise. Being
JAX-only throughout means every chapter is already a JAX chapter, and teaching the API
twice is the redundancy this avoids.

## What This Book Adds

**Measurements on real AMD hardware rather than pure analysis.** The source book is almost
entirely analytical and says so; its GPU chapter is where the theory meets a wall, with
NVIDIA claiming 450 GB/s over NVLink and the authors measuring 370, and 150 at realistic
message sizes. We have machines. Checking the arithmetic against a profile is the single
most valuable thing we can add, and
[How to Read a Number](#how-to-read-a-number-in-this-book) is how we keep that promise
honest.

**Mixture-of-Experts as a first-class subject.** The source book gives MoE about one
subsection and two problems, which is hard to defend when most frontier models are sparse.
Here, sparse accounting appears in
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}) alongside dense accounting,
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) is a full chapter on the routing,
imbalance, capacity and dispatch decisions dense models never have to make, and the sparse
training capstone is [Chapter 10]({{ '/pages/10-deepseek' | relative_url }}).
[Chapter 7]({{ '/pages/7-moe' | relative_url }})'s spine is a question nobody has answered
in public: of the three ways to implement an expert layer, which one can a JAX user on
ROCm actually run fast, and what does the answer cost them in FLOPs.

**An honest account of what happens to the checkpoint after training.** Most scaling
material stops at the loss curve. [Chapter 12]({{ '/pages/12-serving' | relative_url }}) is
about the handoff out of JAX, which is a real engineering problem that nobody has written
up and which has a silent failure mode.

## How to Read a Number in This Book

**Every performance claim here is either computed or measured, and it says which.** Two
tags, defined once, used inline at the point of the claim:

- **[measured]** is a number read off a captured profile on hardware we ran, with the
  protocol in [Appendix B]({{ '/pages/b-appendix-protocol' | relative_url }}) behind it.
- **[analytical]** is derived from published specifications and arithmetic, and not
  checked against hardware.

**Then the scope, stated without hedging.** Every single-node training roofline in this
book is checked against a profile. **Inter-node claims are [analytical]** until we have a
cluster, because we have never had more than eight GPUs at once. **Inference claims are
[analytical] by design**, because we do not run a serving stack. A section that is entirely
one or the other says so in its opening line rather than tagging every sentence.

**There is a third category and it gets named too: measurements that are somebody else's.**
Several of AMD's own published figures are load-bearing here, particularly the realised
xGMI and RCCL bandwidths that
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}) calibrates against, and the occupancy
sweep that [Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) quotes. Those are
attributed to AMD at the point of use and never presented as ours. A borrowed measurement
is stronger evidence than our arithmetic and weaker than our own capture, and you are
entitled to know which you are looking at.

**A book that measures what it can and labels the rest is trusted. One that quietly derives
while implying it measured is not, and the reader always finds out.**

## Reading Paths

**Fifteen pages is enough that "read it in order" is not the only useful instruction.**
Reading in order works and is what we would recommend. These are the shortcuts.

| Route | Chapters |
|---|---|
| I want to train a dense model efficiently | 1, 2, 3, 4, 5, 6, 8, 9 |
| I want to train a Mixture-of-Experts model | 1, 2, 3, 4, 5, 7, 8, 10 |
| I need to size and ship what I trained | 1, 2, 5, 11, 12 |
| My profile looks wrong and I just need the tooling | 2, 3, 8 |

**Chapters 1, 2 and 5 appear in all four routes, which makes them the irreducible core.**
Rooflines, the hardware constants, and the accounting: everything else is those three
substituted into a specific situation.

**Note that both training routes carry Chapter 3 even though it is a tooling chapter.**
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) is unreadable without it,
and a book whose spine is predict-then-measure cannot honestly let you skip the measuring.
[Chapter 3]({{ '/pages/3-profiling' | relative_url }}) names its own minimum path in its
opening paragraph, so you can take the short version.

**The last row is a reference route rather than a reading route**, for readers who arrive
from a search result with a broken profile and no interest in the rest of the book. There
will be a lot of them, and
[Chapter 3]({{ '/pages/3-profiling' | relative_url }})'s limitations table is what they came
for.

## Notation

Fixed for the whole book, introduced in
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}), and repeated at the top of
every chapter that uses it.

{% include notation.liquid %}

## Links to Chapters

**Part I: Preliminaries.** Everything needed before the word "parallelism" means
anything. None of it depends on a large model or on more than one node.

* [**Chapter 1: All About Rooflines**]({{ '/pages/1-rooflines' | relative_url }}).
  What actually limits how fast this runs? Compute, memory bandwidth and
  communication, and the arithmetic that tells you which one you are up against.
* [**Chapter 2: How to Think About AMD GPUs**]({{ '/pages/2-amd-gpus' | relative_url }}).
  How fast should this run on an MI300X, and what in the hardware decides that?
  Every constant the rest of the book substitutes into.
* [**Chapter 3: How to Profile AMD GPU Programs**]({{ '/pages/3-profiling' | relative_url }}).
  Your model is slower than the arithmetic says it should be. Where did the time go?
* [**Chapter 4: Sharded Matrices and How to Multiply Them**]({{ '/pages/4-sharding' | relative_url }}).
  You split the matrix across eight GPUs. What does that cost?

**Part II: Training Transformers.** Part I taught the hardware and the cost model;
this part spends them.

* [**Chapter 5: All the Transformer Math You Need**]({{ '/pages/5-transformers' | relative_url }}).
  How many parameters, FLOPs and bytes, exactly? Dense and sparse, side by side.
* [**Chapter 6: How to Parallelize a Transformer for Training**]({{ '/pages/6-training' | relative_url }}).
  You added seven more GPUs and got four times the throughput. Where did the rest
  go?
* [**Chapter 7: Mixture-of-Experts at Scale**]({{ '/pages/7-moe' | relative_url }}).
  Only a fraction of the parameters run per token, so why isn't it a fraction of the
  time?

**Part III: Training in Practice.** The gap between the performance you should get
and the performance you get, then closing it twice on real models.

* [**Chapter 8: Getting to Roofline**]({{ '/pages/8-getting-to-roofline' | relative_url }}).
  The prediction says 40% MFU. You measured 22%. A triage order, cheapest checks
  first.
* [**Chapter 9: Training Llama 3 on MI300X**]({{ '/pages/9-llama' | relative_url }}).
  The dense capstone: predict, measure, explain the gap, and survive a run that
  lasts days.
* [**Chapter 10: Training DeepSeek-V2-Lite on MI300X**]({{ '/pages/10-deepseek' | relative_url }}).
  The sparse capstone. Same method, harder model.

**Part IV: After Training.** What the model costs to serve, how it leaves JAX, and
where to read next.

* [**Chapter 11: How to Think About Inference**]({{ '/pages/11-inference' | relative_url }}).
  Serving is not training with `no_grad`. How much memory, at what batch size, and
  what does a token cost?
* [**Chapter 12: Getting Your Model Into Production**]({{ '/pages/12-serving' | relative_url }}).
  You trained it in JAX. Now it has to serve traffic, and that probably isn't JAX.
* [**Chapter 13: Conclusions and Further Reading**]({{ '/pages/13-conclusion' | relative_url }}).
  What we got wrong, and where to go next.

**Appendices.** The two things that rot fastest, kept where they can be revised
without touching a chapter.

* [**Appendix A: Installing JAX on ROCm**]({{ '/pages/a-appendix-install' | relative_url }}).
  Containers, wheels, the version matrix, and the combinations known to be broken.
* [**Appendix B: How We Measure**]({{ '/pages/b-appendix-protocol' | relative_url }}).
  The protocol behind every **[measured]** number in the book.

<h3 markdown=1 class="next-section">Without further ado, [here is Chapter 1 on rooflines]({{ '/pages/1-rooflines' | relative_url }}).</h3>
