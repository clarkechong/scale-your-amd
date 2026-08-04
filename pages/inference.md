---
layout: distill
title: "How to Think About Inference"
description: "Serving is not training with no_grad. How much memory do I need, at what batch size, and what will a token cost? Prefill against decode, the decode step-time formula, KV cache economics, why FSDP inverts at decode, and why Mixture-of-Experts is hardest here."
date: 2026-08-04

section_number: 11

previous_section_url: "/pages/deepseek"
previous_section_name: "Chapter 10: DeepSeek-V2-Lite"

next_section_url: "/pages/serving"
next_section_name: "Chapter 12: Production"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: What Are We Optimizing?
  - name: Cost Per Token, in Dollars
  - name: Two Regimes
  - name: The Step-Time Formula
  - name: KV Cache Economics
  - name: Sharding for Decode
  - name: MoE at Decode
  - name: Quantization for Inference
  - name: Speculative Decoding
  - name: What This Chapter Cannot Fix
  - name: Worked Problems
---

> **Skeleton.** Section structure only; the prose, the tables and the worked answers are still
> to be written. The brief for this chapter is the Chapter 11 section of `docs/structure.md`.

**Depends on:** [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) for HBM bandwidth and the
numeric formats, [Chapter 5]({{ '/pages/transformers' | relative_url }}) for the KV cache and
the attention variants, and [Chapter 7]({{ '/pages/moe' | relative_url }}) for the sparsity
that makes decode hard.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

> **To write.** Open by naming which intuitions invert. Inference has a completely different
> profile signature, a different roofline, and a different set of legal sharding strategies, so
> the training instincts built over the previous six chapters are **actively misleading** here.
> The reader needs to be told that in the first paragraph rather than discovering it.
>
> **Scope this chapter as arithmetic, not as a serving guide.** It answers "how much memory do I
> need to serve this, at what batch size, and what will a token cost", which the roofline
> settles cleanly and which every reader who trains a model immediately needs. It does not teach
> a serving system: that is
> [Chapter 12]({{ '/pages/serving' | relative_url }})'s much smaller job, and the book's focus is
> training. Keeping this chapter analytical is what lets it be short, standalone and correct
> without depending on a stack we do not run.
>
> **Everything in this chapter is [analytical]**, and the opening should say so once rather than
> tagging every claim. We do not run a serving stack, and pretending otherwise would undo the
> credit the measured chapters earned.

## What Are We Optimizing?

> **To write.** Latency, throughput, time to first token, and cost per token, plus the fact that
> these conflict rather than compose.
>
> Name the three workload shapes, because they want different answers and a reader optimizing
> the wrong one will be confused by every table in the chapter: offline batch, chat streaming,
> and agentic or long chain-of-thought.

## Cost Per Token, in Dollars

> **To write.** Short section, high value. Everything else in the book is measured in seconds and
> bytes, but the question a reader is usually being asked is what serving this costs, and the
> conversion is a one-liner: the GPU-hour rate divided by achieved tokens per second.
>
> It is also the natural unit for the latency-throughput tradeoff, since it makes the price of a
> latency target explicit instead of rhetorical. Carry the resulting figure into
> [Chapter 12]({{ '/pages/serving' | relative_url }}), which sizes a deployment against it.

## Two Regimes

> **To write.** Prefill is compute-bound and looks like training. Decode is
> memory-bandwidth-bound and dominated by weight loading.
>
> Ask the same compute-against-memory question of the linear operations and of attention,
> separately, for each regime, and fill in the resulting 2x2. Include the arithmetic for why
> decode is bandwidth-bound rather than asserting it: **it is the single most useful mental model
> for inference performance**, and a reader who derives it once will never mis-predict a decode
> step again.

## The Step-Time Formula

> **To write.** Minimum decode step time as `(params + batch * KV cache) / β_hbm`. If the reader
> takes one thing from the chapter, this is it, so give it room and give it a worked instance
> with real numbers.

## KV Cache Economics

> **To write.** Growth with context, memory cost against 192 GiB, and the batch size at which you
> run out. Grouped-query attention, latent attention and quantized caches as the three levers,
> each quantified rather than described.
>
> Tables sweeping batch size, as the source book does, because the saturation point and the OOM
> point are the whole story and a table shows both at once.

## Sharding for Decode

> **To write.** Why fully-sharded data parallelism is actively harmful at decode time and why
> data parallelism is pointless. What remains is tensor parallelism, plus the important twist that
> **when you are bandwidth-bound rather than FLOPs-bound you can shard past the training bound to
> buy latency**, which is the cleanest example in the book of the same hardware giving a different
> answer to a different question.
>
> Then KV cache sharding and the all-to-alls it costs.

## MoE at Decode

> **To write.** Where Chapters [5]({{ '/pages/transformers' | relative_url }}) and
> [7]({{ '/pages/moe' | relative_url }}) collide. `E / E_a` inflates the critical batch size to
> numbers that are hard to reach in practice, and expert placement interacts with the memory
> ceiling at the same time.
>
> **This is the hardest problem in the book and it deserves to be named as such.** Do not resolve
> it artificially; state the tension, give the arithmetic, and point at what production systems
> actually do about it.

## Quantization for Inference

> **To write.** [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) owns the formats and
> [Chapter 6]({{ '/pages/training' | relative_url }}) owns fp8 *training*, so this section is
> specifically the inference levers: weight-only quantization and KV cache quantization, and what
> each does to the step-time formula above rather than to the FLOP ceiling.
>
> **That distinction is the whole point at decode:** you are buying bandwidth, not FLOPs, which is
> why weight-only quantization helps decode far more than it helps prefill. A reader who
> internalises that will predict the effect of a quantization change correctly without being told.
>
> MI355X's fp6 and fp4 as the forward look, including the CDNA4 oddity from
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) that fp6 costs the same as fp4.

## Speculative Decoding

> **To write.** Short, but present, because it is the cleanest illustration that the step-time
> formula is the thing that governs: it wins precisely by converting spare FLOPs into fewer weight
> loads, which is the one trade a bandwidth-bound decode rewards.
>
> One derivation and an acceptance-rate sensitivity, then stop. The implementation lives in the
> serving engine, not in our stack, and saying so is more useful than a survey of methods.

## What This Chapter Cannot Fix

> **To write.** The closing section, and it is doing structural work: it hands off to
> [Chapter 12]({{ '/pages/serving' | relative_url }}) by naming the three problems that survive a
> perfect single-request roofline.
>
> 1. Requests arrive at different times and in different shapes, which is scheduling.
> 2. The KV cache is allocated dynamically and fragments, which is memory management.
> 3. Prefill and decode want different hardware, which is placement.
>
> Those three are also the honest reason serving engines exist and the reason we are not writing
> one, so ending on them sets up the next chapter's "hand it to vLLM" argument instead of letting
> it arrive as an anticlimax. **The two chapters being adjacent is what makes that handoff land**,
> which it did not in an earlier ordering that put a capstone between them.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number.

**Question 1:** Explain why decode throughput barely improves when you double the FLOPs.

{% details Click here for the answer. %}

To write. The step-time formula has no `C` in it, which is the whole answer.

{% enddetails %}

**Question 2:** Compute the maximum batch size for a given model and context length at 192 GiB.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 3:** Decide whether fp8 weights help at a given batch size.

> **To write.** The answer depends on whether weights or cache dominate the byte count at that
> batch size, which is why the question is worth asking rather than answering in general.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 4:** Convert a measured tokens per second into dollars per million tokens.

{% details Click here for the answer. %}

To write, and carry the figure into
[Chapter 12]({{ '/pages/serving' | relative_url }}).

{% enddetails %}
