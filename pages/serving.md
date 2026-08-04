---
layout: distill
title: "Getting Your Model Into Production"
description: "You trained it in JAX. Now it has to serve traffic, and that probably isn't JAX. Exporting an Orbax checkpoint to something vLLM on ROCm will load, what the serving engine does with your weights, disaggregated prefill and decode, and what it all costs."
date: 2026-08-04

section_number: 12

previous_section_url: "/pages/inference"
previous_section_name: "Chapter 11: Inference"

next_section_url: "/pages/conclusion"
next_section_name: "Chapter 13: Conclusions"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Why This Chapter Is Not a JAX Serving Guide
  - name: The Handoff, Concretely
  - name: What the Serving Engine Is Doing With Your Weights
  - name: Disaggregated Prefill and Decode
  - name: What This Costs and How It Is Operated
  - name: Worked Problems
---

> **Skeleton.** Section structure only; the export recipe, the numbers and the worked answers are
> still to be written. The brief for this chapter is the Chapter 12 section of
> `docs/structure.md`.

**Depends on:** [Chapter 11]({{ '/pages/inference' | relative_url }}) for the single-request
arithmetic this chapter applies to many requests, and
[Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) for the bandwidth numbers that decide whether
disaggregation pays.

> **To write.** This is the honest answer to a question the source book never asks: what happens
> to the checkpoint after training finishes.

## Why This Chapter Is Not a JAX Serving Guide

> **To write.** State the scope decision plainly at the top of the chapter rather than letting the
> reader infer it. Production serving on AMD means exporting weights and running them under vLLM
> or SGLang. **That is not a compromise forced on us by JAX, it is what the whole industry does**,
> including the teams with the most AMD capacity. Four facts make it unambiguous:
>
> - **There is no JAX serving engine for ROCm.** JetStream was archived on 1 February 2026 and its
>   functionality moved into `vllm-project/tpu-inference`, which is TPU-only by name and design.
> - **MaxText's blessed inference path is itself an out-of-tree vLLM plugin.** Even on TPU,
>   Google's own answer to serving MaxText is now vLLM.
> - **AMD's MaxText fork is a training path** and documents no inference story.
> - **vLLM and SGLang on ROCm are genuinely good**, not merely popular: AITER attention backends,
>   MXFP4, and published disaggregated-serving results competitive with B200.
>
> So this chapter does not build a JAX serving stack, and it does not pretend the absence of one is
> a gap in the book. **It teaches the handoff, and enough of the serving concepts that the reader
> can size and tune what they hand off to.**

## The Handoff, Concretely

> **To write.** This is the section nobody has written and the reason the chapter exists.
>
> Orbax checkpoint to a HuggingFace-format checkpoint, what MaxText's conversion scripts do and
> where they break, quantization on the way out with AMD's Quark toolkit, then loading the result
> under vLLM on ROCm and confirming the outputs match.
>
> The fp8 format question from [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) shows up here
> for the third time, now as a checkpoint compatibility problem: a checkpoint quantized to
> gfx942's FNUZ variant is not the same artifact as one quantized to OCP fp8, and the reader needs
> to know which one they produced.
>
> **Verify this path end to end before writing.** It is the chapter's spine and the one thing here
> that could surprise us. It is also the natural place to close the loop on the capstones: the
> checkpoint being exported is the one
> [Chapter 9]({{ '/pages/llama' | relative_url }}) or
> [Chapter 10]({{ '/pages/deepseek' | relative_url }}) just produced.

## What the Serving Engine Is Doing With Your Weights

> **To write.** Continuous batching, chunked prefill with its roofline, paged attention and KV
> cache management, and prefix caching with its routing-affinity consequence.
>
> **Taught as properties of the workload rather than as a library tour**, which is what makes them
> transfer: these are the same arithmetic as
> [Chapter 11]({{ '/pages/inference' | relative_url }}), applied to many requests instead of one.
>
> Keep it tight. The goal is a reader who can read a vLLM configuration and predict what it will
> do, not a reader who could reimplement it.

## Disaggregated Prefill and Decode

> **To write.** Present it as the endpoint of an escalating argument rather than as an
> architecture: naive batching, then interleaved, then disaggregated, with the specific failure
> that motivates each step.
>
> The two real advantages are independent scaling and specialized sharding. The cost is moving KV
> cache over the network, which is a
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) bandwidth question, and **it is the best
> single demonstration in the book that our arithmetic still governs a stack we did not write.**

## What This Costs and How It Is Operated

> **To write.** Sizing a deployment against a service-level objective, using the
> dollars-per-token model from
> [Chapter 11]({{ '/pages/inference' | relative_url }}). Then briefly: what to autoscale on, load
> balancing under heterogeneous request shapes, and reliability.
>
> Short. **This is the part of the chapter most likely to become filler, so cap it** and resist
> the temptation to write a general operations guide.

> **Cut from an earlier draft**, and recorded here so it does not creep back: the
> prefill/generate/transfer thread anatomy of a JAX serving engine, multi-node serving as its own
> section, and any attempt to measure a JAX decode path. The first two are things a reader gets
> from vLLM's own documentation; the third is a porting project, not a writing project.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number.

**Question 1:** Given a request trace and a latency objective, size a deployment.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 2:** Compute the KV transfer cost of disaggregation and say whether it pays.

{% details Click here for the answer. %}

To write. The answer should turn on a bandwidth number from
[Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) rather than on a preference.

{% enddetails %}

**Question 3:** Given a trained checkpoint and a target quantization, work out the served memory
footprint per GPU.

{% details Click here for the answer. %}

To write.

{% enddetails %}
