---
layout: distill
title: "Getting Your Model Into Production"
description: "You trained it in JAX. Now it has to serve traffic, and that probably isn't JAX. Why the handoff to vLLM or SGLang is the correct engineering decision rather than a shortfall, what a serving engine does with your weights, and what disaggregating prefill from decode costs."
date: 2026-08-04

section_number: 12

previous_section_url: "/pages/11-inference"
previous_section_name: "Chapter 11: Inference"

next_section_url: "/pages/13-conclusion"
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
  - name: References
---

> **Draft.** The chapter's spine, the checkpoint export path, is blocked on running it end
> to end: it is the one thing here that could surprise us and writing it from
> documentation would be exactly the wrong call. The scope argument, the serving-engine
> concepts and the disaggregation arithmetic are written.

**Depends on:** [Chapter 11]({{ '/pages/11-inference' | relative_url }}) for the single-request
arithmetic this chapter applies to many requests, and
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) for the bandwidth numbers that decide whether
disaggregation pays.

**Here is the question the source book never asks: what happens to the checkpoint after
training finishes?** You have an Orbax directory full of sharded arrays, a training script
that knows how to load it, and a product that needs to answer HTTP requests. Those are very
different problems, and the bridge between them is this chapter.

## Why This Chapter Is Not a JAX Serving Guide

**Production serving on AMD means exporting weights and running them under vLLM or
SGLang.** That is not a compromise forced on us by choosing JAX, it is what the whole
industry does, including the teams with the most AMD capacity. Stating it at the top is
better than letting the reader discover it at the bottom.

**Four facts make it unambiguous.**

**There is no JAX serving engine for ROCm.** JetStream, which was the JAX-native
throughput-oriented inference engine, was archived on 1 February 2026. Its functionality
moved into `vllm-project/tpu-inference`, which is TPU-only by name and by design.

**MaxText's blessed inference path is itself a vLLM plugin.** Even on TPU, Google's answer
to serving a MaxText checkpoint is now vLLM rather than anything in MaxText. If the
framework's own authors hand off to vLLM on their preferred hardware, doing the same on
AMD is not a workaround.

**AMD's MaxText work is a training path** and documents no inference story. That is a
reasonable division of labour rather than a gap.

**vLLM and SGLang on ROCm are genuinely good**, not merely available: AITER attention
backends, MXFP4 support, and published disaggregated-serving results competitive with
current NVIDIA parts. The thing you are handing off to is better at serving than anything
you would write.

**So this chapter does not build a JAX serving stack, and it does not pretend the absence
of one is a hole in the book.** It teaches the handoff, and it teaches enough of the
serving concepts that you can size and tune what you hand off to. That is a genuinely
useful chapter and a much less risky one.

> **Verified against:** the JetStream archival notice and `vllm-project/tpu-inference`, both
> checked 4 August 2026. This is the fastest-moving paragraph in the book; if you are
> reading it a year later, check whether the answer has changed.

## The Handoff, Concretely

<!-- BLOCKED, and this is the chapter's spine. It has to be run before it is written.

     What it has to deliver: Orbax checkpoint to a HuggingFace-format checkpoint, what
     MaxText's conversion scripts do and where they break, quantization on the way out
     (AMD's Quark toolkit, and the fp8 FNUZ-versus-OCP format question from Chapter 2
     showing up again as a checkpoint compatibility problem), then loading it under vLLM
     on ROCm and confirming the outputs match.

     Why it cannot be written from documentation: this is the one section in the book
     whose failure mode is silent. A conversion script that runs to completion and
     produces a checkpoint that generates plausible-but-wrong text is exactly what will
     happen if the fp8 format or the attention-head permutation is mishandled, and only
     an output comparison catches it. Writing "run this script" without having run it
     would be the single most likely thing in this book to waste a reader's week.

     The verification, which docs/structure.md calls the chapter's only hard dependency
     and estimates at a day of work:
       1. Take a checkpoint from Chapter 9 or Chapter 10 (so the loop closes on the
          capstones rather than on an arbitrary model).
       2. Convert to HuggingFace format with MaxText's conversion script.
       3. Quantize with Quark, targeting fp8. Check which fp8 variant it emits and
          whether it matches the serving GPU's architecture (gfx942 wants FNUZ, gfx950
          wants OCP). This is where Chapter 2's format split becomes a correctness bug.
       4. Load under vLLM on ROCm.
       5. Compare outputs against the JAX model on the same prompts at temperature 0.
          Not "looks reasonable": token-for-token, or at minimum matching logprobs to a
          stated tolerance.
     If the conversion scripts are broken for a model we care about, that is worth
     knowing early and is itself publishable.

     Also to verify while in there: whether Quark is the right recommendation versus
     llm-compressor or vLLM's own quantization path, and whether the MaxText conversion
     scripts cover the specific model the capstones train. -->

## What the Serving Engine Is Doing With Your Weights

**Taught as properties of the workload rather than as a library tour, because that is what
transfers.** These are the same arithmetic as
[Chapter 11]({{ '/pages/11-inference' | relative_url }}), applied to many requests instead of
one. The goal is a reader who can look at a vLLM configuration and predict what it will do,
not a reader who could reimplement it.

**Continuous batching answers "requests arrive at different times".** Rather than
assembling a batch, running it to completion, and starting another, the scheduler
maintains a running set of sequences and swaps finished ones out for waiting ones at every
step. The arithmetic reason it matters is in
[Chapter 11]({{ '/pages/11-inference' | relative_url }})'s step-time table: throughput depends
on the batch being *full*, and with static batching a batch of 32 where 30 sequences have
finished is a batch of 2 paying batch-32 latency. **Continuous batching is what makes the
throughput numbers in that table reachable rather than theoretical.**

**Paged attention answers "the cache fragments".** Instead of reserving contiguous memory
for each sequence's maximum possible length, the cache is allocated in fixed-size blocks
(typically 16 tokens) with a per-sequence block table, exactly like virtual memory. The
consequence for your sizing arithmetic is that you can use nearly all of the cache memory
rather than the fraction a contiguous allocator would leave usable, so
[Chapter 11]({{ '/pages/11-inference' | relative_url }})'s "520 sequences at 8k context" is
approximately achievable instead of optimistic by 2x.

**Chunked prefill answers "a long prefill stalls every decode in flight".** Split a long
prompt into chunks and interleave them with decode steps, so that one 100k-token prefill
does not add seconds to the TPOT of every other request. The trade is visible in
[Chapter 11]({{ '/pages/11-inference' | relative_url }})'s two-regime table: prefill is
compute-bound and decode is bandwidth-bound, so a step that mixes them uses both resources
at once and is more efficient in aggregate, at the cost of a longer TTFT for the chunked
request. **Chunk size is the knob, and the roofline tells you the floor**: chunks below the
247-token ridge point are memory-bound and waste the prefill's one advantage.

**Prefix caching answers "many requests share a prompt".** If two requests begin with the
same 2000-token system prompt, the second can reuse the first's cached keys and values and
skip that prefill entirely. Nearly free, and enormous for agentic workloads where every
request carries the same tool definitions.

**Its consequence is the one people miss: prefix caching makes request routing stateful.**
The cache lives on a specific replica, so a load balancer that round-robins destroys the
hit rate. **Prefix-aware routing is worth more than most other tuning available to you** on
workloads with shared prompts, and it is a property of your infrastructure rather than of
the engine.

## Disaggregated Prefill and Decode

**Presented as the endpoint of an escalating argument, because each step is motivated by a
specific failure of the previous one.**

**Step 1: naive batching.** Assemble a batch, prefill it, decode it to completion. Fails
because sequences finish at different times, so the batch empties and throughput collapses.
Fixed by continuous batching.

**Step 2: interleaved continuous batching.** One GPU pool handles both prefill and decode,
scheduling whichever is pending. Fails for the reason in
[Chapter 11]({{ '/pages/11-inference' | relative_url }}): prefill and decode want different
hardware and different sharding, and every prefill chunk you run is a decode step you did
not. Chunked prefill mitigates it and does not remove it.

**Step 3: disaggregation.** Separate GPU pools for prefill and decode. The prefill pool
runs compute-bound work at high batch and ships the resulting KV cache to the decode pool,
which runs bandwidth-bound work.

**The two real advantages are independent scaling and specialized sharding.** You can size
the pools to your actual prompt-to-output ratio rather than compromising, and each pool can
use the sharding that suits it: heavy tensor parallelism for decode, per
[Chapter 11]({{ '/pages/11-inference' | relative_url }})'s argument that you can shard past
the training bound, and a training-like configuration for prefill.

**The cost is moving the KV cache over the network, and this is a
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) bandwidth question.** For Llama 3 70B at
8k context, one sequence's cache is 2.68 GB. Over a 400 Gbps NIC at 50 GB/s:

```
2.68e9 / 50e9 = 54 ms
```

**Compare that against what prefill cost in the first place**: 8192 tokens through a 70B
model is `2 * 70.6e9 * 8192 = 1.16e15` FLOPs, which on 8 MI300X at
`8 * 1307.4e12` FLOP/s takes 111 ms at 100% MFU, or about 280 ms at a realistic 40%.

**So the transfer is roughly 20% of the prefill it saves, and disaggregation pays.** Not by
a huge margin, and the margin depends on three things worth naming: the number of NICs you
can use in parallel for the transfer, whether the transfer overlaps with the next prefill,
and the context length, since both prefill cost and cache size grow with it but prefill
grows faster once attention dominates.

**Two levers that improve the trade substantially.** Send the cache in fp8, halving the
transfer, which is nearly free if you were going to quantize the cache anyway. And send it
layer by layer as prefill produces it, so the transfer overlaps the prefill instead of
following it, at which point the cost is close to zero.

**This is the best single demonstration that the book's arithmetic still governs a stack we
did not write.** Nothing above required knowing how vLLM is implemented: it required the
KV cache size from [Chapter 5]({{ '/pages/5-transformers' | relative_url }}), the network
bandwidth from [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}), and the two-regime model
from [Chapter 11]({{ '/pages/11-inference' | relative_url }}).

All of this is **[analytical]**, and the network figure especially: we have no multi-node
allocation, so 50 GB/s is a specification rather than a measurement.

## What This Costs and How It Is Operated

**Sizing a deployment is [Chapter 11]({{ '/pages/11-inference' | relative_url }})'s dollar
model plus a request rate.** Take the per-node throughput at the batch size your latency
SLO permits, divide your peak request rate by it, and round up. Then add capacity for the
gap between average and peak, which is usually the larger number.

Worked, for Llama 3 70B in bf16 at 8k context with a 40 ms TPOT budget, using
[Chapter 11]({{ '/pages/11-inference' | relative_url }})'s second worked problem:

- One 8-GPU node delivers about 14300 output tokens per second at batch 512, or about half
  that once scheduling reality is accounted for. Call it **7000 tokens per second per
  node.**
- A workload of 20 requests per second averaging 500 output tokens needs
  `20 * 500 = 10000` tokens per second, so **two nodes**, and a third for headroom and
  rolling deployments.
- At $2 per GPU-hour that is $48 per hour for three nodes, against a throughput of 10000
  tokens per second, which is **$0.44 per million tokens delivered** rather than the $0.31
  the hardware could theoretically produce. The difference is utilization, and it is the
  gap between an engineering number and an invoice.

**Three operational notes, kept short because this is the part of the chapter most likely
to become filler.**

**Autoscale on queue depth or time-to-first-token, not on GPU utilization.** A decode-bound
server shows high GPU utilization while it is bandwidth-saturated and idle in every way
that matters, so utilization is close to meaningless as a scaling signal. Queue wait time is
what your users experience.

**Load balancing has to account for request shape, and prefix affinity beats round-robin.**
A replica running one 100k-token prefill is unavailable in a way that a replica running 400
short decodes is not, and prefix caching makes replica choice consequential, per the
section above.

**Reliability, in one sentence: a GPU falling out of an 8-way tensor-parallel group takes
the whole replica down**, because tensor parallelism has no redundancy, so the unit of
failure is the replica rather than the GPU and your capacity planning should be in replicas.

## Worked Problems

**Question 1:** You have a workload with 4000-token prompts and 100-token outputs, 50
requests per second. Should you disaggregate prefill from decode, and how would you split
the pools?

{% details Click here for the answer. %}

**First, work out which regime dominates**, because that is what decides whether
disaggregation helps at all.

- Prefill FLOPs per request: `2 * 70.6e9 * 4000 = 5.6e14`. At 40% MFU on 8 MI300X that is
  `5.6e14 / (0.4 * 8 * 1307.4e12) = 135 ms` of node time per request.
- At 50 requests per second, prefill alone needs `50 * 0.135 = 6.75` nodes.
- Decode: 100 tokens per request at 50 requests per second is 5000 tokens per second, which
  at 7000 tokens per second per node is **0.71 nodes.**

**This workload is 90% prefill**, which is the important finding and the answer to the
question. Prompt-heavy workloads like retrieval-augmented generation and agentic tool use
look nothing like the chat workloads most serving advice assumes.

**So yes, disaggregate, and split roughly 9 prefill nodes to 1 decode node.** The wins are
larger here than in the general case: the prefill pool can be sharded and configured purely
for compute throughput, and the single decode node can use aggressive tensor parallelism for
latency without wasting FLOPs, since it is bandwidth-bound anyway.

**Check the transfer cost before committing.** 4000 tokens of Llama 3 70B cache is 1.31 GB,
so 54 ms of transfer at 25 GB/s effective, against 135 ms of prefill: 40%, which is worse
than the 8k-context case because prefill scales faster than cache size. Send the cache in
fp8 to halve it, and overlap it with prefill, or the transfer eats the win.

{% enddetails %}

**Question 2:** You quantized a checkpoint to fp8 on an MI300X development box and deployed
it to MI355X servers. The model loads, generates fluent text, and scores near zero on your
evaluations. What happened?

{% details Click here for the answer. %}

**The fp8 format changed between the two generations**, per
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}). gfx942 (MI300X) implements the FNUZ
variants of E4M3 and E5M2, with an exponent bias of 8, no infinities and a maximum
representable value of 240. gfx950 (MI355X) implements the OCP standard variants, with bias
7 and a maximum of 448.

**Identical bits mean different numbers**, and specifically the bit pattern holding the
largest finite value on MI300X decodes to NaN on MI355X. The weights are being reinterpreted
rather than converted.

**Fluent-but-wrong is exactly the expected symptom**, which is what makes this dangerous:
the model is not broken, it is subtly rescaled, so it produces grammatical text with no
factual grounding and no error anywhere in the logs.

**The fix** is the conversion vLLM and SGLang carry for the reverse direction: patch the
NaN encoding and rescale, since identical bits mean half the value on the older part. The
better fix is to quantize on hardware matching the deployment target, or to keep the
checkpoint in bf16 and quantize at load time.

**And the general habit:** any time a checkpoint crosses a generation boundary, run an
output comparison at temperature 0 before believing it. This is the reason
[The Handoff, Concretely](#the-handoff-concretely) insists on token-level output matching
rather than eyeballing a sample.

{% enddetails %}

## References

**The scope argument.**

- [JetStream](https://github.com/AI-Hypercomputer/JetStream) (AI-Hypercomputer). The
  archival notice, dated 1 February 2026, and the redirect to `tpu-inference`.
- [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) (vLLM). The
  successor, and the evidence that it is TPU-only.

**Serving engines on ROCm.**

- [vLLM on ROCm installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
  (vLLM). The supported path onto Instinct hardware.
- [SGLang](https://docs.sglang.ai/) (SGLang). The alternative, with strong AMD support.
- [AMD Instinct inference optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/index.html)
  (AMD). AMD's own serving guidance, including AITER backends and the flags worth knowing.

**Concepts.**

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
  (Kwon et al., 2023). Paged attention and continuous batching, as designed.
- [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)
  (Agrawal et al., 2023). Chunked prefill and the argument for mixing the two regimes in
  one step.
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670)
  (Zhong et al., 2024). The disaggregation argument and its KV-transfer accounting.

**Export and quantization.**

- [AMD Quark](https://quark.docs.amd.com/latest/) (AMD). The quantization toolkit for the
  export path, and where the fp8 variant question has to be answered.
- [Orbax](https://orbax.readthedocs.io/) (Google). The checkpoint format on the JAX side.
- [MaxText checkpoint conversion](https://github.com/AI-Hypercomputer/maxtext/tree/main/tools)
  (AI-Hypercomputer). The conversion scripts the handoff section has to verify.
