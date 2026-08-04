---
layout: distill
title: "Mixture-of-Experts at Scale"
description: "Only a fraction of the parameters run per token, so why isn't it a fraction of the time? Routing and load imbalance, capacity against dropless, the three ways to implement an expert layer and what each costs, all-to-all on a switchless mesh, and expert parallelism."
date: 2026-08-04

section_number: 7

previous_section_url: "/pages/training"
previous_section_name: "Chapter 6: Training"

next_section_url: "/pages/getting-to-roofline"
next_section_name: "Chapter 8: Getting to Roofline"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Why the MoE Roofline Is Different
  - name: Routing
  - name: Load Imbalance
  - name: Capacity, Dropping, Padding, and Going Dropless
  - name: Three Ways to Implement an Expert Layer
  - name: Which of the Three You Can Get on AMD in JAX
  - name: All-to-All Dispatch and Combine
  - name: Expert Parallelism
  - name: Anatomy of Three Real Models
  - name: The Four Numbers to Log for Every MoE Run
  - name: Worked Problems
---

> **Skeleton.** Section structure only; the prose, the measurements and the worked
> answers are still to be written. The brief for this chapter is the Chapter 7 section of
> `docs/structure.md`.

**Depends on:** [Chapter 4]({{ '/pages/sharding' | relative_url }}) for collective costs
and `shard_map`, [Chapter 5]({{ '/pages/transformers' | relative_url }}) for MoE
parameter and FLOP accounting, and
[Chapter 6]({{ '/pages/training' | relative_url }}) for the five-part treatment that
expert parallelism reuses.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

> **To write.** The most technically distinctive chapter in the book and the one with the
> least prior art. Nearly everything in it is a failure mode that does not exist in dense
> models. [Chapter 5]({{ '/pages/transformers' | relative_url }}) did the accounting;
> this chapter does the systems.

> **Verify before writing.** Two things gate this chapter and both are cheap to settle.
> Which of the three implementations below does MaxText actually run on ROCm, and under
> which config fields. And does RCCL schedule an 8-way all-to-all across all seven links,
> as the mesh argument predicts. See the Open questions section of `docs/structure.md`.

## Why the MoE Roofline Is Different

> **To write.** A dense model's roofline predicts well. An MoE's does not, because the
> effective FLOP count depends on routing decisions made at runtime, and because the
> shape of the expert matmul is decided by data rather than by the config.
>
> Set up the gap between the naive prediction and reality here, with a number, and then
> spend the rest of the chapter closing it. The reader should finish this section knowing
> that the gap exists and roughly how big it is, and finish the chapter knowing which
> parts of it are recoverable.

## Routing

> **To write.** What a router computes, token-choice against expert-choice, the auxiliary
> load-balancing loss and what it trades against. Enough mechanism that the reader can
> reason about imbalance rather than only observe it.
>
> **Two lines on router numerics, because they are cheap and the failure they prevent
> looks like a hardware fault.** The router runs in fp32 even when the rest of the layer
> is bf16 or fp8, and it usually carries a z-loss on the logits. An unstable router
> produces a loss spike or a NaN, and in a book full of hardware the reader's first
> instinct will be to suspect the hardware. Name it once here so that instinct is
> corrected before [Chapter 10]({{ '/pages/deepseek' | relative_url }}).

## Load Imbalance

> **To write.** What it looks like in a trace: expert GEMMs of visibly unequal duration
> with the step gated on the slowest. Show the trace before explaining the mechanism.
>
> How to quantify it from a profile, and how it varies with the data and over the course
> of training, since an imbalance measured at step 100 is not the one you will live with
> at step 100,000.

## Capacity, Dropping, Padding, and Going Dropless

> **To write.** The quiet FLOP thief. Fixed capacity means padding when an expert is
> underfull and dropped tokens when it overflows, and **both are invisible in wall-clock
> time and very visible in achieved MFU**, which is what makes them worth a section.
> How to measure the waste from a profile, and how capacity trades against quality.
>
> **Then the dropless alternative, which is where modern implementations have landed.**
> Instead of padding to a fixed capacity you size each expert's matmul to whatever
> actually arrived (`capacity_factor: -1` in MaxText; confirm the field name when
> writing). No padding, no dropping, exact FLOPs, and in exchange the expert matmul
> becomes a ragged shape rather than a rectangular one.
>
> **That trade is the hinge of the whole chapter:** it converts a FLOP-efficiency problem
> into a kernel-availability problem, and on AMD in JAX the kernel is the part that is
> missing. Set it up here and let the next two sections pay it off.

## Three Ways to Implement an Expert Layer

> **To write.** The chapter's spine. The choice is made in one or two config fields and it
> moves the FLOP bill by a factor of `E / E_a`, which for a fine-grained model is an order
> of magnitude. Give the arithmetic for each rather than a ranking, because the ranking
> flips with `E`, `E_a` and the hardware.
>
> 1. **Dense masked compute.** Every device runs every expert over every token and
>    multiplies by a one-hot mask. No dispatch collective at all, and every matmul is a
>    plain dense GEMM at full kernel efficiency, which is why toy implementations look
>    deceptively good. It also does `E / E_a` times the activated FLOPs, which is
>    precisely the sparsity you bought, handed back: 4x for Mixtral's 8 experts with 2
>    active, and 16x for a fine-grained model at 128 and 8. Survivable in the first case,
>    indefensible in the second.
> 2. **One-hot dispatch at fixed capacity.** The GShard formulation: an einsum routes
>    tokens into an `[E, capacity, D]` buffer, so the expert matmul is a dense GEMM of
>    statically known shape and the compiler is happy. You pay the padding and the
>    dropping from the section above.
> 3. **Sort and grouped GEMM.** Sort tokens by expert, then one ragged matmul over
>    variable-sized groups. Dropless, no padding, and the FLOP count is exactly the
>    activated one. The entire cost moves into needing a grouped or ragged GEMM kernel
>    that stays fast on ragged shapes.
>
> Say which MaxText knobs select which, since that is what the reader edits. And note that
> a reader whose stack only offers option 1 is not doing it wrong: they need to know what
> it costs them, not that it is inelegant.

## Which of the Three You Can Get on AMD in JAX

> **To write.** How expert matmuls actually execute, how those kernels appear in Kernel
> Stats, and why their efficiency depends on the token distribution.
>
> **The AMD specifics here are the most important finding in the outline and they need
> verifying before this is written.** AMD's fast MoE kernels live in AITER: fused routing,
> block-scaled grouped GEMM, and the FlyDSL work that is superseding hand-written
> Composable Kernel templates. All of it is reached from PyTorch. The JAX bridge
> (`ROCm/jax-aiter`, over XLA FFI) exposes attention and *dense* GEMM, and as far as we
> can tell **exposes no grouped or ragged MoE GEMM at all.**
>
> So option 3 above is the one a JAX user on ROCm cannot simply pick up: the expert
> matmuls are XLA-generated, or they are yours to write in Pallas or Triton, and Pallas on
> ROCm routes through the Triton backend and is labelled experimental. Mosaic GPU is
> NVIDIA-only.
>
> **If that holds, lead with it rather than burying it.** The gap between AMD's best MoE
> kernels and what a JAX user can reach is the single largest performance factor in this
> chapter, and quantifying it is something nobody has published. Two measurements settle
> it, and they are the most valuable numbers in the chapter:
>
> - XLA-generated expert GEMMs against the AITER figures AMD publishes, stated as a ratio.
> - The three implementations above against each other, same model, same tokens, same `E`
>   and `E_a`, with only the implementation varying. A reader forced away from option 3
>   needs to know whether 1 or 2 is the better consolation prize, and the answer is not
>   obvious.
>
> Treat Composable Kernel as the legacy path it now is, and date every claim in this
> section, because it is the fastest-moving material in the book.

## All-to-All Dispatch and Combine

> **To write.** The two collectives that define MoE performance once experts are spread
> over devices. Derive the cost from
> [Chapter 4]({{ '/pages/sharding' | relative_url }})'s model, including the top-`E_a`
> ragged variant where cost scales with `min(E_a / |Ex|, 1)`.
>
> **The AMD-specific result here is a good one, and the chapter should not bury it under
> the bad one.** All-to-all is the collective a switchless full mesh is best suited to:
> every device has a direct link to every peer, so each of the seven links can carry one
> peer's share concurrently, with no switch to contend for and no multi-hop forwarding. A
> ring schedule lights a fraction of the links at any instant; an 8-way all-to-all can in
> principle light all seven at once, which puts the whole 896 GB/s of per-GPU egress in
> play. Predict it, then check it against
> [Chapter 4]({{ '/pages/sharding' | relative_url }})'s sweep, and be honest that RCCL's
> choice of schedule is what decides whether the prediction lands. **Inside a baseboard,
> MoE dispatch should be cheap**, and that is worth stating clearly because the reader
> expects the opposite.
>
> Then the cliff, which is the same fact from the other side. The ninth GPU is over the
> NIC, so an expert axis that crosses the baseboard trades a 128 GB/s direct link for a
> share of node egress. That is the central placement question of the chapter: **keep `Ex`
> inside the node and spend the slow axis on something that tolerates it.** Mark the
> inter-node arithmetic **[analytical]**.

## Expert Parallelism

> **To write.** The full five-part treatment from
> [Chapter 6]({{ '/pages/training' | relative_url }}), then how expert parallelism
> composes with data, fully-sharded, tensor and pipeline parallelism, and which mesh axis
> the all-to-all should land on given the node topology. **This is where most real MoE
> performance is won or lost.**
>
> **Open with memory rather than with communication**, which is the opposite of how expert
> parallelism is usually introduced and the more honest motivation. An MoE has `E` times
> the MLP parameters at `E_a / E` of the MLP FLOPs, so relative to a dense model of the
> same quality it is memory-hungry and FLOP-light, and the optimizer state scales with the
> total parameter count rather than the activated one. That is why the expert axis exists
> at all: you shard by expert because the weights do not fit, and only then discover you
> have bought an all-to-all.
>
> The dispatch tensors are also large enough that remat policy interacts with routing,
> which is worth a sentence pointing back at
> [Chapter 5]({{ '/pages/transformers' | relative_url }}).
>
> This is also the chapter's best argument for `shard_map` over automatic partitioning,
> per [Chapter 4]({{ '/pages/sharding' | relative_url }}): expert routing is the canonical
> case where you want to write the collective yourself.

## Anatomy of Three Real Models

> **To write.** Mixtral 8x7B as the simple case, Qwen3 30B-A3B as the fine-grained one,
> and DeepSeek v3 as the elaborate one, with its shared experts that run for every token.
> For each, the published parallelism configuration and an explanation of why each degree
> was chosen.
>
> Choosing these three is not arbitrary: all three are in the pre-optimised model list for
> AMD's ROCm MaxText fork, so they are the MoE models we can actually run and measure.
>
> DeepSeek's multi-head latent attention is accounted for in
> [Chapter 5]({{ '/pages/transformers' | relative_url }}) and its serving consequences in
> [Chapter 11]({{ '/pages/inference' | relative_url }}). Reference both; do not re-derive.

## The Four Numbers to Log for Every MoE Run

> **To write.** Short, and the most reusable thing in the chapter:
>
> 1. Tokens per expert, as a histogram rather than a max.
> 2. The dropped-token fraction, or the ragged-shape distribution if dropless.
> 3. The achieved efficiency of the expert GEMM against a dense GEMM of equivalent size.
> 4. The all-to-all share of step time.
>
> Each has a named source in a profile, each maps onto one of the failure modes above, and
> **none of them is on by default.** A reader who instruments these four can diagnose
> their own MoE without this chapter, which is the correct ambition for it.
>
> [Chapter 10]({{ '/pages/deepseek' | relative_url }}) is where these four get used on a
> real run, so keep the definitions precise enough to be reused verbatim.

> **Scope boundary with Chapter 11, stated so the writing does not drift.** This chapter
> owns MoE *mechanism and training*: routing, imbalance, capacity, dispatch implementation,
> expert parallelism, all-to-all.
> [Chapter 11]({{ '/pages/inference' | relative_url }}) owns MoE *at decode*, because the
> reader needs the two-regime model before MoE decode makes any sense. The seam is the
> critical batch size: this chapter derives why sparsity inflates it, Chapter 11 shows why
> that inflation is close to fatal when you are serving.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number.

**Question 1:** From a trace, estimate routing imbalance and its cost in step time.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 2:** Determine from a profile whether a run is dropping tokens.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 3:** Decide whether a given expert-parallel degree helps or hurts, and compute
the all-to-all cost of crossing one node boundary against staying inside.

{% details Click here for the answer. %}

To write. The inter-node half is **[analytical]** and the answer should say so.

{% enddetails %}

**Question 4:** For a given `E` and `E_a`, work out the grouped-GEMM efficiency at which
sort-and-group stops beating dense masked compute.

> **To write.** This is the calculation a reader on a stack without a ragged kernel
> actually has to do, which is what makes it the most useful problem in the chapter rather
> than the most contrived.

{% details Click here for the answer. %}

To write.

{% enddetails %}
