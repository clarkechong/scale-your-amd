---
layout: distill
title: "All the Transformer Math You Need"
description: "How many parameters, FLOPs and bytes, exactly? Per-layer accounting for the MLP, attention and the vocabulary projection, when attention starts to matter, what sparsity does to arithmetic intensity, and the KV cache that decides what you can serve. Dense and Mixture-of-Experts side by side."
date: 2026-08-04

section_number: 5

previous_section_url: "/pages/sharding"
previous_section_name: "Chapter 4: Sharding"

next_section_url: "/pages/training"
next_section_name: "Chapter 6: Training"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: The Counting Rule
  - name: Per-Layer Accounting
  - name: When Does Attention Matter?
  - name: MoE Accounting
  - name: KV Cache and the Attention Variants That Shrink It
  - name: Gradient Checkpointing
  - name: MFU, and Why It Is Not Hardware Utilization
  - name: Summary Table
  - name: Worked Problems
---

> **Skeleton.** Section structure only; the prose, the derivations and the worked
> answers are still to be written. The brief for this chapter is the Chapter 5 section
> of `docs/structure.md`.

**Depends on:** [Chapter 1]({{ '/pages/rooflines' | relative_url }}) for arithmetic
intensity and [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) for the constants
that turn a symbolic ridge point into a number. Assumes you have seen a Transformer
before, but not that you have ever counted its FLOPs.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

> **To write.** This chapter is the notational backbone for everything after it, and it
> should be reference-dense rather than argumentative: the reader will come back to it
> to look things up. The one place to be argumentative is the attention crossover,
> because that result is what licenses the rest of the book to model a Transformer as a
> stack of MLPs.
>
> **Mixture-of-Experts accounting appears here, alongside dense accounting, not in an
> appendix.** Treating sparsity as a footnote is one of the source book's clearer
> mistakes given that most frontier models are sparse, and the fix costs one section
> rather than a chapter.

## The Counting Rule

> **To write.** For a contraction, FLOPs = 2x the product of all dimensions, with batch
> and contracting dimensions counted once. Then forward against backward, and the
> derivation of the `6 * params * tokens` rule of thumb: roughly 2 for the forward pass
> and 4 for the backward, because the backward pass computes gradients with respect to
> both the inputs and the weights.

$$t_{\text{math}} = \frac{2 \cdot B \cdot D \cdot F}{C}$$

> **To write.** Keep display math for the handful of results the reader will want to
> quote, and inline arithmetic with real numbers for everything else, per the house
> style. An expression like $2BDF / C$ inline is fine; a page of derivation in display
> math is not.

## Per-Layer Accounting

> **To write.** Work through each block and give both parameters and training FLOPs:
>
> - **MLP.** `3DF` parameters with a gated einsum, `18BTDF` training FLOPs.
> - **Attention.** `2D(N+K)H` parameters for grouped-query attention, and
>   `24BTDNH + 12BT^2NH` training FLOPs, with the second term being the one that grows
>   with sequence length.
> - **Norms and the vocabulary projection.** Small in parameters, and the vocabulary
>   projection is not small in FLOPs at short sequence lengths, which surprises people.
>
> Notes on multi-head, multi-query and grouped-query attention, and on pre-norm against
> post-norm, kept to what changes the count.

## When Does Attention Matter?

> **To write.** The crossover result, which is the licence for the rest of the book to
> model a Transformer as a stack of MLPs. Set the attention term against the MLP term
> and solve for the sequence length at which they meet.
>
> **Give the crossover for real models rather than in the abstract.** A reader wants to
> know whether it applies to the thing they are training, and the abstract inequality
> does not tell them that. Name two or three models and where each one lands.

## MoE Accounting

> **To write.** Total against activated parameters, sparsity as `E / E_a`, and then the
> result that matters for every later chapter: **an MoE's effective arithmetic intensity
> is `E_a / E` of the dense equivalent, so its critical batch size is `E / E_a` times
> larger.**
>
> Derive the MoE critical batch size on MI300X here, with real numbers, because
> [Chapter 7]({{ '/pages/moe' | relative_url }}) uses it as a given and
> [Chapter 11]({{ '/pages/inference' | relative_url }}) shows why that inflation is
> close to fatal at decode.
>
> Shared experts and fine-grained experts as modifiers on the count, not as separate
> topics: both change `E` and `E_a` and nothing else about the arithmetic.

## KV Cache and the Attention Variants That Shrink It

> **To write.** Shape `[2, S, L, K, H]`, size in bytes, and the observation that a
> handful of long-context sequences can exceed the parameter memory. That observation is
> what sets up [Chapter 11]({{ '/pages/inference' | relative_url }}).
>
> **Multi-head latent attention belongs here, not in the MoE chapter.** MLA is an
> attention mechanism that happens to appear in a model that is also sparse, and its
> content is exactly the accounting this section is already doing. So run the sequence
> multi-head, then multi-query, then grouped-query, then latent, asking the same question
> each time: how many bytes of cache per token, and what did you pay in parameters or
> quality to get there.
>
> Two things follow from putting it here. [Chapter 7]({{ '/pages/moe' | relative_url }})
> gets to be purely about sparsity instead of becoming the DeepSeek chapter, and this
> section gets to finish its own KV story instead of deferring a third of it forward. The
> serving economics stay in [Chapter 11]({{ '/pages/inference' | relative_url }}).

## Gradient Checkpointing

> **To write.** Two named policies with their FLOP costs, motivated by an actual
> activation-memory figure for a model we care about rather than in the abstract. The
> reader should finish the section able to say what remat buys them in bytes and costs
> them in FLOPs for their own model.

## MFU, and Why It Is Not Hardware Utilization

> **To write.** This has to be here, because every chapter from
> [Chapter 7]({{ '/pages/moe' | relative_url }}) onward quotes an MFU figure.
>
> Model FLOPs utilization is the `6 * params * tokens` count above, over elapsed time,
> over [Chapter 2]({{ '/pages/amd-gpus' | relative_url }})'s dense peak. Hardware FLOPs
> utilization counts the FLOPs the device actually issued, which rematerialization
> inflates: **a run with full remat can sit at 55% HFU and 40% MFU with nothing whatever
> wrong with it.** Published figures rarely say which they are. This book always says.
>
> It sits immediately after gradient checkpointing on purpose, because remat is exactly
> what separates the two numbers.

## Summary Table

> **To write.** Dense and MoE side by side, covering parameters, forward FLOPs, training
> FLOPs, activation bytes and KV cache bytes per token. This is the page people will
> screenshot, so it is worth over-investing in and worth keeping to one screen.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number.

**Question 1:** Back out achieved FLOPs utilization from a published training cost.

> **To write.** The DeepSeek v3 version of this exercise is excellent and reusable: the
> reported GPU-hours, the parameter count and the token count are all public, so the
> reader can recover the MFU and then ask whether it is plausible. It also forces them
> to notice which of MFU and HFU the published figure was.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 2:** Compute the KV cache for Llama 3 70B at 128k context and say what it
means for the batch size you can serve on one MI300X.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 3:** For a given `E` and `E_a`, compute the MoE critical batch size on
MI300X and compare it with the dense figure from
[Chapter 2]({{ '/pages/amd-gpus' | relative_url }}).

{% details Click here for the answer. %}

To write.

{% enddetails %}
