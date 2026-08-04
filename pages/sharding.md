---
layout: distill
title: "Sharded Matrices and How to Multiply Them"
description: "You split the matrix across eight GPUs. What does that cost? Named-axis sharding notation, the four collectives and their cost on a switchless xGMI mesh, what RCCL actually achieves against what the wiring promises, and who inserts the collective: GSPMD or you."
date: 2026-08-04

section_number: 4

previous_section_url: "/pages/profiling"
previous_section_name: "Chapter 3: Profiling"

next_section_url: "/pages/transformers"
next_section_name: "Chapter 5: Transformer Math"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Notation for Sharded Arrays
  - name: The Four Collectives
  - name: RCCL in Practice
  - name: Measured Versus Spec Bandwidth
  - name: The Four Sharded-Matmul Cases
  - name: One Program, Many Processes
  - name: "Who Inserts the Collective: GSPMD or You?"
  - name: Is the Collective Overlapping?
  - name: Worked Problems
---

> **Skeleton.** Section structure only; the prose, the arithmetic and the measured
> sweep are still to be written. The brief for this chapter is the Chapter 4 section of
> `docs/structure.md`.

**Depends on:** [Chapter 1]({{ '/pages/rooflines' | relative_url }}) for rooflines,
[Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) for the xGMI mesh and its
bandwidths, and [Chapter 3]({{ '/pages/profiling' | relative_url }}) for reading a
trace, since half of this chapter is looking at collectives in one.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

> **To write.** Open on the keystone role this chapter plays. Without it,
> [Chapter 6]({{ '/pages/training' | relative_url }}) would have to introduce sharding
> notation, collective cost *and* five parallelism strategies at once. Every inequality
> in Chapters [6]({{ '/pages/training' | relative_url }}),
> [7]({{ '/pages/moe' | relative_url }}) and
> [11]({{ '/pages/inference' | relative_url }}) is a substitution into the cost model
> built here, so it is worth paying full price for it now.

## Notation for Sharded Arrays

> **To write.** Meshes, named axes, `A[I_X, J_Y]`, and the "unreduced" or partial-sums
> annotation that makes the collective cases below readable. Toy shapes first.
>
> **Introduce the notation and the JAX API together**, so the notation never feels like
> a parallel vocabulary invented for the book: `A[I_X, J_Y]` is the `PartitionSpec` the
> reader passes to `NamedSharding`. Being JAX-only throughout is what makes that
> one-to-one mapping possible, and it is the reason this book has no separate
> "programming in JAX" chapter.

## The Four Collectives

> **To write.** AllGather, ReduceScatter, AllReduce and AllToAll: what each does to a
> sharding, and what each costs.
>
> **Derive against the mesh, in one recap line and then two consequences.**
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) established the wiring:
> switchless, fully connected, seven direct xGMI links per GPU. Recap that in a sentence
> and do not re-derive it. Note that AMD sits in neither position the reader may arrive
> with, not a torus like a TPU pod and not a crossbar like an NVSwitch domain, so the
> TPU result that all-reduce cost is independent of axis size does not carry over
> cleanly.
>
> Then the two consequences, which are this chapter's own material:
>
> - **Physical topology is not the collective algorithm.** The links form a mesh, but
>   RCCL still chooses a ring or a tree schedule over it, so achieved cost follows the
>   algorithm rather than the wiring. That is exactly why the measured sweep below
>   matters more than the derivation does.
> - **The scale-up domain stops hard at eight.** Every cost in this chapter therefore
>   has two regimes with a cliff between them rather than one smooth curve. That
>   discontinuity drives placement decisions in Chapters
>   [6]({{ '/pages/training' | relative_url }}),
>   [7]({{ '/pages/moe' | relative_url }}) and
>   [12]({{ '/pages/serving' | relative_url }}), which is why those chapters can cite
>   one fact instead of re-arguing topology.

## RCCL in Practice

> **To write.** How collectives appear in the trace and in HLO: matching
> `all-reduce-start` to `all-reduce-done`, and reading `replica_groups` to work out the
> sharding actually in use. Inferring the parallelism strategy purely from replica
> groups is a genuinely useful trick and worth teaching properly, because it is how you
> audit a config you did not write.

## Measured Versus Spec Bandwidth

> **To write.** The most valuable section in the chapter, and the one that justifies
> shipping it in the first release.
>
> Sweep message size and device count, plot achieved against theoretical, and state the
> message size at which RCCL reaches asymptotic bandwidth. The source book does exactly
> this for NCCL and finds 370 GB/s against a claimed 450, and 150 GB/s at realistic LLM
> message sizes. **Nobody has published the AMD equivalent.**
>
> Two things to measure while the harness exists, because both are needed later and
> neither costs an extra script. First, all-reduce and reduce-scatter across two through
> eight devices, which is what Chapter 6 needs. Second, an 8-way all-to-all, because
> [Chapter 7]({{ '/pages/moe' | relative_url }}) predicts that a switchless full mesh
> should be unusually good at it and that prediction needs checking.
>
> Everything here is **[measured]**, on one baseboard, with
> [Appendix B]({{ '/pages/appendix-protocol' | relative_url }}) behind it. Say so once
> at the top of the section rather than tagging every figure.

## The Four Sharded-Matmul Cases

> **To write.** Which sharding of inputs and outputs requires which collective, and the
> resulting cost. This is the lookup table
> [Chapter 6]({{ '/pages/training' | relative_url }}) indexes into, so it should read
> like a table the reader will come back to rather than like a narrative.

## One Program, Many Processes

> **To write.** The missing prerequisite for every placement argument later in the book.
> A multi-node JAX job is one SPMD program running in as many processes as there are
> hosts: `jax.distributed.initialize`, arrays that are globally shaped but locally
> backed, and `jax.process_index` for the things that genuinely differ per host.
>
> **The reason this belongs here rather than in a capstone is that the order of the axes
> in the `Mesh` decides which collective crosses the NIC**, and that is precisely the
> question Chapters [6]({{ '/pages/training' | relative_url }}) and
> [7]({{ '/pages/moe' | relative_url }}) keep asking. A mesh laid out so that the expert
> axis straddles two baseboards is a one-line mistake with a large bill, and the reader
> cannot see it without this section.
>
> Keep it short and keep it honest: the mechanism is exact, the eight-GPU numbers are
> **[measured]**, and anything spanning hosts is **[analytical]** until we have a
> cluster.

## Who Inserts the Collective: GSPMD or You?

> **To write.** The missing half of the JAX story. Annotating an array with a
> `NamedSharding` and letting the compiler derive the collectives is the default, and
> for the strategies in [Chapter 6]({{ '/pages/training' | relative_url }}) it is
> usually the right one. `shard_map` hands you the per-device view and makes you write
> the collective yourself.
>
> The reader needs the distinction *here*, before
> [Chapter 7]({{ '/pages/moe' | relative_url }}), because MoE routing is the case where
> the compiler's choice is not good enough and a hand-written all-to-all is the norm.
>
> **Show the same sharded matmul both ways and diff the HLO.** That comparison teaches
> more about what GSPMD is doing than any amount of explanation.

## Is the Collective Overlapping?

> **To write.** Serialized against overlapped is immediately visible in the Trace
> Viewer, so this is a "look at the artifact" section rather than an explanation.
>
> Show the same workload with and without
> `--xla_gpu_enable_latency_hiding_scheduler=true` and
> `--xla_gpu_enable_highest_priority_async_stream=true`, both of which are already set
> in `scripts/transformer_block.py`, and give the measured step-time difference.
>
> **This section owns overlap for the whole book.**
> [Chapter 8]({{ '/pages/getting-to-roofline' | relative_url }})'s triage list opens
> with "are the collectives overlapping", which is the same question, and the two must
> not both teach it. This chapter teaches how to see it and what the flags do; Chapter 8
> gets one line and a cross-reference.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number.

**Question 1:** How long should an all-reduce of a 1 GB gradient buffer take on one
node, and on two?

> **To write.** The second half is **[analytical]** and should say so. It is also the
> first place the reader feels the eight-GPU cliff as a number rather than as a claim.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 2:** From a set of replica groups, name the parallelism strategy.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 3:** At what message size does RCCL stop being latency-bound?

{% details Click here for the answer. %}

To write, with the measured crossover quoted from the sweep above.

{% enddetails %}
