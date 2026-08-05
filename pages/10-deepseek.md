---
layout: distill
title: "Training DeepSeek-V2-Lite on MI300X"
description: "The sparse capstone, and the hardest thing in the book we can run end to end. Same method as the dense chapter, harder model: routing and imbalance on a real run, expert parallelism against the eight-GPU mesh ceiling, and what latent attention does to the memory profile."
date: 2026-08-04

section_number: 10

previous_section_url: "/pages/9-llama"
previous_section_name: "Chapter 9: Llama 3"

next_section_url: "/pages/11-inference"
next_section_name: "Chapter 11: Inference"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: The Configuration, and Why Each Degree
  - name: Routing and Imbalance in a Real Run
  - name: Expert Parallelism and All-to-All Placement
  - name: What Latent Attention Does to the Memory Profile
  - name: MFU Against the Roofline
  - name: Fallbacks If This Model Disappoints
  - name: Worked Problems
  - name: References
---

> **Not started, and blocked twice.** Like
> [Chapter 9]({{ '/pages/9-llama' | relative_url }}), everything here is a measurement of a
> training run we have not done. It is additionally blocked on the open kernel question
> from [Chapter 7]({{ '/pages/7-moe' | relative_url }}): until we know which expert-layer
> implementation a JAX user on ROCm can actually run fast, this chapter cannot report an
> MFU figure that means anything, because the answer would be an artifact of an
> implementation choice rather than a property of the model or the hardware.
>
> That dependency is the right way round. **Chapter 7 has to be settled before this
> chapter is worth running**, and the four numbers it tells you to log are this chapter's
> instrumentation.

**Depends on:** [Chapter 5]({{ '/pages/5-transformers' | relative_url }}) for MoE and latent
attention accounting, [Chapter 7]({{ '/pages/7-moe' | relative_url }}) for everything about
routing and expert parallelism, [Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }})
for triage, and [Chapter 9]({{ '/pages/9-llama' | relative_url }}) for the method this chapter
repeats.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

<!-- WHY THIS WHOLE CHAPTER IS BLOCKED, and the dependency chain behind it.

     BLOCKER 1: no training run. Same as Chapter 9. Needs DeepSeek-V2-Lite training under
     MaxText on 8x MI300X, captured. One node is enough for this model, which is the
     reason it was chosen as the sparse capstone over anything larger.

     BLOCKER 2 (upstream, and the more interesting one): Chapter 7's
     "Which of the Three You Can Get on AMD in JAX" is unresolved. We know from source
     that MaxText's default grouped-GEMM backend is a TPU Pallas kernel that runs in
     interpret mode on a GPU mesh, and that jax.lax.ragged_dot is the only
     platform-general path. What we do not know is how fast that path is on ROCm. Until
     that is measured:
       - Any MFU figure here is a measurement of a kernel choice, not of the model.
       - The configuration section cannot justify sparse_matmul / megablox settings.
       - The expert-parallelism section can still be written, since dispatch cost is
         independent of the GEMM backend, but it would be the only measured section.
     So: run Chapter 7's implementation comparison first. It needs the same node and the
     same model.

     BLOCKER 3 (verification, cheap): confirm DeepSeek-V2-Lite trains at all on the
     pinned stack. The roadmap says AMD's ROCm MaxText fork lists it as pre-optimised;
     we have confirmed only that upstream MaxText has deepseek2-16b.yml with
     num_experts 64, num_experts_per_tok 6, base_moe_mlp_dim 1408. Confirming it trains
     is a day of work and gates the whole wave.

     SECTION BRIEFS, from the roadmap:

     The Configuration, and Why Each Degree.
       Same treatment as Chapter 9, plus the two decisions dense models do not have:
       the expert-parallel degree, justified against Chapter 7's F_moe > 1589 threshold
       (DeepSeek-V2-Lite's F_moe is 1408, so it sits marginally *under* the threshold at
       8-way expert parallelism, which is a genuinely interesting prediction to test),
       and the dispatch implementation, justified against Chapter 7's win condition.

     Routing and Imbalance in a Real Run.
       The token-per-expert histogram over the course of training, the imbalance factor
       and its cost in step time, and whether load_balance_loss_weight was needed. This
       is where Chapter 7's claim that imbalance improves early and worsens with
       specialisation gets tested against a curve rather than asserted.

     Expert Parallelism and All-to-All Placement.
       Measured all-to-all share of step time at |Ex| = 2, 4, 8 on one baseboard, against
       Chapter 7's prediction of roughly 36% of expert compute at 8-way. This is the most
       valuable measurement in the chapter and the one least dependent on Blocker 2.

     What Latent Attention Does to the Memory Profile.
       MLA is accounted for in Chapter 5; this is where its consequences show up in an
       XProf memory profile during *training*, which is a different question from the
       serving economics in Chapter 11. Reference Chapter 5's arithmetic, do not
       re-derive it.

     MFU Against the Roofline.
       Predicted from Chapters 5 and 7, measured, gap explained. Blocked on Blocker 2 as
       described above. When written, it must state which dispatch implementation the
       figure is for, because the answer varies by up to E/E_a between them.

     Fallbacks If This Model Disappoints.
       Qwen3 30B-A3B is the alternative and Mixtral 8x7B the simpler fallback below that.
       All three are in the same MaxText model-config directory, so switching costs a
       re-run rather than a rewrite. Note that Qwen3 is the *harder* model by Chapter 7's
       arithmetic (F_moe of 768, 2x under the dispatch roofline), so it is a fallback in
       the sense of availability, not of difficulty.

     Worked Problems.
       From the roadmap: from this chapter's trace, estimate routing imbalance and its
       cost in step time; determine whether the run is dropping tokens; decide whether the
       chosen expert-parallel degree helped or hurt. Chapter 7's worked problems are the
       analytical versions of all three and should be referenced rather than repeated.

     ONE SCOPE RULE TO HOLD WHEN WRITING: make no serving claims here at all. This is a
     training chapter. Chapters 11 and 12 own that ground, and the temptation to say
     something about serving a model with MLA and 64 experts will be strong. Close the
     chapter by pointing forward at Part IV instead. -->

## The Configuration, and Why Each Degree

## Routing and Imbalance in a Real Run

## Expert Parallelism and All-to-All Placement

## What Latent Attention Does to the Memory Profile

## MFU Against the Roofline

## Fallbacks If This Model Disappoints

## Worked Problems

## References

**Provisional.** The sources the chapter will be written against.

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (DeepSeek, 2024). The model family,
  including DeepSeek-V2-Lite's configuration and the derivation of multi-head latent
  attention.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) (Dai et al., 2024). Fine-grained and
  shared experts, both of which this model uses.
- [MaxText DeepSeek configurations](https://github.com/AI-Hypercomputer/maxtext/tree/main/src/maxtext/configs/models)
  (AI-Hypercomputer). `deepseek2-16b.yml`, the configuration this chapter starts from.
- [MaxText MoE configuration reference](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/reference/core_concepts/moe_configuration.md)
  (AI-Hypercomputer). The dispatch-implementation decision tree that the configuration
  section has to justify.
