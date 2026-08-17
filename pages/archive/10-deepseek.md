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

> **Draft.** DeepSeek-V2-Lite ran on 8x MI300X and the chapter is written from it, with
> [Chapter 7]({{ '/pages/7-moe' | relative_url }})'s kernel question settled first, which
> is what makes the MFU figure here mean something. One section is still owed: the
> token-per-expert histogram over training, which needs MaxText instrumented to emit the
> per-expert counts it already computes.

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

**AMD's Primus recipe for DeepSeek-V2-Lite, reduced to what matters:**

```
ici_expert_parallelism: -1     # all eight devices on the expert axis
sparse_matmul: false
megablox: false
capacity_factor: 1.25
max_target_length: 4096
per_device_batch_size: 8
sharding_tolerance: 0.05
```

**The two decisions a dense model does not have are the dispatch implementation and the
expert-parallel degree, and [Chapter 7]({{ '/pages/7-moe' | relative_url }}) measured both
before we got here.**

**`sparse_matmul: false` with a finite capacity factor is the only fast path.** Of the five
implementations Chapter 7 tried, `megablox: true` fails to compile on GPU, `tokamax`
refuses to run on MI300X, and `jax.lax.ragged_dot` runs between 5.6 and 11.8 times slower
than one-hot capacity dispatch. **The recipe is not a conservative choice; it is the
surviving one.**

**`ici_expert_parallelism: -1`, meaning all eight devices, is right and the threshold does
not say so.** Chapter 7's roofline gives `F_moe > 1589` for 8-way expert parallelism to be
compute-bound, and DeepSeek-V2-Lite's `F_moe` is **1408**, marginally under. That is the
prediction this chapter exists to test, and the next section tests it.

## Routing and Imbalance in a Real Run

**One thing we can report and one we cannot.**

**`load_balance_loss_weight` is 0.0 in AMD's recipe and the model trains.** No auxiliary
balancing loss, no collapse into a dead run, and throughput steady across the measured
steps. **[measured]** That is worth knowing because the auxiliary loss is usually presented
as mandatory, and at least for a short throughput run at `capacity_factor: 1.25` it is not.

**What we cannot report is the histogram**, which is the interesting half.

<!-- BLOCKED: the token-per-expert histogram over training, which is the section's
     actual subject. MaxText does not log per-expert token counts by default, and with
     load_balance_loss_weight: 0.0 there is no auxiliary-loss series to read imbalance
     out of either.

     What it needs: instrument layers/moe.py to emit the group_sizes tensor (the
     per-expert token count the dispatch already computes) to the metrics writer each
     step, then plot it over a run long enough for specialisation to appear. That is a
     patch to MaxText rather than a configuration, which is why it is not here.

     Also owed: the same run with load_balance_loss_weight nonzero, to measure what the
     auxiliary loss costs in step time and what it buys in imbalance. Chapter 7's claim
     that imbalance improves early and worsens with specialisation needs a curve over
     thousands of steps, not the tens we ran. -->

## Expert Parallelism and All-to-All Placement

**Eight MI300X, expert parallelism traded against FSDP so the device count stays at
eight.** **[measured]**

| `\|Ex\|` | FSDP | Step time | TFLOP/s per device | Collectives, share of device time | Of which hidden |
|---|---|---|---|---|---|
| 2 | 4 | 4.23 s | 127.2 | 26.4% | 74.7% |
| 4 | 2 | 3.83 s | 140.4 | 21.2% | 75.6% |
| 8 | 1 | **3.54 s** | **151.8** | **11.6%** | **77.8%** |

**More expert parallelism is better, and by the time all eight devices are on the expert
axis the all-to-all is 11.6% of device time with three quarters of it hidden behind
compute.** In wall-clock terms the dispatch is costing this model a few percent of its
step.

**Which means the `F_moe > 1589` threshold called this one wrong, and the reason is
instructive.** DeepSeek-V2-Lite sits at `F_moe = 1408`, below the line, and should be
dispatch-bound at 8-way. It is comfortably compute-bound instead. **The threshold models a
layer as pure routed MoE, and this model is not one:**

- **Two shared experts** run for every token, dense, with no dispatch at all.
- **The first layer is dense**, `first_num_dense_layers: 1`, at `base_mlp_dim: 10944`.
- **Multi-head latent attention** adds the two low-rank projections that
  [Chapter 5]({{ '/pages/5-transformers' | relative_url }}) accounts for, again with no
  dispatch.

**All of that is compute the all-to-all can hide behind, and none of it is in `F_moe`.**
Compare against [Chapter 7]({{ '/pages/7-moe' | relative_url }})'s Qwen3 measurement, which
is a purer fine-grained MoE at `F_moe = 768` and spends **45.8%** of its device time in
collectives with only 10.2% hidden. Same threshold verdict, opposite outcome.

**So the rule to carry away is narrower than the threshold suggests.** `F_moe > 1589`
predicts whether *the routed part of the layer* pays for its own dispatch. **A model with
substantial dense compute alongside its experts can sit below the threshold and be
perfectly healthy**, and DeepSeek's architects appear to have known that: the shared
experts exist for quality reasons, and they happen to make the dispatch affordable.

## What Latent Attention Does to the Memory Profile

**MLA's serving economics are [Chapter 11]({{ '/pages/11-inference' | relative_url }})'s
subject and its arithmetic is [Chapter 5]({{ '/pages/5-transformers' | relative_url }})'s.
What shows up in training is less dramatic than either.** **[measured]**

| | Llama 3 8B, 8k context | DeepSeek-V2-Lite, 4k context |
|---|---|---|
| Total compiled memory | 126.5 GB | 160.2 GB |
| Arguments | 11.2 GB | 22.0 GB |
| Temporaries | 115.3 GB | 138.3 GB |

**Temporaries dominate both, and that is the point.** MLA compresses the KV
representation, which is decisive when the KV cache is the resident state you are paying
for across a long decode. **In training there is no KV cache**: keys and values are
recomputed every step, live inside the attention kernel, and never persist. The saving MLA
exists to deliver is a serving saving.

**What DeepSeek-V2-Lite pays instead is a larger argument footprint**, 22.0 GB against
Llama's 11.2 GB, which is the 64 experts: a sparse model holds far more parameters for the
same active compute, and every one of them is resident. **The memory story of a training
MoE is expert weights, not attention.**

**The expert-parallel degree does not change it.** At `|Ex| = 2` the compiled footprint is
160.2 GB and at `|Ex| = 8` it is 160.2 GB, identical to a tenth of a gigabyte, because
FSDP and expert parallelism shard the same parameters along different axes and eight ways
is eight ways either way.

## MFU Against the Roofline

**DeepSeek-V2-Lite reaches 151.8 TFLOP/s per device**, which against the two rooflines is
**11.6% of the data sheet's boost-clock peak and 15.3% of the peak at the clock MI300X
actually sustains**. **[measured]**

**State the dispatch implementation with the number, because it is a property of the
kernel choice as much as the model.** This is one-hot capacity dispatch at
`capacity_factor: 1.25`, which per
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) is the fastest of the five available and
between 5.6 and 11.8 times faster than the alternative that runs.

**15% MFU next to Llama 3 8B's 44% is not a defect, it is what sparsity costs on this
hardware.** Three terms, in order:

**The experts are narrow.** `F_moe = 1408` against Llama's `F = 14336`. Every expert GEMM
is a tenth the width, and [Chapter 3]({{ '/pages/3-profiling' | relative_url }})'s size
sweep shows what narrow GEMMs do: a 1024-cubed matmul reaches 11% of peak where a
4096-cubed one reaches 70%, because the tile no longer fills the machine. **Fine-grained
MoE is a machine for generating small matmuls.**

**Capacity padding wastes 25% by construction.** `capacity_factor: 1.25` allocates a
quarter more slots than tokens so that routing imbalance does not drop anything, and the
padding is computed on.

**Dispatch is the smallest term.** 11.6% of device time, three quarters hidden. **The
communication this chapter was most worried about turns out to be the least of it**, and
the missing throughput is in the shape of the GEMMs rather than in the wires between them.

**Which is the chapter's real result.** [Chapter 7]({{ '/pages/7-moe' | relative_url }})
argues that MoE on AMD in JAX is limited by the absence of a good grouped GEMM rather than
by dispatch cost, and a full model at 15% MFU with a healthy communication profile is that
argument cashed out. **A grouped-GEMM kernel that reached even half of hipBLASLt's dense
efficiency would move this number more than any amount of parallelism tuning.**

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
