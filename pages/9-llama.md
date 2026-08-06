---
layout: distill
title: "Training Llama 3 on MI300X"
description: "The dense capstone. A parallelism strategy justified against the inequalities rather than copied, MaxText on ROCm, capturing a profile from a run that lasts days, a per-layer breakdown, and the two operational things that decide whether a real training run finishes."
date: 2026-08-04

section_number: 9

previous_section_url: "/pages/8-getting-to-roofline"
previous_section_name: "Chapter 8: Getting to Roofline"

next_section_url: "/pages/10-deepseek"
next_section_name: "Chapter 10: DeepSeek-V2-Lite"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: The Configuration, and Why Each Degree
  - name: Running MaxText on ROCm
  - name: Capturing a Profile at Production Scale
  - name: A Per-Layer Breakdown
  - name: MFU Against the Roofline
  - name: Checkpointing
  - name: The Input Pipeline at Scale
  - name: The Failure That Is Not a Performance Problem
  - name: Worked Problems
  - name: References
---

> **Draft.** The run happened and the chapter is written from it: Llama 3 8B on 8x MI300X,
> a parallelism sweep, a per-layer breakdown and a checkpoint cost, all measured on
> `rocm/jax-training:maxtext-v26.5`. Two sections are still owed. The input-pipeline
> number needs a real dataset attached, which is now a small job rather than a dependency.
> The MI355X RCCL footnote needs hardware we do not have and a citation we have not
> found.

**Depends on:** Chapters [1]({{ '/pages/1-rooflines' | relative_url }}) through
[6]({{ '/pages/6-training' | relative_url }}) for the predictions this chapter checks, and
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) for the triage that explains
the gap. Nothing from [Chapter 7]({{ '/pages/7-moe' | relative_url }}): this model is dense.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

<!-- WHY THIS WHOLE CHAPTER IS BLOCKED, and what unblocks it.

     Hard dependency: a Llama 3 training run on MI300X under MaxText, captured. Nothing
     in the chapter is writable without it. Specifically needed:
       - One 8x MI300X node for Llama 3 8B. This is the minimum viable version of the
         chapter and it is not gated on a cluster.
       - Multi-node for 70B and for anything Llama 3.1 405B related. We have no cluster,
         so the multi-node half of this chapter is blocked twice over.
       - Capture wrappers around the MaxText configs (listed as a needed script in
         docs/structure.md).

     Verification owed before writing, because the roadmap makes claims we have not
     checked in this pass:
       - RESOLVED, partly. AMD's fork is ROCm/maxtext and the container this book pins,
         rocm/jax-training:maxtext-v26.5, ships it at branch release/v26.5, commit
         a7c6c7e5, installed editable at /workspace/maxtext. Two corrections to the
         roadmap: there is no run_rocm.py in that tree, and the layout is src/maxtext/
         rather than MaxText/. There is also no rocm/ or MI300-specific config directory;
         configs/gpu/ is NVIDIA A3-oriented. The AMD-tuned MI300X recipes live in Primus
         instead, at /workspace/Primus/examples/maxtext/configs/MI300X/, and those are
         what Chapters 9 and 10 should start from.
       - Whether those recipes run as documented on the stack this book now pins
         (rocm/jax-training:maxtext-v26.5, ROCm 7.14.0, jax 0.10.0). Note Primus's own
         README targets an older image, maxtext-v26.4-jax0.9.1-te2.12.0, so the recipes
         were tuned against a slightly different stack than the one we measure on.

     SECTION BRIEFS, from the roadmap:

     The Configuration, and Why Each Degree.
       Start from the published configuration rather than inventing one, then spend the
       section explaining *why* each parallelism degree was chosen, against Chapter 6's
       inequalities. That explanation is the part AMD's documentation does not do and the
       part the reader needs. Every degree should trace to a specific inequality: the
       FSDP degree to the memory ledger, the tensor-parallel degree to the F threshold,
       the per-device batch to the ridge point.

     Running MaxText on ROCm.
       The launch path, container tag, and the config file, kept short. This is the
       section most likely to rot, so it should point at Appendix A for anything that is
       really installation.

     Capturing a Profile at Production Scale.
       A few steps out of thousands: which steps (not the first, and after any autotuning
       has settled), trace file sizes, multi-host capture and where the per-host traces
       land, and how to avoid capturing 40 GB of trace by accident. Chapter 3 taught the
       mechanics on a toy; this is the same thing when the run costs money.

     A Per-Layer Breakdown.
       Where the step time goes, layer by layer and op by op, against Chapter 5's
       accounting. This is where the "stack of MLPs" approximation gets checked against
       a real model: Chapter 5 predicts attention is roughly 40% of layer FLOPs at 8k
       context, and this is the section that finds out.

     MFU Against the Roofline.
       The book's central promise cashed out for a dense model: predicted MFU from
       Chapter 6, measured MFU, and the gap explained using Chapter 8's triage order. Say
       plainly whether the model held. If it did not, the explanation is the most
       valuable paragraph in the chapter.

     Checkpointing.
       What a checkpoint costs in time and bytes at this scale, how often to take one,
       and how long a restart takes. At 70B this is a first-order throughput term and no
       other chapter owns it. Orbax is the mechanism; the numbers are what matter.

     The Input Pipeline at Scale.
       How a sharded dataset gets fed to a multi-process mesh without every host reading
       the same shard, and what a deterministic resume costs. This is checkpointing's
       sibling and the other thing that stalls real runs. Chapter 8 taught the reader to
       *recognise* host starvation in a trace; this is where they see it prevented at
       production scale rather than diagnosed after the fact.

     The Failure That Is Not a Performance Problem.
       AMD's own benchmark scripts disable an RCCL feature to avoid NaN losses on MI355X.
       That is a real, slightly uncomfortable example of the kind of thing no roofline
       predicts, and one honestly told footnote is worth more than a page of generalities
       about robustness. VERIFY THE SPECIFIC FLAG AND THE MODELS AFFECTED before writing;
       we have not checked this claim in this pass, and getting it wrong would be worse
       than omitting it.

     Worked Problems.
       From the roadmap's general pattern: given this chapter's trace, re-derive the
       parallelism strategy from replica groups alone (tests Chapter 4); compute what
       fraction of step time should be the gradient all-reduce and compare (tests
       Chapter 6); estimate the checkpoint overhead as a fraction of throughput at a
       given interval (tests this chapter). -->

## The Configuration, and Why Each Degree

**Start from AMD's published recipe rather than inventing one.** The Primus repository
ships MI300X pretraining configs, and the Llama 3 8B one reduces to this:

```
ici_fsdp_parallelism: 8        # everything else at 1
max_target_length: 8192
per_device_batch_size: 4
remat_policy: minimal_flash
dataset_type: synthetic
```

**Every one of those degrees traces to an inequality in
[Chapter 6]({{ '/pages/6-training' | relative_url }}), and we checked the one that matters
most by measuring the alternatives.** **[measured]**

| Parallelism | Step time | TFLOP/s per device | MFU vs the clock it holds | Collectives, share of device time |
|---|---|---|---|---|
| **FSDP 8** | **3.90 s** | **432.8** | **43.7%** | **6.9%** |
| FSDP 4 x TP 2 | 6.65 s | 253.8 | 25.6% | 43.6% |
| FSDP 2 x TP 4 | 6.85 s | 246.3 | 24.9% | 44.7% |
| TP 8 | 6.81 s | 247.6 | 25.0% | 37.7% |

**Pure FSDP is 1.7x faster than any configuration containing tensor parallelism, and the
collective share says why.** FSDP spends 6.9% of its device time in collectives; every
tensor-parallel variant spends between 38% and 45%.

**This is [Chapter 6]({{ '/pages/6-training' | relative_url }})'s tensor-parallelism
correction arriving in a real model.** The threshold `F > 1816 * (|Y|-1)` says Llama 3
8B's `F = 14336` clears 8-way tensor parallelism with 13% to spare. It does not, because
the threshold assumes 320 GB/s and tensor parallelism's activation-sized messages get
about 212 GB/s. **Substituting the bandwidth those messages actually reach moves the
requirement to 19191 and Llama 3 8B fails it**, which is exactly what the table shows.

**The other degrees, briefly.** `per_device_batch_size: 4` at 8192 tokens is 32768 tokens
per device, comfortably above [Chapter 6]({{ '/pages/6-training' | relative_url }})'s FSDP
threshold of 4086 and far enough above it that the exposed communication is under 5% of
the step. `remat_policy: minimal_flash` is the memory ledger: full activations do not fit
at this batch, and recomputing the cheap parts buys the batch that makes the collectives
hide. **The recipe is internally consistent, and the way to see that is that changing any
one degree makes it worse.**

## Running MaxText on ROCm

**Two settings are not optional and neither is in the Primus recipe**, because they are
properties of MaxText's defaults rather than of the model:

```bash
python -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    model_name=llama3-8b \
    hardware=gpu \
    attention=cudnn_flash_te max_segments_per_seq=1 \
    dataset_type=synthetic enable_checkpointing=false \
    ici_fsdp_parallelism=8 max_target_length=8192 \
    per_device_batch_size=4 remat_policy=minimal_flash \
    run_name=llama3_8b base_output_directory=./out
```

**`hardware=gpu`, because `base.yml` defaults to `tpu`.** You will not get a helpful error.

**`attention=cudnn_flash_te`, because the other two do not run.**
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }})'s attention section has
the details: `dot_product` asks for a 182 GiB allocation and `flash` asks for 96 KB of LDS
against MI300X's 64 KB. Despite the name, `cudnn_flash_te` routes to AMD's Transformer
Engine port on this platform. **It also needs `max_segments_per_seq`** or it refuses to
configure with a pydantic error that does not mention attention.

**Everything else about getting here is in
[Appendix A]({{ '/pages/a-appendix-install' | relative_url }})**, including the container
tag, which is the only version statement that matters.

## Capturing a Profile at Production Scale

**The mechanics are [Chapter 3]({{ '/pages/3-profiling' | relative_url }})'s. What changes
at this scale is that a careless capture is expensive**, and the two settings that control
that are:

```
profiler=xplane
skip_first_n_steps_for_profiler=10
profiler_steps=5
```

**Skip at least ten steps.** The first is compilation, the next several are the input
pipeline reaching steady state and the clock settling, and
[Appendix B]({{ '/pages/b-appendix-protocol' | relative_url }}) asks for both to be gone
before anything is quoted.

**Five steps is enough and more is worse.** Five steps of Llama 3 8B on eight devices
produces a **28 MB `xplane.pb` and a 12 MB `trace.json.gz`**, about 40 MB of run
directory. **[measured]** That is comfortable. The same capture left running for a
thousand steps is not, and the failure is not subtle: XProf loads the whole file.

**Trace what you will read.** The capture above holds roughly 20 seconds of an eight-GPU
run and every number in the rest of this chapter came out of it.

## A Per-Layer Breakdown

**This is where [Chapter 5]({{ '/pages/5-transformers' | relative_url }})'s accounting gets
checked against a real model.** Grouping device kernel time by `jax.named_scope`, from the
capture above: **[measured]**

| Scope | Share of device kernel time |
|---|---|
| `decoder/body/closed_call/layers` | 52.5% |
| `decoder/body/closed_call` | 16.9% |
| `layers/DotProductAttention_0/_FusedDotProductAttention_0` | **15.4%** |
| Kernels with no op name | 5.1% |
| `decoder.apply_output_head/logits_dense` | 4.1% |
| everything else | 6.0% |

**Attention is 15.4% of the step and 22.7% of the time spent inside the decoder layers.**

**Chapter 5 predicts 38% of MLP FLOPs at 8k context for this model**, which, once the
attention projections and the MLP are added to the denominator, is **23.5% of layer
FLOPs**. Measured: 22.7% of layer *time*. **The accounting holds.**

**One caveat on reading that as agreement.** A share of time equals a share of FLOPs only
if the two run at the same efficiency, and the fused attention kernel exploits causal
masking to do about half the score FLOPs the count assumes. **Matching time shares
therefore means attention is running at roughly half the MLP's efficiency**, which is
unsurprising for a kernel that is part softmax, and is the more useful reading of the
number.

**The "stack of MLPs" model is wrong by about a third and right about the scaling**, which
is what Chapter 5 said it would be. For parallelism decisions that is enough. For an MFU
figure it is not, which is why the next section counts the score FLOPs explicitly.

## MFU Against the Roofline

**Llama 3 8B on eight MI300X reaches 432.8 TFLOP/s per device.** **[measured]** Against
the two rooflines this book carries:

```
432.8 / 1307.4 = 33.1%      # the data sheet's boost-clock peak
432.8 /  990   = 43.7%      # the peak at the clock MI300X actually sustains
```

**Quote the second and say which.** [Chapter 3]({{ '/pages/3-profiling' | relative_url }})
measures MI300X settling at about 1590 MHz under sustained dense matmul, power-limited at
750 W rather than thermally throttled, and 2100 MHz is a boost clock the part does not
hold. **A 33% MFU and a 44% MFU are the same measurement against different denominators**,
and the first one makes the machine look worse than it is for reasons that have nothing to
do with the code.

**So: 44% against the achievable roofline. Where is the other 56%?** Working
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }})'s triage order against
the breakdown above:

**Communication is not it.** 6.9% of device kernel time, and FSDP at 32768 tokens per
device is far past the point where it stops being the constraint.

**Attention is a large part of it.** 15.4% of the step running at roughly half the MLP's
efficiency is worth about 8 points of MFU on its own, and it is not obviously recoverable:
the fused kernel is the best of the three available and the other two do not run.

**The 5.1% with no op name is the next thing to look at**, and it is the frustrating one.
Those are kernels XProf shows with an empty Op Name, and per
[Chapter 3]({{ '/pages/3-profiling' | relative_url }}) the escalation is `rocprofv3`.

**And the largest single term is the one Chapter 3 already found in the plain matmul.** A
4096-cubed bf16 GEMM in isolation reaches 93% of the achievable roofline and no more,
because the tile hipBLASLt picks fills a compute unit with one workgroup and there is no
second wave to hide memory latency behind. **A model built out of those GEMMs cannot beat
them.** 44% of the achievable peak on a real training step, against 93% for the bare GEMM
that step is mostly made of, is a gap made of attention, remat recompute, the optimizer
and the unnamed 5%, and none of those is a bug.

**Did the model hold? Yes, once the clock is right.** The prediction machinery in Chapters
1, 5 and 6 gets the parallelism choice right, the layer breakdown right to a point, and
the communication behaviour right. **The one constant that needed correcting was the peak
FLOP rate**, and it needed correcting by 24%.

## Checkpointing

**A checkpoint of Llama 3 8B is 89.7 GiB and takes 70.3 seconds to write.** **[measured]**
Against a 3.9 second step, that is **eighteen steps of throughput for one checkpoint**.

```
/jax/orbax/write/blocking_gbytes_per_sec: 1.277 GiB/s
    (total gbytes: 89.7 GiB) (time elapsed: 70.26 s)
```

**The size is the optimizer, not the model.** 8.03B parameters at fp32, plus Adam's two
fp32 moments, is three copies of a 32 GB model. **The weights are a third of what you are
writing**, which is why "how big is your model" is the wrong question when sizing
checkpoint storage or planning a save interval.

**1.277 GiB/s is the number to be suspicious of.** That is local disk, not the fabric, and
it is roughly what a single write stream to this filesystem sustains. **Checkpoint time
scales with your storage, not with your GPUs**, and it is the one term in this chapter that
gets worse as you add nodes rather than better.

**It is also the least repeatable number in this chapter.** A second run writing the same
89.7 GiB took **124.6 seconds at 738 MiB/s**, 1.8x slower than the first, with nothing
changed but the state of the filesystem underneath. **Measure your own, more than once**,
and do not plan a save interval against a single observation.

**Two levers, and take both.** `async_checkpointing: true` overlaps the write with
subsequent steps, which is the default and which we disabled here to get a clean blocking
number. And the save interval is a straight throughput trade: at one checkpoint every 100
steps, blocking, this costs 18% of throughput; asynchronous and every 1000 steps, it is
noise.

**Restarting reads the same 89.7 GiB back, and reading is the faster direction: 32.7
seconds at 2.744 GiB/s, 2.1x quicker than the write.** **[measured]** That is still half a
minute before the first step of a restarted run, on top of compilation, and unlike the
write it is on the critical path of a job that is already down.

## The Input Pipeline at Scale

**Everything above ran on synthetic data, and that is a real limitation of this chapter
rather than a simplification.** `dataset_type: synthetic` generates tokens on device: no
file reads, no host-to-device copies, no shard assignment, and therefore **no possibility
of observing the host starvation this section exists to teach**.

**What we can say from the numbers we have.** The capture shows no meaningful gap between
steps and no host-side stall, which is what you would expect when there is no input
pipeline at all. **Any real pipeline can only make this worse**, so the 43.7% MFU above is
an upper bound on what this configuration achieves with data attached.

**What it would take, since egress from this container works.** MaxText supports
`dataset_type=hf` with `hf_path=allenai/c4`, and a Llama 3 tokenizer ships in the image at
`src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken`, so the ingredients are present.
**The measurement owed is the same step time with a real pipeline attached and the gap
between them.**

<!-- BLOCKED: the input-pipeline number, and it is now a small job rather than a
     dependency. Everything needed is present: HF egress works from this container
     (verified 5 August 2026 against the datasets API), MaxText supports dataset_type=hf,
     and a Llama 3 tokenizer ships in the image. What is owed:
       - The same fsdp8 configuration with dataset_type=hf, hf_path=allenai/c4 and the
         bundled tokenizer, and the step-time delta against synthetic.
       - A trace showing whether the host keeps up, which is the thing the section is
         actually about. Look for gaps between StepTraceAnnotation boundaries.
       - Deterministic resume cost, which needs grain rather than hf.
     Not done here purely for time. -->

## The Failure That Is Not a Performance Problem

<!-- BLOCKED, and the roadmap is explicit that getting this wrong is worse than omitting
     it. The claim is that AMD's benchmark scripts disable an RCCL feature to avoid NaN
     losses on MI355X. We have no MI355X, so this can only ever be a cited claim here,
     and we have not located the flag in AMD's published scripts.

     Before writing: find the specific flag and the specific models affected in AMD's
     own repository, quote it with a URL and a date, and mark it clearly as someone
     else's observation rather than ours. If it cannot be found, drop the section: a
     vague warning about robustness is worth less than nothing. -->

## Worked Problems

## References

**Provisional, and to be replaced with what the chapter actually uses.** These are the
sources the chapter will be written against; none of them is cited in prose yet because
there is no prose.

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Meta, 2024). The model
  being trained, and the architecture numbers
  [Chapter 5]({{ '/pages/5-transformers' | relative_url }}) already uses.
- [MaxText](https://github.com/AI-Hypercomputer/maxtext) (AI-Hypercomputer). The training
  framework, its Llama 3 configurations, and its ROCm support.
- [Training a model with ROCm MaxText](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/)
  (AMD). AMD's own documentation for JAX and MaxText training on Instinct hardware. The
  exact page to cite depends on which repository this chapter ends up targeting; see the
  verification note in the source of this page.
- [Orbax](https://orbax.readthedocs.io/) (Google). The checkpointing library, for the
  checkpointing section.
