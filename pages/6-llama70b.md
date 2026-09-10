---
layout: distill
title: "Training Llama 70B"
description: "Mixed-precision training of Llama 70B in MaxText on eight GPUs: throughput, memory, and convergence."
date: 2026-09-10

section_number: 6

previous_section_url: "/pages/5-llama7b"
previous_section_name: "Chapter 5: Llama 7B"

next_section_url: "/pages/7-mixtral8-22b"
next_section_name: "Chapter 7: Mixtral 8x22B"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

---
What are the tradeoffs when training in mixed precision?

Ideally we would like to train our models with the highest precision so that our weights can fine-tune as much as possible and we can reach a theoretically higher model performance. However, with large DNN models such as LLMs, we do not have the compute power to do everything in fp64 in a reasonable amount of time. If you do a rough estimate, <...> X hours!

Training at a lower precision greatly increases the throughput of training. We know that lower-precision dtypes have higher peak FLOPs performance on the GPU. However, doing so often comes at a loss of model quality.

So with mixed precision training, we are trying to train with the increased throughput of lower precision, whilst maintaining the higher model quality of higher precisions.

---

Maxtext supports this through config parameters. You can control:

```
weight_dtype: "float32"
grad_dtype: "float32"
mu_dtype: "float32"
dtype: "bfloat16"
quantization: ""
```

<explain each of these dtype params>

<diagram of slice of llm layer, and which blocks use which precision>
<and hence derive amdahls peak speedup. ie not always 2x, its more like 1.5x ceiling when you consider attribution ie not everything is a gemm>

maxtext supports standard training types: bf16, fp16, fp8, mxfp8. also nvfp4

<diagram of maxtext when going through ROCm backend>
<e.g. at what point maxtext branches on vendor or jax call>
<what that jax call looks like when lowered in xla (or if these are already custom calls like TE)>

mxfp4 is in progress.
<explain jax aiter>

so how does mixed precision affect throughput and execution performance on rocm? lets look at a train step profile

<train step profiling metrics: mfu%, throughput, proportions of gemm to other, etc>
<consult plan.md for llama70b>

---

performance improvements are great but what about model quality?

we can measure this with some following metrics
<consult plan.md phase b>

in order to train to convergence we will want to change a few things in our setup
<maxtext config changes, other jax or script changes>

estimate X epochs to reach X quality, takes around X time based on our train step times

<present phase B results>

discuss and analyze.

---

1. how to train mixed precision with maxtext
    - the relevent maxtext flags
    - fp32 (note slow due to no te kernels)
    - bf16 works out of the box
    - fp16 works out of the box
    - fp8 works out of the box
    - mxfp8 seems to work out of the box however custom TE patch required for desired performance in this case
        - te2.17 patch branch, explained
    - mxfp4 is currently unsupported. walk through the (current) steps to get this working.
        - explain why: not yet integrated into upstream maxtext so in the meantime we have to manually patch maxtext to issue mxfp4 instructions. how? we do this via mxfp4 ffi (custom calls)
        - rocm maxtext mxfp4 branch has this patch for us
        - jax aiter alpha 2, we need to build the relevent ffi calls
        - Explicit FP4 environment recipe. AITER_FP4_ATTN=1 for equal scope with FP8. JA_FP4_PACK_GATEUP_AG=0 because the enabled path currently double-shuffles weights. TE/BF16 attention core retained.
        - BASICALLY, JUST CONSULT THE PROVIDED PYTHON RUNNER
2. present the train step results, including throughput, memory footprint, and roofline analysis
    - based on calculation estimations for step time, throughput, memory footprint, do the achieved results align
3. present convergence results
    - how to precision stability vary, do the loss curves overlap, time to quality, etc
4. does external research support these findings