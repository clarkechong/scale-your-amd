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

1. 