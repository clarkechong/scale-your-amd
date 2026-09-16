1. why we want to train in mixed precision
    - based on the premise that we 1) utilize lower-precision FLOPs 2) lower training memory footprint, whilst maintaining higher precision quality
2. well what does it mean to train in mixed precision? we cant simply move the whole model from fp32 to fp8, that would degrade the model
    - we have to apply quantization to selective tensors hence the term mixed precision
    - you can already see the concept in an MX mfma instruction itself: compute in mxfp, accumulate in bf16
3. lets take a prominent example of this: deepseekv3
    - show their mixed precision diagram, including how their logical separation of tensors relates to their quantizations
5. a full persepctive of quantization formats
    - bf16, fp16
    - fp8 training (deepseek style, requires delayed scaling, other such things to make viable)
    - MXFP formats, including diagrams of how they work, and external research papers demosntrating their viability
4. how would we implement this in jax?
    - pure jax 
    - maxtext allows you to control flags: these are predetermined quantization recipes
        - cover these configs

6. CASE STUDY of llama70b
    - expectations:
        - published mi355 format FLOPs
        - amdahl speedup bound: we know from 3-profiling the attribution of a train step. lets try and predict the speedup accordingly
        - expected reductions in memory footprint
    - a discussion of mixed precision recipes. which components should use which precision formats.
    - train step results for llama70b:
        - do the results align with our predictions/attribution estimates
        - do they aligh estimates seen in third party research papers
    - convergence results (local results)
    - convergence results of third party papers.

---

## Why train in lower precision
- based on the premise that we 1) utilize lower-precision FLOPs 2) lower training memory footprint, whilst maintaining higher precision quality

## What does 'mixed' precision mean
- we have to apply quantization to selective tensors hence the term mixed precision
- you can already see the concept in an MX mfma instruction itself: compute in mxfp, accumulate in bf16
- look at deepseek v3 technical report, where they discuss their quantization infrastructure
- present their diagram of how their logical tensors use different precisions for different reasons (deepseek-precision.png) but they say only linear operator. meaning?

## Precision and quantization formats
### bf16, fp16
- standard explanations, reference material for deeper reading
### fp8
- requires delayed scaling and other features. reference third party research
- https://arxiv.org/pdf/2209.05433 useful or not? nvidia fp8 formats for deep learning
### MXFP microscaling formats
- including diagrams of how they work, and external research papers demosntrating their viability
- reference material for further reading
### stability and divergence
- summarise content about concerns about divergenec when training in mixed precision

## Implementing in JAX
### JAX datatypes
- an explanation of how dtypes are handled in jax: jax.numpy needs to be able to process these
- (what if it doesnt? how do i specify mxfp6 on amd?)
### MaxText
- maxtext provides quantization flags that implement standardized categories of tensors for maxtext-supported models
- maxtext is external and rocm cant rely on this to be up to date. may need manual patching if insisting on maxtext, otherwsie for custom quantization infra you should work at jax level
- diagram of what actually needs to happen? if you select quantization strategy, it will gate at jax level (e.g. if some_flag=fp8). then as explained in chapter2, it has to edit at jax level by...
    - by what? ffi? jax dtype? insert relevent code blocks on how this works in practice, tracing the stack
### Transform Engine and JAX-AITER
- what ffi calls does transform engine provide? which dtypes, how are they formatted, specifics we need to know

## Case study: Llama 70B
### Expectations
- published mi355 format FLOPs
- amdahl speedup bound: we know from 3-profiling the attribution of a train step. lets try and roughly estimate our speedup
- expected reductions in memory footprint
### Mixed precision recipes
- sets of diagrams to explain this
### Train step results
- present results: step time, tokens/s, mfu, speedup compared to bf16
### Convergence results
- present our own convergence results on 1B tokens
- refer to third party research for established mixed precision stability etc
