---
layout: distill
title: "Training in Mixed Precision"
description: "How storage, operand, accumulation, output, gradient, and optimizer dtypes combine into MI355X mixed-precision training."
date: 2026-09-13

section_number: 4

previous_section_url: "/pages/3-profiling-and-analysis-of-one-training-step"
previous_section_name: "Chapter 3: Profiling and Analysis of a Training Step"

next_section_url: "/pages/5-making-the-model-fit"
next_section_name: "Chapter 5: Making the Model Fit"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "What Mixed-Precision Training Actually Means"
    subsections:
      - name: "A Matrix Instruction Already Mixes Precisions"
      - name: "Casts, Conversions, and Emulation"
      - name: "Precision in HLO"
      - name: "Representative Training Recipes"
  - name: "Precision Across a Llama Training Step"
  - name: "Formats and Scaling"
    subsections:
      - name: "BF16 and FP16"
      - name: "FP8"
      - name: "MXFP8, MXFP6, and MXFP4"
  - name: "MI355X Training Paths"
    subsections:
      - name: "Published Matrix Peaks"
      - name: "Amdahl Speedup Bound"
      - name: "Compute, Memory, and Communication Effects"
  - name: "Published Convergence and Time-to-Quality Evidence"
  - name: "References"
---

Mixed-precision training is not a model-wide dtype switch. It is a contract assigning
formats to storage, matrix operands, accumulators, outputs, gradients, reductions,
optimizer state, and scale metadata. Saying that a model "trains in FP8" hides most of
that contract.

The distinction matters on MI355X because the Matrix Core instruction already has
separate operand and accumulator formats, and a JAX-facing library can add conversions
on both sides. A BF16 graph can therefore contain an MXFP4 matrix operation with an FP32
accumulator and a BF16 result, while its master parameters and optimizer remain FP32.

The software examples are pinned to `rocm/jax-training:maxtext-v26.6`,
JAX/JAXLIB and ROCm PJRT/plugin 0.11.0, Transformer Engine 2.17, the ROCm
[MaxText MXFP4 branch at `b437942a`](https://github.com/ROCm/maxtext/tree/b437942a),
and [JAX-AITER alpha2 at `35b7175c`](https://github.com/ROCm/jax-aiter/tree/35b7175c),
checked on **13 September 2026**. The chapter describes the available execution
shapes; the later case study measures them.

## What Mixed-Precision Training Actually Means

A useful recipe starts with a dtype ledger rather than one headline format:

| Role | Question | Representative choice |
|---|---|---|
| Master parameter | What does the optimizer update and retain? | FP32 |
| GEMM weight operand | What format enters the matrix instruction? | BF16, FP16, FP8, or MX |
| Activation or residual | What format flows between layers? | BF16 or FP16 |
| GEMM activation operand | What format enters the matrix instruction? | BF16, FP16, FP8, or MX |
| Accumulator | In what format are partial products summed? | commonly FP32 on the MI355X paths here |
| GEMM output | What format is written back to the graph? | commonly BF16 or FP16 |
| Gradient | What format is retained after backward? | BF16 or FP32 |
| Collective payload | What bytes cross RCCL? | BF16 or FP32 unless explicitly reduced |
| Optimizer state | What formats hold moments and update arithmetic? | commonly FP32 |
| Scale metadata | What describes the low-precision range? | FP32 scale, `amax` history, or E8M0 |

These roles can differ within one operation and again across operations. `dtype`,
`weight_dtype`, `grad_dtype`, `mu_dtype`, and `quantization` are therefore not
synonyms. They control different portions of the training state and graph.

### A Matrix Instruction Already Mixes Precisions

[Chapter 1]({{ '/pages/1-mi355x-as-a-training-machine' | relative_url }}#mfma-lane-fragments)
showed that an MFMA instruction names both its operand family and its accumulator.
For example, the CDNA 4 scaled forms are named like:

```text
v_mfma_scale_f32_16x16x128_f8f6f4
```

The `f8f6f4` portion describes the allowed low-precision operand encodings. The
leading `f32` describes the accumulator and result tile held in registers. All 64
lanes contribute low-precision fragments while the instruction updates FP32 partial
sums.

A library call can surround that instruction with a wider graph-level interface:

```text
FP32 master weight ─quantize─┐
                             ├─ low × low MFMA ─ FP32 accumulator ─cast─ BF16 output
BF16 activation   ─quantize─┘
```

The persistent parameter is still FP32, the activation arriving at the layer is still
BF16, and the result returned to the residual stream is still BF16. Only the operands
inside the eligible matrix operation are MXFP4 in this example. Writing simply
`weights: MXFP4` is ambiguous unless it states whether it means the master parameter,
a cached packed copy, or the transient GEMM operand.

Backward introduces the same distinctions again. The activation-gradient and
weight-gradient GEMMs can have their own operand recipes, accumulator format, and
output dtype. The final gradient tree, its RCCL reduction, and the optimizer update
are separate choices.

### Casts, Conversions, and Emulation

A cast is one mechanism used to cross a precision boundary. It is often part of a
mixed-precision recipe, but it is not itself the definition of mixed precision.

Three superficially similar graphs can mean different things:

1. A BF16 tensor is quantized to FP8, consumed by an FP8 MFMA, accumulated in FP32,
   and converted back to BF16. This is a genuine low-precision matrix route.
2. A low-precision stored tensor is converted to BF16 before an ordinary BF16 GEMM.
   Storage changed, but the matrix computation did not.
3. A requested numerical mode is implemented using another hardware mode. CDNA 4,
   for example, implements TF32 semantics in software through BF16. That is backend
   emulation, not by itself a model-level mixed-precision training recipe. FP32 matrix
   arithmetic itself remains a native MI355X path.

An input cast and output cast therefore do not prove which instruction executed. The
meaningful description is the complete route:

```text
stored dtype → operand conversion → matrix operand dtype
             → accumulator dtype → output conversion → graph dtype
```

### Precision in HLO

Optimized HLO is mainly a route-and-scope check for a precision recipe. It can show
which contractions stayed on a BF16 `dot` or GEMM custom call and which entered a
low-precision custom call. Around the latter, look for FP8 `convert` operations,
scale and `amax` calculations, or MX operands accompanied by E8M0 metadata.
Use [Chapter 2's compiler-delta convention]({{ '/pages/2-lowering-jax-jit-on-rocm' | relative_url }}#reading-a-compiler-delta)
and retain the matched arm layout from
[Appendix D]({{ '/pages/d-profiler-and-hlo-cookbook' | relative_url }}#feature-comparison-bundles).

There is deliberately no generic low-precision HLO drawing here. Transformer
Engine FP8, Transformer Engine MXFP8, and JAX-AITER MXFP4 expose different literal
targets, state tuples, scale layouts, and workspaces. Replacing those artifacts
with one invented custom call would hide the part this comparison is meant to
verify. Chapter 11 therefore owns the captured BF16/FP8/MXFP8/MXFP4 SVG excerpts
from its pinned stack.

Inspect the complete training computation, not only the first matching target. The
forward contraction, activation-gradient contraction, and weight-gradient contraction
can take different routes:

```text
forward:              x  @ w
activation gradient:  dy @ transpose(w)
weight gradient:      transpose(x) @ dy
```

For each direction, record operand and result layouts, copies or transposes at the
boundary, scale-metadata shapes, output dtype, and declared workspace. A matching
custom-call name with an unexpected local layout or insufficient workspace can still
select another internal implementation or fail eligibility.

HLO does not prove the matrix instruction inside an opaque library or FFI call. Actual
instruction proof requires a device trace to identify the launched kernel, disassembly
of its code object, and relevant instruction counters from a separate counter run.

### Representative Training Recipes

The following are representative BF16-surrounded recipes, not universal standards:

| Recipe | Master parameters | Surrounding tensors | Eligible GEMM operands | Accumulator | GEMM output | Optimizer |
|---|---|---|---|---|---|---|
| FP32 reference | FP32 | FP32 | FP32 | FP32 | FP32 | FP32 |
| BF16 mixed | FP32 | BF16 | BF16 | FP32 | BF16 | FP32 |
| FP16 mixed | FP32 | FP16 | FP16 | FP32 | FP16 | FP32, commonly with loss scaling |
| FP8 delayed | FP32 | BF16 | FP8, recipe-specific E4M3/E5M2 | FP32 | BF16 | FP32 |
| MXFP8 | FP32 | BF16 | MXFP8 plus block scales | FP32 | BF16 | FP32 |
| MXFP4 | FP32 | BF16 | MXFP4 plus block scales | FP32 | BF16 | FP32 |

Norms, softmax, loss, residual additions, collectives, and optimizer state can make
different choices from the dense projections. A recipe also decides which forward,
activation-gradient, and weight-gradient contractions receive the low-precision path.

## Precision Across a Llama Training Step

A standard Transformer block diagram is useful for locating projections, but it
cannot show the full training recipe by itself: accumulators live inside kernels,
while gradients, optimizer moments, master parameters, and scaling state live outside
the forward block. The figure separates those three views.

{% include figure.liquid path="pages/img/mixed-precision-llama-attribution.png" class="img-fluid" alt="A Llama training graph annotated with surrounding BF16 tensors, low-precision matrix operands, FP32 reductions and accumulators, and optimizer state" caption="An illustrative BF16-surrounded FP8 or MX recipe. Eligible projections quantize their operands, accumulate in FP32, and return BF16. Norms, attention softmax, residuals, loss, gradients, collectives, master parameters, and optimizer state retain independent dtype choices." %}

The important boundaries are:

- **Projections:** Q/K/V/O and the MLP contain the large GEMMs most likely to use
  FP8 or MX operands. The vocabulary projection may or may not be included.
- **Attention core:** QK, softmax, and probability-value multiplication are a separate
  backend path. Low-precision projections do not imply low-precision attention.
- **Norms and loss:** reductions commonly promote to FP32 even when their inputs and
  outputs are BF16.
- **Residual stream:** activations commonly remain BF16 between blocks rather than
  staying packed as FP8 or MX.
- **Backward and collectives:** gradient GEMMs may use the recipe, but retained
  gradients and RCCL payloads have their own dtype.
- **Optimizer state:** FP32 master parameters and Adam moments can dominate persistent
  memory even when the matrix operands are four bits.

The raw-JAX Llama 7B model makes one concrete BF16 recipe explicit: parameters and Adam
moments are FP32, dense inputs and outputs are BF16, RMSNorm variance is FP32, logits
are promoted to FP32 before cross-entropy, and the optimizer updates the FP32 parameter
tree. That is already mixed-precision training without an FP8 or MX quantization mode.

## Formats and Scaling

### BF16 and FP16

BF16 and FP16 both occupy two bytes. They have the same MI355X matrix peak, but they
distribute their bits differently.

BF16 keeps an eight-bit exponent and has much less significand precision than FP16. Its
range is close to FP32, which makes it the safer default for gradients and values whose
magnitudes vary during training.

FP16 has more significand bits and a much narrower exponent range. It can preserve
small relative differences better when values are in range, but small gradients can
underflow and large values can overflow. Loss scaling multiplies the loss before
backpropagation and divides the resulting gradients before the optimizer update. It
addresses gradient underflow; it does not repair overflow in activations, logits, or
optimizer state.

BF16 is the baseline in this book because it uses the two-byte tensor path without
requiring scale metadata and has enough range for the models considered here. This is
a numerical default, not a claim that every BF16 kernel is fast.

### FP8

FP8 training normally uses two encodings:

- E4M3 has more significand precision and less range. It is commonly used for forward
  weights and activations.
- E5M2 has more range and less precision. It is commonly used for backward values.

The MI355X uses the OCP FP8 encodings. MI300X and MI325X use FNUZ variants, so an FP8
checkpoint or recipe carrying scales is not portable merely because both machines say
"FP8." AMD documents this gfx942/gfx950 split in its
[workload-optimization guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html).

An FP8 value is useful only with a scale. For a tensor `x`, a simplified symmetric cast
is:

```text
scale = fp8_max / max(abs(x))
q = cast_fp8(x * scale)
x_approx = cast_high(q) / scale
```

Current scaling reads the current tensor to find its range before casting. Delayed
scaling predicts the next scale from an `amax` history. Transformer Engine documents
that delayed scaling avoids the extra range-finding read, but it adds persistent
history and scale variables that must be threaded through the JAX step
**[cited]**. The scaling state is part of the training state: dropping it while
retaining only the parameters changes the numerical recipe.

The exact E4M3/E5M2 assignment, scaling interval, clipping behavior, and forward versus
backward coverage are recipe choices. "FP8" names a family of encodings, not one
complete training algorithm.

### MXFP8, MXFP6, and MXFP4

OCP microscaling divides a tensor into blocks of 32 values and stores an E8M0 scale for
each block. The smaller range within one block lets a low-bit element format retain
useful local detail. The scale is a power of two, so applying it is cheap.

Ignoring alignment and any second layout, the encoded payload is:

| Format | Element payload | One E8M0 scale per 32 | Encoded bytes per value |
|---|---:|---:|---:|
| MXFP8 | 1 byte | 1/32 byte | 1.03125 |
| MXFP6 | 0.75 byte | 1/32 byte | 0.78125 |
| MXFP4 | 0.5 byte | 1/32 byte | 0.53125 |

These are **[analytical]** lower bounds for one packed orientation. Training
implementations may retain rowwise and columnwise forms, BF16 inputs or outputs, cached
weight workspaces, stochastic-rounding state, and alignment padding. Do not use this
table as an allocator prediction.

MXFP8 uses eight-bit elements with local block scales. Transformer Engine's
`MXFP8BlockScaling` recipe uses a block size of 32 and, unlike delayed FP8 scaling,
does not require an `amax` history across steps.

MXFP6 and MXFP4 share the same published MI355X dense matrix peak: approximately
10 PFLOP/s, versus 5 PFLOP/s for FP8/MXFP8 and 2.5 PFLOP/s for BF16/FP16. Bit width
alone therefore does not define throughput. The instruction family, operand layout,
conversion route, and available library kernel matter.

Scaled MFMA consumes the low-bit blocks and their E8M0 scales as part of the matrix
operation. That does not make quantization free: a training route must generate scales,
pack both operand layouts needed by forward and backward, handle tails, and return the
requested graph dtype.

## MI355X Training Paths

Native hardware support and a complete training path are different layers. The table
below describes the intended dataflow of the routes used in this book, without treating
availability as a performance result:

| Route | Ordinary graph tensors | Eligible matrix operands | Accumulator and output | JAX-facing path |
|---|---|---|---|---|
| BF16 | BF16 | BF16 | FP32 accumulate, BF16 output | XLA or hipBLASLt |
| FP16 | FP16 | FP16 | FP32 accumulate, FP16 output | XLA or hipBLASLt |
| FP8 delayed | BF16 | OCP FP8 with per-tensor scales | FP32 accumulate, BF16 output | Transformer Engine delayed scaling |
| MXFP8 | BF16 | MXFP8 with 32-value scales | FP32 accumulate, BF16 output | Transformer Engine block scaling |
| MXFP6 | recipe-dependent | MXFP6 with 32-value scales | FP32 accumulate, higher-precision output | native hardware; no tested recipe here |
| MXFP4 | BF16 | MXFP4 with 32-value scales | FP32 accumulate, BF16 output | MaxText plus JAX-AITER FFI |

These routes describe eligible dense contractions. They do not imply identical
coverage for the attention core, vocabulary projection, norms, loss, collectives, or
optimizer.

MaxText exposes three layers of control:

1. `dtype`, `weight_dtype`, `grad_dtype`, and `mu_dtype` assign ordinary tensor roles.
2. `quantization` selects a matrix implementation and scaling recipe.
3. Backend environment variables and installed libraries determine which route can be
   constructed.

Transformer Engine provides delayed FP8 and MXFP8 recipes. JAX-AITER exposes lower-level
FFI operations and custom gradients for the MXFP4 integration. Both still present a
higher-precision interface to most of the surrounding JAX graph.

### Published Matrix Peaks

AMD publishes the following dense matrix ceilings for one MI355X OAM:

| Matrix operand format | Published peak | Relative to BF16 |
|---|---:|---:|
| FP32 | 157.3 TFLOP/s | 0.0625x |
| BF16 or FP16 | 2.5166 PFLOP/s | 1x |
| OCP FP8 or MXFP8 | 5.0332 PFLOP/s | 2x |
| MXFP6 or MXFP4 | 10.0663 PFLOP/s | 4x |

The table describes the Matrix Core ceiling for supported operand formats. It does not
say that an FP32 master parameter occupies four bytes and somehow executes at the FP32
matrix rate: if it is quantized into an MXFP4 operand before MFMA, the relevant matrix
ceiling is the MXFP4 row.

Likewise, the peak says nothing about how much of a training step consists of eligible
matrix instructions. Vector operations, reductions, memory movement, scale generation,
collectives, optimizer arithmetic, and launch gaps have different ceilings.

### Amdahl Speedup Bound

Let `p` be the fraction of the BF16 baseline step spent in operations that a recipe
actually accelerates, and let `r` be their achieved speedup. If the rest of the step is
unchanged, Amdahl's law gives:

$$
S_{\mathrm{step}}=\frac{1}{(1-p)+p/r}.
$$

Using the published matrix ratios as optimistic values of `r` produces these upper
bounds:

| Accelerated share `p` | FP8/MXFP8, `r=2` | MXFP6/MXFP4, `r=4` |
|---:|---:|---:|
| 60% | 1.43x | 1.82x |
| 70% | 1.54x | 2.11x |
| 80% | 1.67x | 2.50x |
| 85% | 1.74x | 2.76x |
| 90% | 1.82x | 3.08x |

Even an ideal four-times-faster matrix route cannot give a four-times-faster step when
15% of the baseline remains unchanged. At `p=85%`, the Amdahl ceiling is 2.76x.

For a representative MXFP4 recipe, the attribution might be:

```text
Master parameters       FP32
Residual activations    BF16
Eligible weight operand MXFP4
Eligible input operand  MXFP4
MFMA accumulation       FP32
GEMM output             BF16
Attention core          BF16 / backend-specific
Gradients/collectives   BF16 or FP32
Adam state and update   FP32
```

Only the contractions that receive MXFP4 operands belong in the accelerated fraction
`p`. A BF16 attention core, an unquantized vocabulary projection, norms, loss, optimizer,
and exposed collectives remain outside it.

The peak ratio is also only an optimistic value of `r`. Quantization, scale reduction,
packing, layout conversion, tail handling, and a slower kernel can reduce the achieved
matrix speedup. Conversely, reduced operand traffic can help a bandwidth-limited matrix
by more than a FLOP-only model predicts. The baseline profile determines `p`; a matched
kernel comparison determines `r`.

### Compute, Memory, and Communication Effects

Mixed precision changes three different budgets:

**Compute.**

- Eligible matrix operands select a higher-throughput MFMA family.
- FP32 accumulation preserves wider partial sums without paying the FP32 operand rate.
- Casts, scales, packing, and fallback kernels add work around the matrix instruction.
- Faster projections increase the visible share of attention, normalization, optimizer,
  communication, and launch overhead.

**Memory.**

- A transient FP8 or MX operand uses fewer bytes while it remains encoded.
- FP32 master parameters and Adam moments do not shrink merely because GEMMs quantize
  their operands.
- BF16 residuals and GEMM outputs continue to occupy two bytes when the graph returns to
  BF16 after each contraction.
- Scale histories, E8M0 blocks, dual row/column layouts, alignment padding, and
  workspaces consume memory omitted by a simple bits-per-value calculation.
- Persistent memory falls only when the stored tensor role changes, not when a temporary
  compute operand changes.

**Communication.**

- An FSDP AllGather shrinks only if quantization occurs before communication and the
  low-precision representation is what crosses RCCL.
- Quantizing after an FP32 or BF16 AllGather saves matrix traffic but not collective
  traffic.
- Gradient and activation collectives remain unchanged when their payloads stay BF16 or
  FP32.
- Faster local GEMMs can expose collectives that were previously hidden behind compute.

[Chapter 5]({{ '/pages/5-making-the-model-fit' | relative_url }}) builds the complete
memory ledger. The later Llama 70B case study measures `p`, `r`, transient workspace,
collective payloads, and the resulting end-to-end speedup rather than assuming them
from the peak table.

## Published Convergence and Time-to-Quality Evidence

The literature does not support one blanket claim that low-precision training is
lossless. Results depend on which tensor roles are quantized, the scaling granularity,
and whether the comparison is made at equal tokens or equal wall time.

- Micikevicius et al.,
  [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740) (ICLR 2018),
  matched FP32 accuracy across vision, speech, and language tasks using FP32 master
  weights, FP32 accumulation, and loss scaling. The experiments show that loss
  scaling can determine whether FP16 converges, but they do not establish a
  hardware-independent time-to-quality ratio.

- Kalamkar et al.,
  [*A Study of BFLOAT16 for Deep Learning Training*](https://arxiv.org/abs/1905.12322),
  reported FP32-matching results in the same number of iterations across image,
  speech, language, generative, and recommendation workloads without loss scaling
  or hyperparameter changes. This is the main empirical basis for treating BF16 as
  the control format.

- Micikevicius et al.,
  [*FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433), reported
  quality close to FP16/BF16 for CNNs, RNNs, Transformers, and GPT models up to
  175B parameters with unchanged hyperparameters. Their hybrid recipe uses E4M3
  in the forward pass and E5M2 for gradients.

- Wortsman et al.,
  [*Stable and Low-Precision Training for Large-Scale Vision-Language Models*](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20bd42d82998bc61732c00452228e814-Abstract.html)
  (NeurIPS 2023), kept weight-gradient GEMMs in higher precision while quantizing
  forward and activation-gradient GEMMs. SwitchBack matched BF16 within 0.1
  percentage points on a 1B-parameter CLIP model and improved end-to-end speed by
  13–25%. This result identifies weight gradients as a numerically sensitive path.

- Peng et al.,
  [*FP8-LM: Training FP8 Large Language Models*](https://arxiv.org/abs/2310.18313),
  reported BF16-comparable pretraining and downstream results from GPT-7B through
  GPT-175B. For GPT-175B, their system reduced training time by 37% relative to
  Transformer Engine and used 42% less memory. This is a preprint, and its
  distributed system differs from the JAX/ROCm path used here.

- The
  [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437) describes a
  14.8T-token FP8 run without irrecoverable loss spikes or rollbacks. A controlled
  DeepSeek-V2 experiment over approximately one trillion tokens kept relative loss
  error below 0.25% against BF16. The full V3 run demonstrates feasibility at
  scale, but it is not a paired BF16 time-to-quality experiment.

- Rouhani et al.,
  [*Recipes for Pre-training LLMs with MXFP8*](https://arxiv.org/abs/2506.08027),
  found that E4M3 operands with upward-rounded E8M0 scales matched BF16 accuracy
  on models up to 8B parameters. Its convergence recipe is relevant to MI355X.
  Its Blackwell throughput results are not directly portable to ROCm.

- [*Pretraining Large Language Models with MXFP4 on Native FP4 Hardware*](https://arxiv.org/abs/2605.09825)
  reports Llama 3.1-8B pretraining on MI355X. Quantizing forward and
  activation-gradient GEMMs required approximately 8–11% more tokens to reach the
  target perplexity. Adding MXFP4 weight gradients increased token overhead to
  26–27%; a deterministic Hadamard transform reduced it to 8–9%. A 20% increase
  in step throughput then produced approximately 9–10% lower time to target than
  FP8. This is the closest external comparison to the MXFP4 path considered here.

Two adjacent results clarify the role of scaling granularity. Dettmers et al.,
[*8-bit Optimizers via Block-wise Quantization*](https://arxiv.org/abs/2110.02861)
(ICLR 2022), retained FP32-level optimizer behavior while reducing optimizer-state
memory. Xi et al.,
[*Jetfire*](https://proceedings.mlr.press/v235/xi24b.html) (ICML 2024), found that
per-block INT8 data flow preserved FP16-level quality more reliably than coarser
quantization and reported a 1.42x transformer-block speedup. Neither paper proves
that an equivalent JAX/ROCm route is available.

These results motivate three reporting requirements for the Llama 70B case study:

1. compare loss against consumed tokens;
2. report step throughput separately; and
3. report time to a fixed quality target only when every precision arm reaches it.

## References

- [AMD CDNA 4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf).
  Operand, accumulator, and scale layouts for dense and scaled MFMA.
- [AMD Instinct MI300/MI350 workload optimization](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html).
  MI355X peaks, OCP FP8, and native MXFP8/MXFP6/MXFP4 support.
- [OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
  The MX block formats and E8M0 scales.
- [Transformer Engine FP8 delayed scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html).
  Per-tensor scale history and JAX recipe.
- [Transformer Engine MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html).
  Block size, recipe, and scale behavior. Its published hardware requirement is
  NVIDIA-specific; the ROCm route in this chapter is verified from the pinned AMD fork
  and experiment.
- [MaxText base configuration](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml).
  Dtype, quantization, and optimizer fields.
- [JAX-AITER alpha2](https://github.com/ROCm/jax-aiter/tree/35b7175c).
  The gfx950 MXFP4 FFI, custom gradients, and supported operation surface.
- [Llama 7B JAX fundamentals](https://github.com/clarkechong/llama7b-jax-fundamentals/tree/5f996a88).
  Explicit tensor-role casts and optimizer state.
