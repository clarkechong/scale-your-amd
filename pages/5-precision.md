---
layout: distill
title: "Precision as a Training Decision"
description: "Choose tensor dtypes, scaling recipes, and MI355X execution paths without mistaking a selected config for a low-precision kernel."
date: 2026-09-13

section_number: 5

previous_section_url: "/pages/4-profiling"
previous_section_name: "Chapter 4: Profiling"

next_section_url: "/pages/6-memory"
next_section_name: "Chapter 6: Making the Model Fit"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "A Recipe Assigns Dtypes to Tensor Roles"
  - name: "Formats and Scaling"
    subsections:
      - name: "BF16 and FP16"
      - name: "FP8"
      - name: "MXFP8, MXFP6, and MXFP4"
  - name: "MI355X Training Paths"
  - name: "Compute and Memory Effects"
  - name: "The Llama 70B Precision Sweep"
  - name: "Verify the Executed Path"
  - name: "Convergence Is the Guardrail"
  - name: "A Precision Decision Procedure"
  - name: "Decision Table"
  - name: "References"
---

Low-precision training is not a model-wide dtype switch. It is an assignment of
storage, compute, accumulation, and scaling formats to individual tensor roles. On the
MI355X, choosing `quantization=te_mxfp8` requests a recipe. It does not prove that every
dense projection reached an MXFP8 matrix instruction, that attention used MXFP8, or that
the optimizer stopped using FP32.

This chapter separates four kinds of evidence:

- **[source]** comes from checked-in code, configuration, or a manifest.
- **[measured]** requires a complete Appendix F bundle from an MI355X run.
- **[analytical]** follows from format widths, published peaks, or arithmetic and has
  not been confirmed by a profile.
- **[cited]** is a result or behavior reported by a named external source.

Support statements are verified against
`rocm/jax-training:maxtext-v26.6`, JAX/JAXLIB and ROCm PJRT/plugin 0.11.0,
[Transformer Engine 2.17 with the gfx950 workspace patch](https://github.com/clarkechong/TransformerEngine/tree/fix/jax-gfx950-mxfp8-workspace),
the ROCm [MaxText MXFP4 branch at `b437942a`](https://github.com/ROCm/maxtext/tree/b437942a),
and [JAX-AITER alpha2 at `35b7175c`](https://github.com/ROCm/jax-aiter/tree/35b7175c),
on **13 September 2026**. A hardware format can be native while the required JAX
training integration is absent.

## A Recipe Assigns Dtypes to Tensor Roles

Start by writing a dtype ledger. These MaxText fields control different objects:

```yaml
dtype: "bfloat16"
weight_dtype: "float32"
grad_dtype: "float32"
mu_dtype: "float32"
quantization: "te_fp8_delayedscaling"
```

`dtype` is the ordinary activation and compute dtype. A quantization recipe can replace
eligible `dot_general` calls with lower-precision operations while the rest of the model
continues to use `dtype`.

`weight_dtype` is the stored trainable parameter dtype. In the Llama 70B experiments it
is FP32, so each step casts or quantizes from an FP32 parameter to the compute format.
This is the master copy that the optimizer updates.

`grad_dtype` is a conditional cast in the pinned MaxText trainer: a gradient leaf is
cast only when it arrives in FP32. The trainer does not upcast a BF16 gradient to
FP32. Reducing `grad_dtype` can save memory and communication for FP32 leaves, but
setting it to FP32 does not prove that every stored gradient is FP32. Record gradient
dtypes from the lowered program or runtime state.

`mu_dtype` controls Adam's first moment. In the pinned MaxText source, Adam's second
moment has no independent field and inherits `weight_dtype`. Lowering `mu_dtype` does
not lower the second moment.

`quantization` selects the dense-dot implementation and its scaling recipe. It does not
change every operation in the layer. Norm reductions, softmax, loss evaluation, gradient
accumulation, optimizer arithmetic, and scale or `amax` calculations commonly stay at
BF16 or FP32.

The raw-JAX Llama 7B model makes the exception list visible:

- parameters and Adam moments are FP32;
- dense and attention inputs use `dtype`, normally BF16;
- RMSNorm computes the variance in FP32;
- logits are cast to FP32 before cross-entropy;
- the optimizer update is applied to the FP32 parameter tree.

That is a mixed-precision recipe even though it has no `quantization` value. The
[Llama 7B source](https://github.com/clarkechong/llama7b-jax-fundamentals/tree/5f996a88)
is useful because each cast is explicit.

The Mixtral recipe is different:

```yaml
dtype: "bfloat16"
weight_dtype: "bfloat16"
grad_dtype: "float32"
mu_dtype: "bfloat16"
quantization: ""
```

Its parameters and both Adam moments are BF16 in the pinned MaxText implementation.
Although the config requests `grad_dtype: float32`, the trainer does not upcast BF16
parameter gradients. Treat those gradients as BF16 unless the lowered program or
runtime state proves otherwise. Expert sparsity changes how many weights execute per
token, but optimizer state still exists for all eight experts. Precision and memory
must therefore be accounted for against total parameters, not the two experts selected
for one token.

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

The Llama 70B FP16 arm sets `dtype=float16` and keeps FP32 parameters, gradients, and
moments. It does not add a separate loss-scaling control. That makes finite gradients,
gradient norms, and validation behavior part of the acceptance test rather than
something inferred from the name FP16.

BF16 is the baseline in this book because it uses the two-byte tensor path without
requiring scale metadata and has enough range for the tested models. This is a
numerical default, not a claim that every BF16 kernel is fast.

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
**[cited]**. Dropping those variables while retaining only `params` silently changes the
recipe.

The tested MaxText path is:

```yaml
dtype: "bfloat16"
quantization: "te_fp8_delayedscaling"
```

BF16 remains the surrounding dtype. Eligible dense dots use Transformer Engine's FP8
delayed-scaling recipe. Treat the exact E4M3/E5M2 assignment, backward coverage, and
collective dtype as properties to inspect in the pinned implementation and HLO, not as
facts implied by `dtype=bfloat16`.

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
`MXFP8BlockScaling` recipe is stateless across steps, unlike delayed scaling, and uses a
block size of 32. The Llama 70B arm selects it with:

```yaml
dtype: "bfloat16"
quantization: "te_mxfp8"
```

On the tested gfx950 stack it also requires:

```bash
export NVTE_ROCM_ENABLE_MXFP8=1
```

and a Transformer Engine workspace-size patch. The runner probes the installed Python
source and stops if the patch is missing. This is a useful pattern: fail before timing
rather than let an unpatched path run slowly and call it MXFP8.

MXFP6 has native MI355X matrix instructions and the same published dense matrix peak as
MXFP4: 10 PFLOP/s, versus 5 PFLOP/s for FP8/MXFP8 and 2.5 PFLOP/s for
BF16/FP16 **[analytical]** from AMD's published peak table. MXFP6 may therefore be the preferable
numerical format when both routes are equally optimized. The tested MaxText,
Transformer Engine, and JAX-AITER integrations do not expose an MXFP6 training recipe,
so that comparison is theory, not a runnable recommendation.

The tested MXFP4 route is not Transformer Engine's `te_nvfp4`; NVFP4 is a different
format and hardware path. It uses ROCm's MaxText feature branch and JAX-AITER alpha2:

```yaml
dtype: "bfloat16"
quantization: "aiter_fp4"
use_jax_aiter: true
aiter_attention: false
```

`jax_aiter.gemm_fp4_bf16` accepts BF16 tensors, performs MXFP4 casts and matrix
operations, returns BF16, and supplies custom gradients for forward, activation
gradient, and weight gradient. The pinned MaxText branch applies MXFP4 to MLP and
Q/K/V/O projections; the logits projection remains unquantized. The fused attention
core remains Transformer Engine BF16. The
`AITER_FP4_ATTN=1` environment variable refers to attention *projections* in this
recipe, not the score/softmax/value attention kernel.

## MI355X Training Paths

Hardware support and framework support are separate:

| Requested compute | MI355X hardware | Tested JAX route | Current status |
|---|---|---|---|
| BF16 | Native | XLA/hipBLASLt; TE or JAX-AITER for selected ops | Unverified under Appendix F |
| FP16 | Native | XLA/hipBLASLt with `dtype=float16` | Unverified under Appendix F |
| FP8 | Native OCP E4M3/E5M2 | Transformer Engine delayed scaling | Unverified under Appendix F |
| MXFP8 | Native, 32-value blocks | Transformer Engine block scaling | Experimental patch; post-patch run unverified |
| MXFP6 | Native, 32-value blocks | No recipe in the tested repositories | Unsupported in the tested stack |
| MXFP4 | Native, 32-value blocks | ROCm MaxText feature branch plus JAX-AITER alpha2 FFI | Experimental; artifact bundle blocked |

The table is about training the dense projections in these experiments. It does not
claim equal coverage for attention, norms, embeddings, loss, collectives, or optimizer
updates.

MaxText has three layers of controls:

1. `dtype`, `weight_dtype`, `grad_dtype`, and `mu_dtype` set the ordinary tensor roles.
2. `quantization` selects the dot implementation and scale recipe.
3. Backend-specific environment variables control which implementation is reachable.

Transformer Engine also has a direct JAX API. A custom model can create a
`DelayedScaling` or `MXFP8BlockScaling` recipe and pass it through `te.autocast`, or use
`te_flax.make_dot_general_cls(recipe)` for selected dense layers. With delayed scaling,
initialize the layer inside the autocast context and carry the full variable collection.

JAX-AITER is lower level. It exposes JAX functions backed by XLA FFI and supplies
`custom_vjp` rules and sharding behavior. That makes it suitable for a MaxText
integration, but it also means the integration owns restrictions on shape, layout,
workspace caching, rematerialization, and FSDP weight gradients.

## Compute and Memory Effects

AMD publishes these dense matrix peaks for MI355X:

| Matrix format | Published peak | Relative to BF16 |
|---|---:|---:|
| BF16 or FP16 | 2.5 PFLOP/s | 1x |
| FP8 or MXFP8 | 5.0 PFLOP/s | 2x |
| MXFP6 or MXFP4 | 10.0 PFLOP/s | 4x |

These are **[analytical]** ceilings, not expected train-step ratios. A lower-precision
matrix unit does not accelerate the data loader, FP32 optimizer, norms, unfused casts,
scale calculation, BF16 attention, collectives that remain high precision, or host
gaps.

If a fraction `f` of BF16 step time can accelerate by `r`, the ideal Amdahl bound is:

```text
speedup <= 1 / ((1 - f) + f / r)
```

For FP8 or MXFP8, `r` is at most 2 against BF16. For MXFP4, `r` is at most 4. Casts,
scale reductions, layout conversions, extra workspaces, and slower fallback kernels
make the achieved ratio smaller. A profile supplies `f`; the hardware peak table does
not.

Low precision changes more than compute:

- Encoded weights and activations use fewer HBM bytes while they are in the low format.
- FSDP weight AllGathers can shrink if the gathered buffer is the quantized operand.
- Gradient and activation collectives do not shrink when they remain BF16 or FP32.
- FP32 master weights and optimizer state dominate persistent memory in the Llama 70B
  recipe, so an FP8 GEMM does not make the model state eight bits wide.
- Scale histories, block scales, dual layouts, and workspaces add memory that a
  bytes-per-element estimate omits.
- A faster matrix path can expose communication or fixed-cost kernels that were hidden
  in the BF16 run.

Chapter 6 builds the complete state and activation ledgers. The useful rule here is
that a compute format saves persistent memory only when the stored tensor role also
changes.

## The Llama 70B Precision Sweep

The [Llama 70B repository](https://github.com/clarkechong/llama70b-mixed-precision-training/tree/f3dab369)
defines a current fixed train-step study: 8x MI355X, FSDP-8, sequence length
4096, 491,520 token positions per update, and 30 synthetic steps.

The point estimates below are a historical source record, not measurements under
Appendix B. An audit traced them to
`archive/v26.6-migration-20260908`, where all six arms used the feature-branch
MaxText tree at `b437942a`. BF16, FP16, FP8, and MXFP8 were marked
noncanonical because completion was not verified. MXFP4 has a separate retained
row, but no active Appendix F bundle. The current main-branch launchers request
stock MaxText for BF16, FP16, and FP8, so these historical values cannot rank the
current cohort.

| Arm | Tokens/s/GPU | Seconds/update | Ratio to BF16 | Historical qualification |
|---|---:|---:|---:|---|
| FP32 | BLOCKED | BLOCKED | — | Archived result invalid; no current timing |
| BF16 | 2,319.7 | 26.486 | 1.000x | **[source]** feature-branch cohort; completion unverified |
| FP16 | 2,502.4 | 24.552 | 1.079x | **[source]** feature-branch cohort; completion unverified |
| FP8 delayed scaling | 4,109.7 | 14.950 | 1.772x | **[source]** feature-branch cohort; completion unverified |
| MXFP8, unpatched | 2,052.2 | 29.938 | 0.885x | **[source]** three fallbacks per layer; completion unverified |
| MXFP4 | 5,254.9 | 11.692 | 2.265x | **[source]** feature branch; MLP and Q/K/V/O projections |

Tokens/s/GPU is `491,520 / (seconds/update * 8)`. No post-patch MXFP8 timing is present,
so there is no valid MXFP8 performance result to compare. The unpatched value is
retained as a fallback warning, not as an MXFP8 result.

The historical FP8 ratio is 1.772x against a 2x matrix ceiling; the historical
MXFP4 ratio is 2.265x against a 4x ceiling. **[analytical]** Amdahl's law says
that lower whole-step ratios are expected
whenever attention, optimizer work, casts, communication, or other fixed paths occupy a
material fraction of the step. Attributing the gap among those causes requires a
matched, accepted profile. These values do not supply one.

The FP16/BF16 peak ratio is 1x. The historical 1.079x difference is therefore a
kernel, schedule, or run-level observation, not an FP16 hardware-throughput
entitlement.

None of the current launchers sets `RCCL_WARP_SPEED_AUTO=0`, which AMD's MI355X
MaxText guidance requires to avoid a documented NaN-loss hazard. The historical
environment has not been recovered. The current timing and convergence cohorts
must be rerun with this value frozen or prove that it was present in the captured
environment.

## Verify the Executed Path

Treat each low-precision run as a proof obligation.

First, record the effective configuration and exact binaries:

```bash
python3 -c "import jax; print(jax.__version__, jax.devices())"
python3 -c "import transformer_engine; print(transformer_engine.__version__)"
```

For MXFP8, retain the runner's source probe and stop if the workspace patch is absent.
For MXFP4, check for the MaxText integration and the three required JAX-AITER FFI
libraries before starting the run.

Second, inspect lowered HLO. Find the eligible projections and answer:

- Are their operands represented in the requested format or passed to the expected
  custom call?
- Are block-scale tensors present for MX formats?
- Does an unexpected `convert` return the operands to BF16 before the dot?
- Did sharding insert a BF16 AllGather followed by local quantization, or move the
  quantized representation?
- Which attention operations remain BF16?

Third, capture a warmed train step with `rocprofv3 --kernel-trace`. Positive evidence is
the expected Transformer Engine or JAX-AITER custom-call kernel at the projection
shapes. Negative evidence is equally important: only BF16 hipBLASLt GEMMs, repeated
quantize/dequantize fusions around a BF16 dot, or a generic fallback dominating the
step.

Fourth, check numerics:

```text
loss is finite
gradient norm is finite
parameter and optimizer trees contain the intended dtypes
scale or amax state changes when the recipe requires it
fraction clipped or saturated is logged
```

Finally, report tokens/s/GPU before MFU. Performance alone is not proof of a format, but
a large regression against BF16 is a reason to stop and inspect the path before running
convergence.

## Convergence Is the Guardrail

The train-step sweep uses synthetic reused data and says nothing about model quality.
The separate convergence launchers request C4, one verified Llama 2 tokenizer, five
precision arms, 2,034 updates, 491,520 token positions per update, 5% warmup,
cosine decay, and 20 validation batches every 100 steps. The product
`2,034 × 491,520 = 999,751,680` is nominal packed-sequence capacity; actual
non-padding training tokens require the retained weight or segmentation metrics.

The experiment owner reports that the completed curves were near-identical over
this horizon. The current branch and archived result inventory do not contain the
metric files, plot, immutable data/tokenizer revisions, or arm-to-run mapping.
This is therefore recorded project status, not a publishable **[measured]** claim.

Once those artifacts are retained, the result can serve as a descriptive guardrail:

- it rejects immediate divergence and large short-horizon regressions for this exact
  Llama 2 70B recipe;
- it does not prove equal final quality after a full pretraining token budget;
- it does not establish seed variance;
- it does not transfer automatically to a different model, optimizer, scale policy,
  dataset order, sequence length, or sharding;
- it does not turn the unpatched MXFP8 throughput result into an acceptable execution
  path.

No retained pre-run equivalence threshold has been identified. The current
description must remain “near-identical curves” rather than “equivalent
convergence.”

For a new recipe, compare against BF16 with the same tokens, data order, initialization,
optimizer, and evaluation batches. Log training and validation loss, gradient norm,
non-finite counts, and range saturation. Stop on persistent divergence from the BF16
curve, repeated overflows, scale collapse, or a gradient-norm regime change. A faster
step that needs more tokens to reach the target can increase total training time.

## A Precision Decision Procedure

1. Start with BF16 compute and high-precision loss, reductions, gradients, and optimizer
   state. Confirm the model and input pipeline are correct.
2. Write the tensor-role ledger. Do not use one phrase such as "FP8 training" in place
   of `dtype`, `weight_dtype`, `grad_dtype`, `mu_dtype`, quantized operations, scaling,
   and exceptions.
3. Choose the hardware route. On MI355X use OCP FP8 or MX formats; do not reuse a
   gfx942 FNUZ recipe.
4. Select the narrowest low-precision scope that covers the expensive dense dots.
   Keep norms, loss, reductions, and the optimizer high precision until evidence
   supports changing them.
5. Compile one warmed step and prove the path from effective config, HLO, scale state,
   and kernel trace.
6. Measure tokens/s/GPU, step-time distribution, peak memory, and the fixed-time
   fraction. Use Amdahl's bound to explain the maximum useful speedup.
7. Run a fixed-token convergence comparison against BF16. Preserve raw metric files,
   seeds, data order, and saturation statistics.
8. Adopt the format when it raises tokens/s/GPU at the fixed workload, the intended
   kernels execute, and the predeclared convergence guardrail passes. Report
   time-to-quality only when all arms reach a target chosen before reading the final
   curves.

## Decision Table

Status applies to the pinned MI355X JAX stack, not to all ROCm software.

| Knob | What it buys | What it costs | How to set it | Status on ROCm | Verified on |
|---|---|---|---|---|---|
| BF16 compute | Safe two-byte baseline | Lower matrix peak than FP8/MX | `dtype=bfloat16`, `quantization=` | Unverified under Appendix F | Source recipe, v26.6, 2026-09-13 |
| FP16 compute | Two-byte path with more significand bits | Narrow range; monitor underflow/overflow | `dtype=float16`, `quantization=` | Unverified under Appendix F | Source recipe, v26.6, 2026-09-13 |
| FP8 delayed scaling | Up to 2x BF16 matrix peak; one-byte dot operands | Scale history, casts, convergence check | `dtype=bfloat16 quantization=te_fp8_delayedscaling` | Unverified current cohort | Source recipe, v26.6, 2026-09-13 |
| MXFP8 block scaling | Up to 2x BF16 peak with local 32-value scales | Scale/layout metadata; patched workspace sizing | `quantization=te_mxfp8`, `NVTE_ROCM_ENABLE_MXFP8=1` | Experimental; post-patch run unverified | TE 2.17 patch source, 2026-09-13 |
| MXFP6 | Up to 4x BF16 peak; more precision than MXFP4 at the same published peak | No tested JAX training integration | No field in tested stack | Unsupported in tested stack | Repository audit, 2026-09-13 |
| MXFP4 dense projections | Up to 4x BF16 matrix peak; smallest encoded payload here | Alpha FFI, branch patch, dual-layout/workspace and remat constraints | `quantization=aiter_fp4 use_jax_aiter=true` plus the pinned environment | Experimental; current artifact blocked | MaxText `b437942a`, JAX-AITER `35b7175c`, 2026-09-13 |
| FP32 gradients | Stable gradient storage and reduction where gradient leaves are FP32 | Four bytes per FP32 gradient and larger collectives | `grad_dtype=float32` | Unverified per leaf | Llama configs request FP32; Mixtral requires HLO/runtime verification, 2026-09-13 |
| FP32 master and Adam state | Stable updates | Dominates persistent memory | `weight_dtype=float32 mu_dtype=float32` | Unverified runtime state | Llama 7B/70B source configs, 2026-09-13 |
| BF16 optimizer state | Lower persistent state memory | Changes optimizer numerics | `weight_dtype=bfloat16 mu_dtype=bfloat16` | Unverified runtime state | Mixtral source config, 2026-09-13 |

**Recommendation status: BLOCKED.** BF16 remains the control recipe. FP8 and
MXFP4 are candidates for a fixed-workload rerun; no current arm has the required
throughput, memory, kernel-proof, environment, and convergence artifacts. The
fallback is BF16. Retest after any ROCm, JAX, MaxText, Transformer Engine,
JAX-AITER, or RCCL change.

## References

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
- [Llama 70B mixed-precision experiments](https://github.com/clarkechong/llama70b-mixed-precision-training/tree/f3dab369).
  Fixed recipes, timing summary, and convergence runners.
- [Llama 7B JAX fundamentals](https://github.com/clarkechong/llama7b-jax-fundamentals/tree/5f996a88).
  Explicit tensor-role casts and optimizer state.
- [Mixtral 8x22B distributed strategies](https://github.com/clarkechong/mixtral8-22b-distributed-strategies/tree/a32b51d6).
  BF16/FP32 state assignment for the MoE case.
