---
layout: distill
title: "A Map of Kernel Backends on JAX"
description: "How dense GEMM, attention, fused pointwise work, and MoE reach MI355X kernels through XLA, libraries, Triton, and FFI."
date: 2026-09-13

section_number: 7

previous_section_url: "/pages/6-jax-shardings-to-a-training-mesh"
previous_section_name: "Chapter 6: Sharding"

next_section_url: "/pages/8-mixture-of-experts-on-mi355x"
next_section_name: "Chapter 8: Mixture of Experts"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "The Decision"
  - name: "How a JAX Operation Reaches a Kernel"
  - name: "Dense GEMM"
  - name: "Attention Routes"
  - name: "Fused Pointwise and Reduction Work"
  - name: "MoE Kernel Preview"
  - name: "Restrictions That Change the Route"
  - name: "Correctness and Kernel Proof"
  - name: "Fallback Order"
  - name: "Versioned Reachability"
  - name: "Primary References"
---

## The Decision

For each expensive operation, choose a route that has a correct forward pass, a
correct gradient, acceptable memory use, and a kernel implementation for the exact
MI355X shape. A framework setting is only a request. The optimized HLO and device
trace establish what ran.

This chapter uses the following evidence labels:

- **[analytical]** follows from shapes, dtypes, or program structure.
- **[source]** comes from checked-in code, configuration, or a pinned source tree.
- **[measured]** requires a retained artifact from a repository experiment. A
  measured route does not imply a speed ranking unless timing artifacts are linked.
- **[cited]** comes from a named external source.

The three case repositories use
`docker.io/rocm/jax-training:maxtext-v26.6`. The Llama 7B route comparison pins
JAX, `jaxlib`, the ROCm plugin, and PJRT to 0.11.0. Llama 70B additionally uses
MaxText `release/v26.6` at `b47d74bf`, a Transformer Engine 2.17 workspace patch,
and JAX-AITER alpha 2 for MXFP4. These pins matter because kernel eligibility,
fallbacks, and flag defaults change between releases.

## How a JAX Operation Reaches a Kernel

**[cited]** XLA:GPU has four routes relevant to these cases:

1. **XLA emitter.** A fusion is lowered through an XLA loop, reduction, transpose,
   scatter, or related emitter to LLVM and AMDGPU code. Elementwise chains usually
   start here.
2. **Library custom call.** XLA rewrites an operation to a library call. On ROCm,
   dense matrix products normally reach rocBLAS or hipBLASLt, and collectives reach
   RCCL. HLO target names may retain `cublas`, `cudnn`, or `nccl`; those compatibility
   names do not prove that a CUDA library ran.
3. **Triton.** XLA can form a Triton GEMM or softmax-shaped fusion. Pallas also uses
   Triton as its ROCm GPU backend.
4. **XLA FFI.** A Python package can register an external operation and its ROCm
   implementation. Direct JAX-AITER uses this route.

The route is selected after tracing and can change after partitioning, layout
assignment, fusion, and autotuning:

```text
JAX primitive + shape + dtype + sharding
  -> StableHLO
  -> partitioned and optimized HLO
  -> XLA emitter | library custom call | Triton fusion | FFI custom call
  -> thunk
  -> one or more HIP kernel launches
```

One HLO fusion produces one GPU kernel, but one library custom call can enqueue
several kernels. Conversely, a source-level expression can be split into several
HLO operations before code generation. Count kernels from the trace, not from the
Python expression.

### Selection evidence

Save all three levels:

- the post-optimization HLO, including `custom_call_target` and fusion
  `backend_config`;
- the effective `.debug_options` file;
- a `rocprofv3 --kernel-trace` capture for a warmed step.

For a Triton GEMM, the optimized HLO normally contains a fusion kind such as
`__triton_gemm` and its selected tile. For a library GEMM, look for a GEMM custom
call, then identify the rocBLAS or hipBLASLt launch in the trace. For an FFI route,
prove both the FFI target and the AITER kernel below it.

## Dense GEMM

### Default route

**[cited]** `jax.lax.dot_general`, `jnp.matmul`, and Flax dense layers lower to HLO
`dot`. For the FP32, FP16, and BF16 projections in the case studies, XLA may select:

- a rocBLAS or hipBLASLt custom call;
- a Triton GEMM fusion when the operation and layout are eligible;
- an XLA-generated kernel for a small or unsupported dot.

The choice is shape-specific. The local matrix dimensions, transpose state, batch
dimensions, physical layout, epilogue, dtype, and available workspace are all part
of the problem. A result for one projection cannot be transferred to every
projection in the model.

All three repositories currently set:

```bash
--xla_gpu_autotune_level=4
--xla_gpu_enable_triton_gemm=true
```

These flags make candidates eligible; they do not force every dot through Triton.
The autotuner can still choose a library route. Chapter 9 gives the controlled
sweep and cache rules.

### Training requires three matrix products

**[analytical]** A custom dense kernel used for training must cover the forward
matrix product and the two products generated by the vector-Jacobian product:
activation gradient and weight gradient. A fast forward-only implementation is an
inference kernel until a correct VJP and both backward routes are present.

Plain JAX dots obtain gradients by transposing dot operations and sending those
new dots back through XLA selection. Transformer Engine and JAX-AITER low-precision
paths instead own more of the quantization, scale handling, workspace, and gradient
contract. Keep the kernel proof for each direction.

### Low-precision case routes

- **FP8 and MXFP8:** the Llama 70B scripts request Transformer Engine through
  `te_fp8_delayedscaling` and `te_mxfp8`. The outer model dtype remains BF16.
- **MXFP4:** the pinned feature branch uses JAX-AITER FFI for selected projections.
  Attention remains BF16 in the case recipe, and `aiter_attention=false` and
  `aiter_rmsnorm=false` are explicit.
- **BF16 JAX-AITER GEMM:** current JAX-AITER exposes a BF16 GEMM FFI, but none of the
  three case baselines selects it. It is not part of their measured comparison.

### Workspace is part of eligibility

**[source]** The pinned Transformer Engine patch declares 64 MiB of base GEMM
workspace on gfx950. Its MXFP8 path raises that allocation when the packed scale
scratch is larger:

```text
max(64 MiB, rhs_scale_elements + 2 * lhs_scale_elements) + alignment
```

The unpatched Llama 70B MXFP8 run is recorded as a slow fallback in its repository.
No post-patch timing or memory artifact is present, so this chapter makes no
performance claim for the patch. The important rule is that a workspace fix can
change route validity and peak memory; it is not a harmless timing-only change.

## Attention Routes

The Llama 7B experiment holds BF16, causal masking, sequence length 4096, batch
four, and `minimal_with_context` rematerialization fixed while changing only the
attention call.

### XLA dot-product attention

```python
jax.nn.dot_product_attention(
    q, k, v, scale=scale, is_causal=True, implementation="xla"
)
```

**[analytical]** This is the broadest baseline in the experiment. It builds
attention from JAX operations and has a quadratic score matrix in sequence length.
It is therefore a correctness fallback only when that intermediate fits.

**Unverified historical observation.** The repository source records that this route
emitted XLA fusions and GEMM-fusion kernels, with no AITER attention kernel. It
also records that `implementation="cudnn"` did not provide the ROCm fused route
in the pinned environment. The raw trace and timing artifact are not checked in,
so this is attribution to reproduce, not a speed result.

### Transformer Engine

The raw-JAX arm calls Transformer Engine's `DotProductAttention` directly. The
MaxText arm requests `attention=cudnn_flash_te`. On ROCm, that historical MaxText
name does not mean cuDNN executed.

The experiment pins the ROCm selector:

```bash
NVTE_FUSED_ATTN_CK=1
NVTE_FUSED_ATTN_AOTRITON=0
```

**[cited]** Transformer Engine first filters by dtype, GPU architecture, QKV
layout, mask, bias, dropout, head counts, sequence lengths, and head dimensions.
In the pinned ROCm source it prefers the CK backend before AOTriton when both are
eligible. If no fused backend is eligible, a higher-level call can fall back to
unfused JAX attention. That is why the environment variables alone are not proof.

### CK/AITER through Transformer Engine

**Unverified historical observation.** The Llama 7B source records these prior kernel signatures under
the Transformer Engine route:

```text
aiter::fmha_fwd_hd128_bf16_causal
aiter::fmha_bwd_hd128_bf16_causal_a32_psskddv
```

This is one route with two layers: Transformer Engine owns the JAX-facing
attention interface and selects a ROCm CK/AITER implementation. Do not report
“TE versus AITER” unless the AITER arm is the direct FFI route below.
The raw trace is not checked in, so Chapter 10 must reproduce this attribution.

### Direct JAX-AITER

The direct arm calls:

```python
from jax_aiter.mha import flash_attn_func

out = flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=scale,
    causal=True,
    deterministic=True,
)[0]
```

**[source]** JAX-AITER supplies a custom VJP and separate forward and backward FFI
handlers. The Llama 7B setup requires the full MHA build, including
`mha_fwd_ja.so`, `mha_bwd_ja.so`, `libmha_fwd.so`, and `libmha_bwd.so`, compiled
for `gfx950`. The tested arm covers BF16, batched BSHD tensors, causal masking, no
dropout, and head dimension 128. Broader features in newer JAX-AITER documentation
are not results for this alpha-2 experiment.

Current JAX-AITER source also contains route-specific backward guards. Dropout,
bias, sliding windows, short-sequence kernels, packed layouts, and unequal query
and key lengths can select a different CK/ASM backward. Record forward and
backward kernel names separately.

### Tokamax Pallas-Triton

The fourth arm calls `tokamax.dot_product_attention(...,
implementation="triton")`. **[source]** Tokamax is a JAX/Pallas kernel library with
an XLA fallback when automatic selection is used. The experiment asks for Triton
explicitly so an unsupported route should fail rather than silently become XLA.

The pinned wrapper has two compatibility adjustments:

- it supplies an explicit JAX dot algorithm for BF16;
- it patches Tokamax's private GPU guard because versions 0.0.12 through 0.0.14
  expected an NVIDIA-style compute capability rather than `gfx950`.

The ROCm JAX fork later added direct architecture detection for Pallas-Triton.
That does not remove the experiment's Tokamax-version pin or turn its private
patch into a supported public API. Treat this arm as experimental until its
forward and backward kernels are captured under the frozen environment.

### Attention restrictions to record

For every arm, report:

- Q, K, and V layout and local shapes;
- BF16, FP16, FP8, or mixed input and accumulation dtypes;
- training versus inference and forward versus backward;
- causal, padding, packed-sequence, local-window, and bias semantics;
- dropout and determinism;
- head dimension, query heads, and KV heads;
- rematerialized residuals and reported temporary memory;
- the selected forward and backward kernels.

Changing any item can change the eligible backend. A training route is supported
only for the combination actually tested.

## Fused Pointwise and Reduction Work

**[cited]** XLA's main opportunity for elementwise chains is fusion. Values
inside one HLO fusion remain in registers or shared memory instead of being
materialized in HBM. A chain such as `silu(gate) * up` is therefore normally one
kernel if layouts, users, and sharding allow it.

Relevant work in these cases includes:

- RMSNorm reductions and scaling;
- RoPE arithmetic;
- bias-free residual adds;
- SwiGLU's SiLU-and-multiply;
- softmax or log-softmax;
- integer-label cross entropy.

The raw Llama 7B implementation uses integer labels, avoiding a vocabulary-sized
one-hot target. That changes the operation graph before fusion and should remain
fixed during a kernel comparison.

Current JAX-AITER exposes RMSNorm and SiLU-and-multiply FFI kernels. The three case
recipes do not form a controlled comparison for them: the Llama 70B MXFP4 arm
explicitly disables AITER RMSNorm, and the pinned MaxText feature branch leaves
the AITER SiLU-and-multiply option off by default. A kernel's presence in a package
is not evidence that the case used it.

To prove an XLA fusion, inspect the fusion body and verify that the intermediate
has no external user. To prove an FFI fusion, identify the custom call and the
single corresponding device launch.

## MoE Kernel Preview

Chapter 8 develops the routing and communication model. The kernel choices needed
for the Mixtral case are:

- **Fixed-capacity one-hot:** the BF16 baseline. Tokens are placed into fixed
  expert capacity before dense expert products.
- **Dense masked:** dropless BF16 execution that evaluates masked dense work.
- **Dense padded:** tokens are routed, each expert problem is padded, and regular
  dense dots are issued.
- **Ragged GroupedGEMM:** `jax.lax.ragged_dot` is rewritten to one hipBLASLt
  grouped-GEMM route when eligible.

The controlled sparse pair is FP16 in both arms. **[source]** The pinned XLA
hipBLASLt grouped-GEMM integration supports FP16 on gfx950 but not BF16, and was
described upstream as not yet tuned. The general hipBLASLt datatype table showing
BF16 GEMM support does not override this narrower XLA grouped-GEMM restriction.
Comparing the FP16 sparse arms with the BF16 baseline confounds expert algorithm
and dtype; only the two FP16 sparse arms isolate the grouped-versus-padded route.

The relevant grouped-GEMM request is:

```bash
--xla_gpu_enable_cublaslt=true
--xla_gpu_experimental_use_ragged_dot_grouped_gemm=true
```

The first name is retained for compatibility on ROCm. The HLO must still contain
the grouped custom call, and the trace must show a grouped hipBLASLt kernel. The
separate ragged AllToAll flags select token movement, not the expert GEMM.

The MaxText/JAX-AITER feature tree also contains a fused MXFP4 MoE path, but its
own configuration describes it as forward-only. It is not a training fallback for
this Mixtral case.

## Restrictions That Change the Route

### Gradient

- Require a VJP and prove dQ/dK/dV or dInput/dWeight kernels separately.
- A forward-only MoE or GEMM route is excluded from training.
- Gradient atomics and conversion modes can alter determinism and numeric error.

### Dtype and accumulation

- Record operand, output, scale, gradient, and accumulator dtypes separately.
- BF16 support for ordinary GEMM does not imply BF16 support for grouped GEMM.
- A BF16 model with FP8 or MXFP4 projections is a mixed route, not an all-FP8 or
  all-FP4 train step.

### Shape and layout

- Local sharded shapes determine eligibility.
- Transposes, padding, head dimension, packed sequences, and group counts can
  select another kernel.
- A correct global shape does not prove that each device received a supported
  local problem.

### Workspace and rematerialization

- Workspace is live device memory and belongs in the peak-memory ledger.
- Deterministic and nondeterministic backward routes can require different
  temporaries.
- Rematerialization can execute a fused forward kernel again during backward.
  Count that work when explaining the trace.

## Correctness and Kernel Proof

Use this order for every new kernel route:

1. **Freeze the operation contract.** Save shapes, layouts, dtypes, mask, dropout,
   sharding, remat policy, seed, and package commits.
2. **Build an independent reference.** Use FP32 or the broad XLA route on a small
   problem. Do not use the candidate kernel as its own oracle.
3. **Check forward values.** Report absolute and relative error, NaNs/Infs, and
   the tolerance chosen for the dtype.
4. **Check gradients.** Compare every differentiated input or parameter. A finite
   loss alone does not validate a backward kernel.
5. **Run a multi-step smoke test.** Confirm finite loss and stable updates from
   identical initial state and data.
6. **Inspect optimized HLO.** Prove the intended custom call or fusion and save
   `.debug_options`.
7. **Trace one warmed step.** Prove forward and backward kernel names with
   `rocprofv3 --kernel-trace`.
8. **Measure memory.** Save XLA `memory_analysis()` and allocator high-water marks,
   including workspaces.
9. **Time in a clean process.** Do not use profiler or PMC timings. Report
   tokens/s/GPU first.
10. **Apply the quality guardrail.** For a low-precision training change, use the
    convergence protocol in Chapter 4 before making a final recommendation.

An HLO custom call without a trace is incomplete proof: the library can dispatch
internally. A kernel name without HLO is also incomplete: it does not show which
model operation or sharded shape invoked it.

## Fallback Order

Fallback means the least invasive route that satisfies the same mathematical
contract, not the route with the most optimistic peak FLOP/s.

1. Use the already-proven case route at the same dtype and layout.
2. Use the plain JAX/XLA implementation if its intermediates fit.
3. Use a different fused library backend only after forward and gradient checks.
4. Use direct JAX-AITER or Pallas-Triton only with a pinned build and full kernel
   proof.
5. For MoE, fall back from ragged GroupedGEMM to the matched dense-padded route,
   then to fixed-capacity or dense-masked execution as memory permits.
6. Change dtype, packing, or model semantics only as a new experiment arm. Do not
   call it a backend fallback.

Private monkeypatches, unsupported XLA flags, and forward-only kernels are
bring-up tools. They do not outrank a slower route with a complete training proof.

## Versioned Reachability

Status on 2026-09-13:

- **Dense FP32/FP16/BF16 via XLA:** source paths exist in the pinned common image;
  execution remains unverified under Appendix F. Exact rocBLAS, hipBLASLt, Triton,
  or emitter selection is shape-specific.
- **Transformer Engine CK/AITER attention:** requested by the pinned Llama 7B BF16
  causal route. Source comments record historical kernel attribution, but the trace
  needed to mark it available is absent.
- **Direct JAX-AITER attention:** wired by the Llama 7B alpha-2 setup with forward
  and backward FFI libraries. The repository contains no completed timing table.
- **Tokamax Pallas-Triton attention:** wired through a private compatibility patch
  for Tokamax 0.0.12–0.0.14. Treat as experimental.
- **TE FP8 delayed scaling:** requested by the Llama 70B MaxText recipe; a
  kernel-proof artifact is not present.
- **TE MXFP8:** requires the pinned gfx950 workspace patch. The repository has only
  an unpatched slow-fallback timing, not a post-patch result.
- **JAX-AITER MXFP4 projections:** requested by the pinned MaxText feature branch
  and alpha-2 FFI build. The case keeps the attention core in BF16; a
  per-projection kernel-proof artifact is not present.
- **hipBLASLt ragged GroupedGEMM:** the Mixtral source requests FP16 because BF16
  support is reported absent for the pinned route. Both the dtype restriction and
  executed kernel remain unverified without v26.6 compile and trace artifacts.
- **AITER fused MXFP4 MoE:** forward-only in the inspected feature tree; excluded
  from training.

Revalidate this list whenever JAX, the ROCm plugin/PJRT, XLA, MaxText,
Transformer Engine, JAX-AITER, Tokamax, or ROCm changes.

**Recommendation status: BLOCKED.** XLA attention is the correctness fallback.
TE/CK, direct JAX-AITER, and Tokamax are candidate fused-attention arms; dense
padded and fixed-capacity execution are candidate MoE fallbacks. No candidate
outranks another without forward/backward correctness, optimized HLO, kernel
trace, workspace, memory, and tokens/s/GPU artifacts for the exact shape.

## Primary References

- [XLA:GPU architecture](https://openxla.org/xla/gpu_architecture)
- [XLA:GPU emitters](https://openxla.org/xla/emitters)
- [From HLO to thunks](https://openxla.org/xla/hlo_to_thunks)
- [XLA FFI custom calls](https://openxla.org/xla/custom_call)
- [ROCm JAX installation and compatibility](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html)
- [hipBLASLt datatype support](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html)
- [hipBLASLt grouped GEMM API](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/ext-reference.html)
- [XLA gfx950 grouped-GEMM restriction](https://github.com/openxla/xla/commit/ae3a4afa4f3247377423405adc6eb6da8e0baa83)
- [Transformer Engine JAX API](https://docs.nvidia.com/deeplearning/transformer-engine/api/jax.html)
- [JAX-AITER](https://github.com/ROCm/jax-aiter)
- [Tokamax](https://github.com/openxla/tokamax)

<h3 markdown=1 class="next-section">Next: [Mixture of Experts]({{ '/pages/8-mixture-of-experts-on-mi355x' | relative_url }}).</h3>
