---
layout: distill
title: "Predicting One Training Step"
description: "A compact compute, HBM, communication, model-state, and batching ledger for MI355X training."
date: 2026-09-13

section_number: 3

previous_section_url: "/pages/2-software"
previous_section_name: "Chapter 2: What jax.jit Runs on ROCm"

next_section_url: "/pages/4-profiling"
next_section_name: "Chapter 4: Profiling"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "The Three Time Bounds"
  - name: "Transformer Ledger"
  - name: "Training-State Ledger"
  - name: "Batching Ledger"
  - name: "MoE Ledger"
  - name: "Metrics"
  - name: "Three Worked Ledgers"
  - name: "Prediction Worksheet"
---

This chapter supplies the arithmetic used by the experiments. It predicts which
resource should limit a step and checks whether the model can fit. It does not try to
replace the full derivations in the
[JAX Scaling Book roofline chapter](https://jax-ml.github.io/scaling-book/roofline/),
[Transformer chapter](https://jax-ml.github.io/scaling-book/transformers/), or
[training chapter](https://jax-ml.github.io/scaling-book/training/).

All numbers in this chapter are **[analytical]** unless marked otherwise. Decimal
units are used for bandwidth and capacity: `1 TB = 10^12 bytes`.

## The Three Time Bounds

For one operation or one training step, write three ideal times:

```text
t_compute = FLOPs / compute_rate
t_hbm     = bytes_moved_to_or_from_HBM / HBM_bandwidth
t_comms   = bytes_moved_over_links / achieved_link_bandwidth
```

Within this simplified resource model, complete overlap gives the largest term and
no overlap gives their sum:

```text
max(t_compute, t_hbm, t_comms)
    <= modeled resource-service time
    <= t_compute + t_hbm + t_comms
```

Actual elapsed time cannot beat the lower bound, but it can exceed the upper end of
this bracket. The model omits fixed costs, idle gaps, inefficient kernels, and
collective startup. Those costs matter most for small kernels and messages.

The inputs must describe the execution, not only the model:

- Compute FLOPs depend on rematerialization, padding, and the selected MoE
  implementation.
- HBM bytes are traffic, not allocated memory. Cache hits and fusion can remove HBM
  transfers; casts and temporary buffers can add them.
- Communication bytes depend on the collective algorithm, participant count, and
  local shard shape. Link specification bandwidth is only a ceiling.

### The MI355X BF16 ridge point

Chapter 1 gives a dense BF16 matrix peak of `2.5166 PFLOP/s` and peak HBM bandwidth
of `8 TB/s` for one MI355X. The hardware ridge point is therefore:

```text
2.5166e15 FLOP/s / 8e12 byte/s = 314.6 FLOP/byte
```

Use **315 FLOP/byte**, not 288. An operation below 315 FLOP/byte cannot reach the
dense BF16 compute ceiling if HBM supplies every counted byte.

Arithmetic intensity is:

```text
intensity = FLOPs / HBM bytes
roofline  = min(compute_rate, intensity * HBM_bandwidth)
```

For a projection `X[B_tok, D] @ W[D, F]`, if the weight dominates traffic and each
element occupies `w` bytes:

```text
intensity ~= 2 * B_tok / w
```

With BF16 operands, `w = 2`, so the approximate intensity is `B_tok` FLOP/byte.
This is a useful first check, not a claim about actual traffic. FP32 parameter
storage, a separate cast buffer, imperfect cache reuse, or small local shards all
change the result.

Communication has the same form. Divide useful compute by bytes sent over xGMI or
the network, then compare that intensity with `compute_rate / achieved_link_bandwidth`.
Use the bandwidth for the actual message size. Small tensor-parallel collectives do
not necessarily reach the asymptotic bandwidth of a large gradient reduction.

## Transformer Ledger

Use the following symbols:

- `L`: decoder layers
- `D`: model width
- `F`: SwiGLU hidden width
- `N`: query heads
- `K`: key/value heads
- `H`: head dimension
- `V`: vocabulary size
- `B`: sequences in the local microbatch
- `T`: sequence length

A Llama-style SwiGLU MLP has three matrices:

```text
P_mlp = 3 * D * F
```

The attention projections have one query, key, value, and output matrix:

```text
P_attn = D*N*H + 2*D*K*H + N*H*D
       = 2 * D * (N + K) * H
```

With two RMSNorm scales per layer, untied input and output embeddings, and one final
norm:

```text
P_dense = L * (P_mlp + P_attn + 2D) + 2VD + D
```

The input embedding is a gather. It contributes parameters and bytes but no matrix
multiply FLOPs. The output projection contributes `2BTDV` forward FLOPs.

For every weight matrix, the forward pass performs two FLOPs per parameter per
token. Backpropagation adds an input-gradient and a weight-gradient contraction of
the same size. The common training count is therefore three times the forward
projection count:

```text
training weight FLOPs ~= 6 * used_weight_parameters * tokens
```

Attention scores need a separate term because they have no learned weights and grow
with `T^2`. A full forward `QK^T` plus `AV` costs:

```text
4 * B * N * T^2 * H
```

The experiment repositories follow MaxText and charge half of that for causal
attention, then multiply by three for training. This convention assumes the
implementation avoids the masked half:

```text
causal attention training FLOPs = 6 * B * N * T^2 * H * L
```

State the convention with every MFU result. A kernel may issue a different number of
hardware operations, especially with rematerialization or padding.

## Training-State Ledger

Parameter count alone does not predict memory. Record each persistent and transient
tensor separately:

| Item | First estimate | Qualification |
|---|---:|---|
| Parameters | `P * bytes(weight_dtype)` | These may themselves be the master weights. Do not add a second master copy unless the implementation creates one. |
| Gradients | `P * bytes(grad_dtype)` | Usually live during backward and the update. |
| Adam first moment | `P * bytes(mu_dtype)` | MaxText exposes `mu_dtype`. |
| Adam second moment | `P * bytes(weight_dtype)` | In the inspected MaxText/Optax source, `nu` inherits the weight dtype. |
| Activations | policy- and shape-dependent | Attention backend, layer scanning, and rematerialization dominate. |
| Workspaces | backend-dependent | GEMM, attention, and grouped-GEMM libraries may reserve large buffers. |
| Collective buffers | sharding-dependent | FSDP gathers and MoE dispatch add temporary live ranges. |
| XLA temporaries | executable-dependent | Fusion, scheduling, and donation decide which buffers overlap. |

Treat the tensor sum as a feasibility estimate, not peak HBM. The check is the
compiled executable's memory analysis plus runtime peak allocation, covered in
Chapters 4 and 6.

Sharding applies per tensor. An `n`-way FSDP estimate may divide parameters,
gradients, and optimizer state by `n`, but gathered weights and some small leaves can
remain replicated. Expert parallelism divides expert state while leaving dense
layers replicated unless another axis shards them.

## Batching Ledger

The experiment unit is one optimizer update:

```text
tokens per GPU per update = microbatch_sequences_per_GPU
                            * sequence_length
                            * gradient_accumulation_steps

global tokens per update  = tokens per GPU per update * GPU_count
```

This convention matches the three experiment repositories. Keep these terms
distinct:

- A microbatch is the activation set that must fit at once.
- Gradient accumulation runs several microbatches before one update.
- Per-device tokens determine local matrix shapes and arithmetic intensity.
- Global tokens determine optimization behavior and the amount of training data
  consumed per update.

FSDP shards state and also uses the FSDP axis in the input batch layout in these
recipes. It does not make the global batch equal to the batch on one GPU.

## MoE Ledger

For `E` experts with `E_a` selected per token and expert hidden width `F_e`:

```text
total expert parameters     = E   * 3 * D * F_e
activated expert parameters = E_a * 3 * D * F_e
router parameters           = D * E
```

Persistent state follows the total parameter count. Useful expert FLOPs follow the
activated count. This split is why an MoE can be lighter in arithmetic than its HBM
footprint suggests.

Implementation details can invalidate the useful-FLOP estimate:

- Dense masked execution computes all `E` experts, a factor of `E/E_a` more expert
  work than the activated ledger.
- Fixed-capacity execution pads under-filled experts and may drop overflow tokens.
- Ragged grouped GEMM avoids padding but can lose matrix efficiency on small or
  uneven groups.

With `B_tok` input tokens per GPU, perfect balance sends:

```text
mean tokens per expert = B_tok * E_a / E
imbalance factor       = max(tokens_per_expert) / mean(tokens_per_expert)
```

The router histogram and dropped-token rate are part of the measurement contract.
A profile cannot reconstruct them after the run.

## Metrics

### Primary: tokens/s/GPU

```text
tokens/s/GPU = global tokens per optimizer update
               / elapsed seconds per update
               / GPU count
```

Use a fixed model, sequence length, global batch, accumulation count, and data
semantics when comparing two runs. Report useful input tokens. Do not count padded
tokens twice or silently exclude dropped MoE assignments.

### Diagnostics

- **Step time:** the synchronized, steady-state time for one optimizer update.
- **MFU:** useful model FLOPs divided by elapsed time and the dense peak of all
  participating GPUs. Use activated MoE FLOPs and state the attention convention.
- **HFU:** issued hardware FLOPs divided by peak. It includes rematerialized,
  padded, and dense-masked work, so it can exceed MFU substantially.
- **Peak HBM:** runtime high-water mark per GPU, compared with the static tensor
  ledger and XLA's buffer assignment.
- **Exposed communication:** the portion of step wall time during which a required
  collective runs without useful compute overlapping it. Summed kernel duration
  across eight GPUs is not wall time.
- **Compile and autotuning cost:** report separately from steady-state throughput,
  then state how many steps amortize it.
- **Validation loss:** a guardrail for precision or algorithm changes. The Llama
  70B experiment owner reports near-identical curves over approximately one
  billion nominal token positions, but the metrics, plot, and arm mapping are
  BLOCKED. Chapter 12 records the qualification.

Time-to-quality is useful only when a defensible quality target exists. It is not
the primary metric in this book.

## Three Worked Ledgers

The architecture values below come from the checked-in Llama 7B implementation and
the inspected MaxText model configs used by the Llama 70B and Mixtral experiments.

| Quantity | Llama 2 7B | Llama 2 70B | Mixtral 8x22B |
|---|---:|---:|---:|
| `L` | 32 | 80 | 56 |
| `D` | 4096 | 8192 | 6144 |
| `F` or `F_e` | 11008 | 28672 | 16384 |
| `N` / `K` / `H` | 32 / 32 / 128 | 64 / 8 / 128 | 48 / 8 / 128 |
| `V` | 32000 | 32000 | 32768 |
| Experts / active | dense | dense | 8 / 2 |
| Total parameters | 6.738B | 68.977B | 140.630B |
| Activated parameter ledger | 6.738B | 68.977B | 39.161B |
| Matrix-weight parameters charged by MFU | 6.607B | 68.713B | approximately 38.959B |

The activated ledger includes embeddings and norm scales, which contribute
parameters but not the same matrix FLOPs as dense projections. The MFU row keeps
only matrix weights charged by this chapter's FLOP convention: 6,607,077,376 for
Llama 7B, 68,713,185,280 for Llama 70B, and approximately 38,959,448,064 for
Mixtral. The Mixtral activated ledger is not the number of parameters stored on
one GPU.

### Work per optimizer update

| Quantity | Llama 2 7B | Llama 2 70B | Mixtral 8x22B |
|---|---:|---:|---:|
| GPUs | 1, or 8 for the FSDP sweep | 8 | 8 |
| Microbatch sequences/GPU | 4 | 15 | 4 |
| Sequence length | 4096 | 4096 | 4096 |
| Accumulation steps | 1 | 1 | 2 |
| Tokens/GPU/update | 16384 | 61440 | 32768 |
| Global tokens/update | 16384 on 1 GPU; 131072 on 8 | 491520 | 262144 |
| Model FLOPs/GPU/update | 702.279 TFLOPs | 26.320 PFLOPs | 7.937 PFLOPs |

The FLOP row uses the causal-attention convention above. It excludes optimizer
elementwise work, rematerialized forward work, padding, and communication.

The Llama 7B one-GPU and FSDP-8 arms keep four sequences per GPU, so the global
workload grows eightfold. This is weak scaling, not a fixed-global-batch speedup
test. A scaling-efficiency claim needs a separate FSDP-8 arm with the original
global batch of four sequences.

The Llama 70B FP32 arm also needs separate treatment. It uses microbatch 1 with
15 accumulation steps and dot-product attention, while the other arms use
microbatch 15, one step, and Transformer Engine attention. The nominal tokens and
model FLOPs per optimizer update are unchanged, but local matrix shapes, activation
memory, kernel launches, and attention cost are not.

If the synchronized update takes `s` seconds, the primary throughputs are:

```text
Llama 7B:        16384 / s tokens/s/GPU
Llama 70B:       61440 / s tokens/s/GPU
Mixtral 8x22B:   32768 / s tokens/s/GPU
```

### State before activations and workspaces

The two dense recipes use FP32 parameters and FP32 Adam moments:
`12 persistent bytes/parameter`. One live FP32 gradient tree adds another four
bytes per local parameter.

```text
Llama 7B persistent:       6.738B * 12 = 80.86 GB
Llama 7B live gradients:   6.738B *  4 = 26.95 GB
Llama 7B ideal FSDP-8 persistent:       10.11 GB/GPU
Llama 7B ideal FSDP-8 gradients:         3.37 GB/GPU

Llama 70B persistent:     68.977B * 12 = 827.72 GB
Llama 70B live gradients: 68.977B *  4 = 275.91 GB
Llama 70B ideal FSDP-8 persistent:      103.46 GB/GPU
Llama 70B ideal FSDP-8 gradients:        34.49 GB/GPU
```

The Mixtral recipe uses BF16 parameters and BF16 first and second Adam moments:
`6 persistent bytes/parameter`. Its `grad_dtype=float32` request does not upcast
BF16 parameter gradients in the inspected trainer, so use two bytes per live gradient
unless the compiled state proves otherwise.

```text
Mixtral global persistent state: 140.630B * 6 = 843.78 GB
```

For the EP-8 baseline, the architecture has 135.291B expert parameters and 5.339B
other parameters. If expert state is split eight ways and all non-expert state is
replicated, the structural estimate is:

```text
P_local = 135.291B / 8 + 5.339B = 22.25B
persistent = 22.25B * 6 = 133.50 GB/GPU
one live BF16 gradient tree = 22.25B * 2 = 44.50 GB/GPU
```

Together these two terms consume about 178 GB, leaving about 110 GB of the
advertised 288 GB capacity for activations, gathered buffers, workspaces, runtime
state, and headroom. The actual local shard shapes must be checked in the
compiled program.

> **BLOCKED (measurement):** the current repositories do not contain a common
> v26.6 peak-HBM artifact bundle for these three recipes. The values above are
> tensor ledgers, not measured peaks.

## Prediction Worksheet

Fill this in before each run:

```text
Hardware
  GPUs and partition mode:
  dense compute peak for the actual matrix dtype:
  HBM capacity and bandwidth:
  relevant link and measured message-size bandwidth:

Model
  L, D, F/F_e, N, K, H, V:
  E and E_a:
  total parameters:
  parameters used per token:

Batch
  sequence length:
  microbatch sequences/GPU:
  accumulation:
  tokens/GPU/update:
  global tokens/update:

Memory
  parameters:
  gradients:
  optimizer moments:
  activation estimate and remat policy:
  known workspaces and collective buffers:

Time bounds
  model FLOPs/update:
  expected issued-FLOP modifiers:
  HBM bytes:
  collective bytes by type:
  t_compute, t_hbm, t_comms:

Report
  median seconds/update:
  tokens/s/GPU:
  MFU convention:
  peak HBM:
  exposed communication:
  validation-loss guardrail:
```

Chapter 4 turns this prediction into a reproducible measurement and records the
artifacts needed to explain any gap.
