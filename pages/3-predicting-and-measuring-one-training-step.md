---
layout: distill
title: "Predicting and Measuring One Training Step"
description: "A unified workflow for predicting training-step costs, measuring throughput, and tracing gaps through HLO and ROCm profiling tools."
date: 2026-09-13

section_number: 3

previous_section_url: "/pages/2-lowering-jax-jit-on-rocm"
previous_section_name: "Chapter 2: Lowering jax.jit on ROCm"

next_section_url: "/pages/4-training-in-mixed-precision"
next_section_name: "Chapter 4: Training in Mixed Precision"

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
  - name: "Measurement Contract"
  - name: "Artifact Bundle"
  - name: "XProf"
  - name: "rocprofv3"
  - name: "rocprof-compute"
  - name: "From HLO to a Kernel"
  - name: "Multi-Level Rooflines"
  - name: "Triage Order"
  - name: "Reporting Template"
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
compiled executable's memory analysis plus runtime peak allocation, covered below
and in Chapter 5.

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
  BLOCKED. Chapter 11 records the qualification.

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

The measurement process below tests this prediction and records the artifacts needed
to explain any gap.

Profiling starts with one synchronized number: tokens/s/GPU for a fixed workload.
The tools in this chapter explain that number. They do not replace it.

Use four separate runs:

1. a clean timing run;
2. an XProf trace for framework and HLO attribution;
3. a `rocprofv3` trace for ROCm runtime, kernel, memory-copy, and RCCL events;
4. a counter run with `rocprofv3` or `rocprof-compute`.

Counter collection and tracing perturb execution. Never take the headline step time
from a counter run.

Tool semantics linked to JAX, XProf, or ROCm documentation are **[cited]**.
Commands and settings taken from the experiment repositories are **[source]**.
This chapter contains no **[measured]** MI355X profiler result. Archived gfx942
observations and untested gfx950 fields are identified as unverified where they
appear.

## Measurement Contract

### Freeze the workload

A comparison is valid only if these fields match:

- model configuration and parameter count;
- sequence length and packing behavior;
- microbatch per GPU and gradient accumulation;
- global tokens per optimizer update;
- optimizer and rematerialization policy;
- attention and MoE semantics;
- synthetic or real data;
- device count, mesh, and partition mode.

For precision experiments, keep the input tokens and optimizer recipe fixed. For
MoE experiments, also keep routing inputs fixed and report dropped or padded tokens.
If a configuration must change to fit, describe the comparison as a different
workload.

### Record the machine and software

Store the raw output of:

```bash
date -Iseconds
rocminfo
amd-smi static
python3 - <<'PY'
import jax, jaxlib
print("jax", jax.__version__)
print("jaxlib", jaxlib.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
PY
```

Also record:

- the full container tag and image digest;
- ROCm, JAX ROCm plugin, PJRT, RCCL, XProf, and profiler versions;
- `git rev-parse HEAD` for MaxText and every patched dependency;
- `git status --porcelain=v1` for each source tree;
- a retained patch plus `git diff --binary | sha256sum` when a tree is dirty;
- the experiment-repository commit;
- `HIP_VISIBLE_DEVICES`, `JAX_PLATFORMS`, `XLA_FLAGS`, memory-fraction settings,
  and profiler-specific variables;
- the compute and memory partition modes;
- power, clock, and temperature samples during the measured interval.

Do not dump the complete environment into a public artifact. It can contain tokens
and credentials. Save an allowlist of performance-related variables.

The three experiment repositories declare the mutable tag
`docker.io/rocm/jax-training:maxtext-v26.6`. The Llama 7B plan requests JAX,
JAXLIB, ROCm plugin, and PJRT 0.11.0. Record the image digest and what the
process actually imports rather than treating those declarations as pins.

### Separate compile, warmup, and measurement

The first execution can compile and autotune. Relevant XLA flags normally
participate in JAX's compilation-cache key. Reuse risk remains when a custom
implementation omits state from its key, a shared directory contains stale
artifacts, or a dependency changes without changing the recorded signature. State
whether the cache is cold, warm, or disabled and do not share one writable cache
across unlike experiment cohorts.

The checked-in experiments run 30 steps and reserve the first 10 for warmup. For
the clean timing run:

- launch each arm in a fresh process;
- discard steps 0 through 9;
- retain every later synchronized step time;
- report the median as the headline;
- report a dispersion measure such as p10–p90 or median absolute deviation;
- keep the raw samples.

Raw JAX timing must block:

```python
for _ in range(10):
    state, loss = train_step(state, batch)
    jax.block_until_ready((state, loss))

times = []
for _ in range(20):
    start = time.perf_counter()
    state, loss = train_step(state, batch)
    jax.block_until_ready((state, loss))
    times.append(time.perf_counter() - start)
```

Time one optimizer update. If gradient accumulation is two, the interval includes
both microbatches and the optimizer update.

### Keep timing and profiling independent

Use the same config, flags, data, and warmed step in every run, but do not expect
profiler wall time to equal clean-run wall time.

| Run | Use its time? | Purpose |
|---|---|---|
| Clean timing | Yes | Step time and tokens/s/GPU |
| XProf trace | No | Host/device timeline, HLO attribution, memory views |
| `rocprofv3` trace | No | Runtime calls, dispatches, copies, RCCL, kernel metadata |
| PMC or `rocprof-compute` | No | Counters, cache traffic, MFMA activity, empirical roofline |

Synthetic reused data isolates the compiled train step. It does not test the input
pipeline. Real data is required for convergence and end-to-end input throughput.
Label both.

## Artifact Bundle

One run should be inspectable without the original machine. Appendix F defines the
normative directory and schema. Its top level is:

```text
artifacts/<case_id>/<run_id>/
├── manifest.yaml
├── run-contract.yaml
├── checksums.sha256
├── config/
├── hardware/
├── data/
├── logs/
├── results/
├── ledgers/
├── components/
├── monitoring/
├── hlo/
├── profiles/
├── checkpoints/
└── fallback/
```

`manifest.yaml` identifies which files came from the clean timing run and which
came from instrumented runs. Its `omissions` list records missing artifacts and
reasons. Preserve raw profiler databases; CSV summaries are derived data.

The bundle does not need every directory for every experiment. A timing-only result
still needs the manifest, resolved config, flags, versions, raw steps, and summary.

## XProf

XProf reads the trace produced by JAX and retains the most useful automatic bridge
between JAX scopes, HLO operations, and GPU kernels.

### Capture

Warm up before opening the trace:

```python
for _ in range(10):
    state, loss = train_step(state, batch)
    jax.block_until_ready((state, loss))

jax.profiler.start_trace("/tmp/run/xprof")
for step in range(5):
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
        state, loss = train_step(state, batch)
        jax.block_until_ready((state, loss))
jax.profiler.stop_trace()
```

Five steps are usually enough to show repetition and overlap without producing an
unwieldy trace. The Llama 7B profiling script follows this pattern with 10 warmup
steps and a 5-step capture.

Open the result:

```bash
xprof --logdir=/tmp/run/xprof --port=6006
```

JAX normally writes an `.xplane.pb` plus a timeline-oriented
`.trace.json.gz`. Preserve the XPlane file. The JSON trace is convenient for
Perfetto but does not retain every relationship used by XProf.

### Read the data model

- **XSpace** is the complete capture.
- **XPlane** represents a host or device.
- **XLane** is one stream or framework lane within a plane.
- **XEvent** is an operation, runtime call, or kernel dispatch.
- **XStat** attaches metadata such as HLO op name, correlation ID, launch geometry,
  or kernel details.

Use the views in this order:

1. **Trace Viewer:** find idle gaps, synchronization, overlap, and repeated steps.
2. **Op Profile and HLO Stats:** identify scopes and HLO operations with the most
   self-time.
3. **Kernel Stats:** identify kernel families, durations, occurrences, and launch
   geometry.
4. **Memory Viewer/Profile:** find the peak interval and largest live buffers.
5. **Graph Viewer:** inspect producers, consumers, and fusion boundaries around one
   selected HLO op.

Add `jax.named_scope` around stable model components such as `attention`,
`mlp/up_proj`, and `optimizer`. A trace cannot reliably recover a source-level name
that the program never supplied.

### Validate fields on ROCm

Archived XProf 2.23 captures on gfx942 showed missing device step grouping, an
unsupported AMD compute ceiling in the roofline view, zeros for some occupancy and
register fields, and multi-device op times summed across devices. These observations
are not measurements of the current gfx950/v26.6 stack.

Treat an implausible zero as missing data until the XPlane or a ROCm profiler
corroborates it. Never convert summed device time directly into step wall time.

> **BLOCKED (current-tool audit):** the repositories do not yet contain a checked-in
> gfx950/v26.6 XPlane field audit. Revalidate the archived limitations before
> describing any one of them as current.

## rocprofv3

`rocprofv3` observes the ROCm runtime and device below XLA. It has richer ROCm
events and kernel metadata than XProf, but it does not automatically know the HLO
or Python source that caused a dispatch.

### Timeline capture

Wrap the unchanged experiment command:

```bash
rocprofv3 \
  --kernel-trace \
  --memory-copy-trace \
  --marker-trace \
  --rccl-trace \
  --stats \
  -f rocpd pftrace \
  -d /tmp/run/rocprof \
  -o train-step \
  -- python3 scripts/train_step/bf16.py
```

Use the exact `rocprofv3` path shipped in the container if it is not on `PATH`. The
experiment READMEs locate it under the ROCm SDK Python package.

- `rocpd` is the lossless SQLite form used for queries and later re-analysis.
- `pftrace` opens directly in Perfetto and is useful for visual inspection.
- `--runtime-trace` adds a broad runtime bundle when HIP/HSA launch behavior is the
  question.
- `--kernel-include-regex` narrows a follow-up capture to a known kernel family.

Example:

```bash
rocprofv3 --kernel-include-regex "Cijk_" --kernel-trace --stats \
  -d /tmp/run/rocprof-gemm -o gemm -- python3 train.py
```

List generated tables from SQLite's `sqlite_master`; rocprofiler table names can
carry a session-specific suffix.

### ROCTx ranges

Use ROCTx to mark one optimizer update or a small number of stable phases. Capture
them with `--marker-trace`. Match range names to `jax.named_scope` names where
possible, but remember that they are independent annotations. Do not assume XLA
automatically emits an ROCTx range for every HLO operation.

The raw Llama 7B runner demonstrates another useful technique: call
`roctxProfilerPause` before the loop, resume for one warmed step, then pause again.
This keeps initialization and unrelated steps out of a counter capture.

### Hardware counters

First inspect counters available on the installed version:

```bash
rocprofv3 --list-avail
```

Then collect one compatible group per run:

```bash
rocprofv3 --kernel-trace \
  --pmc "SQ_WAVES GRBM_GUI_ACTIVE SQ_BUSY_CYCLES SQ_INSTS_VALU_MFMA_MOPS_BF16" \
  -- python3 model/train.py configs/jax.yml
```

Counter names and legal groupings vary by architecture and profiler release. The
Llama 7B repository stores its intended groups in `configs/pmc/rocm.txt`.

PMC collection can serialize dispatches and may replay work. Its trace is evidence
about the selected kernels and counters, not overlap or step throughput.

## rocprof-compute

Use `rocprof-compute` after the trace has identified a small workload or kernel
family worth deeper analysis. It collects base counters, often through multiple
replays, then derives cache, occupancy, speed-of-light, and roofline metrics.

```bash
PROFILE_DIR=/tmp/rocprof-compute/llama7b-step
rocprof-compute profile --output-directory "$PROFILE_DIR" -- \
  python3 model/train.py configs/jax.yml

rocprof-compute analyze --path "$PROFILE_DIR"
```

Restrict the command to one warmed step when possible. A complete 30-step training
process multiplied by many counter passes is expensive and hard to interpret.

The workload directory contains raw pass output, `pmc_perf.csv`, and system
information. Preserve the complete directory. Use:

```bash
rocprof-compute analyze --path "$PROFILE_DIR" --experimental --gui
```

The GUI flag changed across releases. Check `rocprof-compute analyze --help` in
the recorded container; older versions may accept `--gui` without
`--experimental`.

for the interactive view, or select only the needed analysis blocks for a text
report. The useful outputs are:

- achieved FLOP rate and matrix-instruction activity;
- HBM and cache traffic;
- cache hit rates;
- wave and occupancy limits;
- LDS, register, and scratch pressure;
- empirical roofline position.

Do not compare the replayed profile's elapsed time with the clean timing run.

## From HLO to a Kernel

The reliable correlation starts at the framework and moves downward.

### 1. Name the source operation

```python
with jax.named_scope("mlp/up_proj"):
    hidden = inputs @ w_up
```

### 2. Dump the compiled program

Set dump flags before importing JAX:

```bash
export XLA_FLAGS="$XLA_FLAGS \
  --xla_dump_to=/tmp/run/hlo \
  --xla_dump_hlo_as_text"
python3 train.py
```

Keep the optimized HLO, debug options, and buffer assignment. The effective
`.debug_options` file is evidence that the intended flags reached the module.

### 3. Select the expensive event in XProf

From Kernel Stats, record:

- complete kernel name;
- HLO/op name;
- duration and occurrence count;
- device and stream;
- grid and workgroup shape;
- the exact captured step.

Search the optimized HLO for the op name. Read the operation's shape, operands,
layout, custom-call target, backend config, and fusion body.

On ROCm, some XLA compatibility strings retain CUDA-oriented names. A custom-call
target containing `cublas` is not proof that an NVIDIA library ran. The loaded code
object, kernel family, and ROCm trace settle the backend path.

### 4. Follow source metadata

When present, HLO metadata carries an `op_name` and `stack_frame_id`. Follow the
stack-frame ID through the dump's `StackFrames`, `FileLocations`, `FunctionNames`,
and `FileNames` tables to the Python line.

A fusion's metadata may name only one contributing operation. Inspect the
computation referenced by `calls=` before assigning the whole fusion to that name.
An activation may also disappear into a library GEMM epilogue, in which case the
backend config is the evidence.

### 5. Hand the kernel to ROCm tools

Re-run the same warmed step under `rocprofv3`, filtered by the kernel family. Confirm
the shape, scope, and selected kernel again: autotuning or cache state can choose a
different kernel in a fresh process.

The correlation chain is:

```text
Python scope
  -> HLO metadata and stack frame
  -> XProf kernel event
  -> ROCm kernel dispatch
  -> rocprof-compute counters and roofline
```

XProf supplies the automatic HLO bridge. ROCTx supplies manually chosen phase
boundaries in the ROCm trace. They complement each other.

## Multi-Level Rooflines

One HBM roofline cannot distinguish cache reuse from HBM traffic. For each memory
level `m`, compute a separate intensity:

```text
I_m = FLOPs / bytes_crossing_level_m
P_m = min(compute_peak, I_m * bandwidth_m)
```

Useful levels on MI355X are registers/LDS, L1, per-XCD L2, Infinity Cache, HBM, xGMI,
and the scale-out network. Do not reuse one byte count at every level. Each boundary
needs its own counters or a clearly labelled analytical lower bound.

Use three scopes:

1. **Kernel:** counters can place one dispatch against cache and HBM ceilings.
2. **HLO or named component:** aggregate only kernels that implement that component,
   and account for fusion.
3. **Training step:** compare useful model FLOPs with wall time, peak HBM, and
   exposed communication. Preserve overlap rather than summing all device-kernel
   durations.

The theoretical BF16/HBM line uses the constants calculated above:

```text
compute peak = 2.5166 PFLOP/s
HBM peak     = 8 TB/s
ridge point  = 315 FLOP/byte
```

An empirical roofline substitutes sustained compute and measured bandwidth from the
same machine state. It must not quietly mix a boost-clock compute ceiling with a
replay-derived bandwidth.

> **BLOCKED — MI355X cache-level rooflines.** Add cache-level bandwidth ceilings
> only with a checked-in `rocprof-compute` bundle. No current experiment
> repository contains those measurements.

## Triage Order

Work down this list and stop when the gap is explained.

1. **Validate the comparison.** Confirm tokens/update, model config, data semantics,
   device count, precision, remat, and synchronization.
2. **Check device occupancy over time.** Long gaps between device kernels while the
   host is active indicate input or host starvation.
3. **Check compilation and recompilation.** Separate compile time, inspect cache
   state, and look for repeated compilation or autotuning.
4. **Verify sharding.** Read local shapes and HLO replica groups. Unexpected
   AllGather, AllReduce, or AllToAll operations can dominate a step while every
   individual kernel remains healthy.
5. **Verify the backend path.** Confirm that GEMM, attention, and MoE operations
   reached the intended library or FFI kernel. Look for casts and slow fallbacks.
6. **Check HBM pressure.** Compare arithmetic intensity and measured HBM traffic
   with the 315 FLOP/byte BF16 ridge. Inspect fusion and temporary traffic.
7. **Check communication.** Compare message bytes with bandwidth at that message
   size, then measure the exposed portion. A long collective that is fully hidden
   is not the first target.
8. **Check MFMA efficiency.** For the dominant GEMMs, compare shape, tile count,
   MFMA activity, LDS/register limits, and achieved FLOP rate. Low occupancy alone
   is not proof of a problem.
9. **Check numerical behavior.** Compare loss and finite-value checks with the
   reference. Faster invalid steps do not count.

This order keeps kernel-level investigation behind cheaper checks for invalid
workloads, idle devices, wrong shardings, and fallbacks.

## Reporting Template

Every case-study result should include this block or an equivalent machine-readable
manifest:

```text
Result:
  evidence: [analytical] | [measured] | [cited]
  claim:

Workload:
  model and parameter count:
  sequence length and packing:
  microbatch/GPU:
  gradient accumulation:
  global tokens/update:
  data: synthetic-reused | real
  optimizer, remat, attention, MoE:

System:
  GPU count and partition mode:
  topology:
  container and digest:
  ROCm, JAX, plugin/PJRT, RCCL, XProf:
  MaxText and dependency commits:
  XLA_FLAGS and relevant environment:

Protocol:
  fresh process:
  cache state:
  warmup steps:
  measured steps:
  synchronization:
  statistic and dispersion:
  clock/power handling:

Primary:
  median seconds/update:
  tokens/s/GPU:

Diagnostics:
  model-FLOP convention and MFU:
  measured or estimated peak HBM:
  dominant HLO scopes/kernels:
  collective time and exposed fraction:
  compile/autotuning time:
  validation-loss guardrail:

Artifacts:
  bundle path:
  missing files and reason:

Interpretation:
  predicted bound:
  observed bound:
  gap explained by:
  next controlled change:
```

References:

- [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html)
- [XProf](https://github.com/openxla/xprof)
- [ROCm Systems Profiler and rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/)
- [JAX Scaling Book: Rooflines](https://jax-ml.github.io/scaling-book/roofline/)
