---
layout: distill
title: "Llama 2 70B: Mixed Precision Training"
description: "A throughput-first precision study of Llama 2 70B on eight MI355X GPUs, with FSDP memory accounting and a one-billion-token convergence guardrail."
date: 2026-09-13

section_number: 12

previous_section_url: "/pages/11-llama-7b-exposing-the-complete-stack"
previous_section_name: "Chapter 11: Llama 7B"

next_section_url: "/pages/13-mixtral-8x22b-sharding-meshes-and-moe-optimizations"
next_section_name: "Chapter 13: Mixtral 8x22B"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "Decision and Evidence"
  - name: "Frozen Experiment Manifest"
  - name: "Predicting FSDP Memory"
  - name: "Precision and Backend Paths"
  - name: "Train-Step Results"
  - name: "Analysis Boundaries"
  - name: "One-Billion-Token Guardrail"
  - name: "External Evidence"
  - name: "Reproducing the Study"
  - name: "Rerun Decision"
---

## Decision and Evidence

The current source does not support a default precision recommendation. The
README repeats historical point estimates in which FP8 was 1.77 times the BF16
baseline and an MXFP4 arm was 2.27 times BF16. Those values came from an archived
feature-branch cohort, not the current stock-path launchers, and most rows were
marked noncanonical. They are useful inputs to a rerun plan, not current
measurements.

The experiment owner also reports near-identical reduced-precision loss curves
through approximately one billion nominal token positions. The retained source
does not identify the completed arm-to-run mapping or include the metrics and
plot. This remains recorded project status until provenance is packaged. It does
not establish full-pretraining or downstream-task equivalence.

This chapter uses four evidence labels:

- **[source]** comes from checked-in code, configuration, or a manifest.
- **[measured]** requires a complete Appendix F artifact bundle.
- **[analytical]** is calculated from the fixed recipe or standard Llama 2 70B
  dimensions.
- **[cited]** comes from a named external source.

**BLOCKED** marks an absent artifact; it is not an evidence label.

The historical throughput values and convergence observation are **[source]**
records. Current Appendix F bundles are BLOCKED. These limits are carried through
the chapter rather than filled with estimates.

## Frozen Experiment Manifest

The train-step sweep changes precision while holding the model, mesh, batch, and
optimizer fixed.

| Field | Fixed value | Evidence status |
|---|---|---|
| Hardware | One node, 8× AMD Instinct MI355X (`gfx950`) | [source] plan; runtime inventory BLOCKED |
| Experiment repository | `clarkechong/llama70b-mixed-precision-training` at `f3dab3694da64afb87a74fcb20a4142577d668fe` | [source] |
| Container | `docker.io/rocm/jax-training:maxtext-v26.6` | [source] tag; image digest BLOCKED |
| Stock MaxText | `release/v26.6` at `b47d74bf`, under `/workspace/maxtext` | [source] |
| Model | MaxText `llama2-70b` | [source] |
| Mesh | FSDP-8; every other ICI and DCN axis is 1 | [source] |
| Sequence length | 4,096 tokens | [source] |
| Per-device batch | 15 sequences for all throughput-comparison arms | [source] |
| Global batch | 120 sequences, 491,520 tokens per step | [analytical] |
| Rematerialization | `remat_policy: full` | [source] |
| Layer representation | `scan_layers: true` | [source] |
| Attention | `cudnn_flash_te`, except FP32 | [source] request; selected kernel BLOCKED |
| Optimizer | AdamW, LR `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1` | [source] |
| Training state dtypes | FP32 weights, gradients, and optimizer state | [source] request |
| Train-step data | Synthetic, with one reused batch | [source] |
| Train-step length | 30 steps; first 10 intended for JIT warmup | [source] protocol; aggregation BLOCKED |
| XLA autotuning | Level 4; Triton GEMM enabled | [source] request |
| Collective combine thresholds | 8 GiB for AllGather, ReduceScatter, and AllReduce | [source] request |
| Allocator fraction | 0.97, except MXFP8 at 0.94 | [source]; MXFP8 is outside a one-variable comparison |

The common non-FP32 batch is

\[
15\ \text{sequences/GPU}\times 8\ \text{GPUs}\times
4096\ \text{tokens/sequence}=491{,}520\ \text{tokens/step}.
\]

The FP32 runner preserves the same global batch by using a per-device microbatch
of 1 and 15 gradient-accumulation steps. It also selects
`attention=dot_product` and `matmul_precision=highest`. This arm is useful as a
numeric reference, but it is not a controlled throughput comparison with the
five arms that share Transformer Engine attention.

### Precision is the intended independent variable

The common configuration fixes:

```yaml
weight_dtype: "float32"
grad_dtype: "float32"
mu_dtype: "float32"

ici_fsdp_parallelism: 8
max_target_length: 4096
per_device_batch_size: 15
remat_policy: "full"
scan_layers: true
```

Each runner changes `dtype`, `quantization`, and the environment or batch
exception required by that precision. MXFP8 also lowers
`XLA_PYTHON_CLIENT_MEM_FRACTION` from 0.97 to 0.94, so it is not a
one-variable timing comparison until that allocator setting is aligned. FP8 and MX formats
quantize selected matrix operands; they do not turn the complete training state
into an 8-bit or 4-bit checkpoint.

## Predicting FSDP Memory

The first fit calculation should use persistent training state, then add
activations and implementation overhead. It should not start from the format
used inside a GEMM.

### Model ledger

For the standard Llama 2 70B dimensions used by MaxText:

- layers \(L=80\)
- model width \(d=8192\)
- MLP width \(d_{\mathrm{ff}}=28672\)
- 64 query heads and 8 key/value heads, with head dimension 128
- vocabulary size 32,000
- separate input embedding and logits matrices

The bias-free parameter count is [analytical]:

\[
\begin{aligned}
P_{\mathrm{attn/layer}}
  &= 2d^2 + 2d(8\times128)
   = 150{,}994{,}944, \\
P_{\mathrm{mlp/layer}}
  &= 3d\,d_{\mathrm{ff}}
   = 704{,}643{,}072, \\
P
  &= 80(P_{\mathrm{attn/layer}}+P_{\mathrm{mlp/layer}}+2d) \\
  &\quad +2(32000d)+d \\
  &= 68{,}976{,}648{,}192.
\end{aligned}
\]

The resolved MaxText parameter count is not committed, so this ledger remains a
prediction. A publication bundle should compare it with the parameter tree
reported by the actual run.

### Persistent state under FSDP-8

The configuration requests FP32 parameters, gradients, and Adam state. Parameters
and two Adam moments persist between steps:

\[
\frac{P(4+4+4)}{8}=96.36\ \text{GiB/GPU}.
\]

One live FP32 gradient shard adds 32.12 GiB, bringing these terms to
128.48 GiB during the relevant part of the step.

If MaxText keeps a persistent BF16 working copy of the weights, add

\[
\frac{2P}{8}=16.06\ \text{GiB/GPU},
\]

for 112.42 GiB/GPU of persistent state, or 144.54 GiB with the live gradient
shard. The run's buffer assignment is needed to determine whether that cast copy
is persistent and to avoid double-counting aliased buffers.

FP8, MXFP8, and MXFP4 use `dtype=bfloat16` and change the quantization recipe at
the matrix operation. Their FP32 persistent state and live-gradient terms
therefore have the same leading memory cost as BF16. Quantized operands, scales, amax histories, packed
temporaries, and backend workspaces are additional terms. The repository has no
measurements for those terms.

### Activation proxy

With full rematerialization and scanned layers, a useful first proxy is one
saved layer input per layer. For the non-FP32 arms, all of which use a two-byte
activation dtype:

\[
80\times15\times4096\times8192\times2
=75.00\ \text{GiB/GPU}.
\]

This gives a pre-workspace range of 203.48 to 219.54 GiB/GPU, depending on
whether the BF16 weight copy is persistent.

For FP32, the microbatch of 1 reduces the same proxy to

\[
80\times1\times4096\times8192\times4
=10.00\ \text{GiB/GPU}.
\]

Adding the FP32 state floor gives 138.48 GiB/GPU before attention buffers,
collective buffers, accumulation temporaries, compiler workspaces, executable
state, and allocator effects.

These are [analytical] planning values, not peak-memory results. The train-step
timings show that the five reported comparison arms ran, but they do not reveal
their high-water marks. Per-arm XProf memory output or device-memory telemetry
is BLOCKED.

## Precision and Backend Paths

### FP32: stock MaxText with two exceptions

The FP32 arm requests:

```text
dtype=float32
quantization=
per_device_batch_size=1
gradient_accumulation_steps=15
matmul_precision=highest
attention=dot_product
```

The microbatch and accumulation preserve 491,520 tokens per optimizer step.
Dot-product attention is a different backend from the other arms, so a future
FP32 timing must be reported separately rather than used as the speedup
baseline. No FP32 timing, peak memory, profile, or loss is available.

### BF16 and FP16: stock paths

BF16 sets `dtype=bfloat16`; FP16 sets `dtype=float16`. Both leave
`quantization` empty and request MaxText's `cudnn_flash_te` attention. The field
name is retained for MaxText compatibility on ROCm. It requests the Transformer
Engine route, but the optimized HLO and kernel trace are required to prove the
selected attention and GEMM kernels.

BF16 is the throughput baseline because it is the common production precision
and shares the batch and attention request with FP16, FP8, MXFP8, and MXFP4.

### FP8: Transformer Engine delayed scaling

FP8 keeps `dtype=bfloat16` and sets:

```text
quantization=te_fp8_delayedscaling
```

MaxText therefore routes eligible dense operations through Transformer Engine's
delayed-scaling FP8 recipe while the rest of the step remains in its configured
precision. The measured speedup does not prove that every intended projection
used FP8. That requires HLO custom-call attribution and a kernel trace, both of
which are absent.

### MXFP8: patched Transformer Engine workspace path

MXFP8 also keeps `dtype=bfloat16`, then sets:

```text
quantization=te_mxfp8
NVTE_ROCM_ENABLE_MXFP8=1
```

The current runner refuses to start unless
`transformer_engine.jax.cpp_extensions.gemm._get_gemm_workspace_size` contains
the patched scale-workspace expression. This source probe is useful because an
unpatched run can execute through a slow fallback rather than fail clearly.

The repository's only MXFP8 timing, 29.938 seconds/step, is explicitly labelled
an **unpatched fallback**. It is 11.5% slower than BF16. This is a negative
result about that fallback path, not a performance result for MXFP8 hardware.
Without a kernel trace, this chapter does not name the fallback kernel or assign
its time to a particular operation.

There is also a provenance mismatch to resolve. The README links a
`fix/jax-gfx950-mxfp8-workspace` branch, while
`scripts/setup/install_te_mxfp8.sh` checks out
`experiment/v2.17-gfx950-mxfp8-workspace`. The setup script does not pin a
Transformer Engine commit. The source probe can reject the wrong implementation,
but it cannot identify the exact implementation that was measured.

Post-patch MXFP8 timing, memory, profile, and convergence provenance are
BLOCKED. No replacement number is inferred from an older branch or another
experiment.

### MXFP4: ROCm MaxText branch and JAX-AITER FFI

MXFP4 is not a stock v26.6 configuration path. Its setup pins:

- ROCm MaxText `feature/jax-aiter-mxfp4-v26.6` at
  `b437942a5f33704f8438deb948488ad08164285c`
- JAX-AITER `release/v0.1.0-alpha2` at
  `35b7175c763153ddb5da50c47d33dec436d5f191`
- JAX-AITER's AITER submodule at
  `31350226161346314b3d8882c8085bd31dce6a34`

The build must produce:

```text
libjax_aiter.so
gemm_fp4_ja.so
cast_mxfp4_ja.so
```

The runner adds both the MaxText feature branch and JAX-AITER to
`PYTHONPATH`, selects `quantization=aiter_fp4`, and enables the JAX-AITER path.
Its FP4 environment includes MLP and attention-projection coverage. The
Q/K/V/O projections, MLP projections, and logits projection use the MXFP4
path; `aiter_attention=false` keeps the fused attention core in BF16 through
the Transformer Engine configuration. `aiter_rmsnorm=false` likewise avoids
changing the RMSNorm backend.

Two settings are part of the tested path:

```text
AITER_FP4_ATTN=1
JA_FP4_PACK_GATEUP_AG=0
```

The first gives the FP4 arm projection scope comparable with the FP8 arm. The
second disables a path that currently double-shuffles packed Gate/Up weights.
The remaining `JA_FP4_*` controls select the tested Hadamard, stochastic
rounding, dgrad partition, reuse, and rematerialization behavior.

The README reports 11.692 seconds/step for the MXFP4 arm. The pinned branch
quantizes the MLP and Q/K/V/O projections; the logits projection remains
unquantized. This historical value applies to the branch-and-FFI stack, not
stock MaxText.
Optimized HLO and a kernel trace are still needed to publish operation-by-
operation proof of coverage.

## Train-Step Results

The README repeats prior v26.6 step times for 30-step synthetic runs. An audit
traced the exact values to `archive/v26.6-migration-20260908`. That cohort used
MaxText `b437942a` for every arm. Its index marks BF16, FP16, FP8, and MXFP8
noncanonical because result completion was not verified. The historical runner
did retain step JSON and some memory and profile artifacts, but they do not
belong to the current stock-launcher cohort.

For arms that retain the fixed 491,520-token global batch:

\[
\text{tokens/s/GPU}
=\frac{491{,}520}{8\times\text{seconds/step}}.
\]

| Arm | Reported seconds/step | Derived tokens/s/GPU | Speedup vs BF16 | Historical status |
|---|---:|---:|---:|---|
| BF16 | 26.486 | 2,320 | 1.00× | Feature branch; completion unverified |
| FP16 | 24.552 | 2,502 | 1.08× | Feature branch; completion unverified |
| FP8 delayed scaling | 14.950 | 4,110 | 1.77× | Feature branch; completion unverified |
| MXFP8 | 29.938 | 2,052 | 0.88× | Three fallbacks/layer; completion unverified |
| MXFP4 | 11.692 | 5,255 | 2.27× | Feature branch; retained historical row |
| FP32 | **BLOCKED** | **BLOCKED** | Not comparable | Different attention and accumulation |

The conversion is valid only for these fixed-batch train-step runs. It must not
be applied to evaluation steps, compilation, a convergence run with input
overhead, or a run whose effective global batch differs.

The historical source records:

- FP16 was 7.9% faster than BF16 in that cohort.
- FP8 was 77.2% faster than BF16 in that cohort.
- The unpatched MXFP8 fallback is 11.5% slower than BF16.
- The experimental MXFP4 row is 126.5% faster than the historical BF16 row.

These are **[source]** historical observations. They do not support a current
stock-path speed ranking or a precision recommendation. Current Appendix F
bundles remain BLOCKED.

## Analysis Boundaries

### MFU is blocked

MFU needs a declared model-FLOP convention, an accepted step time, and the
correct hardware denominator for each format. This repository supplies the
historical step-time record but not a current accepted cohort or per-arm
operation mix.
Using the BF16 peak for every arm would also make the cross-format comparison
misleading. No MFU value is reported here.

The missing artifact is a per-arm calculation that records:

1. the model FLOPs assigned to one optimizer step;
2. any attention, rematerialization, and accumulation corrections;
3. the fraction executed in each arithmetic format;
4. the MI355X peak used for each fraction.

### Peak memory is blocked

The FSDP ledger predicts the leading terms, but allocator telemetry or XLA
buffer assignment is needed for the peak. Historical branches contain memory
records for a different cohort; no accepted current HBM bundle exists. This
chapter therefore does not claim that FP8 or MX formats save memory. Under this
recipe their FP32 state and BF16 activation terms do not shrink; FP16 uses a
different 16-bit format with the same byte width.

### Amdahl attribution is blocked

If a fraction \(f\) of the BF16 step were accelerated by a factor \(r\), the
ideal whole-step speedup would be

\[
S=\frac{1}{(1-f)+f/r}.
\]

The measured speedups do not determine \(f\): format conversion, workspace
management, kernel selection, attention, collectives, optimizer work, and
fallbacks can all change between arms. A BF16 profile that attributes steady-
state time to quantized projections and unchanged work is required before
assigning an Amdahl ceiling. No non-GEMM percentage is invented.

### Profiles and kernel proof are blocked

The README gives a valid starting command:

```bash
NVTE_FRAMEWORK=jax \
/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3 \
  --kernel-trace -- python3 scripts/train_step/mxfp8.py
```

Historical MXFP4 traces and HLO exist on the archived branch, but they are not
part of the current cohort or an Appendix F bundle. Until current traces and
optimized HLO are accepted, the chapter distinguishes requested backend paths
from proved kernel paths. This is particularly important for MXFP8 and the
feature-branch MXFP4 FFI route.

## One-Billion-Token Guardrail

### Convergence recipe

The convergence launchers replace synthetic data with local C4 JSON shards and
hold the training order fixed:

- 10 C4 training shards and 2 validation shards by default
- one local Llama 2 tokenizer, checked against MaxText's reference
- data shuffle seed `20260823`
- packing enabled, with at most 32 segments per sequence
- 2,034 optimizer steps
- 491,520 token positions per step
- 5% warmup and cosine decay over the 2,034-step run
- evaluation from step 0, then every 100 steps
- 20 evaluation batches per evaluation
- checkpointing disabled

The nominal packed-sequence capacity is [analytical]:

\[
2034\times491{,}520=999{,}751{,}680\ \text{token positions}.
\]

Actual non-padding training tokens require retained segmentation or
`total_weights` metrics.

There are launchers for BF16, FP16, FP8, patched MXFP8, and MXFP4. FP32 has no
convergence launcher.

### Data and tokenizer provenance

`fetch_c4.py` writes `manifest.json` with every local shard's byte count and a
digest, plus a digest over the file inventory. Its per-file digest covers the
file size and the first and last 1 MiB rather than the complete file. It also
records the Hugging Face revision as mutable `main`. This identifies the local
campaign inputs but is not an immutable upstream snapshot.

`make_hf_tokenizer.py` downloads a Llama 2 tokenizer mirror, verifies token IDs
against MaxText's SentencePiece model on 5,000 C4 documents by default, checks
the vocabulary size, and writes SHA-256 hashes for the copied files and the
reference model.

The generated data manifest and tokenizer `PROVENANCE.json` are not committed,
so the exact inputs used by the completed runs remain BLOCKED.

### Result and qualification

The experiment owner reports near-identical reduced-precision loss curves over
approximately one billion nominal token positions. This is recorded project
status, not **[measured]** evidence: no retained artifact identifies which arms
completed or maps them to metric files and plotted curves.

That statement is deliberately narrower than "the formats converge
identically." The current evidence does not establish:

- numeric final validation losses or pairwise deltas;
- equivalence across multiple seeds;
- equivalence beyond the nominal 999,751,680 positions;
- full-pretraining or downstream-task quality;
- equivalence to FP32;
- time to a predeclared target loss;
- NaN/Inf counts, gradient-norm excursions, or other numeric stability values.

Once arm/run provenance is recovered, the curves can provide a descriptive
tested-horizon guardrail. They cannot yet support a per-arm recommendation, and
they do not justify extrapolation to another schedule, dataset, scale, or
training duration.

## External Evidence

This draft does not import throughput or convergence values from another
system. A useful external comparison would have to match the model revision,
sequence length, global batch, rematerialization policy, training-state dtypes,
quantized projection scope, and software pins. A peak-throughput claim for a
format or a result from a different model would not close any blocker in this
case study.

The format and backend descriptions above are tied to the declared MaxText,
Transformer Engine, and JAX-AITER source paths. General format specifications and
vendor documentation can explain why those paths exist, but the train-step
timings and convergence guardrail remain results of this experiment.

## Reproducing the Study

### 1. Use the recorded container and stock tree

Start from:

```text
docker.io/rocm/jax-training:maxtext-v26.6
```

Confirm the stock MaxText revision:

```bash
git -C /workspace/maxtext rev-parse HEAD
```

It should resolve to the `release/v26.6` pin beginning `b47d74bf`. Record the
container digest, ROCm/JAX versions, `rocminfo`, `rocm-smi`, partition mode, and
GPU topology; those fields are missing from the current result record.

The runners accept path overrides:

```text
MAXTEXT_ROOT
MAXTEXT_MXFP4_ROOT
JAX_AITER_ROOT
DATA_ROOT
OUTPUT_ROOT
```

### 2. Install the precision-specific backends

BF16, FP16, and FP8 use the stock environment. For MXFP8:

```bash
bash scripts/setup/install_te_mxfp8.sh
```

Before treating the run as reproducible, pin and record the exact Transformer
Engine commit and reconcile the branch-name mismatch described above.

For MXFP4:

```bash
bash scripts/setup/setup_mxfp4.sh
```

This checks out the recorded MaxText and JAX-AITER commits and builds the
required FFI libraries for `gfx950`.

### 3. Run the train-step arms

```bash
python3 scripts/train_step/bf16.py
python3 scripts/train_step/fp16.py
python3 scripts/train_step/fp8.py
python3 scripts/train_step/mxfp8.py
python3 scripts/train_step/mxfp4.py
python3 scripts/train_step/fp32.py
```

Extra arguments are forwarded to MaxText. For a publication run, retain every
post-warmup sample and state the synchronization and aggregation method. Do not
replace the common batch with a smaller one after an out-of-memory failure;
preserve the global batch with accumulation or report the result as a different
experiment.

### 4. Prepare convergence inputs

```bash
python3 scripts/setup/fetch_c4.py
python3 scripts/setup/make_hf_tokenizer.py
```

Archive:

```text
data/c4-en/manifest.json
data/tokenizer-llama2-hf/PROVENANCE.json
```

For a new campaign, replace the mutable C4 `main` revision with an immutable
revision and use complete-file hashes if strict byte-for-byte provenance is
required.

### 5. Run convergence arms

Run one launcher at a time:

```bash
python3 scripts/convergence/bf16.py
python3 scripts/convergence/fp16.py
python3 scripts/convergence/fp8.py
python3 scripts/convergence/mxfp8.py
python3 scripts/convergence/mxfp4.py
```

Each launcher writes its metrics under `OUTPUT_ROOT`, which defaults to
`/tmp/llama70b`. Copy the JSONL metrics and complete stdout/stderr log out of
`/tmp` before the container exits. Checkpointing is disabled, so these metrics
are the durable convergence artifact.

## Rerun Decision

| Arm | Historical source value | Guardrail status | Current-path status | Next action |
|---|---|---|---|---|
| FP32 | BLOCKED | Not run | Different attention and accumulation | Establish numeric reference |
| BF16 | 2,320 tokens/s/GPU | Arm mapping BLOCKED | Stock recipe unverified | Rerun as control |
| FP16 | 2,502 tokens/s/GPU | Arm mapping BLOCKED | Stock recipe unverified | Rerun after BF16 |
| FP8 delayed scaling | 4,110 tokens/s/GPU | Arm mapping BLOCKED | Stock TE recipe unverified | Candidate for controlled rerun |
| MXFP8 | 2,052 tokens/s/GPU, unpatched | Arm mapping BLOCKED | Patched TE path unverified | Reject fallback; measure pinned patch |
| MXFP4 | 5,255 tokens/s/GPU | Arm mapping BLOCKED | Experimental branch/FFI | Reproduce with kernel proof |

No precision arm is the current default on this evidence. BF16 is the control
recipe. FP8 and MXFP4 are candidates because the historical source values justify
the cost of a controlled rerun. Patched MXFP8 remains undecided.

The exact blockers are:

1. FP32 train-step timing and successful-run record.
2. A pinned Transformer Engine commit and post-patch MXFP8 timing.
3. Raw post-warmup samples, synchronization method, aggregation, and variance
   for every reported timing.
4. Per-arm peak HBM and XLA buffer-assignment output.
5. Optimized HLO and kernel traces proving FP8, MXFP8, and MXFP4 coverage and
   identifying the unpatched MXFP8 fallback.
6. A declared FLOP convention and format-weighted denominator for MFU.
7. A BF16 profile breakdown before any non-GEMM fraction or Amdahl ceiling is
   reported.
8. Convergence JSONL files, arm/run manifests, numeric loss and stability
   values, and the plotted curves.
9. The generated C4 manifest and tokenizer provenance, preferably with an
   immutable upstream data revision and full-file hashes.
10. A captured environment proving `RCCL_WARP_SPEED_AUTO=0`, or a complete rerun
    with that MI355X safety setting frozen across every arm.
