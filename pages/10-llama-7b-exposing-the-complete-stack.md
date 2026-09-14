---
layout: distill
title: "Llama 7B: Exposing the Complete Stack"
description: "A fixed Llama 2 7B training step implemented in raw JAX and MaxText, then varied across precision, attention, rematerialization, and FSDP."
date: 2026-09-13

section_number: 10

previous_section_url: "/pages/9-compiler-runtime-and-rccl-controls"
previous_section_name: "Chapter 9: Compiler, Runtime, and RCCL Controls"

next_section_url: "/pages/11-llama-2-70b-mixed-precision-training"
next_section_name: "Chapter 11: Llama 70B"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "Question and Evidence State"
  - name: "Frozen Experiment Manifest"
  - name: "Predict Before Running"
    subsections:
      - name: "Model Ledger"
      - name: "Training FLOPs"
      - name: "Persistent State and Activations"
  - name: "The Raw-JAX Implementation"
  - name: "Stage 1: Raw JAX and MaxText"
  - name: "One Complete Profiling Walkthrough"
  - name: "Stage 2: Attention Backends"
  - name: "Stage 3: Rematerialization on One GPU"
  - name: "Stage 4: Rematerialization under FSDP-8"
  - name: "Reconcile Prediction and Measurement"
  - name: "Negative Results and Traps"
  - name: "Pre-run Defaults"
  - name: "Reproduce the Matrix"
  - name: "Missing Artifacts"
---

## Question and Evidence State

This case study asks how much performance is lost or gained as one fixed Llama 2
7B training step moves between implementations, precision modes, attention
backends, rematerialization policies, and one versus eight GPUs. Tokens/s/GPU is
the primary metric. Step time, analytic TFLOP/s, peak HBM, compiled memory, HLO,
and kernel traces explain that metric.

The experiment code is complete, but the current repository does not contain the
timing, memory, HLO, or profile outputs for this matrix. This draft therefore
separates source-backed facts from analytical predictions and leaves narrow
blocked result slots. Older runs in repository history use a different manifest
and must not be copied into the current tables.

Evidence labels in this chapter mean:

- **[source]** The current experiment configuration or implementation establishes
  the claim.
- **[analytical]** The value follows from the frozen dimensions or a stated
  formula.
- **[measured]** A captured artifact supports the value. No current cell has this
  label yet.
- **[cited]** A named external source supports the claim.

**BLOCKED** means the named artifact is absent; it is a status, not an evidence
label.

This is a throughput study. It uses one synthetic batch repeatedly, so its falling
training loss is neither a convergence result nor evidence about model quality.
Real data, checkpointing, evaluation, convergence, and multi-node execution are
out of scope.

## Frozen Experiment Manifest

The comparison is valid only if every arm resolves to this manifest.

| Field | Frozen value | Evidence or status |
|---|---|---|
| Accelerator | AMD Instinct MI355X, `gfx950` | [source] experiment plan |
| Node | Eight MI355X devices; stages 1–3 expose device 0, stage 4 exposes all eight | [source] launch scripts |
| Partition mode | Must be recorded for the run | **BLOCKED:** no current hardware manifest |
| Container | `docker.io/rocm/jax-training:maxtext-v26.6` | [source] README; digest not captured |
| JAX stack | JAX/JAXLIB and ROCm plugin/PJRT 0.11.0 | [source] plan and setup script |
| MaxText | Stock checkout at `MAXTEXT_ROOT`, default `/workspace/maxtext` | **BLOCKED:** commit and import origin not captured |
| Transformer Engine | Version supplied by the container | **BLOCKED:** package/build identity not captured |
| JAX-AITER | `release/v0.1.0-alpha2` at `35b7175c763153ddb5da50c47d33dec436d5f191` | [source] setup script |
| AITER submodule | `31350226161346314b3d8882c8085bd31dce6a34` | [source] setup script |
| Tokamax | 0.0.14 | [source] setup script |
| Model source | Current `llama7b` commit `5f996a88f2cd4f3c1ffe969458483f04cbcaa2e6` | [source] repository |
| Sequence length | 4096 | [source] both configs |
| Per-device batch | 4 sequences | [source] both configs |
| Tokens per GPU per step | \(4\times4096=16{,}384\) | [analytical] |
| Steps | 30 | [source] both configs |
| Data | One reused synthetic batch | [source] raw JAX and MaxText configs |
| Optimizer | AdamW, \(b_1=0.9\), \(b_2=0.95\), \(\epsilon=10^{-8}\), weight decay 0.1 | [source] both configs |
| Schedule | \(3\times10^{-4}\), 1% warmup, cosine decay to 10%, 100,000 schedule steps | [source] both configs |
| Gradient clipping | Global norm 1.0 | [source] both configs |
| Parameter and moment dtype | FP32 | [source] both implementations |
| Layer representation | Scanned 32-layer stack | [source] both configs |
| Checkpointing | Disabled | [source] MaxText config; absent from raw JAX |
| JAX compilation cache | Disabled | [source] raw config and launch contract |
| Allocator fraction | 0.9 | [source] all launch scripts |

The shared XLA flag file contains:

```text
--xla_gpu_autotune_level=4
--xla_gpu_enable_triton_gemm=true
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_all_gather_combine_threshold_bytes=8589934592
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
--xla_gpu_all_reduce_combine_threshold_bytes=8589934592
--xla_gpu_enable_command_buffer=
```

Each launcher removes ambient `XLA_FLAGS` and reconstructs the value from this
file. This prevents inherited container flags from making two arms silently
different.

For publication, save the container digest, package list, MaxText commit and
import path, partition mode, topology, config hashes, flag hash, and complete
command beside every result. A tag and a prose claim that the machine was an
MI355X are not a sufficient manifest.

### Measurement rule

The raw-JAX loop runs 30 steps, compiles before the loop, and synchronizes each
step with `jax.block_until_ready`. Use steps 10–29 for its steady-state median
and report the minimum, maximum, and median absolute deviation. Step 0 is
first-dispatch warmup, not compile time.

The MaxText launchers delegate to the checkout at `MAXTEXT_ROOT`. Treat the same
warmup and synchronization policy as an acceptance requirement until the pinned
MaxText loop and logs prove it. Do not merge raw-JAX and MaxText samples under one
protocol assumption.

For any arm with median step time \(t\):

\[
\text{tokens/s/GPU}=\frac{16{,}384}{t}.
\]

The FSDP-8 runs keep the local batch at four, so this formula remains unchanged.
Their global batch is 32 and their global tokens per step are 131,072.

## Predict Before Running

### Model Ledger

**[source]** The raw implementation matches MaxText's Llama 2 7B dimensions:

- 32 decoder layers
- model width 4096
- 32 query and 32 key/value heads
- head dimension 128
- SwiGLU hidden width 11,008
- vocabulary 32,000
- RMSNorm epsilon \(10^{-5}\)
- RoPE maximum timescale 10,000
- separate token embedding and output projection

For \(E=4096\), \(M=11008\), \(L=32\), and \(V=32000\), one layer contains:

\[
\begin{aligned}
P_{\text{attention}} &= 4E^2 = 67{,}108{,}864,\\
P_{\text{MLP}} &= 3EM = 135{,}266{,}304,\\
P_{\text{norm}} &= 2E = 8{,}192.
\end{aligned}
\]

The complete model is:

\[
P=L(P_{\text{attention}}+P_{\text{MLP}}+P_{\text{norm}})
  +2VE+E
  =6{,}738{,}415{,}616.
\]

The two implementations must log this exact parameter count. A mismatch stops the
comparison before timing.

### Training FLOPs

The raw implementation reproduces MaxText's analytical training count term for
term. For batch \(B\), sequence \(T\), model width \(E\), MLP width \(M\), head
count \(H\), head dimension \(D\), and layer count \(L\):

\[
\begin{aligned}
F_{\text{FFN}} &= 6BTEM,\\
F_{\text{QKV}} &= 2BTE(3H)D,\\
F_{\text{out}} &= 2BTEHD,\\
F_{\text{vocab}} &= 2BTEV,\\
F_{\text{weights,train}} &=
  3\left[L(F_{\text{FFN}}+F_{\text{QKV}}+F_{\text{out}})
  +F_{\text{vocab}}\right],\\
F_{\text{attention,train}} &=
  3L\left(\frac{4BT^2HD}{2}\right).
\end{aligned}
\]

The factor of three approximates one forward pass plus two backward passes. The
division by two applies the causal-attention convention. Embedding lookup is not
charged, while the output projection is.

At \(B=4\) and \(T=4096\):

- learnable-weight work is 649.50 TFLOPs per GPU per step;
- attention work is 52.78 TFLOPs;
- total analytical work is 702.28 TFLOPs;
- 92.48% is assigned to weights and 7.52% to attention.

These are model FLOPs, not executed hardware FLOPs. Full rematerialization
executes extra work without changing this numerator, so MFU comparisons across
remat policies need that qualification.

The idealized compute-only floor is:

\[
t_{\min}=\frac{702.28\ \text{TFLOPs}}{P_{\text{effective}}}.
\]

Using the MI355X dense BF16 data-sheet ceiling of 2516.6 TFLOP/s gives 0.279
seconds and 58,712 tokens/s/GPU. This is a ceiling, not a throughput prediction:
non-matrix work, HBM traffic, attention implementation, optimizer work, launch
overhead, and sustained clock all lower the achieved value. FP32 requires its own
dtype-appropriate peak.

### Persistent State and Activations

With \(P=6{,}738{,}415{,}616\), one FP32 copy occupies:

\[
\frac{4P}{2^{30}}=25.10\ \text{GiB}.
\]

The returned training state contains:

- FP32 parameters: 25.10 GiB;
- FP32 Adam first moment: 25.10 GiB;
- FP32 Adam second moment: 25.10 GiB;
- total persistent returned state: 75.31 GiB.

An FP32 gradient tree is another 25.10 GiB while live. A complete BF16 copy of the
weights would be 12.55 GiB, but the compiler may cast and schedule smaller pieces
rather than retain one full copy. Donation permits parameter and optimizer-state
buffers to be reused. For those reasons, adding every item gives a useful pressure
check but not a valid peak-memory result.

Two activation calculations set expectations:

\[
\text{attention scores}=BHT^2\times\text{bytes per element}.
\]

At this shape, one score tensor is 4.00 GiB in BF16 or 8.00 GiB in FP32. More
than one score-related value may be live during backward. The XLA path can
materialize this \(T^2\) state; flash-attention paths are designed not to retain
the complete matrix.

The loss intentionally converts logits to FP32:

\[
4\times4096\times32000\times4\ \text{bytes}=1.95\ \text{GiB}.
\]

This is a large activation even after attention is fused. Integer labels avoid a
second 1.95 GiB one-hot target.

For FSDP-8, ideal full sharding would reduce the 75.31 GiB returned state to about
9.41 GiB per GPU and an FP32 gradient tree to about 3.14 GiB per GPU. This is a
lower-bound state ledger. Replicated leaves, gathered parameters, activations,
collective buffers, executable temporaries, and allocator behavior must be
measured.

The memory ordering prediction is firm:

\[
\text{full remat} < \text{minimal\_with\_context} < \text{none}
\]

for saved activations. The speed ordering is not firm. Recomputing fewer values
usually helps, but retaining more activations can increase HBM traffic and
interfere with scheduling.

## The Raw-JAX Implementation

The raw path is deliberately a complete training step without a training
framework. It is small enough to trace from Python to kernels while retaining the
parts that change performance.

**[source] Model**

- The training-only Flax model is adapted from the Hugging Face Flax Llama
  implementation.
- `DenseGeneral` layouts are chosen to match MaxText's projections.
- RMSNorm computes its variance in FP32.
- The final logits and cross-entropy path use FP32.
- Layers are represented with `nn.scan`, so XLA sees one loop body instead of 32
  unrolled copies.
- Named checkpoints identify Q, K, V, context, output, and MLP projections for
  selective rematerialization.
- Attention dispatch is explicit: `xla`, `te`, `aiter`, or `triton`.

**[source] Optimizer and data**

- Optax applies global-norm clipping followed by AdamW.
- Adam's first moment is forced to FP32; the second moment follows the FP32
  parameter state.
- One random token batch is generated once and reused for all 30 steps.
- Inputs and next-token labels have shape \([4,4096]\).

**[source] Compilation and timing**

The complete loss, backward pass, optimizer update, and parameter update are
compiled ahead of the loop:

```python
compiled = jax.jit(
    step_fn,
    donate_argnums=(0, 1),
).lower(params, opt_state, example).compile()
```

Donation matters because an update without aliasing can temporarily retain old
and new copies of a state already tens of GiB in size. Every timed call then uses
`jax.block_until_ready`; otherwise Python enqueue latency would be reported as
GPU step time.

The harness logs:

- parameter count;
- allocator use after parameter and optimizer initialization;
- analytical TFLOPs;
- compiled output, temporary, argument, alias, and host-temporary memory;
- step time, TFLOP/s/GPU, tokens/s/GPU, and loss;
- allocator use and peak after training.

The memory logger labels binary GiB as `GB`. Preserve the raw field for parser
compatibility, but label it GiB in analysis.

## Stage 1: Raw JAX and MaxText

Stage 1 isolates implementation and compute dtype. All four arms use one GPU, XLA
dot-product attention, `minimal_with_context` remat, sequence 4096, batch 4, FP32
master weights, FP32 gradients/moments, the same optimizer, and the same XLA
flags.

| Implementation | Compute dtype | Tokens/s/GPU | Median step | Compiled memory | Allocator peak |
|---|---|---:|---:|---:|---:|
| Raw JAX | FP32 | — | — | — | — |
| MaxText | FP32 | — | — | — | — |
| Raw JAX | BF16 | — | — | — | — |
| MaxText | BF16 | — | — | — | — |

> **BLOCKED — Stage 1 results:** the four 30-step logs and memory reports are
> absent. Fill this table only from runs produced by
> `scripts/precision/{jax_fp32,jax_bf16,maxtext_fp32,maxtext_bf16}.py` under the
> frozen manifest.

The predictions are:

1. BF16 should exceed FP32 throughput because its large projections can use the
   higher-throughput matrix path. FP32 state is unchanged, so its memory benefit
   is concentrated in compute-side values and activations.
2. Raw JAX and MaxText should be close only if optimized HLO selects comparable
   layouts, fusions, and GEMM implementations. Equal Python-level dimensions do
   not prove equal executables.
3. Any large framework gap with attention fixed to XLA should first be checked
   for layout, fusion, optimizer, or effective-config differences. It should not
   be attributed to framework overhead without a trace.

The acceptance checks are:

- exact parameter count: 6,738,415,616;
- exact analytical work: 702.28 TFLOPs per GPU per step;
- finite losses for all 30 steps;
- no recompilation after the ahead-of-time compile;
- optimized HLO and buffer assignment saved for each arm;
- identical effective flags and allocator fraction;
- tokens/s/GPU computed from the same steady-state step range.

Raw and MaxText loss values need not match exactly because each implementation
generates its own synthetic batch and initializes parameters through its own
framework path. This study compares execution, not learning curves.

## One Complete Profiling Walkthrough

Use the raw-JAX BF16, XLA-attention, `minimal_with_context` arm for the first
walkthrough. Run timing, HLO dumping, XProf, kernel tracing, and hardware-counter
collection in separate processes. A profiler run does not replace the unprofiled
timing result.

### 1. Establish the timing result

```bash
cd /home/clchong/work/llama7b
python3 scripts/precision/jax_bf16.py
```

Record the parameter count and analytical FLOPs first. Then compute the median of
steps 10–29 and derive tokens/s/GPU as \(16{,}384/t\). Save the compiled-memory
line and the allocator peak separately; they answer different questions.

> **BLOCKED — baseline:** the raw log from this command is absent.

### 2. Capture optimized HLO and buffer assignment

The standalone wrapper replaces ambient `XLA_FLAGS`, so use the raw entry point
when adding dump flags:

```bash
mkdir -p /tmp/llama7b/profile-hlo
FLAGS="$(awk '
  { sub(/#.*/, ""); if ($0 ~ /[^[:space:]]/) printf "%s ", $0 }
' configs/flags/rocm.txt)"

HIP_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=rocm \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
XLA_FLAGS="$FLAGS --xla_dump_to=/tmp/llama7b/profile-hlo --xla_dump_hlo_as_text" \
python3 model/train.py configs/jax.yml \
  base_output_directory=/tmp/llama7b/profile-hlo \
  dtype=bfloat16 attention=xla remat_policy=minimal_with_context
```

Inspect the `jit_train_step` module in this order:

1. entry parameter and result layouts;
2. the scanned decoder loop;
3. dot operations and selected custom calls;
4. attention score, mask, softmax, and value-product lowering;
5. remat clones in the backward body;
6. buffer assignment and peak live range.

The XLA attention arm should be treated as unfused until its optimized HLO and
kernel trace prove otherwise. Search terms alone are not proof of memory
behavior; use the buffer live range.

> **BLOCKED — HLO:** no current optimized HLO, buffer assignment, memory-usage
> report, or compiler debug options were captured.

### 3. Capture XProf

```bash
python3 scripts/profile/xprof.py
python3 -m tensorboard.main \
  --logdir=/tmp/llama7b/profile-xprof/trace \
  --port=6006
```

The script profiles steps 10–14. In XProf:

1. select one complete warmed-up step;
2. separate host activity from device execution;
3. check whether device work has idle gaps;
4. group the major GEMM, attention, normalization, loss, and optimizer regions;
5. use HLO attribution to connect the largest regions to the optimized module;
6. open Memory Viewer and compare its peak and largest allocations with the
   analytical state and activation ledger.

Write down the selected step, trace timestamps, device ID, top regions, idle
fraction, and memory peak. Do not report the profiled step as the throughput
result.

> **BLOCKED — XProf:** the `.xplane.pb` trace and an exported step/memory summary
> are absent.

### 4. Capture the kernel timeline

```bash
ROCPROF=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3

ROCPROF_SELECTED_STEP=10 \
$ROCPROF --kernel-trace -- \
  python3 scripts/precision/jax_bf16.py steps=12
```

`ROCPROF_SELECTED_STEP` pauses rocprof until the selected warmed-up step and
pauses it again afterwards. In the resulting database:

1. confirm the selected interval contains exactly one step;
2. rank kernels by total duration and call count;
3. identify GEMM and XLA-generated attention kernels;
4. compare gaps and concurrency with XProf;
5. connect kernel names to HLO thunks where the metadata permits;
6. record launch geometry for the dominant kernels.

XProf provides the framework and HLO context. `rocprofv3` provides the ROCm
runtime and kernel names. Neither view should be forced to answer the other's
question.

> **BLOCKED — kernel trace:** the selected-step `rocpd` database and top-kernel
> export are absent.

### 5. Collect hardware counters

Each non-comment line in `configs/pmc/rocm.txt` is a separate process:

```text
SQ_WAVES GRBM_GUI_ACTIVE SQ_BUSY_CYCLES SQ_INSTS_VALU_MFMA_MOPS_BF16
FETCH_SIZE
WRITE_SIZE
TCC_HIT_sum TCC_MISS_sum
```

For example:

```bash
ROCPROF_SELECTED_STEP=10 \
$ROCPROF --kernel-trace \
  --pmc "SQ_WAVES GRBM_GUI_ACTIVE SQ_BUSY_CYCLES SQ_INSTS_VALU_MFMA_MOPS_BF16" \
  -- python3 scripts/precision/jax_bf16.py steps=12
```

Repeat for the other three lines. Counter collection serializes dispatches, so
step times and overlap in these runs are invalid. Use the counters only to
answer a named question about a kernel or the selected step, such as MFMA
activity or cache traffic.

> **BLOCKED — PMC:** the four selected-step counter databases and joined
> kernel/counter summary are absent.

### 6. Reconcile the views

The walkthrough is complete only when one chain can be stated with evidence:

\[
\text{JAX region}\rightarrow
\text{HLO instruction or fusion}\rightarrow
\text{thunk/custom call}\rightarrow
\text{kernel}\rightarrow
\text{counter interpretation}.
\]

The final walkthrough should report:

- unprofiled tokens/s/GPU and median step time;
- compiled memory and runtime peak HBM;
- XProf's largest regions and idle gaps;
- rocprof's dominant kernels and launch geometry;
- one counter-backed statement about a named kernel;
- any mismatch between the analytical prediction and captured execution.

No such current chain exists yet, so this section defines the procedure rather
than presenting a profile result.

## Stage 2: Attention Backends

Stage 2 holds the raw-JAX model, BF16 compute, one GPU, batch, sequence, optimizer,
flags, and `minimal_with_context` remat fixed. Only the attention call changes.

| Arm | Python entry point | Expected lowering to verify | Tokens/s/GPU | Median step | Peak HBM |
|---|---|---|---:|---:|---:|
| XLA | `jax.nn.dot_product_attention(..., implementation="xla")` | XLA-generated dots, mask, softmax, and fusions | — | — | — |
| Transformer Engine | `DotProductAttention` | TE fused-attention custom call; CK selected by environment | — | — | — |
| JAX-AITER | `jax_aiter.mha.flash_attn_func` | Direct JAX FFI call into AITER forward and backward libraries | — | — | — |
| Tokamax | `tokamax.dot_product_attention(..., implementation="triton")` | Pallas-Triton forward and backward kernels | — | — | — |

> **BLOCKED — attention sweep:** all four timing logs, compiled-memory reports,
> allocator peaks, correctness comparisons, optimized HLO modules, and
> selected-step kernel traces are absent.

The XLA arm is the portability baseline. Its score state scales as \(T^2\). The
three flash-style arms should reduce score materialization, but lower memory does
not guarantee higher end-to-end throughput. Attention is only 7.52% of the
analytical FLOP count, and integration overhead or a slow backward kernel can
erase a faster forward pass.

The TE launcher sets:

```text
NVTE_FUSED_ATTN_CK=1
NVTE_FUSED_ATTN_AOTRITON=0
```

This requests the CK path. The kernel trace must still prove the selected route.
The raw model's source notes that prior TE runs reached AITER FMHA kernels, but
that observation is not a substitute for a current trace.

The direct JAX-AITER arm checks for five built libraries before launch:

- `libjax_aiter.so`
- `mha_fwd_ja.so`
- `mha_bwd_ja.so`
- `libmha_fwd.so`
- `libmha_bwd.so`

This makes a missing build explicit instead of falling through to another
backend.

Tokamax 0.0.14 assumes an NVIDIA-style compute capability in two guards. The raw
model overrides those guards for a `gfx*` device and supplies an explicit
`DotAlgorithmPreset`. This is a source-level compatibility workaround. It must be
recorded with the result and retested when Tokamax changes.

Before ranking the arms, compare forward outputs and gradients on a smaller
deterministic fixture, then check finite losses in the full run. Completion and a
plausible loss do not by themselves establish backend parity.

## Stage 3: Rematerialization on One GPU

Stage 3 fixes raw JAX, BF16, one GPU, and TE attention. The named policy determines
which layer intermediates survive the forward pass:

- `none`: save every saveable intermediate;
- `minimal_with_context`: save Q, K, V, attention context, attention output, and
  the three MLP projections; recompute cheaper surrounding work;
- `full`: save the layer input and recompute the layer body during backward.

| Policy | Tokens/s/GPU | Median step | Compiled memory | Allocator peak | Executed-work qualification |
|---|---:|---:|---:|---:|---|
| `none` | — | — | — | — | No deliberate layer recomputation |
| `minimal_with_context` | — | — | — | — | Selective recomputation |
| `full` | — | — | — | — | Full layer recomputation |

> **BLOCKED — one-GPU remat:** the three logs, compiled-memory analyses, allocator
> peaks, and traces are absent.

The decision should be made on the Pareto frontier:

1. reject any policy that does not fit with operating headroom;
2. among policies that fit, rank tokens/s/GPU;
3. if throughput is close, prefer more headroom for sequence or batch changes;
4. qualify MFU because the fixed 702.28-TFLOP numerator does not count
   recomputation.

`minimal_with_context` is the planned default, but it is not declared the winner
before the sweep.

## Stage 4: Rematerialization under FSDP-8

The final stage repeats the three policies in MaxText with
`ici_fsdp_parallelism=8` and `attention=cudnn_flash_te`. The scripts expose all
eight GPUs, keep the per-device batch at four, and therefore raise the global
batch from four to 32.

| Policy | Tokens/s/GPU | Median step | Per-GPU compiled memory | Per-GPU allocator peak | Collective share |
|---|---:|---:|---:|---:|---:|
| `none` | — | — | — | — | — |
| `minimal_with_context` | — | — | — | — | — |
| `full` | — | — | — | — | — |

> **BLOCKED — FSDP-8 remat:** the three eight-GPU logs, per-device memory reports,
> optimized HLO, collective replica groups, and RCCL traces are absent.

This is not a strong-scaling experiment: the global work increases by eight when
the device count increases by eight. Tokens/s/GPU remains comparable because the
local tokens per step remain 16,384. The difference from the one-GPU remat sweep
includes both implementation and sharding: stage 3 is raw JAX, while stage 4 is
MaxText. Do not attribute a cross-stage difference to FSDP alone.

Rematerialization can change an FSDP executable beyond activation storage. It can
move recomputed projections relative to AllGather and ReduceScatter, change the
lifetime of gathered parameters, and alter overlap. For each policy, verify:

- parameter and optimizer-state shardings;
- collective types, tensor sizes, and replica groups;
- gathered-parameter live ranges;
- whether backward recomputation changes collective ordering;
- exposed RCCL duration in the selected-step trace;
- per-device rather than node-summed memory.

## Reconcile Prediction and Measurement

Use the same sequence for every table rather than explaining a result from the
most convenient profiler view.

### 1. Confirm the workload

- parameter count is 6,738,415,616;
- model FLOPs are 702.28 TFLOPs per GPU per step;
- local tokens are 16,384;
- compute, weight, gradient, and moment dtypes match the arm;
- attention, remat, and mesh values match the table label;
- the effective XLA flags match the fixed file.

### 2. Reconcile memory

Start from 75.31 GiB of returned FP32 state, a possible 25.10 GiB gradient tree,
compute-side casts, 1.95 GiB FP32 logits, attention state, and saved layer
activations. Then compare:

1. XLA's compiled output, temporary, argument, and alias sizes;
2. the buffer-assignment peak live range;
3. XProf Memory Viewer;
4. the allocator's runtime high-water mark.

Do not force these numbers to agree exactly. Static planning, trace capture, and
allocator reservation measure different boundaries. Explain the difference.

### 3. Reconcile throughput

Convert every median to tokens/s/GPU first. Then use:

\[
\text{analytic TFLOP/s/GPU}=\frac{702.28}{t}
\]

as a diagnostic. State the peak and clock convention before reporting MFU.
For remat, repeat that the numerator excludes recomputation.

If an attention arm improves its attention region by a fraction \(s\), a simple
Amdahl bound is:

\[
\text{whole-step speedup}\leq
\frac{1}{(1-f_{\text{attn}})+f_{\text{attn}}/s},
\]

where \(f_{\text{attn}}\) must come from the measured baseline trace. The 7.52%
analytical FLOP share is not a measured time share.

### 4. Reconcile raw JAX and MaxText

The implementations intentionally agree on architecture, analytical FLOPs,
optimizer hyperparameters, synthetic-data intent, and stage-1 attention/remat.
They can still differ in:

- scanned parameter layout: raw JAX stacks layers on axis 0, while MaxText uses
  its `param_scan_axis` layout;
- initialization and generated token values;
- embedding implementation (`use_iota_embed` is enabled in MaxText);
- fusion boundaries and layout assignment;
- optimizer lowering and donation/alias decisions;
- resolved defaults inherited from MaxText's `base.yml`.

Save the fully resolved MaxText config. The small experiment YAML alone does not
show every effective value.

## Negative Results and Traps

These are source-backed constraints, invalid interpretations, or historical
observations. Items without retained runtime artifacts remain unverified.

1. **Raw `implementation="cudnn"` is not the ROCm fused-attention route.**
   The model comments record a historical `cuDNN is not detected` failure and
   XLA-generated fallback. Retest and retain the exception and HLO before treating
   this as a current compatibility result.
2. **A Python API name does not prove a kernel.** TE, direct AITER, and Tokamax
   each need optimized HLO and a kernel trace. A successful import is not path
   verification.
3. **Tokamax needs a gfx950 compatibility workaround in this source revision.**
   The result must record that monkey patch and cannot be generalized to an
   unmodified Tokamax release.
4. **PMC time is invalid.** `rocprofv3 --pmc` serializes dispatches. Use a
   separate timing process.
5. **TE can conflict with rocprof's ROCm preload during PMC collection.** The raw
   model contains a `ROCPROF_SKIP_TE_ROCM_PRELOAD=1` compatibility path based on
   a historical observation. Retest it; if used, record it as part of the profiler
   environment.
6. **Step 0 is not compile time.** The raw harness compiles before entering the
   loop. First-step excess is warmup and first dispatch.
7. **A warm persistent cache can invalidate a compiler comparison.** The config
   leaves `jax_cache_dir` empty and the launchers rebuild `XLA_FLAGS`.
8. **The repeated synthetic batch makes the loss collapse.** This is expected
   memorization and cannot support a convergence or quality claim.
9. **FSDP-8 does not pool eight memories into one JAX allocation.** State is
   explicitly sharded and parameters may be gathered; per-GPU capacity and
   collective traffic still matter.
10. **The FSDP-8 stage is weak scaling.** Global batch changes from 4 to 32. It
    does not measure speedup for one fixed global batch.
11. **Model MFU is not hardware FLOP execution.** The fixed numerator excludes
    remat work and optimizer details.
12. **An older result cohort is not a missing cell filler.** Historical runs in
    Git used another container and experiment contract. Keep them separate from
    this v26.6 matrix.

## Pre-run Defaults

The source and analytical ledger support a provisional recipe, not a measured
winner:

- Use BF16 compute with FP32 parameters and Adam moments as the normal throughput
  baseline. Keep FP32 as a diagnostic and equivalence arm.
- Keep `minimal_with_context` as the starting remat policy. Move to `full` when
  capacity requires it; use `none` only if measured throughput improves and peak
  HBM leaves enough headroom.
- Use XLA attention as the portable baseline and path-debugging reference.
- Test TE/CK first as the fused-attention candidate because it is packaged with
  the MaxText route. Promote it only after output/gradient checks and a current
  kernel trace.
- Treat direct JAX-AITER and Tokamax as explicit alternatives, not automatic
  fallbacks. Require correctness, forward and backward kernel proof, memory, and
  tokens/s/GPU.
- Use raw JAX to explain lowering and isolate kernels. Use MaxText for the defined
  FSDP-8 experiment and configuration workflow.
- Choose the FSDP remat policy from its own eight-GPU Pareto frontier. Do not copy
  the one-GPU winner.

No backend or remat policy can be named fastest until the blocked artifacts are
captured.

## Reproduce the Matrix

Run these commands inside
`docker.io/rocm/jax-training:maxtext-v26.6` on an eight-MI355X node.

### Setup

```bash
cd /home/clchong/work/llama7b

export MAXTEXT_ROOT="${MAXTEXT_ROOT:-/workspace/maxtext}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/llama7b}"
export JAX_AITER_ROOT="${JAX_AITER_ROOT:-/workspace/jax-aiter-alpha2}"

bash scripts/setup/setup_aiter.sh

python3 - <<'PY'
import importlib.metadata as m
import jax

print("jax", jax.__version__)
for package in (
    "jaxlib",
    "jax-rocm7-plugin",
    "jax-rocm7-pjrt",
    "transformer-engine",
    "tokamax",
):
    try:
        print(package, m.version(package))
    except m.PackageNotFoundError:
        print(package, "NOT INSTALLED")
print("devices", jax.devices())
PY

git -C "$MAXTEXT_ROOT" rev-parse HEAD
git -C "$JAX_AITER_ROOT" rev-parse HEAD
git -C "$JAX_AITER_ROOT/third_party/aiter" rev-parse HEAD
```

Also capture the container digest, `rocm-smi --showproductname
--showcomputepartition --showmemorypartition`, `rocminfo`, the topology, `pip
freeze`, config hashes, and flag hash in the result directory.

### Raw JAX versus MaxText

```bash
python3 scripts/precision/jax_fp32.py
python3 scripts/precision/jax_bf16.py
python3 scripts/precision/maxtext_fp32.py
python3 scripts/precision/maxtext_bf16.py
```

### Attention

```bash
python3 scripts/attention/xla.py
python3 scripts/attention/te.py
python3 scripts/attention/aiter.py
python3 scripts/attention/triton.py
```

### One-GPU rematerialization

```bash
python3 scripts/remat/none_1gpu.py
python3 scripts/remat/minimal_1gpu.py
python3 scripts/remat/full_1gpu.py
```

### FSDP-8 rematerialization

```bash
python3 scripts/remat/none_fsdp8.py
python3 scripts/remat/minimal_fsdp8.py
python3 scripts/remat/full_fsdp8.py
```

### Profiling

```bash
python3 scripts/profile/xprof.py

ROCPROF=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3
ROCPROF_SELECTED_STEP=10 \
$ROCPROF --kernel-trace -- \
  python3 scripts/precision/jax_bf16.py steps=12

ROCPROF_SELECTED_STEP=10 \
$ROCPROF --kernel-trace --pmc "SQ_WAVES GRBM_GUI_ACTIVE" -- \
  python3 scripts/precision/jax_bf16.py steps=12
```

Run the remaining PMC groups in separate processes as described in the profiling
walkthrough.

## Missing Artifacts

The following artifacts are required to replace every blocked marker:

1. **Frozen manifest**
   - container image digest;
   - host and in-container ROCm versions;
   - `pip freeze`;
   - MaxText commit, clean-tree status, and imported module path;
   - Transformer Engine package/build version;
   - MI355X product, clock/power mode, SPX/NPS mode, visible-device mapping, and
     eight-GPU topology;
   - SHA-256 values for `configs/jax.yml`, `configs/maxtext.yml`, and
     `configs/flags/rocm.txt`.
2. **Stage 1**
   - complete 30-step logs for raw-JAX FP32, raw-JAX BF16, MaxText FP32, and
     MaxText BF16;
   - a machine-readable steady-state summary for steps 10–29;
   - compiled memory analysis and allocator peak for each arm;
   - optimized `jit_train_step` HLO and buffer assignment for each arm;
   - fully resolved MaxText config.
3. **Profiling walkthrough**
   - unprofiled raw-JAX BF16/XLA timing log;
   - optimized HLO, thunk metadata, buffer assignment, live range, compiler
     debug options, and autotune results;
   - XProf `.xplane.pb` plus exported selected-step and Memory Viewer summaries;
   - selected-step `rocprofv3` kernel-trace database plus top-kernel export;
   - four selected-step PMC databases and a kernel/counter join;
   - one written HLO-to-kernel-to-counter correlation.
4. **Attention sweep**
   - four 30-step logs and steady-state summaries;
   - four compiled-memory reports and runtime peaks;
   - deterministic forward-output and gradient comparison;
   - finite-loss record for every full-shape run;
   - optimized HLO/custom-call proof and selected-step kernel trace for every
     backend;
   - package/build identities for TE, JAX-AITER/AITER, and Tokamax.
5. **One-GPU remat**
   - three 30-step logs and steady-state summaries;
   - compiled memory, buffer live range, and allocator peak for each policy;
   - selected-step trace showing recomputation for each policy.
6. **FSDP-8 remat**
   - three eight-GPU 30-step logs and per-device summaries;
   - per-device compiled memory and allocator peaks;
   - optimized HLO with parameter shardings, collective tensor sizes, and
     replica groups;
   - selected-step RCCL trace with exposed collective duration;
   - device/topology mapping for the run.

Until these artifacts exist, the chapter can publish the implementation,
accounting, protocol, known constraints, and provisional recommendation, but not
a performance ranking.
