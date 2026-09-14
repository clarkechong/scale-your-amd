---
layout: distill
title: "Mixtral 8x22B: Topology Meets Sparse Kernels"
description: "A one-node MI355X case study of FSDP and expert parallelism, MoE routing, ragged collectives, GroupedGEMM, and latency hiding."
date: 2026-09-13

section_number: 13

previous_section_url: "/pages/12-llama70b"
previous_section_name: "Chapter 12: Llama 70B"

next_section_url: "/pages/14-multinode"
next_section_name: "Chapter 14: Multi-Node Training"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Evidence Status
  - name: The Fixed Recipe
  - name: Model and Routing Ledger
  - name: The Baseline
  - name: Analytical Predictions
  - name: FSDP by EP Sweep
  - name: Verify the Mesh in HLO
  - name: Expert Execution Sweep
  - name: Ragged AllToAll Controls
  - name: Latency Hiding
  - name: Metrics and Diagnostics
  - name: Expected Versus Achieved
  - name: Predeclared Fallback Plan
  - name: Reproduce the Study
  - name: Publication Blockers
---

> **Draft evidence status.** Eight timing-arm launchers exist, but HLO capture,
> router instrumentation, and `maxtext-v26.6` result bundles do not. Every
> achieved-result field is therefore marked **BLOCKED**. The analytical ledgers
> are predictions, not substitutes for a run.

**Depends on:** [Chapter 3]({{ '/pages/3-cost-model' | relative_url }}) for the
training-step ledger, [Chapter 4]({{ '/pages/4-profiling' | relative_url }}) for the
measurement contract, [Chapter 7]({{ '/pages/7-sharding' | relative_url }}) for HLO
collectives, [Chapter 8]({{ '/pages/8-kernels' | relative_url }}) for kernel
attribution, and [Chapter 9]({{ '/pages/9-moe' | relative_url }}) for MoE routing.

The decision in this case study is narrower than “what is the fastest Mixtral
configuration?” The eight runs answer three controlled questions:

1. On eight MI355X GPUs, how should the fixed mesh product of eight be divided between
   FSDP and expert parallelism?
2. For one fixed mesh, which source-defined expert implementation gives the best useful
   throughput without hiding token drops?
3. Does the latency-hiding scheduler earn the extra live memory it can require?

The primary metric is tokens/s/GPU at a fixed workload. Step time, useful MFU, peak
memory, routing balance, and exposed collective time explain that result. Thirty
synthetic steps do not measure convergence.

## Evidence Status

This chapter uses four evidence labels:

- **[source]** comes from checked-in model, configuration, or launcher code.
- **[analytical]** is a calculation from stated inputs.
- **[measured]** requires a retained `maxtext-v26.6` artifact from an eight-MI355X run.
- **[cited]** comes from a named external source.
- **BLOCKED** identifies the exact result and artifact still missing.

The source tree defines eight timing-arm launchers and states explicitly that no
v26.6 timings have been recorded. HLO capture and router instrumentation still need
implementation. Older timings, recollections, and results from another container do
not fill these slots.

The current Mixtral page stub described the baseline as possibly having duplicated
weights. That is not the current sharding rule. `FSDP=1` means there is no active FSDP
axis, but `EP=8` is still a real sharding axis. Expert weights are sharded on their
expert dimension, and the default MaxText rule also maps the general `embed` weight
axis across the active `expert` mesh axis. HLO remains the final authority on the
physical layout.

## The Fixed Recipe

All eight cells request the following values unless an exception is stated:

| Item | Fixed value |
|---|---|
| Hardware | One node, 8x MI355X |
| Runtime | `docker.io/rocm/jax-training:maxtext-v26.6` |
| Model | MaxText `mixtral-8x22b` |
| Layers | 56 |
| Width | 6,144 |
| Attention | 48 query heads, 8 KV heads, head dimension 128 |
| Experts | 8 experts in every decoder layer |
| Routing | Top 2 experts per token |
| Expert width | 16,384 |
| Sequence length | 4,096 |
| Microbatch | 4 sequences per device |
| Gradient accumulation | 2 microsteps |
| Global batch | 64 sequences, or 262,144 tokens per update |
| Data | Synthetic, one batch reused |
| Run length | 30 updates |
| Attention backend | Transformer Engine fused-attention request |
| Partitioning | Shardy request |
| Layer storage | Scanned layers |
| Rematerialization | `save_dot_with_context_except_mlp` |
| Optimizer | AdamW |
| Checkpointing | Disabled |

The main recipe uses BF16 weights and compute and BF16 Adam moments. Although the
config requests `grad_dtype: float32`, the pinned trainer does not upcast BF16
parameter gradients; record them as BF16 unless HLO or runtime state proves
otherwise. The two `jax.lax.ragged_dot` cells change the compute dtype to FP16
because the experiment plan reports that the targeted gfx950 hipBLASLt
GroupedGEMM route does not support BF16. That restriction is unverified until a
BF16 compile/path test is retained. The checked-in `weight_dtype` remains BF16;
the sparse implementation casts operands to the configured compute dtype before
the ragged dot.

The global batch follows directly from the local recipe:

$$
4\ \text{sequences/device}
\times 8\ \text{devices}
\times 2\ \text{microsteps}
=64\ \text{sequences/update}.
$$

At length 4,096, that is 262,144 input tokens and 524,288 top-2 expert assignments per
update.

Values intentionally excluded from the study are real-data input performance,
checkpoint overhead, convergence, precision selection, tensor parallelism, pipeline
parallelism, and multi-node scaling. The next chapter owns the multi-node boundary.

## Model and Routing Ledger

The model config supplies the dimensions used below:

```yaml
base_emb_dim: 6144
base_num_query_heads: 48
base_num_kv_heads: 8
head_dim: 128
base_moe_mlp_dim: 16384
base_num_decoder_layers: 56
num_experts: 8
num_experts_per_tok: 2
vocab_size: 32768
logits_via_embedding: false
```

### Parameter ledger

For one decoder layer:

- Query projection: $6{,}144 \times 6{,}144 = 37{,}748{,}736$
  parameters.
- Key projection: $6{,}144 \times 1{,}024 = 6{,}291{,}456$.
- Value projection: another $6{,}291{,}456$.
- Output projection: $6{,}144 \times 6{,}144 = 37{,}748{,}736$.
- Attention projections in total: 88,080,384.
- Router: $6{,}144 \times 8 = 49{,}152$.
- One SwiGLU expert:
  $3 \times 6{,}144 \times 16{,}384 = 301{,}989{,}888$.
- Eight experts: 2,415,919,104.
- Two RMSNorm vectors: 12,288.
- Total per decoder layer: 2,504,060,928.

The input embedding and separate output projection each contain
$32{,}768 \times 6{,}144 = 201{,}326{,}592$ parameters. Including the final norm, the
model has **140,630,071,296 parameters [analytical]**, or about 140.63 billion. The
eight experts account for 135.29 billion of them.

Only two experts run for each token in a sparse top-2 implementation. Counting two
experts, attention, router, norms, embedding, and output weights gives about
**39.16 billion activated parameters per token [analytical]**. This is a useful model
size, not a memory size: all 140.63 billion parameters and their optimizer state must
still be stored.

### Routing ledger

The balanced top-2 load for one update is:

$$
\frac{262{,}144\ \text{tokens} \times 2}{8\ \text{experts}}
=65{,}536\ \text{assignments/expert}.
$$

The fixed-capacity path computes capacity separately for each sequence. At sequence
length 4,096 and `capacity_factor=1.0`, each expert receives 1,024 slots per sequence:

$$
\left\lceil \frac{4{,}096 \times 2}{8} \right\rceil \times 1.0
=1{,}024.
$$

This capacity equals the balanced mean. It does **not** guarantee that all assignments
fit. An expert receiving more than 1,024 assignments from one sequence drops the
overflow, even if another expert has empty slots. The experiment also sets
`load_balance_loss_weight=0.0`, so no auxiliary loss pushes the router toward equal
loads during these synthetic steps.

That distinction affects every comparison:

- the fixed-capacity one-hot path has predictable tensor shapes but may drop
  assignments;
- dense masked execution is dropless because it evaluates every expert for every
  token;
- the sparse ragged paths are dropless because `ragged_buffer_factor=-1.0` selects a
  worst-case receive buffer.

For one sparse EP-8 microstep, each device handles 16,384 input tokens and 32,768
top-2 assignments. MaxText's worst-case factor is
`min(EP, experts / top_k) = min(8, 4) = 4`, so one microstep's receive buffer can
contain 131,072 rows. The two sequential accumulation microsteps process up to
262,144 rows cumulatively. At width 6,144 and two bytes per FP16 element, one live
activation buffer is exactly **1.5 GiB per device [analytical]**. Three GiB is the
cumulative capacity processed across the update, not simultaneous storage. This is a
buffer-size bound, not a peak-memory prediction; buffer aliasing and simultaneous
lifetimes come from XLA's memory analysis.

## The Baseline

The baseline is:

```text
compute dtype     bfloat16
FSDP              1
EP                8
sparse_matmul     false
capacity_factor   1.0
one-hot dispatch  enabled by the fixed-capacity dense path
latency hiding    on
```

The complete overrides in `scripts/baseline.py` also disable Megablox, ragged sort,
ring-of-experts, and Tokamax GMM. `moe_dispatch_no_expert_sharding=true` peels the
expert axis from the batch dimension inside the fixed-capacity MoE dispatch. The
expert dimension remains sharded. This is intended to produce expert-parallel token
movement instead of an FSDP-style AllGather/ReduceScatter fallback.

“FSDP-1, EP-8” therefore does not mean that every GPU stores a full copy of all
weights. In the intended layout:

- each device stores one eighth of the expert dimension;
- MoE weight input dimensions have no active FSDP split at this point;
- general matrices using the default `embed` rule can still be sharded over the
  active expert mesh axis; and
- activations are sharded across the eight-device batch mesh.

The baseline is the reference for the mesh sweep and the latency-hiding toggle. It is
not a dropless correctness reference for the sparse expert sweep unless its measured
drop rate is zero.

### Baseline prediction

EP-8 keeps each expert intact along the inactive FSDP axis and pays token movement
across the eight expert shards. The fixed capacity gives stable shapes and computes
the same number of expert slots as an ideally balanced top-2 route. It may waste those
slots on zeros and may drop overflowing assignments.

At balanced routing, one dispatch direction carries:

$$
524{,}288\ \text{assignments}
\times 6{,}144\ \text{elements}
\times 2\ \text{bytes}
=6\ \text{GiB}
$$

of logical node-wide BF16 payload per layer. Dispatch plus combine is 12 GiB per
layer. If routing is uniform, one eighth stays on its source device; the expected
off-device payload is 1.3125 GiB per device per layer, or 73.5 GiB per device over 56
layers. These values exclude collective protocol overhead and backward-pass
collectives.

### Baseline achieved result

> **BLOCKED — MIXTRAL-V26.6-BASELINE.** Run `scripts/baseline.py` on one verified
> eight-MI355X node. Retain all 30 step records, report the median and dispersion over
> the declared steady-state window, and fill:
>
> - median step time: `BLOCKED`
> - node tokens/s: `BLOCKED`
> - tokens/s/GPU: `BLOCKED`
> - useful MFU: `BLOCKED`
> - compiled peak bytes/device: `BLOCKED`
> - observed peak HBM/device: `BLOCKED`
> - assignment drop rate: `BLOCKED`
> - max/mean expert load: `BLOCKED`
> - exposed dispatch plus combine time: `BLOCKED`

### Baseline diagnosis

Do not explain a gap until the result exists. The minimum diagnosis bundle is the
optimized HLO, XLA memory analysis, one unprofiled timing log, one XProf trace, one
RCCL-aware `rocprofv3` trace, and router counts from the same configuration.

## Analytical Predictions

### Compute lower bound

MaxText's useful-FLOP convention counts:

- the two selected experts;
- the router;
- attention projections;
- causal attention score/value products;
- the output vocabulary projection; and
- one forward plus two backward matmul equivalents.

For this recipe, the result is **242.212 GFLOPs per token [analytical]**, or
**63.495 PFLOPs per update [analytical]**. This excludes normalization, sorting,
collectives, optimizer elementwise work, rematerialized operations, and dropped-token
effects.

Using a nominal MI355X BF16 peak of 2.5166 PFLOP/s per GPU gives a node peak of
20.1328 PFLOP/s. The corresponding hard lower bound is:

$$
t_\text{compute}
\geq \frac{63.495\ \text{PFLOPs}}{20.1328\ \text{PFLOP/s}}
=3.154\ \text{s/update}.
$$

The equivalent 100%-peak ceiling is 10,390 tokens/s/GPU. A real end-to-end train step
must be slower because the lower bound omits communication, non-matmul work,
rematerialization, and runtime overhead.

### Dense-masked lower bound

Dense masked execution evaluates all eight experts instead of two. The expert
arithmetic is exactly four times larger. Once attention and shared work are included,
the arm executes about **851.024 GFLOPs per token [analytical]**, or
**223.091 PFLOPs per update [analytical]**. Its nominal compute lower bound is
11.081 seconds, or 2,957 tokens/s/GPU at 100% peak.

MaxText's standard logged TFLOP figure is based on `num_experts_per_tok=2`. It
therefore describes useful top-2 work and undercounts the physical arithmetic of the
dense-masked arm. Report useful MFU for cross-arm comparisons and an
executed-arithmetic rate for kernel diagnosis. Do not mix them.

### State-memory lower bound

The current logical rules shard parameter storage across the active product of FSDP
and EP, which remains eight in the mesh sweep. If the HLO realizes that layout, each
device stores about 17.58 billion parameter elements.

- BF16 parameters alone: 32.74 GiB/device.
- BF16 parameters plus BF16 Adam `mu` and `nu`: 98.23 GiB/device.
- Add one FP32 gradient tree: 163.71 GiB/device.

This is a floor. It excludes gradient-accumulation temporaries, optimizer update
temporaries, activations, rematerialization checkpoints, attention workspaces, routing
buffers, collective buffers, executable storage, and allocator slack. The config has
no separate FP32 master-weight field; do not add one to the ledger without finding it
in the compiled state.

### Predicted ordering

The pre-run hypotheses are:

1. FSDP-1/EP-8 should avoid FSDP gathers of the large expert input dimension but pay
   the largest expert AllToAll.
2. Moving mesh degree from EP to FSDP should reduce token-routing participants while
   increasing expert-weight FSDP communication.
3. Persistent parameter-state memory should remain broadly similar because
   `FSDP x EP = 8`, but activation and communication-buffer peaks can change.
4. Dense masked BF16 should have the largest executed arithmetic and no token drops.
5. The two FP16 sparse arms should do the same useful math and route the same tokens.
   Their direct difference should be the `ragged_dot` lowering.
6. Latency hiding should reduce exposed communication when the scheduler finds
   independent work, with a possible increase in peak live memory.

These are falsifiable predictions. The measured section must retain any result that
reverses them.

## FSDP by EP Sweep

The mesh sweep holds the BF16 fixed-capacity one-hot expert implementation and latency
hiding constant. Only two values change:

| Cell | FSDP | EP | Pre-run communication expectation | Achieved |
|---|---:|---:|---|---|
| Baseline | 1 | 8 | Most EP token movement; no active FSDP split | `BLOCKED` |
| Mesh 2x4 | 2 | 4 | One FSDP degree introduced; EP group halved | `BLOCKED` |
| Mesh 4x2 | 4 | 2 | More weight gather/reduce-scatter; small EP group | `BLOCKED` |
| Mesh 8x1 | 8 | 1 | No EP AllToAll; maximum FSDP degree | `BLOCKED` |

The product remaining eight does not make these configurations equivalent. The two
axes assign different dimensions and induce different collectives. The sweep is
valuable only if HLO confirms that distinction.

### Sweep achieved result

> **BLOCKED — MIXTRAL-V26.6-FSDP-EP-SWEEP.** Run
> `mesh_fsdp2_ep4.py`, `mesh_fsdp4_ep2.py`, and `mesh_fsdp8_ep1.py` under the
> baseline protocol. For every cell, fill:
>
> - median step time and steady-state spread: `BLOCKED`
> - tokens/s/GPU and relative change from baseline: `BLOCKED`
> - useful MFU: `BLOCKED`
> - compiled and observed peak HBM: `BLOCKED`
> - bytes and exposed time by collective type: `BLOCKED`
> - fixed-capacity drop and padding rates: `BLOCKED`
> - HLO verification status: `BLOCKED`

### Sweep diagnosis

For each adjacent move, explain the observed delta as:

$$
\Delta t_\text{step}
=\Delta t_\text{expert-weight communication}
+\Delta t_\text{token movement}
+\Delta t_\text{compute efficiency}
+\Delta t_\text{unhidden overhead}.
$$

Do not infer those terms from the mesh names. Count collective bytes from HLO and
measure exposed time in the trace. If peak memory changes, reconcile the change
against XLA's buffer assignment rather than against parameter storage alone.

## Verify the Mesh in HLO

The config states what was requested. Optimized HLO records what the compiler built.
Each mesh cell passes only after the following checks.

### Parameter layout

Record representative shardings and local shapes for:

- one attention query or output matrix;
- router weights;
- one expert `wi_0`, `wi_1`, and `wo` matrix;
- input embeddings and the output vocabulary projection;
- Adam `mu` and `nu`; and
- one gradient leaf.

The expert weight shape begins with the expert dimension. EP should shard that
dimension. FSDP should affect the `embed_moe` input dimension according to the current
logical rules. Confirm both from HLO instead of assuming that all state is evenly
divided.

### Collective inventory

For each compiled train step, retain:

- collective opcode;
- operand and result shape;
- dtype;
- replica groups;
- channel ID where present;
- source HLO name;
- byte count; and
- whether the trace shows the operation on the critical path.

Expected signatures are:

- EP greater than one: token layout changes around dispatch and combine, expected to
  include AllToAll semantics in the fixed-capacity path.
- FSDP greater than one: parameter AllGather and gradient ReduceScatter associated
  with FSDP-sharded dimensions.
- EP equal to one: no expert-axis token AllToAll.
- FSDP equal to one: no collective whose only cause is an active FSDP group.

A collective merely appearing is not enough. Replica groups must match the intended
physical axis.

### Expert lowering

The baseline and mesh sweep should show fixed-capacity dispatch/combine masks and
expert-shaped dense dots. The dense-masked arm should show all-expert dots without a
capacity dimension. The sparse arms should retain ragged routing and
`ragged-all-to-all`; their `ragged_dot` lowering must then diverge:

- GroupedGEMM arm: backend evidence for the hipBLASLt GroupedGEMM route.
- Dense-padded arm: padded dense dots/GEMMs and no GroupedGEMM backend call.

Use a kernel trace as the final proof. HLO custom-call names can be compatibility
names and need not contain “HIP.”

> **BLOCKED — MIXTRAL-V26.6-HLO.** No optimized v26.6 HLO bundle is retained for the
> eight cells. Required output: text HLO, buffer assignment or memory analysis,
> collective inventory, representative local shapes, and kernel attribution for both
> `ragged_dot` lowerings.

The current launchers always populate `XLA_FLAGS` from `configs/flags/rocm.txt`.
MaxText will not add its default HLO-dump flags when `XLA_FLAGS` already exists.
Before publication, add a dedicated, checked-in HLO-capture mode that appends the dump
destination and module filter to the same effective flag string. Do not edit the
timing flag file by hand and then call the result reproducible.

## Expert Execution Sweep

The expert study uses the FSDP-1/EP-8 mesh and latency hiding on.

| Arm | Compute dtype | Capacity | Expert implementation | Directly comparable? |
|---|---|---|---|---|
| Fixed-capacity one-hot | BF16 | 1.0 | Capacity-shaped dense dots | Baseline only |
| Dense masked | BF16 | Dropless | Every token through all 8 experts | No |
| Ragged GroupedGEMM | FP16 | Dropless | `jax.lax.ragged_dot` to hipBLASLt GroupedGEMM | Yes, to next row |
| Ragged dense-padded | FP16 | Dropless | Same `ragged_dot`, lowered to padded dense dots | Yes, to previous row |

### Fixed-capacity one-hot

This arm is expected to have stable shapes and regular dense kernels. Its performance
can be overstated as useful training throughput if overflow assignments are dropped.
Report both raw input tokens/s and retained expert assignments/s.

### Dense masked

`expert_dense_masked.py` sets `capacity_factor=-1.0` while leaving
`sparse_matmul=false`. MaxText then computes all eight experts for all tokens and
combines only the routed outputs. It avoids capacity overflow, sorting, and sparse
GroupedGEMM, but executes four times the expert arithmetic of top-2 routing.

This arm is a BF16 dropless fallback and a diagnostic. High dense-GEMM utilization
does not by itself make it a good end-to-end result.

### Ragged GroupedGEMM

`expert_ragged_dot.py` sets `sparse_matmul=true`, keeps
`ragged_buffer_factor=-1.0`, and enables:

```text
--xla_gpu_enable_cublaslt=true
--xla_gpu_experimental_use_ragged_dot_grouped_gemm=true
```

On the ROCm backend, the first flag is a compatibility spelling for the BLASLt route;
the implementation under test is hipBLASLt. The arm sorts and exchanges only routed
token assignments, then presents variable group sizes to GroupedGEMM.

### Ragged dense-padded

`expert_dense_padded.py` is identical to the GroupedGEMM arm except:

```text
--xla_gpu_experimental_use_ragged_dot_grouped_gemm=false
```

The frontend still issues `jax.lax.ragged_dot`. XLA lowers it to padded dense work
instead of GroupedGEMM. Because dtype, route, buffer policy, mesh, and AllToAll flags
match, this pair isolates the lowering more closely than any comparison with the BF16
arms.

### FP16 comparison caveat

Only the two sparse FP16 rows support a direct kernel-path claim. Comparing either one
with the BF16 baseline changes at least three things:

1. compute dtype changes from BF16 to FP16;
2. fixed-capacity routing changes to a dropless ragged route; and
3. expert execution changes from capacity-shaped dense dots to `ragged_dot`.

Comparing sparse FP16 with dense-masked BF16 still changes dtype and executed
arithmetic. A faster FP16 sparse arm does not establish that GroupedGEMM beats the
BF16 baseline. It establishes the performance of the whole FP16 dropless recipe.

### Expert sweep achieved result

> **BLOCKED — MIXTRAL-V26.6-EXPERT-SWEEP.** Run the four arms under one timing
> protocol. Fill:
>
> - median step time and tokens/s/GPU for each arm: `BLOCKED`
> - useful top-2 MFU for each arm: `BLOCKED`
> - executed-arithmetic rate for dense masked: `BLOCKED`
> - compiled and observed peak memory: `BLOCKED`
> - tokens/expert distribution: `BLOCKED`
> - fixed-capacity dropped assignments: `BLOCKED`
> - ragged bytes sent/received by peer: `BLOCKED`
> - GroupedGEMM versus dense-padded speed ratio: `BLOCKED`
> - backend proof for each lowering: `BLOCKED`

If dense-padded wins, retain it as a measured negative result for the GroupedGEMM
route. Possible causes include small or uneven groups, GroupedGEMM launch overhead,
workspace behavior, and lower kernel efficiency. Select the cause from profile
evidence, not from the outcome alone.

## Ragged AllToAll Controls

Both sparse arms append:

```text
--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false
--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=true
```

The source comments state that the first flag disables a communicator device
barrier reported as unsupported by the targeted ROCm XLA/RCCL route. The second
requests the one-shot ragged AllToAll kernel. Neither behavior has current v26.6
execution proof. They are a matched pair of requested controls, not general
recommendations for other JAX versions.

Verification requires three records:

1. the startup log showing both effective flags;
2. optimized HLO containing the expected ragged AllToAll operations; and
3. a kernel/RCCL trace proving that the one-shot path ran without a barrier failure.

Do not silently remove an unrecognized or failing flag. That creates a different
experiment. Record the parser or runtime failure, then use the fallback ladder below.

> **BLOCKED — MIXTRAL-V26.6-RAGGED-A2A.** Required artifacts: accepted effective
> flags, forward and reverse ragged AllToAll HLO, peer byte counts, kernel names,
> exposed time, and confirmation that no unsupported device barrier ran.

## Latency Hiding

`configs/flags/rocm.txt` enables:

```text
--xla_gpu_enable_latency_hiding_scheduler=true
```

`overlap_lhs_off.py` starts from the same file and replaces that value with `false`.
Every model and mesh override remains at the BF16 FSDP-1/EP-8 one-hot baseline.

The expected trade is simple:

- LHS on may overlap collectives with independent compute and shorten the critical
  path.
- LHS on may extend buffer lifetimes and raise peak memory.
- LHS off may reduce live memory but expose more communication.

The HLO operation set should remain mathematically equivalent. The useful evidence is
the schedule, buffer assignment, and trace overlap.

### LHS achieved result

> **BLOCKED — MIXTRAL-V26.6-LHS.** Run `baseline.py` and `overlap_lhs_off.py` in
> separate fresh processes. Fill:
>
> - LHS-on median step time: `BLOCKED`
> - LHS-off median step time: `BLOCKED`
> - relative latency change: `BLOCKED`
> - LHS-on compiled/observed peak memory: `BLOCKED`
> - LHS-off compiled/observed peak memory: `BLOCKED`
> - exposed collective time in each arm: `BLOCKED`
> - overlap intervals and critical-path change: `BLOCKED`

Use LHS off as a fit fallback only if the memory reduction is real and needed. Leave
it on if the memory saving is negligible or the run already fits.

## Metrics and Diagnostics

### Headline metrics

For a measured step time $t$:

$$
\text{node tokens/s}=\frac{262{,}144}{t},
\qquad
\text{tokens/s/GPU}=\frac{32{,}768}{t}.
$$

Report:

- median step time;
- p10, p90, minimum, and maximum steady-state step time;
- node tokens/s and tokens/s/GPU;
- useful MFU against the arm's useful top-2 FLOP ledger;
- peak HBM from XLA memory analysis and an observed device high-water mark;
- compile and autotuning time separately from steady-state execution; and
- loss and gradient norm only as finite-value smoke checks.

The 30-step timing protocol must be fixed before the first accepted run. A suitable
single-cohort rule is to retain all steps, discard the first ten from the headline,
and report the median of steps 10 through 29. If a different window is chosen, rerun
every cell. Do not time profiled processes.

### Router diagnostics

Record per layer, and summarize across layers:

- assignments per expert;
- mean, maximum, minimum, standard deviation, and coefficient of variation;
- maximum divided by mean;
- fixed-capacity slots per expert;
- assignments dropped by capacity overflow;
- padding slots and padding fraction;
- effective retained assignments per input token; and
- worst-case and actual ragged receive rows.

The current recipe disables the load-balance loss and reuses synthetic data. Router
imbalance is therefore part of the workload, not noise to average away.

MaxText's standard scalar metrics include loss, gradient norms, step time,
tokens/s/device, and TFLOP/s/device. They do not provide the complete router ledger
above. Add instrumentation before accepting the expert comparison; reconstructing
drop rate from step time is impossible.

### Collective diagnostics

For AllGather, ReduceScatter, AllToAll, and ragged AllToAll, report:

- logical bytes;
- bytes by peer where available;
- call count;
- total device duration;
- exposed critical-path duration;
- overlap with compute;
- achieved payload bandwidth; and
- replica groups.

Sum device times only after accounting for the eight devices. A trace viewer that
sums identical per-device events can overstate wall-clock time by about eight times.

### Kernel diagnostics

For attention, expert up projections, expert down projections, sorting, dispatch, and
combine, retain:

- HLO name and framework scope;
- backend custom-call target where present;
- kernel name;
- input/output shapes and dtype;
- group sizes for GroupedGEMM;
- duration and call count;
- useful and executed FLOPs; and
- achieved FLOP/s.

An MoE kernel can look short while the step waits on one overloaded expert group.
Always read kernel efficiency beside the tokens-per-expert distribution.

## Expected Versus Achieved

Fill this section only after all eight cells pass the same environment, HLO, timing,
and correctness gates.

### 1. Baseline

**Expected:** regular fixed-capacity expert dots, EP-8 token movement, possible
overflow drops, and no active FSDP degree.

**Achieved:** `BLOCKED — MIXTRAL-V26.6-BASELINE`.

**Explanation:** `BLOCKED — requires HLO, profile, memory, and router artifacts`.

### 2. FSDP by EP

**Expected:** shifting mesh degree toward FSDP trades expert token movement for
weight/gradient communication. Parameter-state storage remains broadly stable while
temporary memory can change.

**Achieved:** `BLOCKED — MIXTRAL-V26.6-FSDP-EP-SWEEP`.

**Explanation:** `BLOCKED — requires per-collective bytes and exposed time`.

### 3. Expert execution

**Expected:** dense masked pays much more arithmetic; the sparse pair performs matched
useful work; GroupedGEMM wins only if its efficiency and reduced padded work outweigh
grouping overhead.

**Achieved:** `BLOCKED — MIXTRAL-V26.6-EXPERT-SWEEP`.

**Explanation:** `BLOCKED — requires lowering proof, group sizes, drop rate, and kernel
efficiency`.

### 4. Latency hiding

**Expected:** LHS on reduces exposed communication at a possible memory cost.

**Achieved:** `BLOCKED — MIXTRAL-V26.6-LHS`.

**Explanation:** `BLOCKED — requires schedule, buffer assignment, and overlap trace`.

### Final recommendation

> **BLOCKED — MIXTRAL-V26.6-RECOMMENDATION.** Select a mesh, expert path, and LHS
> setting only after the cells above are complete. State the recommendation as an
> exact config and flag set, followed by its tokens/s/GPU, peak memory, drop rate,
> useful MFU, and applicable software revisions.

## Predeclared Fallback Plan

No v26.6 result exists yet. Apply this fallback order only when the stated
condition is captured:

1. **GroupedGEMM is correct but slower than dense-padded.** Keep FP16 sparse routing
   and set
   `--xla_gpu_experimental_use_ragged_dot_grouped_gemm=false`. Retain the
   GroupedGEMM profile as the negative result.
2. **GroupedGEMM fails to compile or selects the wrong backend.** Use the matched
   dense-padded sparse arm. Do not describe that as a GroupedGEMM result.
3. **The FP16 sparse route is numerically or operationally unacceptable.** Use BF16
   dense masked for a dropless correctness fallback, accepting its larger arithmetic
   cost.
4. **Ragged AllToAll is unsupported on the pinned communicator.** Return to the BF16
   fixed-capacity one-hot path and report its drop rate. If drops are unacceptable,
   increase capacity in a new experiment or use dense masked; do not hide the change
   inside this sweep.
5. **The selected mesh runs out of memory with LHS on.** Test LHS off. Keep it only if
   the measured memory reduction makes the run fit.
6. **An intended EP layout lowers to FSDP-style collectives.** Fail the HLO gate.
   Check `moe_dispatch_no_expert_sharding`, logical axis rules, and local expert
   shapes before timing again.

A fallback changes the recommendation, not the historical cell. Keep both artifacts
and state which constraint forced the change.

## Reproduce the Study

### Source revisions

The draft was derived from:

```text
experiment repository
/home/clchong/work/mixtral8-22b
commit a32b51d6b4032afb19fb30340c84a57139ae4bb7

Reference model config inspected while drafting
/home/clchong/work/maxtext-mxfp4-v26.6
commit b437942a5f33704f8438deb948488ad08164285c
model config src/maxtext/configs/models/mixtral-8x22b.yml
```

The launchers default to `MAXTEXT_ROOT=/workspace/maxtext`, not the feature tree
above. The local tree establishes the model dimensions used in the analytical
ledger, but it cannot establish implementation behavior for a stock run. Before
accepting a run, record `git rev-parse HEAD` and `git status --short` from the tree
actually imported inside the container, then re-audit routing, capacity, sharding,
and ragged lowering against that revision. A result from `b437942a` belongs to a
separate feature-branch cohort.

Also retain:

- container tag and immutable image digest;
- ROCm, JAX, jaxlib, PJRT plugin, RCCL, Transformer Engine, and hipBLASLt versions;
- `jax.devices()` output;
- MI355X compute and memory partition modes;
- `rocm-smi --showtopo`;
- effective `XLA_FLAGS`;
- full resolved MaxText config; and
- hashes of the experiment config and all launchers.

### Commands

Run each cell in a fresh process:

```bash
cd /home/clchong/work/mixtral8-22b

python3 scripts/baseline.py
python3 scripts/mesh_fsdp8_ep1.py
python3 scripts/mesh_fsdp4_ep2.py
python3 scripts/mesh_fsdp2_ep4.py
python3 scripts/expert_dense_masked.py
python3 scripts/expert_ragged_dot.py
python3 scripts/expert_dense_padded.py
python3 scripts/overlap_lhs_off.py
```

The scripts:

- expose devices `0,1,2,3,4,5,6,7`;
- set `JAX_PLATFORMS=rocm`;
- remove inherited `XLA_FLAGS` and rebuild them from the checked-in flag file;
- set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.97`;
- prepend the selected MaxText `src` directory to `PYTHONPATH`;
- write under `OUTPUT_ROOT`, defaulting to `/tmp/mixtral8-22b`; and
- pass additional command-line overrides directly to MaxText.

Use a unique, durable `OUTPUT_ROOT` for accepted results. `/tmp` is only the launcher
default and is not an artifact-retention policy.

For a short compile or launch smoke test, append `steps=2`. Do not mix smoke-test
numbers with the 30-step timing cohort.

### Artifact layout

Each accepted cell should retain:

```text
manifest.yaml
run-contract.yaml
checksums.sha256
config/effective.yaml
config/xla_flags.txt
logs/steps.jsonl
results/metrics.json
ledgers/moe.json
hlo/optimized.txt
hlo/memory-analysis.txt
profiles/xprof/
profiles/rocprof/
```

Appendix F is normative; this list names only the case-specific minimum. Timing,
XProf, `rocprofv3`, and HLO capture should run in separate fresh processes
with the same resolved config. Profiler overhead invalidates headline timing.

## Publication Blockers

The chapter cannot make a v26.6 performance recommendation until these are complete:

1. **Runtime identity:** immutable container digest, imported MaxText commit, package
   versions, device partitioning, and topology.
2. **Eight timing cells:** retained 30-step logs under one declared steady-state
   protocol.
3. **Memory:** compiled memory analysis and observed peak HBM for every cell.
4. **HLO:** representative parameter shardings, replica groups, collective byte
   counts, and both `ragged_dot` lowerings.
5. **Routing:** tokens per expert, fixed-capacity drop/padding rates, and actual ragged
   receive sizes.
6. **Profiles:** unperturbed timing plus separate XProf and RCCL-aware `rocprofv3`
   captures.
7. **Correctness smoke checks:** finite loss and gradients, and matched short-run loss
   behavior for the directly compared sparse pair.
8. **Recommendation:** one exact mesh, expert implementation, and LHS setting with a
   documented fallback.

Until then, the fixed recipe, arithmetic ledger, predicted tradeoffs, and timing
commands are available. HLO capture, router instrumentation, and all achieved
performance fields remain **BLOCKED**.
