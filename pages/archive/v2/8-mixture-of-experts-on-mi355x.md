---
layout: distill
title: "Training Mixture-of-Experts on MI355X"
description: "Apply memory, parallelism, communication, and kernel decisions together when sparse training work is determined by routing."
date: 2026-09-13

section_number: 8

previous_section_url: "/pages/7-a-map-of-kernel-backends-on-jax"
previous_section_name: "Chapter 7: A Map of ROCm Kernel Backends on JAX"

next_section_url: "/pages/9-compiler-runtime-and-rccl-controls"
next_section_name: "Chapter 9: Tuning the Compiler, Runtime, and RCCL"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "Why Sparse Training Is Different"
  - name: "Routing and Precision"
  - name: "Load Balance and Imbalance"
  - name: "Capacity Padding Dropping and Dropless Execution"
  - name: "Four Expert Execution Paths"
    subsections:
      - name: "Dense Masked"
      - name: "Fixed Capacity One Hot"
      - name: "Sparse Dense Padded"
      - name: "Sparse Grouped GEMM"
      - name: "Read the Expert Lowering"
  - name: "The Grouped GEMM Win Condition"
  - name: "Token Movement and Ragged Collectives"
  - name: "Expert Parallelism on One MI355X Node"
    subsections:
      - name: "Mixtral state under EP-8"
      - name: "EP versus FSDP"
      - name: "EP versus TP"
      - name: "One-node placement"
  - name: "Rematerialization and Custom VJPs"
  - name: "Four Required Diagnostics"
  - name: "Decision Procedure"
  - name: "Current Status and Blockers"
  - name: "References"
---

A sparse Mixture-of-Experts layer stores many feed-forward networks but sends each
token to only a few of them. This reduces the model FLOPs per token. It does not
automatically reduce step time by the same factor. The router makes the expert batch
sizes data-dependent, the tokens may move between GPUs twice per layer, and the
resulting matrix multiplications are smaller and less regular than a dense MLP.

Those costs matter on MI355X for two hardware-specific reasons. CDNA4 Matrix Cores are
fast when a GEMM is large and regularly tiled; a collection of narrow or ragged expert
GEMMs can leave much of that throughput unused. In the other direction, an eight-GPU
MI355X node is a one-hop full mesh, which is a good topology for an intra-node
AllToAll. Achieved step time therefore depends on expert-kernel efficiency and the
physical placement of the expert mesh axis, in addition to the reduced useful FLOP
count.

This chapter is the integration point for sparse training. Earlier chapters only
signpost MoE-specific exceptions; routing, capacity, expert kernels, token
collectives, expert parallelism, rematerialization, and diagnostics are owned here
and evaluated as one training contract.

This chapter uses four evidence labels:

- **[source]** comes from checked-in code, configuration, or a pinned source tree.
- **[analytical]** follows from the model dimensions and stated hardware constants.
- **[cited]** is supported by a named external paper or documentation source.
- **[measured]** requires an MI355X artifact produced under the book's measurement
  protocol.

There are no `[measured]` performance results in this chapter yet. In particular, the
MI300X timings in the archived draft are not MI355X evidence and are not reused here.
The current implementation discussion is pinned to
`rocm/jax-training:maxtext-v26.6`, the matching MaxText source tree used by this
project, and the experiment scripts inspected on 13 September 2026.

For the full generic derivations behind Transformer and MoE accounting, read the
[Transformer chapter](https://jax-ml.github.io/scaling-book/transformers/) of the JAX
Scaling Book. Its [sharding](https://jax-ml.github.io/scaling-book/sharding/) and
[training](https://jax-ml.github.io/scaling-book/training/) chapters provide the
general collective and parallelism theory. The purpose here is to turn that theory into
MI355X, MaxText, and ROCm decisions.

## Why Sparse Training Is Different

Use the following symbols for one routed SwiGLU layer:

- $T$: tokens processed by one device before expert dispatch
- $D$: model width
- $F$: intermediate width of one expert
- $E$: routed experts
- $k$: experts selected per token, called `num_experts_per_tok` in MaxText
- $E_s$: shared experts that run for every token
- $P$: expert-parallel degree
- $w_a$: bytes per communicated activation element
- $w_p$: bytes per communicated weight element

One SwiGLU expert has a gate projection, an up projection, and a down projection. Its
parameter count is

$$
N_{\text{one expert}} = 3DF.
$$

All routed expert weights occupy

$$
N_{\text{routed weights}} = 3EDF
$$

parameters, while one token activates only

$$
N_{\text{active expert weights/token}} = 3(k + E_s)DF.
$$

The corresponding expert arithmetic is $6(k+E_s)DF$ FLOPs per token in the forward
pass and approximately $18(k+E_s)DF$ FLOPs per token for forward plus backward. The
router projection adds $2DE$ forward FLOPs per token, or approximately $6DE$ for
training. **[analytical]**

This produces two counts that must remain separate:

1. **Total parameters** determine model-state memory, checkpoint size, optimizer state,
   and how much weight data FSDP may gather.
2. **Activated parameters** determine the useful model FLOPs for each token.

An MoE can therefore have the compute cost of a much smaller model while retaining the
optimizer-state cost of the full model. Expert parallelism is often required for memory
before it is considered as a performance optimization.

The dense-masked implementation runs all $E$ routed experts for every token. Ignoring
shared experts and the router, its arithmetic multiplier over ideal sparse execution is

$$
\text{dense-masked multiplier} = \frac{E}{k}.
$$

Current MaxText model configs show how quickly that multiplier grows. **[cited]**

| MaxText model config | $D$ | $F$ | $E$ | $k$ | Shared | Dense-masked multiplier |
|---|---:|---:|---:|---:|---:|---:|
| Mixtral 8x22B | 6144 | 16384 | 8 | 2 | 0 | 4x |
| Qwen3 30B-A3B | 2048 | 768 | 128 | 8 | 0 | 16x |
| DeepSeek V3 | 7168 | 2048 | 256 | 8 | 1 | 32x on the routed experts |

The shared DeepSeek expert is not part of the 32x ratio. It runs for every token and
must be added to both the useful and issued FLOP counts.

Three FLOP counts are useful in an MoE profile:

- **Useful model FLOPs** count selected routed experts and shared experts.
- **Issued expert FLOPs** include dense masking, fixed capacity, kernel padding, and
  rematerialization.
- **Hardware FLOPs** are what counters attribute to the actual matrix instructions.

Tokens/s/GPU is the primary comparison metric in this book. MFU is still useful, but
only after stating which of these FLOP counts forms its numerator. A dense-masked path
can have good hardware utilization while performing four, sixteen, or thirty-two times
the useful expert arithmetic. A dropping path can make a useful-FLOP MFU look better by
skipping work the model was meant to perform.

## Routing and Precision

For ordinary token-choice routing, MaxText computes one score per token and expert,
selects the top $k$, then normalizes the selected weights:

```python
gate_logits = gate(inputs)                  # [batch, sequence, experts]
topk_values, topk_indices = jax.lax.top_k(
    gate_logits, num_experts_per_tok
)
topk_weights = jax.nn.softmax(
    topk_values.astype(jnp.float32), axis=-1
).astype(dtype)
```

The exact scoring and normalization differ by model. DeepSeek can use sigmoid scores,
routing groups, and a bias for load balancing. Qwen3 enables
`norm_topk_prob`. Gemma 4 computes a full fp32 router softmax before gathering the
selected probabilities. Treat the model config as part of the architecture, not as a
performance-only setting.

The router is small compared with the experts, but its output controls every later
shape and collective. Small score perturbations can change a top-k decision. MaxText
v26.6 exposes two relevant precision controls:

- `float32_gate_logits` chooses fp32 for the gate projection. It is `false` in
  `base.yml`, so the projection follows the model dtype unless the recipe overrides it.
- `float32_weight_sum` controls the weighted sum of expert outputs. It is `true` by
  default.

The ordinary top-k path casts selected logits to fp32 for its softmax, but top-k
selection itself still sees the gate-logit dtype. Setting `float32_gate_logits=true`
therefore protects more than the softmax; it can also stabilize expert selection.
Whether the extra gate cost changes tokens/s/GPU is an MI355X measurement, but the gate
is normally a small fraction of the layer FLOPs. **[analytical]**

Reduced precision in the expert MLP is a separate decision. The current Mixtral
experiment uses BF16 for dense-masked and fixed-capacity paths. Its two
`jax.lax.ragged_dot` arms use FP16 because the ROCm grouped-GEMM lowering under test is
currently exercised only in FP16. The FP16 sparse arms must therefore be compared with
each other, not silently merged into a BF16 ranking. FP16 also changes the numerical
requirements: loss scaling and overflow checks become part of the experiment.

The local MaxText branch contains a JAX-AITER MXFP4 fused-MoE forward path. It is marked
inference-only and has no training VJP. It is not one of the four training paths below.
[Chapter 7]({{ '/pages/7-a-map-of-kernel-backends-on-jax' | relative_url }}) defines
the forward/backward kernel-proof standard; this chapter records the sparse routes
that pass or fail it.

## Load Balance and Imbalance

Token-choice routing does not guarantee equal expert batches. For a layer with $T$
local tokens, the mean number of routed assignments per expert is

$$
\bar{n} = \frac{Tk}{E}.
$$

Define the imbalance factor as

$$
I = \frac{\max_e n_e}{\bar{n}}.
$$

This ratio has different effects in the four implementations:

- Dense masked always runs every expert on every token, so router imbalance does not
  change its expert GEMM work.
- Fixed capacity always runs the allocated slots. Imbalance appears as more padding in
  underfull experts and more drops in overflowing experts.
- Dense-padded sparse lowering and grouped GEMM receive the actual group sizes.
  Imbalance changes their shapes and may make the longest or least efficient group
  determine layer time.
- Expert parallelism also creates **device imbalance**. What matters for communication
  and device idle time is the sum of assignments sent to all experts on each EP shard,
  not only the busiest individual expert.

MaxText has two balancing mechanisms.

The auxiliary loss is enabled by a positive `load_balance_loss_weight`. The v26.6 base
value is `0.0`, so it is off unless the model config or command line changes it. The
implemented loss is proportional to the product of each expert's selected-token
density and mean router probability. It changes the optimization objective, so a
throughput benefit is not enough to justify a larger weight; convergence remains the
guardrail.

DeepSeek-style loss-free balancing uses `routed_bias` and
`routed_bias_update_rate`. The bias changes expert selection but not the mixing weight
computed from the original score. In the current DeepSeek V3 model config,
`routed_bias` is `true`, while the base update rate remains `0.0`. A recipe that does
not set a positive update rate has declared the mechanism without activating its
updates. **[cited]**

Random or forced-balanced routing is useful as a performance control. It answers
whether the slow step is caused by the learned distribution or by the kernel at a
balanced shape. It is not a training recipe and cannot establish model quality.

Imbalance changes with the data and with training. Log it over time and by layer. A
single synthetic batch is sufficient for a kernel comparison only when the routing
histogram is fixed and reported.

## Capacity Padding Dropping and Dropless Execution

Static GEMMs need a static number of token slots per expert. In the pinned MaxText
training path, the exact Python calculation is

$$
C =
\operatorname{int}\left(
\max\left[
\left\lceil \frac{Sk}{E}\right\rceil
\texttt{capacity_factor},
\texttt{capacity_factor}
\right]\right).
$$

This is slots per expert and batch example, where \(S\) is sequence length. The
outer `int` truncates a non-integer product after the `max`. The inference-only
subgroup path substitutes its context subsequence length for \(S\); this book's
training path uses the full sequence length.

Capacity creates two costs:

- An expert receiving fewer than $C$ assignments runs padded slots.
- Assignments after slot $C$ are masked out and dropped.

For top-k routing, report both an **assignment drop rate** and a **token drop rate**.
A token can lose one of its $k$ expert assignments while retaining another. The two
rates therefore answer different correctness questions.

`capacity_factor=1.0` is dropless only under perfect balance. The current Mixtral
baseline uses that value, so a real-data run must measure drops rather than assume
there are none. Raising the factor reduces drops but increases allocated and issued
work. Lowering it may improve step time by changing the computation, which is not a
valid speedup unless the resulting quality trade is accepted.

The sparse path uses `ragged_buffer_factor` instead:

- A negative value allocates a worst-case receive buffer and guarantees that the buffer
  itself does not force dropping.
- A positive value allocates a multiple of the balanced receive size. Overflow is
  deterministically truncated and dropped.

The v26.6 default is `ragged_buffer_factor=-1.0`. Its worst-case factor is bounded in
MaxText by

$$
\min\left(P,\frac{E}{k}\right).
$$

This protects semantics at a potentially large memory cost. The ragged collective
communicates actual send and receive sizes, but XLA still needs a statically sized
output buffer. A "dropless" run can therefore carry substantial padding in allocated
memory even when its grouped GEMM uses only valid rows.

Keep three properties separate:

1. **Dropping:** whether a selected assignment is discarded.
2. **Allocated padding:** unused rows reserved in a static buffer.
3. **Issued padding:** unused rows on which a lowered kernel performs matrix work.

The sparse dense-padded path below is dropless at the model level while still issuing
padded matrix work. Calling it simply "dropless" hides the performance question.

## Four Expert Execution Paths

Current MaxText and the MI355X Mixtral experiment expose four useful training paths.
The last two have the same JAX-level sparse algorithm and differ in the XLA lowering of
`jax.lax.ragged_dot`.

| Path | MaxText fields | Relevant XLA control | Model semantics |
|---|---|---|---|
| Dense masked | `sparse_matmul=false`, `capacity_factor=-1` | Normal dense GEMM selection | Dropless; all experts run |
| Fixed capacity one hot | `sparse_matmul=false`, `capacity_factor>0` | Normal dense GEMM selection | May drop; static expert slots |
| Sparse dense padded | `sparse_matmul=true`, `megablox=false`, `ragged_buffer_factor=-1` | `--xla_gpu_experimental_use_ragged_dot_grouped_gemm=false` | Dropless buffer; padded lowering |
| Sparse grouped GEMM | Same MaxText fields | `--xla_gpu_experimental_use_ragged_dot_grouped_gemm=true` | Dropless buffer; grouped lowering |

`use_tokamax_gmm=false` is also set in the current sparse experiment. The base MaxText
defaults select `sparse_matmul=true` and `megablox=true`; those defaults are not a
statement that the Megablox TPU-oriented kernel is the correct MI355X path. On this
ROCm experiment, `megablox=false` is deliberate.
[Chapter 7]({{ '/pages/7-a-map-of-kernel-backends-on-jax' | relative_url }}) explains how to prove the kernel
that was actually reached.

### Dense Masked

Dense masked execution applies every routed expert to every token and gives unselected
experts a zero combine weight:

```text
BSM,EMH -> BSEH
BSEH,EHM -> BSEM
BSEM,BSE -> BSM
```

Its advantages are regular shapes, ordinary dense GEMMs, no token sort, and no
data-dependent expert capacity. These are favorable properties for MI355X Matrix
Cores. Its disadvantages are the $E/k$ arithmetic multiplier and large intermediate
activations carrying an expert dimension.

Dense masked is useful for:

- numerical reference checks;
- bringing up a new model before sparse dispatch works;
- small $E/k$ when the sparse kernel is exceptionally inefficient;
- separating routing quality from sparse-kernel behavior.

It is not automatically a reasonable fallback for a fine-grained model. A 16x or 32x
expert multiplier is a large bill even when the dense GEMMs are efficient.

Current Mixtral experiment settings:

```yaml
dtype: bfloat16
sparse_matmul: false
capacity_factor: -1.0
ici_expert_parallelism: 8
moe_dispatch_no_expert_sharding: true
```

The path is semantically dropless. It does not exploit sparse activation.

### Fixed Capacity One Hot

The fixed-capacity path constructs dispatch and combine masks with shape containing
`[experts, capacity]` and runs static expert buffers:

```text
BSM,BSEC -> EBCM
EBCM,EMH -> EBCH
EBCH,EHM -> EBCM
EBCM,BSEC -> BSM
```

This gives the compiler regular expert GEMMs and makes the maximum memory footprint
predictable. It also adds dense dispatch and combine contractions. Their work grows
with both sequence length and capacity; because capacity itself grows with sequence
length, this route becomes increasingly unattractive at long context.

The current MI355X Mixtral baseline uses:

```yaml
dtype: bfloat16
sparse_matmul: false
capacity_factor: 1.0
ici_fsdp_parallelism: 1
ici_expert_parallelism: 8
moe_dispatch_no_expert_sharding: true
```

`moe_dispatch_no_expert_sharding=true` matters only on MaxText's dense MoE path. It
removes the `expert` mesh axis from the dispatch batch dimension so that the expert
dimension remains expert-sharded. The v26.6 base default is `false`. Verify the
compiled HLO: the intended EP layout should not turn into replicated activations or an
FSDP-style gather of all expert weights.

This is the correct baseline for the current case study, not a proven winner. At
`capacity_factor=1.0`, the drop rate is part of the result.

### Sparse Dense Padded

The sparse path first duplicates each token $k$ times, sorts assignments by expert,
forms `group_sizes` with `jnp.bincount`, and calls `jax.lax.ragged_dot`. With the
grouped-GEMM XLA lowering disabled, the current ROCm experiment asks XLA to lower that
ragged operation through its dense-padded route.

```yaml
dtype: float16
sparse_matmul: true
megablox: false
use_tokamax_gmm: false
ragged_buffer_factor: -1.0
```

```text
--xla_gpu_experimental_use_ragged_dot_grouped_gemm=false
```

Every selected assignment is retained by the worst-case buffer, but the matrix
lowering may issue work on padded rows. This path is valuable because it separates the
benefit of sparse routing and ragged AllToAll from the quality of the grouped-GEMM
kernel.

Record the lowered GEMM shapes. "Dense padded" is an implementation description, not a
fixed padding factor; XLA can change its padding and batching strategy between
versions.

### Sparse Grouped GEMM

This path keeps the same sort, group sizes, ragged dispatch, and MaxText fields, then
enables the ROCm grouped-GEMM lowering:

```text
--xla_gpu_experimental_use_ragged_dot_grouped_gemm=true
```

The intended expert work is one GEMM group per local expert, with runtime group sizes.
There is no model-level capacity padding. Kernel tiles still have tails, and empty or
small groups can reduce occupancy.

The attraction is exact sparse arithmetic. The risk is poor Matrix Core utilization on
small and uneven groups. Mixtral has only eight wide experts and is the easier bring-up
case. Qwen3 30B-A3B has 128 experts of width 768; it saves more arithmetic relative to
dense masking but presents much narrower expert GEMMs.

The current experiment constrains both sparse arms to FP16. Its README states that no
v26.6 timings have been recorded. The grouped path is therefore a candidate, not the
recommended MI355X implementation.

Megablox and Tokamax are additional `jax.lax.ragged_dot`-style backends in upstream
MaxText's decision tree, while the local JAX-AITER branch has a forward-only fused
MXFP4 MoE path. They do not create additional validated MI355X training paths here.
Their platform, dtype, gradient, and workspace restrictions are part of this
chapter's versioned sparse-route status.

### Read the Expert Lowering

Read the optimized HLO as a delta between paths, not as proof by operation name alone.
The four graphs below are literal XLA DOT output from
`bench/hlo_feature_fixtures.py`, captured on gfx950 with JAX 0.11.0. They isolate
expert execution for 64 tokens, four experts, and width 128. The fixtures do not
model the preceding token sort or cross-device AllToAll; those remain
case-study evidence requirements. Raw HLO, DOT, debug options, flags, and
provenance are retained under `artifacts/hlo-fixtures/moe/`.

{% include figure.liquid path="pages/archive/img/hlo-moe-dense-masked.svg" class="img-fluid" zoomable=true alt="Literal HLO graph for dense-masked expert execution" caption="Captured `before_optimizations` HLO. The first `dot_general` produces `[tokens,experts,width]`; the second contracts routing weights while keeping every expert's issued work. Raw artifact: `artifacts/hlo-fixtures/moe/dense-masked/`." %}

{% include figure.liquid path="pages/archive/img/hlo-moe-fixed-capacity.svg" class="img-fluid" zoomable=true alt="Literal HLO graph for fixed-capacity one-hot expert execution" caption="Captured `before_optimizations` HLO. Dispatch `[tokens,experts,capacity]` creates `[experts,capacity,width]`, which feeds the expert dot and final combine dot. Raw artifact: `artifacts/hlo-fixtures/moe/fixed-capacity/`." %}

{% include figure.liquid path="pages/archive/img/hlo-moe-ragged-padded.svg" class="img-fluid" zoomable=true alt="Literal optimized HLO graph for ragged dot lowered to padded dense work" caption="Captured gfx950 optimized HLO with grouped GEMM disabled. The runtime group sizes build masks and padded operands around an ordinary GEMM fusion. Raw artifact: `artifacts/hlo-fixtures/moe/ragged-padded/`." %}

{% include figure.liquid path="pages/archive/img/hlo-moe-ragged-grouped.svg" class="img-fluid" zoomable=true alt="Literal optimized HLO graph for ragged dot lowered to hipBLASLt GroupedGEMM" caption="Captured gfx950 optimized HLO with grouped GEMM enabled. The same `ragged_dot` frontend becomes the compatibility-named `__cublas$lt$groupedMatmul` custom call implemented by hipBLASLt on ROCm. Raw artifact: `artifacts/hlo-fixtures/moe/ragged-grouped/`." %}

The last pair is the cleanest compiler comparison. Hold the sparse frontend, router
indices, FP16 dtype, mesh, remat policy, and ragged collective flags fixed. The HLO
should match through `ragged_dot` and diverge only at its lowering. A different sort,
`group_sizes`, collective, sharding, or backward route means the test changed more than
the grouped-GEMM lowering.

For every path, record:

- local input, weight, intermediate, and output shapes before and after dispatch;
- fixed capacity \(C\), or the runtime `group_sizes` for every local expert;
- issued padding as rows and FLOPs, separately from allocated buffer padding;
- the collective operation and replica groups, including whether it is
  `ragged-all-to-all`;
- the custom-call target and traced kernel name, or their confirmed absence;
- both forward and backward routes, including any route repeated by rematerialization;
- temporary and library-reported workspace bytes.

[Chapter 2]({{ '/pages/2-lowering-jax-jit-on-rocm' | relative_url }}#reading-a-compiler-delta) locates
`ragged_dot`, GPU lowering, custom calls, and runtime dispatch in the compilation
pipeline. [Appendix D]({{ '/pages/d-profiler-and-hlo-cookbook' | relative_url }}#feature-comparison-bundles)
gives the dump, search, trace, and memory-recording procedure for collecting this
delta.

## The Grouped GEMM Win Condition

A sparse implementation wins only when the arithmetic it avoids is worth more than its
loss in GEMM efficiency and its routing overhead.

Let:

- $\eta_d$ be dense-masked GEMM efficiency relative to the applicable MI355X dtype
  ceiling;
- $\eta_g$ be grouped-GEMM efficiency;
- $F_u$ be useful routed-expert FLOPs;
- $t_r$ include sorting, permutation, dispatch, combine, and other sparse-only time;
- $C_{\text{peak}}$ be the applicable MI355X peak for the dtype and sparsity mode.

Then

$$
t_{\text{dense}} =
\frac{F_u(E/k)}{\eta_d C_{\text{peak}}}
$$

and

$$
t_{\text{grouped}} =
\frac{F_u}{\eta_g C_{\text{peak}}} + t_r.
$$

Grouped GEMM wins when

$$
\frac{1}{\eta_g}
+ \frac{t_r C_{\text{peak}}}{F_u}
<
\frac{E/k}{\eta_d}.
$$

If routing overhead is temporarily ignored, the familiar lower bound is

$$
\eta_g > \eta_d\frac{k}{E}.
$$

For Mixtral, grouped efficiency must exceed one quarter of dense efficiency. For Qwen3
30B-A3B it must exceed one sixteenth. For the routed part of DeepSeek V3 it must exceed
one thirty-second. **[analytical]** These ratios explain why fine-grained sparsity can
tolerate a poor grouped kernel in theory.

They do not prove that the current ROCm lowering wins. Fine-grained models also make
the grouped GEMMs narrower, which can reduce $\eta_g$, and the router plus collectives
remain outside the simplified inequality.

For the sparse dense-padded comparison, define $p$ as issued padded expert FLOPs divided
by useful expert FLOPs, and $\eta_p$ as the efficiency of those dense-padded GEMMs.
Because the two sparse arms otherwise share the routing graph, grouped GEMM wins their
expert-compute comparison when

$$
\eta_g > \frac{\eta_p}{p}.
$$

For fixed-capacity one-hot, use its actual issued FLOPs and include the dispatch and
combine contractions. First require equal model semantics. A capacity path that drops
assignments is not a valid throughput winner over a dropless path until the drop policy
has been accepted.

The practical measurement is straightforward:

1. Hold tokens, model, router indices, dtype, EP mesh, and remat policy fixed.
2. Measure tokens/s/GPU.
3. Extract expert-kernel time and shapes from the trace.
4. Calculate useful and issued expert FLOPs separately.
5. Substitute measured efficiencies and routing time into the inequality.

The current experiment cannot hold dtype fixed between its BF16 fixed-capacity
baseline and FP16 sparse arms. It can still compare the two FP16 `ragged_dot` lowerings
directly. Any broader ranking must state the dtype caveat.

## Token Movement and Ragged Collectives

With expert parallelism, the current sparse MaxText path performs the following work in
each routed layer:

1. Compute router scores and top-k assignments locally.
2. Repeat each input $k$ times and sort the assignments by destination expert.
3. Count assignments per expert and per EP shard.
4. Dispatch variable-sized segments with `jax.lax.ragged_all_to_all`.
5. Sort the received rows by local expert.
6. Run the local expert GEMMs.
7. Undo the local sort.
8. Return outputs with a second ragged AllToAll.
9. Undo the original permutation and sum the $k$ weighted outputs per token.

Under balanced routing, one forward dispatch moves approximately

$$
w_aTkD\frac{P-1}{P}
$$

bytes out of each device. The combine moves the same volume back, giving

$$
B_{\text{forward pair}} \approx
2w_aTkD\frac{P-1}{P}.
$$

This is an activation-volume estimate, not an RCCL time prediction. Protocol overhead,
buffer padding, peer distribution, and overlap still matter. Backward introduces the
transpose communication for activation and output gradients and must be counted from
the compiled training HLO. **[analytical]**

The MI355X node's one-hop full mesh makes an eight-way AllToAll plausible: each peer is
directly reachable. Do not turn the aggregate link sum into an assumed application
bandwidth. RCCL scheduling, message size, and simultaneous peer traffic determine the
measured result.

The current sparse experiment adds:

```text
--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false
--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=true
```

The `unsupported` prefix is important. This is a versioned experiment, not a stable
XLA contract. Preserve the effective flag string with every result, check the HLO for
`ragged-all-to-all`, and check the trace for the intended one-shot lowering.
[Chapter 9]({{ '/pages/9-compiler-runtime-and-rccl-controls' | relative_url }}) covers controlled flag sweeps and
initialization order.

MaxText also has `use_ring_of_experts`. In v26.6 this replaces the pair of AllToAll
operations with an AllGather before local routing and a ReduceScatter afterward.
`num_moe_token_chunks` can split this path so communication for one chunk overlaps
expert compute for another. The trade depends on $k$, expert placement, message sizes,
and exposed overlap. It should be treated as a separate algorithm, not toggled inside
an otherwise matched AllToAll test.

Ragged collectives solve communication semantics, not static allocation. The output
buffer is still statically shaped. With `ragged_buffer_factor=-1`, measure its HBM
footprint as well as the actual bytes sent.

## Expert Parallelism on One MI355X Node

Expert parallelism shards the expert dimension:

```text
expert_weights[E / P, D, F]
```

Each device holds complete local experts. Tokens travel to the devices that own their
selected experts. This reduces per-device expert weights and optimizer state by roughly
$P$, subject to shared experts, dense layers, replicated router state, and optimizer
layout. It introduces dispatch and combine on the critical path.

The memory argument usually comes first. An MoE optimizer stores state for all expert
parameters, not only the selected experts. If the state does not fit on one MI355X,
the minimum EP or FSDP degree is constrained before throughput tuning begins.

### Mixtral state under EP-8

The current Mixtral recipe stores BF16 parameters and BF16 Adam moments. In the
inspected trainer, its BF16 parameter gradients remain BF16 despite the
`grad_dtype=float32` request. The first estimate is therefore six persistent bytes
and two live-gradient bytes per local parameter.

Of approximately 140.63 billion total parameters, 135.291 billion belong to routed
experts and 5.339 billion are shared attention, router, norm, embedding, and output
parameters. With FSDP-1 and EP-8:

$$
\begin{aligned}
P_{\mathrm{local}}
  &=\frac{135.291\times10^9}{8}+5.339\times10^9
   =22.25\times10^9,\\
M_{\mathrm{persistent}}
  &\approx124.33\ \mathrm{GiB/GPU},\\
M_{\mathrm{persistent+gradients}}
  &\approx165.78\ \mathrm{GiB/GPU}.
\end{aligned}
$$

This is **[analytical]** and must be checked against resolved parameter shardings,
compiled memory, and runtime peak allocation. Top-2 routing reduces activated FLOPs;
it does not reduce these stored weights or optimizer moments.

### EP versus FSDP

EP sends activations to stationary expert weights. FSDP shards expert weights, then
gathers them for computation. Let \(P_{\mathrm{FSDP}}\) be the FSDP degree and
\(P_{\mathrm{EP}}\) the expert-parallel degree. For one SwiGLU MoE layer, the
per-device forward weight-gather traffic is approximately

$$
3w_pEDF\frac{P_{\mathrm{FSDP}}-1}{P_{\mathrm{FSDP}}}.
$$

The forward dispatch-and-combine activation volume per device is approximately

$$
2w_aTkD\frac{P_{\mathrm{EP}}-1}{P_{\mathrm{EP}}}.
$$

These formulas expose the choice: gather a parameter volume proportional to $EF$, or
move an activation volume proportional to $Tk$. They are not complete time models;
the exact weight traffic depends on logical rules, gather reuse, and whether backward
regathers the weights. FSDP can overlap gathers, while MoE dispatch is tied to runtime
routes. Remat can also repeat either work.

On an eight-GPU node, compare meshes at fixed device count:

```text
FSDP 8 x EP 1
FSDP 4 x EP 2
FSDP 2 x EP 4
FSDP 1 x EP 8
```

The current Mixtral case study uses `FSDP 1 x EP 8` as its baseline to beat. That is a
test hypothesis, not an MI355X result. Verify each mesh in HLO: the intended sparse path
should show expert ownership and activation dispatch, while an unintended path may
show large expert-weight AllGathers.

MaxText v26.6 also exposes `shard_exp_on_fsdp` and
`moe_fsdp_use_two_stage_all_gather`. These alter the weight-sharding algorithm and
belong in a separately controlled sweep.

### EP versus TP

Tensor parallelism shards each expert matrix. It can reduce the weight and GEMM work
per device, but adds collectives around expert MLP projections and makes already-small
GEMM dimensions smaller. EP and TP also compete for the same eight intra-node devices:

$$
P_{\text{EP}}P_{\text{TP}} \le 8
$$

when no other intra-node axis consumes devices.

Use EP first when complete experts fit individually and the primary memory problem is
the number of experts. Add TP when one expert is itself too large or when measured
expert GEMMs remain large enough after sharding. Do not choose TP from total parameter
count alone.

### One-node placement

Keep the EP axis inside the MI355X full mesh. Extending EP across nodes sends
data-dependent activation traffic through the scale-out network for every routed layer.
The book has no MI355X multi-node MoE measurements. Data parallelism, FSDP, or
pipeline parallelism across nodes with `dcn_expert_parallelism=1` remain untested
hypotheses rather than recommendations.

Within one node:

- set `ici_expert_parallelism` to the intended degree;
- keep `dcn_expert_parallelism=1`;
- check `activation_batch_moe` and expert weight shardings;
- inspect the actual replica groups and collective shapes;
- confirm that tokens/s/GPU, HBM, and the four diagnostics improve together.

## Rematerialization and Custom VJPs

Block rematerialization can recompute more than the expert GEMMs. If the remat boundary
encloses the whole routed layer, backward may repeat:

- the gate projection and top-k;
- token repetition and sorting;
- dispatch communication;
- local expert computation;
- combine communication.

The memory saved must therefore be compared with issued FLOPs and repeated collective
time. A dense MLP remat estimate is not sufficient for an MoE layer.

MaxText names the expert projection outputs `moe_mlpwi_0`, `moe_mlpwi_1`, and
`moe_mlpwo` for checkpoint policies. Policies such as
`save_dot_with_context_except_mlp` intentionally do not save MLP projections. Do not
infer the MoE recompute boundary from the policy name. Compile one step and compare the
forward and backward HLO or trace.

For the sparse path, `use_custom_sort_vjp=true` is the v26.6 default. The custom VJP
implements the backward of a sort as the inverse permutation:

```python
@jax.custom_vjp
def sort_activations(inputs, sort_indices):
    return inputs[sort_indices]

def backward(sort_indices, grads):
    return grads[jnp.argsort(sort_indices)], None
```

This avoids an inefficient gather gradient that can lower to scatter-add. It does not
make the router differentiable through top-k, eliminate dispatch communication, or
prevent the routed layer from being rematerialized.

The custom-sort compiler delta is narrow. With the generic VJP, the transpose of
`inputs[sort_indices]` is a gather backward that can become a `scatter` or
`scatter-add`. With the custom VJP, backward computes the inverse permutation with
`argsort` and gathers gradients in that order. Preserve three evidence layers for the
comparison: the backward jaxpr (`scatter-add` versus `argsort` plus gather), optimized
HLO (`scatter` versus inverse-sort/permutation operations), and the trace (generic
scatter kernels and time versus inverse-sort and gather kernels and time). This is a
backward-only delta; the forward route should remain unchanged. Use
[Chapter 2]({{ '/pages/2-lowering-jax-jit-on-rocm' | relative_url }}#reading-a-compiler-delta) to locate the
compiler boundary and [Appendix D]({{ '/pages/d-profiler-and-hlo-cookbook' | relative_url }}#feature-comparison-bundles)
to capture the HLO and trace evidence.

Test the custom VJP in two ways:

1. compare loss and gradients with `use_custom_sort_vjp=false` on a small deterministic
   model;
2. compare backward HLO and trace, looking for the removed scatter and the replacement
   inverse sort.

If a finite ragged buffer drops assignments, include that exact condition in the
correctness test. A permutation proof for the dropless expanded token list does not by
itself prove a truncated route.

## Four Required Diagnostics

Log these by layer. An aggregate over all MoE layers can hide one dead expert or one
poorly placed layer.

### 1. Tokens per expert

Compute a histogram from the selected expert indices:

```python
counts = jnp.bincount(
    selected_experts.reshape(-1),
    length=num_experts,
)
```

Record the mean, maximum, maximum divided by mean, coefficient of variation, and empty
expert count. For EP, also sum counts by destination shard. This distinguishes
per-expert imbalance from device imbalance.

### 2. Drop and padding rates

For fixed capacity, record:

- selected assignments;
- retained assignments;
- assignment drop rate;
- fraction of tokens losing at least one assignment;
- occupied slots divided by allocated slots.

For the sparse path, record actual group sizes, receive-buffer occupancy, and any
truncation caused by positive `ragged_buffer_factor`. For dense-padded lowering, also
derive issued padded rows from the lowered GEMM shapes.

### 3. Expert GEMM efficiency

For each expert kernel, retain:

- kernel identity;
- dtype;
- group or padded shapes;
- useful FLOPs;
- issued FLOPs;
- kernel duration.

Compare with a dense GEMM that has the same dtype and comparable dimensions. Substitute
the resulting $\eta_d$, $\eta_p$, or $\eta_g$ into the win condition. A single
whole-step MFU cannot tell whether sparse routing saved arithmetic or merely moved time
into sorting and collectives.

### 4. Exposed AllToAll time

Record dispatch and combine duration, bytes, participant count, and overlap with
compute. Report exposed collective time on the critical path, not only the sum of
kernel durations. Confirm whether the trace contains ragged AllToAll, ordinary
AllToAll, or the ring-of-experts AllGather and ReduceScatter pair.

These diagnostics accompany the primary outcome:

```text
tokens/s/GPU at fixed model, tokens, dtype, mesh, remat, and routing input
```

If any of those invariants change, label the comparison accordingly.

## Decision Procedure

1. **Freeze the software and hardware manifest.** Record MI355X partition mode, ROCm,
   JAX and jaxlib, MaxText commit, container, effective `XLA_FLAGS`, and kernel-library
   versions.

2. **Account for the model.** Write down $D$, $F$, $E$, $k$, shared experts, MoE-layer
   count, total parameters, activated parameters, and expert-state bytes.

3. **Choose the minimum memory sharding.** Determine whether complete experts fit.
   Prefer intra-node EP for the expert-count problem; add TP only when an individual
   expert or its GEMM requires it.

4. **Start with an auditable baseline.** For the current Mixtral study this is BF16,
   FSDP 1, EP 8, fixed-capacity one-hot, `capacity_factor=1.0`, and
   `moe_dispatch_no_expert_sharding=true`. It is a baseline, not a recommendation.

5. **Make routing numerics explicit.** Decide whether to enable
   `float32_gate_logits`, and report `float32_weight_sum`. Configure either a justified
   auxiliary-loss weight or a positive DeepSeek bias update rate. Do not assume the
   base defaults balance the load.

6. **Choose model semantics before kernels.** Decide whether dropping is allowed. If
   not, use a worst-case ragged buffer or enough fixed capacity to make the observed
   drop rate zero, then verify it.

7. **Run matched expert-path comparisons.** Dense masked is the correctness reference.
   Compare the FP16 dense-padded and grouped-GEMM sparse arms directly. Compare
   fixed-capacity BF16 only with its dtype and drop-rate caveats stated.

8. **Apply the win condition.** Use measured kernel efficiencies and routing time. Do
   not select grouped GEMM only because it performs fewer model FLOPs.

9. **Sweep FSDP and EP at fixed device count.** Verify expert weight gathers, token
   collectives, and activation shardings in HLO before interpreting throughput.

10. **Choose remat after the route is visible.** Check whether backward repeats routing
    and collectives. Validate the custom sort VJP separately.

11. **Prove the kernel and collective path.** A config value is an intention. The HLO,
    custom-call target, kernel name, and trace are the evidence. Use
    [Chapter 7]({{ '/pages/7-a-map-of-kernel-backends-on-jax' | relative_url }}) for kernel reachability and
    [Chapter 9]({{ '/pages/9-compiler-runtime-and-rccl-controls' | relative_url }}) for the flags.

12. **Publish the four diagnostics with tokens/s/GPU.** A result without routing
    histogram, drop or padding rate, expert efficiency, and exposed collective time is
    not sufficient to generalize.

The current control surface can be summarized as follows. **[source]**

| Decision | Primary field | What it buys | Main cost or risk |
|---|---|---|---|
| Gate projection in fp32 | `float32_gate_logits` | More stable ranking | Small extra gate cost |
| Expert sum in fp32 | `float32_weight_sum` | More stable combine | Cast and fp32 reduction |
| Auxiliary balance | `load_balance_loss_weight` | Lower imbalance | Changes objective |
| DeepSeek bias balance | `routed_bias_update_rate` | Balance without auxiliary gradient | Model-specific update |
| Dense masked | `sparse_matmul=false`, `capacity_factor=-1` | Regular GEMMs and simple reference | $E/k$ expert FLOPs |
| Fixed capacity | `capacity_factor>0` | Static shapes | Padding and dropping |
| Sparse route | `sparse_matmul=true`, `megablox=false` | Activated expert work | Sort, ragged buffers, kernel requirement |
| Sparse buffer | `ragged_buffer_factor` | Memory versus drop control | Worst-case allocation or truncation |
| Custom sort backward | `use_custom_sort_vjp=true` | Avoids scatter-add backward | Must validate permutation path |
| Expert mesh | `ici_expert_parallelism` | Shards expert state | Two routed collectives per layer |
| Dense-path EP layout | `moe_dispatch_no_expert_sharding` | Keeps dense dispatch expert-parallel | Path-specific sharding change |
| Ring of experts | `use_ring_of_experts` | Different communication and overlap opportunity | AllGather volume and separate algorithm |

## Current Status and Blockers

The following statements are current for this project's v26.6 source and scripts, not
permanent ROCm guarantees:

- MaxText has dense masked, fixed-capacity one-hot, and sparse
  `jax.lax.ragged_dot` training routes. **[source]**
- The MI355X Mixtral experiment splits the sparse route into FP16 dense-padded and
  grouped-GEMM XLA lowerings with one flag. **[source]**
- Both sparse arms use ragged AllToAll with a one-shot kernel flag carrying an
  `unsupported` prefix and with the RCCL device barrier disabled. **[source]**
- The base config defaults to sparse Megablox, but the MI355X experiment selects
  `megablox=false`; default does not establish ROCm reachability. **[source]**
- The local JAX-AITER fused MXFP4 MoE route is forward-only and cannot support the
  training comparison without a VJP. **[source]**
- No v26.6 MI355X timings, peak-HBM records, routing histograms, drop rates, or trace
  artifacts have been recorded for the Mixtral expert-path sweep. The four paths
  cannot yet be ranked on MI355X.
- The FP16 restriction in the current grouped-GEMM experiment prevents a direct
  like-for-like ranking against the BF16 fixed-capacity baseline.
- The performance and correctness of the one-shot ragged AllToAll lowering on MI355X
  remain to be established under the book's protocol.
- Multi-node EP remains future work.

These blockers are intentionally narrow. The model accounting, required diagnostics,
config mapping, and acceptance test are usable now.
[Chapter 12]({{ '/pages/12-mixtral-8x22b-sharding-meshes-and-moe-optimizations' | relative_url }}) will fill the result slots
once the MI355X artifacts exist.

**Recommendation status: BLOCKED.** Fixed-capacity one-hot, dense masked,
dense-padded, and ragged GroupedGEMM remain experiment arms. None is the MI355X
default until Chapter 12 records tokens/s/GPU, peak HBM, routing correctness,
expert efficiency, and collective exposure. The fixed-capacity BF16 route is the
predeclared fallback. Retest after changes to model routing, MaxText, XLA, RCCL,
or the grouped-GEMM backend.

## References

- [How To Scale Your Model: Transformers](https://jax-ml.github.io/scaling-book/transformers/)
  (Google DeepMind). Generic dense and MoE parameter and FLOP accounting.
- [How To Scale Your Model: Sharding](https://jax-ml.github.io/scaling-book/sharding/)
  (Google DeepMind). Sharded-matrix and collective derivations.
- [How To Scale Your Model: Training](https://jax-ml.github.io/scaling-book/training/)
  (Google DeepMind). General training-parallelism treatment.
- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)
  (Shazeer et al., 2017). Sparsely gated expert layers.
- [GShard](https://arxiv.org/abs/2006.16668)
  (Lepikhin et al., 2020). Fixed-capacity expert dispatch and automatic sharding.
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
  (Fedus et al., 2021). Top-1 routing and the auxiliary load-balancing loss.
- [ST-MoE](https://arxiv.org/abs/2202.08906)
  (Zoph et al., 2022). Router stability and precision.
- [MegaBlocks](https://arxiv.org/abs/2211.15841)
  (Gale et al., 2022). Dropless MoE with block-sparse expert computation.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
  (Dai et al., 2024). Fine-grained routed experts and shared experts.
- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)
  (Wang et al., 2024). The routing-bias update used by DeepSeek-style models.
- [MaxText MoE configuration](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/reference/core_concepts/moe_configuration.md)
  (Google). Current upstream field definitions and dispatch decision tree.
- [JAX `ragged_dot`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html)
  and [JAX `ragged_all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_all_to_all.html).
  The JAX primitives used by MaxText's sparse route.
