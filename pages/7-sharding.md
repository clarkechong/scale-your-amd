---
layout: distill
title: "From JAX Shardings to a Training Mesh"
description: "How named JAX shardings become collectives, how to place training parallelism on one eight-GPU MI355X node, and how to verify the compiled result."
date: 2026-09-13

section_number: 7

previous_section_url: "/pages/6-memory"
previous_section_name: "Chapter 6: Making the Model Fit"

next_section_url: "/pages/8-kernels"
next_section_name: "Chapter 8: Kernels Reachable from JAX"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Prerequisites and Scope
  - name: A Compact Sharding Notation
  - name: The Four Core Collectives
  - name: The Four Matmul Cases
  - name: Automatic, Constrained, and Manual Partitioning
  - name: Verify the Compiled Mesh
  - name: MI355X and RCCL
  - name: Training Parallelism
    subsections:
      - name: Data Parallelism
      - name: Fully Sharded Data Parallelism
      - name: Tensor and Sequence Parallelism
      - name: Context Parallelism
      - name: Pipeline Parallelism
      - name: Expert Parallelism
  - name: MaxText Sharding Surface
  - name: Place an Eight-GPU Mesh
  - name: Combining and Overlap
  - name: Mesh Decision Procedure
  - name: References
---

## Prerequisites and Scope

Read the JAX Scaling Book chapters
[Sharded Matrices and How to Multiply Them](https://jax-ml.github.io/scaling-book/sharding/)
and
[How to Parallelize a Transformer for Training](https://jax-ml.github.io/scaling-book/training/)
first. They derive the general rules. This chapter keeps only the notation and decisions
needed to configure JAX and MaxText on MI355X.

[Chapter 1]({{ '/pages/1-hardware' | relative_url }}) supplies the topology, and
[Chapter 6]({{ '/pages/6-memory' | relative_url }}) supplies the state and activation
ledgers. A mesh is acceptable only if it fits in memory, preserves tensor divisibility,
and puts its frequent collectives on suitable links.

The hardware discussion here covers one eight-GPU MI355X node. Multi-node placement is
analytical until the work in
[Chapter 14]({{ '/pages/14-multinode' | relative_url }}) is completed. There is no
MI355X RCCL benchmark bundle in the current experiment repositories, so this chapter
does not reuse the archived MI300X bandwidth or overlap measurements.

Evidence labels have their literal meanings:

- **[source]** describes the checked-in experiment or MaxText configuration.
- **[cited]** comes from linked JAX or AMD documentation.
- **[analytical]** follows from array shapes, byte counts, or topology.

**BLOCKED** marks a result slot that still needs an MI355X artifact.

## A Compact Sharding Notation

A device mesh is a named grid. An array sharding maps array dimensions to mesh axes.
For a global array `A[I, J]`:

- `A[I, J]` is replicated over the mesh axis under discussion.
- `A[I_X, J]` shards rows over mesh axis `X`.
- `A[I, J_Y]` shards columns over mesh axis `Y`.
- `A[I_X, J_Y]` uses both axes. If `|X| = 2` and `|Y| = 4`, each device holds
  `[I/2, J/4]`.
- `A[I, J]{U_X}` is full-shaped on each device but contains an unreduced partial sum
  over `X`. It is not the final value.

The same layouts in JAX are:

```python
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(
    np.asarray(jax.devices()).reshape(2, 4),
    axis_names=("fsdp", "tensor"),
)

row_and_column = NamedSharding(mesh, P("fsdp", "tensor"))
rows_only = NamedSharding(mesh, P("fsdp", None))
replicated = NamedSharding(mesh, P(None, None))
```

`jax.Array.shape` is the global shape. `array.addressable_shards` exposes the
process-local pieces and their global indices. Replication is also relative: an array
may be sharded over `tensor` and replicated over `fsdp`.

The names describe roles, not hardware. `tensor` does not become an intra-node axis
because of its name. The device ordering used to construct `Mesh` decides which
physical devices belong to each axis.

## The Four Core Collectives

Let `n` be the number of devices in one collective group and `V` the bytes in the full
logical array named below. The payloads are ideal per-device traffic terms. Startup,
protocol, channel scheduling, contention, and arrival skew are excluded.

| Collective | Layout change | Ideal bytes moved per device |
|---|---|---:|
| AllGather | `A[I_X, J] -> A[I, J]` | `V(n-1)/n` |
| ReduceScatter | `A[I, J]{U_X} -> A[I_X, J]` | `V(n-1)/n` |
| AllReduce | `A[I, J]{U_X} -> A[I, J]` | `2V(n-1)/n` |
| AllToAll | `A[I_X, J] -> A[I, J_X]` | `V(n-1)/n²` |

For AllToAll, each device begins with `V/n` local bytes and sends the fraction
`(n-1)/n` elsewhere. If a tool reports a local send-buffer size `L`, use
`L(n-1)/n` instead.

The operations serve different layout changes:

- **AllGather** removes a sharding. FSDP gathers a weight shard before a layer uses it.
- **ReduceScatter** resolves partial sums while leaving the result sharded. It is the
  useful half of an AllReduce when the consumer does not need a replicated result.
- **AllReduce** resolves partial sums and replicates the answer. Data-parallel
  gradients and row-parallel tensor outputs commonly use it.
- **AllToAll** moves a sharding from one dimension to another. Expert dispatch moves a
  token-owned layout to an expert-owned layout, then the combine reverses it.

AllGather and ReduceScatter are transposes under reverse-mode differentiation.
AllReduce is self-transpose. This is why a forward layout decision can create a
different collective in the backward pass.

A usable time model needs the measured curve, not a peak number:

```text
t_collective = latency(kind, n)
             + payload / achieved_bandwidth(kind, payload, n, version)
```

Small tensor-parallel messages can be latency-bound while a large gradient collective
is bandwidth-bound. Treating both at one asymptotic bandwidth gives the wrong mesh
decision.

## The Four Matmul Cases

Write `C[I, K] = A[I, J] @ B[J, K]`, where `J` is the contracting dimension. Every
distributed dense matmul reduces to one of four cases.

| Case | Example over `X` | Required communication |
|---|---|---|
| 1. No contracting split | `A[I_X,J] @ B[J,K] -> C[I_X,K]` | None |
| 2. One contracting split | `A[I,J_X] @ B[J,K]` | AllGather the split operand, or change the other layout |
| 3. Matching contracting splits | `A[I,J_X] @ B[J_X,K] -> C[I,K]{U_X}` | Local matmul, then AllReduce or ReduceScatter |
| 4. Same mesh axis on two output dimensions | `A[I_X,J] @ B[J,K_X]` | AllGather one operand before the matmul |

Case 1 includes column parallelism:
`A[I,J] @ B[J,K_X] -> C[I,K_X]` is local and communication-free.

Case 2 is a layout mismatch. A device does not have the full contracting dimension of
one operand. Gathering that operand is the usual repair, but changing the other
operand to the same contracting split turns it into case 3. Compare the resulting
weight-sized gather with the output-sized reduction before choosing.

Case 3 produces partial sums. An elementwise operation can sometimes run on those
partials, but nonlinear operations generally cannot. Resolve the reduction before the
first consumer that needs actual values.

Case 4 tries to use `X` twice in `C[I_X, K_X]`. One mesh axis cannot independently
partition both output dimensions in an ordinary `PartitionSpec`; gather one operand
or assign the dimensions to different mesh axes.

These rules apply to forward and backward matmuls. Before estimating a strategy, write
the shardings for the activation, weight, output, input gradient, and weight gradient.

## Automatic, Constrained, and Manual Partitioning

Current JAX documents three levels of control.

### Automatic global view

The program uses global array shapes. Input and output shardings, logical rules, and
occasional constraints guide Shardy. The partitioner propagates layouts and inserts
collectives.

The March 2026 JAX migration guide describes Shardy as the post-migration
partitioner. **[cited]** The inspected MaxText tree still exposes:

```yaml
shardy: true
shard_mode: "auto"
```

The source says `shardy` selects the compatibility path in that version. The
declared Mixtral environment still needs a post-propagation dump to prove which
partitioner ran. In `auto` mode, MaxText's
sharding helper lowers logical names to a `NamedSharding` and applies
`jax.lax.with_sharding_constraint`. **[source]**

A constraint is a request for an intermediate layout. It can cause a resharding
collective, so use it where a layout is ambiguous or known to be poor, then inspect
the HLO. Scattering constraints throughout the model makes propagation harder to
reason about.

### Explicit global view

With explicit mesh axes, shardings are part of traced JAX types and can be inspected
with `jax.typeof`. Operations propagate those shardings or reject ambiguous rules.
Current MaxText selects this path with:

```yaml
shard_mode: "explicit"
```

In the pinned tree, the same helper uses `jax.sharding.reshard` in explicit mode
instead of applying a soft constraint. **[source]** Explicit mode improves
auditability, but it does not mean that collectives are hand-written.

### Manual per-device view

`jax.shard_map` exposes local shards and makes communication explicit:

```python
from functools import partial
import jax.numpy as jnp

@partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=(P(None, "tensor"), P("tensor", None)),
    out_specs=P(None, None),
)
def row_parallel_matmul(x_local, w_local):
    partial_y = jnp.dot(x_local, w_local)
    return jax.lax.psum(partial_y, "tensor")
```

The `psum` is the AllReduce debt from matmul case 3. Use `shard_map` for schedules the
partitioner cannot infer or when the collective order itself is the algorithm. Ragged
expert routing and collective matmuls are typical examples. It can be composed with
automatic partitioning on other mesh axes.

`shard_map` is not inherently faster. If manual and automatic programs express the
same local matmul, collective, and schedule, they should lower to equivalent work.
Compare their optimized HLO rather than assuming an API-level speed difference.

## Verify the Compiled Mesh

The configuration records intent. Optimized HLO records what the compiler produced.
Dump both the Shardy-stage and final HLO:

```bash
XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=shardy" \
  python your_program.py

rg -n "all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute|replica_groups" \
  /tmp/hlo
```

For MaxText, also retain the resolved configuration and set `dump_hlo=true` when the
pinned runner supports it.

A collective may appear as one operation or as asynchronous start/done operations:

{% raw %}
```text
all-reduce-start(...),
  replica_groups={{0,1,2,3,4,5,6,7}},
  use_global_device_ids=true
all-reduce-done(...)
```
{% endraw %}

Read six fields:

1. **Kind.** An unexpected AllGather often means an operand reached matmul case 2.
2. **Shape and dtype.** These give message bytes and distinguish weight-, activation-,
   and gradient-shaped traffic.
3. **Replica-group size.** This is the degree of the mesh axis used by the collective.
4. **Group membership.** It identifies which logical devices communicate together.
5. **Global device IDs and assignment.** Map IDs back to hosts and local GPU indices
   before calling a group intra- or inter-node.
6. **Start/done distance.** Compute scheduled between them is an overlap opportunity,
   not proof of achieved overlap.

On eight devices:

{% raw %}
```text
{{0,1,2,3,4,5,6,7}}       one group of eight
{{0,1,2,3},{4,5,6,7}}     two groups of four
{{0,4},{1,5},{2,6},{3,7}} four strided groups of two
```
{% endraw %}

Contiguous and strided groups reveal the logical mesh traversal order. On one full-mesh
node, both remain one-hop groups. Across nodes, neither notation proves link locality
without the runtime device assignment.

Verification is complete only when:

- every expected strategy has the expected collective kind and group size;
- no large unexpected resharding collective appears;
- parameter and activation shardings divide their tensor dimensions;
- the intended local groups map to the eight GPUs in one node;
- the trace attributes the corresponding kernels to RCCL; and
- tokens/s/GPU is measured in a separate, unprofiled timing run.

## MI355X and RCCL

An eight-GPU MI355X node is a one-hop full mesh: every GPU has a direct xGMI
connection to the other seven. Each link is 76.8 GB/s per direction; seven links give
a specification ceiling of 537.6 GB/s aggregate egress per GPU if traffic uses all
links concurrently. AMD also publishes 153.6 GB/s bidirectional per link and
1,075.2 GB/s bidirectional aggregate; do not use those bidirectional values as
one-way payload bandwidth. **[cited]**

The topology has three consequences:

- eight participants expose all seven peer links;
- two- and four-participant groups expose only one and three peer links per GPU; and
- the scale-up domain ends at eight GPUs.

AMD recommends using all eight GPUs for best collective performance on this topology.
**[cited]** That does not establish achieved bandwidth for a particular JAX release,
RCCL version, collective, message size, or channel count. RCCL chooses the protocol and
schedule; the full mesh only sets the available paths.

The archived 320 GB/s model and overlap tables were measured on MI300X/gfx942. They
are not MI355X results. This chapter therefore uses:

- xGMI values only as specification ceilings;
- `payload / measured_time` only after an MI355X sweep exists; and
- symbolic `beta_rccl(kind, bytes, participants)` in strategy thresholds.

The missing sweep should cover AllGather, ReduceScatter, AllReduce, and AllToAll;
participant counts 2, 4, and 8; and payloads from the latency regime through the
largest case-study messages. Record latency, algorithm bandwidth, bus bandwidth,
RCCL/ROCm versions, SPX or partition mode, in-place status, channel settings, and
correctness. Run standalone bandwidth and in-step overlap as separate experiments.
**BLOCKED**

The Mixtral ragged route adds these experimental XLA controls:

```text
--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false
--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=true
```

They describe the planned compatibility path, not a measured MI355X result.
**[source]** The `unsupported` name is a warning to pin the build, verify the
HLO, check token-routing correctness, and keep the dense-padded fallback.

## Training Parallelism

The degree of every active mesh axis multiplies to the total device count. On one
eight-GPU run:

```text
data × fsdp × tensor × tensor_sequence × context × stage × expert = 8
```

Size-one and specialized axes are omitted above. Two strategies can share a physical
role only if their implementation intentionally does so; setting two degrees to eight
does not create sixteen virtual devices.

### Data Parallelism

```text
X[B_data, T, D]    W[...] replicated
```

- Shards independent examples or tokens; forward and backward matmuls are local.
- Replicates model state, so it does not make a too-large model fit.
- AllReduces gradients once they are produced. Buckets can overlap with later
  backward compute.
- MaxText fields: `ici_data_parallelism` and `dcn_data_parallelism`.

The communication threshold is where backward compute available during one gradient
bucket exceeds that bucket's measured AllReduce time. **[analytical]** A global batch
or convergence limit can cap data parallelism before bandwidth does.

### Fully Sharded Data Parallelism

```text
X[B_fsdp, T, D]    W[..., P_fsdp]
```

- Shards parameters, gradients, master weights, and optimizer state when their
  logical rules include `fsdp`.
- AllGathers each needed weight layout and ReduceScatters gradients.
- Reduces persistent state roughly with the FSDP degree, subject to replicated leaves,
  workspaces, and collective buffers.
- MaxText fields: `ici_fsdp_parallelism`, `dcn_fsdp_parallelism`, and the specialized
  `fsdp_transpose` fields.

Weight AllGather is on the forward critical path unless prefetched. Use the memory
ledger to choose the minimum FSDP degree that fits, then test whether its gather and
reduce-scatter are exposed.

The checked-in Llama 70B recipe uses `ici_fsdp_parallelism: 8` and leaves the other
main axes at one. **[source]**

### Tensor and Sequence Parallelism

Megatron-style tensor parallelism pairs a column-parallel projection with a
row-parallel projection:

```text
X[B,T,D] @ W_in[D,F_tensor] -> H[B,T,F_tensor]       # case 1
H[B,T,F_tensor] @ W_out[F_tensor,D] -> Y[B,T,D]{U_tensor}
```

The second matmul is case 3 and resolves an activation-sized partial sum. This traffic
is frequent and commonly on the critical path, so tensor parallelism belongs inside
the eight-GPU scale-up domain.

Sequence parallelism is overloaded:

- Megatron sequence parallelism shards norms and residual activations over the
  tensor-parallel group, replacing replicated activation phases with paired
  ReduceScatter/AllGather operations.
- MaxText exposes `tensor_sequence` separately from `tensor`, including
  `ici_tensor_sequence_parallelism`.
- MaxText also defines plain `ici_sequence_parallelism` and
  `dcn_sequence_parallelism` fields, but `sequence` is absent from the pinned default
  `mesh_axes`. Changing one of those fields alone therefore does not add a mesh axis.
  Do not assume plain `sequence` is interchangeable with `tensor_sequence` or
  `context`; a custom mesh needs matching logical rules and HLO verification.

Tensor parallelism reduces weight and selected activation memory, but it also shrinks
local GEMMs and adds per-layer collectives. Its usable degree is constrained by model
dimensions, head counts, local GEMM efficiency, and the measured bandwidth at the
actual activation message size.

The current MI355X case-study configs set tensor, tensor-sequence, and sequence degrees
to one. Their performance on MI355X is therefore **BLOCKED**, not inferred from the
archived MI300X sweep.

### Context Parallelism

```text
X[B, T_context, D]
```

Context parallelism splits tokens from the same sequence. Pointwise MLP work remains
local, but attention must exchange or circulate key/value blocks and combine
partial-softmax state. It addresses long-context activation memory; ordinary data
parallelism is cheaper when independent sequences already fill the devices.

MaxText fields are `ici_context_parallelism` and `dcn_context_parallelism`; the
logical rules place activation length on `context`. The current case configs leave
both at one. A usable MI355X threshold needs the selected attention algorithm,
key/value bytes, overlap schedule, and measured communication curve.

### Pipeline Parallelism

Pipeline parallelism assigns layer ranges to `stage` and passes activations and
gradients between adjacent stages. Its principal costs are stage imbalance and the
fill/drain bubble, not a group collective:

```text
bubble fraction = (stages - 1) / (microbatches + stages - 1)
```

This is an ideal 1F1B-style bound; schedule and transfer overhead add to it.
**[analytical]** MaxText uses `ici_pipeline_parallelism`,
`dcn_pipeline_parallelism`, and `num_pipeline_microbatches`. Pipeline parallelism is
usually a later memory lever because it complicates execution and makes each local
stage smaller.

### Expert Parallelism

```text
W[E_expert, D, F]    tokens[token_owner, D]
```

Expert parallelism shards expert weights and changes token ownership:

1. route and sort tokens;
2. AllToAll from token owners to expert owners;
3. run local expert GEMMs; and
4. AllToAll outputs back before unpermuting.

It divides expert-weight memory but adds routing buffers, imbalance, and two layout
changes per MoE layer. Keep the expert axis inside the eight-GPU full mesh when
possible. Tensor and expert parallelism compete for that same fast domain.

MaxText fields are `ici_expert_parallelism` and `dcn_expert_parallelism`. The Mixtral
plan fixes eight experts and tests:

```text
FSDP-8 × EP-1
FSDP-4 × EP-2
FSDP-2 × EP-4
FSDP-1 × EP-8
```

The checked-in baseline is BF16, FSDP-1, EP-8, fixed-capacity one-hot execution.
These are experiment definitions; no v26.6 result artifacts are present.
**[source] BLOCKED**

## MaxText Sharding Surface

MaxText separates three naming levels:

1. `mesh_axes` lists physical mesh roles.
2. Model variables and activations carry logical names such as `mlp`,
   `activation_batch`, `activation_length`, and `exp`.
3. `logical_axis_rules` maps those logical names to mesh roles.

The pinned base configuration includes:

```yaml
mesh_axes:
  [diloco, data, stage, fsdp, fsdp_transpose, context,
   context_autoregressive, tensor, tensor_sequence, expert, autoregressive]

logical_axis_rules:
  - [activation_batch, [data, fsdp, fsdp_transpose, expert]]
  - [activation_length, [context]]
  - [activation_mlp, [tensor, tensor_sequence]]
  - [mlp, [fsdp_transpose, tensor, tensor_sequence, autoregressive]]
  - [exp, expert]
  - [layers, stage]
```

This excerpt is illustrative. The effective rules vary by model and custom mesh file;
save the fully resolved configuration for every run.

| Configuration surface | What to check |
|---|---|
| `shardy` | Compatibility path requested by the inspected MaxText source; runtime proof still required |
| `shard_mode` | `auto` constraints or explicit sharding propagation |
| `ici_*_parallelism` | Within-slice degree; these axes consume the eight local GPUs in the current studies |
| `dcn_*_parallelism` | Cross-slice degree; all are one in the current one-node studies |
| `mesh_axes` | Axis order and names used to construct the JAX mesh |
| `logical_axis_rules` | Which tensor dimensions each role actually shards |
| `data_sharding` | Physical layout of input batches |
| `debug_sharding` | Parameter sharding diagnostics |
| `sharding_tolerance` | Allowed fraction of parameters not sharded as requested |
| `check_vma` | Optional validation for supported auto-mode EP/FSDP configurations |

MaxText requires DCN axes to precede ICI axes in its data-sharding list. Do not edit
axis order as cosmetic YAML. The product of ICI degrees must match devices per slice,
and the product of DCN degrees must match slices, apart from one supported `-1`
auto-filled degree.

## Place an Eight-GPU Mesh

All eight GPUs are one hop apart, so there is no faster pair hidden inside the node.
Placement still matters because a degree-two group can use one peer link per GPU while
a degree-eight group can involve seven peers. The group with more frequent or larger
critical-path traffic should generally receive the larger local axis, subject to
memory and GEMM efficiency.

Use these defaults:

- Put TP inside the node. Its activation collectives recur within each layer.
- Put EP inside the node. Every MoE layer dispatches and combines tokens.
- Put CP inside the node when long context requires it.
- Use DP or FSDP for the remaining devices. Across nodes, these are usually better
  candidates than TP or EP, but FSDP weight gathers still need measurement.
- Use PP across a slow boundary only after checking its microbatch bubble and stage
  balance.

On one node, current recipes instantiate two useful endpoints:

```yaml
# Llama 70B: maximize state sharding
ici_fsdp_parallelism: 8
ici_tensor_parallelism: 1
ici_expert_parallelism: 1

# Mixtral baseline: one whole expert per device
ici_fsdp_parallelism: 1
ici_tensor_parallelism: 1
ici_expert_parallelism: 8
```

The intermediate Mixtral cells trade expert ownership for FSDP state sharding. They
are a controlled way to determine whether weight collectives, token movement, memory,
or expert-kernel shape is the active constraint. Do not predict the winner from
topology alone.

## Combining and Overlap

Asynchronous collectives create an interval between start and done in which
independent compute may run. Three conditions are required:

1. the collective has an asynchronous lowering;
2. useful independent work is ready; and
3. the runtime can progress communication while kernels run.

The current Llama and Mixtral flag files enable the latency-hiding scheduler and set
the AllGather, ReduceScatter, and AllReduce combine thresholds to 8 GiB:

```text
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_all_gather_combine_threshold_bytes=8589934592
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
--xla_gpu_all_reduce_combine_threshold_bytes=8589934592
```

These values are **[source]**, not evidence that communication is hidden.

Combining and overlap can work against each other. Combining reduces launch latency
and can move a transfer into the bandwidth-efficient message regime, but one large
collective may have less independent compute around it and require a larger temporary
buffer. Splitting creates more scheduling opportunities but pays more startup cost.
Pipelined collective matmuls go further by exchanging a chunk while multiplying the
previous chunk; use `shard_map` when that schedule must be explicit.

Evaluate the trade in this order:

1. dump HLO and count collective operations, payloads, and start/done pairs;
2. measure the MI355X latency/bandwidth curve at those payloads;
3. inspect the trace for compute concurrent with RCCL kernels;
4. compute exposed collective time rather than total collective duration;
5. check peak memory and recompilation; and
6. compare tokens/s/GPU in an unprofiled run.

Change combine thresholds and latency hiding as separate experiment axes. An archived
MI300X result cannot select the MI355X setting.

## Mesh Decision Procedure

Use this procedure for each model and batch.

### 1. Write the constraints

Record:

- device and host count;
- parameter, optimizer, gradient, activation, and workspace bytes;
- global batch, sequence length, and gradient accumulation;
- model width, MLP width, head counts, expert count, and top-k;
- dimensions that each candidate axis must divide; and
- the largest acceptable pipeline bubble.

### 2. Choose axes required for memory

Use the Chapter 6 ledger.

- Add the minimum FSDP degree that makes persistent state fit.
- Add TP only if weight or activation sharding is still needed and local GEMMs remain
  viable.
- Add CP only when per-sequence activation memory requires splitting the context.
- For MoE, choose an EP divisor that gives valid expert ownership and enough memory.
- Add PP only if tensor sharding and rematerialization still do not produce a workable
  stage.

### 3. Add throughput axes

Use DP to consume devices only while per-device work remains efficient and the global
batch remains valid. Increase TP, CP, or EP for throughput only after pricing their
critical-path collective at the actual payload size.

### 4. Fit the device equation

For one node, the active ICI degrees must multiply to eight. Enumerate divisors rather
than guessing:

```text
(fsdp, expert) = (8,1), (4,2), (2,4), (1,8)
(fsdp, tensor) = (8,1), (4,2), (2,4), (1,8)
```

Reject candidates that fail tensor, head, expert, layer, or batch divisibility.

### 5. Place frequent communication locally

Keep TP, EP, and usually CP within the eight-GPU node. In a future multi-node mesh,
Chapter 14 will test DP, FSDP, and PP as candidate cross-node axes. Map global
device IDs to host/GPU coordinates before testing them. Axis names and reshape
order alone are insufficient proof, and this chapter does not select a cross-node
axis without those measurements.

### 6. List the collective messages

For each candidate, list every material collective:

```text
kind, dtype, global shape, local shape, bytes, group size, calls per layer/step,
critical-path or overlap candidate
```

Use the four matmul cases to derive the list. Estimate time with the matching MI355X
RCCL curve. Until that curve exists, keep the comparison symbolic or use specification
bandwidth only as a lower bound.

### 7. Compile before a long run

Inspect resolved MaxText rules, parameter shardings, optimized HLO, collective shapes,
replica groups, and device assignment. Unexpected communication invalidates the
prediction even if the YAML looks right.

### 8. Measure one change at a time

Hold model, batch, precision, remat, kernels, versions, and flags fixed while sweeping
one mesh factorization. Report tokens/s/GPU first, then step time, peak HBM, collective
payload and exposed duration, and kernel attribution. A mesh is selected only after it
passes correctness and memory checks.

**Recommendation status: BLOCKED.** The Llama 70B and Mixtral configs define
FSDP-8 and EP-8 endpoints, but no accepted MI355X RCCL curve or Mixtral mesh-sweep
bundle ranks them. Use those endpoints as experiment arms, not general defaults.
Retest after changes to XLA partitioning, RCCL, device ordering, model shape, or
message size.

## References

- [Sharded Matrices and How to Multiply Them](https://jax-ml.github.io/scaling-book/sharding/)
  (JAX Scaling Book). Full collective and four-case derivations.
- [How to Parallelize a Transformer for Training](https://jax-ml.github.io/scaling-book/training/)
  (JAX Scaling Book). DP, FSDP, TP, sequence, context, and pipeline cost models.
- [JAX distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).
  `Mesh`, `NamedSharding`, `PartitionSpec`, global arrays, and addressable shards.
- [JAX parallelism](https://docs.jax.dev/en/latest/parallel.html). Automatic, explicit,
  and manual parallelism.
- [Manual parallelism with `shard_map`](https://docs.jax.dev/en/latest/201/shard-map.html).
  Per-device programs and explicit collectives.
- [Shardy JAX migration](https://docs.jax.dev/en/latest/shardy_jax_migration.html).
  Current partitioner status and migration diagnostics.
- [AMD Instinct workload optimization](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html).
  MI355X capacity, xGMI specifications, topology, and RCCL guidance.
- [RCCL usage tips](https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/rccl-usage-tips.html).
  Versioned channel, partition-mode, affinity, and profiling guidance.
- [MaxText base configuration](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml).
  Mesh axes, logical rules, sharding modes, and parallelism fields. Use the experiment
  repository's pinned commit rather than assuming `main` matches a run.
