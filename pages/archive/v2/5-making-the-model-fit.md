---
layout: distill
title: "Making the Model Fit"
description: "Account for the peak memory of one optimizer update, then apply donation, rematerialization, accumulation, FSDP, sharded initialization, and offload in a measured order."
date: 2026-09-13

section_number: 5

previous_section_url: "/pages/4-training-in-mixed-precision"
previous_section_name: "Chapter 4: Training in Mixed Precision"

next_section_url: "/pages/6-jax-shardings-to-a-training-mesh"
next_section_name: "Chapter 6: Parallelism for Higher Throughput"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "Start with the Memory Peak"
  - name: "Persistent Training State"
    subsections:
      - name: "Llama 7B"
      - name: "Llama 70B under FSDP"
      - name: "Sparse Models"
  - name: "Activation Memory"
  - name: "Measure the Compiled Program"
  - name: "Donate Replaced State"
  - name: "Scan and Rematerialization"
    subsections:
      - name: "Scan in HLO"
      - name: "Rematerialization in HLO"
      - name: "The Policy Surface"
      - name: "Interactions That Change the Answer"
  - name: "Gradient Accumulation"
  - name: "FSDP and Sharded Initialization"
    subsections:
      - name: "Choose the minimum FSDP degree"
      - name: "Initialize directly into shards"
  - name: "Host Offload"
  - name: "A Memory Decision Procedure"
  - name: "Decision Table"
  - name: "References"
---

An out-of-memory failure occurs when concurrently live allocations exceed device
capacity. This can happen while initializing the optimizer, compiling the step,
materializing an attention score, gathering an FSDP weight, creating a
low-precision workspace, or returning a new state before the old state dies.

Separate two classes of memory before compiling:

1. persistent training state: parameters, optimizer moments, scale state, and
   counters;
2. live step state: gradients, saved activations, temporary buffers, collective
   buffers, library workspaces, and outputs.

Then compare both with XLA's buffer assignment and the allocator high-water mark.
The estimates explain the terms. The compiled program decides which terms overlap
in time.

This chapter uses the same evidence tags as Chapter 4. **[source]** identifies
checked-in code or configuration, **[measured]** requires a complete Appendix F
bundle, **[analytical]** means arithmetic not confirmed by a profile, and
**[cited]** means a named external source's result or behavior. Current
support is verified against MaxText v26.6, the ROCm MaxText MXFP4 branch at `b437942a`,
JAX/JAXLIB and ROCm PJRT/plugin 0.11.0, and 8x MI355X experiment configs, on
**13 September 2026**.

## Start with the Memory Peak

Define:

```text
P       total parameter count
P_local parameters resident on one GPU after sharding
b_w     bytes per stored parameter
b_g     bytes per stored gradient
b_mu    bytes per first Adam moment
b_nu    bytes per second Adam moment
Q       quantizer state and cached low-precision layouts
```

A first persistent-state estimate is:

```text
persistent bytes per GPU =
    P_local * (b_w + b_mu + b_nu) + Q + counters

live gradient bytes =
    P_grad_local * b_g
```

For the Llama experiments, parameters and both Adam moments are FP32, so the
persistent coefficient is 12 bytes per local parameter. One live FP32 gradient tree
adds four bytes per local parameter. In the pinned MaxText source, `mu_dtype`
controls the first moment and the second moment inherits `weight_dtype`; there is no
independent `nu_dtype`.

This formula is a steady-state estimate. It omits several possible peaks:

- a separate input checkpoint while parameters are being restored;
- the old and new state at a functional update boundary;
- FSDP-gathered working weights;
- FP8 history and scale collections;
- MX rowwise and columnwise layouts or weight caches;
- temporary cast, reduction, communication, and library workspaces;
- input batches, logits, executable code, and allocator fragmentation.

Do not solve those omissions by adding one arbitrary "safety factor." Name each missing
term, measure the compiled step, and reserve headroom for terms the compiler report
cannot see.

## Persistent Training State

### Llama 7B

The raw-JAX implementation has exactly 6,738,415,616 parameters. Its persistent
returned state stores parameters and both Adam moments in FP32:

```text
6,738,415,616 * 12 bytes = 75.31 GiB
```

One live FP32 gradient tree adds 25.10 GiB, bringing parameters, moments, and
gradients to 100.41 GiB during the relevant part of the step. These values are
**[analytical]** and match the roles in the
[Llama 7B source](https://github.com/clarkechong/llama7b-jax-fundamentals/tree/5f996a88).
It is not the peak. The model also produces FP32 logits; at batch 4, sequence 4096,
and vocabulary 32,000, that single tensor is about 1.95 GiB. Saved layer activations
and attention implementation decide whether the full step fits.

The train step donates both `params` and `opt_state`. Without donation, the update
boundary can require old and new state buffers at once. That is enough to turn a
100 GiB steady-state estimate into an OOM despite a 288 GB device.

### Llama 70B under FSDP

The MaxText Llama 2 70B dimensions imply 68.98 billion parameters, including separate
embedding and output matrices. With FP32 parameters and Adam moments:

```text
unsharded persistent state = 68.98e9 * 12 bytes = 770.9 GiB
FSDP-8 persistent state    = 770.9 / 8           =  96.4 GiB per GPU
```

Both figures are **[analytical]**. The experiment sets
`ici_fsdp_parallelism=8`, so the sharded figure is the relevant starting point. It
still omits a live FP32 gradient shard of about 32.1 GiB, gathered weights,
activations, Transformer Engine state, and workspaces.

The same config uses sequence length 4096 and microbatch 15 per device. Its
`remat_policy=full` is not incidental. A no-remat activation estimate is much larger
than the sharded optimizer state, so FSDP alone cannot make the run fit.

### Sparse Models

Top-k routing reduces the expert FLOPs used by each token, but it does not remove
inactive expert weights or their optimizer state. A sparse model therefore sizes
persistent memory from **total parameters**, not activated parameters.

Expert parallelism can divide expert state while leaving attention, embeddings,
routers, norms, and other dense state replicated. It is not equivalent to dividing
the complete model by the expert-parallel degree. This chapter uses that distinction
only to decide whether the step can fit.

[Chapter 8, Training Mixture-of-Experts on MI355X]({{ '/pages/8-mixture-of-experts-on-mi355x' | relative_url }})
owns the Mixtral state calculation, routing buffers, expert-parallel placement, and
FSDP-versus-EP comparison.

## Activation Memory

For a dense Transformer without a materialized attention-score matrix, the following
formula gives a partial estimate of commonly saved named residuals per GPU:

```text
activation bytes ≈
    B_micro * S * L * w * [4D + (N + 2K)H + 3F]
```

where:

- `B_micro` is the per-GPU microbatch;
- `S` is sequence length and `L` is layer count;
- `D` is model width, `F` is MLP width;
- `N`, `K`, and `H` are query heads, KV heads, and head dimension;
- `w` is bytes per saved activation element.

The terms represent residual and normalization values, Q/K/V projections, and MLP
intermediates. This is **[analytical]** and is not an upper bound. It omits logits, masks, dropout state,
attention-backend workspace, padding, quantization metadata, and compiler reuse. It
also assumes the named values survive until backward. A remat policy changes that
assumption.

For Llama 7B at batch 4, sequence 4096, and BF16, the formula gives about 60.25 GiB
before logits and attention-specific storage. For Llama 70B at microbatch 15 it gives
about 1,181 GiB per GPU. The latter cannot fit; a selective or full remat policy is a
requirement, not a tuning preference.

Naive attention adds a term proportional to:

```text
B_micro * N * S * S * w
```

for each live score or probability tensor. Flash attention avoids materializing that
full matrix in HBM. Changing `remat_policy` cannot make a materialized quadratic
attention algorithm behave like flash attention; select the attention backend first.

For an MoE layer, add token indices, routing weights, permutations, capacity
padding, and AllToAll buffers. Their sizes depend on the selected routing and expert
execution path, so Chapter 8 owns the complete sparse activation estimate.

Three practical rules follow:

1. Saved activation memory scales with microbatch, sequence length, and layer count.
2. Persistent state scales with total local parameters and is independent of
   microbatch.
3. Temporary peaks depend on lifetime. Summing every tensor ever created overstates
   the peak, while omitting simultaneous FSDP, attention, or quantization workspaces
   understates it.

## Measure the Compiled Program

JAX exposes XLA's static buffer plan on a compiled executable:

```python
compiled = (
    jax.jit(train_step, donate_argnums=(0,))
    .lower(state, batch)
    .compile()
)
stats = compiled.memory_analysis()
```

When the backend supplies the fields, a useful device estimate is:

```text
generated_code_size_in_bytes
+ argument_size_in_bytes
+ output_size_in_bytes
- alias_size_in_bytes
+ temp_size_in_bytes
```

OpenXLA calls this the minimum on-device memory needed by the executable
**[cited]**. `alias_size_in_bytes` is the input storage reused by outputs, often due to
donation. `host_temp_size_in_bytes` is separate and should not be added to the HBM
total.

The Llama 7B runner and MaxText both print:

```text
output + temporary + arguments - aliases
```

MaxText also reports allocator usage from `device.memory_stats()`. Keep both views:

- compiled memory is the static buffer assignment for one executable;
- allocator high-water includes runtime allocations visible to the allocator;
- neither replaces a phase-by-phase record after parameter initialization, optimizer
  initialization, compilation, and warmed execution.

JAX documents `memory_analysis()` as a debugging interface whose availability and
shape can vary across versions and backends. Record raw output rather than building a
parser that assumes every future field exists.

If the estimate and XLA disagree, reconcile these categories:

| Difference | Likely place to inspect |
|---|---|
| Arguments larger than the state estimate | Extra parameter copy, scale history, batch, or replicated shard |
| Outputs almost as large as arguments | Donation missing or unusable |
| Temporary storage dominates | Saved activations, fused-op workspace, collective buffer, padding, or poor lifetime reuse |
| Host temporary storage is nonzero | Offload or host-memory placement |
| Runtime peak exceeds compiled estimate | Library workspace, asynchronous overlap, allocator fragmentation, input pipeline, or initialization |
| Compiled estimate exceeds runtime peak | Conservative buffer plan or buffers whose branches did not execute |

## Donate Replaced State

A training step returns new parameters and optimizer state while the old values are no
longer needed. Donation lets XLA reuse compatible input buffers for those outputs:

```python
compiled_step = jax.jit(
    train_step,
    donate_argnums=(0,),
    in_shardings=(state_shardings, batch_sharding),
    out_shardings=(state_shardings, None),
)
```

The donated object can be a PyTree; all array leaves in that positional argument are
offered for reuse. The caller must not use the old object after dispatch. JAX raises an
error if code later accesses a donated buffer.

The raw-JAX Llama 7B runner donates `params` and `opt_state` separately with
`donate_argnums=(0, 1)`. MaxText donates train-state argument 0. Evaluation does not
donate state because evaluation keeps it.

The mathematical dataflow often remains identical with donation enabled: the step
still computes the same new state from the same old state. The useful compiler delta
is therefore often not a new HLO operation. Look instead for input-output alias
metadata on the executable and for the resulting buffer assignment, where an output
is assigned compatible storage formerly owned by a donated input. Compare
`alias_size_in_bytes` and the assigned buffer/live range, not only an HLO instruction
diff.

Donation is present in the checked-in raw-JAX and MaxText paths. Its runtime effect
still requires verification:

- `alias_size_in_bytes` should increase when compatible outputs reuse inputs;
- JAX should not warn that donated buffers were unusable;
- argument and output shapes, dtypes, and shardings must match;
- the allocator high-water should fall in a controlled on/off comparison.

Donation does not free saved activations and does not reuse a BF16 input for an FP32
output. It fixes the functional-update boundary, not every memory problem.

## Scan and Rematerialization

`jax.checkpoint` and `jax.remat` are aliases. During reverse-mode autodiff they control
which forward values may be saved and which must be recomputed in backward.

`jax.lax.scan` solves a different problem. A Python layer loop inside `jit` is normally
unrolled into a large graph. `scan` lowers the repeated body to one loop, reducing
compile size and time **[cited]**. It stacks layer parameters along
`param_scan_axis`; it does not remove those parameters or automatically reduce
activation memory.

The tested configs use:

```yaml
scan_layers: true
param_scan_axis: 1
```

The raw-JAX Llama 7B model applies remat to the scanned layer body and sets
`prevent_cse=False` under scan. JAX documents this pairing: compiler
rematerialization is less effective across forward and backward scans, so checkpointing
the scan body is often useful, and CSE prevention is usually unnecessary inside scan
**[cited]**.

The lowering vocabulary used below is established in
[Chapter 2]({{ '/pages/2-lowering-jax-jit-on-rocm' | relative_url }}#reading-a-compiler-delta).
Use the matched dump procedure in
[Appendix D]({{ '/pages/d-profiler-and-hlo-cookbook' | relative_url }}#feature-comparison-bundles)
and compare
the same before-optimization and optimized stages for every variant. The SVGs below
are rendered directly from XLA's `--xla_dump_hlo_as_dot` output for the small
fixtures in `bench/hlo_feature_fixtures.py`. Their raw HLO, DOT, debug options, and
provenance are retained under `artifacts/hlo-fixtures/`. They are real gfx950
compiler artifacts, but they are explanatory fixtures rather than Llama
case-study measurements.

### Scan in HLO

With an unrolled Python loop, each layer has its own copy of the layer operations in
the module. With scanned layers, the expected lowering is one `while` body plus an
induction variable and `dynamic-slice` operations that select layer `i` from each
stacked parameter. XLA may subsequently transform that structure, so verify it in the
selected dump rather than inferring it from `scan_layers=true`.

{% include figure.liquid path="pages/archive/img/hlo-scan-unrolled.svg" class="img-fluid" zoomable=true alt="Literal XLA HLO graph for four unrolled matrix layers" caption="Captured `before_optimizations` HLO: four parameter slices feed four distinct `dot_general` operations. Raw artifact: `artifacts/hlo-fixtures/scan/unrolled/`." %}

{% include figure.liquid path="pages/archive/img/hlo-scan-scanned.svg" class="img-fluid" zoomable=true alt="Literal XLA HLO graph for the same matrix layers expressed with lax.scan" caption="Captured `before_optimizations` HLO: one `while` carries the activation and stacked weights; its body dynamically slices one weight and executes one `dot_general`. Raw artifact: `artifacts/hlo-fixtures/scan/scanned/`." %}

The primal forward scan walks layer indices from 0 to `L - 1`. Reverse-mode
transposition commonly produces a second scan/`while` that walks saved layer data in
the opposite order while carrying cotangents. Its body may dynamically slice stacked
parameters and saved residual arrays using `L - 1 - i`. A forward `while` in the
primal is therefore not evidence that the complete training step contains only one
loop; count the forward and reverse loops and inspect both carries.

Use a fixed model, batch, sharding, remat policy, compiler flags, and cache state for
the scan on/off pair:

| Comparison target | Unrolled-layer expectation | Scanned-layer expectation | Required evidence |
|---|---|---|---|
| Optimized HLO instruction count and size | Layer body repeated approximately `L` times | One loop body plus tuple and slice machinery | Count all computations at the same dump stage; record HLO text/proto bytes |
| Compile time | Usually grows with the repeated graph | Usually lower because the body is compiled once | Separate lowering and compilation wall times; do not time exhaustive dump I/O |
| Runtime | No loop-index or parameter-slice overhead; more cross-layer optimization opportunities | Loop and dynamic-slice overhead; potentially less fusion across layers | Warmed step latency and tokens/s/GPU |
| Memory | Autodiff may expose separate residuals from every unrolled layer | Reverse scan may still carry or stack residuals for every layer | `memory_analysis()`, buffer assignment/live ranges, and runtime HBM high-water |

The first two directional statements are the motivation for scan, not measured
results for these experiments. Scan can reduce HLO size and compile time while leaving
activation memory unchanged or making runtime slower. Report all four comparisons;
none is a proxy for another.

### Rematerialization in HLO

For a scanned, differentiated layer, no explicit remat commonly makes the forward
loop tuple large: besides the hidden-state carry, it contains arrays into which the
body writes normalization inputs, projections, attention context, MLP intermediates,
or other residuals needed by the reverse loop. The reverse loop slices those saved
arrays. A full-remat policy should retain a smaller checkpoint tuple, such as layer
inputs, and clone the omitted forward computations into the backward body.

{% include figure.liquid path="pages/archive/img/hlo-remat-none.svg" class="img-fluid" zoomable=true alt="Literal XLA HLO graph for a two-layer differentiated function without explicit rematerialization" caption="Captured `before_optimizations` HLO without explicit remat. Forward `tanh` values feed the transpose calculation directly. Raw artifact: `artifacts/hlo-fixtures/remat/none/`." %}

{% include figure.liquid path="pages/archive/img/hlo-remat-full.svg" class="img-fluid" zoomable=true alt="Literal XLA HLO graph for the same differentiated function with jax checkpoint rematerialization" caption="Captured `before_optimizations` HLO with `jax.checkpoint`: `remat2` tuples and `checkpoint/rematted_computation` dots and `tanh` operations appear in backward. Raw artifact: `artifacts/hlo-fixtures/remat/full/`." %}

These are three different observations:

1. A jaxpr `remat`/`remat2` primitive records the autodiff policy's intent about which
   values may be recomputed.
2. Optimized HLO records the consequence after lowering, CSE, fusion, and other
   rewrites. Evidence that the policy took effect is a smaller saved-residual
   tuple/live set together with cloned or fused equivalent work in the backward
   region; the literal `remat` name need not survive.
3. XLA's compiler rematerialization is a separate memory-scheduling optimization. It
   may recompute an operation even without `jax.remat`, and it may alter the final
   schedule after the explicit policy has been lowered. Compiler rematerialization is
   not proof that a JAX policy matched the intended checkpoint names.

HLO operation counts alone are insufficient: fusion can reduce the count while
extending a buffer lifetime, and a cloned recomputation can disappear through CSE.
For a no-remat/selective/full comparison, require buffer-assignment evidence showing
the allocation sizes and live ranges at the peak, the raw `memory_analysis()` fields,
and the allocator high-water from warmed execution. Pair that memory evidence with
the optimized HLO's forward/reverse loop carries and a runtime trace or tokens/s/GPU
measurement. Without the buffer-lifetime evidence, describe the remat policy as
configured, not as a demonstrated memory saving.

### The Policy Surface

MaxText labels important values with `checkpoint_name`. A policy names what may remain
in HBM; unnamed or excluded residuals are recomputed.

| Policy | What it retains or moves | Expected direction |
|---|---|---|
| `none` | Normal autodiff residuals; no explicit layer remat | Highest activation memory, least explicit recompute |
| `minimal_with_context` | Projection and MLP dot outputs plus attention context | High memory, avoids expensive recompute |
| `minimal` | Similar named dot outputs without attention context | Less memory, recomputes context |
| `save_dot_with_context_except_mlp` | Attention projections, output, and context; MLP intermediates recompute | Useful middle point for large MLP/MoE layers |
| `save_dot_except_mlpwi` | Attention projections/output and MLP down projection | More MLP recompute |
| `save_dot_except_mlp` | Attention projections/output only | Lower memory |
| `save_qkv_proj` | Q/K/V projections only | Lower memory |
| `save_out_proj` | Attention output projection only | Low memory |
| `full` | Layer input checkpoint; recompute the layer body | Lowest named-residual memory, highest recompute |
| `qkv_proj_offloaded` | Q/K/V residuals in pinned host memory | Trades HBM for transfers |
| `minimal_offloaded` | Most named dot residuals in pinned host memory | Larger transfer volume |
| `custom` | Per-name `device`, `remat`, or `offload` assignment | Explicit, model-specific control |

This is a semantic ordering, not a promised performance ranking. XLA fusion, attention,
quantization, scan lowering, and FSDP can change both bytes and recomputation. Compile
and measure each candidate.

The experiment repos provide a controlled sweep:

- Llama 7B on one GPU: `none`, `minimal_with_context`, and `full`;
- the same three policies under FSDP-8;
- Llama 70B: `full`;
- Mixtral 8x22B: `save_dot_with_context_except_mlp`.

No timing or peak-memory output is committed for the Llama 7B sweep. The scripts prove
the intended comparison exists, but they do not support a numeric recommendation.

### Interactions That Change the Answer

**Policy names must match model names.** A custom policy that says `mlpwi_0` has no
effect if the model never applies `checkpoint_name(..., "mlpwi_0")`. Inspect the model
and HLO rather than assuming a borrowed policy transfers.

**Scan changes the remat boundary.** Apply the policy to the layer body, not blindly
around the complete network. MaxText warns that saving quantization residuals while
scanning can be slower than recomputing them for some configurations.

**Quantization adds its own residuals.** FP8 delayed scaling carries history and scales.
MX formats may save packed columnwise or rowwise forms. The JAX-AITER MXFP4 branch has
special checkpoint names for reusable column layouts; saving them spends HBM to avoid
recasting during backward. A policy written for BF16 is incomplete for that path.

**FSDP can repeat communication.** Rematerializing a function that contains or depends
on a just-in-time weight AllGather can cause the gather or conversion to reappear in
backward. Compare HLO collective counts and the trace across policies. Do not report a
memory saving without the corresponding tokens/s/GPU.

**Attention backend comes first.** Flash attention removes the quadratic score
materialization. Remat then decides which remaining outputs survive. A no-remat flash
run can use less memory than an aggressively rematerialized naive-attention run.

**Gradient accumulation is another scan.** On GPU, the pinned MaxText code disables
CSE prevention when `gradient_accumulation_steps > 1`. Layer scan, microbatch scan, and
remat nesting can alter compile time, loop buffers, and recomputation. Test the combined
configuration, not each knob in isolation.

## Gradient Accumulation

Gradient accumulation splits an update into sequential microbatches:

```text
global sequences per update =
    per_device_batch_size
    * device_count
    * gradient_accumulation_steps
```

The Mixtral config uses `per_device_batch_size=4`, eight devices, and
`gradient_accumulation_steps=2`, giving 64 sequences per optimizer update.

MaxText reshapes the update batch into microbatches, runs forward and backward in a
`lax.scan`, sums gradients, and applies one optimizer update. Activation memory tracks
one microbatch rather than the full update batch. Parameter, optimizer, and gradient
accumulator memory does not shrink.

At HLO level, inspect the microbatch loop carry. It should identify the induction
variable, unchanged or donated train state, full gradient accumulator, scale/RNG
state, and accumulated metrics. Extra carried activation arrays can defeat the
expected one-microbatch lifetime.

{% include figure.liquid path="pages/archive/img/hlo-accumulation-direct.svg" class="img-fluid" zoomable=true alt="Literal XLA HLO graph for one gradient over the full update batch" caption="Captured `before_optimizations` HLO for a direct full-batch gradient. The reshaped update batch feeds one forward dot and one weight-gradient dot. Raw artifact: `artifacts/hlo-fixtures/accumulation/direct/`." %}

{% include figure.liquid path="pages/archive/img/hlo-accumulation-scanned.svg" class="img-fluid" zoomable=true alt="Literal XLA HLO graph for gradients accumulated through a microbatch scan" caption="Captured `before_optimizations` HLO for four microbatches. A `while` carries the gradient accumulator and stacked input, dynamically slices one microbatch, computes its forward and weight-gradient dots, and adds the result into the carry. Raw artifact: `artifacts/hlo-fixtures/accumulation/scanned/`." %}

Collective placement is a separate comparison target. A gradient `all-reduce` or
`reduce-scatter` inside the microbatch body executes once per microbatch; the same
collective after the loop executes once per optimizer update, usually with a
different overlap and accumulator requirement. FSDP weight gathers or other
collectives may still execute per microbatch even when gradient reduction is outside.
Record collective location and replica groups in optimized HLO, then use the trace to
count dynamic executions and transferred bytes. One textual collective inside a
`while` is not one runtime collective per update.

There are two distinct experiments:

- Hold `per_device_batch_size` fixed and increase accumulation. Tokens per update rise,
  so step time rises and optimizer/collective cost is amortized over more tokens.
- Hold global batch fixed, reduce `per_device_batch_size`, and increase accumulation.
  Activation peak falls, while more sequential microbatches add loop and launch
  overhead.

Only the second is a pure memory trade. Compare tokens/s/GPU at a fixed global batch and
fixed token sequence.

Accumulation also connects the memory and convergence constraints. The microbatch must
fit HBM and be large enough for efficient kernels. The global batch must remain below
the point where extra examples stop improving time to target. Accumulation lets those
two batch sizes differ.

Check precision compatibility. The pinned MaxText source explicitly rejects its
`fp8`, `nanoo_fp8`, `fp8_gpu`, and `te_fp8_delayedscaling` paths when
`gradient_accumulation_steps != 1`. `te_mxfp8` and `aiter_fp4` do not share that
static check in the pinned source; this is not proof that accumulation works. Validate
their forward, backward, scale state, and loss explicitly.

## FSDP and Sharded Initialization

If persistent state dominates, shard it before changing activation policy. This is
FSDP's primary role in the memory decision: parameters, gradients, master weights,
and optimizer state are distributed according to their logical axis rules, then
weights are gathered when a layer needs them.

### Choose the minimum FSDP degree

A useful capacity condition is:

$$
M_{\mathrm{replicated}}
+\frac{M_{\mathrm{shardable}}}{d_{\mathrm{FSDP}}}
+M_{\mathrm{temporary}}(d_{\mathrm{FSDP}})
\leq M_{\mathrm{budget}}.
$$

The replicated term includes leaves whose logical rules do not use `fsdp`. The
temporary term includes gathered weights, collective buffers, workspaces, and
overlap-induced live ranges. Consequently, dividing total state by the FSDP degree is
only a lower-level estimate.

The Llama 70B configuration uses:

```yaml
ici_fsdp_parallelism: 8
dcn_fsdp_parallelism: 1
```

Choose the smallest degree that satisfies the compiled and runtime peak-memory
checks. Chapter 6 prices the resulting AllGather and ReduceScatter traffic and
decides whether that degree is efficient.

`shard_optimizer_over_data=true` is a separate ZeRO-1-style option. It shards optimizer
state over the data axis while model parameters remain replicated for compute. Do not
call ordinary FSDP and optimizer-only sharding the same configuration; their memory and
collective costs differ.

For sparse models, expert parallelism can shard expert state while shared attention
and embedding state remains replicated. Chapter 8 owns that separate memory and
communication decision.

### Initialize directly into shards

Sharding must apply during initialization. This pattern is unsafe for a model larger
than one GPU:

```python
state = init_full_state(rng)       # transient full state on one device
state = jax.device_put(state, state_shardings)
```

Build an abstract state, resolve its logical axes to `NamedSharding`, and compile the
initializer with output shardings:

```python
abstract_state = jax.eval_shape(init_state, rng)
state_shardings = logical_to_mesh_sharding(abstract_state, mesh)
state = jax.jit(init_state, out_shardings=state_shardings)(rng)
```

MaxText follows this structure for its train state. The `out_shardings` contract lets
JAX produce each shard on its target device instead of materializing one unsharded
intermediate. Verify initialization separately from steady-state execution:

1. record memory after parameters;
2. record memory after optimizer state;
3. print every large leaf's global shape, local shape, dtype, and sharding;
4. confirm no device briefly holds the full model;
5. test checkpoint restore, because restore can reintroduce a host or device staging
   copy that fresh initialization avoided.

Scanned checkpoints carry a stacked-layer axis. MaxText records `scan_layers` in
checkpoint metadata and rejects an explicit mismatch. Preserve `scan_layers` and
`param_scan_axis` when moving between initialization, training, and restore.

## Host Offload

Host offload places selected residuals, parameters, or optimizer state in pinned host
memory and transfers them when needed. JAX exposes pinned-host memory kinds and
`save_and_offload_only_these_names`; MaxText exposes:

```yaml
remat_policy: "qkv_proj_offloaded"  # or minimal_offloaded/custom
optimizer_memory_host_offload: false
parameter_memory_host_offload: false
```

For `custom`, each named tensor can be assigned `device`, `remat`, or `offload`.
Current upstream MaxText source also contains pretraining paths for optimizer and
parameter host placement. Parameter offload requires `param_scan_axis=0` in the
inspected implementation, while the tested experiment configs use axis 1.

In compiler artifacts, compare the program before and after memory-space assignment.
A realized offload should assign the selected buffer to a host memory space and
introduce `copy-start`/`copy-done` pairs, or the backend's equivalent asynchronous
copy operations, around its next device use. Names and exact placement are
backend-dependent, so absence of those literal strings in one dump is not by itself
proof that no transfer exists.

The API surface is current. The performance recommendation is not. None of the Llama or
Mixtral experiment repositories measures host offload on MI355X, so this book marks all
three offload routes experimental on the pinned ROCm stack.

PCIe Gen5 x16 provides about 64 GB/s per direction, while MI355X HBM provides about
8 TB/s. A byte fetched from host therefore has roughly 125x less peak bandwidth than a
byte read from HBM **[analytical]**. Offload helps only when the transfer:

- replaces an HBM term that otherwise prevents the run;
- is smaller or cheaper than the recomputation alternative;
- is prefetched early enough to hide behind useful compute;
- does not contend with checkpointing, data input, or other PCIe traffic.

Measure host bytes, transfer intervals, overlap, tokens/s/GPU, and HBM high-water.
`host_temp_size_in_bytes > 0` proves host memory exists in the compiled plan; it does
not prove transfer is hidden. Require trace proof of each host-to-device and
device-to-host interval, its bytes and duration, and whether it overlaps useful
compute; correlate those events with the HLO copy pairs. Prefer remat when
recomputation is cheap. Prefer state sharding when another GPU can own the bytes. Use
offload after those routes are insufficient or when capacity matters more than
throughput.

## A Memory Decision Procedure

1. Freeze the workload: model, sequence length, global batch, precision assignment,
   attention backend, mesh, and optimizer.
2. Estimate persistent state from actual parameter leaves and dtypes. Separate
   shared, FSDP-sharded, tensor-sharded, and expert-sharded terms.
3. Estimate no-remat activation memory. Include naive-attention scores, logits, MoE
   dispatch, quantizer state, and known workspaces as separate terms.
4. Initialize directly with target shardings. Record allocator memory after parameters
   and after optimizer creation.
5. Compile the donated train step. Save raw `memory_analysis()` output and reconcile
   arguments, outputs, aliases, temporaries, and host memory with the estimates.
6. If persistent state dominates, increase FSDP or optimizer sharding before remat.
   Recheck temporary gathered-weight buffers and communication.
7. If activations dominate, select flash attention, then sweep `none`, a selective
   policy, and `full`. Test each with the real `scan_layers`, precision, and FSDP
   configuration.
8. If the required global batch does not fit, lower the microbatch and increase
   `gradient_accumulation_steps` while holding global batch fixed.
9. Add host offload only after sharding, donation, attention, remat, and accumulation
   are understood. Treat it as a capacity experiment until overlap is measured.
10. Choose the lowest-memory configuration that meets the tokens/s/GPU target and
    retains the Chapter 4 convergence guardrail. Keep headroom for checkpoint restore,
    profiler overhead, and runtime variance.

## Decision Table

Status distinguishes source availability from validated execution. None of these
rows has a complete Appendix F bundle yet.

| Knob | What it buys | What it costs | How to set it | Status on ROCm | Verified on |
|---|---|---|---|---|---|
| Static memory analysis | Buffer-plan estimate before execution | Compiler-specific, omits some runtime peaks | `compiled.memory_analysis()` | Unverified [source] | JAX 0.11.0 source, 2026-09-13 |
| Allocator high-water | Runtime HBM use visible to JAX | Can include allocator policy and miss external detail | `device.memory_stats()` | Unverified [source] | Llama 7B and MaxText source, 2026-09-13 |
| Train-state donation | Reuses old state buffers for outputs | Donated inputs become invalid; shape/dtype must match | `jax.jit(..., donate_argnums=(0,))` | Unverified [source] | Llama 7B source, 2026-09-13 |
| Layer scan | Smaller compiled graph and shorter compile | Loop constraints; parameters become stacked | `scan_layers=true param_scan_axis=1` | Unverified [source] | Three experiment configs, 2026-09-13 |
| No explicit remat | Avoids deliberate recomputation | Highest saved-activation demand | `remat_policy=none` | Unverified [source] | Llama 7B sweep definition, 2026-09-13 |
| Selective remat | Removes chosen residuals while saving expensive ones | Policy- and backend-specific recomputation | `remat_policy=minimal_with_context` or `save_dot_with_context_except_mlp` | Unverified [source] | Llama 7B/Mixtral configs, 2026-09-13 |
| Full remat | Lowest named-residual memory | Recomputes the layer body; can repeat casts or gathers | `remat_policy=full` | Unverified [source] | Llama 7B/70B configs, 2026-09-13 |
| Custom remat | Per-tensor save/recompute assignment | Requires stable checkpoint names and retesting | `remat_policy=custom`, tensor fields=`device`/`remat` | Unverified [source] | MaxText source audit, 2026-09-13 |
| Gradient accumulation | Smaller activation microbatch at fixed global batch | Sequential microbatches and one full gradient accumulator | `gradient_accumulation_steps=N` | Unverified; precision-specific | Mixtral config and source audit, 2026-09-13 |
| FSDP state sharding | Divides parameter, gradient, and optimizer state | Weight AllGather/ReduceScatter and gathered temporaries | `ici_fsdp_parallelism=N` | Unverified [source] | Llama 70B config, 2026-09-13 |
| Optimizer-only sharding | Divides optimizer state over data replicas | Different update/collective path from FSDP | `shard_optimizer_over_data=true` | Unverified [source] | MaxText source audit, 2026-09-13 |
| Sharded initialization | Avoids a full one-device initialization peak | Requires correct abstract axes and restore path | JIT initializer with `out_shardings=state_shardings` | Unverified [source] | MaxText source audit, 2026-09-13 |
| Activation host offload | Replaces selected HBM residuals with pinned-host storage | PCIe traffic and prefetch requirement | `qkv_proj_offloaded`, `minimal_offloaded`, or `custom` | Experimental, unmeasured here | API/source audit, 2026-09-13 |
| Optimizer host offload | Removes optimizer state from steady HBM | Transfers state for update; some metrics unavailable | `optimizer_memory_host_offload=true` | Experimental, unmeasured here | MaxText source audit, 2026-09-13 |
| Parameter host offload | Removes parameter state from steady HBM | Frequent transfers; requires `param_scan_axis=0` in inspected path | `parameter_memory_host_offload=true param_scan_axis=0` | Experimental, unmeasured here | MaxText source audit, 2026-09-13 |

**Recommendation status: BLOCKED.** Donation and sharded initialization are
source-defined defaults to verify. Rematerialization, accumulation, and host
offload require fixed-workload tokens/s/GPU and peak-HBM results before a policy
can be selected. The fallback is BF16 with the minimum FSDP degree that the
compiled memory plan shows will fit. Retest when compiler scheduling, attention,
quantization, or model shape changes.

## References

- [JAX buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html).
  Donation semantics, valid use, and unusable-buffer warnings.
- [JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html).
  Remat policies, named residuals, scan interaction, and offload.
- [JAX memories and host offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html).
  `memory_analysis()`, memory kinds, and pinned-host placement.
- [JAX `lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html).
  WhileOp lowering and compile-size motivation.
- [OpenXLA `CompiledMemoryStats`](https://github.com/openxla/xla/blob/main/xla/pjrt/compiled_memory_stats.h).
  Device and host memory fields and the minimum-memory expression.
- [MaxText base configuration](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml).
  Remat, scan, accumulation, sharding, and host-offload fields.
- [MaxText decoder policies](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/layers/decoders.py).
  Named save, recompute, and pinned-host policies.
- [Llama 7B JAX fundamentals](https://github.com/clarkechong/llama7b-jax-fundamentals/tree/5f996a88).
  Donation, compiled memory reporting, and one-GPU/FSDP remat sweeps.
- [Llama 70B mixed-precision experiments](https://github.com/clarkechong/llama70b-mixed-precision-training/tree/f3dab369).
  FSDP-8 state roles, full remat, and accumulation control.
- [Mixtral 8x22B distributed strategies](https://github.com/clarkechong/mixtral8-22b-distributed-strategies/tree/a32b51d6).
  EP-8 versus FSDP configurations, BF16 optimizer state, and two-step accumulation.
