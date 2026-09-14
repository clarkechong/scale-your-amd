---
layout: distill
title: "DeepSeek V3"
description: "The future acceptance specification for a DeepSeek V3 training capstone on a multi-node MI355X mesh."
date: 2026-09-13

section_number: 15

previous_section_url: "/pages/14-multinode"
previous_section_name: "Chapter 14: Multi-Node Training"

next_section_url: "/pages/a-appendix-install"
next_section_name: "Appendix A: Environment"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Status and Acceptance Boundary
  - name: Freeze the Model Ledger
  - name: MLA Ledger
  - name: MoE Ledger
  - name: Training-State Ledger
  - name: Multi Token Prediction Ledger
  - name: Mesh Plan
  - name: Staged Validation
  - name: Scaling and Quality Gates
  - name: Fallback Requirement
  - name: Required Outputs
---

> **Future acceptance specification.** No DeepSeek V3 performance number is claimed here.
> The chapter becomes a case study only after the model, mesh, numerics, scaling, and
> artifact gates below pass on the pinned MI355X stack.

**Depends on:** [Chapter 9]({{ '/pages/9-moe' | relative_url }}) for MoE execution,
[Chapter 13]({{ '/pages/13-mixtral8-22b' | relative_url }}) for the one-node sparse-model
method, and [Chapter 14]({{ '/pages/14-multinode' | relative_url }}) for launch, data,
checkpoint, restart, and monitoring.

## Status and Acceptance Boundary

The current Mixtral runners do not implement MLA, shared experts, DeepSeek routing,
multi-token prediction, or multi-node launch. Each component therefore needs an
independent correctness and lowering test before the end-to-end run.

The source model configuration must be pinned to a repository and commit. Record any local
change as a patch. The chapter must not copy dimensions from a paper, a Hugging Face config,
and a MaxText config without proving that they describe the same model.

The final chapter must answer:

1. What model was compiled?
2. Which parameters and activations are dense, shared, routed, or latent?
3. Which mesh axis owns each tensor dimension and collective?
4. Which kernel handled MLA and each expert path?
5. Did one-device, one-node, and multi-node results agree within declared tolerances?
6. Did aggregate throughput increase at each accepted scaling point?
7. Did the quality smoke test match the reference recipe?
8. Can the job survive a forced stop and resume the same data stream?

Until all eight have retained evidence, the page remains a specification.

## Freeze the Model Ledger

Export a machine-readable ledger directly from the effective model config. It must contain:

| Group | Required fields |
|---|---|
| Identity | Model/config name, source URL, commit, local patch hash |
| Core | Layer count, vocabulary, hidden width, normalization, sequence length |
| Attention | Attention type, head counts, every query/key/value dimension, RoPE dimensions |
| Dense MLP | Dense-prefix layer count, intermediate width, activation |
| Routed MoE | Routed expert count, active experts per token, routed expert width |
| Shared MoE | Shared expert count and width |
| Router | Scoring function, top-k rule, group restriction, precision, balance mechanism |
| Training | Parameter dtype, compute dtype, accumulation dtype, optimizer state, remat policy |

Add three independently checked totals:

- total parameter count;
- parameters activated per token; and
- trainable bytes by state category.

The config export, parameter-tree count, and analytical count must agree within a
predeclared tolerance. A discrepancy blocks compilation measurements because it means the
workload is not identified.

## MLA Ledger

Do not reduce multi-head latent attention to a single hidden dimension. Record every
projection that affects shape, FLOPs, sharding, or saved activations:

```text
number of attention heads
query LoRA rank
KV LoRA rank
query/key non-positional head dimension
query/key RoPE head dimension
value head dimension
input and output projection shapes
normalization points
RoPE placement
attention mask and packing rule
```

For forward and backward, the ledger must list:

| Operation | Global shape | Dtype | Sharding | Saved or rematerialized | Expected kernel |
|---|---|---|---|---|---|
| Query compression and expansion | Required | Required | Required | Required | Required |
| KV compression | Required | Required | Required | Required | Required |
| Key/value expansion | Required | Required | Required | Required | Required |
| RoPE path | Required | Required | Required | Required | Required |
| Attention scores and softmax | Required | Required | Required | Required | Required |
| Output projection | Required | Required | Required | Required | Required |

Derive FLOPs and live bytes from these shapes. Keep score-matrix work separate from linear
projections. Verify the ledger against optimized HLO and the compiled memory report. Any
materialized tensor missing from the ledger fails the MLA accounting gate.

MLA component acceptance requires:

- forward outputs against a simple reference implementation;
- input and parameter gradients against the same reference;
- packed and unpacked sequence cases;
- causal-mask and sequence-boundary cases;
- each intended dtype path;
- optimized HLO proving the selected implementation; and
- a kernel trace proving the expected backend or recording a fallback.

Numerical tolerances are dtype-specific and written before the test. A finite output alone
does not pass.

## MoE Ledger

Keep shared experts and routed experts separate throughout the accounting.

The static ledger records:

```text
shared expert count and width
routed expert count and width
active routed experts per token
router input and logits dtype
top-k and group-selection rules
capacity, padding, dropping, or dropless policy
load-balance loss or bias update
sort/permute and inverse-permute method
expert GEMM implementation
dispatch and combine collective
```

For every MoE layer, derive:

- shared-expert FLOPs and bytes per token;
- routed-expert FLOPs and bytes per token;
- router and top-k work;
- token metadata and dispatch bytes;
- expected tokens per expert under uniform routing;
- buffer capacity and padding policy; and
- total versus activated parameter bytes.

The dynamic ledger is measured per step and per layer:

| Metric | Required aggregation |
|---|---|
| Tokens per expert | Minimum, median, maximum, histogram |
| Routing concentration | Per layer and over time |
| Dropped or padded tokens | Count and fraction |
| Dispatch and combine | Bytes, duration, participant group, exposed duration |
| Expert GEMMs | Shape distribution, dtype, kernel family, duration |
| Shared experts | Duration kept separate from routed experts |

MoE correctness tests compare dispatch, expert output, combine, and gradients with a small
dense reference. Use token counts that exercise empty experts, uneven routing, repeated
expert choices, and capacity boundaries. The full model cannot compensate for a routing
test that was never passed.

## Training-State Ledger

Before selecting a mesh, account for:

- parameters by MLA, dense MLP, shared expert, routed expert, embedding, and output head;
- master weights, gradients, and optimizer moments by dtype;
- saved activations and rematerialized operations;
- attention and expert workspaces;
- sort, routing, dispatch, and collective buffers;
- compiler temporaries and command-buffer allocations;
- checkpoint bytes written by each process; and
- per-device high-water mark from the compiled plan and runtime.

The analytical ledger, XLA memory analysis, and runtime high-water mark need not be equal,
but every material difference must have an identified buffer or accounting rule. Choose no
multi-node mesh until at least one one-node configuration fits with a declared safety margin.

## Multi Token Prediction Ledger

DeepSeek V3 training includes a multi-token prediction (MTP) module in addition to the
core 671B model. Keep the two counts separate. The model identity record must state
whether a parameter total covers the core model alone or the complete training
checkpoint with MTP.

Record:

- the number and placement of MTP modules;
- every projection and normalization shape;
- whether embeddings or output weights are shared with the main model;
- the auxiliary loss definition and weight;
- additional forward and backward FLOPs per token;
- saved activations and rematerialization policy;
- parameter, gradient, optimizer, and checkpoint bytes;
- logical and physical shardings; and
- whether the exported inference checkpoint omits MTP state.

The effective config, parameter tree, optimizer tree, and checkpoint inventory must
agree with the MTP-inclusive ledger. A core-model match does not pass if the training
checkpoint contains unaccounted MTP parameters or optimizer state.

## Mesh Plan

Express the mesh as named ICI and DCN factors:

```text
ICI: data, fsdp, tensor, sequence/context, expert, pipeline
DCN: data, fsdp, tensor, sequence/context, expert, pipeline
```

The product of active factors must equal the global device count, and each logical model
axis must have one documented physical mapping.

Start from these candidates; they are experiments, not recommendations:

| Candidate | Purpose | Constraint |
|---|---|---|
| One device | Reduced-model numerical reference | Preserve operation shapes and routing semantics where possible |
| One node | Reduced-model integration and TP/EP/FSDP comparison | Do not imply that the full training state fits |
| Two nodes, DCN data parallel | Reduced-model scale-out reference | Replicate the accepted one-node model mesh |
| Two nodes, one alternate DCN axis | Reduced-model FSDP or pipeline test | Change one cross-node factor at a time |
| Full-model minimum | First complete-model run | Ledger proves fit with declared memory headroom |
| Larger allocation | Extend the accepted full-model mesh | Keep workload and protocol fixed |

Treat TP and EP as competing users of the fast intra-node axes. Keep a latency-sensitive
axis within a node unless a measured collective sweep and memory requirement justify
crossing nodes. Do not put DP, FSDP, PP, TP, and EP across DCN simultaneously and then try
to infer which one caused the result.

For each candidate, publish:

- all `ici_*_parallelism` and `dcn_*_parallelism` fields;
- the JAX device mesh in rank order;
- logical-axis rules for every parameter group;
- local shard shapes for MLA and expert weights;
- optimized HLO replica groups;
- collective type, tensor shape, bytes, and participants;
- physical node, GPU, NIC, and rail placement; and
- whether communication is exposed or overlapped in the trace.

## Staged Validation

Advance in this order:

### Stage 0: configuration and lowering

Export the model ledger, instantiate an abstract parameter tree, lower one train step, and
save StableHLO, optimized HLO, sharding annotations, and compiled memory analysis. No timing
from this stage is a training result.

### Stage 1: component correctness

Run MLA and MoE forward/backward tests independently. Prove kernel routes with HLO and
traces. Record unsupported paths, slow fallbacks, and numerical failures in
[Appendix E]({{ '/pages/e-appendix-compatibility' | relative_url }}).

### Stage 2: one-device model correctness

Run a reduced model with deterministic inputs. Compare loss, selected activations,
gradients, and one optimizer update against the reference implementation.

### Stage 3: one-node integration

Run a shape-faithful reduced model. Run the full model only if the complete
MTP-inclusive state ledger proves that it fits with the declared safety margin.
Validate every candidate one-node mesh, router diagnostics, memory ledger, checkpoint
write, and same-allocation restore. Label reduced-model results explicitly.

### Stage 4: two-node operations

Use the accepted Chapter 14 launcher and the reduced workload. Validate synthetic and
real-data operation, checkpointing, restart, one forced process failure, and one
interrupted checkpoint. A two-node reduced-model result is an operations test, not a
full-model scaling point.

### Stage 5: scaling series

Begin full-model scaling at the smallest allocation that the complete state ledger
proves can fit. Collect unprofiled timing samples at each allocation and capture
profiles in separate runs. Keep the global batch rule fixed or start a new, clearly
named weak-scaling series.

### Stage 6: quality smoke test

Use a pinned dataset, tokenizer, initialization, optimizer, and token budget. Compare the
selected recipe with the BF16 reference using validation loss and stability diagnostics.
This is a smoke test, not evidence of full pretraining quality.

## Scaling and Quality Gates

| Gate | Acceptance evidence |
|---|---|
| Model identity | Config, parameter tree, and analytical ledgers agree |
| MLA | Forward, backward, masks, packing, HLO, and kernel route pass |
| MoE | Routing, dispatch, combine, gradients, and edge cases pass |
| MTP | Parameters, auxiliary loss, gradients, sharding, and checkpoint state agree |
| Memory | Predicted and observed allocations are reconciled and the run fits safely |
| Mesh | Device map and HLO replica groups match the declared placement |
| One node | Stable loss, complete router metrics, checkpoint and restore |
| Two nodes | Same correctness plus Chapter 14 data and failure-recovery gates |
| Scaling | Aggregate tokens/s rises and the predeclared efficiency floor is met |
| Quality | No unexplained divergence from the reference over the pinned smoke-test budget |
| Reproducibility | A clean allocation can recreate the accepted result from published artifacts |

Report failed gates with the same prominence as passed gates. A faster run that changes
token order, drops tokens unexpectedly, uses a fallback kernel, or diverges from the
reference is rejected.

## Fallback Requirement

Every experimental fast path needs a fallback selected from components that passed earlier
stages. The final artifact bundle must contain one runnable fallback config that:

- uses the verified BF16 numerical baseline;
- uses the simplest accepted MLA implementation;
- replaces a failed ragged/grouped expert path with the accepted padded or fixed-capacity
  path;
- uses the smallest mesh that fits and has passed restart;
- disables only flags tied to a recorded incompatibility; and
- preserves the same model, data, optimizer, and metric definitions.

The fallback is not described as performant. Its purpose is to separate model and
infrastructure correctness from the fast path under investigation.

## Required Outputs

The completed chapter requires:

- pinned model source, exported config, local patches, and ledger generator;
- MLA, MoE, MTP, and training-state ledgers in machine-readable and rendered form;
- component test vectors and reference outputs;
- all candidate mesh configs and rank/topology maps;
- StableHLO, optimized HLO, compiled memory reports, and kernel-route evidence;
- unprofiled raw timing samples for every scaling point;
- XProf plus lossless `rocprofv3` captures for selected stages;
- per-layer routing histograms and collective exposure records;
- real-data and tokenizer manifests;
- clean, forced-stop, and interrupted-write checkpoint/restore records;
- quality-smoke-test metrics and comparison script;
- accepted fallback config and command;
- negative-result entries with stack, symptom, workaround, and retest status; and
- an [Appendix F]({{ '/pages/f-appendix-artifacts' | relative_url }}) manifest with
  checksums.

Only after these outputs exist should the specification be replaced with measured results.
