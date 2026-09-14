---
layout: distill
title: "Configuration Quick Reference"
description: "MaxText fields, mesh axes, XLA flags, and environment variables used by the MI355X case studies."
date: 2026-09-13

section_label: "Appendix C"

previous_section_url: "/pages/b-measurement-and-convergence-protocol"
previous_section_name: "Appendix B: Protocol"

next_section_url: "/pages/d-profiler-and-hlo-cookbook"
next_section_name: "Appendix D: Profiling"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Run Identity and Batch
  - name: Precision and Kernels
  - name: Memory and State
  - name: Mesh Axes
  - name: MoE Fields
  - name: Data and Evaluation
  - name: Checkpoint and Metrics
  - name: XLA Flags
  - name: Environment Variables
---

This page is a lookup for fields exercised by the current repositories. A field being
present does not prove that its requested backend ran. Verify optimized HLO and a kernel
trace.

## Run Identity and Batch

| Field | Meaning | Record with it |
|---|---|---|
| `model_name` | MaxText model configuration | MaxText commit |
| `run_name` | Output namespace | Unique run ID |
| `hardware` | Backend selector; current cases use `gpu` | JAX platform version |
| `steps` | Total train steps | Warmup and measured ranges |
| `max_target_length` | Tokens per sequence | Packing policy |
| `per_device_batch_size` | Sequences per addressable device | Device count |
| `gradient_accumulation_steps` | Microsteps per optimizer update | Global batch formula |
| `learning_rate_schedule_steps` | Schedule horizon | Restore step |

Use:

$$
B_{\mathrm{global}}
=
B_{\mathrm{device}}\times N_{\mathrm{devices}}\times N_{\mathrm{accumulation}},
$$

unless the input pipeline or process model defines a narrower data-parallel group. State
the actual formula used by the run.

## Precision and Kernels

| Field | Current use | Verification |
|---|---|---|
| `dtype` | Compute/activation baseline | HLO tensor dtypes |
| `weight_dtype` | Stored working weights | Parameter tree |
| `grad_dtype` | Gradient storage | Parameter/optimizer tree |
| `mu_dtype` | Adam first-moment storage | Optimizer tree |
| `quantization` | Empty, `te_fp8_delayedscaling`, `te_mxfp8`, or `aiter_fp4` in current runners | Custom calls and kernel names |
| `attention` | XLA dot product or `cudnn_flash_te` | Attention HLO and trace |
| `use_jax_aiter` | Enable MaxText JAX-AITER integration | Imported source commit and FFI target |
| `aiter_attention` | Select AITER attention within that integration | Forward/backward kernel route |

Current source requirements:

| Path | Required source |
|---|---|
| BF16, FP16, delayed-scaling FP8 | Stock MaxText `b47d74bf` |
| MXFP8 | Patched Transformer Engine 2.17 branch; pin commit and wheel hash |
| MXFP4 | MaxText `b437942a`, JAX-AITER `35b7175c`, AITER `31350226` |

## Memory and State

| Field | Effect | Required check |
|---|---|---|
| `remat_policy` | Selects saved versus recomputed activations | Compiled memory and HLO |
| `scan_layers` | Lowers repeated layers through a scan | HLO structure and compile count |
| `reuse_example_batch` | Reuses one batch; `1` is synthetic timing only | Data declaration |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | Caps allocator fraction; current runners use `0.97`, Llama 7B direct attention uses `0.9` | Effective environment |
| `use_iota_embed` | Avoids materialized embedding inputs in current recipes | HLO and model equivalence |

Precision, remat, batch, and FSDP all change memory. Change one at a time when attributing a
memory result.

## Mesh Axes

The current configs expose the same factors at two topology levels:

```text
ici_data_parallelism
ici_fsdp_parallelism
ici_fsdp_transpose_parallelism
ici_tensor_parallelism
ici_tensor_sequence_parallelism
ici_sequence_parallelism
ici_context_parallelism
ici_context_autoregressive_parallelism
ici_expert_parallelism
ici_pipeline_parallelism
ici_autoregressive_parallelism
ici_diloco_parallelism

dcn_<same axis names>
```

| Field | Shards | Typical critical communication |
|---|---|---|
| `*_data_parallelism` | Batch replicas | Gradient AllReduce |
| `*_fsdp_parallelism` | Parameters, gradients, optimizer state | AllGather and ReduceScatter |
| `*_tensor_parallelism` | Model width or heads | AllReduce/ReduceScatter/AllGather |
| `*_sequence_parallelism` | Sequence activations | Gather/scatter around dependent ops |
| `*_context_parallelism` | Attention context | Attention-specific exchange |
| `*_expert_parallelism` | Expert weights and routed tokens | AllToAll dispatch/combine |
| `*_pipeline_parallelism` | Layer stages | Point-to-point activation transfer |

Set unused factors to `1`. Verify that the product of active factors matches the intended
device mesh and that HLO replica groups match physical placement. `num_slices` must agree
with the launch topology.

## MoE Fields

| Field | Meaning | Current exercised values |
|---|---|---|
| `num_experts` | Routed expert count | `8` in Mixtral |
| `num_experts_per_tok` | Top-k routed experts | `2` in Mixtral |
| `ici_expert_parallelism` | Intra-node expert factor | `1`, `2`, `4`, `8` |
| `sparse_matmul` | Sparse routed-expert path | `false` or `true` |
| `megablox` | MegaBlocks path selector | `false` in current MI355X runners |
| `capacity_factor` | Fixed capacity; `-1.0` denotes dropless in current runners | `1.0`, `-1.0` |
| `ragged_buffer_factor` | Ragged buffer sizing | `-1.0` in current runners |
| `moe_dispatch_no_expert_sharding` | Dispatch layout control | `true` |
| `use_custom_sort_vjp` | Custom gradient for token sort | `true` |
| `use_ragged_sort` | Ragged sorting path | `false` |
| `use_ring_of_experts` | Ring expert execution | `false` |
| `use_tokamax_gmm` | Tokamax grouped GEMM | `false` |
| `load_balance_loss_weight` | Auxiliary balancing loss | `0.0` in current Mixtral config |
| `float32_gate_logits` | Router-logit precision | `false` in current Mixtral config |

For `sparse_matmul=true`, prove whether the run used GroupedGEMM or dense-padded lowering.
The config field alone does not distinguish them.

## Data and Evaluation

| Field | Current convergence setting |
|---|---|
| `dataset_type` | `hf` |
| `hf_path` | `json` |
| `hf_train_files` / `hf_eval_files` | Pinned local C4 glob |
| `train_split` / `hf_eval_split` | `train` |
| `tokenizer_path` | Verified local Llama 2 tokenizer |
| `enable_data_shuffling` | `true` |
| `data_shuffle_seed` | `20260823` |
| `packing` | `true` |
| `max_segments_per_seq` | `32` |
| `eval_start_step` | `0` |
| `eval_interval` | `100` |
| `eval_steps` | `20` |
| `metrics_file` | Per-arm JSONL output |

Synthetic train-step runs use `dataset_type=synthetic` and `reuse_example_batch=1`. Do not
compare their tokens/s with a real-data run without measuring the input cost.

## Checkpoint and Metrics

| Field | Purpose |
|---|---|
| `enable_checkpointing` | Create checkpoints |
| `async_checkpointing` | Overlap supported writes with training |
| `save_checkpoint_on_completion` | Emit a final checkpoint |
| `enable_tensorboard` | TensorBoard-compatible metrics |
| `enable_goodput_recording` | Goodput records |
| `monitor_goodput` | Runtime goodput monitor |
| `profiler` | MaxText profiler selector |
| `dump_hlo` | HLO dump selector |
| `jax_cache_dir` | Persistent JAX compilation cache |

The current single-node timing and convergence configs disable checkpointing.
Any future multi-node study must add and validate save and restore settings.

## XLA Flags

The shared checked-in baseline is:

```text
--xla_gpu_autotune_level=4
--xla_gpu_enable_triton_gemm=true
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_all_gather_combine_threshold_bytes=8589934592
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
--xla_gpu_all_reduce_combine_threshold_bytes=8589934592
--xla_gpu_enable_command_buffer=
```

The matched sparse Mixtral arms additionally set:

```text
--xla_gpu_enable_cublaslt=true
--xla_gpu_experimental_use_ragged_dot_grouped_gemm=<true|false>
--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false
--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=true
```

Experimental and unsupported flags are pinned to the exact JAX/XLA revision. Retest them
after every stack change.

## Environment Variables

| Variable | Purpose |
|---|---|
| `JAX_PLATFORMS=rocm` | Select ROCm without probing other backends |
| `HIP_VISIBLE_DEVICES` | Process-local GPU visibility |
| `XLA_FLAGS` | Compiler/runtime controls |
| `NVTE_FRAMEWORK=jax` | Select Transformer Engine's JAX path |
| `NVTE_FUSED_ATTN_CK=1` | Select CK fused attention in current Llama 7B runners |
| `MAXTEXT_ROOT` | Stock MaxText checkout |
| `MAXTEXT_MXFP4_ROOT` | MXFP4 MaxText checkout |
| `JAX_AITER_ROOT` | JAX-AITER checkout |
| `DATA_ROOT` | Dataset and tokenizer root |
| `OUTPUT_ROOT` | Result root |
| `HF_DATASETS_OFFLINE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | Prevent convergence runs from fetching mutable inputs |

Capture the effective values, imported module paths, and source commits with every run.
