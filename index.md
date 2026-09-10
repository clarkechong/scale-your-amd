---
layout: distill
title: "How To Scale Your Model with AMD"
subtitle: "A Systems View of LLMs on AMD GPUs"
description: "Given a model and some number of MI355X-class GPUs, how do I run it in JAX so that adding GPUs adds throughput?"
date: 2026-09-10

section_number: 0

previous_section_url: ""
previous_section_name: "Chapter 0: Intro"

next_section_url: "/pages/1-hardware"
next_section_name: "Chapter 1: Hardware"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Hardware
  - name: Software
  - name: DL (Performance) Theory
  - name: Analysis
  - name: "Case Study 1: Training Llama 7B in JAX (No MaxText)"
  - name: "Case Study 2: Training Llama 70B in JAX (MaxText)"
  - name: "Case Study 3: Mixtral 8x22B (MaxText) (Sharded MoE)"
  - name: "Case Study 4: DeepSeek V3 (needs multinode)"
---

**One sentence: given a model and some number of MI355X-class GPUs, how do I run it
in JAX so that adding GPUs adds throughput?** The chapters below are the path through
that question, from a single GPU up to a multi-node frontier run.

## Hardware

[Chapter 1: Hardware]({{ '/pages/1-hardware' | relative_url }}).

- Single GPU architecture
- Heterogeneous systems (host, device)
- Node architecture and topology (NVLink, InfiniFabric), extended to multinode (RDMA)

## Software

[Chapter 2: Software]({{ '/pages/2-software' | relative_url }}).

- XLA compiler (debug, lgas.cc), HLO, LLVM
- Backends:
  - hipBLAS
  - cuBLAS
  - Triton
  - XLA own HIP codegen?
  - Fusion
  - RCCL
- AITER

## DL (Performance) Theory

[Chapter 3: Deep Learning Methods]({{ '/pages/3-dl-methods' | relative_url }}).

- Transformers, MoE
- Shardy
  - Distributed methods:
    - DDP
    - FSDP (sharding)
    - TP
    - Collectives
  - Shardy is the framework/infra to implement sharding strategies in JAX
  - It is what generates the collectives
  - Example:
    - Specify sharding config in Shardy
    - Generate collectives
- Attention
  - Flash Attention
  - (Paged) KV Cache
  - Multi Query Attention
  - Grouped Query Attention
- Checkpointing
  - `jax.checkpoint = jax.remat = rematerialisation`
  - Instead of storing activations, recompute them in another forward pass
  - Trade memory for compute

## Analysis

[Chapter 4: Profiling]({{ '/pages/4-profiling' | relative_url }}).

- Roofline Analysis
  - Theoretical based first, e.g.
  - Arithmetic intensity
  - FLOPs
  - Datatypes
- trace analysis with XProf (cover XSpace `.pb` format)
- trace analysis with `roofprofv3`
- hardware counter analysis with `rocprof-compute`
- debugging with `rocgdb`
- end to end:
  - Tokens/sec
  - Roofline (utilisation)
  - Peak memory usage (VRAM + bandwidth)
  - Convergence time

## Case Study 1: Training Llama 7B in JAX (No MaxText)

[Chapter 5: Training Llama 7B]({{ '/pages/5-llama7b' | relative_url }}).

### 1 GPU Fundamentals

### Back-of-the-Hand Peak Memory Calculation

(7B in fp32 + Adam optimizer states in bf16)

- If bf16, approx 7B × 2 bytes × 4 (optimizer) = 56 GB
- But actually there is:
  - fp32 (master) copy
  - bf16 (working) copy of weights
- fp32 used to store full-precision copy, downcast to bf16 for forward-pass compute

### Expected Step Time Calculation

- Quick estimate:
  - Based on FLOPs?
  - Based on links and bandwidth?

## Case Study 2: Training Llama 70B in JAX (MaxText)

[Chapter 6: Training Llama 70B]({{ '/pages/6-llama70b' | relative_url }}).

### 8 GPU Large Dense Model

### Peak Memory Usage

- Again, quick calculation for peak memory usage

### Mixed Precision Training Showcase

- fp32
- fp16
- bf16
- fp8
- mx-fp8

## Case Study 3: Mixtral 8x22B (MaxText) (Sharded MoE)

[Chapter 7: Mixtral 8x22B]({{ '/pages/7-mixtral8-22b' | relative_url }}).

- requires FSDP + expert parallelism

### MoE Communication Patterns

- Group GEMM / ragged dot
  - Token routing
  - All-to-all collective

### Kernels

- Try out `jax-aiter` kernels

## Case Study 4: DeepSeek V3 (needs multinode)

[Chapter 8: DeepSeek V3]({{ '/pages/8-deepseek-v3' | relative_url }}).

- Most complicated case
- Focus on end-to-end showcase of frontier performance on MI355 node

<h3 markdown=1 class="next-section">Without further ado, [here is Chapter 1 on hardware]({{ '/pages/1-hardware' | relative_url }}).</h3>
