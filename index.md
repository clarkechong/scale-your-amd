---
layout: distill
title: "Training on MI355X with JAX and ROCm"
subtitle: "An AMD Companion to How To Scale Your Model"
description: "How MI355X hardware, ROCm kernels, JAX compilation, and MaxText configuration affect training throughput."
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
  - name: "What This Book Covers"
  - name: "How Claims Are Supported"
  - name: "Reader and Prerequisites"
  - name: "Reading Paths"
  - name: "Versioning and Attribution"
  - name: "Part I: The MI355X Execution Contract"
  - name: "Part II: The Configuration Surface"
  - name: "Part III: Case Studies and Measurement Plans"
  - name: "Part IV: Future Multi-node Work"
  - name: "Appendices"
---

The [JAX Scaling Book](https://jax-ml.github.io/scaling-book/) already gives a
careful treatment of rooflines, Transformer accounting, sharded matrix
multiplication, and training parallelism. This book uses the same foundation for a
narrower question:

> Given a fixed training workload on MI355X, which JAX and MaxText configuration
> produces the most tokens per second per GPU without changing the learning
> behaviour?

The answer depends on details that a hardware-independent treatment cannot settle:
which ROCm kernel XLA selects, which low-precision paths carry gradients, how Shardy
maps logical axes onto an eight-GPU baseboard, and whether a nominally supported
feature runs quickly on the declared software stack.

## What This Book Covers

The current evidence scope is one MI355X or one eight-GPU MI355X UBB 2.0 node. The
software path is JAX, XLA, ROCm, and MaxText. The main subject is pre-training
throughput. Validation loss is used as a guardrail when a performance feature changes
numerics.

Multi-node MI355X operation is a future acceptance specification. No current
chapter presents a multi-node performance result.

The book does not cover production serving. For KV-cache economics, continuous
batching, speculative decoding, and serving-engine design, use the inference chapters
of the Scaling Book and the documentation for vLLM or SGLang.

Generic theory is repeated only when it is needed to understand an AMD result. Each
recap links to the fuller derivation in the Scaling Book, substitutes MI355X
constants, and then moves to the JAX or MaxText control that the reader can use.

## How Claims Are Supported

Results use four labels:

- **`[source]`**: established by checked-in code, configuration, or a manifest.
  It proves intent or implementation, not device execution.
- **`[measured]`**: collected on the stated MI355X system and retained in a
  complete Appendix F bundle.
- **`[analytical]`**: calculated from model shapes or published hardware
  specifications.
- **`[cited]`**: supported by a named external specification, document, or
  measurement.

Software support is dated and tied to exact versions. The status vocabulary is
`available`, `experimental`, `unsupported`, `fallback`, `slow fallback`,
`deprecated`, `no-op`, and `unverified`. Status is reported separately from the
theoretical value of a feature.

Tokens per second per GPU is the main performance metric. Step time, MFU, peak HBM,
compile time, and exposed collective time explain that result. Precision comparisons
also report validation loss. The Llama 70B experiment owner reports near-identical
curves over approximately one billion nominal token positions, but the metrics,
plot, and arm-to-run provenance are not yet available as a measured artifact.

## Reader and Prerequisites

The intended reader knows Python, basic JAX, and the main Transformer blocks. The
book introduces AMD hardware, the ROCm execution path, and the sharding and
profiling details needed by the experiments. Readers who want the complete generic
derivations should use the corresponding Scaling Book chapters linked throughout.

## Reading Paths

- For a first MI355X run, read Chapters 1 through 4, then the Llama 7B case.
- For precision choices, read Chapters 3 through 6, Chapter 8, and the Llama 70B
  case.
- For distributed dense training, read Chapters 6, 7, 10, and 12.
- For MoE training, read Chapters 7 through 10 and the Mixtral case.
- For a performance problem, start with Chapter 4 and Appendix D.
- For exact controls and known failures, use Appendices C and E.

## Versioning and Attribution

Every software claim states the relevant version or commit and its verification
date. A new ROCm, JAX, MaxText, Transformer Engine, or JAX-AITER release triggers
retesting rather than silent carryover.

The book reuses concepts and, where noted, adapted material from the MIT-licensed
JAX Scaling Book. Citations accompany reused derivations and figures. AMD, JAX,
OpenXLA, and OCP specifications are cited where their facts are used.

## Part I: The MI355X Execution Contract

1. [**MI355X as a Training Machine**]({{ '/pages/1-hardware' | relative_url }})
   explains CDNA4, wave-level MFMA, the memory hierarchy, native low-precision
   formats, partition modes, and the eight-GPU Infinity Fabric topology.
2. [**What `jax.jit` Runs on ROCm**]({{ '/pages/2-software' | relative_url }})
   follows a training step from Python through StableHLO and XLA to ROCm libraries,
   generated kernels, FFI calls, and the HIP runtime.
3. [**Predicting One Training Step**]({{ '/pages/3-cost-model' | relative_url }})
   defines the compute, memory, and communication ledgers used by every experiment.
4. [**Measuring and Explaining a Training Step**]({{ '/pages/4-profiling' | relative_url }})
   defines the benchmark protocol and the path from XProf to HLO, `rocprofv3`, and
   hardware counters.

## Part II: The Configuration Surface

5. [**Precision as a Training Decision**]({{ '/pages/5-precision' | relative_url }})
   covers BF16, FP16, FP8, MXFP8, MXFP6, and MXFP4 as per-tensor training recipes.
6. [**Making the Model Fit**]({{ '/pages/6-memory' | relative_url }}) covers
   activation memory, optimizer state, donation, scanned layers, rematerialization,
   gradient accumulation, and sharded initialization.
7. [**From JAX Shardings to a Training Mesh**]({{ '/pages/7-sharding' | relative_url }})
   connects `Mesh`, `PartitionSpec`, and Shardy to RCCL traffic and MaxText
   parallelism fields.
8. [**Kernels Reachable from JAX**]({{ '/pages/8-kernels' | relative_url }})
   compares the dense GEMM, attention, and fused-kernel paths that are available on
   ROCm and shows how to confirm which path ran.
9. [**Mixture-of-Experts on MI355X**]({{ '/pages/9-moe' | relative_url }}) covers
   routing, capacity, dropping, dropless execution, expert kernels, all-to-all
   dispatch, and expert parallelism.
10. [**Compiler, Runtime, and RCCL Controls**]({{ '/pages/10-flags' | relative_url }})
    covers the flags used by the experiments, including autotuning, collective
    combining, latency hiding, command buffers, and RCCL controls.

## Part III: Case Studies and Measurement Plans

11. [**Llama 7B: Exposing the Complete Stack**]({{ '/pages/11-llama7b' | relative_url }})
    starts with a training-only Flax implementation. The source defines comparisons between raw JAX and
    MaxText, four attention paths, three rematerialization policies, and single-GPU
    with FSDP-8 execution. A consolidated v26.6 result bundle is still blocked.
12. [**Llama 70B: Dense Low-Precision Training**]({{ '/pages/12-llama70b' | relative_url }})
    defines FP32, BF16, FP16, FP8, MXFP8, and MXFP4 arms under FSDP-8. Historical
    timing observations require a controlled rerun; convergence provenance is
    blocked.
13. [**Mixtral 8x22B: Topology Meets Sparse Kernels**]({{ '/pages/13-mixtral8-22b' | relative_url }})
    defines FSDP and expert-parallel mesh, expert-path, and latency-hiding sweeps.
    No v26.6 performance result exists yet.

Case-study sections without captured artifacts are marked as blocked. Planned
measurements are not presented as results.

## Part IV: Future Multi-node Work

14. [**Operating Multi-node MI355X Training**]({{ '/pages/14-multinode' | relative_url }})
    specifies the launch, topology, data, checkpoint, restart, and scaling evidence
    required for multi-node claims.
15. [**Future Capstone: DeepSeek V3**]({{ '/pages/15-deepseek-v3' | relative_url }})
    defines the model ledger, mesh choices, component tests, and acceptance criteria
    for a later multi-node study. It contains no frontier-performance claim before
    those measurements exist.

## Appendices

- [**Appendix A: Reproducible Environment**]({{ '/pages/a-appendix-install' | relative_url }})
- [**Appendix B: Measurement Protocol**]({{ '/pages/b-appendix-protocol' | relative_url }})
- [**Appendix C: Configuration Reference**]({{ '/pages/c-appendix-config' | relative_url }})
- [**Appendix D: Profiler and HLO Cookbook**]({{ '/pages/d-appendix-tooling' | relative_url }})
- [**Appendix E: Compatibility and Negative Results**]({{ '/pages/e-appendix-compatibility' | relative_url }})
- [**Appendix F: Case-study Artifacts**]({{ '/pages/f-appendix-artifacts' | relative_url }})

<h3 markdown=1 class="next-section">Next: [Chapter 1, MI355X as a Training Machine]({{ '/pages/1-hardware' | relative_url }}).</h3>
