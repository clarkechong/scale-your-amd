---
layout: distill
title: "Training on MI355X with JAX and ROCm"
subtitle: "An AMD Companion to How To Scale Your Model"
description: "How MI355X hardware, ROCm kernels, JAX compilation, and MaxText configuration affect training throughput."
date: 2026-09-10

section_number: 0

previous_section_url: ""
previous_section_name: "Chapter 0: Intro"

next_section_url: "/pages/1-mi355x-as-a-training-machine"
next_section_name: "Chapter 1: Hardware"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "What This Book Covers"
  - name: "Reader and Prerequisites"
  - name: "Part I: The JAX Stack on ROCm using MI355X"
  - name: "Part II: JAX Performance Features on ROCm"
  - name: "Part III: Case Studies: Expectations and Results"
  - name: "Appendices"
  - name: "Versioning and Attribution"
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

The current book covers one MI355X or one eight-GPU MI355X node. It does not
present multi-node performance results.

The book does not cover production serving. For KV-cache economics, continuous
batching, speculative decoding, and serving-engine design, use the inference chapters
of the Scaling Book and the documentation for vLLM or SGLang.

Generic theory is repeated only when it is needed to understand an AMD result. Each
recap links to the fuller derivation in the Scaling Book, substitutes MI355X
constants, and then moves to the JAX or MaxText control that the reader can use.

## Reader and Prerequisites

The intended reader knows Python, basic JAX, and the main Transformer blocks. The
book introduces AMD hardware, the ROCm execution path, and the sharding and
profiling details needed by the experiments. Readers who want the complete generic
derivations should use the corresponding Scaling Book chapters linked throughout.

## Part I: The JAX Stack on ROCm using MI355X

1. [**MI355X as a Training Machine**]({{ '/pages/1-mi355x-as-a-training-machine' | relative_url }})
   explains CDNA4, wave-level MFMA, the memory hierarchy, native low-precision
   formats, partition modes, and the eight-GPU Infinity Fabric topology.
2. [**Lowering `jax.jit` on ROCm**]({{ '/pages/2-lowering-jax-jit-on-rocm' | relative_url }})
   follows a training step from Python through StableHLO and XLA to ROCm libraries,
   generated kernels, FFI calls, and the HIP runtime.
3. [**Predicting and Measuring One Training Step**]({{ '/pages/3-predicting-and-measuring-one-training-step' | relative_url }})
   defines the compute, memory, and communication ledgers, then tests them using a
   fixed benchmark protocol, HLO, XProf, `rocprofv3`, and hardware counters.

## Part II: JAX Performance Features on ROCm

4. [**Training in Mixed Precision**]({{ '/pages/4-training-in-mixed-precision' | relative_url }})
   covers BF16, FP16, FP8, MXFP8, MXFP6, and MXFP4 as per-tensor training recipes.
5. [**Making the Model Fit**]({{ '/pages/5-making-the-model-fit' | relative_url }}) covers
   activation memory, optimizer state, donation, scanned layers, rematerialization,
   gradient accumulation, and sharded initialization.
6. [**JAX Shardings to a Training Mesh**]({{ '/pages/6-jax-shardings-to-a-training-mesh' | relative_url }})
   connects `Mesh`, `PartitionSpec`, and Shardy to RCCL traffic and MaxText
   parallelism fields.
7. [**A Map of Kernel Backends on JAX**]({{ '/pages/7-a-map-of-kernel-backends-on-jax' | relative_url }})
   compares the dense GEMM, attention, and fused-kernel paths that are available on
   ROCm and shows how to confirm which path ran.
8. [**Mixture-of-Experts on MI355X**]({{ '/pages/8-mixture-of-experts-on-mi355x' | relative_url }}) covers
   routing, capacity, dropping, dropless execution, expert kernels, all-to-all
   dispatch, and expert parallelism.
9. [**Compiler, Runtime, and RCCL Controls**]({{ '/pages/9-compiler-runtime-and-rccl-controls' | relative_url }})
    covers the flags used by the experiments, including autotuning, collective
    combining, latency hiding, command buffers, and RCCL controls.

## Part III: Case Studies: Expectations and Results

10. [**Llama 7B: Exposing the Complete Stack**]({{ '/pages/10-llama-7b-exposing-the-complete-stack' | relative_url }})
    starts with a training-only Flax implementation. The source defines comparisons between raw JAX and
    MaxText, four attention paths, three rematerialization policies, and single-GPU
    with FSDP-8 execution. A consolidated v26.6 result bundle is still blocked.
11. [**Llama 2 70B: Mixed Precision Training**]({{ '/pages/11-llama-2-70b-mixed-precision-training' | relative_url }})
    defines FP32, BF16, FP16, FP8, MXFP8, and MXFP4 arms under FSDP-8. Historical
    timing observations require a controlled rerun; convergence provenance is
    blocked.
12. [**Mixtral 8x22B: Sharding Meshes and MoE Optimizations**]({{ '/pages/12-mixtral-8x22b-sharding-meshes-and-moe-optimizations' | relative_url }})
    defines FSDP and expert-parallel mesh, expert-path, and latency-hiding sweeps.
    No v26.6 performance result exists yet.

Case-study sections without captured artifacts are marked as blocked. Planned
measurements are not presented as results.

## Appendices

- [**Appendix A: Reproducible Environment**]({{ '/pages/a-reproducible-mi355x-environment' | relative_url }})
- [**Appendix B: Measurement Protocol**]({{ '/pages/b-measurement-and-convergence-protocol' | relative_url }})
- [**Appendix C: Configuration Reference**]({{ '/pages/c-configuration-quick-reference' | relative_url }})
- [**Appendix D: Profiler and HLO Cookbook**]({{ '/pages/d-profiler-and-hlo-cookbook' | relative_url }})
- [**Appendix E: Compatibility and Negative Results**]({{ '/pages/e-compatibility-and-negative-results' | relative_url }})
- [**Appendix F: Case-study Artifacts**]({{ '/pages/f-case-study-artifact-schema' | relative_url }})

## Versioning and Attribution

Every software claim states the relevant version or commit and its verification
date. A new ROCm, JAX, MaxText, Transformer Engine, or JAX-AITER release triggers
retesting rather than silent carryover.

The book reuses concepts and, where noted, adapted material from the MIT-licensed
JAX Scaling Book. Citations accompany reused derivations and figures. AMD, JAX,
OpenXLA, and OCP specifications are cited where their facts are used.

<h3 markdown=1 class="next-section">Next: [Chapter 1, MI355X as a Training Machine]({{ '/pages/1-mi355x-as-a-training-machine' | relative_url }}).</h3>
