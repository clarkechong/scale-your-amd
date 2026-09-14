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
    subsections:
      - name: "1. MI355X as a Training Machine"
        url: "/pages/1-mi355x-as-a-training-machine"
      - name: "2. Lowering jax.jit on ROCm"
        url: "/pages/2-lowering-jax-jit-on-rocm"
      - name: "3. Profiling and Analysis of a Training Step"
        url: "/pages/3-profiling-and-analysis-of-one-training-step"
  - name: "Part II: JAX Performance Features on ROCm"
    subsections:
      - name: "4. Training in Mixed Precision"
        url: "/pages/4-training-in-mixed-precision"
      - name: "5. Making the Model Fit"
        url: "/pages/5-making-the-model-fit"
      - name: "6. Parallelism Strategies for Higher Throughput"
        url: "/pages/6-jax-shardings-to-a-training-mesh"
      - name: "7. A Map of ROCm Kernel Backends on JAX"
        url: "/pages/7-a-map-of-kernel-backends-on-jax"
      - name: "8. Training Mixture-of-Experts on MI355X"
        url: "/pages/8-mixture-of-experts-on-mi355x"
      - name: "9. Tuning the Compiler, Runtime, and RCCL"
        url: "/pages/9-compiler-runtime-and-rccl-controls"
  - name: "Part III: Case Studies: Expectations and Results"
    subsections:
      - name: "10. Llama 7B: Exposing the Complete Stack"
        url: "/pages/10-llama-7b-exposing-the-complete-stack"
      - name: "11. Llama 2 70B: Mixed Precision Training"
        url: "/pages/11-llama-2-70b-mixed-precision-training"
      - name: "12. Mixtral 8x22B: Sharding Meshes and MoE Optimizations"
        url: "/pages/12-mixtral-8x22b-sharding-meshes-and-moe-optimizations"
  - name: "Appendices"
  - name: "Attribution"
---

## What This Book Covers

The famous [JAX Scaling Book](https://jax-ml.github.io/scaling-book/) teaches a comprehensive understanding on TPU architecure, rooflines, Transformer analysis, sharding behaviour, training parallelism, and much more. 

This ROCm book serves as complementary material for readers to understand the ROCm stack for JAX, including ROCm experimental features, and in general how to extract the best training performance out of AMD GPU's with JAX.

Namely,

> For a given training workload on MI355X, how can we achieve the best tokens/s/gpu with JAX and MaxText without changing the learning behaviour.

But perhaps more importantly, readers should be aware of the performance features available within the JAX/XLA/MaxText ecosystem, understand their underlying mechanisms, and know where to find further information.

Given the pace of active development, the JAX/ROCm software stack evolves rapidly, with new optimizations and capabilities often arriving faster than documentation can be updated. Consequently, the goal is not merely to present performance recommendations for the current state of the ecosystem, but to provide readers with the context necessary to discover new developments, evaluate their applicability, and incorporate optimizations suited to their own training environments.

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
3. [**Profiling and Analysis of a Training Step**]({{ '/pages/3-profiling-and-analysis-of-one-training-step' | relative_url }})
   predicts one optimizer update, measures it cleanly, and traces performance gaps
   through HLO, XProf, `rocprofv3`, ROCTx, and hardware counters.

## Part II: JAX Performance Features on ROCm

The chapters follow one configuration decision in order: choose the numeric recipe,
make one optimizer update fit, distribute it for throughput, select kernels for the
resulting local operations, apply those decisions together for sparse models, and
only then tune compiler or runtime controls.

4. [**Training in Mixed Precision**]({{ '/pages/4-training-in-mixed-precision' | relative_url }})
   covers BF16, FP16, FP8, MXFP8, MXFP6, and MXFP4 as per-tensor training recipes.
5. [**Making the Model Fit**]({{ '/pages/5-making-the-model-fit' | relative_url }})
   covers persistent and activation memory, donation, rematerialization, accumulation,
   FSDP as a capacity tool, sharded initialization, and offload.
6. [**Parallelism Strategies for Higher Throughput**]({{ '/pages/6-jax-shardings-to-a-training-mesh' | relative_url }})
   connects `Mesh`, `PartitionSpec`, and Shardy to local shapes, collectives,
   parallelism strategies, and eight-GPU placement.
7. [**A Map of ROCm Kernel Backends on JAX**]({{ '/pages/7-a-map-of-kernel-backends-on-jax' | relative_url }})
   selects and proves dense GEMM, attention, and fused forward/backward kernel paths
   for the local operations produced by the precision and mesh decisions.
8. [**Training Mixture-of-Experts on MI355X**]({{ '/pages/8-mixture-of-experts-on-mi355x' | relative_url }})
   integrates sparse-model memory, routing, capacity, expert kernels, AllToAll
   traffic, expert parallelism, rematerialization, and diagnostics.
9. [**Tuning the Compiler, Runtime, and RCCL**]({{ '/pages/9-compiler-runtime-and-rccl-controls' | relative_url }})
   changes autotuning, collective combining, latency hiding, command buffers, and
   RCCL controls only after a profile identifies the mechanism.

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

## Attribution


The book reuses concepts and, where noted, adapted material from the MIT-licensed
JAX Scaling Book. Citations accompany reused derivations and figures. AMD, JAX,
OpenXLA, and OCP specifications are cited where their facts are used.

<h3 markdown=1 class="next-section">Next: [Chapter 1, MI355X as a Training Machine]({{ '/pages/1-mi355x-as-a-training-machine' | relative_url }}).</h3>
