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
next_section_name: "Chapter 1: MI355X as a Training Machine"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "Motivation"
  - name: "Reader and Prerequisites"
  - name: "Chapters"
    subsections:
      - name: "1. MI355X as a Training Machine"
        url: "/pages/1-mi355x-as-a-training-machine"
      - name: "2. The JAX Software Stack on ROCm"
        url: "/pages/2-the-jax-software-stack-on-rocm"
      - name: "3. Profiling a Training Step"
        url: "/pages/3-profiling"
      - name: "4. Training in Mixed Precision"
        url: "/pages/4-mixed-precision"
      - name: "5. Sharding, Meshes, and Parallelism"
        url: "/pages/5-sharding"
      - name: "6. Memory and Kernel Optimizations"
        url: "/pages/6-mem-and-kernel-optimizations"
  - name: "Attribution"
---

## Motivation

The [JAX Scaling Book](https://jax-ml.github.io/scaling-book/) explains
rooflines, Transformer math, sharding, and training parallelism largely through
TPU systems. This companion applies those methods to JAX training on AMD
MI355X GPUs and follows the software path through XLA and ROCm.

> For a fixed MI355X training workload, how can JAX and MaxText maximize
> tokens/s/GPU without changing the learning behavior?

The ROCm/JAX stack changes quickly. Each chapter therefore combines the
current versioned results with the mechanisms and evidence needed to evaluate
new compiler passes, libraries, and kernels as they become available.

## Reader and Prerequisites

The intended reader knows Python, basic JAX, and the main Transformer blocks. The
book introduces AMD hardware, the ROCm execution path, and the sharding and
profiling details needed by the experiments. Readers who want the complete generic
derivations should use the corresponding Scaling Book chapters linked throughout.

## Chapters

The six chapters move from the machine and compiler stack to the decisions that
change a training step:

1. [**MI355X as a Training Machine**]({{ '/pages/1-mi355x-as-a-training-machine' | relative_url }})
   explains CDNA 4 matrix execution, memory, native precision formats, and the
   eight-GPU Infinity Fabric topology.
2. [**The JAX Software Stack on ROCm**]({{ '/pages/2-the-jax-software-stack-on-rocm' | relative_url }})
   follows a JAX program through tracing, StableHLO, XLA, ROCm libraries, FFI,
   and device execution.
3. [**Profiling a Training Step**]({{ '/pages/3-profiling' | relative_url }})
   connects roofline estimates to XProf, `rocprofv3`, hardware counters, and
   `rocprof-compute`.
4. [**Training in Mixed Precision**]({{ '/pages/4-mixed-precision' | relative_url }})
   treats BF16, FP16, FP8, MXFP8, MXFP6, and MXFP4 as per-operation training
   recipes and tests them with Llama 70B.
5. [**Sharding, Meshes, and Parallelism**]({{ '/pages/5-sharding' | relative_url }})
   connects JAX meshes and Shardy to local tensor shapes, collectives, FSDP,
   expert parallelism, and Mixtral.
6. [**Memory and Kernel Optimizations**]({{ '/pages/6-mem-and-kernel-optimizations' | relative_url }})
   covers rematerialization, ROCm attention backends, and grouped GEMM lowering
   for routed experts.

## Attribution


The book reuses concepts and, where noted, adapted material from the MIT-licensed
JAX Scaling Book. Citations accompany reused derivations and figures. AMD, JAX,
OpenXLA, and OCP specifications are cited where their facts are used.

<h3 markdown=1 class="next-section">Next: [Chapter 1, MI355X as a Training Machine]({{ '/pages/1-mi355x-as-a-training-machine' | relative_url }}).</h3>
