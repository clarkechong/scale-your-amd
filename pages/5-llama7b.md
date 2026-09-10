---
layout: distill
title: "Training Llama 7B"
description: "A single-GPU Llama 7B in raw JAX, then the same model in MaxText, with remat and attention-backend comparisons."
date: 2026-09-10

section_number: 5

previous_section_url: "/pages/4-profiling"
previous_section_name: "Chapter 4: Profiling"

next_section_url: "/pages/6-llama70b"
next_section_name: "Chapter 6: Llama 70B"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

---
1. training llama7b in raw jax (fp32 master, bf16 compute)
    - what is the expected: memory footprint, throughput FLOPs?
    - what is the achieved: memory footprint, throughput FLOPs?
    - use the jax profiler in practise and identify signature things in the xprof trace. demonstrate framework level op correlation
    - use the rocprof profiler as a comparison as to what we are missing (if we dont use our own roctx ranges)
2. how to implement the equivalent model in maxtext
    - how maxtext works (use config file, specify experiement parameters in the config, call the runner py file)
    - a minimal setup, config + runner
    - how did step time change? why? what about hlo lowering?
3. in maxtext, investigation into jax.remat
    - Compare `minimal_with_context`, `full`, and `none`
    - what kind of memory footprint vs throughput tradeoff should we expect?
    - what kind of memory footprint vs throughput tradeoff did we achieve?
    - does third party research have similar findings?
4. in raw jax, investigate into attention backends
    - xla, te, aiter, triton
    - (raw-JAX BF16 XLA, TransformerEngine CK, direct JAX-AITER, and Tokamax, Pallas-Triton attention)
