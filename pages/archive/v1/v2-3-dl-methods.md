---
layout: distill
title: "Deep Learning Methods"
description: "Transformer, MoE, sharding, remat, and the inference tricks that change the cost model."
date: 2026-09-10

section_number: 3

previous_section_url: "/pages/2-software"
previous_section_name: "Chapter 2: Software"

next_section_url: "/pages/4-profiling"
next_section_name: "Chapter 4: Profiling"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "OUTLINE"
---
## OUTLINE

prerequisites: standard LLM transformer architecture (transformerblock=att+ffn)

- (theory):
    - MoE
    - attention
        - multihead att
        - multi query att (MQA)
        - grouped query att (GQA)
    - RoPE position embedding

- (execution methods)
    - sharding (DDP, FSDP, TP, PP)
        - collectives
        - mesh configurations
    - activation checkpointing (remat)
    - mixed precision training

- (inference)
    - KV Cache
    - Paged KV Cache
    - Continuous Batching
    - Speculative Decoding
    - Flash Attention
    - Prefix Caching
    - Quantization (INT8, FP8, 4-bit)
