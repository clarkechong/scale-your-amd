---
layout: distill
title: "Mixtral 8x22B"
description: "Sharded MoE on one node: FSDP plus expert parallelism, token routing, and GroupGEMM."
date: 2026-09-10

section_number: 7

previous_section_url: "/pages/6-llama70b"
previous_section_name: "Chapter 6: Llama 70B"

next_section_url: "/pages/8-deepseek-v3"
next_section_name: "Chapter 8: DeepSeek V3"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

---
1. baseline fp32/bf16, LHS on, fsdp1+ep8 (no sharding ie weights dupcliated across 8 gpus? and ep8 means that each gpu holds unique expert in full)
    - estimate step time and memory consumption
    - compare to achieved step time and memory consumption

2. fsdp mesh experiments: fsdp 
    - how to do this from maxtext config flags
    - estimate effect on step time and memory footprint
    - compare to achieved step time and memory footprint

3. MoE token routing experiments:
    - dense masked (all tokens sent to all experts), raggeddot groupedgemm, dense padded (tokens routed, padded)
    - how to do each of these from maxtext configuration
        - on mi355, groupedgemm is only supported (hipblaslt) for fp16
        - however it is currently untuned and a worse performer than dense padded path
        - work in progress to enable triton grouped gemm which can readily outperform dense padded path
    - expected relationship between them for memory footprint and step time throughput
    - vs achieved step time and memory footprint

4. lhs off
    - improvement in peak memory footprint, vs latency

