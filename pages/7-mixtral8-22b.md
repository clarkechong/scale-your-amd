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

