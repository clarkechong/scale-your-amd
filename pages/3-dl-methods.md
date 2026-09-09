# OUTLINE

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
