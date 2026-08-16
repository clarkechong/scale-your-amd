## Hardware

- Single GPU architecture
- Heterogeneous systems (host, device)
- Node architecture and topology (NVLink, InfiniFabric), extended to multinode (RDMA)

## Software

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

---



# Case Study 1: Training Llama 7B in JAX (No MaxText)



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

---



# Case Study 2: Training Llama 70B in JAX (MaxText)



### 8 GPU Large Dense Model



### Peak Memory Usage

- Again, quick calculation for peak memory usage



### Mixed Precision Training Showcase

- fp32
- fp16
- bf16
- fp8
- mx-fp8

---



# Case Study 3: Mixtral 8x22B (MaxText) (Sharded MoE)

- requires FSDP + expert parallelism
- 



### MoE Communication Patterns

- Group GEMM / ragged dot
  - Token routing
  - All-to-all collective



### Kernels

- Try out `jax-aiter` kernels

---



# Case Study 4: DeepSeek V3 (needs multinode)

- Most complicated case
- Focus on end-to-end showcase of frontier performance on MI355 node

