6-Memory and Kernel Optimizations

1. Rematerialization, Checkpointing
    - https://jax-ml.github.io/scaling-book/transformers/#gradient-checkpointing
    - https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html
    - explanation of remat policies
    - expectations of memory reduction vs compute tradeoff (some math is covered in scaling book)
    - changes at the HLO level (how flow of collectives are affected?)
    - results on llama7b

2. Attention Backend Implementations
    - clarify: forward or backward attention
    - brief reintroduction to amd backends, and a deeper dive into each. specifically, how do their implementations differ ie what causes the difference in performance:
        - eg TE/CK focuses on XYZ which affects tiling or something, whereas aiter focuses on ABC, .... ?
    - visually explain how the tiling sizes for these backends differered when they were lowered.
    - (we have to do this from kernel traces as hlo can only show us custom call) (apart from xla attention route) (and maybe pallas/triton?)
    - results on llama7b
    

3. MoE Kernels and Grouped GEMM
    - with moe token routing, the bottom line is, experts receive different # of tokens. hence the gemm sizes that each device needs to perform are uneven
    - at the op level:
        - we can pad everything
        - we can leave uneven as ragged dot op
    - during lowering, if we pad everything, we go down dense gemm path
    - with ragged dot, we would like to execute groupedGEMM. otherwise the ragged dot op can still lower through a dense masked gemm which is inefficient.
    - how do we lower with ragged dot in maxtext, and how does it affect the jax-level implementation (Trace through maxtext and find relevent snippets that can demonstrate this)
    - with maxtext flags (use_ragged_dot, sparse_matmul): how does the hlo lower differently
        - specifically, the handoff from the router -> expert gemm -> output

---

## Rematerialization and checkpointing
### Activations in training memory
### What is rematerialization
### Implementing in JAX
### Changes at the HLO level
### Case study: Llama 7B
- estimate memory reduction and compute increase
- compare with actual measurements
- compare effect of remat on 1gpu vs 8gpu

## Attention backend implementations
### Components of attention
- diagram to break into logical components
### ROCm attention backends
### Case study: Llama 7B

## Grouped GEMM kernels for MoE
### Why MoE Requires Grouped GEMM
- with moe token routing, the bottom line is, experts receive different # of tokens. hence the gemm sizes that each device needs to perform are uneven
- at the op level:
    - we can pad everything
    - we can leave uneven as ragged dot op
- during lowering, if we pad everything, we go down dense gemm path
- with ragged dot, we would like to execute groupedGEMM. otherwise the ragged dot op can still lower through a dense masked gemm which is inefficient.
- how do we lower with ragged dot in maxtext, and how does it affect the jax-level implementation (Trace through maxtext and find relevent snippets that can demonstrate this)
### Grouped GEMM lowering paths
- with maxtext flags (use_ragged_dot, sparse_matmul)
- ffi custom calls to TE, hipblasLT, ... needs investigation what paths are available and how their tiling and/or other characteristics differ
### Changes at the HLO Level
- how does the hlo lower differently (ragged dot, vs non ragged dot dense masked)
- specifically, the handoff from the router -> expert gemm -> output
### Case Study: Mixtral 8x22B
- short: just present results and conclude that this is current WIP
