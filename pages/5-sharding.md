5-sharding, meshes, parallelism

1. revisiting xla: shardy as the partitioner
- shardy is a mechanism, simply a mechanism, by which we can implement sharding strategies such as zero/fsdp
- to instructionally demonstrate sharding effects at a hlo level we dont necessarily haev to be under fsdp or tp or ddp strategy. demonstrating shardy can be simply as demonstrating how 
    a) matmul -> split into collectives (maybe arbitrarily for sake of it) across devices before actual compute, 
    b) demonstrate fsdp is just the same thing, but we're splitting components more intentionally
    c) ep again just splitting in different ulterior motive. shardy simply implements the stratgy being described
    https://jax-ml.github.io/scaling-book/sharding/
- main thing is to make sure we demonstrate the hlo before and after the shardy partitioning, highlighting how sharding{} transforms into collectives.

2. parallelism strategies
- likewise, fsdp and zero are just standardized partitioning recipes
- https://jax-ml.github.io/scaling-book/training/ covers the theory of parallelism stratgies in detail

4. deepseekv3 again as a prominent example of frontier preferences for training meshes:
    - dense models, no ep parallelism (like llama), which means fsdp for making the model fit is a priority
    - moe itself is a way to parallelize the mlp/ff layer in architecture. which means we still parallelize to fit, but along an independent axis (The experts!)
    - extreme ep parallelism eg. deepseekv3 pushes further into high ep (smaller experts, smaller independently parallelisable units, more incentize to parallelize along expert axis) but further emphasis on collectives

3. how does this get implemented in jax?
    - at the jax level, https://jax-ml.github.io/scaling-book/jax-stuff/ explains sharding modes
    - at maxtext level we can specify fsdp and ep parallelism for standard shemes
    - implications on the device mesh: want to be able to explain why fsdp8 ep8 indicates a 64-devices mesh (orthogonal mesh axis)



4. CASE STUDY of mixtral
    - with fsdp,ep: 1,8 and 2,4 and 4,2 and 8,1
    - more ep=alltoall, more fsdp=allgather/reduce-scatter
    - the point is, fsdp,ep are also system engineering decisions not purely model implementation details
        - hardware platform can influence which path to go down. so we should teach how to determine for yourself and the correct profiling approach to identify which meshes to use
    - Predict memory and communication costs
    - Measure FSDP scaling
    - Measure EP scaling
    - Analyze HLO collective patterns
    - Relate findings to DeepSeek-style MoE training

5. opportunity for extended case study of multinode mesh configurations and how the scaling of communication is affected

---

## Sharding as a compiler transformation

### Shardy
- xla's partitioner. what does that mean? it needs to insert collectives etc etc. you describe a sharding configuration using shardy api's, and it will implement it
- shardy is a mechanism, simply a mechanism, by which we can implement sharding strategies such as zero/fsdp

### Changes at the HLO level
- to instructionally demonstrate sharding effects at a hlo level we dont necessarily haev to be under fsdp or tp or ddp strategy. demonstrating shardy can be simply as demonstrating how 
    a) matmul -> split into collectives (maybe arbitrarily for sake of it) across devices before actual compute, 
    b) demonstrate fsdp is just the same thing, but we're splitting components more intentionally
    c) ep again just splitting in different ulterior motive. shardy simply implements the stratgy being described
    https://jax-ml.github.io/scaling-book/sharding/
- main thing is to make sure we demonstrate the hlo before and after the shardy partitioning, highlighting how sharding{} transforms into collectives.

## Parallelism strategies
### FSDP, EP 
- likewise, fsdp and zero are just standardized partitioning recipes
- https://jax-ml.github.io/scaling-book/training/ covers the theory of parallelism stratgies in detail
- scale to fit parameters, vs scaling for throughput
- dense models, no ep parallelism (like llama), which means fsdp for making the model fit is a priority

### Deepseek V3
- moe itself is a way to parallelize the mlp/ff layer in architecture. which means we still parallelize to fit, but along an independent axis (The experts!)
- extreme ep parallelism eg. deepseekv3 pushes further into high ep (smaller experts, smaller independently parallelisable units, more incentize to parallelize along expert axis) but further emphasis on collectives

## Implementing in JAX
### JAX sharding meshes
- at the jax level, https://jax-ml.github.io/scaling-book/jax-stuff/ explains sharding modes
- implications on the device mesh: want to be able to explain why fsdp8 ep8 indicates a 64-devices mesh (orthogonal mesh axis)
### MaxText
- at maxtext level we can specify fsdp and ep parallelism for standard shemes
- ICI? DCN?
- trace through the source again, and a diagram of how maxtext logically treats sharding meshes

# Case study: Mixtral 8x22b
- with fsdp,ep: 1,8 and 2,4 and 4,2 and 8,1
- more ep=alltoall, more fsdp=allgather/reduce-scatter
- the point is, fsdp,ep are also system engineering decisions not purely model implementation details
    - hardware platform can influence which path to go down. so we should teach how to determine for yourself and the correct profiling approach to identify which meshes to use
- Predict memory and communication costs
- Measure FSDP scaling
- Measure EP scaling
- Analyze HLO collective patterns
