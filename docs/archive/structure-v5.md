
---

2-lowering jax.jit on rocm (rename to: The JAX software stack on ROCm)

1. the bigger picture purpose of jax
    - as a functional language, a program is expressed as composition of transformations. no mutable state. general functional advatnages apply: easier to reason about function behaviour, etc
    - this naturally lends to tracing and compiler transformations JIT
    - this section should introduce the idea of tracing, and jaxpr such that we (the reader) is positioned to read the following xla content. (remember xla is not striclty jax! TF and other frameworks use it too, so here we should cover the jax-level specifics)

2. XLA and the GPU Pipeline:
    - openxla has [extensive documentation](https://openxla.org/xla). here is how to navigate it in context of rocm.
    - what is HLO? it is compiler IR
        - compiler primitives are described at https://openxla.org/xla/operation_semantics
    - the overall compiler pipeline is descibed at https://openxla.org/xla/hlo_to_thunks and gpu details covered in https://openxla.org/xla/gpu_architecture. this gpu page also walks thorugh hlo examples of sharding, layout assignment, fusion.
    - https://openxla.org/xla/emitters discusses the XLA emitter, which is one of the paths by which an hlo OP is lowered to its kernel implementation. also quote as written in https://openxla.org/xla/gpu_architecture#compiler_backend_codegen_and_library_selection:
        a) library selection (custom calls) b) xla codegen c) triton codegen
        lets elaborate rocm lowering paths below:

3. A map of ROCm kernel backends
    - a diagram of the lowering path of HLO ops:
        - a jax primitive branches into 1) custom-call 2) XLA lowering
        - XLA lowering branches into XLA gpu emitter vs triton
    - a text list of all these backneds and where to find further documentation

4. maxtext
    - explain maxtext is a framework built on top of jax.
        - a diagram to show one level above jax primitive:
        - maxtext -> a) custom jax primitive = FFI call, b) standard jax = jax primitive path
        - a) FFI call -> custom call, path as described above
    - maxtext works via config flags: how do these config flags actually lower?
    - maxtext changes the jax source. we have explained the xla lowering path, but maxtext given it operates above jax, will modify at the jax layer and insert ffi calls

5. worked example: tracing a maxtext flag (use_te=true) down to its lowering via the xla pipeline
- where is the gated logic
- the relevent snippets of how it lowers differently
- finally just show the hlo with and without

6. worked example: lowering a jax.jit transformer module
- block diagram of a basic transformer block
- follow it after each logical collection of passes:
    - right after jaxpr: no optimizations
    - shardy
    - optimization passes (lowered to rocm custom calls)
    - collectives passes (show whatever this changed)
    - layout assignment
    - jump to final optimized hlo



---

3-profiling

1. prerequisite knowledge
    - roofline analysis

2. JAX/ROCm profiling tool stack
    - jax profiler, xprof
        - starting trace collection in jax
        - the jax profiler backend: xla profiler backend + XSpace
        - xprof pages: https://jax-ml.github.io/scaling-book/profiling/ explains most
        - explain gpu specific pages eg. kernel stats
    - rocprofv3
        - trace collection
        - roctx annotations
        - hardware counter collection with --pmc
    - tracelens

3. profiling a mixtral training step
    - attribution of a full train step
    - decompose into logical components of the architecture (e.g. forward[att, moe router, dispatch, expert gemm, activations] etc)
    - showing roofline analysis per logical parts


---

4-mixed precision

1. why we want to train in mixed precision
    - based on the premise that we 1) utilize lower-precision FLOPs 2) lower training memory footprint, whilst maintaining higher precision quality
2. well what does it mean to train in mixed precision? we cant simply move the whole model from fp32 to fp8, that would degrade the model
    - we have to apply quantization to selective tensors hence the term mixed precision
    - you can already see the concept in an MX mfma instruction itself: compute in mxfp, accumulate in bf16
3. lets take a prominent example of this: deepseekv3
    - show their mixed precision diagram, including how their logical separation of tensors relates to their quantizations
4. how would we implement this in jax?
    - pure jax 
    - maxtext allows you to control flags: these are predetermined quantization recipes
        - cover these configs
5. a full persepctive of quantization formats
    - bf16, fp16
    - fp8 training (deepseek style, requires delayed scaling, other such things to make viable)
    - MXFP formats, including diagrams of how they work, and external research papers demosntrating their viability
6. CASE STUDY of llama70b
    - expectations:
        - published mi355 format FLOPs
        - amdahl speedup bound: we know from 3-profiling the attribution of a train step. lets try and predict the speedup accordingly
        - expected reductions in memory footprint
    - a discussion of mixed precision recipes. which components should use which precision formats.
    - train step results for llama70b:
        - do the results align with our predictions/attribution estimates
        - do they aligh estimates seen in third party research papers
    - convergence results (local results)
    - convergence results of third party papers.

---

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

3. how does this get implemented in jax?
    - at the jax level, https://jax-ml.github.io/scaling-book/jax-stuff/ explains sharding modes
    - at maxtext level we can specify fsdp and ep parallelism for standard shemes
    - implications on the device mesh: want to be able to explain why fsdp8 ep8 indicates a 64-devices mesh (orthogonal mesh axis)

4. deepseekv3 again as a prominent example of frontier preferences for training meshes:
    - dense models, no ep parallelism (like llama), which means fsdp for making the model fit is a priority
    - moe itself is a way to parallelize the mlp/ff layer in architecture. which means we still parallelize to fit, but along an independent axis (The experts!)
    - extreme ep parallelism eg. deepseekv3 pushes further into high ep (smaller experts, smaller independently parallelisable units, more incentize to parallelize along expert axis) but further emphasis on collectives

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

you have to keep in mind that: a model model architecture is simply a specification for a model
whether you implement that through jax or through maxtext or (pytorch for that matter) is simply an implementation detail. as long as you are describing the same archicture within the framework, you are training the same model.
with a framework like maxtext, its difficult to describe custom workflows and you likely need to know how to implement custom designs at jax level in order to fully describe what you want. maxtext can only cover generic structures