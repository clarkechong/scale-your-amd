3-profiling

1. prerequisite knowledge
    - roofline analysis

2. JAX/ROCm profiling tool stack
    - jax profiler, xprof
        - starting trace collection in jax
        - the jax profiler backend: xla profiler backend + XSpace
        - xprof pages: https://jax-ml.github.io/scaling-book/profiling/ explains most xprof functionalities
        - explain gpu specific pages eg. kernel stats
    - rocprofv3
        - trace collection
        - roctx annotations
        - hardware counter collection with --pmc
    - tracelens as a WIP

3. profiling a mixtral training step
    - attribution of a full train step
    - decompose into logical components of the architecture (e.g. forward[att, moe router, dispatch, expert gemm, activations] etc)
    - showing roofline analysis per logical parts

---

## Profiler metrics
### Roofline analysis
### End-to-end metrics
- explain train step time, tokens/s, MFU, memory footprint

## The JAX profiler
### XLA profiler backend
- a block digram to explain: tracer, collector, TracerEvent, ... -> how the XSpace is actually populated
- explain the general internals: rocprof-sdk provides the API, which the tracer calls. needs explanation on how profiling is conceptually done: where does XLA have a part to play? is HLO attribution tagged simply as metadata?
- diagram to explain the XSpace proto (img/what-is-xspace image)
### XProf
- refer to https://jax-ml.github.io/scaling-book/profiling/ for general usage
- demonstrate kernel stats page
- demonstrate roofline analysis page for GPU

## rocprofv3
### Trace collection
- refer to rocm documentation
- screenshots of perfetto trace
### PMC collection
- PMC serialises dispatch: what does this actually mean? GPU streams parallelise at hardware level. if serialized, it means essentially one gpu stream, no stream level parallelism
- example of PMC collection eg attention kernel
- hopefully some example where rocm documents their tested MFU and etc, and we can compare and validate
### ROCTx annotations
- roctx is traditionally host-side api which means when host dispatches a jax.jit module, we cannot apply device-side roctx annotations
- show an example of applying roctx annotation for non jit program: see the roctx annotation lane in the xplane (?)
### TraceLens

## rocprof-compute
- used for individual kernels. refer to documentation
- it will replay kernel to obtain all hardware counters


## Worked example: profiling a Mixtral8x22b training step
### Decomposing a train step
- decompose into logical components:
    - forward = traditional diagram
    - backward = block diagram
### Time attribution of in JAX
- mixtral train step in jax, manually insert jax profiler and named scopes
### Time attribution in MaxText
- use default jax profiler maxtext setting
### Roofline analysis of components
- in xprof it looks fine, but in reality can be misleading
- use rocprof for a more finegrained roofline per kernel and manually weigh kernels into components for an accurate roofline

