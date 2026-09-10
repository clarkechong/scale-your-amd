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
