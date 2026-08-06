---
layout: distill
title: "Appendix B: How We Measure"
description: "The protocol behind every measured number in this book: the exact software stack, warmup and repeat counts, clock and power state, and device count and partitioning mode. Plus what the analytical tag means and what it would take to remove it."
date: 2026-08-04

section_label: "Appendix B"

previous_section_url: "/pages/a-appendix-install"
previous_section_name: "Appendix A: Installing"

next_section_url: ""
next_section_name: "End of the book"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: The Software Stack
  - name: Warmup and Repeats
  - name: Clocks and Power
  - name: Device Count and Partitioning Mode
  - name: What Analytical Means Here
---

**Depends on:** nothing. Every **[measured]** number in the book links here, so this page has to
stand alone and stay short enough that following the link is not a punishment.

**"We measured 370 GB/s" is an anecdote. The same number with a stack and a method behind it
is evidence.** This page is the difference, and it is the convention that decides whether
this book's central claim survives contact with a skeptical reader.

**Every [analytical] number links here too**, because the useful thing to say about an
unmeasured claim is what it would take to measure it.

## The Software Stack

**Everything in this book is one `docker pull`.** That is the whole of the stack
specification, and it is deliberately a single string rather than a list of wheel versions,
because a list of wheel versions is five things a reader has to reconcile and an image tag
is one thing they can actually run.

```bash
docker pull rocm/jax-training:maxtext-v26.5
```

> **The stack every [measured] number in this book was taken on**, unless the claim says
> otherwise. All of it comes with the image above:
>
> | Component | Version |
> |---|---|
> | Container | `rocm/jax-training:maxtext-v26.5`, build `5ffe026e`, 17 July 2026 |
> | ROCm | 7.14.0 |
> | `jax` | 0.10.0 |
> | `jaxlib` | 0.10.0 |
> | `jax-rocm7-pjrt` | 0.10.0+rocm7.14.0 |
> | `jax-rocm7-plugin` | 0.10.0+rocm7.14.0 |
> | RCCL | 2.30.4 |
> | XProf | 2.23.0 |
> | MaxText | ROCm fork, `release/v26.5`, commit `a7c6c7e5` |
> | Hardware | 8x MI300X (gfx942), SPX compute partition, NPS1 memory partition |
> | Dates | Captures from August 2026 onward; individual claims carry their own date |

**Note the RCCL line, because it is not the one in the ROCm tarball.** The image rebuilds
RCCL from `ROCm/rocm-systems` rather than shipping the stock library, so a reader who
installs ROCm 7.14.0 from scratch and a reader who pulls this image are not running the same
collectives. Chapter 4's sweep is a measurement of the image.

**The image is also not a clean-room environment, and pretending otherwise would invalidate
half this book.** It presets `XLA_FLAGS`, and two of those flags change numbers we care
about:

```
--xla_gpu_autotune_level=0                      # autotuning OFF
--xla_gpu_enable_latency_hiding_scheduler=True  # overlap ON
--xla_gpu_enable_triton_gemm=False
--xla_gpu_enable_cublaslt=True
--xla_gpu_enable_command_buffer=''              # the Appendix A workaround, pre-applied
--xla_gpu_all_gather_combine_threshold_bytes=8589934592
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
--xla_gpu_memory_limit_slop_factor=95
--xla_gpu_enable_all_gather_combine_by_dim=FALSE
```

**We measure the container as it ships, and we say so.** The alternative was to clear
`XLA_FLAGS` and define our own baseline, which would produce numbers nobody could reproduce
without also copying our flag list. So the as-shipped configuration is the baseline
throughout, and where a flag matters we give both arms explicitly: the matmul in
[Chapter 3]({{ '/pages/3-profiling' | relative_url }}) is quoted at both autotune levels, and
the overlap section in [Chapter 4]({{ '/pages/4-sharding' | relative_url }}) toggles the
latency-hiding scheduler rather than assuming it.

**Tip:** the container also ships `libtpu`, which initialises on import and prints a wall of
TPU warnings before every JAX program runs correctly on ROCm. It is harmless, and
`JAX_PLATFORMS=rocm` silences it. Every command in this book sets it.

**The protocol below is implemented, not just described.** `bench/_harness.py` in the
book's repository enforces the warmup and repeat counts, and `bench/_env.py` writes an
`env.json` next to every result recording the table above as it actually was at run time.
If a number here and a number in the repository disagree, the repository is right.

## Warmup and Repeats

**Three iterations discarded, ten measured, median reported.** That is the protocol
throughout, and each part of it is there for a reason.

**Discard the first iterations because they are measuring compilation, not
computation.** The first call to a `jit`-compiled function compiles it, and on ROCm it may
also run kernel autotuning, which picks a GEMM implementation by trying several. Both are
one-time costs and neither is what you want to know.

**Report the median, not the mean.** Clock ramp and occasional scheduling noise produce a
right tail, and a mean over ten iterations is sensitive to one slow one in a way that
tells you nothing useful about steady-state throughput. If the median and the mean differ
by more than a few percent, that is itself worth reporting.

**Block before stopping the clock.** JAX dispatches asynchronously, so a timing loop
without `block_until_ready()` measures how fast Python can enqueue work. Every measurement
in this book blocks.

```python
for _ in range(3):                       # warmup: compile, autotune, ramp
    f(x).block_until_ready()

with jax.profiler.trace("/tmp/traces/workload"):
    for _ in range(10):                  # measured
        f(x).block_until_ready()
```

**For training-step measurements the equivalent is steps 10 through 20 of a real run**, for
the same reasons plus one more: the input pipeline needs a few steps to reach steady state,
and a step-time measurement taken while it is still filling its prefetch buffer is a
measurement of the data loader.

## Clocks and Power

**We do not lock clocks, and you should expect a few percent of unexplained gap because of
it.** Saying so is the point of this section.

MI300X is a 750 W part and its 2100 MHz figure is a peak boost clock rather than a
sustained one. Under a long bf16 matmul, a real device does not hold boost, so a
measurement compared against a boost-clock roofline will land a few percent low for reasons
that have nothing to do with the code being measured.

**Two consequences for how to read this book.**

**A measurement within a few percent of the roofline is at the roofline**, as far as this
book is concerned. We do not claim differences of that size mean anything.

**A measurement that is 20% off is not explained by clocks.** If someone offers thermal
throttling as an explanation for a 20% gap, ask for the clock trace;
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }})'s triage order will find the
real cause faster.

`rocm-smi` reports current clocks, power draw and any throttling reasons, and it is worth
watching alongside a long measurement rather than only afterwards.

## Device Count and Partitioning Mode

**A number without a device count is unreadable, and on AMD there are two separate reasons
for that.**

**First, the profiler sums op times across devices.** Per
[Chapter 3]({{ '/pages/3-profiling' | relative_url }})'s limitations table, a value read off
an 8-GPU trace is roughly 8x the wall-clock figure. Every device op time in this book is
either from a single-device capture or has been divided, and each claim says which.

**Second, partitioning changes how many compute units a process sees.** MI300X can be
presented as one logical device with all 304 CUs (SPX) or split along XCD boundaries into
several smaller logical devices, with memory partitioned separately by NPS mode.
**Everything in this book is SPX, one process per physical GPU, all 304 CUs.** A partitioned
device invalidates every FLOP figure in
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) by exactly the fraction of CUs it
holds.

Check both before comparing anything:

```bash
rocm-smi --showproductname          # which part, and the gfx target
rocm-smi --showcomputepartition     # SPX / CPX / ...
rocm-smi --showmemorypartition      # NPS1 / NPS2 / ...
```

**Conventions for how counts are quoted.** Where the book says "per device" it means one
physical MI300X in SPX mode. Where it gives a node figure it means eight of them on one
baseboard. Where a claim involves more than eight GPUs it is
**[analytical]**, because we have never had more than one node.

## What Analytical Means Here

**[analytical]** means derived from published specifications and arithmetic, and not
checked against hardware. It is not a hedge and it is not an apology: most of this book is
analytical by design, because the arithmetic is the transferable part. What it *is* is a
promise that we are not implying a measurement we did not take.

**Three categories of analytical claim in this book, with what it would take to remove the
tag from each.**

**Anything inter-node.** Every claim involving more than eight GPUs, which includes all of
[Chapter 6]({{ '/pages/6-training' | relative_url }})'s multi-node placement arithmetic,
[Chapter 7]({{ '/pages/7-moe' | relative_url }})'s expert-axis-crossing-hosts numbers, and
[Chapter 12]({{ '/pages/12-serving' | relative_url }})'s KV-transfer cost. **What it would
take:** a multi-node allocation with a documented fabric topology and a per-node egress
figure. This is the largest single gap between what the book promises and what it
demonstrates, and it is why the measurement promise is scoped to single-node claims.

**Everything inference-side.** All of
[Chapter 11]({{ '/pages/11-inference' | relative_url }}) and most of
[Chapter 12]({{ '/pages/12-serving' | relative_url }}). **What it would take:** a serving stack
and a load generator, which is a different project. The arithmetic there is not in doubt;
what is untested is whether a real engine achieves it, and
[Chapter 11]({{ '/pages/11-inference' | relative_url }})'s closing section is honest about the
three reasons it will not.

**Numbers borrowed from someone else's measurements.** AMD's realised xGMI and RCCL
bandwidth figures, which
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}) calibrates its cost model against, and
AMD's occupancy sweep, which
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) quotes. **These are attributed
to AMD at the point of use and are never presented as ours.** That distinction matters more
than the analytical tag does: a borrowed measurement is stronger evidence than our
arithmetic and weaker evidence than our own capture, and the reader is entitled to know
which they are looking at. **What it would take:** running the equivalent sweep ourselves,
which for the RCCL numbers needs one node and is the highest-value outstanding measurement
in the book.

## References

- [rocm-smi documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
  (AMD). The partitioning, clock and power queries used above.
- [AMD Instinct MI300X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf).
  The 750 W board power and 2100 MHz peak boost clock that the clocks section is about.
- [JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html) (JAX).
  `jax.profiler.trace` and `StepTraceAnnotation`, and the asynchronous-dispatch behaviour
  that makes `block_until_ready()` mandatory.
- [Appendix A]({{ '/pages/a-appendix-install' | relative_url }}) for how to obtain the stack
  in the table above.
