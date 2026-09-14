---
layout: distill
title: "Measurement and Convergence Protocol"
description: "The evidence labels, timing method, quality checks, and comparison rules used throughout the book."
date: 2026-09-13

section_label: "Appendix B"

previous_section_url: "/pages/a-appendix-install"
previous_section_name: "Appendix A: Environment"

next_section_url: "/pages/c-appendix-config"
next_section_name: "Appendix C: Configuration"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Evidence Labels
  - name: Required Run Identity
  - name: Train-Step Timing
  - name: Profile Runs
  - name: Metrics
  - name: Convergence Guardrail
  - name: Comparison Rules
---

## Evidence Labels

- **[source]**: established by checked-in code, configuration, or a manifest. This
  proves what was requested or implemented, not what executed on a device.
- **[analytical]**: derived from stated shapes, algorithms, or published specifications; no
  hardware run is implied.
- **[measured]**: produced by this project with a complete
  [Appendix F]({{ '/pages/f-appendix-artifacts' | relative_url }}) bundle.
- **[cited]**: produced by another named source. The citation must identify the stack,
  workload, and method well enough to judge comparability.

Use **unverified** for a planned or observed behavior that lacks the required evidence.
Software support claims also carry a version and verification date.

## Required Run Identity

Every result states:

```text
run_id and UTC start time
MI355X count, node count, SPX/compute and NPS/memory partition
container tag and digest
ROCm, JAX, jaxlib, PJRT plugin, Python, RCCL, XProf
MaxText, Transformer Engine, JAX-AITER, AITER commits and dirty patches
effective XLA_FLAGS and relevant environment variables
model config, precision roles, mesh, sequence length and batch vocabulary
dataset/tokenizer manifest or synthetic-data declaration
```

The runtime inventory is authoritative. A README pin does not prove which wheel or source
tree a process imported.

## Train-Step Timing

The current case-study convention is:

1. run 30 train steps;
2. use the first 10 for compilation, autotuning, cache population, and clock ramp;
3. synchronize device work before stopping each timer;
4. report the median of steps 10 through 29; and
5. retain all 20 samples plus minimum, maximum, median, mean, and median absolute
   deviation (MAD).

JAX dispatch is asynchronous. The timed value must be blocked:

```python
start = time.perf_counter()
state, metrics = train_step(state, batch)
jax.block_until_ready((state, metrics))
seconds = time.perf_counter() - start
```

The workload must state whether data is a reused synthetic batch or a real pipeline.
Synthetic timing excludes input cost and must not be presented as end-to-end training
throughput.

Use separate run IDs when changing global batch, sequence length, gradient accumulation,
model dimensions, precision, rematerialization, attention, expert execution, mesh, flags,
or source revisions. Those changes do not belong in one paired timing series.

Do not lock clocks unless the entire series uses the same documented policy. Capture clocks,
power, throttling state, and partition mode during the run.

## Profile Runs

Timing, XProf, kernel tracing, and performance-counter collection are separate processes
using the same config:

| Run type | Use |
|---|---|
| Unprofiled | Published step time and tokens/s |
| XProf | Framework/HLO attribution, memory, overlap |
| `rocprofv3` trace | Kernel, runtime, memory, marker, and RCCL timelines |
| `rocprofv3 --pmc` | Selected hardware counters |
| `rocprof-compute` | Replay-based counter and roofline analysis |

Counter collection may serialize dispatches and replay the workload. Never use a PMC or
`rocprof-compute` duration as the application timing.

For XProf, skip the first 10 steps and capture five unless the case declares another bounded
window. Call `block_until_ready()` inside the trace.

## Metrics

Report definitions with values:

$$
\text{tokens/s/GPU}
=
\frac{\text{global non-padding training tokens per step}}
{\text{median step seconds}\times\text{GPU count}}.
$$

Required train-step fields are:

- median step time and all retained samples;
- aggregate tokens/s and tokens/s/GPU;
- global sequences and non-padding tokens per step;
- peak device memory per GPU;
- compile time and recompilation count;
- exposed and total collective time from a trace;
- loss and gradient norm;
- kernel-route evidence for the path under test; and
- MFU only with its FLOP convention and hardware ceiling stated.

For MoE, add tokens per expert, drop/padding fraction, expert GEMM shape distribution, and
dispatch/combine exposure.

## Convergence Guardrail

Throughput is the primary objective. Convergence checks whether a precision or kernel change
preserves learning behavior over the tested horizon.

The Llama 70B convergence protocol is:

| Field | Value |
|---|---|
| Reference | BF16 |
| Arms | BF16, FP16, delayed-scaling FP8, MXFP8, MXFP4 |
| Data | Local pinned `allenai/c4` shards with a file manifest |
| Tokenizer | Verified Llama 2 tokenizer with file hashes |
| Budget | 2,034 steps, 999,751,680 nominal token positions |
| Schedule | 5% warmup, then cosine decay |
| Evaluation | 20 batches every 100 training steps |
| Data controls | Packing on, shuffle seed `20260823`, offline reads |
| Primary quality field | Validation loss versus BF16 |

Publish the dataset manifest, tokenizer provenance, config, per-step train metrics, every
evaluation result, and the comparison script. State seed count. One seed supports a
tested-horizon guardrail, not a general equivalence claim.

No retained pre-run validation-loss threshold has been identified for the completed
campaign. Its result is therefore descriptive: the curves were near-identical over
the tested horizon. It is not a passed equivalence test. A repeated campaign must
declare the acceptable validation-loss difference and numerical-failure criteria
before reading the final curves. Time-to-quality is optional and is reported only
when the target loss is reached by all compared arms.

## Comparison Rules

A comparison is accepted only when:

- one named variable changes;
- the same run identity fields are equal or the difference is explained;
- timing uses unprofiled synchronized samples;
- the batch and token denominator are identical;
- the expected backend is proved rather than inferred from a config field;
- peak memory comes from the same measurement method;
- numerical or convergence checks appropriate to the change pass; and
- failed and fallback paths remain in the record.

If any condition fails, report separate observations instead of a speedup ratio.
