---
layout: distill
title: "Compatibility and Negative Results"
description: "Versioned support status, failed paths, fallbacks, patches, and retest requirements for the MI355X stack."
date: 2026-09-13

section_label: "Appendix E"

previous_section_url: "/pages/d-appendix-tooling"
previous_section_name: "Appendix D: Profiling"

next_section_url: "/pages/f-appendix-artifacts"
next_section_name: "Appendix F: Artifacts"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Status Vocabulary
  - name: Current Compatibility Register
  - name: Negative Result Registry
  - name: Adding or Retesting an Entry
---

## Status Vocabulary

| Status | Meaning |
|---|---|
| Available | Passed the stated test on the pinned stack |
| Experimental | Passed only with an unstable interface, patch, or source branch |
| Unsupported | No correct path exists on the pinned stack |
| Fallback | Runs through a different backend than requested |
| Slow fallback | Correct fallback whose measured throughput is materially lower |
| Deprecated | Supported by an older stack and scheduled for removal |
| No-op | Accepted but does not control behavior on the stated revision |
| Unverified | Present in config, source, or documentation without a retained test artifact |

Each row needs a stack, hardware, verification date, test, HLO/kernel proof, workaround, and
artifact link. Without those fields, its publication evidence is **unverified** even if a
repository note records the observation.

## Current Compatibility Register

These rows preserve the current repository requests as of **13 September 2026**.
They are **[source]**, not runtime proof. Retest against the immutable container
digest before changing a row from `Unverified`.

| Path | Status | Requested stack | Required proof | Last verified | Artifact |
|---|---|---|---|---|---|
| BF16/FP16 MaxText training | Unverified | v26.6, MaxText `b47d74bf` | Unprofiled run, HLO dtype, kernels | Not verified under Appendix F | BLOCKED |
| Delayed-scaling FP8 | Unverified | Same stock path | TE custom calls, amax state, convergence guardrail | Not verified under Appendix F | BLOCKED |
| MXFP8 | Unverified | Patched TE 2.17 branch | Exact commit/wheel hash and no fallback | Not verified under Appendix F | BLOCKED |
| MXFP4 | Unverified | MaxText `b437942a`, JAX-AITER `35b7175c`, AITER `31350226` | FFI libraries, projection coverage, BF16 attention core | Not verified under Appendix F | BLOCKED |
| TE fused attention | Unverified | `NVTE_FRAMEWORK=jax`; Llama 7B source requests CK | Forward/backward AITER kernel names | Not verified under Appendix F | BLOCKED |
| Raw JAX XLA attention | Unverified | JAX 0.11.0 | Correctness plus absence of fused-attention custom call | Not verified under Appendix F | BLOCKED |
| Direct JAX-AITER attention | Unverified | JAX-AITER alpha2 for `gfx950` | FFI libraries and `fmha` kernels | Not verified under Appendix F | BLOCKED |
| Tokamax Pallas-Triton attention | Unverified | Local ROCm hardware-guard adaptation | Pallas/Triton lowering and correctness | Not verified under Appendix F | BLOCKED |
| Ragged GroupedGEMM experts | Unverified | FP16 and experimental ragged flags | hipBLASLt GroupedGEMM route | Not verified under Appendix F | BLOCKED |
| Dense-padded sparse experts | Unverified | Same FP16 workload as ragged arm | GroupedGEMM flag off, matched routing | Not verified under Appendix F | BLOCKED |

## Negative Result Registry

These entries are source-recorded observations. None has a validated Appendix F
bundle, so each remains `Unverified` until retested.

| ID | Stack and scope | Result | Detection | Workaround or next test | Verification | Artifact |
|---|---|---|---|---|---|---|
| NR-001 | JAX 0.11.0 ROCm, raw JAX attention | `implementation="cudnn"` reports that cuDNN is not detected | Exception plus no executable | Use `implementation="xla"`, TE, direct JAX-AITER, or Tokamax; retain a fresh log | Source audit 2026-09-13 | BLOCKED |
| NR-002 | Same raw JAX XLA attention path | Does not reach the fused AITER attention route | HLO lacks the custom call; trace shows XLA fusions and GEMMs | Treat XLA as its own arm | Source audit 2026-09-13 | BLOCKED |
| NR-003 | Tokamax 0.0.12–0.0.14 hardware guard | Expects an NVIDIA-style numeric capability while ROCm reports `gfx950` | Guard failure before lowering | Pin the local adaptation; retest upstream | Source audit 2026-09-13 | BLOCKED |
| NR-004 | v26.6 MI355X Mixtral GroupedGEMM | BF16 is unsupported by the exercised gfx950 hipBLASLt path | Backend/dtype rejection | Compare GroupedGEMM and dense-padded arms in FP16 | Source audit 2026-09-13 | BLOCKED |
| NR-005 | Declared Mixtral ragged AllToAll path | RCCL device barrier is unsupported in the exercised communicator | Failure with barrier path | Disable the barrier path; retain the one-shot flag; retest | Source audit 2026-09-13 | BLOCKED |
| NR-006 | v26.6 Llama 70B MXFP8 without workspace patch | Slow fallback does not represent the intended path | Workspace check plus backend proof | Reconcile the TE branch, pin a commit and wheel hash, then rerun | Source audit 2026-09-13 | BLOCKED |
| NR-007 | `rocprofv3 --pmc` and `rocprof-compute` | Collected durations do not represent normal timing | Dispatch serialization or replay | Use a separate unprofiled process for timing | Source audit 2026-09-13 | BLOCKED |
| NR-008 | Current checked-in `XLA_FLAGS` | Command buffers are disabled with an empty assignment | Effective flag capture | Preserve the syntax; reproduce the original failure before assigning a cause | Source audit 2026-09-13 | BLOCKED |
| NR-009 | Archived v26.5 training image | `libtpu` initialization printed irrelevant TPU errors on ROCm | Startup log | Set `JAX_PLATFORMS=rocm`; retest v26.6 | Archive only | BLOCKED |
| NR-010 | Multi-process container launch | Default shared-memory failure has not been reproduced on the current stack | No current artifact | Test host IPC and a sized private namespace separately | Unverified | BLOCKED |

NR-008 through NR-010 deliberately separate a retained workaround from an unverified cause.
Do not turn an old observation into a current platform claim.

## Adding or Retesting an Entry

Create one record with:

```yaml
id: NR-011
status: unsupported
first_seen: 2026-09-13
last_tested: 2026-09-13
hardware: MI355X/gfx950
stack:
  container: tag@digest
  jax: version
  xla_commit: commit
  rccl: version
workload: case/run id
requested_path: exact config and flags
symptom: exception, fallback, hang, numerical error, or regression
detection: command and artifact path
workaround: exact patch, flag, or fallback
retest_trigger: version or upstream issue
evidence_label: measured
```

A failed run is **[measured]** when its complete inputs and failure artifacts are retained.
Close an entry only with a new artifact bundle; keep the old result and link the superseding
test.
