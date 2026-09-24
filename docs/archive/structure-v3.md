# MI355X JAX/ROCm book v3 repository roadmap

Status: chapter skeleton migrated; initial drafts under evidence and editorial review  
Scope of this document: editorial and repository plan only  
Reference platform: AMD Instinct MI355X (`gfx950`)  
Last updated: 2026-09-13

This document is the source of truth for the v3 book structure. It records what the
book owns, what it reuses, where every planned chapter will live, which evidence each
claim needs, and the order in which the repository should change.

The active pages now follow this roadmap. Initial drafts exist for every chapter and
appendix. A heading or drafted result does not authorize a measured claim; measured
prose must still pass the evidence rules below.

## 1. Editorial contract

### 1.1 Positioning

- The book is a self-contained companion to the
  [JAX Scaling Book](/home/clchong/work/jax-ml-scaling-book-outline.md), not a
  replacement for it.
- A reader should be able to follow this book without first reading the Scaling Book.
- Generic derivations are recapped only far enough to support the MI355X decision at
  hand. The full derivation remains the Scaling Book's responsibility.
- The book owns the link from model arithmetic to MI355X hardware, JAX/XLA lowering,
  ROCm kernels, MaxText configuration, and measured behavior.

### 1.2 Platform and workload boundary

- The only target accelerator is MI355X/CDNA4 (`gfx950`).
- Other AMD accelerators and NVIDIA or TPU systems appear only in short contrasts
  that explain an MI355X result.
- The subject is model training.
- Serving engines, KV-cache management, continuous batching, speculative decoding,
  and deployment are out of scope. Link to the Scaling Book, vLLM, or SGLang instead.
- Single-node work uses the eight-GPU MI355X scale-up domain.
- Multi-node operation is future work until hardware and captured evidence exist.

### 1.3 Reader and prerequisites

The primary reader can read Python and basic JAX, knows the main Transformer blocks,
and wants to configure or diagnose training on MI355X. The book must introduce GPU
execution, ROCm-specific paths, sharding costs, and profiling without assuming prior
AMD experience.

Each chapter should support one or more of these paths:

- new MI355X user;
- existing JAX user moving to ROCm;
- performance investigation;
- precision selection;
- dense-model training;
- MoE training.

### 1.4 Optimization objective and validation hierarchy

The primary performance metric is **tokens/s/GPU** for a fixed, fully specified
training workload.

Every comparison must freeze or report:

- model architecture and parameterization;
- sequence length and packing;
- global batch and per-device token count;
- optimizer and schedule;
- gradient accumulation;
- mesh and device count;
- precision by tensor role;
- rematerialization policy;
- attention and expert-kernel path;
- software versions, commits, patches, and effective flags;
- data source or synthetic-data construction;
- warmup, synchronization, repetitions, and statistic.

Supporting metrics explain the tokens/s/GPU result:

- step time;
- MFU or HFU;
- peak HBM use;
- exposed collective time;
- compile cost;
- kernel attribution.

MFU is diagnostic, not the objective. A higher MFU does not win if
tokens/s/GPU is lower for the same workload.

Convergence is the correctness guardrail when a configuration changes numerics.
Time-to-quality is optional and should appear only when there is a meaningful,
predeclared quality threshold.

### 1.5 Completed Llama 70B convergence finding

The Llama 70B low-precision convergence experiment is complete through approximately
one billion training tokens. The tested loss curves were nearly identical, and no
quality separation was observed over that tested horizon.

This is a completed result, not a planned experiment. It must retain the tested-horizon
qualification. It does not prove that all precisions converge identically at longer
horizons or on other workloads. Publication-ready plots, exact provenance, and the
final artifact index are still blockers for publishing the claim.

### 1.6 Evidence vocabulary

Every substantive technical claim uses one of these labels:

- `[source]`: established by checked-in code, configuration, or a manifest. It
  proves the requested or implemented path, not device execution.
- `[analytical]`: calculated from model shapes, algorithms, or published hardware
  specifications. Show enough inputs and units to reproduce the calculation.
- `[measured]`: produced on the stated MI355X system under the frozen protocol. Link
  the claim to its artifact manifest.
- `[cited]`: taken from an identified external source. Cite the source next to the
  claim and distinguish its system from this book's system.

Rules:

- Do not convert planned measurements, console observations, or remembered numbers
  into `[measured]` prose.
- A derived number based on a cited specification remains `[analytical]`; cite the
  input specification.
- A borrowed benchmark remains `[cited]`, even when its hardware matches.
- Negative results need the same environment and artifact record as positive results.
- A section with one evidence type may declare it once at the start. Mixed sections
  label claims inline.

### 1.7 Software-status vocabulary

Software behavior changes quickly. Every support statement must include a date and
the relevant version or commit. Use these states:

- `available`: works through the documented path on the pinned stack;
- `experimental`: works but depends on an unstable interface or patch;
- `unsupported`: no working path exists on the pinned stack;
- `fallback`: the requested route lowers to a different implementation;
- `slow fallback`: the route is correct but materially slower than the recommended
  path;
- `deprecated` or `no-op`: accepted by configuration but no longer controls behavior;
- `unverified`: plausible or documented elsewhere, but not proved on the pinned stack.

The text must distinguish API availability from an optimized training path.

### 1.8 Recommendation standard

A recommendation is publishable only when the chapter gives:

1. the fixed workload and decision being made;
2. the analytical expectation;
3. the exact JAX, MaxText, XLA, and ROCm controls;
4. proof of the path that ran;
5. tokens/s/GPU and supporting measurements;
6. a correctness or convergence check where numerics change;
7. known failures and a fallback;
8. the software versions for which the recommendation holds.

## 2. Content ownership relative to the JAX Scaling Book

### 2.1 Material owned by the Scaling Book

Link to the Scaling Book for the full, hardware-independent treatment of:

- roofline derivations;
- arithmetic intensity;
- Transformer parameter and FLOP accounting;
- sharded matrix multiplication;
- collective cost models;
- data, tensor, sequence, context, and pipeline parallelism;
- general JAX sharding notation;
- broad accelerator comparisons.

This book may restate a compact formula or definition when the next paragraph
substitutes MI355X constants or maps it to a configuration decision. Preserve
attribution for adapted MIT-licensed notation, diagrams, and derivations.

### 2.2 Material owned by this book

This book is authoritative for:

- MI355X execution units, memory hierarchy, partition modes, and scale-up topology;
- MI355X constants used in compute, memory, and communication estimates;
- the JAX-to-XLA-to-ROCm execution path on the pinned software stack;
- how to prove that a requested kernel or collective path ran;
- MaxText controls and patches used by the experiments;
- MI355X-specific precision support and fallback behavior;
- measured single-node tokens/s/GPU, MFU, HBM, and profile results;
- rematerialization behavior on the tested models;
- RCCL behavior in the eight-GPU scale-up domain;
- JAX-reachable dense, attention, and MoE kernels;
- negative results, compatibility records, and retest status;
- the Llama 7B, Llama 70B, and Mixtral 8x22B case studies;

### 2.3 Boundary by chapter

- Chapters 1 and 2 own platform facts and the concrete execution path. The Scaling
  Book supplies only broad architecture or compiler context.
- Chapter 3 borrows generic cost-model forms and owns the MI355X constants,
  workload definition, worked prediction, measurement contract, and ROCm/JAX
  profiling workflow.
- Chapters 4 through 9 own configuration decisions on the pinned MI355X stack.
  They may link to generic parallelism or numerical background.
- Chapters 10 through 12 own the experiment design, artifacts, results, and
  recommendations for their named workloads.
- Appendices own volatile setup, protocol, configuration, tooling, compatibility,
  and artifact lookup material.

### 2.4 Duplication test

Keep a recap only if removing it would prevent a reader from understanding the next
MI355X mechanism, control, prediction, or result. Otherwise link out.

Before retaining a section, verify that it contributes at least one of:

- an MI355X mechanism;
- a JAX/ROCm/MaxText control;
- measured or cited evidence;
- a direct prerequisite for one of those items.

## 3. Full reading-order hierarchy

### Chapter 0 — What this companion answers

Planned file: `index.md`

- Reader and prerequisites
- Relationship to the JAX Scaling Book
- MI355X-only scope
- Training-only boundary
- Fixed-workload optimization
  - tokens/s/GPU as the primary metric
  - step time, MFU, HBM, communication, and compile cost as diagnostics
  - why MFU alone does not rank configurations
- Numerical validation
  - convergence as a correctness guardrail
  - optional time-to-quality
  - completed one-billion-token Llama 70B finding
- Evidence vocabulary
  - `[source]`
  - `[analytical]`
  - `[measured]`
  - `[cited]`
- Software-status vocabulary
- Reproducibility promise
- Reading paths
  - new MI355X user
  - existing JAX user
  - performance investigation
  - precision selection
  - dense training
  - MoE training
- Case-study progression
- Versioning and retest policy
- Attribution and MIT reuse

## Part I — The JAX Stack on ROCm using MI355X

### Chapter 1 — MI355X as a training machine

Planned file: `pages/1-mi355x-as-a-training-machine.md`

- Device identity
  - CDNA4
  - `gfx950`
  - OAM and UBB 2.0
  - physical device versus logical JAX device
- Chiplet hierarchy
  - I/O die
  - accelerator complex dies
  - compute units
  - SIMD units
  - Wave64
- Matrix execution
  - Matrix Cores
  - MFMA instructions
  - lane fragments
  - accumulators
  - 16x16 tiles
  - 32x32 tiles
  - macrotiles
  - shape and alignment tails
- Short TPU contrast
  - MFMA versus MXU
  - distributed compute units versus a systolic array
- Non-matrix execution
  - vector pipelines
  - scalar pipelines
  - reductions
  - the non-GEMM Amdahl limit
- Memory hierarchy
  - VGPR and AccVGPR
  - LDS
  - direct-to-LDS
  - L1
  - per-XCD L2
  - Infinity Cache
  - HBM3E
  - cache scope
  - chiplet locality
- Resource pressure
  - register pressure
  - LDS pressure
  - occupancy
  - instruction-level parallelism
  - scratch spills
  - bank conflicts
- Native numerical formats
  - FP32 accumulation
  - BF16 and FP16
  - OCP FP8
  - MXFP8
  - MXFP6
  - MXFP4
  - E8M0 block scales
  - dense versus sparse peak specifications
- Partition modes
  - SPX, DPX, QPX, and CPX
  - NPS modes
  - effects on JAX device visibility
- Capacity and bandwidth constants
  - 288 GB HBM
  - 8 TB/s HBM bandwidth
  - 2.5166 PFLOP/s BF16 peak
  - approximately 315 FLOPs/byte machine balance
  - source and qualification for each constant
- Eight-GPU scale-up topology
  - seven peers per GPU
  - one-hop full mesh
  - directional xGMI bandwidth
  - aggregate injection bandwidth
  - partial collective participation
  - the eight-GPU scale-up ceiling
  - no transparent pooled JAX memory
- Scale-out preview
  - PCIe path
  - GPU-direct RDMA
  - reference NIC topology
  - rail placement
  - explicit future-evidence label
- Hardware constants sheet
- Forward links to precision, sharding, kernels, and case studies

### Chapter 2 — Lowering `jax.jit` on ROCm

Planned file: `pages/2-lowering-jax-jit-on-rocm.md`

- JAX transformations used by the book
  - PyTrees
  - `jax.jit`
  - automatic differentiation
  - `jax.lax.scan`
  - asynchronous dispatch
- Intermediate-representation ladder
  - jaxpr
  - StableHLO
  - HLO
  - LLVM IR
  - AMDGCN ISA
  - `hsaco`
- Partitioning
  - Shardy
  - sharding propagation
  - collective insertion
  - `shard_map`
- Compiler stages
  - graph rewrites
  - layout assignment
  - fusion
  - scheduling
  - buffer assignment
  - code generation
- Kernel routes
  - hipBLASLt
  - rocBLAS
  - RCCL
  - Transformer Engine
  - Composable Kernel and AITER
  - Triton
  - XLA emitters
  - XLA FFI
- Runtime
  - thunks
  - streams
  - PJRT
  - HIP and HSA
  - command buffers
- Autotuning and caches
- Backend initialization order
- Vendor-named compatibility fields
- Path-verification ladder
  - optimized HLO
  - custom-call target
  - kernel name
  - ISA or hardware counters
- Failure taxonomy
  - unsupported
  - compilation failure
  - silent fallback
  - slow fallback
  - numerical failure
- Reference boundary
  - source-file tours move to Appendix D
  - exhaustive compiler-pass catalogues remain external

### Chapter 3 — Profiling and analysis of a training step

Planned file: `pages/3-profiling-and-analysis-of-one-training-step.md`

- Investigation question
  - one synchronized optimizer update
  - compute work
  - HBM traffic
  - communication
  - runtime overhead
- Predicting one training step
  - compute, HBM, and communication bounds
  - MI355X BF16 ridge point
  - model dimensions and FLOPs
  - memory capacity versus traffic
  - batch and optimizer-update semantics
  - sparse-model adjustments
  - expected trace signatures
- Measuring one training step
  - tokens/s/GPU
  - MFU and HFU
  - peak HBM
  - exposed communication
  - compile cost
  - validation loss
  - one Llama 7B worked prediction
  - reusable prediction worksheet
  - workload contract
  - compile, warmup, and timing separation
  - timing versus instrumented runs
  - evidence preservation
- Profiling stack
  - XProf finds the framework or HLO component
  - `rocprofv3` and ROCTx identify runtime dispatches
  - `rocprof-compute` explains kernel efficiency
- XProf
  - XSpace, XPlane, XLane, XEvent, and XStat
  - Trace Viewer
  - operation-to-HLO attribution
  - Kernel Stats
  - Memory Viewer
  - known ROCm limitations
- `rocprofv3`
  - kernel, runtime, copy, and RCCL traces
  - ROCTx annotations
  - lossless `rocpd`
  - targeted PMC collection
  - perturbation warning
- `rocprof-compute`
  - counters
  - speed-of-light analysis
  - cache and HBM behavior
  - occupancy and resource pressure
  - empirical roofline
- One-operation correlation
  - named JAX scope
  - optimized HLO
  - XProf event
  - ROCm dispatch
  - kernel counters
- Interpreting the profile
  - multi-level rooflines
  - triage order
  - host or input starvation
  - compilation or recompilation
  - wrong sharding
  - wrong kernel or fallback
  - HBM limit
  - communication limit
  - low MFMA efficiency
  - numerical regression
- Reporting template used by later chapters
- Reference boundary
  - generic derivations remain in the Scaling Book
  - full command cookbook remains in Appendix D
  - full measurement protocol remains in Appendix B
  - artifact schema remains in Appendix F

## Part II — JAX Performance Features on ROCm

Decision order: choose precision, make one update fit, distribute it for throughput,
select the local kernel routes, integrate those decisions for sparse training, then
tune compiler and runtime controls only when profiling identifies a mechanism.

### Chapter 4 — Training in mixed precision

Planned file: `pages/4-training-in-mixed-precision.md`

- Per-tensor precision rather than one global dtype
- Tensor roles
  - compute
  - activations
  - weights
  - gradients
  - master weights
  - optimizer moments
  - accumulation
  - sensitive exceptions
- BF16 baseline
- FP16 and loss scaling
- FP8
  - E4M3 and E5M2
  - delayed scaling
  - amax history
- MX formats
  - block scaling
  - scale layout
  - MXFP8
  - MXFP6
  - MXFP4
- MI355X hardware path
- Transformer Engine path
- JAX-AITER path
- MaxText configuration fields
- Partial quantization scope
- Communication dtype
- Memory effect
- Peak-compute effect
- Non-GEMM Amdahl ceiling
- Fallback detection
- Throughput protocol
  - fixed-workload requirement
  - tokens/s/GPU ranking
  - profile evidence
- One-billion-token convergence validation
  - fixed C4 and tokenizer provenance
  - tested precision arms
  - nearly identical loss curves
  - no observed quality separation over the tested horizon
  - limits of the finding
- Throughput-first decision procedure
- Convergence guardrail
- Versioned support table
- Inputs to the Llama 70B case study

### Chapter 5 — Making the model fit

Planned file: `pages/5-making-the-model-fit.md`

- Capacity before speed
- Persistent state
- Activation memory
- Attention-score materialization
- Sparse-model capacity signpost
- XLA memory analysis
- Runtime high-water mark
- Donation and aliasing
- Layer scanning
- Rematerialization
  - `jax.checkpoint` and `jax.remat`
  - `none`
  - selective and named policies
  - `full`
  - scan and common-subexpression-elimination interaction
  - attention interaction
- sparse-layer interaction
- Gradient accumulation
- FSDP state sharding
- Sharded initialization
- Host-offload status
- Checkpoint-memory boundary
- Expected-versus-observed reconciliation
- Memory decision procedure
- Inputs to the Llama 7B and Llama 70B case studies

### Chapter 6 — Parallelism strategies for higher throughput

Planned file: `pages/6-jax-shardings-to-a-training-mesh.md`

- Concise sharding recap
  - global versus local arrays
  - `Mesh`
  - `NamedSharding`
  - `PartitionSpec`
  - logical axis rules
- Four collectives
  - AllGather
  - ReduceScatter
  - AllReduce
  - AllToAll
- Four sharded-matmul cases
- Partitioning choices
  - automatic Shardy propagation
  - `with_sharding_constraint`
  - `shard_map`
- HLO verification
  - replica groups
  - collective shapes
  - axis placement
- MI355X RCCL behavior
  - message-size curve
  - participant count
  - directional bandwidth
- Training strategies
  - data parallelism
  - FSDP
  - tensor parallelism
  - sequence parallelism
  - context parallelism
  - pipeline parallelism
  - expert-parallelism signpost
- Required treatment for each strategy
  - state being sharded
  - persistent-memory effect
  - activation effect
  - critical-path collective
  - MI355X threshold
  - MaxText axis fields
- Eight-GPU placement
  - TP and EP competition
  - reserved scale-out axes
- Collective payload and overlap opportunities
- Concrete mesh decision procedure
- Inputs to all distributed case studies

### Chapter 7 — A map of ROCm kernel backends on JAX

Planned file: `pages/7-a-map-of-kernel-backends-on-jax.md`

- Kernel-selection model
  - XLA-generated kernel
  - library custom call
  - Triton
  - FFI
  - Pallas status
- Dense GEMM
  - hipBLASLt
  - rocBLAS
  - Triton GEMM
  - shape and layout sensitivity
  - workspace
  - autotuning
- Attention
  - XLA DPA
  - Transformer Engine
  - CK/AITER FMHA
  - direct JAX-AITER
  - Tokamax and Pallas-Triton
  - forward and backward availability
  - masks and packing
  - workspace and precision
- Fused pointwise kernels
  - RMSNorm
  - SwiGLU
  - cross entropy
- Correctness validation
- Kernel-proof workflow
- Fallback ranking
- Versioned reachability table
- Inputs to the Llama 7B and Llama 70B case studies

### Chapter 8 — Training Mixture-of-Experts on MI355X

Planned file: `pages/8-mixture-of-experts-on-mi355x.md`

- Dense versus sparse accounting
- Router
  - gate projection
  - top-k selection
  - router precision
  - load-balance loss or bias
- Load imbalance
- Capacity policies
  - padding
  - dropping
  - dropless execution
- Expert implementations
  - dense masked
  - fixed-capacity one-hot
  - dense padded
  - ragged or GroupedGEMM
  - win condition for each path
- Token movement
  - sort and permute
  - dispatch AllToAll
  - combine AllToAll
  - ragged collectives
- Expert parallelism
  - memory rationale
  - Mixtral EP-8 state
  - FSDP-by-EP trade
  - TP-by-EP competition
  - one-node placement
- Rematerialization and routing interaction
- Custom VJPs
- Four required diagnostics
  - tokens per expert
  - drop or padding rate
  - expert GEMM efficiency
  - exposed AllToAll time
- Kernel-availability cross-reference
- MoE decision procedure
- Inputs to the Mixtral case study

### Chapter 9 — Tuning the compiler, runtime, and RCCL

Planned file: `pages/9-compiler-runtime-and-rccl-controls.md`

- Flag initialization order
- Controlled flag-sweep method
- Autotuning levels
- Persistent caches
- Latency-hiding scheduler
- Async stream priority
- Memory-limit slop
- Collective combining
- Collective pipelining
- Collective reordering
- Command buffers and HIP graphs
- Triton GEMM selection
- RCCL algorithm and transport controls
- MI355X-specific numerical hazards
- Deprecated and no-op flags
- Before-and-after evidence requirement
- Versioned flag table
- Scope boundary
  - include only controls exercised by a case study
  - include controls required to reproduce or explain a case
  - move unused flag catalogues to external documentation

## Part III — Case Studies: Expectations and Results

### Chapter 10 — Llama 7B: exposing the complete stack

Planned file: `pages/10-llama-7b-exposing-the-complete-stack.md`

- Status and artifact-completeness banner
- Chapter contract and frozen invariants
- Environment and setup
- Model anatomy
- Parameter, FLOP, memory, and throughput prediction
- Raw-JAX training implementation
  - Flax model
  - loss and optimizer
  - whole-step JIT
  - donation
  - scan
  - asynchronous timing
- Raw JAX versus MaxText
  - FP32
  - BF16
  - HLO and layout differences
- First complete profiling walkthrough
  - XProf
  - optimized HLO
  - `rocprofv3`
  - PMC perturbation caveat
- Attention sweep
  - XLA
  - Transformer Engine
  - JAX-AITER
  - Tokamax or Triton
  - kernel-path proof
- One-GPU rematerialization sweep
  - `none`
  - `minimal_with_context`
  - `full`
- FSDP-8 rematerialization sweep
- Tokens/s/GPU as the primary comparison
- Expected-versus-measured reconciliation
- Negative results and fallbacks
- Exact final recommendation
- Artifact index
- Non-goals
  - real-data convergence
  - checkpointing
  - multi-node execution

### Chapter 11 — Llama 2 70B: mixed precision training

Planned file: `pages/11-llama-2-70b-mixed-precision-training.md`

- Status and artifact-completeness banner
- Chapter contract and frozen invariants
- Environment and branch manifest
- Model and FSDP memory prediction
- Shared MaxText recipe
- Precision matrix
  - FP32
  - BF16
  - FP16
  - FP8 delayed scaling
  - MXFP8
  - MXFP4
- Backend path for every arm
- FP32 microbatch or accumulation exception
- MXFP8 workspace patch
- MXFP4 MaxText and JAX-AITER branch
- Train-step protocol
- Primary and supporting results
  - tokens/s/GPU
  - step time
  - MFU
  - peak HBM
- Non-GEMM and Amdahl analysis
- Profile comparison
- Silent and slow fallback analysis
- Convergence setup
  - pinned C4 data
  - tokenizer provenance
  - packing
  - learning-rate schedule
  - evaluation cadence
- Completed one-billion-token result
  - experiment completed
  - approximately one billion training tokens
  - nearly identical tested loss curves
  - no observed quality separation
  - tested-horizon qualification
- Convergence as a guardrail
- Optional time-to-quality
- Comparison with identified external evidence
- Final precision decision record
- Artifact index
- Remaining publication artifacts
  - FP32 timing
  - post-patch MXFP8 timing
  - peak memory for every arm
  - profile artifacts
  - convergence plot and exact provenance

### Chapter 12 — Mixtral 8x22B: sharding meshes and MoE optimizations

Planned file: `pages/12-mixtral-8x22b-sharding-meshes-and-moe-optimizations.md`

- Status and artifact-completeness banner
- Chapter contract and frozen invariants
- Model and routing ledger
- Baseline definition
  - BF16
  - FSDP-1
  - EP-8
  - fixed-capacity one-hot experts
  - latency hiding enabled
- Baseline prediction
  - total and activated parameters
  - per-device memory
  - expert FLOPs
  - dispatch bytes
- FSDP-by-EP sweep
  - 8x1
  - 4x2
  - 2x4
  - 1x8
- Mesh and HLO verification
- MoE execution sweep
  - dense masked BF16
  - ragged GroupedGEMM FP16
  - dense-padded FP16
  - controlled-comparison qualifications
- Ragged AllToAll flags
- GroupedGEMM dtype restriction
- Latency-hiding comparison
- Primary and supporting results
  - tokens/s/GPU
  - step time
  - MFU
  - peak HBM
- Router and imbalance diagnostics
- Exposed collective time
- Kernel attribution
- Expected-versus-measured reconciliation
- Negative result and fallback ladder
- Final mesh and kernel recommendation
- Artifact index
- Explicit blocker
  - no v26.6 result artifacts are available yet

## Appendices

### Appendix A — Reproducible MI355X environment

Planned file: `pages/a-reproducible-mi355x-environment.md`

- hardware and partition-mode checks;
- container identity;
- ROCm, JAX, XLA, and PJRT versions;
- MaxText commit;
- Transformer Engine commit and build;
- AITER and JAX-AITER commit and build;
- topology and RCCL checks;
- initialization-order requirements;
- minimal smoke test.

### Appendix B — Measurement and convergence protocol

Planned file: `pages/b-measurement-and-convergence-protocol.md`

- frozen workload fields;
- warmup and compilation separation;
- device synchronization;
- cache policy;
- repetitions and summary statistics;
- tokens/s/GPU calculation;
- MFU calculation;
- memory collection;
- profile collection;
- evidence labels;
- synthetic-data protocol;
- real-data provenance;
- convergence controls;
- quality-equivalence language;
- exception-record process.

### Appendix C — Configuration quick reference

Planned file: `pages/c-configuration-quick-reference.md`

- MaxText fields;
- JAX configuration;
- XLA flags;
- environment variables;
- RCCL controls;
- software status;
- default and tested values;
- verification method;
- owning chapter and case study.

### Appendix D — Profiler and HLO cookbook

Planned file: `pages/d-profiler-and-hlo-cookbook.md`

- exact tool commands;
- XProf views;
- optimized-HLO extraction;
- custom-call patterns;
- kernel-name patterns;
- collective patterns;
- `rocprofv3` traces;
- `rocprof-compute` counter sets;
- common failure signatures;
- known ROCm tooling limitations.

### Appendix E — Compatibility and negative-results register

Planned file: `pages/e-compatibility-and-negative-results.md`

- feature and requested path;
- pinned environment;
- observed state;
- minimal reproducer;
- fallback;
- patch or upstream issue;
- first-tested and last-retested dates;
- chapter affected.

### Appendix F — Case-study artifacts

Planned file: `pages/f-case-study-artifact-schema.md`

- artifact schema;
- manifest validation rules;
- Llama 7B artifact index;
- Llama 70B artifact index;
- Mixtral 8x22B artifact index;
- full configurations and commands;
- logs and metrics;
- HLO and traces;
- result tables and plot sources;
- data and tokenizer provenance;
- repository commit references;
- checksums or immutable object references for large files.

## 4. Recurring chapter templates

### 4.1 Configuration or mechanism chapter

Use this order for Chapters 1 through 9:

1. Decision the reader needs to make.
2. Prerequisite recap and Scaling Book link.
3. MI355X mechanism.
4. JAX, XLA, ROCm, and MaxText control surface.
5. Analytical prediction.
6. Controlled experiment or explicit evidence status.
7. Result, with `[measured]` or `[cited]` labels.
8. Correctness or convergence guardrail when numerics change.
9. HLO and profile explanation.
10. Negative result and fallback.
11. Exact configuration recommendation.
12. Versioned decision record.
13. Artifact links and open blockers.

Omit a step only when it does not apply. State why briefly rather than filling it
with generic prose.

### 4.2 Measured case-study chapter

Use this order for Chapters 10 through 12:

1. Status banner: measured, partially measured, or blocked.
2. Decision and chapter contract.
3. Frozen invariants.
4. Environment and source-commit manifest.
5. Model, FLOP, memory, and communication ledger.
6. Analytical prediction and expected bottleneck.
7. Baseline configuration.
8. Experimental arms and the one variable each changes.
9. Path proof from HLO, custom calls, kernel names, or counters.
10. Measurement protocol and deviations.
11. Tokens/s/GPU result.
12. Supporting step-time, MFU, HBM, compile, and collective results.
13. Expected-versus-measured reconciliation.
14. Correctness or convergence evidence.
15. Negative results and fallback ladder.
16. Exact recommendation and scope.
17. Artifact manifest.
18. Remaining blockers.

### 4.3 Case-study artifact bundle

Every measured arm must provide:

- human-readable arm name and run ID;
- source repository URL or local path and commit;
- container digest and package versions;
- hardware, firmware, partition mode, and topology;
- complete configuration and effective flags;
- launch command;
- workload and data provenance;
- warmup, timing, and synchronization details;
- raw log;
- machine-readable metrics;
- optimized HLO;
- profile or an explicit reason it was not collected;
- memory output;
- correctness output;
- checksum or immutable reference for large artifacts;
- text explaining protocol deviations.

An aggregate plot must point back to the run IDs used to generate it.

### 4.4 Recommendation record

End each configuration chapter and case study with:

- recommended setting;
- workload range;
- tokens/s/GPU effect;
- memory effect;
- numerical effect;
- software versions;
- proof method;
- fallback;
- retest trigger.

## 5. Exact proposed repository map

The chapter and appendix paths below are the current active layout. The proposed
`artifacts/` manifests remain evidence work rather than published result bundles.

```text
scale-your-amd/
├── index.md
├── docs/
│   ├── structure.md
│   └── structure-v3.md
├── pages/
│   ├── 1-mi355x-as-a-training-machine.md
│   ├── 2-lowering-jax-jit-on-rocm.md
│   ├── 3-profiling-and-analysis-of-one-training-step.md
│   ├── 4-training-in-mixed-precision.md
│   ├── 5-making-the-model-fit.md
│   ├── 6-jax-shardings-to-a-training-mesh.md
│   ├── 7-a-map-of-kernel-backends-on-jax.md
│   ├── 8-mixture-of-experts-on-mi355x.md
│   ├── 9-compiler-runtime-and-rccl-controls.md
│   ├── 10-llama-7b-exposing-the-complete-stack.md
│   ├── 11-llama-2-70b-mixed-precision-training.md
│   ├── 12-mixtral-8x22b-sharding-meshes-and-moe-optimizations.md
│   ├── a-reproducible-mi355x-environment.md
│   ├── b-measurement-and-convergence-protocol.md
│   ├── c-configuration-quick-reference.md
│   ├── d-profiler-and-hlo-cookbook.md
│   ├── e-compatibility-and-negative-results.md
│   ├── f-case-study-artifact-schema.md
│   └── archive/
│       ├── index.md
│       ├── 1-rooflines.md
│       ├── 2-amd-gpus.md
│       ├── 3-profiling.md
│       ├── 4-sharding.md
│       ├── 5-transformers.md
│       ├── 6-training.md
│       ├── 7-moe.md
│       ├── 8-getting-to-roofline.md
│       ├── 9-llama.md
│       ├── 10-deepseek.md
│       ├── 11-inference.md
│       ├── 12-serving.md
│       ├── 13-conclusion.md
│       ├── a-appendix-install.md
│       ├── b-appendix-protocol.md
│       └── v2-3-dl-methods.md
└── artifacts/
    ├── README.md
    ├── schema.md
    ├── llama7b/
    │   └── <run-id>/
    │       ├── manifest.yaml
    │       └── checksums.sha256
    ├── llama70b/
    │   └── <run-id>/
    │       ├── manifest.yaml
    │       └── checksums.sha256
    └── mixtral8-22b/
        └── <run-id>/
            ├── manifest.yaml
            └── checksums.sha256
```

Large logs, HLO dumps, and traces may remain in durable external storage or the
experiment repositories. The book repository manifests must still provide immutable
references, checksums, and enough metadata to audit every published number.

### 5.1 Completed page migration

- `index.md`, Chapters 1 through 12, and Appendices A through F now use the paths
  listed above.
- The former `pages/3-dl-methods.md` is preserved as
  `pages/archive/v1/v2-3-dl-methods.md`.
- The former Chapter 4 through 7 stubs were replaced by the renumbered case-study
  drafts.
- Archived appendix sources remain under `pages/archive/v1/`; active appendices contain
  the MI355X revisions.
- Front matter, previous/next links, section numbers, index navigation, TOC anchors,
  and the Jekyll build are validated as one chain.

## 6. Archive reuse map

Archived pages are source material, not drop-in chapters. Remove MI300X constants,
serving scope, stale support claims, and old chapter-number assumptions before reuse.

### `pages/archive/v1/1-rooflines.md`

- Reuse compact explanations of compute, memory, and communication bounds in
  Chapter 3.
- Reuse profiler-oriented roofline material in Chapter 3 or Appendix D.
- Do not recreate a standalone generic roofline chapter.
- Replace every hardware constant with a cited MI355X value.

### `pages/archive/v1/2-amd-gpus.md`

- Reuse MI355X-relevant architecture explanations in Chapter 1.
- Retain only short MI300X comparisons that explain a changed MI355X result.
- Correct BF16 peak and machine balance.
- Correct xGMI directionality and aggregate-bandwidth wording.
- Remove any implication that JAX sees pooled memory across GPUs.

### `pages/archive/v1/3-profiling.md`

- Reuse tooling limitations, command patterns, and profiler signatures in Chapter 3
  and Appendix D.
- Recheck all zero-field, counter, and XProf limitations on the pinned stack.
- Prefer the tighter active `pages/3-profiling-and-analysis-of-one-training-step.md` structure when material overlaps.

### `pages/archive/v1/4-sharding.md`

- Reuse notation, collective-cost examples, sharded matmul cases, and parallelism
  mechanics in Chapters 3 and 6.
- Link to the Scaling Book for full generic derivations.
- Replace MI300X topology assumptions with the MI355X eight-GPU topology.
- Move implementation recipes into the MaxText field sections of Chapter 6.

### `pages/archive/v1/5-transformers.md`

- Reuse parameter, FLOP, activation, and MoE accounting examples in Chapter 3.
- Reuse only compact architectural context in Chapters 4 and 8.
- Do not repeat a general Transformer tutorial.
- Recalculate all examples for Llama 7B, Llama 70B, and Mixtral 8x22B.

### `pages/archive/v1/6-training.md`

- Reuse parallelism, rematerialization, gradient accumulation, and optimizer-state
  material in Chapters 5 and 6.
- Split conceptual sharding from measured MaxText recommendations.
- Remove unsupported multi-node implications.
- Revalidate all configuration fields.

### `pages/archive/v1/7-moe.md`

- Reuse routing, imbalance, capacity, expert implementation, and AllToAll material in
  Chapter 8.
- Move model-specific observations into Chapter 12.
- Recheck grouped-kernel and ragged-collective support on the pinned stack.

### `pages/archive/v1/8-getting-to-roofline.md`

- Reuse the cheapest-first diagnosis sequence in Chapter 3.
- Reuse worked performance-gap patterns in Chapters 7 and 9.
- Remove broad tuning advice that has no exercised MI355X control or evidence.

### `pages/archive/v1/9-llama.md`

- Use as background for the structure of Chapters 10 and 11.
- Do not carry MI300X results into the MI355X case studies.
- Preserve useful prediction-versus-measurement framing after updating the workload.

### `pages/archive/v1/10-deepseek.md`

- Keep the model-ledger and MoE acceptance material archived for possible future use.
- Do not reuse old measurements as new performance evidence.
- Mark every unexecuted path as future.

### `pages/archive/v1/11-inference.md` and `pages/archive/v1/12-serving.md`

- Keep archived.
- Link externally when Chapter 0 explains the training-only boundary.
- Do not migrate their content into active chapters.

### `pages/archive/v1/13-conclusion.md`

- Keep archived.
- Reuse only concrete limitations or retest items in Appendix E.
- Do not add a generic conclusion chapter.

### `pages/archive/v1/a-appendix-install.md`

- Use as the starting point for `pages/a-reproducible-mi355x-environment.md`.
- Replace the environment matrix with the pinned MI355X stack.
- Add commit, patch, topology, and smoke-test requirements.

### `pages/archive/v1/b-appendix-protocol.md`

- Use as the starting point for `pages/b-measurement-and-convergence-protocol.md`.
- Make tokens/s/GPU the primary output.
- Add artifact-manifest, convergence, and protocol-exception rules.

### Existing active pages

- `pages/1-mi355x-as-a-training-machine.md` is the primary source for Chapter 1.
- `pages/2-lowering-jax-jit-on-rocm.md` is the primary source for Chapter 2.
- `pages/3-profiling-and-analysis-of-one-training-step.md` is the primary source for Chapter 3.
- The current Llama and Mixtral pages provide intent and notes, while the experiment
  repositories and captured artifacts provide evidence.

## 7. Experiment-repository ownership

### Llama 7B

Source repository: `/home/clchong/work/llama7b`

Book ownership:

- narrative;
- normalized comparisons;
- selected configurations;
- analytical ledgers;
- profile interpretation;
- recommendation and fallback.

Experiment-repository ownership:

- raw-JAX and MaxText runners;
- attention variants;
- rematerialization variants;
- profiler scripts;
- exact flags and setup;
- raw result artifacts.

The book manifest at `artifacts/llama7b/manifest.yaml` should point to exact commits
and artifacts rather than duplicate changing scripts.

### Llama 70B

Source repository: `/home/clchong/work/llama70b`

Book ownership:

- precision comparison;
- tokens/s/GPU ranking;
- Amdahl and profile interpretation;
- completed one-billion-token convergence finding;
- tested-horizon qualification;
- final precision decision.

Experiment-repository ownership:

- train-step runners;
- convergence runners;
- C4 and tokenizer setup;
- Transformer Engine and MXFP4 setup;
- MaxText configuration and flags;
- raw results and checkpoints.

The book manifest at `artifacts/llama70b/manifest.yaml` must distinguish completed
runs from missing publication artifacts.

### Mixtral 8x22B

Source repository: `/home/clchong/work/mixtral8-22b`

Book ownership:

- model and routing ledger;
- normalized mesh and expert-path comparison;
- tokens/s/GPU ranking;
- profile and communication interpretation;
- recommendation and fallback.

Experiment-repository ownership:

- baseline and mesh runners;
- expert implementations;
- latency-hiding comparison;
- flags and setup;
- raw v26.6 result artifacts.

The book manifest at `artifacts/mixtral8-22b/manifest.yaml` remains blocked until the
v26.6 result bundle exists.

## 8. Dependencies

### 8.1 Reader dependencies

- Chapter 0 has no dependency.
- Chapter 1 has no chapter dependency.
- Chapter 2 assumes Chapter 1's device and memory vocabulary.
- Chapter 3 assumes Chapters 1 and 2.
- Chapter 4 assumes Chapters 1 through 3.
- Chapter 5 assumes Chapters 1 through 3 and previews Chapter 6 where state sharding
  affects capacity.
- Chapter 6 assumes Chapters 1 through 3.
- Chapter 7 assumes Chapters 1 through 4.
- Chapter 8 assumes Chapters 3, 6, and 7.
- Chapter 9 assumes Chapters 2, 3, 6, 7, and 8.
- Chapter 10 assumes Chapters 1 through 7 and Chapter 9.
- Chapter 11 assumes Chapters 1 through 7 and Chapter 9.
- Chapter 12 assumes Chapters 1 through 9.
- Appendix A supports every measured chapter.
- Appendix B is normative for Chapter 3 and Chapters 10 through 12.
- Appendix C is generated from the controls established in Chapters 4 through 9.
- Appendix D supports Chapters 2, 3, 7, and every case study.
- Appendix E collects failures from Chapters 2, 4, 7, 8, 9, and 10 through 12.
- Appendix F is the evidence index for Chapters 10 through 12.

### 8.2 Authoring dependencies

Reading order and writing order differ. Chapters 4 through 9 appear before the case
studies, but their recommendations must be grounded in case-study measurements.

- Chapter 1 requires corrected and cited MI355X constants.
- Chapter 2 requires a pinned software stack and path-verification examples.
- Chapter 3 requires frozen case-study workload ledgers, Appendix B's protocol,
  and Appendix F's artifact contract.
- Chapters 4, 5, and 7 require Llama 7B or Llama 70B artifacts.
- Chapters 6, 8, and 9 require Mixtral artifacts for MoE and collective
  recommendations.
- Chapters 10 through 12 require Chapters 1 through 3 for vocabulary and method, but
  their measured sections should be drafted before final recommendations in
  Chapters 4 through 9.

### 8.3 Repository dependencies

- Do not change active navigation before roadmap review.
- Freeze `pages/b-measurement-and-convergence-protocol.md` and the artifact schema before promoting
  planned results to `[measured]`.
- Every active page needs final front matter and previous/next links.
- `index.md` must link only to files created in the skeleton migration.
- Appendix C should be assembled from chapter-owned controls, not written as an
  independent list.
- Appendix E should be updated whenever a chapter records a fallback or incompatibility.
- Appendix F should be generated from validated manifests where practical.

## 9. Writing and migration order

### Phase 0 — Review this roadmap (complete)

1. Check the outline against MI355X-only and training-only scope.
2. Check Chapters 10 through 12 against the current experiment repositories.
3. Remove sections without a mechanism, control, evidence source, or prerequisite role.
4. Confirm the exact file map and archive destinations.
5. Approve the roadmap before active-page migration.

### Phase 1 — Migrate the skeleton (complete)

1. Preserve `pages/3-dl-methods.md` at
   `pages/archive/v1/v2-3-dl-methods.md`.
2. Rename the three current case-study pages to Chapters 10, 11, and 12.
3. Create empty, front-matter-complete skeletons for Chapters 3 through 9.
4. Create active appendix skeletons A through F.
5. Update `index.md`, all previous/next links, section numbers, and navigation.
6. Build the site and check every route.

### Phase 2 — Freeze evidence contracts

1. Finalize Appendix B's measurement and convergence protocol.
2. Finalize `artifacts/schema.md`.
3. Create and validate the three case-study manifests.
4. Record allowed protocol exceptions.
5. Define plot provenance and checksum requirements.

### Phase 3 — Draft foundations

Draft Chapters 1 through 3 in order:

1. correct hardware constants and topology;
2. pin the software path and failure vocabulary;
3. freeze workload ledgers and prediction worksheets;
4. freeze profiling, measurement, and reporting methods.

### Phase 4 — Draft measured case-study sections

Draft evidence-backed portions in this order:

1. Llama 7B sections supported by captured artifacts;
2. Llama 70B train-step sections;
3. the completed Llama 70B one-billion-token convergence result;
4. Mixtral sections after the v26.6 result bundle exists.

Keep missing sections visibly blocked. Do not wait for every artifact before drafting
claims that already have complete evidence.

### Phase 5 — Draft configuration chapters

Use case-study evidence to complete Chapters 4 through 9:

1. precision;
2. memory;
3. sharding;
4. kernels;
5. MoE;
6. compiler, runtime, and RCCL controls.

Each recommendation should point to the case-study evidence that supports it.

### Phase 6 — Complete case-study narratives

Finish Chapters 10 through 12 with:

- analytical-versus-measured reconciliation;
- negative results;
- recommendations;
- limits;
- validated artifact links.

### Phase 7 — Complete appendices

1. Pin Appendix A to the final experiment environments.
2. Keep Appendix B normative and versioned.
3. Aggregate tested controls into Appendix C.
4. Move detailed tool recipes into Appendix D.
5. Populate Appendix E from recorded failures.
6. Validate every Appendix F artifact link and checksum.

## 10. Evidence and artifact blockers

### 10.1 Global blockers

- The shared measurement protocol is not yet frozen in active Appendix B.
- The artifact schema and manifests do not yet exist in the proposed book paths.
- Hardware, container, package, repository, patch, and effective-flag manifests must
  be consistent across case studies.
- Plot sources and run-to-plot provenance must be publishable.
- Every support statement needs a version and test date.
- Every measured comparison needs fixed-workload verification.

### 10.2 Chapter-specific blockers

Chapter 1:

- authoritative sources for MI355X capacity, bandwidth, and peak values;
- corrected BF16 machine balance;
- precise directional and aggregate xGMI wording;
- confirmation of JAX-visible partition behavior;
- removal of pooled-memory implications.

Chapter 2:

- pinned JAX, XLA, ROCm, PJRT, MaxText, TE, and AITER versions;
- representative HLO, custom-call, and kernel-name examples;
- tested initialization-order behavior;
- dated fallback examples.

Chapter 3:

- frozen workload ledgers for all three case studies;
- checked FLOP, memory, and communication arithmetic;
- one reusable worksheet with consistent units;
- final warmup, synchronization, repetition, and statistic policy;
- documented profiler commands on the pinned stack;
- verified XProf and counter limitations;
- artifact bundle schema.

Chapter 4:

- complete Llama 70B train-step outputs;
- backend-path proof for each precision arm;
- publication-ready convergence plot and provenance;
- versioned support and fallback records.

Chapter 5:

- comparable Llama 7B rematerialization memory and timing outputs;
- Llama 70B expected-versus-observed memory records;
- evidence for donation, scan, and policy interactions.

Chapter 6:

- MI355X RCCL message-size and participant-count evidence;
- HLO replica-group examples;
- Mixtral mesh-sweep results;
- explicit single-node versus future scale-out boundary.

Chapter 7:

- forward and backward path checks for every attention route;
- kernel names or custom-call proof;
- correctness comparisons;
- dated support state for Pallas, Triton, TE, and JAX-AITER paths.

Chapter 8:

- Mixtral v26.6 expert-path results;
- tokens-per-expert, padding or drop, GEMM-efficiency, and AllToAll diagnostics;
- verified GroupedGEMM dtype restriction;
- evidence for the fallback ranking.

Chapter 9:

- before-and-after results for every recommended flag;
- effective-flag capture;
- deprecation and no-op checks;
- numerical-hazard reproductions where applicable.

Chapter 10:

- consolidated Llama 7B result bundle;
- comparable raw-JAX and MaxText runs;
- attention sweep artifacts;
- one-GPU and FSDP-8 rematerialization artifacts;
- HLO and profile correlation.

Chapter 11:

- FP32 train-step timing;
- post-patch MXFP8 timing;
- peak memory for every arm;
- representative profiles;
- final convergence plot, run mapping, and data/tokenizer provenance.

The one-billion-token convergence experiment itself is complete. The blockers above
concern publication and auditability, not whether the experiment ran.

Chapter 12:

- all v26.6 result artifacts;
- mesh-sweep metrics;
- expert-path metrics;
- latency-hiding comparison;
- router and collective diagnostics;
- HLO and kernel attribution.

### 10.3 Case-study publication states

- Llama 7B: implementation sources exist; publication waits on a consolidated,
  validated result bundle.
- Llama 70B train-step study: partially publishable after missing timing, memory, and
  profile outputs are captured.
- Llama 70B convergence study: completed through approximately one billion tokens;
  publication waits on plot and provenance packaging.
- Mixtral 8x22B: blocked because v26.6 result artifacts do not yet exist.

## 11. Roadmap completion criteria

The v3 migration is complete when:

- all planned files exist at the paths in section 5;
- active navigation matches the reading order;
- generic theory is linked or compactly recapped rather than duplicated;
- every substantive claim has an evidence label;
- every software claim is date- and version-pinned;
- tokens/s/GPU is the primary result in every performance comparison;
- the completed Llama 70B one-billion-token convergence finding is published with its
  tested-horizon qualification and provenance;
- every measured number resolves to a validated artifact manifest;
- negative results and fallbacks are indexed;
- Mixtral remains blocked until its v26.6 artifacts exist;
