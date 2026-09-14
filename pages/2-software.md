---
layout: distill
title: "What jax.jit Runs on ROCm"
description: "The ROCm stack, HLO, and the XLA compiler path from a jax.jit to a HIP executable."
date: 2026-09-10

section_number: 2

previous_section_url: "/pages/1-hardware"
previous_section_name: "Chapter 1: Hardware"

next_section_url: "/pages/3-cost-model"
next_section_name: "Chapter 3: Cost Model"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "The Executable Contract"
  - name: "JAX Transformations"
  - name: "The IR Ladder"
  - name: "Shardy and Partitioning"
  - name: "The XLA GPU Pipeline"
    subsections:
      - name: "Rewrites Layout and Fusion"
      - name: "Scheduling and Buffer Assignment"
      - name: "Code Generation and the Executable"
  - name: "Kernel Routes on MI355X"
  - name: "XLA FFI"
  - name: "PJRT and the Runtime"
  - name: "Autotuning and Caches"
  - name: "Initialize the ROCm Backend Once"
  - name: "Verify the Path"
  - name: "Failure Taxonomy"
---

## The Executable Contract

`jax.jit` does not map each JAX operation to one fixed ROCm library call. It
specializes a function, partitions it across the selected devices, optimizes the
result, chooses kernel routes, and packages those routes into an executable. On
MI355X, a successful call proves only that the program ran. It does not prove that
JAX initialized the ROCm backend, that the intended sharding was used, or that a
particular kernel ran.

This chapter builds the proof chain:

```
JAX function
  -> jaxpr
  -> StableHLO with sharding information
  -> partitioned and optimized HLO
  -> generated kernels, library calls, collectives, and FFI calls
  -> PJRT executable
  -> HIP/HSA dispatches on gfx950
```

The practical questions are therefore concrete:

1. Which function and input signature did JAX compile?
2. Which per-device program and communication did Shardy produce?
3. Which XLA route was selected for each expensive operation?
4. Which kernel name reached the MI355X?

The mechanisms below are documented by JAX, OpenXLA, and ROCm and are marked
`[cited]`. The experiment examples are source-verified against the checked-in Llama 7B,
Llama 70B, and Mixtral launchers. This chapter reports no timing result.

## JAX Transformations

A JAX program is a composition of transformations, not a sequence of eager library
calls. [JAX transformations](https://docs.jax.dev/en/latest/101/transformations.html)
trace array operations and produce a new program. `[cited]`

- A **PyTree** gives parameters, optimizer state, and batches a nested structure.
  Transformations flatten that structure to array leaves and reconstruct it at the
  boundary. The tree structure, leaf shapes, dtypes, shardings, and static arguments
  help define the compiled signature.
- `jax.jit` specializes a function for that signature. A new shape, dtype, static
  value, or relevant sharding can require another trace or compilation.
- `jax.value_and_grad` or `jax.grad` transforms the function before XLA compilation.
  The backward calculation is therefore present in jaxpr and HLO; PJRT does not
  invoke a separate automatic-differentiation runtime.
- `jax.lax.scan` lowers a repeated body to a loop. A Python loop inside `jit` is
  normally unrolled during tracing, while
  [`scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) lowers to
  one `WhileOp`. This can make a repeated-layer HLO smaller and reduce compile time.
  `[cited]`
- [JAX dispatch is asynchronous](https://docs.jax.dev/en/latest/async_dispatch.html).
  A returned `jax.Array` can be a future, so elapsed host time is not device time
  until an output is made ready. Use
  `jax.block_until_ready(result)` when timing. `[cited]`

The raw-JAX Llama 7B experiment follows this contract. It differentiates the loss,
JIT-compiles the whole update, donates parameters and optimizer state, lowers and
compiles before the step loop, and blocks after each dispatch:

```python
def step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

compiled = jax.jit(step, donate_argnums=(0, 1)).lower(
    params, opt_state, batch
).compile()

params, opt_state, loss = compiled(params, opt_state, batch)
jax.block_until_ready((params, opt_state, loss))
```

Decide the compilation boundary before tuning a backend. A whole-step boundary lets
XLA see more fusion and aliasing opportunities. It also makes variable shapes and
changing static arguments more expensive because they can compile another whole
step. The first verification is to log the signature and separate lowering,
compilation, and execution time.

## The IR Ladder

The representations answer different questions. Treating all of them as “HLO”
hides where a decision was made.

1. **jaxpr** records the specialized JAX primitives after transformations. Inspect it
   with `jax.make_jaxpr(f)(*args)` or `jax.jit(f).trace(*args).jaxpr`. Use it to check
   that autodiff, control flow, casts, and custom primitives entered the program.
2. **StableHLO** is the portable input to XLA. Obtain it from
   `lowered.compiler_ir(dialect="stablehlo")` or `lowered.as_text()`. It contains
   tensor operations, types, source metadata, and sharding annotations, but it does
   not show the final MI355X kernel choice.
3. **HLO** is XLA's internal optimizer representation. A pre-optimization HLO is
   still close to StableHLO. The optimized HLO shows fusions, layouts, copies,
   collectives, custom calls, backend configuration, and selected algorithms.
4. **MLIR and LLVM IR** appear on XLA-generated and Triton-generated code paths.
   XLA's native GPU emitters lower fusions through GPU-specific MLIR to LLVM IR.
5. **AMDGPU ISA and HSACO** are the generated device instructions and code objects
   for gfx950. An LLVM module targeting `amdgcn-amd-amdhsa` confirms AMDGPU code
   generation, but only disassembly or counters establish which instructions the
   final kernel contains.

The ladder branches at optimized HLO. A library or FFI `custom-call` does not expose
the called kernel's LLVM IR to XLA. XLA emits a runtime call, while hipBLASLt,
RCCL, Transformer Engine, or AITER supplies and launches its own code object.
Consequently, the absence of a library kernel from XLA's `.ll` files is expected.

A small unoptimized HLO still reads as a data-flow graph:

```text
HloModule jit_matmul

ENTRY main (a: bf16[16,2048], b: bf16[2048,10]) -> bf16[16,10] {
  a = bf16[16,2048]{1,0} parameter(0)
  b = bf16[2048,10]{1,0} parameter(1)
  ROOT dot = bf16[16,10]{1,0} dot(a, b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
```

`{1,0}` is a minor-to-major physical layout: dimension 1 is contiguous. The
contracting dimensions state which axes are reduced. Neither field says whether
the final dot uses hipBLASLt, rocBLAS, Triton, or an XLA emitter. That decision is
visible only after optimization.

JAX's
[ahead-of-time API](https://docs.jax.dev/en/latest/aot.html) exposes the stage
boundaries:

```python
traced = jax.jit(f).trace(*args)
print(traced.jaxpr)

lowered = traced.lower()
print(lowered.as_text())       # normally StableHLO

compiled = lowered.compile()
print(compiled.memory_analysis())
```

These text and analysis methods are debugging interfaces. JAX does not guarantee
their availability or output format across backends and versions. Save the package
manifest with every dump rather than parsing it as a stable API. `[cited]`

## Shardy and Partitioning

JAX presents global arrays to `jit`. `NamedSharding`, `PartitionSpec`, input and
output shardings, and explicit constraints describe how those arrays may be divided.
[Shardy](https://openxla.org/shardy/overview) propagates that information through
the program, resolves unspecified shardings, partitions the global computation into
a per-device SPMD program, and inserts the data movement required between local
shapes. `[cited]`

This distinction matters on an eight-MI355X node. A global dot can become smaller
local dots with no communication, or it can require an AllGather, ReduceScatter, or
AllReduce. The Python operation alone does not determine which case occurred.

`jax.shard_map` provides a different control point. Its body is written in the local
per-device view and can contain explicit collectives. It composes with `jit`, so a
program can use manual partitioning across one mesh axis and compiler partitioning
across another. The
[`shard_map` guide](https://docs.jax.dev/en/latest/201/shard-map.html) describes this
division. `[cited]`

The [JAX migration guide](https://docs.jax.dev/en/latest/shardy_jax_migration.html)
describes Shardy-only operation as the post-migration state. The Mixtral config
requests `shardy: true`, but no retained post-propagation dump proves which
partitioner ran in the declared JAX/ROCm environment. Treat the declared-stack
behavior as unverified until the dump is captured. Older GSPMD fallback switches
should not be used as current configuration advice. `[cited]`

Verify partitioning at two points:

- In Shardy's MLIR dump, inspect the propagated axis shardings. The
  [Shardy JAX guide](https://openxla.org/shardy/getting_started_jax) identifies
  `sdy_module_after_sdy_export.mlir` as the useful post-propagation artifact.
- In optimized HLO, inspect local tensor shapes, collective opcodes, channel IDs,
  replica groups, and source metadata. Confirm that groups match the intended mesh
  axis. A valid result can contain no collective when an operation is local.

The Mixtral experiment explicitly enables Shardy and varies FSDP and expert axes.
Its configuration fields express intent. Only the optimized HLO proves the local
shapes and collectives generated from that intent.

## The XLA GPU Pipeline

OpenXLA documents the complete
[HLO-to-thunks path](https://openxla.org/xla/hlo_to_thunks). The exact pass list
changes between XLA revisions, but the decisions remain recognizable. `[cited]`

### Rewrites Layout and Fusion

Sharding and general simplification run before backend-specific code generation.
Collective passes can combine, reorder, or make collectives asynchronous. Layout
assignment then attaches a physical minor-to-major order to each shape.

Layout is a real performance constraint. Library calls accept particular operand
orders, and generated kernels need coalesced accesses. XLA propagates compatible
layouts through the graph. If two requirements cannot be reconciled, it inserts a
`copy`; if the change can be represented as a bitcast, no data moves. Search
optimized HLO for copies around large dots and collectives before blaming the
kernel itself.

Post-layout passes rewrite eligible operations to library calls or specialized
emitters and run autotuning. Fusion then groups compatible HLO operations. An XLA
GPU fusion becomes one GPU kernel, allowing intermediate values to remain in
registers or LDS instead of being written to HBM. A missed fusion is therefore
observable as both separate HLO fusions and separate dispatches.

### Scheduling and Buffer Assignment

The scheduler chooses one valid execution order from the HLO dependency graph.
XLA first evaluates schedules for peak memory. The latency-hiding scheduler can
then move asynchronous communication relative to compute, which may lengthen buffer
lifetimes. If the estimated peak exceeds the available memory, compiler
rematerialization may shorten lifetimes by recomputing values. `[cited]`

Buffer assignment runs after scheduling because lifetimes depend on execution
order. It maps logical HLO values to slices of device allocations and reuses a slice
when lifetimes do not overlap. Donation adds legal input-output aliasing; it does not
change the numerical function. JAX exposes the compiler estimate through
`compiled.memory_analysis()` when the backend provides it.

The useful artifacts are the optimized HLO and the file ending in
`after_optimizations-buffer-assignment.txt`. Compare their layouts, copies, buffer
lifetimes, and aliases with the runtime high-water mark. A static buffer plan and an
allocator peak answer different questions.

### Code Generation and the Executable

After scheduling and buffer assignment, XLA lowers the entry computation to a
linear sequence of **thunks**. A thunk is one runtime action: launch a generated
kernel, call a BLAS routine, execute a collective, copy memory, or run control flow.
One HLO instruction can produce zero, one, or several thunks.

For a native fusion, the current emitter path is:

```
HLO fusion
  -> XLA GPU MLIR
  -> LLVM IR
  -> AMDGPU code generation
  -> gfx950 HSACO
  -> kernel-launch thunk
```

The final executable combines the thunk sequence, buffer assignment, generated
code objects, and references to external library calls. Command-buffer conversion
may replace compatible runs of thunks with one replayable command-buffer thunk.
On ROCm that is a HIP Graph path. Whether it helps or works for a declared workload is
a versioned runtime question, not something that follows from `jax.jit`.

## Kernel Routes on MI355X

An expensive HLO operation can take several routes. Selection can also be nested:
XLA may select a library entry point, and that library may select one of its own
kernels.

- **hipBLASLt or rocBLAS.** Eligible dots and grouped matmuls become library custom
  calls with algorithm and workspace configuration. The optimized HLO proves the
  library call; the kernel trace proves the solution launched by the library.
- **RCCL.** Shardy-produced collectives become collective thunks and RCCL calls.
  Optimized HLO proves the operation, local shape, and replica group. An RCCL API
  trace and kernel trace prove runtime execution.
- **Triton.** XLA can select Triton for eligible GEMM- or reduction-shaped fusions.
  JAX code can also reach the Triton AMD backend through a Pallas-based package such
  as Tokamax. These are different entry routes even if both eventually use Triton
  and LLVM-AMDGPU.
- **XLA emitters.** Loop, reduction, transpose, and other emitters generate code
  directly. A fusion in optimized HLO, corresponding `.ll` module, and matching
  kernel dispatch form the proof.
- **External extensions.** Transformer Engine and JAX-AITER register opaque
  operations. Their HLO custom-call target proves entry into the extension. A trace
  is still required to distinguish an internal CK, AITER ASM, hipBLASLt, or fallback
  kernel.

ROCm uses a shared XLA GPU backend, so CUDA-derived names remain in flags, HLO
targets, and internal APIs. They are compatibility names, not proof that an NVIDIA
library loaded. For example, current XLA
[lowers a supported ROCm grouped matmul](https://github.com/openxla/xla/pull/38735)
to `custom_call_target="__cublas$lt$groupedMatmul"` and executes it with
hipBLASLt. On gfx950, the
[eligibility check](https://github.com/openxla/xla/pull/38732) documents FP16
support for this grouped route. `[cited]`

The same caution applies to controls. The supplied Mixtral launcher sets
`--xla_gpu_enable_cublaslt=true` for its declared stack. The flag's upstream
status changed more than once: an earlier deprecation was reverted, and a later
change deprecated it again. That history does not prove whether the v26.6 XLA
treats it as active, deprecated, or a no-op. Record the loaded XLA revision and
effective debug options, then inspect optimized HLO and the trace.

## XLA FFI

[XLA FFI](https://openxla.org/xla/custom_call) connects an HLO `custom-call` to an
external handler registered in the process. On GPU, the handler receives device
buffer views and the active ROCm stream, then enqueues its work on that stream.
JAX's typed interface uses `jax.ffi.register_ffi_target` and
`jax.ffi.ffi_call`. `[cited]`

An FFI wrapper must define more than a function pointer:

- the target name and ROCm platform registration;
- output shapes and dtypes known during tracing;
- accepted input and output layouts;
- static attributes and side-effect behavior;
- input-output aliases, if any;
- batching, differentiation, and partitioning rules.

The kernel body is opaque to XLA. XLA can schedule the call and allocate its visible
buffers, but it cannot fuse through the call or repair an incorrect ABI. Gradients
do not appear automatically: a package must provide a `custom_vjp`, `custom_jvp`, or
primitive transpose rule. Multi-device use likewise needs a valid sharding rule or
an explicit `shard_map`.

The JAX-AITER alpha2 source used by the experiments makes these layers visible. Its
raw attention wrappers call the registered ROCm targets `MhaFwdUnifiedJA` and
`MhaBwdUnifiedJA`. The public attention API adds a custom VJP and custom
partitioning around those calls. Its BF16 and MXFP4 GEMM APIs use the same pattern.
The [JAX-AITER project](https://github.com/ROCm/jax-aiter) describes the route as
JAX buffer to FFI handler to AITER kernel launch, without a PyTorch runtime.
`[cited]`

For an FFI path, verify three separate contracts:

1. Import and target registration succeed before lowering.
2. The optimized HLO contains both forward and backward targets where gradients are
   required, with the expected local shapes and layouts.
3. A kernel trace shows the intended AITER or CK implementation, and a small
   reference comparison checks outputs and gradients.

## PJRT and the Runtime

PJRT is the API boundary between JAX and the backend. The ROCm installation has two
plugin packages: AMD's
[OpenXLA and JAX on ROCm guide](https://rocm.blogs.amd.com/software-tools-optimization/openxla-jax-rocm/README.html)
documents the `jax-rocm{N}-pjrt` wheel as the native PJRT C-API plugin with HIP
runtime and RCCL integration, and the `jax-rocm{N}-plugin` wheel as the Python
wrapper that JAX discovers and registers. Their versions must match JAX, jaxlib,
and the installed ROCm release. `[cited]`

The runtime path is:

```
JAX
  -> ROCm PJRT client
  -> loaded XLA executable
       - buffer assignment
       - thunk sequence
       - generated HSACO
       - library and FFI call sites
  -> StreamExecutor streams and events
  -> HIP runtime
  -> HSA runtime and amdgpu driver
  -> MI355X
```

PJRT creates buffers, compiles or loads executables, transfers ownership, and
submits execution. StreamExecutor supplies XLA's device, stream, event, memory, and
library abstractions. HIP/HSA performs the device-level load and dispatch. A BLAS
thunk can call hipBLASLt, which may enqueue more than one kernel; a collective thunk
can do the same through RCCL. Do not assume one HLO op equals one dispatch.

JAX returns before queued device work necessarily finishes. This is why backend
initialization, compilation, and steady execution must be timed separately, and why
the last result must be blocked before stopping a timer.

## Autotuning and Caches

Autotuning is part of compilation. XLA constructs valid candidates for eligible
library and generated-kernel routes, runs candidates on the target GPU, checks them
according to the configured level, and stores the selected algorithm or tile in the
compiled program. The supplied experiment flag files request
`--xla_gpu_autotune_level=4`; they contain no measured comparison against another
level.

Three caches are easy to confuse:

1. The **in-process JIT cache** reuses an executable for the same function identity
   and compatible signature. New shapes, dtypes, static values, or newly created
   function objects can miss it.
2. The **JAX persistent compilation cache** stores compiled executables across
   processes. The
   [JAX cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)
   says to set `jax_compilation_cache_dir` before the first compilation. Its key
   includes non-optimized HLO, jaxlib, relevant XLA flags, and device configuration.
3. The **XLA autotune cache** stores per-fusion tuning results. OpenXLA recommends
   `--xla_gpu_per_fusion_autotune_cache_dir=DIR`; whole-program dump/load flags are
   also available. `[cited]`

```python
import jax

jax.config.update("jax_compilation_cache_dir", "/cache/jax/gfx950-xla-version")
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)
```

The
[persisted-autotuning guide](https://openxla.org/xla/persisted_autotuning) requires
the directory to exist and places invalidation on the user. Separate caches by GPU
type and XLA version. A different HLO can reuse matching fusion entries and tune the
rest. A changed compiler can produce different fusions, making old entries
irrelevant. `[cited]`

State cache conditions in every comparison: cold executable cache, warm executable
cache, cold autotune cache, or warm autotune cache. The raw-JAX experiment defaults
to a cold persistent cache unless a cache directory is explicitly configured. This
keeps compiler-route comparisons attributable; production restarts may prefer a
versioned warm cache.

## Initialize the ROCm Backend Once

Environment variables and XLA flags must be fixed before the ROCm backend is
initialized. Some libraries import JAX or register FFI targets as a side effect, so
the safest pattern is to construct a clean environment and launch the workload in a
new process. This is the pattern used by all three experiment repositories.

At minimum, set the device visibility, required platform, and complete XLA flag
string before importing the workload:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export JAX_PLATFORMS=rocm
export XLA_FLAGS="--xla_gpu_autotune_level=4"
python3 workload.py
```

`JAX_PLATFORMS=rocm` is a useful assertion. JAX's
[configuration reference](https://docs.jax.dev/en/latest/config_options.html#platforms)
says that every explicitly listed platform must initialize successfully; without
it, automatic discovery can fall back to CPU when an accelerator backend is
unavailable. `[cited]`

The first call to `jax.devices()`, array placement, or compilation can initialize
the client. Changing `HIP_VISIBLE_DEVICES`, `JAX_PLATFORMS`, allocator controls, or
`XLA_FLAGS` after that point does not rebuild the client or previously compiled
executables. Set Python configuration such as the persistent-cache directory before
the first compilation as well.

Log the loaded stack rather than inferring it from an image tag:

```python
from importlib.metadata import version
import jax
from jax.extend import backend

for package in (
    "jax",
    "jaxlib",
    "jax-rocm7-plugin",
    "jax-rocm7-pjrt",
):
    try:
        print(package, version(package))
    except Exception:
        pass

print("backend:", jax.default_backend())
print("devices:", jax.devices())
print("device_count:", jax.device_count())
print("backend_xla_version:", backend.backend_xla_version("rocm"))
print("platform_version:", jax.devices()[0].client.platform_version)
```

The last `client` field is a diagnostic interface and may change. Pair it with
`rocminfo` output showing `gfx950`, the container digest, package freeze, and every
environment variable used by the launcher. XLA dumps also include a
`.debug_options` artifact that records the options attached to each module.

## Verify the Path

Use the cheapest evidence that can disprove the intended route, then move down one
level.

**1. Backend and device.** Require `jax.default_backend() == "rocm"`, the expected
device count, and `gfx950` in the ROCm hardware inventory. This catches CPU fallback,
visibility mistakes, and the wrong partition mode.

**2. Transformed program.** Inspect jaxpr when a cast, custom primitive, scan, or
custom derivative is in doubt. Inspect StableHLO for operation types, static shapes,
source locations, and initial sharding annotations.

**3. Partitioned program.** Inspect the Shardy export and optimized HLO. Check local
shapes, collectives, replica groups, copies, and layouts against the mesh. This
catches a semantically valid but expensive sharding.

**4. Backend route.** In optimized HLO, search for the relevant `custom_call_target`,
fusion, and backend configuration. A custom-call name proves entry to a library or
extension, not its internal kernel.

**5. Dispatched kernel.** Capture a warmed execution with
[`rocprofv3`](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html):

```bash
rocprofv3 --kernel-trace --output-format csv -- python3 workload.py
```

Add `--rccl-trace` when the collective API call matters. Kernel names are
implementation details and can change, so store the full trace and software
manifest instead of matching one permanent string.

**6. Generated code and counters.** For an XLA-generated kernel, inspect the dumped
LLVM IR and any emitted HSACO. ROCm's LLVM tools can disassemble an available code
object:

```bash
llvm-objdump -d --mcpu=gfx950 kernel.hsaco
```

Use `rocprofv3 --pmc` or `rocprof-compute` only after identifying the dispatch.
Counters verify MFMA issue, memory traffic, occupancy, and cache behavior. Counter
collection can serialize same-GPU streams, so its elapsed time is not a steady-run
measurement.

Create compiler artifacts in a new directory so runs cannot mix:

```bash
DUMP_DIR="/tmp/xla-dump-$$"
mkdir -p "$DUMP_DIR"
export XLA_FLAGS="${XLA_FLAGS:-} \
--xla_dump_to=$DUMP_DIR \
--xla_dump_hlo_as_text \
--xla_gpu_dump_llvmir"
python3 workload.py
```

OpenXLA names the key files `before_optimizations.txt`,
`after_optimizations.txt`, and
`after_optimizations-buffer-assignment.txt`. LLVM files such as
`.ir-no-opt.ll` and `.ir-with-opt.ll` apply to generated code paths.

Two experiment routes illustrate the full proof:

- For Mixtral's ragged dot, the config and flag show intent. A ragged operation in
  early IR, `__cublas$lt$groupedMatmul` in optimized HLO, and a hipBLASLt kernel in
  the trace prove the grouped route. If the custom call is absent, the operation
  took another lowering.
- For direct JAX-AITER attention, importing the package proves only that the wrapper
  loaded. Forward and backward JAX-AITER targets in optimized HLO prove FFI
  lowering. The AITER or CK dispatch names prove the kernel route.

Chapter 4 applies this ladder to traces and counters. Later kernel and flag chapters
own route-specific selection controls; this chapter owns the evidence that a control
had an effect.

## Failure Taxonomy

Classify the failure before changing flags. Different classes require different
evidence.

**Unsupported.** The requested dtype, shape, layout, derivative, sharding rule, or
collective has no implementation in the declared stack. Expect an explicit
`UNIMPLEMENTED`, registration error, or validation error. Record the exact signature
and version, then select a documented fallback or change the shape. Repeated flag
changes cannot add a missing kernel.

**Compile failure.** Tracing, StableHLO verification, Shardy propagation, an XLA
pass, code generation, or compiler memory use fails before execution. Reduce to the
smallest lowering reproducer and save `before_optimizations.txt`, logs, flags, and
the package manifest. If only one sharding triggers it, include both mesh and input
shardings.

**Runtime or ABI failure.** Compilation succeeds, but loading a code object,
resolving an FFI target, launching a kernel, or executing a collective fails. A
missing shared object, invalid stream use, device illegal-address fault, or RCCL
timeout belongs here. Keep the optimized HLO, runtime trace, first failing dispatch,
and extension build manifest.

**Silent fallback.** The program is correct, but the requested implementation never
appears. The decisive evidence is the missing custom-call or unexpected kernel, not
the selector setting. Check eligibility restrictions, package imports, registration,
and deprecated or ignored controls.

**Slow fallback.** The intended family appears, but a generic algorithm, poor tile,
extra copy, materialization, or unfavorable local shape makes it slow. Inspect the
selected algorithm and workspace, layouts around the call, dispatch count, local
dimensions, and counters. This class cannot be diagnosed from the Python API name.

**Numerical failure.** The intended path executes but outputs, gradients, or repeated
updates diverge from a reference, produce NaNs, or corrupt memory. Compare the
smallest forward and backward operation in a higher-precision reference, then test
the composed step. Record tolerances, seeds, accumulation dtype, autotune cache, and
the exact forward and backward kernels. A fast dispatch is not acceptable evidence
of correctness.

A path is accepted only when the backend, transformed program, partitioning,
optimized HLO route, dispatched kernel, and numerical check agree. If one link is
missing, report that link as unverified rather than treating a configuration field
as proof.
