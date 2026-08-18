# The ROCm Ecosystem

Before diving into the JAX stack, it helps to know what we're working with on the ROCm side before reaching the framework level. You can find a brief overview [here](https://rocm.docs.amd.com/en/latest/about/what-is-rocm.html).

![](img/what-is-rocm.png)

The most relevent components for us are the math and compute libraries (hipBLAS, rocBLAS, CK, ...) and communication libraries (RCCL), along with the universally useful profiling and debugging tools (rocprof-sdk, rocprof-compute, rocgdb).

> Note that as of ROCm 7.14, AMD has released their own ROCm build platform [TheRock](https://github.com/ROCm/TheRock/tree/main) which enables modular installation of ROCm components through a centralised build manager.

<INSERT some content about the ROCm compiler stack>:
> amdclang++ (aka HIP-Clang): Clang/LLVM-based compiler shipped in the rocm-llvm package
> hipcc is the driver wrapper that invokes clang (or nvcc on nvidia) with the right HIP include/lib flags
> https://rocm.docs.amd.com/projects/llvm-project/en/latest/index.html
> - and maybe quickly mention the later section on rocm backends (llvm) which involve this

---

# The JAX/XLA/ROCm Stack

Normally, if you are simply researching and building models in JAX, you may not need to concern yourself with XLA internals. However, if you are looking to extract the maximum performance from a large distributed training/inference workload, it is really quite helpful to look into this and understand what certain XLA performance flags achieve and what parts of the stack they affect.

XLA documents their [GPU backend architecture](https://openxla.org/xla/gpu_architecture) and [how HLO is lowered to binary](https://openxla.org/xla/hlo_to_thunks).

![](img/jax-rocm-stack.png)

A high-level crash course summary to understand JAX/XLA:

- Jax supports JIT compilation of a function via `jax.jit()`. Commonly used in its decorator form `@jax.jit` around a function.
    - For example:
        ```python
        @jax.jit
        def matmul(a, b):
            return a @ b
        ```
- Jax traces this function and lowers it into the highest form of IR in the stack, `jaxpr`
- `jaxpr` is subequently lowered to `StableHLO` which you can consider a portable, unoptimised representation of the function, or rather, "module" that you have just JIT'd (e.g. `matmul`)
- `StableHLO` is the entry format into the XLA compiler pipeline. From here it is lowered into `HLO` and pipelined through many XLA
optimization stages, some of which are platform specific.
- After this, you have an optimised `HLO` representation of your JIT'd module. It then passes through the XLA backend, where stages such as code generation, op scheduling, and buffer assignment transform the `HLO` into a sequence of executable actions ("Thunks") for the target device.
- By combining thunks with their respective codegen output, you construct the runnable executable.

This all makes much more sense once you see what `StableHLO` and `HLO` actually look like.

---

# What is HLO and how is it used?

HLO is an intermediate representation. In `.txt` dump form, a single module looks like this:
```
<INSERT HLO FOR MATMUL>
```

Not particularly helpful...

But HLO is a DAG (Directed Acyclic Graph) of operations and can be visualized intuitively as such:

![](img/matmul-hlo.png)

- A module can be considered the top-level unit of HLO.
    - if you `jax.jit(matmul)` then `matmul` is your top level operation and hence it is the module.
    - if you `jax.jit(train_step)` then `train_step` is your module, albeit probably magnitudes larger and more complex.
- It follows that `ROOT` is simply the output point of the module.
- `f32[2048,10]{1,0}` represents tensor metadata:
    - `f32` is the datatype
    - `[2048,10]` is the tensor dimensions
    - `{1,0}` is the memory storage format, here indicating row-major as `1` (the row dimension) is listed first.
- `dot_general.0` is an op, namely a dot-product (matmul) op.

Realistically, an HLO module can contain hundreds or thousands of ops. Ultimately, to create the executable, we would like a scheduled sequence of ops to run, ie. to flatten the DAG representation of HLO into a dependency-aware sequence.

# XLA Components

## HLO Optimisation Passes

<HLO diagram before and after a pass>
<example of lowering ie generic to custom call>
<example of fusion pass>
<further description on: layout assignment>

## XLA Scheduler

The XLA scheduler is what transforms the optimised HLO into a planned sequence of ops. The two primary top-level objectives are:

1. Peak memory usage 
    - Buffer lifetimes
    - Buffer reuse
    - Rematerialization/recomputation
    - Eviction/offloading decisions

2. End-to-end latency
    - Exposing parallelism
    - Overlapping communication and compute
    - Critical-path reduction
    - Data locality
    - Resource utilization

Without going into too much detail, scheduling can be treated as an optimization problem. Latency-oriented scheduling relies heavily on execution-time estimates, while memory-oriented scheduling relies more on buffer sizes and lifetimes. In both cases, a more accurate cost model enables better scheduling decisions and ultimately a better memory-versus-latency tradeoff.

By default, the XLA scheduler optimises for peak memory usage (ie. first make sure the model actually fits and can therefore run!). 
To optimize scheduling for latency, the recommended XLA flags are:

```bash
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_command_buffer=''
--xla_disable_hlo_passes=collective-permute-motion
--xla_gpu_experimental_pipeline_parallelism_opt_level=PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE
```

It is recommended to check this against [XLA's documentation for recommended XLA flags](https://docs.jax.dev/en/latest/gpu_performance_tips.html#pipeline-parallelism-on-gpu).

## Buffer Assignment

<INSERT explanation: motive=reduce mem usage, method=reassign expired variables>

---

# Where does LLVM appear?


- IN XLA CODEGEN (AS OPPOSE TO KERNEL LIBRARY SELECTION) LOWERED TO LLVM FOR CODEGEN

---

# XLA Autotuner and ROCm Backends

- explain autotuner:
a GEMM may have several implementations

a variety of implementations e.g. using different rocblas kernels

each kernel may tile the work different, or use a better MFMA instruction pattern

during compilation time, XLA performs autotuning, which essentially empirically measures performance of candidate kernels and selects the best performing kernel


- explain rocm backends and where they fit in the stack (eg. after hlo optimisation, maybe alongside layout decisions/autotuning?)hipblas, rocblas, triton, xla, rccl... 
- jax-aiter https://github.com/ROCm/jax-aiter as FFI calls to bring ROCm AITER (optimised AMD kernels for AI) to jax

---

# The PJRT

PJRT and backend responsibilities
essentially, the PJRT runtime is an API defined by XLA, that backends need to implement
OPAQUE, VENDOR AGNOSTIC.

contrived example:

- XLA defines the PRJT API
```
// PJRT BUFFER API EXAMPLE (defined by XLA)

class PjRtBuffer {
public:
    virtual ~PjRtBuffer() = default;
};

class PjRtLoadedExecutable {
public:
    virtual ~PjRtLoadedExecutable() = default;

    virtual std::vector<std::unique_ptr<PjRtBuffer>> Execute(
        const std::vector<PjRtBuffer*>& inputs) = 0;
};

class PjRtClient {
public:
    virtual ~PjRtClient() = default;

    virtual std::unique_ptr<PjRtBuffer>
    BufferFromHost(const void* data, size_t bytes) = 0;

    virtual std::unique_ptr<PjRtLoadedExecutable>
    LoadExecutable(const std::string& binary) = 0;
};
```

- ROCm defines its own implementation
```
// Backend (ROCm) PJRT IMPLEMENTATION

class RocmBuffer : public PjRtBuffer {
public:
    void* device_ptr;
    size_t bytes;

    RocmBuffer(void* ptr, size_t size)
        : device_ptr(ptr), bytes(size) {}
};

class RocmExecutable : public PjRtLoadedExecutable {
public:
    hipFunction_t kernel_;
    hipStream_t stream_;

    std::vector<std::unique_ptr<PjRtBuffer>> Execute(
        const std::vector<PjRtBuffer*>& inputs) override {

        // Convert PJRT buffers to HIP buffers
        auto* a =
            static_cast<RocmBuffer*>(inputs[0]);

        auto* b =
            static_cast<RocmBuffer*>(inputs[1]);

        // Launch code-generated kernel
        hipLaunchKernel(
            kernel_,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            0,
            stream_,
            /* args = */ {a->device_ptr, b->device_ptr});

        return {/* output buffers */};
    }
};

class RocmClient : public PjRtClient {
public:
    std::unique_ptr<PjRtBuffer>
    BufferFromHost(const void* data, size_t bytes) override {

        void* dev_ptr;
        hipMalloc(&dev_ptr, bytes);

        hipMemcpy(
            dev_ptr,
            data,
            bytes,
            hipMemcpyHostToDevice);

        return std::make_unique<RocmBuffer>(
            dev_ptr,
            bytes);
    }

    std::unique_ptr<PjRtLoadedExecutable>
    LoadExecutable(const std::string& hsaco_binary) override {

        // Load GPU binary produced by XLA codegen
        hipModule_t module;
        hipModuleLoadData(
            &module,
            hsaco_binary.data());

        hipFunction_t kernel;
        hipModuleGetFunction(
            &kernel,
            module,
            "fused_kernel");

        return std::make_unique<RocmExecutable>(
            kernel);
    }
};
```

- XLA runtime runs the PJRT implementation that is passed to it.
```
// XLA RUNTIME CODE (backend-agnostic)
void RunXlaProgram(
    PjRtClient* client,
    const std::string& compiled_binary) {
    auto executable =
        client->LoadExecutable(compiled_binary);
    auto inputA =
        client->BufferFromHost(a_data, bytes);
    auto inputB =
        client->BufferFromHost(b_data, bytes);
    executable->Execute({
        inputA.get(),
        inputB.get()
    });
}
```

<THIS ALL NEEDS TO BE SHORTENED, ESPECIALLY THE CODE SNIPPETS. REMOVE MOST OF THE NOISE IN THE CODE SNIPPETS, KEEP ONLY THE PARTS THAT DEMONSTRATE THE ESSENSE IE DONT NEED ALL THE CLASS SCAFFOLDING AROUND IT>

---

# XLA Compiler Performance Flags

<debug_options.cc>
<how to use these in actual program (set env variables?) code snippet to show this>
<actual list of flags>
https://rocm.docs.amd.com/projects/cvs/en/latest/reference/configuration-files/jax.html#xla-flags

---

# MY OWN NOTES

```
stage 1: RunHloPasses() from gpu_compiler.cc
vendor specific amdgpu_compiler.cc overrides what exactly? if not RunHloPasses?

it only overrides HOOKS that GpuCompiler calls at specific points

examples:

OptimizeHloConvolutionCanonicalization — how convolutions get canonicalized (MIOpen vs cuDNN expectations differ)
AddPaddingForGpublasGemms — hipBLAS vs cuBLAS padding requirements
OptimizeHloPostLayoutAssignment — adds AMD-specific passes (then calls the base)
CompileTargetBinary — the actual LLVM→GCN/HSACO step (vs NVPTX→PTX)
GetLLVMCommandLineOptions — target-specific LLVM flags
RunHloPasses is the outer wrapper for the passes

within this is OptimizeHloModule which is layout-independent passes (simplification, expanders, SPMD partitioning …)

what does this even mean? what does layout mean here?

LAYOUT=PHYSICAL MEMORY ARRANGEMENT OF TENSOR DIMENSIONS

row vs column major (but applied to 3, 4, Nth dimensions

a shape [batch, height, width, channel] holds nothing about stride order in memory but the layout does

therefore layout-independent passes means passes that are valid regardless of memory order i.e. algebraic simplification (x*1 → x), constant folding, decomposing high-level ops into primitives ("expanders"), SPMD partitioning. these do not care how bytes are laid out.

versus layout-DEPENDENT passes e.g. fusion needs to know access patterns which are only known after a layout is pinned per tensor

OptimizeHloPostLayoutAssignment for all passes after layouts are pinned

fusion, GEMM rewriting, vendor library matching, triton fusion

e.g. fusion_pipeline.cc (resides in xla/service/gpu)

“the passes are grouped into named HloPassPipeline’s” what does this mean? what does a hlopasspipeline object represent?

literally the pipeline to pass unoptimised HLO through

i.e. HLO module → (hlo pass pipeline) → optimised HLO

an ordered list of passes and a runner

i.e. pipeline.Run(module)

which makes runhlopasses a wrapper around the pipeline runner?

just know that RunHloPasses is what drives unoptimized HLO → optimized HLO

stage 2: scheduling and buffer assignment
buffer assignment (xla/service/buffer_assignment.cc) decides which HLO values share memory (what does this mean)

Every HLO instruction produces a value that needs to live somewhere in device memory. Naively you'd give each its own buffer, but that's wildly wasteful. Buffer assignment (buffer_assignment.cc) does liveness analysis: if value A is dead by the time B is computed (their lifetimes don't overlap), B can reuse A's memory.

Result: a memory plan mapping each value to an offset in a set of allocations — this is XLA's answer to a register/stack allocator, but for GPU DRAM.

(basically, analagous to heap allocation in a c/cpp compiler ie managing memory regions)

runs AFTER scheduling as lifetimes depend on execution order

scheduling for determining execution order and async overlap (collectives, compute)

RunPostSchedulingPipelines what is this?

same pipeline idea as above: just the pipeline for passes that only run after scheduling is determined

for example passes involving collectives

stage 3: RunBackend() (codegen)
at a high level, mapping the ops → vendor library/manual codegen/triton

EMITTERS → LLVM IR, which compiles to target platform via LLVM backend compiler

what is meant by emitters?

like type HloOp.emitIR() codegen

kinda. more precisely, an object that consumes an HLO Fusion → emit IR

e.g. SomeEmitter(HloFusion): HloFusion → IR

MORE PRECISELY: HloFusion → [Emitter] → MLIR → [MLIR lowering passes] → LLVM IR → PTX/GCN

other: StreamExecutor and runtime
xla/stream_executor/ for hardware abstraction (device memory, kernel launch, …)

basically the logistics associated with handling a stream

MORE PRECISELY:

a StreamExecutor is an abstraction layer for a single device ie. one StreamExecutor=1GPU

Look at what its methods are (:93-363). they're pure device-driver primitives:

Allocate / HostMemoryAllocate — device DRAM allocation (this is what buffer assignment's plan ultimately calls into)

CreateStream, CreateEvent — make execution queues and sync points

LoadModule / LoadKernel — load compiled PTX/GCN, get a launchable Kernel handle

SynchronousMemcpy, SynchronizeAllActivity — host↔device transfer, sync

CreateCommandBuffer — CUDA graph / equivalent

xla/backend/gpu/runtime/ for thunks. the executable is ultimately a sequence of THUNKS

what is a THUNK??? what would the equivalent be in a c++ stack?

“one runtime action”. 

thunk: “launch kernel K”

kernel: the actual kernel

btw the k in eg. kKernel just indicates a constant (google programming convention

the executable itself is a sequence of thunks ie. vector<Thunk>

CORE: for Thunk in ThunkSequence: Thunk.ExecuteOnStream(params)



ThunkSequence            Stream
(vector<Thunk>)          (ordered queue of driver ops)
─────────────            ──────────────────────────────
kGemm       ──executes──▶  [cuBLAS internally enqueues N kernel launches]
kCopy       ──executes──▶  [Memcpy]
kKernel     ──executes──▶  [LaunchKernel]
kAllReduce  ──executes──▶  [NCCL enqueues kernels + comms]
the ThunkSequence is the software/program perspectiveof the executable

the Stream is the closer-to-hardware perspective of actual ops that run



// setup phase (once)
stream = StreamExecutor->CreateStream()
// execution phase (per run)
executable = ThunkSequence
for thunk in ThunkSequence:
    thunk.ExecuteOnStream(params) // params.stream = the stream from setup
        └─ params.stream->LaunchKernel(...)  // enqueue onto that queue
        └─ params.stream->Memcpy(...)
```