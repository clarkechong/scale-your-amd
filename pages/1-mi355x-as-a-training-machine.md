---
layout: distill
title: "MI355X as a Training Machine"
description: "The gfx950 execution, memory, precision, and interconnect limits that govern JAX training on one MI355X or one eight-GPU node."
date: 2026-09-13

section_number: 1

previous_section_url: "/"
previous_section_name: "Chapter 0: Intro"

next_section_url: "/pages/2-what-jax-jit-runs-on-rocm"
next_section_name: "Chapter 2: What jax.jit Runs on ROCm"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "How to read this chapter"
  - name: "Device identity and JAX visibility"
  - name: "From chiplets to wavefronts"
  - name: "Matrix execution"
    subsections:
      - name: "From lane fragments to macrotiles"
      - name: "Shape tails"
      - name: "A useful TPU contrast"
  - name: "The rest of a training kernel"
  - name: "Memory hierarchy"
    subsections:
      - name: "Cache scope and chiplet locality"
      - name: "Direct global to LDS loads"
  - name: "Resource pressure and occupancy"
    subsections:
      - name: "Worked occupancy example"
      - name: "Spills and bank conflicts"
  - name: "Native training formats"
    subsections:
      - name: "OCP FP8"
      - name: "Microscaling formats"
      - name: "Dense and sparse peaks"
  - name: "Partition modes"
  - name: "Capacity bandwidth and the BF16 roofline"
  - name: "Eight GPU scale up"
    subsections:
      - name: "Directional bandwidth"
      - name: "Partial participation"
      - name: "No transparent pooled JAX memory"
  - name: "Scale out preview"
  - name: "Hardware constants sheet"
  - name: "Where these constants reappear"
  - name: "References"
---
This chapter supplies the hardware constants used by the rest of the book. The
scope is one AMD Instinct MI355X OAM, its `gfx950` execution target, and the
eight-OAM UBB 2.0 scale-up domain. Other accelerators appear only when a contrast
changes how a JAX training program should be reasoned about.

## How to read this chapter

Facts copied from a linked specification or architecture document are marked
**[cited]**. Arithmetic derived from those facts is marked **[analytical]**.
There are no **[measured]** claims in this chapter. In particular, the existing
benchmark artifacts in this project identify MI300X `gfx942` devices, so they
cannot support MI355X performance claims.

The peak rates below are ceilings, not promised sustained rates. They assume the
published 2.4 GHz peak engine clock, a supported dense Matrix Core instruction,
enough independent work, and no time lost to memory, communication, launch
overhead, or non-matrix operations. Later chapters compare traces against these
ceilings.

> Verified against the linked AMD, ROCm, JAX, and OCP documentation on
> **13 September 2026**. Capacities and rates use the decimal labels in AMD's
> product material unless a binary unit such as KiB is written explicitly.

## Device identity and JAX visibility

**[cited]** MI355X is an OCP Accelerator Module based on AMD CDNA 4. Its LLVM
target is `gfx950`. The package contains two I/O dies (IODs), eight Accelerator
Complex Dies (XCDs), eight HBM3E stacks, and 256 active Compute Units (CUs). It
has 288 GB of HBM3E and a 2.4 GHz peak engine clock. Eight OAMs fit on the AMD
Universal Base Board 2.0, or UBB 2.0. These identities come from the
[MI355X GPU product brief](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf)
and the
[CDNA 4 architecture white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf).

In the full-chip SPX partition used throughout this book, one physical OAM is one
logical GPU. JAX normally prints it as a device such as `rocm:0`; XLA compiles
GPU code for `gfx950`. An eight-OAM node in SPX mode therefore contributes eight
logical JAX devices, subject to process and container visibility. The name
`rocm:0` describes a runtime device, not an XCD, a CU, or the whole UBB.

The host CPU still launches work and manages the process. Each OAM has one PCIe
Gen 5 x16 connection for host or I/O traffic. Device-to-device traffic within
the UBB uses Infinity Fabric links instead. The distinction matters because
host staging, xGMI peer traffic, and HBM access have different limits.

{% comment %}
> **Figure 1 placeholder.** *Caption: One MI355X OAM as JAX sees it in SPX mode.
> Show two IODs below eight XCDs, eight 36 GB HBM3E stacks around the IODs,
> 256 MB Infinity Cache on the IODs, and the external interfaces: seven xGMI
> peer links plus one PCIe Gen 5 x16 link. Label the whole OAM `rocm:n`; do not
> label an XCD as a JAX device.*
{% endcomment %}

## From chiplets to wavefronts

**[cited]** The execution hierarchy is:

```text
MI355X OAM: one physical GPU
├── 2 IODs: HBM controllers, Infinity Cache, PCIe, and external Infinity Fabric
├── 8 XCDs: 36 physical CUs each, 32 active
│   ├── 4 MB L2 shared by the active CUs on that XCD
│   └── 32 active CUs
│       ├── 4 SIMD vector units
│       ├── 4 Matrix Cores
│       ├── scalar and memory pipelines
│       ├── 32 KiB L1 vector cache
│       └── 160 KiB LDS
└── 8 HBM3E stacks: 288 GB total
```

Across the package, `8 XCDs × 32 active CUs/XCD = 256 CUs`. There are four SIMDs
and four Matrix Cores per CU, giving 1,024 of each across the OAM. An XCD and a
Shader Engine are not synonyms. The XCD is a compute chiplet containing CUs,
cache, and scheduling resources.

**[cited]** A CU has 64 stream processors arranged as four SIMD16 vector
pipelines. A Wave64 contains 64 logical work-items and is issued through a SIMD16
pipeline as four 16-lane quarter-waves. MFMA is still a wave-level operation:
all 64 logical lanes contribute register fragments to one matrix instruction.
A HIP workgroup or an XLA-generated GPU block can contain several waves. For
example, a 256-thread workgroup contains four waves. The hardware can hold at
most eight resident waves per SIMD, or 32 per CU, if registers, LDS, and other
resources permit.

For readers coming from CUDA, the small translation table is:

| NVIDIA term | AMD CDNA 4 term | MI355X fact |
|---|---|---|
| Streaming Multiprocessor | Compute Unit | 256 per OAM |
| warp | wavefront | 64 threads, not 32 |
| CUDA core | SIMD execution lane | 16 physical lanes per SIMD; 64 stream processors per CU |
| Tensor Core | Matrix Core using MFMA | four per CU |
| thread block | workgroup | one or more waves |
| shared memory | LDS | 160 KiB per CU |
| local memory spill | scratch | backed outside the register file |
| NVLink | xGMI over Infinity Fabric | seven direct peer links |

The mapping is for vocabulary, not performance. Scheduling rules, register
allocation, instruction shapes, and topology still differ.

## Matrix execution

Transformer training spends most of its FLOPs in matrix multiplications. On
CDNA 4, those products should lower to Matrix Fused Multiply-Add (MFMA)
instructions. An MFMA is a wave-level operation: all 64 lanes provide fragments
of the input matrices and receive fragments of the output accumulator.

For a tile \(D=A B+C\), the logical FLOP count is

$$
F_{\mathrm{tile}} = 2mnk,
$$

because every multiply-accumulate is counted as two floating-point operations.
The instruction name states its logical shape. CDNA 4 adds BF16 and FP16 forms
with output tiles of \(16\times16\) and \(32\times32\), including
`16x16x32` and `32x32x16` forms. Its low-precision scaled family includes
`v_mfma_scale_f32_16x16x128_f8f6f4` and
`v_mfma_scale_f32_32x32x64_f8f6f4`. The exact operand layout is defined by the
[CDNA 4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf).

The `f32` in those names describes the accumulator. Low-precision inputs do not
force low-precision accumulation. A normal training GEMM can multiply BF16,
FP16, FP8, or MX values while keeping partial sums in FP32 registers, then
convert the stored output to the requested dtype. The later precision
experiments must identify all three roles: input format, accumulator format,
and output format.

### From lane fragments to macrotiles

**[cited]** AMD's
[CDNA 4 FP8 GEMM guide](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)
shows the lane mapping for a `16x16x128` FP8 instruction. Each of the 64 lanes
holds 32 FP8 elements from \(A\), 32 from \(B\), and four FP32 accumulator
values. Together the lanes update a \(16\times16\) output tile:

$$
2(16)(16)(128)=65{,}536\ \text{FLOPs per instruction}.
$$

That instruction tile is smaller than the region assigned to a workgroup.
Several waves usually cooperate through LDS to build a *macrotile*. One
published MI355X example assigns eight waves to a
\(256\times256\times128\) workgroup tile. The workgroup loads operand tiles,
each wave updates its output fragments, and the fragments are written back as
one larger result. The example establishes a mapping, not a universal best
tile. hipBLASLt, Triton, AITER, and XLA can choose different shapes.

The important data path is:

```text
HBM or cache → LDS staging tile → VGPR operand fragments
             → MFMA → FP32 accumulator registers → output
```

Reuse at the LDS and register levels is what turns an HBM-limited dot product
into a compute-limited GEMM.

{% comment %}
> **Figure 2 placeholder.** *Caption: One `16x16x128` low-precision MFMA inside
> a larger GEMM macrotile. Show 64 lanes contributing A and B fragments, four
> FP32 outputs per lane, four or eight waves sharing LDS, and repeated K tiles
> accumulating into register-resident C fragments. Distinguish instruction
> tile, wave tile, and workgroup macrotile.*
{% endcomment %}

### Shape tails

Matrix dimensions rarely arrive as one hardware instruction. A kernel tiles
\(M\), \(N\), and \(K\), then masks, pads, or sends incomplete edge tiles to a
cleanup path. For tile sizes \(T_M,T_N,T_K\), a simple padded-work estimate is

$$
F_{\mathrm{padded}} =
2\left\lceil\frac{M}{T_M}\right\rceil T_M
 \left\lceil\frac{N}{T_N}\right\rceil T_N
 \left\lceil\frac{K}{T_K}\right\rceil T_K.
$$

**[analytical]** If only \(M\) has a tail, \(M=257\), and the macrotile step is
32 rows, padding to 288 rows adds

$$
\frac{288}{257}-1=12.1\%
$$

to this upper estimate. A predicated kernel may avoid some arithmetic while
still paying for inactive lanes and less efficient memory transactions. This is
why batch, sequence, hidden, and expert dimensions that look almost identical
at the model level can select different kernels or show different utilization.
No one tile divisibility rule covers every backend; inspect the selected kernel
and profile the actual shape.

### A useful TPU contrast

A TPU MXU is a comparatively large systolic array. Values flow through its
two-dimensional multiply-accumulate grid. MI355X distributes matrix execution
across 1,024 Matrix Cores in 256 CUs, and each MFMA starts with wave-lane
fragments. The JAX source can contain the same `dot_general`, but the utilization
problem differs:

- On a TPU, array dimensions and MXU tiling determine how well the systolic
  array is filled.
- On MI355X, MFMA shape, wave and workgroup tiles, register allocation, LDS
  staging, and the number of CUs receiving work all matter.

The [JAX Scaling Book TPU chapter](https://jax-ml.github.io/scaling-book/tpus/)
explains the systolic side. This companion uses the same roofline method but
substitutes the MI355X execution and memory limits.

## The rest of a training kernel

Matrix Cores do not execute the whole step. CDNA 4 CUs also contain vector,
scalar, memory, and control pipelines:

- SIMD vector instructions handle elementwise activations, optimizer updates,
  dtype conversion, normalization arithmetic, masking, and address-dependent
  work.
- Scalar instructions handle values uniform across a wave, including loop
  bounds, base addresses, and control decisions.
- Reductions combine lane values through shuffle or LDS stages. Larger
  reductions may require several workgroups or a device collective.
- Memory instructions move values among HBM, caches, LDS, and registers.

These paths explain why a twofold matrix peak does not imply a twofold training
step speedup. If a fraction \(p\) of step time improves by a factor \(s\), the
largest end-to-end speedup with the remainder unchanged is

$$
S_{\mathrm{step}}=\frac{1}{(1-p)+p/s}.
$$

**[analytical]** If GEMMs account for 85% of a hypothetical step and their time
halves, then

$$
S_{\mathrm{step}}=\frac{1}{0.15+0.85/2}=1.74,
$$

not 2. This is an illustration, not a measurement. The real \(p\) comes from a
profile. Attention softmax, normalization, optimizer work, routing, launch
gaps, and collectives determine the remaining fraction.

## Memory hierarchy

The useful memory hierarchy is organized by both speed and sharing scope.

| Level | Published MI355X capacity | Scope | Training role |
|---|---:|---|---|
| VGPR and AccVGPR | 512 32-bit entries per lane | one SIMD; allocated per wave | operands, accumulators, live values |
| LDS | 160 KiB | one CU; shared by a workgroup | software-managed staging and cross-wave exchange |
| L1 vector cache | 32 KiB | one CU | cached vector and global loads |
| L2 | 4 MiB | one XCD, 32 active CUs | coalesces traffic before Infinity Fabric |
| Infinity Cache | 256 MiB | one OAM, shared across eight XCDs | memory-side last-level cache |
| HBM3E | 288 GB at 8 TB/s peak | one OAM | parameters, optimizer state, activations, and workspaces |

**[cited]** ROCm's
[GPU specification table](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
also lists the 32 KiB vector L1, 4 MiB of L2 per XCD, 256 MiB last-level cache,
and 512 KiB total VGPR storage per CU. The table above expresses the register
budget in the per-lane unit needed for occupancy calculations.

### Cache scope and chiplet locality

The cache labels alone are not enough. A value reused by waves on one CU can
remain in its L1. Work spread across CUs on one XCD can share that XCD's L2.
Traffic crossing an XCD boundary leaves that L2 and enters the on-package
Infinity Fabric toward the IODs, Infinity Cache, or HBM controllers.

In SPX mode, workgroups are distributed across XCDs; ordinary JAX code does not
pin an HLO operation to a chosen XCD. Library kernels may use XCD-aware tile
ordering, but a model author should not assume that two successive workgroups
share an L2 slice. Cache reuse is a kernel and schedule property that must be
checked with counters.

The 256 MiB Infinity Cache is shared at package scope, but it does not turn HBM
into an on-chip scratchpad. A 70-billion-parameter model occupies about 140 GB
in BF16, roughly 530 times the cache capacity. Training kernels still depend on
tiling, prefetch, and high HBM bandwidth.

### Direct global to LDS loads

**[cited]** CDNA 4 can move data from the global-memory path directly into LDS
without first staging the payload in VGPRs. AMD documents this through the
`llvm.amdgcn.raw.buffer.load.lds` intrinsic and the wider CDNA 4
`GLOBAL_LOAD_LDS` path. The destination is still LDS; waves later read the
fragments into registers before MFMA.

Direct-to-LDS can reduce temporary VGPR use and remove explicit LDS stores. It
does not remove synchronization, guarantee a conflict-free LDS layout, or make
the data immediately available to MFMA. A tuned pipeline overlaps the load of a
future K tile with MFMA on the current tile, then waits at the point where the
new tile is consumed.

{% comment %}
> **Figure 3 placeholder.** *Caption: MI355X memory scopes and the two global
> load paths. The conventional path is HBM/cache to VGPR to LDS; the CDNA 4
> direct path is HBM/cache to LDS, followed by LDS to VGPR before MFMA. Mark L1
> as per CU, L2 as per XCD, Infinity Cache as per OAM, and HBM as attached to
> one OAM.*
{% endcomment %}

## Resource pressure and occupancy

A kernel can be ready to execute but unable to place another wave on a SIMD
because one resource is exhausted. The main limits are:

- regular VGPR and accumulator-register use per wave;
- scalar-register use per wave;
- LDS bytes per workgroup;
- workgroup and barrier slots.

**[cited]** On gfx950, regular VGPRs and AccVGPRs share one
512-entry-per-lane budget. Each type can use at most 256 entries per wave, with a
flexible split, but they are not two independent 512-entry pools. The register
limit in waves per SIMD is approximately

$$
W_{\mathrm{VGPR}} =
\left\lfloor
\frac{512}{R_{\mathrm{VGPR}}+R_{\mathrm{AccVGPR}}}
\right\rfloor,
$$

after allocation-granularity rounding. LDS is allocated per workgroup, so its
first result is workgroups per CU:

$$
G_{\mathrm{LDS}} =
\left\lfloor
\frac{160\ \mathrm{KiB}}{L_{\mathrm{workgroup}}}
\right\rfloor.
$$

Convert all limits to waves per SIMD, take the minimum, and clamp at eight.
Occupancy is that result divided by eight. The
[MI355X occupancy guide](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html)
derives the conversion for different workgroup sizes.

More occupancy can hide latency because the scheduler can issue another ready
wave. More instruction-level parallelism (ILP) can hide the same latency inside
one wave by keeping independent loads or accumulators in flight. Larger tiles
usually increase ILP and data reuse but consume more registers. The target is
enough parallel work to feed the limiting pipeline, not the highest possible
occupancy percentage.

### Worked occupancy example

**[cited]** Consider the example from AMD's occupancy guide: a 256-thread,
four-wave MXFP8 workgroup uses 128 total VGPR entries per lane, 50 SGPRs per
wave, and 32 KiB of LDS.

The register ceiling is

$$
W_{\mathrm{VGPR}}=\left\lfloor\frac{512}{128}\right\rfloor=4
\quad\text{waves per SIMD}.
$$

The LDS holds

$$
G_{\mathrm{LDS}}=\left\lfloor\frac{160}{32}\right\rfloor=5
\quad\text{workgroups per CU}.
$$

Each four-wave workgroup places one wave on each of the four SIMDs, so the LDS
limit converts to five waves per SIMD. SGPRs and workgroup slots do not bind in
the published example. The register limit wins:

$$
\mathrm{occupancy}=\frac{\min(4,5,8)}{8}=50\%.
$$

That 50% is a resource ceiling, not a prediction of throughput. A matrix kernel
with enough ILP may keep the Matrix Cores busy at lower occupancy; a
memory-latency-bound kernel may need more resident waves.

### Spills and bank conflicts

If live values exceed the register allocation, the compiler can spill them to
scratch memory. Scratch has a private-address-space programming model, but it is
not a hidden extension of the register file. Spill loads and stores enter the
memory hierarchy and can add dependencies in a hot loop. Kernel names,
compiler resource reports, ISA metadata, and profiler counters are the evidence
for a spill diagnosis.

LDS pressure has two forms. Capacity pressure reduces resident workgroups.
Access pressure occurs when several lanes address the same bank in one phase.
**[cited]** CDNA 4 LDS has 64 banks and up to 256 bytes per clock of read
bandwidth. A poor lane layout serializes bank-conflicting accesses even when
capacity and occupancy look healthy. Padding or swizzling the LDS tile can
change the bank mapping, but the benefit is shape- and schedule-dependent.

## Native training formats

The hardware supports more formats than a training recipe should use. A format
is useful only when the JAX-to-kernel path selects the corresponding instruction
and the training run passes its numerical checks. Hardware support is necessary,
not sufficient.

BF16 is the baseline format for this book. It keeps FP32's eight-bit exponent
with fewer significand bits, which makes it easier to use for training than
FP16's narrower exponent range. BF16 and FP16 have the same published MI355X
matrix peak. Both commonly accumulate into FP32.

FP32 remains important for accumulation and sensitive state even though its
matrix peak is much lower. Optimizer moments, reductions, logits, and selected
normalization operations may stay in FP32 while the large GEMMs use a compact
input format.

### OCP FP8

**[cited]** gfx950 uses the OCP FP8 encodings:

- E4M3 has one sign bit, four exponent bits, and three mantissa bits.
- E5M2 has one sign bit, five exponent bits, and two mantissa bits.

E4M3 trades exponent range for precision; E5M2 trades precision for exponent
range. They are not bit-compatible with the FNUZ FP8 variants used by gfx942.
Checkpoint and scale metadata must therefore identify the encoding, not merely
say `fp8`. The
[OCP FP8 specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf)
defines the encodings.

FP8 training also needs a scaling policy. The scale may be chosen per tensor or
per channel and updated from current or delayed amax statistics. Those are
software and numerical decisions covered in the precision chapter; the hardware
fact here is that MI355X can execute OCP FP8 matrix instructions at twice its
BF16 matrix rate.

### Microscaling formats

**[cited]** CDNA 4 adds native MXFP8, MXFP6, and MXFP4 matrix instructions. The
[OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
groups 32 values under one E8M0 scale. E8M0 stores a power-of-two scale as an
eight-bit biased exponent. The element encodings are:

| Format | Element encoding | Scale block | Effective storage including one scale |
|---|---|---:|---:|
| MXFP8 | E4M3 or E5M2 | 32 values | \(8+8/32=8.25\) bits/value |
| MXFP6 | E3M2 or E2M3 | 32 values | \(6+8/32=6.25\) bits/value |
| MXFP4 | E2M1 | 32 values | \(4+8/32=4.25\) bits/value |

The effective-storage column is **[analytical]** and excludes tensor padding,
alignment, and auxiliary metadata. A block whose length is not a multiple of 32
needs padding or a tail representation. The shared scale preserves more local
range than one scale for a whole tensor, but it does not restore the precision
discarded by a four- or six-bit element.

Scaled MFMA instructions consume element blocks and their scale information as
part of the matrix operation. Whether JAX reaches those instructions through
Transformer Engine, AITER, Triton, or another custom call is a software-stack
question. Chapter 2 shows how to prove the route rather than infer it from a
configuration name.

### Dense and sparse peaks

The table uses the 2.4 GHz MI355X figures. A sparse peak requires the
hardware-supported structured pattern and a kernel that uses the sparse path.
Ordinary dense transformer weights use the dense column.

| Matrix input format | FLOPs per clock per CU | Dense peak per OAM | Published structured-sparse peak |
|---|---:|---:|---:|
| FP32 | 256 | 157.3 TFLOP/s | not listed |
| FP16 | 4,096 | 2.5166 PFLOP/s | 5.0332 PFLOP/s |
| BF16 | 4,096 | **2.5166 PFLOP/s** | 5.0332 PFLOP/s |
| OCP FP8 | 8,192 | 5.0332 PFLOP/s | 10.0663 PFLOP/s |
| MXFP8 | 8,192 | 5.0332 PFLOP/s | not listed in the product table |
| MXFP6 | 16,384 | 10.0663 PFLOP/s | not listed in the product table |
| MXFP4 | 16,384 | 10.0663 PFLOP/s | not listed in the product table |

**[analytical]** The dense BF16 ceiling can be rebuilt from architectural
factors:

$$
\begin{aligned}
C_{\mathrm{BF16}}
&=256\ \mathrm{CUs}
  \times4096\ \frac{\mathrm{FLOPs}}{\mathrm{clock\cdot CU}}
  \times2.4\times10^9\ \frac{\mathrm{clocks}}{\mathrm{s}}\\
&=2.5165824\times10^{15}\ \mathrm{FLOP/s}\\
&\approx\mathbf{2.5166\ PFLOP/s}.
\end{aligned}
$$

The same reconstruction gives 5.0332 PFLOP/s for FP8 and MXFP8, and
10.0663 PFLOP/s for MXFP6 and MXFP4. Do not use a sparse number in MFU unless
the workload and kernel actually use structured sparsity.

## Partition modes

MI355X can split its eight XCDs into logical accelerator partitions. Compute
partitioning and HBM NUMA partitioning are separate controls. AMD's published
MI355X workload table lists:

| Compute mode | Logical GPUs per OAM | XCDs per logical GPU | HBM per logical GPU | Published NPS pairing |
|---|---:|---:|---:|---|
| SPX | 1 | 8 | 288 GB | NPS1 |
| DPX | 2 | 4 | 144 GB | NPS2 |
| QPX | 4 | 2 | 72 GB | NPS2 |
| CPX | 8 | 1 | 36 GB | NPS2 |

**[cited]** The
[ROCm workload guide](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html)
provides these MI355X profiles. Exact support is firmware-dependent, so
`amd-smi partition --accelerator` on the target host is authoritative. The
[AMD SMI partitioning guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/conceptual/partition.html)
explains the logical-device model.

JAX sees the logical GPUs exposed by the driver. Thus one CPX-partitioned OAM
can enumerate as eight devices, while the same OAM in SPX enumerates as one.
The peak compute, memory capacity, and cache scope available to one JAX device
have changed even though `jax.device_count()` increased.

This book assumes **SPX with NPS1** for full-OAM training unless a case study
says otherwise. DPX with NPS2 is AMD's efficiency recommendation for partitioned
MI355X workloads, but it is not a general recommendation to split a large JAX
training job. Always record partition mode with a benchmark. A clean twofold,
fourfold, or eightfold error in a roofline often means the calculation assumed
SPX while the process saw a partition.

## Capacity bandwidth and the BF16 roofline

**[cited]** One MI355X has 288 GB of HBM3E and 8 TB/s peak HBM bandwidth.
Capacity answers whether the local shard, activations, compiler temporaries, and
workspaces fit. Bandwidth limits kernels that do too little arithmetic per byte
fetched from HBM.

For peak compute \(C\), bandwidth \(\beta_{\mathrm{HBM}}\), and arithmetic
intensity \(I=F/Q\), the ideal roofline is

$$
P\leq\min\left(C,\ I\beta_{\mathrm{HBM}}\right),
$$

or, as a time lower bound,

$$
t\geq\max\left(\frac{F}{C},\frac{Q}{\beta_{\mathrm{HBM}}}\right).
$$

**[analytical]** The dense BF16 machine balance is

$$
I^*_{\mathrm{BF16}}
=\frac{2.5165824\times10^{15}}{8.0\times10^{12}}
=314.5728\ \mathrm{FLOP/byte}
\approx\mathbf{315\ FLOP/byte}.
$$

A BF16 matrix operation below roughly 315 FLOP/byte cannot reach the dense BF16
compute ceiling even with a perfect memory implementation. The ridge is not a
cliff, and real kernels lose bandwidth and compute efficiency on both sides.

Chapter 3 applies this ratio to concrete training projections and derives the
corresponding token-row threshold. This chapter only supplies the hardware side
of that calculation.

## Eight GPU scale up

**[cited]** The MI355X UBB 2.0 places eight OAMs in a one-hop, fully connected
mesh. Every GPU has one dedicated xGMI link to each of its seven peers. There is
no scale-up switch between them. The ninth GPU is outside this complete graph
and must be reached through a scale-out network.

The topology is valuable for tensor, expert, and fully sharded parallelism
because every pair can communicate directly. It also sets a hard placement
boundary: a parallelism axis of size eight can remain inside xGMI; a larger
axis crosses NICs.

{% comment %}
> **Figure 4 placeholder.** *Caption: The eight-GPU MI355X UBB 2.0 as a complete
> graph. Draw eight OAM vertices, seven direct xGMI edges incident on each
> vertex, and one PCIe Gen 5 x16 I/O edge leaving each OAM. Annotate one xGMI
> edge as 76.8 GB/s per direction and 153.6 GB/s bidirectional. Annotate the
> per-GPU directional sum as 537.6 GB/s, not 1,075.2 GB/s.*
{% endcomment %}

### Directional bandwidth

Each xGMI link is 16 lanes at 38.4 Gb/s per lane:

$$
\beta_{\mathrm{xGMI,one\ direction}}
=\frac{16\times38.4\ \mathrm{Gb/s}}{8}
=\mathbf{76.8\ GB/s}.
$$

Because transmit and receive directions are independent, AMD also publishes
153.6 GB/s *bidirectional* per link. Summing seven links gives:

$$
\beta_{\mathrm{egress}}=7(76.8)=537.6\ \mathrm{GB/s},
$$

and 1,075.2 GB/s only when transmit and receive are added together. A cost model
for bytes sent in one direction must use 76.8 GB/s per peer link or
537.6 GB/s aggregate egress, not the doubled marketing total.

**[analytical]** Sending 1 GB to one peer has an ideal serialization lower bound
of

$$
\frac{1\ \mathrm{GB}}{76.8\ \mathrm{GB/s}}=13.0\ \mathrm{ms}.
$$

This excludes protocol overhead, synchronization, software latency, and any
collective algorithm. It is not a measured RCCL time.

### Partial participation

The 537.6 GB/s aggregate assumes traffic can use all seven peer links at once.
A two-GPU operation can use only the one physical link between those GPUs. A
four-GPU subgroup has only three participating peer links per GPU. The unused
links do not combine into a faster link to one destination.

**[analytical]** If one GPU sends 1 GB concurrently to each of seven peers, the
ideal time is still

$$
\frac{7\ \mathrm{GB}}{7(76.8)\ \mathrm{GB/s}}=13.0\ \mathrm{ms},
$$

because the seven pairwise links operate in parallel. That equality describes
the physical complete graph. RCCL may choose a ring, tree, or specialized
algorithm with different traffic and synchronization, so later chapters
measure the collective rather than treating this lower bound as achieved
bandwidth.

### No transparent pooled JAX memory

AMD describes the UBB's 2.304 TB of aggregate HBM as coherent shared memory.
That hardware capability supports peer access and coherence. It does **not**
give JAX one transparent 2.304 TB allocator.

In SPX mode JAX sees eight devices, each with 288 GB attached. A `jax.Array`
spanning them has a `Sharding` that maps slices or replicas to the physical
memories of those devices. A replicated 100 GB array consumes about 100 GB on
each participating GPU; a one-dimensionally sharded 100 GB array consumes about
12.5 GB per GPU before overhead. The
[JAX distributed-array documentation](https://docs.jax.dev/en/latest/parallel.html)
defines this per-device layout.

A single unsharded allocation on `rocm:0` cannot silently overflow into the HBM
of `rocm:1`. Peer accessibility also does not let one JAX device execute an
unsharded 400 GB model. Parameters, optimizer state, activations, workspaces,
and temporary buffers must fit each device according to their explicit or
compiler-chosen shardings.

## Scale out preview

An MI355X OAM exposes PCIe Gen 5 x16, with a published 128 GB/s bidirectional
rate, or 64 GB/s per direction before protocol overhead. A server can connect
this I/O path to the host and to RDMA-capable NICs. The NIC is a system
component, not part of the MI355X OAM.

**[cited]** ROCm exposes PeerDirect interfaces that let an RDMA NIC read and
write GPU memory without copying the payload through host memory. The
[ROCm GPU-enabled MPI guide](https://rocm.docs.amd.com/en/develop/how-to/gpu-enabled-mpi.html)
describes this mechanism. GPU-direct RDMA removes a host-memory staging copy;
it does not remove the PCIe and network transfers.

AMD's
[MI3XX cluster reference design](https://instinct.docs.amd.com/projects/MI3XX-reference/latest/)
lists several NIC and switch choices. One option is a 400 Gb/s AMD Pensando
Pollara 400 NIC per GPU in a RoCEv2 rail-optimized fabric. The 400 Gb/s line
rate is 50 GB/s per direction before overhead:

$$
\frac{400\ \mathrm{Gb/s}}{8}=50\ \mathrm{GB/s}.
$$

This is a **reference design**, not an intrinsic MI355X topology. Deployed
servers may use Pollara, Broadcom, NVIDIA, or other supported RDMA adapters;
they may use a fat tree, rail, or hybrid network; and subscription ratios vary.
Do not write "each MI355X has a Pollara NIC."

In a rail layout, GPU index \(i\) on each node is paired with NIC index \(i\),
and equal-index NICs share a low-hop network rail. Traffic that changes rail
must traverse a spine or first move over local xGMI to the correctly placed
GPU. Physical mesh ordering, process ranks, and JAX mesh axes must agree before
rail placement can help.

{% comment %}
> **Figure 5 placeholder.** *Caption: A non-normative two-node rail example.
> Each node has eight MI355X GPUs in an xGMI full mesh and eight external 400G
> NICs connected one-to-one by PCIe. GPU 3 on both nodes maps to rail 3. Show a
> same-rail path through one leaf and a cross-rail path through a spine or local
> xGMI hop. Label Pollara 400 as one reference-design NIC option, not a device
> integrated into MI355X.*
{% endcomment %}

There are no multi-node MI355X training measurements in this book yet.
Multi-node bandwidth, overlap, and scaling claims must remain **[analytical]**
until a run records the server PCIe topology, NIC model and firmware, link
state, switch fabric, rail map, RCCL settings, and message-size curve.

## Hardware constants sheet

Use this sheet for the analytical models in later chapters. Replace a peak with
a measured sustained value only when the measurement protocol and workload are
named.

| Quantity | MI355X value | Scope or qualifier |
|---|---:|---|
| Architecture target | `gfx950` | CDNA 4 |
| Form factor | OAM | UBB 2.0 platform |
| IODs | 2 | per OAM |
| XCDs | 8 | per OAM |
| Active CUs | 256 | 32 per XCD |
| SIMDs and Matrix Cores | 1,024 and 1,024 | four of each per CU |
| Wavefront width | 64 threads | wave64 |
| Maximum resident waves | 8 per SIMD | 32 per CU before resource limits |
| Peak engine clock | 2.4 GHz | ceiling, not guaranteed sustained clock |
| VGPR plus AccVGPR budget | 512 entries per lane | one shared budget per SIMD |
| LDS | 160 KiB per CU | 64 banks; 256 B/clock read peak |
| L1 vector cache | 32 KiB per CU | 128-byte lines |
| L2 | 4 MiB per XCD | 32 MiB total across eight separate slices |
| Infinity Cache | 256 MiB per OAM | shared last-level cache |
| HBM3E capacity | 288 GB per OAM | 2.304 TB across eight OAMs, not one JAX allocator |
| HBM3E bandwidth | 8 TB/s per OAM | peak theoretical |
| Dense BF16 and FP16 | 2.5166 PFLOP/s | peak theoretical |
| Dense OCP FP8 and MXFP8 | 5.0332 PFLOP/s | peak theoretical |
| Dense MXFP6 and MXFP4 | 10.0663 PFLOP/s | peak theoretical |
| BF16 machine balance | about 315 FLOP/byte | dense peak divided by HBM peak |
| Direct xGMI peers | 7 | one-hop complete graph of eight GPUs |
| xGMI per peer | 76.8 GB/s per direction | 153.6 GB/s bidirectional |
| xGMI aggregate | 537.6 GB/s per direction | all seven links active |
| PCIe | Gen 5 x16 | 64 GB/s per direction, 128 GB/s bidirectional |
| Scale-up limit | 8 GPUs | larger groups use scale-out networking |
| MI355X maximum power | 1,400 W | system cooling and power must support it |

## Where these constants reappear

- [Chapter 2]({{ '/pages/2-what-jax-jit-runs-on-rocm' | relative_url }}) follows a `jax.jit`
  computation from StableHLO to `gfx950` code and identifies whether GEMMs reach
  MFMA library kernels.
- [Chapter 3]({{ '/pages/3-predicting-one-training-step' | relative_url }}) uses 288 GB, 8 TB/s,
  2.5166 PFLOP/s, and 76.8 GB/s per direction to derive memory, compute, and
  communication bounds for training parallelism.
- [Chapter 4]({{ '/pages/4-measuring-and-explaining-a-training-step' | relative_url }}) checks those analytical
  bounds against clocks, kernels, counters, and collective traces.
- The [Llama 7B]({{ '/pages/11-llama-7b-exposing-the-complete-stack' | relative_url }}) case separates raw JAX
  execution from optimized attention routes.
- The [Llama 70B]({{ '/pages/12-llama-2-70b-mixed-precision-training' | relative_url }}) case tests BF16,
  FP8, MXFP8, and MXFP4 while treating convergence as a guardrail.
- The [Mixtral 8x22B]({{ '/pages/13-mixtral-8x22b-sharding-meshes-and-moe-optimizations' | relative_url }}) case places
  expert traffic inside the eight-GPU xGMI domain and defines the measurement
  needed to quantify AllToAll exposure.

## References

Primary hardware and architecture sources:

- [AMD Instinct MI355X GPU product brief](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf).
  Package, XCD and CU counts, caches, HBM, clock, precision peaks, PCIe, and
  xGMI.
- [AMD Instinct MI355X platform brief](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-miI355x-platform-brochure.pdf).
  Eight-OAM UBB 2.0 topology and aggregate physical HBM.
- [AMD CDNA 4 architecture white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf).
  IOD and XCD organization, execution pipelines, cache scopes, partitioning,
  data formats, and link derivation.
- [AMD CDNA 4 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf).
  MFMA forms, operand layouts, registers, and allocation rules.
- [ROCm AMD GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html).
  `gfx950`, wave size, and per-level resource capacities.
- [ROCm MI350 Series architecture page](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html).
  Index to the CDNA 4 ISA, white paper, and performance counters.

Primary software-facing and format sources:

- [Occupancy Math on the AMD MI355X GPU](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html).
  Per-SIMD register budget, per-CU LDS, wave limits, and worked occupancy
  arithmetic.
- [FP8 GEMM Optimization on AMD CDNA 4](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html).
  MFMA lane fragments, direct-to-LDS, LDS banks, and macrotile construction.
- [AMD SMI GPU partitioning](https://rocm.docs.amd.com/projects/amdsmi/en/latest/conceptual/partition.html)
  and
  [ROCm workload optimization](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/workload.html).
  Logical-device enumeration and MI355X SPX, DPX, QPX, CPX, and NPS pairings.
- [OCP 8-bit Floating Point Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf)
  and
  [OCP Microscaling Formats Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
  OCP FP8 encodings, MX block size, element formats, and E8M0 scales.
- [JAX explicit parallelism documentation](https://docs.jax.dev/en/latest/parallel.html).
  `jax.Array` shardings and per-device physical storage.
- [ROCm GPU-enabled MPI guide](https://rocm.docs.amd.com/en/develop/how-to/gpu-enabled-mpi.html).
  PeerDirect and GPU-aware RDMA.
- [AMD Instinct MI3XX cluster reference design](https://instinct.docs.amd.com/projects/MI3XX-reference/latest/).
  Non-normative NIC, RoCEv2, tree, rail, and hybrid scale-out designs.

<h3 markdown=1 class="next-section">Next: [how JAX reaches this hardware]({{ '/pages/2-what-jax-jit-runs-on-rocm' | relative_url }}).</h3>
