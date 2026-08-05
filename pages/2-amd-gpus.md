---
layout: distill
title: "How to Think About AMD GPUs"
description: "How fast should this run on an MI300X, and what in the hardware decides that? Compute units, the memory hierarchy, peak FLOPs by dtype, and the switchless xGMI mesh that eight GPUs sit on. Every constant the rest of the book substitutes into."
date: 2026-08-04

section_number: 2

previous_section_url: "/pages/1-rooflines"
previous_section_name: "Chapter 1: Rooflines"

next_section_url: "/pages/3-profiling"
next_section_name: "Chapter 3: Profiling"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: What an MI300X Is
  - name: The Memory Hierarchy
  - name: Peak FLOPs by Dtype
    subsections:
      - name: Numeric Formats, Once, for the Whole Book
  - name: Inside a Node
  - name: Beyond a Node
  - name: "Worked Example: A 4096 Cubed bf16 Matmul"
  - name: What Changes Across the Family
  - name: "A Translation Table: MI300X, H100 and TPU v5e"
  - name: Worked Problems
  - name: References
---

**Depends on:** [Chapter 1]({{ '/pages/1-rooflines' | relative_url }}) for rooflines,
arithmetic intensity and the ridge point. Nothing later.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

[Chapter 1]({{ '/pages/1-rooflines' | relative_url }}) handed you three bounds with
symbols in them: `C` for the FLOP rate, `β_hbm` for memory bandwidth, `β` for
whatever link the bytes cross. **This chapter fills in the symbols, and that is the
whole job.** By the end you will be able to predict how long a matmul takes on an
MI300X, and you will know which of the numbers you used are trustworthy and which are
marketing.

**The editorial rule for this chapter is that every fact has to feed a later
prediction.** CDNA 3 is an interesting microarchitecture and this is not a tour of
it. If a paragraph cannot answer "which later prediction breaks without this?", it is
not here. That means no die photos, no cache-coherence protocol, and nothing about
video decoders.

Everything in this chapter is **[analytical]**: published specifications, mostly from
AMD's own data sheets, with the arithmetic shown so you can rebuild any number for a
part we do not cover. The exception is a handful of figures read out of our own
traces, which are marked where they appear, and which are here because they disagree
with the specifications in an instructive way.

> **Verified against:** AMD MI300X, MI325X and MI355X data sheets and ROCm hardware
> documentation as of **4 August 2026**. Sources are linked in
> [References](#references). Every number below is the dense figure unless it says
> otherwise.

## What an MI300X Is

**An MI300X is 304 compute units at 2.1 GHz with 192 GB of memory strapped to
them.** If you remember one line from this chapter, that is the one: the two numbers
in it produce the FLOP ceiling, and the third decides what fits.

Unpacking it from the top down, because each level is a unit that shows up somewhere
later:

- **The OAM module** is the thing you buy: one MI300X in an OCP Accelerator Module
  socket, drawing up to 750 W. Eight of them make a node.
- **Eight XCDs** (Accelerator Complex Dies) per module, stacked over four I/O dies.
  Each XCD carries 38 active compute units, so `8 * 38 = 304` in total, and its own
  4 MB slice of L2.
- **304 compute units**, the unit of accounting for the whole book. This is the number
  that multiplies into the FLOP ceiling and the number that changes when you
  partition the GPU.
- **Four SIMDs per CU**, each with a 512-entry-per-lane vector register file. A
  wavefront is **64 threads wide**, which is the first thing to reset if you arrive
  from CUDA, where a warp is 32.
- **Four Matrix Cores per CU**, which execute the MFMA (matrix fused multiply-add)
  instructions. Every FLOP that matters in this book comes out of these, not out of
  the vector units.

**The clock is the part worth being careful about.** AMD quotes 2100 MHz as the peak
boost engine clock, and the peak FLOP tables are computed against it. Our own traces
report `clock_rate` of **2.10 GHz** in the device plane, so the profiler and the data
sheet agree, but note that both are describing the boost ceiling rather than a clock
you are guaranteed to hold. A 750 W part under a sustained bf16 matmul is a
thermally interesting object, and if you compare a measurement against a
boost-clock roofline you should expect to give back a few percent that has nothing to
do with your code. [Appendix B]({{ '/pages/b-appendix-protocol' | relative_url }})
records what we do about that.

**Partitioning changes the CU count a process sees, and it is the one configuration
detail here you can get wrong from software.** An MI300X can be presented as one
logical device with all 304 CUs (SPX mode) or split along XCD boundaries so that each
logical device gets a fraction of them (CPX and friends), with memory divided
separately by NPS mode. All numbers in this book are SPX, one process seeing all 304
CUs, and a partitioned device invalidates every FLOP figure below by exactly the
fraction of CUs it holds. If a measurement comes in at a clean fraction of the
prediction, check the partitioning mode before you check anything else.

## The Memory Hierarchy

**192 GB of HBM3 at 5.3 TB/s is the memory-bound side of every roofline in this
book.** The capacity decides what fits, which turns out to drive more parallelism
decisions than time does, and the bandwidth is the `β_hbm` from
[Chapter 1]({{ '/pages/1-rooflines' | relative_url }}).

From the compute units outward:

| Level | Capacity | Notes |
|---|---|---|
| Vector registers | 512 entries per lane per SIMD | Only level that can feed the Matrix Core at peak |
| LDS (Local Data Share) | 64 KB per CU | Software-managed scratchpad; NVIDIA calls it shared memory |
| L1 | 32 KB per CU | |
| L2 | 4 MB per XCD | Per-die, so eight slices per module |
| Infinity Cache | 256 MB per module | Last level, shared across all eight XCDs |
| HBM3 | 192 GB at 5.3 TB/s | 8192-bit bus |

> **Verified against:** AMD MI300X data sheet and the ROCm MI300 microarchitecture
> page, 4 August 2026.

**The gap that matters is the first one.** Only the register file delivers operands
fast enough to keep the Matrix Cores at peak; LDS is on-chip and still meaningfully
slower. That fact is the reason
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }})'s section on occupancy
exists, because it means a kernel that maximises resident wavefronts by shrinking its
register footprint can be *slower* than one that hogs registers and runs four waves.
For now, file it as: registers are the scarce resource, and everything below HBM is a
detail until you are writing kernels, which this book will mostly talk you out of.

**Here is the first place a specification and our own tooling disagree, and it is
worth walking through rather than quietly correcting.** Our MI300X traces report a
peak HBM bandwidth of **2479.6 GB/s** in the device plane, less than half the
5.3 TB/s on the data sheet. The number is not random. The ROCm profiler computes it
as

```
2 * mem_clock_hz * bus_width_bits / 8
```

and HIP reports a 1.3 GHz memory clock and an 8192-bit bus on this part, which gives
`2 * 1.3e9 * 8192 / 8 = 2662.4 GB/s`. The comment next to that `2` in the collector
source explains itself: it was inherited from the CUDA path, and it reads "Times 2
because HBM is DDR memory; it gets two data bits per each data lane." **But MI300X
HBM3 signals at 5.2 Gbps per pin against that 1.3 GHz reference, which is four
transfers per clock, not two.** With the right multiplier,
`4 * 1.3e9 * 8192 / 8 = 5324.8 GB/s`, and there is the 5.3 TB/s from the data sheet.

Then a second, independent bug stacks on top: 2662.4 GB/s is rendered as 2479.6 in
the UI, because the value is divided by 2<sup>30</sup> while the label says GB/s.
That same unit mismatch is why 192 GiB of HBM shows up in the same traces as
"206.1 GB".

**Carry 5.3 TB/s.** Two bugs, one a factor of 2 and one a factor of 1.074, and
between them they turn a correct spec number into a figure that would have made every
memory-bound prediction in this book look 2x too pessimistic. We found it by doing
exactly what [Chapter 1]({{ '/pages/1-rooflines' | relative_url }})'s first worked
problem asks you to do, which is to rebuild the number from its factors and see
whether the tool agrees.<d-footnote>Two follow-ups we owe on this: filing the
collector bug upstream, and checking whether the same 2x applies on MI355X before
quoting its bandwidth from a profile rather than from the data sheet.</d-footnote>

## Peak FLOPs by Dtype

**The FLOP ceiling is three numbers multiplied together, and you should compute it
rather than look it up.** For any CDNA part:

```
C = CUs * FLOPs per clock per CU * clock
```

For dense bf16 on MI300X, where each CU retires 2048 matrix FLOPs per clock:

```
304 * 2048 * 2.10e9 = 1.307e15 = 1307 TFLOP/s
```

**That is the single most important constant in this book.** Every MFU figure from
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}) onward is divided by it.

Here is the full table. The middle column is the one to keep, because it is the
architectural fact; the right column is just the middle column times `304 * 2.10e9`.

| Data type | Matrix FLOPs/clock/CU | Dense peak | With structured sparsity |
|---|---|---|---|
| FP64 (vector) | 128 | 81.7 TFLOP/s | — |
| FP64 (matrix) | 256 | 163.4 TFLOP/s | — |
| FP32 (vector) | 256 | 163.4 TFLOP/s | — |
| FP32 (matrix) | 256 | 163.4 TFLOP/s | — |
| TF32 | 1024 | 653.7 TFLOP/s | 1307.4 TFLOP/s |
| FP16 | 2048 | 1307.4 TFLOP/s | 2614.9 TFLOP/s |
| BF16 | 2048 | 1307.4 TFLOP/s | 2614.9 TFLOP/s |
| FP8 | 4096 | 2614.9 TFLOP/s | 5229.8 TFLOP/s |
| INT8 | 4096 | 2614.9 TOP/s | 5229.8 TOP/s |

> **Verified against:** ROCm MI300 microarchitecture documentation and the MI300X data
> sheet, 4 August 2026. **All figures in this book use the dense column.**

**Three things to notice, in descending order of how likely they are to cost you a
day.**

**First, published tables disagree in the third digit, and one class of source
disagrees by 2x.** AMD's data sheet says 1307.4 TFLOP/s; several OEM pages say 1305
from a slightly different clock assumption; and a few third-party spec sites list 653
because they took the dense figure for the sparse one and halved it. Rebuilding from
`304 * 2048 * 2.10e9` settles all three arguments in one line, which is the actual
reason to show the arithmetic rather than the table.

**Second, fp8 is exactly 2x bf16.** Not 1.8x, not "up to 2x": the Matrix Core does
4096 fp8 FLOPs per clock against 2048 for bf16. That exactness is load-bearing in
[Chapter 6]({{ '/pages/6-training' | relative_url }}), where it lets us say precisely
what fp8 does to each parallelism inequality instead of hand-waving about "faster
matmuls".

**Third, the sparsity column is not for you.** Structured sparsity requires the
weights to be sparse in a hardware-specified pattern. Nothing in this book produces
such weights, so the right column exists here only so you can recognise it when a
vendor comparison quotes it.

**Now substitute into [Chapter 1]({{ '/pages/1-rooflines' | relative_url }})'s ridge
point and get the number this book uses more than any other.** In bf16 the ridge
point in tokens is just `C / β_hbm`:

```
1307.4e12 / 5.3e12 = 247 tokens per device
```

**Below roughly 250 tokens per device, an MI300X matmul is memory-bound and the
Matrix Cores are idling.** Above it you are paying for the FLOPs you bought. Keep it
to two significant figures and do not treat it as a cliff: it is where two straight
lines cross, and the region within 2x of it is a curve.

Note what this means for a realistic training config. A per-device batch of 8
sequences at 2048 tokens is 16384 tokens, which is 66x the ridge point, so dense
training on MI300X is comfortably compute-bound and the interesting question is
communication, not bandwidth. Decode at a batch of one is 1 token, which is 247x
*below* it, and that asymmetry is the entire content of
[Chapter 11]({{ '/pages/11-inference' | relative_url }}).

### Numeric Formats, Once, for the Whole Book

**fp8 is not one format, and the two AMD generations in this book do not agree on
which one it is.** This gets introduced here, once, so that
[Chapters 6]({{ '/pages/6-training' | relative_url }}),
[11]({{ '/pages/11-inference' | relative_url }}) and
[12]({{ '/pages/12-serving' | relative_url }}) can spend it without re-explaining it.

Both generations offer E4M3 (1 sign bit, 4 exponent, 3 mantissa) and E5M2. The
difference is what happens to the bit patterns that a bigger float would spend on
infinities and negative zero:

- **gfx942 (MI300X, MI325X) implements the FNUZ variants**, `e4m3fnuz` and
  `e5m2fnuz`. "Finite and NaN only, unsigned zero": no infinities, no negative zero,
  exactly one NaN sitting where `-0` would be. Dropping those cases frees encodings,
  so AMD shifted the exponent bias to 8 and got a maximum representable value of 240.
- **gfx950 (MI355X) implements the OCP variants**, the Open Compute Project standard
  that NVIDIA has shipped since H100: bias 7, maximum 448, signed zeros, NaN at the
  top of the range.

**The same byte means different numbers on the two parts, and that is a checkpoint
compatibility problem rather than a numerics curiosity.** The bit pattern holding the
largest finite value on MI300X decodes to NaN on MI355X. This is why vLLM and SGLang
carry a load-time conversion that fires when the architecture string contains
`gfx94`: it patches the NaN pattern and doubles every scale factor, because identical
bits mean half the value on the older part.
[Chapter 12]({{ '/pages/12-serving' | relative_url }}) meets this again as an export
problem, and MaxText's two config values, `nanoo_fp8` for gfx942 against `fp8` for
gfx950, are the same distinction showing up as a flag you have to set correctly.

**Note:** the microscaling formats (MXFP8, MXFP6, MXFP4) are defined as blocks of OCP
fp8, and there is no FNUZ version of them. That is the practical reason gfx950
switched, and it means MI300X cannot run the MX formats at all.

## Inside a Node

**Eight MI300X modules sit on one universal baseboard, and they are wired as a
switchless, fully connected mesh.** Each GPU has seven Infinity Fabric (xGMI) links,
one to each peer, plus one PCIe Gen 5 x16 to the host. Every GPU is exactly one hop
from every other GPU, and there is no switch anywhere in the path.

The bandwidths, and please read the direction qualifier carefully because it is a
factor of 2 in every collective estimate in this book:

| Quantity | MI300X |
|---|---|
| Links per GPU | 7 |
| Per link, bidirectional | 128 GB/s |
| Per link, unidirectional | 64 GB/s |
| Per GPU aggregate, bidirectional | 896 GB/s |
| Per GPU aggregate, unidirectional | 448 GB/s |
| Host link | 1x PCIe Gen 5 x16, 128 GB/s bidirectional |

> **Verified against:** AMD MI300X Platform data sheet and the ROCm xGMI blog post,
> 4 August 2026. **This book quotes unidirectional bandwidth in cost models**, so
> `β_xgmi = 64 GB/s` per link unless stated otherwise. AMD's marketing figures are
> bidirectional, so they look 2x larger.

**How this differs from the two topologies you may have arrived with matters more
than the raw numbers.** An NVLink domain puts a switch in the middle, which costs a
hop but scales: you can grow past eight GPUs by adding switch ports. A TPU pod is a
torus, where each chip talks to its neighbours and distant chips are several hops
away, which is why TPU collective costs are famously independent of how many chips
are on the axis. **AMD is neither.** It is a complete graph on eight nodes, which is
the best possible topology at that size and *does not extend to nine*. The eight-GPU
scale-up domain is a hard ceiling: the ninth GPU is over the NIC, at a bandwidth an
order of magnitude lower.

That single fact drives more placement decisions in this book than anything else in
this chapter. [Chapter 6]({{ '/pages/6-training' | relative_url }}) uses it to decide
which parallelism axis must stay inside a node,
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) uses it to argue that the expert axis
in particular has to stay on the baseboard, and both of them cite it rather than
re-deriving it.

**This chapter owns the wiring; [Chapter 4]({{ '/pages/4-sharding' | relative_url }})
owns what it costs.** Everything about which collective algorithm RCCL picks, and
what an all-reduce therefore costs, is over there. The one thing worth previewing,
because it is counterintuitive: a mesh rewards using *all* of it. AMD's own guidance
is that collective performance is best when all eight GPUs participate, because a
2-GPU or 4-GPU collective can only light a fraction of the links. Two GPUs talking to
each other get one link between them, 64 GB/s, not 448.

**An honesty note about the sources, which is a useful lesson early.** AMD's MI300X
*platform* data sheet describes the topology twice on the same page, once as
"fully-meshed 128 GB/s bidirectional Infinity Fabric connectivity" and once as
"connects all GPUs in the 8-node **ring**" with a row labelled "Ring of 8 aggregate
bandwidth". The MI300X *GPU* data sheet says "seven Infinity Fabric links for full
connectivity between eight GPUs in a ring". **The mesh is correct**: the ROCm
architecture documentation, the cluster reference guide and the xGMI blog all describe
seven direct links forming a fully connected eight-GPU graph, and 896 GB/s is just
`7 * 128`, which only makes sense per-GPU across seven links. The "ring" wording
appears to be aggregate-bandwidth boilerplate from an earlier generation. Cross-check
vendor documentation against itself; it is cheaper than cross-checking it against a
benchmark.

## Beyond a Node

**Past eight GPUs you are on Ethernet, and this is the weakest part of the book.**
We have no multi-node allocation, so everything in this section is
**[analytical]** and stays that way until we do. Say the number out loud when you use
it: a claim about 64-GPU training in this book is a claim about arithmetic, not about
a run we watched.

What the reference designs specify, which is the best we can do for now:

- **Eight scale-out NICs per node, one per GPU**, at 400 Gbps each. That is 50 GB/s
  per GPU, or 400 GB/s of node egress, against 448 GB/s of *per-GPU* intra-node
  bandwidth. Call it a factor of 9 per GPU between talking to a peer on the baseboard
  and talking to a peer in the next rack.
- **RoCE over Ethernet is the default**, with InfiniBand as an alternative. AMD's
  cluster guide specifies a leaf-spine fabric with the eight per-node NICs spread
  across leaves in a rail-optimised layout, so that GPU `i` in every node shares a
  leaf with GPU `i` in every other node.
- **The rail layout is why mesh axis order matters so much**, and why
  [Chapter 4]({{ '/pages/4-sharding' | relative_url }})'s section on multi-process JAX
  is not optional reading. A collective that runs "GPU `i` to GPU `i` across nodes"
  rides one rail and behaves well. A collective that needs all-to-all across node
  boundaries hits the spine and does not.

**The single number [Chapters 6]({{ '/pages/6-training' | relative_url }}),
[7]({{ '/pages/7-moe' | relative_url }}) and
[12]({{ '/pages/12-serving' | relative_url }}) need from here is `β_net = 50 GB/s` per
GPU**, and the single fact is that it is roughly 8-9x worse than `β_xgmi`. Use it,
mark the result **[analytical]**, and treat any strategy whose cost model puts a
frequent collective on that axis as suspect until measured.

## Worked Example: A 4096 Cubed bf16 Matmul

**The whole chapter earns its place here.** Multiply two `bf16[4096, 4096]` matrices
on one MI300X. **How long do we expect this to take?**

First, which bound are we against? The matmul does `2 * 4096^3 = 1.37e11` FLOPs and
moves `3 * 4096^2 * 2 = 1.01e8` bytes, so its arithmetic intensity is
`1.37e11 / 1.01e8 = 1365` FLOPs per byte, against the machine's `247`. Comfortably
compute-bound, by more than 5x, so the memory term should hide entirely behind the
arithmetic and `t_math` alone should predict it.

```
2 * 4096^3 / 1307.4e12 = 105 microseconds
```

**How long does it actually take?** We do not know yet, and that is deliberate. This
chapter has no way to find out: we can predict the number but we cannot read a
profile. [Chapter 3]({{ '/pages/3-profiling' | relative_url }}) is where we capture the
trace, find the kernel, and see whether 105 microseconds was right. The script is
`jax_matmul.py`, it runs exactly this shape, and the answer is one chapter away.

**A prediction to hold onto while you wait**, since it is the interesting part: 105
microseconds is short. A kernel that runs for a hundred microseconds is in a regime
where launch overhead, autotuning on the first call, and clock ramp are all
comparable to the work. Expect the first iteration to be much slower than the tenth,
and expect to have to say which one you are quoting.

## What Changes Across the Family

**Short, but read it if you are on anything other than an MI300X, because the
substitutions are not the ones you would guess.**

**MI325X is the easy one and the one most likely to be got wrong.** Same gfx942, same
CDNA 3, same 304 CUs at 2.1 GHz, therefore *the entire FLOP table above is unchanged*.
What changes is memory: **256 GB of HBM3E at 6.0 TB/s** instead of 192 GB at 5.3.
A reader who carries MI300X's memory numbers onto MI325X gets every compute
prediction right and every bandwidth-bound prediction wrong, which is a confusing
failure mode because half your model holds. The ridge point moves down slightly, to
`1307.4 / 6.0 = 218` tokens.

**MI355X changes almost everything, including the direction of one number.**

| Quantity | MI300X | MI355X |
|---|---|---|
| ISA target | gfx942 (CDNA 3) | gfx950 (CDNA 4) |
| Compute units | 304 | **256** |
| Peak clock | 2.10 GHz | 2.40 GHz |
| BF16 matrix FLOPs/clock/CU | 2048 | 4096 |
| Dense BF16 | 1307.4 TFLOP/s | 2516.6 TFLOP/s |
| Dense FP8 | 2614.9 TFLOP/s | 5033.2 TFLOP/s |
| Dense FP6 and FP4 | not supported | 10066.3 TFLOP/s |
| HBM | 192 GB HBM3 at 5.3 TB/s | 288 GB HBM3E at 8.0 TB/s |
| LDS per CU | 64 KB | **160 KB** |
| Vector registers | 512 per lane | 512 per lane, unchanged |
| xGMI per link, bidirectional | 128 GB/s | 153.6 GB/s |
| Board power | 750 W | 1400 W |

> **Verified against:** AMD MI355X product brief and the ROCm occupancy-math blog
> post, 4 August 2026.

Four things in that table are traps.

**The CU count went down.** 256 against 304. CDNA 4 got faster per CU and narrower
overall, so anything you scaled by CU count is wrong in the direction that flatters
the new part. Rebuild from the formula: `256 * 4096 * 2.40e9 = 2516.6 TFLOP/s`.

**LDS tripled while the register file did not.** 64 KB to 160 KB per CU, with the
512-entry-per-lane vector register file unchanged. The common assumption is that
CDNA 4 doubled the registers too, and it did not. This pairing is exactly why
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }})'s occupancy section can
show the *same kernel* being LDS-bound on MI300X and register-bound on MI355X.

**MXFP6 runs at the same rate as MXFP4**, 10066.3 TFLOP/s dense for both, rather than
fp6 sitting halfway between fp8 and fp4 as you would expect from the bit widths. On
gfx950, fp6 is a nearly free accuracy upgrade over fp4. That is a genuine
scaling-arithmetic result and
[Chapter 11]({{ '/pages/11-inference' | relative_url }}) uses it.

**The ridge point barely moved**, despite every number in the table changing:
`2516.6 / 8.0 = 315` tokens against MI300X's 247. Compute and bandwidth grew together,
which is the normal state of affairs and the reason the ridge point is a durable
quantity to reason with.

**One paragraph on what comes next, and no numbers we cannot check.** The MI400 series
(MI455X, CDNA 5, HBM4) and the rack-scale Helios systems landed in mid-2026. CDNA 5
renames the compute unit to a Work Group Processor, so the book's unit of accounting
changes name again, and published per-GPU HBM4 bandwidth figures currently disagree
with each other. Go to the primary data sheet, rebuild the FLOP number from
`units * FLOPs per clock * clock`, and do not trust a third-party table you cannot
reconstruct.

## A Translation Table: MI300X, H100 and TPU v5e

**Most readers arrive holding NVIDIA vocabulary, and the cheapest way to meet them is
a dictionary.** The rows are unit-for-unit equivalences, not performance claims.

| AMD (CDNA 3) | NVIDIA (Hopper) | TPU v5e | Notes |
|---|---|---|---|
| Compute Unit (CU) | Streaming Multiprocessor (SM) | TensorCore | The unit of scheduling and the unit you multiply by |
| SIMD (4 per CU) | Processing block (4 per SM) | — | |
| Wavefront, 64 threads | Warp, 32 threads | — | **The one everyone trips on** |
| Matrix Core, MFMA | Tensor Core, HMMA/QMMA | MXU | Where the FLOPs come from |
| LDS, 64 KB per CU | Shared memory / SMEM | VMEM | Software-managed scratchpad |
| Infinity Cache, 256 MB | L2, 50 MB | CMEM | Last level before HBM |
| xGMI / Infinity Fabric | NVLink | ICI | Scale-up interconnect |
| Switchless mesh, max 8 | NVSwitch, scales past 8 | 2D or 3D torus | Topology, and its ceiling |
| RCCL | NCCL | — | Collective library; RCCL is a port of NCCL |
| HIP | CUDA | — | |
| `rocm-smi` | `nvidia-smi` | | |
| gfx942, gfx950 | sm_90, sm_100 | | Architecture target string |

**We have the same workload captured on all three, which almost nobody does, and it
is worth being precise about what that does and does not support.** The device planes
from those traces agree with the data sheets on the structural numbers: 304 CUs at
2.10 GHz on MI300X against 132 SMs at 1.98 GHz on H100 **[measured]**, read out of
the trace metadata rather than a specification. What the captures cannot support is
any scaling claim, because the device counts differ: 8x MI300X, 4x H100, and a single
v5e chip. Per-device comparisons and vocabulary mapping, yes. "AMD scales better than
NVIDIA", absolutely not, and we will not be making that claim from this data.

<!-- BLOCKED: the per-device comparison table (same workload, three platforms, step
     time and kernel breakdown side by side) needs numbers extracted from the three
     committed traces in gpu_profiling/traces/transformer_block/ via
     gpu_profiling/tools/parse_xplane.py. The traces exist; the extraction has not
     been done. Blockers: (a) AMD op times are summed across devices, so the 8x
     figures need dividing by 8 before any comparison, see Chapter 3's limitations
     table row 5; (b) H100 and v5e numbers need the same normalisation and neither
     has been checked; (c) decide whether a comparison at 8 vs 4 vs 1 devices is
     publishable at all, or whether only the per-device kernel times are.
     Until then this section stays a vocabulary table, which is the part that does
     not need measurement. -->

## Worked Problems

**Question 1:** Your job reports 152 CUs instead of 304 and every FLOP prediction in
this chapter is 2x too high. What happened, and what is the correct peak bf16 figure
for the device you are actually on?

{% details Click here for the answer. %}

The GPU is partitioned. MI300X can be presented as one logical device with all 304
CUs, or split along its eight XCDs into several smaller logical devices; 152 CUs is
four XCDs, so you are seeing half a physical GPU. Recompute from the formula with the
CU count you actually have:

```
152 * 2048 * 2.10e9 = 653.7 TFLOP/s
```

**And check the memory too**, because NPS memory partitioning is configured
separately from compute partitioning: your bandwidth and capacity may or may not have
been halved alongside the CUs, and the ridge point moves accordingly. `rocm-smi` will
tell you the current compute-partitioning and memory-partitioning modes.

This is worth practising because the failure is silent. Nothing errors, the model
trains, and every roofline you compare against is wrong by a clean factor that looks
like a software problem.

{% enddetails %}

**Question 2:** You are moving a training config from MI300X to MI355X. Which of
these change, and by how much: the peak bf16 FLOP rate, the ridge point in tokens,
the maximum model that fits in one GPU's memory, the number of GPUs in a scale-up
domain, and the fp8 checkpoint you already produced?

{% details Click here for the answer. %}

- **Peak bf16: 1307.4 to 2516.6 TFLOP/s**, a 1.92x improvement, from `256 * 4096 *
  2.40e9`. Note that it is *not* the 1.19x you would get by scaling CU count and
  clock, because the per-CU matrix throughput doubled and the CU count fell.
- **Ridge point: 247 to 315 tokens**, from `2516.6 / 8.0`. It went up, so a
  per-device batch that was comfortably compute-bound on MI300X is slightly less so
  on MI355X. Rarely decisive at training batch sizes; occasionally decisive at
  decode.
- **Memory: 192 GB to 288 GB**, a 1.5x increase, so 50% more parameters or KV cache
  per device.
- **Scale-up domain: unchanged at 8.** Same seven-link switchless mesh, same
  baseboard, same hard ceiling. Per-link bandwidth rises from 128 to 153.6 GB/s
  bidirectional, about 20%, which is much less than the 1.92x compute improvement, so
  **every communication-bound inequality in
  [Chapter 6]({{ '/pages/6-training' | relative_url }}) gets worse on the newer part.**
  That is the important one and it is the one nobody expects.
- **The fp8 checkpoint does not transfer.** gfx942 wrote FNUZ, gfx950 reads OCP, and
  the same bytes mean different values. Convert it: patch the NaN encoding and rescale.
  See [Chapter 12]({{ '/pages/12-serving' | relative_url }}).

{% enddetails %}

**Question 3:** Two GPUs on the same baseboard need to exchange 1 GB. How long, at
spec bandwidth? Now do it for eight GPUs each exchanging 1 GB with every peer, and
say why the per-GPU time is not eight times worse.

{% details Click here for the answer. %}

**Two GPUs:** they share exactly one xGMI link, 64 GB/s unidirectional.

```
1e9 / 64e9 = 15.6 ms
```

The other six links on each GPU are idle. This is the case AMD's documentation warns
about: a 2-GPU collective can only use a fraction of the fabric.

**Eight GPUs, all-to-all:** each GPU sends 1 GB to each of seven peers, but it sends
them *down seven different links, concurrently*. Per-GPU egress is
`7 * 64 = 448 GB/s`, and the total each GPU must send is 7 GB:

```
7e9 / 448e9 = 15.6 ms
```

**The same 15.6 ms.** Seven times the data in the same time, because a complete graph
has no contention for an all-to-all pattern: every pair has its own private link and
nothing shares. This is the single best thing about a switchless mesh, and it is why
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) argues that Mixture-of-Experts dispatch
should be *cheap* inside a baseboard, which is the opposite of what most readers
expect.

**Two caveats, both important.** This is spec bandwidth: AMD's own measurements put
realised per-link bandwidth around 45-48 GB/s, roughly 75% of peak, once protocol
overhead and CRC are accounted for. And it assumes RCCL actually schedules the
transfer as a direct all-to-all rather than as a ring;
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}) is where that stops being an
assumption.

{% enddetails %}

## References

**AMD primary sources.**

- [AMD Instinct MI300X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf).
  CU count, clock, HBM capacity and bandwidth, the FLOP table, the seven-link xGMI
  figure. Also the source of the "in a ring" phrasing discussed above.
- [AMD Instinct MI300X Platform data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf).
  The eight-GPU baseboard, 128 GB/s bidirectional per link, 896 GB/s aggregate.
- [AMD Instinct MI325X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf).
  Confirms the identical FLOP table and the 256 GB / 6.0 TB/s memory change.
- [AMD Instinct MI355X GPU brochure](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf).
  256 CUs, 2.4 GHz, 288 GB HBM3E at 8 TB/s, the fp6/fp4 rates, 7x 153.6 GB/s xGMI.
- [AMD Instinct MI300 Series Cluster Reference Architecture Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/other/instinct-mi300-series-cluster-reference-guide.pdf).
  The scale-out side: eight 400 Gbps NICs per node, the rail-optimised leaf-spine
  fabric, and the statement that NICs are not used inside the node.

**ROCm documentation and blogs.**

- [AMD Instinct MI300 Series microarchitecture](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html).
  The FLOPs-per-clock-per-CU table that the peak figures are built from, and the
  node-level diagram showing seven links forming a fully connected eight-GPU system.
- [MI300 and MI350 Series workload optimization](https://rocmdocs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/workload.html).
  The generation-to-generation comparison table, and AMD's guidance that collectives
  perform best when all eight GPUs participate.
- [MI300X RCCL and xGMI](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html).
  Confirms 64 GB/s unidirectional per link, and gives AMD's own realised figures of
  45-48 GB/s per link.
- [Occupancy Math on the AMD MI355X GPU (CDNA4)](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html).
  The CDNA 3 to CDNA 4 resource changes, the 64 KB to 160 KB LDS figure, the
  unchanged 512-entry register file, and the CDNA-to-CUDA vocabulary table this
  chapter's translation table builds on.
- [HIP low precision floating point types](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.0/reference/low_fp_types.html)
  and [ROCm precision support](https://rocm.docs.amd.com/en/docs-6.4.3/reference/precision-support.html).
  The FNUZ versus OCP fp8 definitions, and the note that gfx94x defaults to FNUZ.
