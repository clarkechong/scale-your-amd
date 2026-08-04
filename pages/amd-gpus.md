---
layout: distill
title: "How to Think About AMD GPUs"
description: "How fast should this run on an MI300X, and what in the hardware decides that? Compute units, the memory hierarchy, peak FLOPs by dtype, and the switchless xGMI mesh that eight GPUs sit on. Every constant the rest of the book substitutes into."
date: 2026-08-04

section_number: 2

previous_section_url: "/pages/rooflines"
previous_section_name: "Chapter 1: Rooflines"

next_section_url: "/pages/profiling"
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
---

> **Skeleton.** Section structure only; the prose, the arithmetic and the figures are
> still to be written. The brief for this chapter is the Chapter 2 section of
> `docs/structure.md`.

**Depends on:** [Chapter 1]({{ '/pages/rooflines' | relative_url }}) for rooflines,
arithmetic intensity and the ridge point. Nothing later.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

> **To write.** Open by naming the job: Chapter 1 gave you three bounds with symbols
> in them, and this chapter fills in the symbols. Every number here is one the rest of
> the book substitutes into, which is also the editorial test for the chapter.
>
> **Resist making this a survey of CDNA.** If a fact does not feed an arithmetic
> prediction, cut it. The test to apply to every paragraph is "which later prediction
> breaks without this?" and a paragraph that cannot answer does not belong.

> **Verify before writing.** Every hardware number below needs checking against
> official specifications, with the date of the check stated in the text. Spec tables
> carry a "verified against" line; see the Conventions section of
> `docs/structure.md`.

## What an MI300X Is

> **To write.** XCDs, compute units, SIMDs, MFMA matrix cores, and the register file
> and LDS per CU. The numbers that matter downstream are **304 CUs at roughly
> 2.10 GHz**, both of which our own traces report in the device plane, which is a nice
> early example of the profile agreeing with the spec sheet.
>
> Mention SPX and NPS partitioning only insofar as it changes the CU count a process
> sees, because that is the only way it shows up in an arithmetic prediction. It comes
> back in [Appendix B]({{ '/pages/appendix-protocol' | relative_url }}) as something
> every measurement has to state.

## The Memory Hierarchy

> **To write.** HBM3, Infinity Cache, L2, LDS and registers, with capacity and
> bandwidth at each level. The headline pair is **192 GiB of HBM at 5.3 TB/s**, and it
> supplies the memory-bound side of every later roofline.
>
> Note the honest footnote here rather than hiding it: our own profiles report HBM
> bandwidth as 2479.6, which is wrong twice over, and the write-up of why belongs in
> this chapter as a worked example of checking the tool against the spec sheet. The
> arithmetic is in the Open questions section of `docs/structure.md`, and
> [Chapter 3]({{ '/pages/profiling' | relative_url }}) points back here for it.

## Peak FLOPs by Dtype

> **To write.** A table covering fp32, tf32, bf16, fp16, fp8 and int8, with a
> "verified against" line under it. State plainly that marketing numbers usually quote
> the sparsity-enabled figure and that this book uses dense throughout.
>
> **Show the arithmetic rather than quoting the table.** Dense bf16 on MI300X is
> `304 * 2048 * 2.10e9 = 1307 TFLOP/s`, from 304 CUs at 2048 FLOP per clock per CU at
> 2.10 GHz. Two reasons to derive it: the reader can rebuild the number for any other
> part in the family, and published tables disagree in the third digit. AMD's own
> sheet says 1307.4, several OEM pages say 1305 from a slightly different clock, and a
> few third-party sites have it at 653 because they mistook the dense figure for the
> sparse one.
>
> Then substitute into
> [Chapter 1]({{ '/pages/rooflines' | relative_url }})'s ridge point and get the AMD
> critical batch size. That number is the one the rest of the book keeps quoting.

### Numeric Formats, Once, for the Whole Book

> **To write.** This is the only place the formats get introduced, and Chapters
> [6]({{ '/pages/training' | relative_url }}),
> [11]({{ '/pages/inference' | relative_url }}) and
> [12]({{ '/pages/serving' | relative_url }}) spend it rather than re-explaining it.
>
> Two facts do the work. fp8 is 2x bf16 on MI300X, which halves the ridge point. And
> gfx942's fp8 is the NANOO/FNUZ variant rather than the OCP one that gfx950
> implements, which bites anyone moving a checkpoint between the two: it turns up again
> as a training-config question in Chapter 6 and as a checkpoint-compatibility problem
> in Chapter 12.

## Inside a Node

> **To write.** The 8-GPU OAM baseboard (UBB 2.0) is a *switchless, fully connected
> mesh*: seven xGMI links per GPU at 128 GB/s bidirectional each, 896 GB/s aggregate,
> one hop to every peer, plus one PCIe Gen 5 x16 to the host. Contrast with an NVLink
> domain, which puts a switch in the middle and can therefore grow past eight.
> **The AMD scale-up domain cannot: eight is the ceiling, and the ninth GPU is over
> the NIC.**
>
> That is enough to predict an all-reduce and enough for
> [Chapter 6]({{ '/pages/training' | relative_url }}) to reason about what has to stay
> on the baseboard.
>
> **This chapter owns the wiring; Chapter 4 owns what it costs.** Describe the
> topology once, here, in physical terms.
> [Chapter 4]({{ '/pages/sharding' | relative_url }}) gets one line of recap and
> spends its space on the collective cost model. Chapters
> [6]({{ '/pages/training' | relative_url }}),
> [7]({{ '/pages/moe' | relative_url }}) and
> [12]({{ '/pages/serving' | relative_url }}) cite the eight-GPU ceiling rather than
> restating it. Without that split, this paragraph gets written four times.
>
> One footnote worth including rather than smoothing over: AMD's own MI300X data sheet
> says "seven Infinity Fabric links for full connectivity between eight GPUs **in a
> ring**," while the platform data sheet and the ROCm architecture docs in the same
> family say fully meshed. The mesh is correct. It is a good early demonstration that
> vendor documentation needs cross-checking, and it costs one footnote.

## Beyond a Node

> **To write.** Scale-out NICs, RoCE and Ethernet against InfiniBand, per-node egress
> bandwidth, and the fabric topology of a typical MI300X cluster.
>
> **This is currently a hole in our knowledge and a hole in the source book, and it is
> where a lot of production performance is decided.** Chapters
> [6]({{ '/pages/training' | relative_url }}),
> [7]({{ '/pages/moe' | relative_url }}) and
> [12]({{ '/pages/serving' | relative_url }}) all need a node-egress number. Until we
> have one, say so in the text and mark the affected derivations **[analytical]**.
> Being visibly short of a number is survivable; implying we have one is not.

## Worked Example: A 4096 Cubed bf16 Matmul

> **To write.** The first full predict-then-measure in the book, and the cliffhanger
> that [Chapter 3]({{ '/pages/profiling' | relative_url }}) resolves.
> `2 * 4096**3 / 1307e12` gives the expected time. Then stop, and say plainly that we
> have no way yet to find out whether that was right, which is what the next chapter
> is for.
>
> Use the same shape as `jax_matmul.py` so the two chapters line up on one artifact.

## What Changes Across the Family

> **To write.** Short, but with the counterintuitive parts stated plainly, because a
> reader substituting constants will get them wrong otherwise.
>
> **MI325X first, in one sentence**, because it is the cheapest thing to get wrong:
> same gfx942, same 304 CUs and therefore the same FLOP table as MI300X, but 256 GB of
> HBM3E at 6.0 TB/s. A reader who carries the MI300X memory numbers onto an MI325X
> gets the compute predictions right and every bandwidth-bound one wrong, which is a
> confusing way to be wrong.
>
> **Then MI355X, which changes almost everything.** gfx950, CDNA4, 288 GB HBM3E at
> 8.0 TB/s, and **256 CUs, which is fewer than MI300X's 304**: the generation got
> faster per CU and narrower, so anything the reader scaled by CU count breaks. Dense
> bf16 goes to roughly 2.5 PFLOP/s and fp8 to 5.0. xGMI per-link rises to 153.6 GB/s
> while the seven-link mesh topology is unchanged. **LDS goes from 64 KB to 160 KB per
> CU while the 512-entry-per-lane vector register file does not change**, which is the
> pairing to state explicitly because the common assumption is that CDNA4 doubled the
> registers too.
>
> Revised FLOP table including fp6 and fp4, where the interesting entry is that
> **MXFP6 runs at the same rate as MXFP4** rather than half of it, so on gfx950 fp6 is
> a nearly free accuracy upgrade over fp4. That is a real scaling-arithmetic result and
> worth its own sentence.
>
> One paragraph, no more, on what comes after: the MI400 series (MI455X, CDNA5, HBM4)
> and rack-scale Helios landed in mid-2026, and CDNA5 renames the compute unit to a
> Work Group Processor, so the book's core unit of accounting changes name again. Point
> at primary datasheets rather than quoting numbers we cannot check, and note that
> published per-GPU HBM4 bandwidth figures for MI455X currently disagree with each
> other.

## A Translation Table: MI300X, H100 and TPU v5e

> **To write.** Unit for unit: CU against SM against Tensor Core, LDS against SMEM
> against VMEM, MFMA against Tensor Core against MXU. Readers arrive with NVIDIA
> vocabulary and this is the cheapest way to meet them.
>
> AMD's [occupancy math post](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html)
> has a compact CDNA-to-CUDA table that covers the compute-side rows well, including
> the one people trip on, that an AMD wavefront is 64 threads against NVIDIA's
> 32-thread warp. Use it as a source and add the TPU column, which is ours to write.
>
> We have same-workload traces on all three, which almost nobody else does, so this
> can be empirical rather than a spec-sheet comparison. **Caveat the empirical half
> honestly:** the captures are 8x MI300X, 4x H100 and a single v5e chip, so they
> support per-device comparisons and unit-vocabulary mapping but not scaling claims.
> Say which one the table is doing.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number.

**Question 1:** Rebuild the dense bf16 FLOP rate for a part whose spec sheet you do
not have.

{% details Click here for the answer. %}

To write.

{% enddetails %}

**Question 2:** At what per-device token batch size does an MI300X matmul cross from
memory-bound to compute-bound, in bf16 and in fp8?

{% details Click here for the answer. %}

To write, with both numbers stated and the factor between them explained.

{% enddetails %}

**Question 3:** How long should an all-reduce of a 1 GB buffer take across eight GPUs
on one baseboard, from the numbers in this chapter alone?

> **To write.** The point is that the reader can get a defensible answer from the
> wiring, and that
> [Chapter 4]({{ '/pages/sharding' | relative_url }}) will then show them the measured
> figure and the gap. Set the expectation that the gap exists.

{% details Click here for the answer. %}

To write.

{% enddetails %}
