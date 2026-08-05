---
layout: distill
title: "All About Rooflines"
description: "What actually limits how fast this runs? Algorithms are bounded by three things: compute, memory bandwidth and communication. This chapter builds the arithmetic that tells you which one you are up against, with no hardware constants in it yet."
date: 2026-08-04

section_number: 1

previous_section_url: "/"
previous_section_name: "Chapter 0: Intro"

next_section_url: "/pages/2-amd-gpus"
next_section_name: "Chapter 2: AMD GPUs"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "Three Bounds: Compute, Memory, Communication"
  - name: Arithmetic Intensity
  - name: The Critical Batch Size
    subsections:
      - name: What the Name Does Not Mean
  - name: Communication Rooflines
  - name: Worked Problems
  - name: References
---

**Depends on:** nothing. This is the first chapter and it assumes only that you have
seen a matrix multiply.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

You have a model, you have eight GPUs, and you have a number in your head for how
fast the thing should run. It runs at a third of that. Then you add eight more GPUs
and it gets 1.3x faster instead of 2x. Somewhere between the arithmetic you did and
the wall clock there is a gap, and until you can name what is in that gap you are
guessing.

**This chapter is the vocabulary for naming it.** There are only three things an
accelerator can be waiting on: arithmetic, memory traffic, or the network. Every
performance conversation in the rest of this book is an argument about which of the
three is currently the binding constraint, and every parallelism strategy in
[Chapter 6]({{ '/pages/6-training' | relative_url }}) is a trade of one for another.

**There is deliberately no AMD content in this chapter.** The reasoning here is
hardware-independent, so that when
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) arrives with real constants
there is somewhere for them to land. Everything below is **[analytical]** by
construction: it is algebra about hardware, not a measurement of one. This is also
the shortest chapter in the book, and if you already think in FLOPs per byte you can
skim to [The Critical Batch Size](#the-critical-batch-size) and lose nothing.

## Three Bounds: Compute, Memory, Communication

**Take any single operation, on any accelerator, and there are exactly three clocks
running against it.** The chip has to feed operands from memory into the compute
units, it has to do the arithmetic, and if the operation is split across devices it
has to move bytes between them. Give each one a name and a duration:

- **`t_math`** is the arithmetic time: total FLOPs divided by the device's FLOPs per
  second, `C`.
- **`t_mem`** is the memory time: bytes moved between HBM (high-bandwidth memory)
  and the compute units, divided by `β_hbm`.
- **`t_comms`** is the communication time: bytes moved between devices, divided by
  the relevant link bandwidth.

**The three overlap, but never perfectly, and that gives us a bracket rather than a
number.** If the hardware manages to hide all the memory traffic and all the network
traffic behind the arithmetic, the operation takes `max(t_math, t_mem, t_comms)`. If
it hides nothing, it takes `t_math + t_mem + t_comms`. Reality sits between.

**That bracket is narrower than it looks, which is why this crude model is useful.**
When one term dominates, the sum and the max are pretty much the same number. When
two terms are equal, the sum is exactly twice the max. So for two terms the two
bounds never differ by more than 2x, and the interesting cases are the ones where
they are close to equal, which is precisely where the sum is at its worst relative
to the max.

We will use the max throughout, and call it the roofline: the fastest the operation
can possibly go. **When a measurement comes in at 2x the roofline, that is not
necessarily a bug**, it may just be an operation that overlaps nothing. When it
comes in at 10x, something is wrong, and finding out what is
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}).

**Note:** we are ignoring latency entirely. Every collective has a fixed startup
cost that dominates at small message sizes, and every kernel has a launch overhead.
Both matter, and both are invisible in this model. Where they bite, we will say so.

## Arithmetic Intensity

**The question "am I compute-bound or memory-bound" has a clean answer, and it is a
comparison of two ratios.** Divide the FLOPs an algorithm does by the bytes it has
to move, and you get its *arithmetic intensity*, in FLOPs per byte. Do the same for
the hardware, `C / β_hbm`, and you get the intensity the machine wants. The
algorithm is compute-bound exactly when its intensity is higher than the machine's:

```
t_math > t_mem
  <=>  FLOPs / C > bytes / β_hbm
  <=>  FLOPs / bytes > C / β_hbm
```

That equivalence is the single most reused move in this book. It converts a
statement about durations, which depend on problem size, into a statement about
ratios, which mostly do not.

**Start with the hopeless case.** A dot product of two bf16 vectors of length
`4096` does `2 * 4096 = 8192` FLOPs: one multiply and one add per element. It loads
both vectors, `2 * 4096 * 2 bytes = 16384` bytes. Its intensity is
`8192 / 16384 = 0.5` FLOPs per byte, and *no choice of `4096` changes that*, because
the FLOPs and the bytes both grow linearly in the length. A dot product is
memory-bound on every accelerator ever built and will be on every accelerator ever
built.

**Now the good case, and the reason Transformers are worth running on this hardware
at all.** Take a small matmul: activations `A` of shape `bf16[4, 512]` times weights
`W` of shape `bf16[512, 1024]`, giving `bf16[4, 1024]`.

- FLOPs: `2 * 4 * 512 * 1024 = 4.19e6`. Two per multiply-accumulate, once per
  combination of the three dimensions.
- Bytes: `2 * (4*512 + 512*1024 + 4*1024) = 1.06e6`. Both inputs in, one output
  back.
- Intensity: `4.19e6 / 1.06e6 = 3.95` FLOPs per byte.

That is suspiciously close to 4, which is the batch size, and it is not a
coincidence. Write it out with symbols. For `A[B, D] @ W[D, F]` in a format with `w`
bytes per element:

```
intensity = 2*B*D*F / (w * (B*D + D*F + B*F))
```

**When `B` is much smaller than `D` and `F`, the `D*F` term swamps the other two**,
because it is the only one that does not have a small `B` in it. Reading the weight
matrix is the whole memory bill. So:

```
intensity ≈ 2*B*D*F / (w * D*F) = 2*B / w
```

In bf16, where `w = 2`, the intensity of a matmul *is the batch size*, in tokens.
Nothing else about the shape matters. That is a remarkably clean result and the rest
of the chapter is consequences of it.

**Tip:** the condition `B << D, F` is the one to keep an eye on, not the conclusion.
Once `B` is comparable to `F`, the `B*D` and `B*F` terms come back and the intensity
saturates around `min(B, D, F)`-ish. For LLM training the per-device token batch is
in the hundreds or low thousands and `F` is in the tens of thousands, so we are
comfortably in the regime where the approximation holds. In
[Chapter 11]({{ '/pages/11-inference' | relative_url }}), where `B` can be 1, we are
even more comfortably in it.

## The Critical Batch Size

**Set the algorithm's intensity equal to the machine's and solve for the batch
size.** This is the ridge point of the roofline: below it you are memory-bound and
adding FLOPs buys nothing, above it you are compute-bound and the arithmetic is what
you are paying for.

```
2*B / w = C / β_hbm
  =>  B_crit = w * C / (2 * β_hbm)
```

**In bf16 this collapses to something you can hold in your head: `B_crit = C /
β_hbm`.** The ridge point in tokens is just the hardware's FLOPs-per-byte ratio.
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) substitutes MI300X numbers and
gets an actual figure; for now the shape of the result is the thing. Note what it
does *not* depend on: the model. Not `D`, not `F`, not the layer count. A
per-device token batch below the ridge point is a memory-bound matmul on that device
whatever the model is, and the fix is always the same, put more tokens on the device
or accept that you are paying for silicon you are not using.

**Two warnings about this number, because it is the most-cited quantity in the book
and it is easy to misapply.**

First, it is *per device*, and it is in *tokens*, not sequences. A global batch of 512
sequences of length 2048 spread over 64 GPUs is `512 * 2048 / 64 = 16384` tokens per
device, which is a very different question from "is 512 a big batch".

Second, it assumes the weights are read from HBM once per matmul, which is true for
a training step where each weight is used once per microbatch and false in several
interesting cases. If the matrix is small enough to stay resident in cache across
calls, the memory term goes away and the ridge point is irrelevant.

### What the Name Does Not Mean

**In the training literature, "critical batch size" means something else, and a
reader arriving from that literature will have the wrong quantity in mind.** There it
is the batch size past which more data stops buying convergence: double the batch,
halve the number of steps, and at some point that trade stops working and you are
just burning tokens. It is a property of the optimization problem, it has to be
measured empirically, and it has nothing to do with bandwidth.

**This book always means the hardware ridge point.** When the other meaning is
needed we call it the convergence limit and say so.
[Chapter 6]({{ '/pages/6-training' | relative_url }}) is where both are in play at
once, and the reason it matters there is that they squeeze the data-parallel degree
from opposite sides: the global batch has to be large enough that each device clears
the ridge point, and small enough that the optimizer still makes progress per token.
The number of GPUs you can usefully run data-parallel is the ratio between those two
bounds, which is a sobering way to look at a cluster.

## Communication Rooflines

**Everything so far has been about one device, where the only competition is
arithmetic against memory. Split the matmul across two devices and a third clock
starts.** The point of this section is not the specific inequality we derive; it is
that *which variable controls the roofline changes from strategy to strategy*, and a
reader who expects one universal threshold will misread
[Chapter 6]({{ '/pages/6-training' | relative_url }}) badly.

Take `A[B, D] @ W[D, F]` again, and split it down the contracting dimension `D`
across two devices connected by a link with bandwidth `β`. Device 0 holds the first
half of `D`, device 1 the second half. Each device multiplies its half and gets a
`[B, F]` result that is a *partial sum*: neither answer is right on its own, and the
two have to be added elementwise to get the real output. So each device does half
the arithmetic and then the pair exchanges `B * F` values to reconcile.

- `t_math = 2 * B * (D/2) * F / C = B*D*F / C`
- `t_comms = w * B * F / β`, one pass of the output over the link per device.

Compute-bound means `t_math > t_comms`, so:

```
B*D*F / C > w*B*F / β
  =>  D / C > w / β
  =>  D > w * C / β
```

**The batch size cancelled.** This crossover is controlled entirely by `D`, the
contracted dimension, and no amount of extra batch will save a split that is too
narrow. That is the opposite of the single-device story, where `B` was the only thing
that mattered, and it is why "just increase the batch size" is good advice for a
memory-bound matmul and useless advice for a communication-bound one.

**The size of `w * C / β` is worth previewing, because it is uncomfortable.** A
modern accelerator has a FLOPs-to-link-bandwidth ratio in the thousands, so the
threshold on `D` lands in the tens of thousands, which is larger than the model
dimension of most models you would want to train. Splitting the contracting
dimension of a single matmul across devices is therefore usually communication-bound
unless you have a lot of links or a lot of layers to overlap against. Both of those
are real options, and they are what tensor parallelism actually is.
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) gives the real bandwidths and
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}) gives the real collective costs;
the number to remember from here is that this ratio is a thousand-ish, not a ten-ish.

## Worked Problems

Two problems, both about the ridge point, because it is the one number from this
chapter you will use every day.

**Question 1:** Pull up AMD's
[MI300X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
and find the BF16 figure and the HBM bandwidth. Compute the bf16 ridge point in
tokens per device. Then look again at the BF16 figure and decide whether you used the
right one.

{% details Click here for the answer. %}

The data sheet gives two BF16 numbers in adjacent columns, **1307.4 TFLOPS** and
**2614.9 TFLOPS**, under a heading that reads "AI PEAK THEORETICAL PERFORMANCE with
sparsity". The larger one is the sparsity-enabled figure: it assumes structured
sparsity in the weights, which a dense LLM does not have and which nothing in this
book will produce. **Dense bf16 on MI300X is 1307.4 TFLOP/s**, and that is the number
we carry for the whole book. Memory bandwidth is 5.3 TB/s.

With the dense figure, in bf16, `w = 2` so the ridge point is just `C / β_hbm`:

```
1307.4e12 / 5.3e12 = 246.7 tokens per device
```

With the sparse figure you would have got `2614.9e12 / 5.3e12 = 493.4`, and you would
have spent the rest of the book believing you needed twice the batch size you
actually need to saturate the machine.

**The general habit is the point.** Marketing surfaces the largest defensible number,
which is usually sparse, sometimes a different data type than you thought, and
occasionally a boost clock nobody sustains. Find the dense figure, note it with its
source and its date, and rebuild it from `CUs * FLOPs per clock * clock` when you can
so you can tell whether the table is lying.
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) does exactly that.

{% enddetails %}

**Question 2:** MI300X does fp8 at exactly twice its bf16 rate: 2614.9 TFLOP/s dense
against 1307.4. Two variants, and they do not have the same answer.

**(a)** You train in fp8 end to end, weights and activations both one byte. What
happens to the ridge point?

**(b)** You keep the matmul in bf16 but store the weights in fp8, unpacking them on
the way in. What happens to the ridge point?

{% details Click here for the answer. %}

**(a) Nothing moves, and the reason is worth internalising.** Use
`B_crit = w * C / (2 * β_hbm)` with `w = 1` and `C = 2614.9e12`:

```
1 * 2614.9e12 / (2 * 5.3e12) = 246.7 tokens per device
```

The same 247 we got for bf16. Halving the bytes per element doubled the algorithm's
intensity, and doubling the FLOP rate doubled the hardware's, so the crossover
between them did not budge.

**Be careful about which currency you are quoting, because the answer differs.** In
FLOPs per byte the ridge point *did* double, from `1307.4 / 5.3 = 247` to
`2614.9 / 5.3 = 493`. In tokens per device it stayed at 247. Both statements are
true and they are about the same picture; a claim of the form "fp8 moves the ridge
point" is meaningless without the units attached. What fp8 actually bought you is
not a different crossover but twice the throughput on the same side of it.

**(b) It halves, to about 123 tokens.** Now `w = 1` for the bytes, because that is
what you load, but `C = 1307.4e12` for the arithmetic, because the matmul still runs
at the bf16 rate:

```
1 * 1307.4e12 / (2 * 5.3e12) = 123.3 tokens per device
```

Weight-only quantization buys bandwidth without buying FLOPs, so it moves the
crossover down: it makes small batches compute-bound that were previously
memory-bound. That sounds like a curiosity here and it is the whole ballgame in
[Chapter 11]({{ '/pages/11-inference' | relative_url }}), where decode runs at a batch
of one and is bandwidth-bound by construction. Halving the bytes you read per token
nearly halves the step time; halving the FLOPs does nothing at all.

{% enddetails %}

## References

- [AMD Instinct MI300X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
  (AMD). The dense-versus-sparse FLOP columns used in Question 1, plus the 5.3 TB/s
  HBM figure.
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785)
  (Williams, Waterman and Patterson, 2009). The original roofline paper, and the
  source of the ridge-point framing this chapter uses.
- [How To Scale Your Model, Part 1: Rooflines](https://jax-ml.github.io/scaling-book/roofline/)
  (Google DeepMind). The TPU-side treatment of the same material. Useful if you want
  a second pass over arithmetic intensity in a different hardware idiom.
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)
  (McCandlish et al., 2018). The *other* critical batch size, the convergence one, if
  you want to see where that meaning comes from.
