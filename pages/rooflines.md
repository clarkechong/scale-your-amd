---
layout: distill
title: "All About Rooflines"
description: "What actually limits how fast this runs? Algorithms are bounded by three things: compute, memory bandwidth and communication. This chapter builds the arithmetic that tells you which one you are up against, with no hardware constants in it yet."
date: 2026-08-04

section_number: 1

previous_section_url: "/"
previous_section_name: "Chapter 0: Intro"

next_section_url: "/pages/amd-gpus"
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
---

> **Skeleton.** Section structure only; the prose, the arithmetic and the figures are
> still to be written. The brief for this chapter is the Chapter 1 section of
> `docs/structure.md`.

**Depends on:** nothing. This is the first chapter and it assumes only that you have
seen a matrix multiply.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

> **To write.** Open on the gap: you have a model and some accelerators, you double
> the accelerators, and you do not get double the throughput. The reason is always
> one of three bounds, and this chapter is the vocabulary for saying which.
>
> **There is deliberately no AMD content in this chapter.** Hardware-independent
> reasoning comes first so that
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }})'s constants have somewhere to
> land. Say that in the opening so the reader is not waiting for MI300X numbers that
> arrive one chapter later. This is also the gentlest chapter in the book and should
> stay short.

## Three Bounds: Compute, Memory, Communication

> **To write.** Define `t_math` and `t_comms`, then the bracket: `max(t_math,
> t_comms)` is a decent lower bound on step time and `t_math + t_comms` is the upper
> bound, depending on whether the two overlap. The useful observation is that they
> differ by at most 2x, so it rarely matters which you use. Pick the lower bound and
> say so, because every later chapter substitutes into it.

## Arithmetic Intensity

> **To write.** FLOPs per byte, and the algebraic move the whole book reuses: being
> compute-bound is exactly the statement that the algorithm's intensity exceeds the
> hardware's. Dot product as the hopeless case, since it does 2 FLOPs per 2 loaded
> elements no matter how big it gets. Matmul as the good case, where intensity grows
> with the shared dimension.
>
> Toy shapes first, per the style guide, then generalise.

## The Critical Batch Size

> **To write.** Matmul intensity is roughly `B` when `B` is much smaller than `D` and
> `F`, so the ridge point where compute-bound turns into memory-bound falls at a
> per-device *token* batch size rather than a sequence batch size. Derive it
> symbolically here;
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) substitutes MI300X numbers and
> gets an actual figure.
>
> This is the single most-cited number in the book, so the derivation has to be clean
> enough that a reader can rebuild it from memory a month later.

### What the Name Does Not Mean

> **To write.** One paragraph, and it earns its place because the collision is real.
> In the training literature "critical batch size" means the batch size past which
> more data stops buying convergence: a different quantity, with a different value,
> and the meaning a training-focused reader arrives holding. **This book always means
> the hardware ridge point.**
> [Chapter 6]({{ '/pages/training' | relative_url }}) is where both meanings are in
> play at once, because the global batch is bounded below by this ridge point and
> above by convergence, and the data-parallel degree has to fit between them.

## Communication Rooflines

> **To write.** A two-device sharded matmul, worked all the way through, where the
> crossover depends on `D` rather than on `B`. The point of the section is not the
> specific inequality, it is to prime the reader that *which* variable controls the
> roofline changes from strategy to strategy. That is the entire shape of
> [Chapter 6]({{ '/pages/training' | relative_url }}), and a reader who expects one
> universal threshold will misread it.
>
> Keep the topology abstract here: two devices and a link with bandwidth `β`. Real
> interconnects arrive in the next chapter.

## Worked Problems

> **To write.** Answers behind `{% raw %}{% details %}{% endraw %}`, each with a
> reference number so the reader can tell whether they got it right.

**Question 1:** Spec-sheet skepticism.

> **To write.** Hand the reader an AMD spec sheet and let them notice for themselves
> that the headline bf16 number is the sparsity-enabled figure, then have them redo
> the ridge point with the dense one. Teaching spec-sheet skepticism as an exercise is
> cheap, and it sets up the honesty that
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}) needs about published tables
> and [Chapter 3]({{ '/pages/profiling' | relative_url }}) needs about our own
> tooling.

{% details Click here for the answer. %}

To write, with the dense-versus-sparse factor of 2 stated explicitly and the
resulting ridge point given as a number.

{% enddetails %}

**Question 2:** The same ridge point in int8 and in fp8.

> **To write.** The purpose is to watch the ridge point move: halving the bytes per
> element while doubling the FLOP rate does not leave the crossover where it was.
> This is the seed of the fp8 argument in
> [Chapter 6]({{ '/pages/training' | relative_url }}), where changing precision turns
> out to move every inequality in the chapter.

{% details Click here for the answer. %}

To write.

{% enddetails %}
