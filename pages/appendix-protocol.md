---
layout: distill
title: "Appendix B: How We Measure"
description: "The protocol behind every measured number in this book: the exact software stack, warmup and repeat counts, clock and power state, and device count and partitioning mode. Plus what the analytical tag means and what it would take to remove it."
date: 2026-08-04

section_label: "Appendix B"

previous_section_url: "/pages/appendix-install"
previous_section_name: "Appendix A: Installing"

next_section_url: ""
next_section_name: "End of the book"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: The Software Stack
  - name: Warmup and Repeats
  - name: Clocks and Power
  - name: Device Count and Partitioning Mode
  - name: What Analytical Means Here
---

> **Skeleton.** Section structure only; the versions, the counts and the settings are still to be
> written. The brief for this appendix is the Conventions and Appendices sections of
> `docs/structure.md`.

**Depends on:** nothing. Every **[measured]** number in the book links here, so this page has to
stand alone and stay short enough that following the link is not a punishment.

> **To write.** Open with why this page exists, in two sentences. "We measured 370 GB/s" is an
> anecdote; the same number with a stack and a method behind it is evidence. **This is the
> convention that decides whether the book's central claim survives contact with a skeptical
> reader**, so it is worth fixing all of it before the first measurement is published rather than
> reconstructing it afterwards.

## The Software Stack

> **To write.** ROCm version, JAX, jaxlib, the two plugin wheels, and the container tag if one was
> used.
>
> **Prefer quoting a container tag over a list of wheel versions**, because it is one string and a
> reader can actually reproduce it. AMD's prebuilt JAX images are the obvious baseline and a better
> target than "stock wheels" precisely because they pin everything at once.
> [Appendix A]({{ '/pages/appendix-install' | relative_url }}) has the install detail; this page
> only records what was used.

## Warmup and Repeats

> **To write.** How many iterations discarded, how many measured, and whether the reported figure
> is the median or the mean.
>
> **Pick the median and say so.** Autotuning and clock ramp make the first iterations useless and
> make the mean misleading, and a reader who knows we report medians can compare against their own
> numbers correctly.

## Clocks and Power

> **To write.** An MI300X at 750 W will throttle under a sustained matmul, so a reader comparing
> against a boost-clock roofline will see a gap that has nothing to do with their code.
>
> If we lock clocks, say so and give the command. If we do not, say that too, and expect to explain
> a few percent. Either answer is fine; silence is not.

## Device Count and Partitioning Mode

> **To write.** The device count and the SPX or NPS partitioning mode for every measurement.
>
> This is not bookkeeping: [Chapter 3]({{ '/pages/profiling' | relative_url }})'s limitations table
> includes a row about op times being summed across devices, so **a number without a device count
> next to it is unreadable**, and partitioning mode changes the CU count a process sees, per
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}).

## What Analytical Means Here

> **To write.** The other half of the tagging convention, and the reason a reader should trust the
> **[measured]** half.
>
> **[analytical]** means derived from published specifications and not checked against hardware.
> Two categories are analytical throughout the book and it is better to say so in one place than to
> repeat it in six chapters:
>
> - **Anything inter-node**, because we have no multi-node allocation. Affects
>   Chapters [2]({{ '/pages/amd-gpus' | relative_url }}),
>   [6]({{ '/pages/training' | relative_url }}),
>   [7]({{ '/pages/moe' | relative_url }}) and
>   [12]({{ '/pages/serving' | relative_url }}).
> - **Anything inference-side**, by design, because we do not run a serving stack. Affects
>   Chapters [11]({{ '/pages/inference' | relative_url }}) and
>   [12]({{ '/pages/serving' | relative_url }}).
>
> For each, say what measurement would remove the tag. A reader who knows exactly what is missing
> trusts what is present, and it keeps us honest about which gaps are real constraints and which
> are just work we have not done.
