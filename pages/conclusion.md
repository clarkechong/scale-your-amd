---
layout: distill
title: "Conclusions and Further Reading"
description: "What we got wrong, what is one level down from where this book stops, and where to go for the parts we deliberately did not cover. Plus acknowledgements."
date: 2026-08-04

section_number: 13

previous_section_url: "/pages/serving"
previous_section_name: "Chapter 12: Production"

next_section_url: "/pages/appendix-install"
next_section_name: "Appendix A: Installing"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: What We Got Wrong
  - name: One Level Down
  - name: Where to Go for Serving
  - name: Further Reading
  - name: Acknowledgements
---

> **Skeleton.** Section structure only. The brief for this chapter is the Chapter 13 section of
> `docs/structure.md`.

**Depends on:** nothing, and it should stay that way. This chapter is readable on its own by a
reader deciding whether to start the book.

## What We Got Wrong

> **To write.** Written last, and worth being specific in: predictions that did not hold,
> constants that turned out to be wrong, and any place where the tooling changed under us between
> writing and publishing. A book with a section like this is trusted more than one without, and
> the cost of writing it is one honest afternoon.
>
> This is also the right home for the promises the book scoped down: what is still
> **[analytical]** that we would rather have measured, and what it would take to fix that.

## One Level Down

> **To write.** This book stops at the kernel boundary by design. For the reader who wants to
> cross it, frame the next tier as **"one level down" rather than "further reading"**, because that
> tells them what kind of question the material answers.
>
> The ROCm blog's
> [occupancy math post](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html)
> is the best entry point we know of, along with the CDNA4 ISA guide and the CDNA4 architecture
> whitepaper it cites.

## Where to Go for Serving

> **To write.** Since the book deliberately stops at the handoff, list the vLLM and SGLang ROCm
> documentation, AMD's inference optimization guides, and AITER, **as the genuine next step rather
> than as consolation.**
>
> A reader who trained a model with this book and then serves it well with someone else's has been
> served correctly, and saying that plainly is better than apologising for the scope.

## Further Reading

> **To write.** ROCm documentation, the source book, the HuggingFace Ultra-Scale Playbook, and
> Stas Bekman's ML Engineering handbook. One line each on what it is good for, since an unannotated
> list of links is worth very little.

## Acknowledgements

> **To write.** Name the people consulted. This is both correct and a credibility move, and it is
> the one section that should not be written in a hurry.

> **To write.** Consider closing with a short "if you only remember five things" list. The
> candidates are the ridge point from
> [Chapter 1]({{ '/pages/rooflines' | relative_url }}), the 1307 TFLOP/s and 5.3 TB/s pair from
> [Chapter 2]({{ '/pages/amd-gpus' | relative_url }}), the eight-GPU cliff from
> [Chapter 4]({{ '/pages/sharding' | relative_url }}), the `E / E_a` inflation from
> [Chapter 7]({{ '/pages/moe' | relative_url }}), and the decode step-time formula from
> [Chapter 11]({{ '/pages/inference' | relative_url }}). If the book has done its job, a reader
> should already agree with the list.
