---
layout: distill
title: "Appendix A: Installing JAX on ROCm"
description: "The container path, the wheel path, the version matrix, and the combinations known to be broken. Kept out of the chapters because it is necessary, it is nobody's reason for reading the book, and it rots faster than anything else here."
date: 2026-08-04

section_label: "Appendix A"

previous_section_url: "/pages/conclusion"
previous_section_name: "Chapter 13: Conclusions"

next_section_url: "/pages/appendix-protocol"
next_section_name: "Appendix B: How We Measure"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: The Container Path
  - name: The Wheel Path
  - name: The ROCm Version Matrix
  - name: Building From Source
  - name: Known-Broken Combinations
---

> **Skeleton.** Section structure only; the commands, the versions and the tables are still to be
> written. The brief for this appendix is the Appendices section of `docs/structure.md`.

**Depends on:** nothing. This is a reference page and readers arrive at it directly from
[Chapter 3]({{ '/pages/profiling' | relative_url }}) or from a search result.

> **Verified against.** Every version and command on this page needs a date and the exact
> versions it was checked with. This appendix exists precisely because this material rots, so an
> undated page here is worse than no page.

## The Container Path

> **To write.** The recommended route, and the one the book measures against. AMD's prebuilt JAX
> and MaxText images: which tag, what is inside it, the `docker run` invocation with the device
> and group flags that ROCm needs, and the verification snippet from
> [Chapter 3]({{ '/pages/profiling' | relative_url }}).
>
> **Prefer quoting a container tag over a list of wheel versions**, because it is one string and it
> is actually reproducible by a reader. This is also what
> [Appendix B]({{ '/pages/appendix-protocol' | relative_url }}) records for every measurement in
> the book.

## The Wheel Path

> **To write.** For readers who cannot use containers. The four wheels in the right order, which is
> the part people get wrong:
>
> 1. `jax-rocm7-pjrt`
> 2. `jax-rocm7-plugin`
> 3. `jaxlib`
> 4. `jax`
>
> Explain why ROCm ships two plugin wheels rather than one, briefly, and cross-reference
> [Chapter 3]({{ '/pages/profiling' | relative_url }})'s thousand-foot view for where the PJRT
> plugin actually sits in the stack. `gpu_profiling/readme.md` is the source for this section.

## The ROCm Version Matrix

> **To write.** Which JAX version needs which ROCm version, as a table, with the tested
> combinations marked as tested rather than implied. Include the ROCm versions available in the
> containers above so a reader can tell which row they are on.

## Building From Source

> **To write.** For the reader who needs a patch that is not in a release yet. Keep it to the
> shortest working recipe and say plainly that this is the slow path.

## Known-Broken Combinations

> **To write.** The combinations that install cleanly and then fail at runtime, which are the
> expensive ones to discover yourself. Each row: the combination, the symptom, and either the fix
> or the version to avoid.
>
> This is the sibling of
> [Chapter 3]({{ '/pages/profiling' | relative_url }})'s limitations table and should follow the
> same discipline: dated, version-pinned, and honest about which entries we expect to disappear.
