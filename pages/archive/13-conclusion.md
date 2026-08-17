---
layout: distill
title: "Conclusions and Further Reading"
description: "What we got wrong, what is one level down from where this book stops, and where to go for the parts we deliberately did not cover. Plus acknowledgements."
date: 2026-08-04

section_number: 13

previous_section_url: "/pages/12-serving"
previous_section_name: "Chapter 12: Production"

next_section_url: "/pages/a-appendix-install"
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

> **Draft.** "What we got wrong" is written last by construction, and the
> acknowledgements need names. The three reading-path sections are written, because they
> are useful the moment they exist.

**Depends on:** nothing, and it should stay that way. This chapter is readable on its own by a
reader deciding whether to start the book.

## What We Got Wrong

<!-- BLOCKED by construction: this section is written last, and it depends on every
     measurement in the book having been taken.

     What it has to deliver: predictions that did not hold, constants that turned out to
     be wrong, and any place where the tooling changed under us between writing and
     publishing. A book with a section like this is trusted more than one without, and
     the cost of writing it is one honest afternoon.

     Candidates already visible while the book is being written, which should be checked
     against reality before being asserted here:
       - The 5.3 TB/s versus 2479.6 GB/s HBM discrepancy in Chapter 2. Already written up
         there as a worked example rather than quietly corrected, which is the right
         treatment, but this section should record that we initially carried the wrong
         number.
       - Chapter 6's tensor-parallel roofline swings by 7x depending on whether RCCL
         achieves a mesh-optimal all-reduce or a single ring. Whichever way the
         measurement lands, one version of that section will have been wrong.
       - Chapter 7's kernel-reachability finding is a source reading, not a measurement.
         If the measurement contradicts it, say so here loudly.
       - The scoped measurement promise itself: the book promises that every single-node
         training roofline is checked against a captured profile. Record honestly how much
         of that was achieved.

     This is also the right home for the promises the book scoped down: what is still
     analytical that we wanted measured, and what a reader should therefore distrust. -->

## One Level Down

**This book stops at the kernel boundary, deliberately, and that is a real reading path
rather than an absence.** Everything here answers a different kind of question from the
ones this book answers: not "which parallelism strategy" but "why is this specific kernel
not at peak".

**Framing them as the next level down rather than as further reading is more useful**,
because it tells you what kind of question they answer.

- **[Occupancy Math on the AMD MI355X GPU (CDNA4)](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html)**
  is the best entry point we know of. It derives occupancy from first principles, shows
  the four limiters, and, most usefully, demonstrates with measurements that a
  matrix-bound kernel holds 97% of peak at 12% occupancy.
  [Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) leans on it heavily and
  does not reproduce it.
- **[The AMD CDNA 4 architecture whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)**
  is where the resource budgets come from: the register file, the LDS change from 64 KB to
  160 KB, and the matrix-core throughput per clock that
  [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }})'s FLOP table is built on.
- **[The CDNA 4 instruction set architecture reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)**
  is the level below that: the actual MFMA instructions, their operand layouts, and the
  register allocation rules. This is where you go when you have decided to write the
  kernel that
  [Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) tried to talk you out of.
- **[ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/)**
  is the instrument for that level, giving cache hit rates, MFMA utilization, LDS bank
  conflicts and memory coalescing, none of which XProf can show you.

## Where to Go for Serving

**This book stops at the handoff on purpose, so this is the genuine next step rather than
a consolation.** A reader who trained a model with this book and then serves it well with
someone else's has been served correctly.

- **[vLLM](https://docs.vllm.ai/)**, and specifically its ROCm installation and AMD
  optimization pages. This is where
  [Chapter 12]({{ '/pages/12-serving' | relative_url }})'s checkpoint ends up, and its
  documentation is the reference for the scheduling, paging and speculative-decoding
  features that chapter describes as properties of the workload.
- **[SGLang](https://docs.sglang.ai/)**, the main alternative, with strong AMD support and
  a different scheduler design. Worth reading alongside vLLM rather than instead of it,
  because the two make different trade-offs on prefix caching and structured output.
- **[AMD Instinct inference optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/index.html)**,
  AMD's own guidance: which backends to enable, which environment variables matter, and
  the tuning that is specific to Instinct hardware rather than generic to the engine.
- **[AITER](https://github.com/ROCm/aiter)**, where AMD's fast kernels live. Relevant to a
  serving reader because the engines above use it, and relevant to a JAX reader because
  [Chapter 7]({{ '/pages/7-moe' | relative_url }})'s central open question is how much of it
  is reachable from JAX.

## Further Reading

**Four places to go next, chosen because each one covers something this book deliberately
does not.**

- **[How To Scale Your Model](https://jax-ml.github.io/scaling-book/)** (Google DeepMind).
  The book this one is a response to, and the best treatment of the same material on TPU.
  Read it for the parts we compressed: its treatment of TPU topology and its inference
  chapter are both more thorough than ours, and its arithmetic is the same arithmetic.
- **[The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)**
  (Hugging Face). More empirical and more NVIDIA-shaped than this book, with a great deal
  of measured data on parallelism configurations. The natural complement: we derive
  thresholds, they sweep them.
- **[Machine Learning Engineering](https://github.com/stas00/ml-engineering)** (Stas
  Bekman). The operational half that no roofline covers: what actually goes wrong on a
  large cluster, how to debug a hang, how to survive hardware failures over a
  months-long run. If [Chapter 9]({{ '/pages/9-llama' | relative_url }})'s sections on
  checkpointing and the input pipeline were a book, this would be it.
- **[ROCm documentation](https://rocm.docs.amd.com/)** (AMD). The reference for everything
  this book treats as a given, including the hardware documentation
  [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) is built from and the compatibility
  matrices [Appendix A]({{ '/pages/a-appendix-install' | relative_url }}) points at.

## Acknowledgements

<!-- BLOCKED: needs names, and needs asking the people involved whether they want to be
     named. Not a research problem, just something that cannot be invented.

     Who belongs here, from the roadmap and from the work the book rests on: the people
     consulted on the ROCm profiler internals, whoever reviews the drafts, and the AMD
     teams whose published work this book cites heavily rather than reproduces (the ROCm
     blogs in particular).

     Naming people is both correct and a credibility move, so this should not be
     skipped. -->
