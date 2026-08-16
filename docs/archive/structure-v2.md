# Roadmap v2

Planning document for *How To Scale Your Model with AMD*. Not published; this is the
working outline the chapters get written against. Voice and structure rules live in
`CLAUDE.md` at the repo root and are not repeated here.

**This supersedes `docs/structure.md`, which should be frozen rather than edited.**
V1 remains the detailed section list for the chapters marked *carried* below, and
those bullets are still current. What v2 changes is the book's objective, its centre
of gravity, and the chapter list. Where the two disagree, v2 wins; where v2 is silent
on the interior of a carried chapter, v1 is the reference.

## Contents

- [Why v2 exists](#why-v2-exists)
- [What this book is](#what-this-book-is)
- [What changed from v1](#what-changed-from-v1)
- [Conventions](#conventions)
- [Notation](#notation)
- [Chapter 0 — the landing page](#chapter-0--the-landing-page)
- [Part I — Preliminaries](#part-i--preliminaries)
- [Part II — The configuration surface](#part-ii--the-configuration-surface)
- [Part III — Running it](#part-iii--running-it)
- [Part IV — After training](#part-iv--after-training)
- [Appendices](#appendices)
- [File table and renumbering](#file-table-and-renumbering)
- [Existing assets](#existing-assets)
- [Sequencing](#sequencing)
- [Decisions defaulted in this draft](#decisions-defaulted-in-this-draft)
- [Open questions](#open-questions)

---

## Why v2 exists

**V1 optimizes for the wrong thing, and it optimizes for it very well.** Its thesis
is "given a model and some number of MI300X-class GPUs, how do I run it in JAX so
that adding GPUs adds throughput," and every measurement in it exists to confirm a
roofline prediction. That produces a book about whether the hardware is being used
well.

The reader we actually have is a customer standing up a frontier training run on our
machines. They do not open with "is my MFU defensible." They open with "I want to
train in fp8, can I, what will it cost me, and what else is available." Those are
configuration questions, and v1 has no home for most of them: fp8 is one bullet
inside the parallelism chapter, remat is one bullet inside the Transformer-math
chapter, the entire `XLA_FLAGS` surface is two flags inside a section about overlap,
Shardy is absent, and the attention-backend choice is one bullet inside a tuning
chapter.

V1 saw this failure mode coming and named it one level down. Its own warning was that
the gravity well is tooling, and that "profiling is the instrument, not the subject."
The same critique now applies one level up. **Rooflines are also the instrument. The
subject is the configuration decision, and the thing being optimized is time to a
target loss.**

---

## What this book is

**One sentence: given a model, a token budget and some number of MI300X-class GPUs,
what is every knob available to me in JAX on ROCm, what does each one buy, what does
each one cost, and how do I turn it on?** Everything serves that question. A chapter
earns its place by owning part of the answer.

**The objective function, stated once and used everywhere:**

```
time to target  =  tokens to target  /  tokens per second
```

The denominator is systems: parallelism, precision, kernels, compiler flags, overlap,
pipeline bubbles, the input pipeline, checkpoint overhead, and the restart rate when
something dies. The numerator is numerics and optimization: precision again, batch
size, optimizer, architecture.

**Precision and batch size appear in both, with opposite signs, and that is the whole
reason this book is hard.** fp8 doubles the FLOP ceiling and may cost you tokens to
target. A larger global batch raises tokens per second right up until it stops buying
convergence. V1 had both halves and never joined them: it disambiguated the two
meanings of "critical batch size" in Chapter 1 and noted in Chapter 6 that the global
batch is bounded below by the hardware ridge point and above by the convergence limit.
That seam is now the spine of the book rather than a parenthetical.

Five commitments follow.

**Every chapter terminates in a decision, not in an understanding.** The reader
should be able to close any chapter and change a config file. Each one ends with a
decision table: the knob, what it buys with a number attached, what it costs, the
exact config field or flag that sets it, and its status on ROCm with a date. Those
tables aggregate into Appendix C, which is the page a customer will actually keep
open.

**Standalone.** Unchanged from v1, and still load-bearing. A reader who has never
opened the TPU book can read this one cover to cover. We pay full price for
rooflines, sharding notation and Transformer math rather than borrowing them.

**JAX throughout, MaxText where a real model is needed.** Unchanged. One sharding
vocabulary, one profiler, one compiler to reason about. AMD maintains a ROCm MaxText
fork with documented multi-node training on MI300X through MI355X, which is what
makes the configuration advice concrete rather than notional.

**Predict, then measure, and say which evidence you have.** Also unchanged, and now
carrying more weight, because a configuration recommendation with no number behind it
is just an opinion with formatting. The tag system in [Conventions](#conventions)
grows a third tag to cover convergence claims we cite rather than run.

**Throughput claims and convergence claims are different claims and never share a
sentence.** This is new and it is the thing most likely to embarrass us. "fp8 is 2x"
is a throughput claim we can measure in an afternoon. "fp8 trains" is a convergence
claim that costs machine-weeks or a citation. Every precision section states both,
separately, with its own tag.

**Non-goals.** Not a CDNA microarchitecture reference, not a HIP kernel-writing
tutorial, not a ROCm installation manual, and not a numerical-optimization textbook:
we tell the reader what a precision recipe does to their loss curve, not how to
design a new one.

---

## What changed from v1

Recorded explicitly, because the pull back toward v1's shape will be strong and every
one of these was argued for.

1. **The objective changed** from roofline agreement to time to target loss, and the
   headline number changed from MFU to tokens per second per GPU. MFU is demoted to a
   diagnostic: it is how you find out *why* tokens per second is low.
2. **Three chapters were added** that v1 had no home for: numerics and precision (6),
   kernels you can actually reach (10), and compiler and runtime flags (11).
3. **Two v1 bullets were promoted to chapters**: memory and recompute (8), which was
   a bullet in v1's Chapter 5 and half a bullet in its Chapter 6, and operating a
   training run (13), which was three paragraphs bolted onto the dense capstone.
4. **Profiling shrank twice.** V1's Chapter 3 keeps its first-trace section and its
   limitations table; the XProf tool tour moves to Appendix D. V1's Chapter 8 becomes
   a triage playbook, and its `rocprofv3`, rocprof-compute and occupancy material
   moves to Appendix D as well. This is where the page budget for the new chapters
   comes from.
5. **Shardy was added** to the partitioner discussion in Chapter 4, which previously
   covered only GSPMD versus `shard_map`.
6. **Numerics moved ahead of parallelism**, so that Chapter 7's decision procedure can
   take precision as an input. V1 explicitly wanted this and could not have it while
   fp8 lived inside Chapter 7 itself.
7. **Multi-node moved onto the critical path.** V1 scoped the measurement promise to
   single-node and marked everything else analytical, which is defensible for a
   scaling textbook and is not defensible for a document telling a customer how to run
   a frontier training job. See [Open questions](#open-questions); this is the largest
   unfunded liability in the plan.
8. **Serving is unchanged and stays last.** Two chapters, honest rather than deep, for
   the reasons v1 gives. Nothing in the reframe touches that argument.

---

## Conventions

Everything in v1's Conventions section still applies: one Markdown file per chapter in
`pages/`, `layout: distill`, front matter chaining, the five mechanical rules that the
build will not catch, `tools/check_links.py` after every structural edit, worked
problems with answers behind `{% details %}`, and dated "verified against" lines under
every spec table. Re-read that section before writing; it is unchanged and it is
correct.

Five conventions are new or amended.

**Tokens per second per GPU is the headline number. MFU is a diagnostic.** Every
measured result leads with tokens per second per GPU, because that is the number a
customer compares across vendors and the number that divides into their token budget.
MFU follows in parentheses, because that is the number that tells you whether the
first one has room left in it. V1 quoted MFU alone, which answers a question the
reader did not ask.

**Three evidence tags, not two.** `[measured]` means a number we read off our own
hardware with Appendix B behind it. `[analytical]` means derived from published specs
and not checked. `[cited]` is new: it means someone else's measurement, named
inline with a link and a date, and it exists because the alternative for convergence
claims is either silence or dishonesty. A reader must never have to guess which of
the three a precision recommendation rests on.

**Every chapter ends in a decision table.** Fixed columns: knob, what it buys, what it
costs, how to set it, status on ROCm, verified on. "How to set it" is the literal
MaxText config field, `XLA_FLAGS` entry or environment variable, not a description of
one. "Status" is one of *works*, *works with caveats*, *experimental*, or *absent*,
and *absent* is a perfectly good answer that customers need more than they need
encouragement.

**Software status carries a date and a version, everywhere, without exception.** V1
applied this to the profiling limitations table and the spec tables. In v2 most of the
book is status claims about a fast-moving stack, so the discipline generalizes or the
book rots in a quarter. Any chapter whose decision table has no date is not finished.

**Config keys keep their own names.** Carried from v1 and now much more load-bearing:
no symbol for capacity factor, write `capacity_factor`, because that is the field the
reader edits. Same for every flag and every environment variable.

---

## Notation

V1's notation table is unchanged and stays in `_includes/notation.liquid`: `B`, `T`,
`S`, `D`, `F`, `N`, `K`, `H`, `L`, `V`, `E`, `E_a`, `C`, `β`, mesh axes `X`, `Y`, `Z`,
`Ex`, sharded arrays written `A[I_X, J_Y]`, times lowercase, bandwidth Greek. The four
collision fixes it argues for are all still right.

Three additions, for the objective function:

| Symbol | Meaning |
|---|---|
| `tps` | tokens per second per GPU, the headline throughput number |
| `n_target` | tokens to reach the target loss |
| `t_target` | time to target loss, `n_target / (tps * devices)` |

**Do not invent a symbol for MFU or for convergence loss.** Write them out. They
appear in prose far more often than in algebra, and a reader who has to decode a
symbol in a sentence about loss curves is being taxed for nothing.

---

## Chapter 0 — the landing page

`index.md` · Carried from v1 with a new opening and new reading paths.

The four elements v1 specifies are still right: the thesis, the expected background,
what the book adds and does not, and the reading paths. Three changes.

**The thesis is now the objective function**, stated as the equation above, followed
by the strong-scaling framing v1 opens with. The equation is the single most useful
thing on the page: it tells a customer in one line why the book contains both a
chapter on collective bandwidth and a chapter on loss curves.

**The evidence promise is stated here, in three sentences**, not buried in an
appendix: what we measured, what we derived, what we cite, and the fact that every
claim in the book says which it is.

**The reading paths become customer-shaped.** V1's four routes were organized by
model type and by tooling. Replace with routes organized by the decision the reader
arrived holding:

| Route | Chapters |
|---|---|
| I want to train in fp8 or lower | 1, 2, 5, 6, 8, 10 |
| I want to pick a parallelism strategy | 1, 2, 4, 5, 7, 8 |
| I am training an MoE | 1, 2, 4, 5, 7, 9, 10 |
| My run is slower than it should be | 3, 11, 12 |
| I am standing up a multi-node run | 4, 7, 11, 13 |
| I need to size and ship what I trained | 1, 2, 5, 16, 17 |
| Just give me the knobs | Appendix C |

That last row is not a joke and should be printed. A large fraction of the audience
wants the support matrix and nothing else, and sending them there directly buys more
goodwill than making them find it.

---

## Part I — Preliminaries

Everything needed before a configuration decision means anything. All four chapters
are independent of MaxText and of any large model.

### Chapter 1 — All About Rooflines

`1-rooflines.md` · **What actually limits how fast this runs?**

Carried from v1 essentially intact: three bounds, arithmetic intensity, the critical
batch size derived symbolically, communication rooflines, and the spec-sheet
skepticism exercise. The section list in v1 is still the one to write against.

**One addition, and it is the chapter's new job: introduce the objective function.**
After the three bounds and before arithmetic intensity, state
`time to target = tokens to target / tokens per second`, and say plainly that most of
the book is about the denominator and that only two things in it move the numerator:
precision in Chapter 6 and batch size in Chapter 8.
This is also where v1's careful disambiguation of the two
meanings of "critical batch size" pays off, so keep that paragraph and point it
forward at Chapter 8's gradient-accumulation section, where the two bounds finally
meet.

### Chapter 2 — How to Think About AMD GPUs

`2-amd-gpus.md` · **How fast should this run on an MI300X, and what in the hardware
decides that?**

Carried from v1 intact, including the full-mesh topology discussion, the eight-GPU
ceiling, the family table across MI300X, MI325X and MI355X, the translation table
against H100 and TPU v5e, and the HBM bandwidth discrepancy written up as a worked
example rather than quietly corrected.

**One section grows: the dtype table becomes the reference for the whole book.** V1
listed fp32, tf32, bf16, fp16, fp8 and int8, with the gfx942 NANOO/FNUZ versus gfx950
OCP split and the CDNA4 oddity that MXFP6 runs at MXFP4 rate. Keep all of it and add
the microscaled formats properly: MXFP8, MXFP6, MXFP4, block size, where the scale
factors live, and what the hardware does natively versus what is emulated. Chapter 6
then spends this table rather than rebuilding it, and Chapter 16 spends it again for
inference.

Stay hardware-side here. **What the formats do to a loss curve is Chapter 6 and must
not leak into Chapter 2**, or the two will disagree within a quarter.

### Chapter 3 — How to Profile AMD GPU Programs

`3-profiling.md` · **Your model is slower than the arithmetic says it should be.
Where did the time go?**

**Deliberately shrunk, and this is the chapter where v2 buys its page budget.** V1
called this the biggest length risk in the book and it was right. Keep four things:
the thousand-foot view of the stack, setup in about fifteen lines against a container,
your first trace, and the limitations table with its stable anchor and its pinned
versions. Keep the training-step section that introduces `StepTraceAnnotation`,
because everything downstream depends on step markers existing.

**Move the XProf tool tour to Appendix D.** The one-tool-at-a-time walkthrough with
two dozen screenshots is reference material, it is the single largest unplanned cost
in the book, and it invalidates every time the UI changes. A reader who needs it will
follow a link. A reader who does not is currently paying for it in the middle of the
argument.

The limitations table stays exactly as v1 specifies, seven curated rows with a fourth
"fix in progress" column, dated and version-pinned, and framed as a candid status
report rather than an apology. **It remains the single most-linked thing we will
publish**, and it is now also the template for every decision table in Part II.

### Chapter 4 — Sharding, Collectives and Partitioners

`4-sharding.md` · **You split the matrix across eight GPUs. What does that cost, and
who decides?**

Carried from v1, which is strong here: named-axis notation, the four collectives
derived against the full mesh, RCCL in practice, the measured-versus-spec bandwidth
sweep, the four sharded-matmul cases, and the multi-process section that explains why
mesh axis order decides which collective crosses the NIC. The bandwidth sweep is still
the most novel measurement in the book and still justifies the chapter on its own.

**One section is rewritten and one is added.**

The rewrite is **who inserts the collective**. V1 framed this as GSPMD versus
`shard_map`, which is now incomplete: Shardy is where JAX partitioning has gone, and
the old `xla_use_shardy` escape hatch has been removed from `xla.proto` rather than
merely defaulted, so this is not a knob the reader can opt out of by the time they
read us. Present three levels, not two: annotate and let the partitioner derive the
collectives, take the per-device view with `shard_map` and write them yourself, or
constrain the partitioner's choices with sharding annotations in between. Show the
same sharded matmul each way and diff the HLO. **Confirm the current default and the
flag names against the exact jaxlib we pin**, because this specific area has moved
twice in a year and a wrong sentence here is the kind readers screenshot.

The addition is **ragged collectives**, one short section, placed here because it is a
collective and not an MoE trick. XLA has a `ragged-all-to-all` op with its own
decomposer and one-shot-kernel paths, currently behind flags carrying the
`unsupported_` prefix. That is exactly the mechanism Chapter 9's dropless dispatch
needs, so establishing here that it exists, that it is gated, and that its ROCm status
is a thing we must check, saves Chapter 9 from having to introduce a collective and a
kernel-availability argument in the same breath.

---

## Part II — The configuration surface

**The new centre of the book.** Six chapters, one per family of knobs, each answering
the same three questions: what does this buy, what does it cost, and can I actually
have it on ROCm today. Read in order they escalate from model-level decisions that a
customer makes before the run starts, to implementation-level decisions they make
while it is running.

### Chapter 5 — All the Transformer Math You Need

`5-transformers.md` · **How many parameters, FLOPs and bytes, exactly?**

Carried from v1 intact: the counting rule, per-layer accounting, the `T > 8D` result,
MoE accounting alongside dense accounting rather than in an appendix, the KV cache and
the MHA to MQA to GQA to MLA sequence, and the MFU-versus-HFU distinction.

**Two deletions, both to chapters that now exist.** Gradient checkpointing moves to
Chapter 8, where it sits with the rest of the memory story instead of being stranded
between the KV cache and MFU. The MFU section stays here, because it is accounting,
but it now opens by saying that MFU is a diagnostic for tokens per second rather than
a goal, which is the framing the rest of the book uses.

### Chapter 6 — Numerics and Precision

`6-numerics.md` · **You want to train in fp8. Can you, what does it buy, and what does
it cost you in convergence?** ***New chapter, and the one a customer opens first.***

This is the largest single addition in v2 and the clearest hole in v1, where the
entire subject was one bullet framed as a perturbation of the parallelism
inequalities. That framing is true and it is a third of the story.

- **The format zoo, from the training side.** Chapter 2 gave the hardware table; this
  is what each format is for. fp32 and where it survives, tf32, bf16 as the baseline
  everything is measured against, fp16 and why loss scaling exists, fp8 in its e4m3
  and e5m2 roles, then the microscaled formats: MXFP8, MXFP6, MXFP4, what a block
  scale is, and why block scaling exists at all. Lead with the observation that makes
  the chapter tractable: **you are never training in one format, you are choosing a
  per-tensor assignment**, and the interesting question is always which tensors stay
  high precision.
- **The anatomy of a precision recipe.** The five decisions, each with the arithmetic
  for what it saves: activation, weight and gradient precision; master weight
  precision; optimizer state precision; the scaling strategy, per-tensor with delayed
  scaling versus per-block; and the exception list, which for essentially every recipe
  in production includes the router, the norms, the loss and the accumulate.
- **The gfx942 versus gfx950 format split, spent rather than re-explained.** Chapter 2
  named NANOO/FNUZ versus OCP. Here it becomes a practical problem: the same recipe is
  not the same recipe on the two parts, MaxText's ROCm fork exposes `nanoo_fp8` for
  gfx942 and standard `fp8` for gfx950, and a checkpoint carrying scales moves between
  them badly. Chapter 17 hits the same wall from the serving side.
- **What it buys, measured.** Tokens per second per GPU for a fixed model and mesh
  across bf16, fp8 and where the hardware allows it MXFP8 and MXFP4, with MFU
  alongside. This is the cheap half of the chapter: same workload, change one config
  field, capture a trace. Expect the gap between the 2x FLOP ceiling and the achieved
  speedup to be the most instructive number in the chapter, and explain it from the
  profile rather than apologizing for it.
- **What it costs, in convergence.** The expensive half, and the one that has to be
  scoped before anything is written. See
  [Decisions defaulted in this draft](#decisions-defaulted-in-this-draft): the plan of
  record is proxy-scale runs against a bf16 baseline at a fixed token budget, reporting
  the loss delta and the seed variance so the reader can tell a real regression from
  noise. Whatever we do, the tag is what matters: a `[cited]` convergence claim with a
  link is honest and useful, and an untagged one is the thing that gets us quoted back
  at ourselves.
- **How you would find out yourself.** The transferable skill and the correct ending:
  the short-horizon proxy protocol, what a precision-induced divergence looks like
  versus an ordinary bad run, which quantities to watch (loss delta against baseline,
  gradient norm, the fraction of tensors hitting the format's dynamic range limits),
  and when to stop and go back to bf16.

**Decision table.** Every format, what it buys in tokens per second per GPU, its
convergence status with a tag, the MaxText field that selects it, and whether it works
on gfx942, gfx950 or both.

**Worked problems.** Given the dtype table, compute the new ridge point in fp8 and say
what it does to the minimum per-device batch; given a measured 1.4x from a format with
a 2x ceiling, produce a ranked list of explanations; decide whether MXFP6 or MXFP4 is
the better choice on gfx950 given that they run at the same rate.

### Chapter 7 — How to Parallelize a Transformer for Training

`7-parallelism.md` · **You added seven more GPUs and got four times the throughput.
Where did the rest go?**

Carried from v1's Chapter 6 with its structure intact, which is the best-designed
chapter in the outline: five strategies, each with the same five-part treatment, the
critical-path annotation on every collective, the composition rules, the optimal
split, the sequence-versus-context parallelism disambiguation, and full pipeline
treatment including a roofline.

**Three changes.**

Precision is now an input rather than a section. V1 wanted the decision procedure to
take precision as an input and could not have it while the fp8 section lived inside
this chapter. Now Chapter 6 comes first, so each of the five rooflines is stated with
the FLOP ceiling as a parameter and the reader substitutes their own.

The memory material leaves for Chapter 8, which is a straight lift of v1's "memory,
not just time" bullet plus gradient accumulation.

**The chapter's product changes from an inequality to a config.** Each strategy's
five-part treatment gains a sixth part: the MaxText mesh and axis configuration that
implements it, written out. V1 derived beautifully and left the reader to translate.

### Chapter 8 — Memory, Recompute and the Optimizer

`8-memory.md` · **It does not fit. What do you give up to make it fit?** ***New
chapter, promoted from two bullets.***

Memory decides more real configurations than time does, and in v1 it was a bullet in
Chapter 5 and half a bullet in Chapter 6. It also happens to be where the largest
single-knob throughput swings in the book live.

- **The activation memory budget**, derived for a real model, layer by layer, so the
  reader can see which term dominates before choosing what to do about it.
- **Rematerialization properly, with the API surface.** V1 gave "two named policies
  with their FLOP costs." Give the actual JAX surface: `jax.checkpoint` / `jax.remat`,
  what a policy is, the named policies and what each saves and costs, custom policies
  by name, and the interaction with `scan` over layers that decides whether remat is
  per-layer or per-block. **This is the knob with the widest range in the book**: the
  span from no remat to full remat is a large fraction of both memory and step time,
  and the middle of that range is where every real run sits.
- **Offloading.** Host offload of activations and optimizer state, what PCIe Gen 5 x16
  costs against the HBM bandwidth from Chapter 2, and the honest conclusion about when
  it pays. XLA has host-offloading and pipelined-host-offloading paths; check their
  ROCm status rather than assuming.
- **Optimizer state: sharding and precision.** What Adam actually stores, ZeRO stages
  as a memory story rather than a communication one since Chapter 7 covered the
  communication, and what happens when moments are kept in bf16 instead of fp32. This
  is also the natural place to note that an MoE's optimizer state scales with total
  parameters and not activated ones, which Chapter 9 then leans on.
- **Gradient accumulation as the lever that decouples memory from batch size**, moved
  from v1's Chapter 6. **And this is where the two critical batch sizes from Chapter 1
  finally meet**: the global batch is bounded below by the hardware ridge point and
  above by convergence, and the accumulation count is how you land between them
  without the memory of a large batch. That paragraph is the clearest statement in the
  book of why the objective function has two factors, so give it room.

**Decision table.** Each lever, memory saved, throughput cost, config field, status.

### Chapter 9 — Mixture-of-Experts at Scale

`9-moe.md` · **Only a fraction of the parameters run per token, so why isn't it a
fraction of the time?**

Carried from v1's Chapter 7, which is the most technically distinctive chapter in the
outline and needs very little structural change: the routing mechanism, load imbalance
and how to see it in a trace, capacity and dropping and going dropless, the three
implementations of an expert layer with the arithmetic for each, all-to-all dispatch
against the full mesh, expert parallelism opened from the memory side, the anatomy of
three real models, and the four numbers to log for every MoE run.

**Two amendments.**

The kernel-reachability argument moves to Chapter 10 and is cited from here. V1 made
"which of the three can a JAX user on ROCm actually run fast" the spine of this
chapter, which was right when there was nowhere else to put it. Now that Chapter 10
exists and owns kernel availability across attention and GEMM alike, this chapter
states the requirement (implementation 3 needs a grouped or ragged GEMM), gives the
FLOP arithmetic that says what it is worth, and points at Chapter 10 for whether you
can have it. **The finding is unchanged and stays prominent**; it just stops being
made twice.

The ragged-collective mechanism is cited from Chapter 4 rather than introduced here,
per that chapter's new section. Dispatch then reads as one question, what does the
collective cost, instead of two questions tangled together.

### Chapter 10 — Kernels You Can Actually Reach

`10-kernels.md` · **Vendor kernels exist. Can you call them from JAX?** ***New
chapter.***

V1 scattered this across a tuning section in Chapter 8, an MoE bullet in Chapter 7,
and an open question. For a customer it is a single question asked repeatedly, and the
answer is the difference between a competitive number and an embarrassing one.

- **How a JAX op becomes a kernel**, briefly: XLA-generated fusions, custom calls into
  hipBLASLt and rocBLAS, Triton through the XLA path, XLA FFI for external kernels,
  and Pallas. What each one costs you in portability and what it buys in speed.
- **Attention, as a chooser.** The comparison a customer asks for directly:
  Transformer Engine, `jax-aiter`, `jax-triton`, and XLA-native, on the same shapes.
  For each: does it exist on ROCm today, does it carry a gradient so it is usable in
  training rather than inference only, how it appears in a trace so the reader can
  confirm which one they actually got, and the measured tokens per second. **How to
  tell which implementation you got is as valuable as which is fastest**, because
  silently falling back is the common failure and it is invisible without this.
- **GEMM.** hipBLASLt heuristics and offline tuning, where rocBLAS still gets picked,
  the `xla_gpu_enable_cublaslt` switch which selects hipBLASLt on the ROCm backend,
  Triton GEMM paths, and real before-and-after numbers from tuning.
- **Grouped and ragged GEMM for MoE, which is the chapter's hard finding.** AMD's fast
  MoE kernels live in AITER and are reached from PyTorch; the JAX bridge exposes
  attention and dense GEMM. If it remains true that no grouped or ragged MoE GEMM is
  reachable from JAX, that is the single largest performance fact in Part II and it
  should be stated plainly, quantified against the AITER figures AMD publishes, and
  paired with the honest fallback ranking from Chapter 9. **Verify against current
  wheels immediately before writing**; this is the fastest-moving claim in the book.
- **Writing your own**, and when not to. Mostly not to. Pallas on ROCm goes through
  the Triton backend and is experimental; Mosaic GPU is NVIDIA-only. Say so with a
  date.

**Decision table.** Every kernel path, what it accelerates, its status, how to select
it, and how to confirm in a trace that you got it.

### Chapter 11 — Compiler and Runtime Flags

`11-flags.md` · **Two dozen environment variables stand between your config and your
step time.** ***New chapter.***

The entire `XLA_FLAGS` surface in v1 was two flags in a bullet about overlap. In
practice this is where a large fraction of achievable throughput lives, it is almost
entirely undocumented folklore, and it is unusually cheap to measure: same workload,
flag on, flag off, report the step time. **A measured flag table for JAX on ROCm is
something nobody has published and customers ask for constantly.**

- **How the surface is organized**, which the reader needs before any individual flag
  makes sense: `XLA_FLAGS` is read once at backend initialization, so setting it after
  the first JAX call silently does nothing. Then the optimization-level mechanism,
  because it changes what the individual flags mean: several collective optimizations
  that used to require explicit flags are now enabled by the effort level, so a reader
  copying a flag list from a 2024 blog post is setting things that are already on and
  concluding the flags do nothing.
- **Overlap and scheduling.** `xla_gpu_enable_latency_hiding_scheduler`, the async
  stream priority flag, and `xla_gpu_memory_limit_slop_factor` which governs how much
  memory the scheduler believes it has to play with. **Chapter 4 owns teaching what
  overlap is and how to see it in a trace; this chapter owns the flags and the
  numbers**, one cross-reference each way, and neither teaches the other's half.
- **Collective pipelining, combining and reordering**, which is the largest and least
  documented group. The boolean generation (`xla_gpu_enable_pipelined_all_reduce`,
  `..._all_gather`, `..._reduce_scatter`, `..._p2p`) is being superseded by an enum
  generation (`xla_gpu_pipeline_all_gather` and siblings, taking
  default/off/on/explicit), and `xla_gpu_enable_pipelined_collectives` is already
  deprecated. **Both generations will be live in the wheels customers are running, so
  the section has to name both and say which applies to which version.** Then the
  combiner thresholds, `xla_gpu_all_gather_combine_threshold_bytes` and its all-reduce
  and reduce-scatter siblings, which are the highest-value tunable in the group: the
  standard advice is to raise them until at least one Transformer layer's weights or
  gradients combine into a single collective, and Chapter 5's per-layer accounting
  gives the reader the number to set them to. That connection is worth making
  explicitly, because it turns a magic constant into an arithmetic result.
- **Command buffers, and the ROCm hazard we have already documented.** XLA can capture
  work into HIP graphs via `xla_gpu_enable_command_buffer`, and on ROCm this is the
  flag behind a flaky SIGSEGV we diagnosed ourselves: setting it empty falls back to
  standard kernel dispatch and works correctly. **This is the single best example in
  the book of why a flag chapter exists**, it is our own primary-source material rather
  than a repetition of NVIDIA guidance, and it is honest about the platform in exactly
  the way the limitations table is. There is also a separate flag governing whether
  command buffers run while profiling is active, which matters directly to anyone
  following Chapter 3.
- **Autotuning.** `xla_gpu_autotune_level` and what each level actually checks, plus
  persisting results across runs with the autotune cache flags, which matters because
  autotuning a large model is a real fraction of the first step's wall clock and is
  pure waste on every subsequent run.
- **RCCL environment variables**, treated as a short survey rather than a tour: which
  ones change algorithm selection, which change transport, and the one-line warning
  that AMD's own benchmark scripts disable a feature to avoid NaN losses on MI355X.
  That fact was a footnote in v1's capstone; it belongs here, where a reader tuning
  flags will actually meet it.

**Method, stated once at the top and enforced.** Every flag in the table gets the same
treatment: one workload, one flag changed, median of N steps after warmup, tokens per
second before and after, and the version it was measured on. Flags we did not measure
are listed as unmeasured rather than omitted or, worse, recommended on reputation.

**Decision table.** This chapter's table *is* the chapter, and it is the single most
likely page in the book to be linked from a customer email.

---

## Part III — Running it

Part II says what to configure. Part III is about the gap between a configuration and
a finished training run, and then about closing that gap twice on real models.

### Chapter 12 — Getting to Roofline

`12-getting-to-roofline.md` · **The prediction says 40% MFU. You measured 22%.**

Carried from v1's Chapter 8 but reduced to what it is best at, a triage playbook. The
triage order is unchanged and still correct, with host starvation first because it is
common and nobody looks for it, and with the honest note that the XProf Input Pipeline
page is one of the broken views so the reader needs the trace-viewer symptom instead.

**What leaves.** Kernel selection and tuning move to Chapter 10, where they are a
choice rather than a remedy. The deep-counter escalation, `rocprofv3`, rocprof-compute,
TraceLens and the ISA path, moves to Appendix D. The occupancy section moves to
Appendix D with it, keeping v1's excellent framing intact: low occupancy is usually
not the bug, the same roofline argument one level down, and the section's whole job is
to stop a JAX user chasing a number they cannot move.

**What stays and grows.** The triage order itself, fusion and how to read fusion
decisions from HLO, and a new closing section that turns the whole of Part II into a
checklist: given a gap, which chapter owns the likely cause. That section is the
book's index for a reader in trouble, which is when they are least able to find things.

### Chapter 13 — Operating a Training Run

`13-operations.md` · **The run has to survive a week, not a step.** ***New chapter,
promoted from three paragraphs in v1's dense capstone.***

Everything that decides whether a real run finishes, none of which any other chapter
owns, and all of which v1 had bolted onto the side of Chapter 9.

- **Multi-node launch**, concretely: `jax.distributed.initialize` against the mechanism
  Chapter 4 taught, what Primus does for MaxText on ROCm, and how a job is actually
  started on a cluster.
- **Checkpointing.** What a checkpoint costs in time and bytes at scale, Orbax's
  async and sharded save paths, how often to take one given a failure rate, and how
  long a restart takes. At frontier scale this is a first-order throughput term in the
  objective function and it belongs in the same book as the flag that buys 3%.
- **The input pipeline at scale.** How a sharded dataset feeds a multi-process mesh
  without every host reading the same shard, and what a deterministic resume costs.
  Chapter 12 taught the reader to recognise host starvation; this is where they see it
  prevented rather than diagnosed.
- **Failure and restart.** What actually dies, how to tell a hardware fault from a
  numerics divergence from a deadlock, and how the restart rate enters tokens per
  second as an availability factor. Be honest that our own evidence here is thin.
- **What to monitor**, as a single list a customer can implement: loss and gradient
  norm, tokens per second per GPU, step time distribution rather than mean, the four
  MoE numbers from Chapter 9 when applicable, and the format range-saturation
  statistics from Chapter 6 when training below bf16.

**This chapter is where the multi-node gap bites hardest.** Write the mechanism
regardless, mark the numbers `[analytical]`, and say plainly what we have not run.

### Chapter 14 — Training Llama 3 on MI300X

`14-llama.md` · The dense capstone. Carried from v1's Chapter 9 with the
checkpointing, input-pipeline and RCCL-NaN material removed to Chapter 13, which
leaves it to do the thing it is actually for: take a real model and justify every
configuration decision in Part II against the measured result. AMD's ROCm MaxText fork
ships Llama 3 8B and 70B as pre-optimised configurations with Llama 3.1 405B as a
documented multi-node benchmark, so the config is given and the chapter's value is
explaining why each degree was chosen.

**The capstones now have a fixed reporting format**, which they did not in v1: the
configuration in full, predicted tokens per second per GPU, measured, the gap
explained from a profile, and the decision table of what we changed and what each
change was worth. Chapter 15 reuses it exactly.

### Chapter 15 — Training DeepSeek-V2-Lite on MI300X

`15-deepseek.md` · The sparse capstone, same method, harder model. Carried from v1's
Chapter 10 unchanged in substance: routing and imbalance, expert parallelism,
all-to-all placement against the eight-GPU ceiling, MLA's effect on the memory
profile, and the four numbers from Chapter 9 earning their keep. Qwen3 30B-A3B is the
alternative and Mixtral 8x7B the simpler fallback, all three being in the same
pre-optimised list, so switching costs a re-run rather than a rewrite.

---

## Part IV — After training

Unchanged from v1 in both scope and argument. Serving is covered honestly rather than
deeply, it stays last, and the reasoning in v1 for why is still correct: there is no
JAX serving engine for ROCm, MaxText's own blessed inference path is a vLLM plugin,
and vLLM and SGLang on ROCm are genuinely good.

### Chapter 16 — How to Think About Inference

`16-inference.md` · **Serving is not training with `no_grad`.** Carried from v1's
Chapter 11 intact: the two regimes, the decode step-time formula, KV cache economics,
sharding for decode and why FSDP is actively harmful there, MoE at decode as the
hardest problem in the book, quantization as a bandwidth lever rather than a FLOP
lever, speculative decoding, and cost per token in dollars. Analytical throughout, and
the opening line says so.

One amendment: the quantization section now cites Chapter 6 for the formats and the
recipes and confines itself to what changes at decode, which is a cleaner split than
v1's, where Chapter 2 owned formats and Chapter 6 owned fp8 training and this chapter
owned inference levers.

### Chapter 17 — Getting Your Model Into Production

`17-serving.md` · **You trained it in JAX. Now it has to serve traffic, and that
probably isn't JAX.** Carried from v1's Chapter 12 intact: the handoff done
concretely, what the serving engine does with your weights, disaggregated prefill and
decode, and deployment sizing against the dollars-per-token model.

The handoff section gains one thing from Chapter 6: **the fp8 format split is a
checkpoint compatibility problem**, and a model trained with NANOO fp8 on gfx942 is
the concrete case to walk through when exporting through Quark to a vLLM deployment.

### Chapter 18 — Conclusions and Further Reading

`18-conclusion.md` · Carried from v1's Chapter 13, including the "one level down"
framing for kernel-level reading and pointing serving readers at vLLM and SGLang on
ROCm as the genuine next step. Add one closing section: what in this book will be
wrong first. Given how much of v2 is dated status claims, saying which ones we expect
to rot is both honest and useful.

---

## Appendices

Four, up from two.

**Appendix A — Installing JAX on ROCm.** `a-appendix-install.md`. Unchanged from v1:
the four wheels in the right order, the ROCm version matrix, building from source, and
known-broken combinations.

**Appendix B — How We Measure.** `b-appendix-protocol.md`. V1's protocol, unchanged:
container tag, warmup and repeat counts, median rather than mean, clock and power
state, device count and partitioning mode. **Grows one section: the convergence
protocol**, since v2 makes claims of a kind v1 never did. Model and token budget for
proxy runs, the baseline, how many seeds, and what loss delta we consider
distinguishable from noise. Every `[cited]` claim links here too, because the useful
thing to say about someone else's number is what it would take to reproduce it.

**Appendix C — The Configuration Reference.** `c-appendix-config.md`. ***New, and the
most-used page in the book.*** Every decision table from Parts II and III, aggregated:
knob, what it buys, what it costs, how to set it, status on ROCm, verified on. Nothing
here is new material, which is the point. It is a view, and it should be assembled
mechanically from the chapter tables if we can manage it, because a hand-maintained
copy will disagree with its sources inside a month.

**Appendix D — Profiling Reference.** `d-appendix-tooling.md`. ***New, and mostly
relocated.*** The XProf tool tour with its screenshots from Chapter 3, and the deep
escalation path from Chapter 12: `rocprofv3`, rocprof-compute, TraceLens, ISA
extraction, and the occupancy section with v1's framing intact. This is reference
material and it rots on a different schedule than the argument does, which is the
whole reason to separate them.

---

## File table and renumbering

**Inserting five chapters means renaming almost every file, and the convention is that
this happens in one commit.** V1's rule stands: filenames carry the ordinal, the
prefix is part of the URL, and a half-finished renumber leaves the prev/next chain
broken in a way Jekyll exits 0 on. Run `tools/check_links.py` immediately after.

| v2 file | `section_number` | Title | From |
|---|---|---|---|
| `index.md` | 0 | How To Scale Your Model with AMD | carried |
| `pages/1-rooflines.md` | 1 | All About Rooflines | 1 |
| `pages/2-amd-gpus.md` | 2 | How to Think About AMD GPUs | 2 |
| `pages/3-profiling.md` | 3 | How to Profile AMD GPU Programs | 3, shrunk |
| `pages/4-sharding.md` | 4 | Sharding, Collectives and Partitioners | 4, retitled |
| `pages/5-transformers.md` | 5 | All the Transformer Math You Need | 5 |
| `pages/6-numerics.md` | 6 | Numerics and Precision | new |
| `pages/7-parallelism.md` | 7 | How to Parallelize a Transformer for Training | 6, renamed |
| `pages/8-memory.md` | 8 | Memory, Recompute and the Optimizer | new |
| `pages/9-moe.md` | 9 | Mixture-of-Experts at Scale | 7 |
| `pages/10-kernels.md` | 10 | Kernels You Can Actually Reach | new |
| `pages/11-flags.md` | 11 | Compiler and Runtime Flags | new |
| `pages/12-getting-to-roofline.md` | 12 | Getting to Roofline | 8, shrunk |
| `pages/13-operations.md` | 13 | Operating a Training Run | new |
| `pages/14-llama.md` | 14 | Training Llama 3 on MI300X | 9 |
| `pages/15-deepseek.md` | 15 | Training DeepSeek-V2-Lite on MI300X | 10 |
| `pages/16-inference.md` | 16 | How to Think About Inference | 11 |
| `pages/17-serving.md` | 17 | Getting Your Model Into Production | 12 |
| `pages/18-conclusion.md` | 18 | Conclusions and Further Reading | 13 |
| `pages/a-appendix-install.md` | `section_label: Appendix A` | Installing JAX on ROCm | carried |
| `pages/b-appendix-protocol.md` | `section_label: Appendix B` | How We Measure | carried |
| `pages/c-appendix-config.md` | `section_label: Appendix C` | The Configuration Reference | new |
| `pages/d-appendix-tooling.md` | `section_label: Appendix D` | Profiling Reference | new |

**Eighteen chapters is a big book and the honest reading is that this is now two
books' worth of material.** The mitigation is that Part II chapters are shorter than
Part I chapters by design: a decision chapter is a table plus the reasoning that
justifies it, not a pedagogical arc. If that stops being true while writing, the book
is drifting back toward v1.

---

## Existing assets

V1's asset table is still accurate and still points almost entirely at Chapters 2, 3
and 12. Everything in it carries over with the chapter renumbering applied.

**Three assets v1's table missed or under-used, all of which now have a home:**

| Asset | Feeds |
|---|---|
| `gpu-profiling/docs/writeup/segfault-rocm-command-buffer-launch.md` | Ch 11 command buffers, as primary-source material |
| `gpu-profiling/scripts/segfault_embedding_grad_reproduction.py` | Ch 11, the reproduction behind that section |
| ROCm MaxText `nanoo_fp8` / `fp8` configurations | Ch 6, the measured half |

The command-buffer writeup is the strongest thing we own for the new Part II. It is a
real, diagnosed, AMD-specific flag hazard with a one-flag workaround, it is our own
work rather than a repetition of NVIDIA guidance, and it is exactly the tone the
limitations table established.

**Scripts still needed**, in dependency order. V1's list is unchanged and correct; v2
adds four:

1. A precision sweep harness: one model, one mesh, a format flag, reporting tokens per
   second and MFU per format. Blocks Chapter 6's cheap half.
2. A flag sweep harness: one workload, a list of flags, one changed at a time, median
   step time and tokens per second out. Blocks Chapter 11 entirely, and it is the
   highest value-per-line script in the plan.
3. An attention-backend comparison: the same attention shapes through TE, `jax-aiter`,
   `jax-triton` and XLA-native, reporting throughput and, importantly, which kernel
   name appears in the trace. Blocks Chapter 10.
4. A convergence proxy runner: fixed model, fixed token budget, fixed seed set, loss
   curves out, for comparing a precision recipe against a bf16 baseline. Blocks
   Chapter 6's expensive half and nothing else, which is worth knowing when scheduling
   it.

Items 1 through 3 are small and unblock three chapters. **Item 4 is the one real
capital expenditure in v2 and the one to decide on deliberately rather than drift
into.**

---

## Sequencing

The waves change, because the reader we are now writing for wants Part II first and
the theory second. **Wave order is a publication decision, not a dependency
statement**: the chapters still depend backwards, but a customer reading only the
first wave should get something immediately useful.

**Wave 1 — "What are my knobs?"** Chapters 6, 10, 11 and Appendix C. Precision,
kernel selection and flags, with the aggregated configuration reference. All three
need one 8-GPU node and the three small harnesses above. This wave is the answer to
the question that prompted the reframe, it contains two things nobody has published (a
measured flag table for JAX on ROCm and an attention-backend comparison), and it is
publishable without any of the theory chapters being finished, provided each chapter
links forward to where the derivation will live.

**Wave 2 — "Why those knobs?"** Chapters 1, 2, 3, 4. The preliminaries, including the
RCCL bandwidth sweep, which remains the single most novel measurement in the book.
Shipping this second rather than first is the visible consequence of the reframe.

**Wave 3 — "How do I lay it out?"** Chapters 5, 7, 8. The parallelism and memory arc.
Still the wave most at risk of being deferred, because none of the prose exists;
protect it.

**Wave 4 — "Sparse models, and why am I not at roofline?"** Chapters 9, 12.

**Wave 5 — "Two real runs, and how to keep them alive."** Chapters 13, 14, 15. The
most hardware exposure, and the wave where the central promise is either kept or
visibly not.

**Wave 6 — "What will it cost to serve, and how do I ship it?"** Chapters 16, 17, 18.
Deliberately last and cheap, exactly as in v1.

Waves 1 and 2 together are a coherent standalone document, and Wave 1 alone is a
genuinely useful one, which was not true of v1's Wave 1.

---

## Decisions defaulted in this draft

Three questions were open when v2 was drafted. Each is written above on a default;
these are the defaults and the arguments for them, so that changing one is a decision
rather than an accident.

**Venue: public book, with internal material quarantined.** V1 committed to an
external audience with no internal defect IDs in published prose, and v2 keeps that
for everything in `pages/`. The tension is real: a customer wants to hear "grouped
GEMM is tracked, here is the ticket and the ETA," and that sentence cannot appear in a
public book. The default is that status columns say *absent* or *experimental* with a
date and no ticket number, and that any ticket-level detail lives in a separate
internal companion document that is not part of this site. **If the primary
deliverable is actually a customer enablement doc rather than a public book, several
chapters get more useful and this decision should be revisited first**, because it
changes what Chapters 10 and 11 are allowed to say.

**Convergence evidence: run proxy-scale comparisons.** The alternative, citing
published results only, makes Chapter 6 cheap and materially weaker, since the whole
differentiator of this book over the source book is that we have machines. The default
is therefore a proxy-scale convergence protocol in Appendix B, with `[cited]` used for
anything at a scale we cannot run. **This is the largest new cost in v2** and the only
one that requires machine time on a different order than everything else.

**Shape: eighteen chapters.** The alternative was to keep thirteen and rewrite them
around the new objective. That fails for a concrete reason: precision, kernels and
flags do not fit inside existing chapters without becoming the same buried bullets
that caused the reframe. The cost is the renumbering above and a longer book, and the
mitigation is that Part II chapters are short by design.

---

## Open questions

Carried from v1 where still open, with the resolved ones dropped. V1's resolved
entries, MaxText on ROCm and the JAX decode question and the HBM bandwidth
discrepancy, stay recorded in `docs/structure.md` and do not need repeating.

**What is the scale-out fabric, and when do we get a cluster?** V1 called this the
largest gap between what the book promises and what we can measure, and scoped the
measurement promise to single-node in response. **Under v2 that mitigation no longer
covers us.** Frontier training is multi-node by definition, Chapter 13 is largely
about multi-node operation, and a customer evaluating us on multi-node performance
will not accept an analytical answer. Either an allocation appears, or Chapter 13 ships
with a prominent scope statement and the capstones stay single-node. AMD documents
Llama 3.1 405B as a multi-node MaxText benchmark, so the workload is ready if the
machines are.

**Which XLA flag generation is in the wheels we pin?** The collective pipelining
surface is mid-migration: an older boolean family, a newer enum family, one deprecated
umbrella flag, and an optimization-level mechanism that turns several of them on by
default. Chapter 11 cannot be written without pinning a version and reading the flag
definitions at that commit. This is a half-day of work and it gates the highest-value
chapter in Wave 1.

**What is the current Shardy default, and does it hold on ROCm?** `xla_use_shardy` has
been removed from `xla.proto` rather than merely defaulted, which suggests the
migration is finished on the XLA side, but the JAX-level control and the ROCm
behaviour both need confirming against the pinned wheel. Chapter 4 has a wrong
sentence in it until this is checked.

**Is there a ragged all-to-all path on ROCm?** XLA has the op, a decomposer, a
multi-host decomposer and a one-shot kernel, several of them behind `unsupported_`
flags. If any of that works on ROCm it changes Chapter 9's dropless story materially,
and it is a cheaper thing to test than the grouped GEMM question it partially
substitutes for.

**How much of AMD's kernel stack can a JAX user reach?** Carried from v1 unchanged and
now the spine of Chapter 10 rather than a detail: AITER via `ROCm/jax-aiter` for
attention and dense GEMM, no grouped or ragged MoE GEMM that we have found, Pallas
experimental through Triton, Mosaic GPU NVIDIA-only. Verify the whole list against
current wheels and pin versions when writing.

**Which of the three MoE dispatch implementations does MaxText run on ROCm, and under
which config fields?** Carried from v1. The names to confirm are `sparse_matmul`,
`megablox` and `capacity_factor`, along with whether the dropless path depends on a
Pallas kernel that exists only for TPU.

**Does RCCL schedule an 8-way all-to-all across all seven links?** Carried from v1.
The full-mesh argument predicts it should, which would make intra-node MoE dispatch
cheap. Measure it in the Chapter 4 sweep, where the script is needed anyway.

**Screenshot workflow.** Carried, and cheaper now that the tool tour lives in
Appendix D: manual or scripted against XProf's data endpoints, and either way it no
longer blocks a chapter in Wave 1.

**Is there a runnable companion?** Carried. Cheapest resolution is still to publish
the example scripts in a public repo and link them per chapter. **V2 raises the value
of doing this**, because the four new harnesses are the kind of thing a customer would
rather run than read about.
