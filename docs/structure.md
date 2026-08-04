# Roadmap

Planning document for *How To Scale Your Model with AMD*. Not published; this is
the working outline the chapters get written against.

The book grows. Chapters ship in waves and stand on their own; later chapters
slot in as siblings, because slugs are names rather than numbers. Voice and
structure rules live in `CLAUDE.md` at the repo root and are not repeated here.

## Contents

- [What this book is](#what-this-book-is)
- [Conventions](#conventions)
- [Notation](#notation)
- [Chapter 0 — the landing page](#chapter-0--the-landing-page)
- [Part I — Preliminaries](#part-i--preliminaries)
  - [Chapter 1 — All About Rooflines](#chapter-1--all-about-rooflines)
  - [Chapter 2 — How to Think About AMD GPUs](#chapter-2--how-to-think-about-amd-gpus)
  - [Chapter 3 — How to Profile AMD GPU Programs](#chapter-3--how-to-profile-amd-gpu-programs)
  - [Chapter 4 — Sharded Matrices and How to Multiply Them](#chapter-4--sharded-matrices-and-how-to-multiply-them)
- [Part II — Training Transformers](#part-ii--training-transformers)
  - [Chapter 5 — All the Transformer Math You Need](#chapter-5--all-the-transformer-math-you-need)
  - [Chapter 6 — How to Parallelize a Transformer for Training](#chapter-6--how-to-parallelize-a-transformer-for-training)
  - [Chapter 7 — Mixture-of-Experts at Scale](#chapter-7--mixture-of-experts-at-scale)
- [Part III — Training in Practice](#part-iii--training-in-practice)
  - [Chapter 8 — Getting to Roofline](#chapter-8--getting-to-roofline)
  - [Chapter 9 — Training Llama 3 on MI300X](#chapter-9--training-llama-3-on-mi300x)
  - [Chapter 10 — Training DeepSeek-V2-Lite on MI300X](#chapter-10--training-deepseek-v2-lite-on-mi300x)
- [Part IV — After Training](#part-iv--after-training)
  - [Chapter 11 — How to Think About Inference](#chapter-11--how-to-think-about-inference)
  - [Chapter 12 — Getting Your Model Into Production](#chapter-12--getting-your-model-into-production)
  - [Chapter 13 — Conclusions and Further Reading](#chapter-13--conclusions-and-further-reading)
- [Appendices](#appendices)
- [Existing assets](#existing-assets)
- [Sequencing](#sequencing)
- [Open questions](#open-questions)

---

## What this book is

**One sentence: given a model and some number of MI300X-class GPUs, how do I run
it in JAX so that adding GPUs adds throughput?** Everything in the book serves
that question. A chapter earns its place by answering some part of it.

This matters because the outline drifts otherwise. The obvious gravity well is
tooling: we have far more written material about profilers than about scaling,
so left alone the book becomes *How to Profile AMD GPU Programs* with scaling
chapters attached. That is a good document, but it is not this one. Profiling is
the instrument, not the subject. The subject is the parallelism decision.

Four commitments follow, and they are the things that distinguish this from the
source book:

**Standalone.** A reader who has never opened the TPU book can read this one
cover to cover. We never say "see the source book's Chapter N" for anything
load-bearing; at most we cite it as further reading on a tangent. That means we
pay full price for rooflines, sharding notation, and Transformer math rather
than borrowing them.

**JAX throughout, MaxText where a real model is needed.** Settled, and it
narrows the book usefully: one sharding vocabulary (`Mesh`, `NamedSharding`,
`PartitionSpec`, `shard_map`), one profiler, one compiler to reason about. The
cost lands entirely on the serving side, and once the book stopped pretending
otherwise the cost mostly evaporated: there is no JAX serving engine for ROCm now
that JetStream is archived, so Chapter 12 teaches the handoff to vLLM instead of
inventing one. Training, which is the focus, is well supported: AMD maintains a
ROCm MaxText fork with documented multi-node training on MI300X through MI355X.

**Predict, then measure, on real hardware, and be exact about which.** The source
book is almost entirely analytical, and admits as much in its GPU chapter, where
the theory meets a wall (NVIDIA claims 450 GB/s over NVLink; the authors measure
370, and 150 at realistic message sizes). We have machines. This is the single
most valuable thing we can add, so it is worth protecting from overclaiming.

The promise is therefore scoped: **every single-node training roofline in this book
is checked against a captured profile.** Inter-node claims are analytical until we
have a cluster, and inference claims are analytical by design, since we are not
running a serving stack. Each carries a visible marker saying so. A book that
measures what it can and labels the rest is trusted; a book that quietly derives
while claiming to measure is not, and the reader always finds out.

**MoE is first-class; serving is covered honestly rather than deeply.** The source
book gives MoE about one subsection and two problems, which is indefensible when
most frontier models are sparse: Chapter 7 is a full chapter, MoE accounting appears
in the Transformer-math chapter from the start rather than being quarantined in a
sparsity appendix, and the sparse training capstone is Chapter 10. The chapter's
spine is a question nobody has answered in public: of the three ways to implement
an expert layer, which one can a JAX user on ROCm actually run fast, and what does
the answer cost them in FLOPs.

Serving gets different and lesser treatment, deliberately. **The centre of gravity
of this book is training**, and the chapter order says so: Parts II and III are one
uninterrupted training argument, and everything about serving waits until Part IV.
Chapter 11 covers inference *rooflines*, because "how much memory do I need to serve
this myself" is a question the arithmetic answers cleanly and cheaply. Chapter 12
covers the *handoff* out of JAX, because production serving
on AMD means exporting weights to vLLM or SGLang and that is the correct engineering
decision rather than a shortfall. What the book does not do is teach a JAX serving
stack, since there isn't one on ROCm and building the chapter as though there were
was the outline's weakest bet. Saying this in one paragraph on the landing page is
better than an apologetic chapter.

**Non-goals.** Not a CDNA microarchitecture reference, not a HIP kernel-writing
tutorial, not a ROCm installation manual. Each of those is a real book and none
of them is this one. Where a reader needs one, link out.

**One deliberate omission to record, because readers of the source book will
notice it.** That book has a standalone chapter on programming TPUs in JAX, and
we have no counterpart. The API material is distributed instead: sharding APIs
and `shard_map` in Chapter 4, where they are introduced alongside the notation
they implement, and the parallelism implementations in Chapters 6 and 7 next to
the strategies they realise. Teaching the API twice is the redundancy this
avoids, and being JAX-only throughout means every chapter is already a JAX
programming chapter. Say this once on the landing page so nobody hunts for it.

---

## Conventions

**One Markdown file per chapter in `pages/`**, `layout: distill`, with `index.md`
staying at the repository root because it is the site landing page. Front matter
chains the chapters together; nothing is inferred, so adding a chapter means
editing its neighbours too.

| File | `section_number` | Title |
|---|---|---|
| `index.md` | 0 | How To Scale Your Model with AMD |
| `pages/rooflines.md` | 1 | All About Rooflines |
| `pages/amd-gpus.md` | 2 | How to Think About AMD GPUs |
| `pages/profiling.md` | 3 | How to Profile AMD GPU Programs |
| `pages/sharding.md` | 4 | Sharded Matrices and How to Multiply Them |
| `pages/transformers.md` | 5 | All the Transformer Math You Need |
| `pages/training.md` | 6 | How to Parallelize a Transformer for Training |
| `pages/moe.md` | 7 | Mixture-of-Experts at Scale |
| `pages/getting-to-roofline.md` | 8 | Getting to Roofline |
| `pages/llama.md` | 9 | Training Llama 3 on MI300X |
| `pages/deepseek.md` | 10 | Training DeepSeek-V2-Lite on MI300X |
| `pages/inference.md` | 11 | How to Think About Inference |
| `pages/serving.md` | 12 | Getting Your Model Into Production |
| `pages/conclusion.md` | 13 | Conclusions and Further Reading |
| `pages/appendix-install.md` | `section_label: Appendix A` | Installing JAX on ROCm |
| `pages/appendix-protocol.md` | `section_label: Appendix B` | How We Measure |

All fifteen files exist as skeletons: correct front matter, the section headings
below, and a "To write" blockquote per section carrying the brief for it. Writing a
chapter means replacing those blockquotes, and
`grep -rn '^> \*\*To write' pages/` is the remaining-work list, currently 164
sections.

Three mechanical rules the build will not catch if you break them:

- **Internal links go through the `relative_url` filter.** Section URLs in front
  matter are site-root-relative (`/pages/moe`) and the layout pipes them through
  `relative_url`; body links pipe it themselves. A bare `/pages/moe` drops the
  `baseurl` and 404s on the deployed site.
- **Every `toc` name must match a heading exactly**, because the anchor is the
  slugified name. Avoid apostrophes and slashes in headings: kramdown deletes them
  when it builds heading IDs, while Liquid's `slugify` turns them into hyphens, so
  the two disagree and the sidebar link silently 404s. Colons, commas and question
  marks are handled the same way by both and are safe.
- **Appendices carry `section_label` instead of `section_number`**, which the layout
  uses in place of "Chapter N".

`tools/check_links.py` checks all three against the built site, plus that the
prev/next chain is symmetric. Run it after every structural edit; Jekyll exits 0 on
all of these.

Note the tension in the numbering: slugs are stable but `section_number` is
ordinal, so inserting a chapter mid-book means editing every file after it. That
is cheap and worth it for readable navigation, but do the insert-and-renumber in
one commit rather than leaving the chain half-broken.

**"Chapter" is the file, "Part" is the grouping.** The source book uses "Part"
for both and it reads ambiguously. Navbar and front matter say Chapter N;
`index.md` groups them under Parts.

**Every chapter ends in worked problems** with answers behind `{% details %}`,
including reference numbers so the reader can tell whether they got it right.
For the long chapters, prefer several short quizzes placed after each major
section over one terminal problem set; the source book's GPU chapter does this
and it paces much better.

**Every performance claim is either computed or measured.** The predict-then-
measure pattern is the spine of the whole book: state the expected number from
hardware first, then show what the profile says. Sections below note which
number each one hangs on.

**Which of the two it was is visibly tagged.** The scoped promise in
[What this book is](#what-this-book-is) needs a mechanism or it stays a promise.
Two tags, defined once on the landing page and used inline at the point of the
claim: **[measured]** means a number read off a captured profile on hardware we
ran, with Appendix B behind it, and **[analytical]** means derived from published
specs and not checked. Everything inter-node and everything inference-side is
**[analytical]** by default. A section that is entirely one or the other says so in
its opening line instead of tagging every sentence. The tag is cheap; the trust it
buys is the whole differentiator over the source book, so do not let it lapse in
the chapters where the honest answer is the less impressive one.

**Measured numbers follow one stated protocol, defined once and linked from every
chapter.** This is the convention that decides whether the book's central claim
survives contact with a skeptical reader. "We measured 370 GB/s" is an anecdote;
the same number with a stack and a method behind it is evidence. Fix all of the
following now, in Appendix B, which every chapter links to:

- The exact software stack: ROCm version, JAX, jaxlib, the two plugin wheels, and
  the container tag if one was used. **Prefer quoting a container tag over a list
  of wheel versions**, because it is one string and it is actually reproducible by
  a reader. AMD's prebuilt JAX images are the obvious baseline and are a better
  target than "stock wheels" precisely because they pin everything at once.
- Warmup and repeats: how many iterations discarded, how many measured, and
  whether we report median or mean. Pick median and say so; autotuning and clock
  ramp make the first iterations useless and the mean misleading.
- Clock and power state, since MI300X at 750 W will throttle under a sustained
  matmul and a reader comparing against a boost-clock roofline will see a gap that
  has nothing to do with their code. If we lock clocks, say so; if we do not, say
  that too and expect to explain a few percent.
- Device count and partitioning mode (SPX/NPS), because Chapter 3's limitations
  table includes a row about op times being summed across devices, and a number
  without a device count is unreadable.

**Software behaviour gets a "verified against" line too, not just hardware.** The
spec-table rule below covers hardware numbers, which drift once a generation.
Tooling limitations, wheel names, install commands and library recommendations
drift every few months and are the more likely thing to embarrass us. Same
discipline, same one-line treatment, applied to Chapter 3's limitations table,
Chapter 8's kernel-library advice and Chapter 12's stack survey.

**Every chapter states its dependencies in one line at the top**, and those
dependencies are strictly backward. If a chapter needs a concept, either an
earlier chapter taught it or this chapter teaches it. There is no third option;
"see the source book" is not a dependency we are allowed to have.

**Target build is a stock, publicly available stack**, not a patched fork, so
everything is reproducible by an outside reader. In practice that now means AMD's
prebuilt JAX and MaxText container images as the primary target, with the wheel
install documented in an appendix for readers who cannot use containers. Both are
public; the container is one line and pins the whole stack. Known-broken views are
documented rather than worked around.

**Audience is external**: JAX users running LLMs on MI300X-class hardware. No
internal defect IDs in published prose. The internal audit
(`gpu_profiling/docs/writeup/xprof-for-amd.md`) is the source for the
limitations table but is not linked from it.

**Spec tables carry a "verified against" date and source.** Hardware numbers go
stale within a generation and the source book has no discipline here at all.
One line under each table.

---

## Notation

Fixed for the whole book, introduced in Chapter 5, and repeated at the top of
every chapter that uses it. Getting this wrong is expensive later, so decide now.

The table lives in `_includes/notation.liquid` and is included rather than pasted
thirteen times, because a notation table that drifts between chapters is worse than
no table. Chapters wrap the include in a collapsible `details` block so it does not
push the opening paragraph down the page.

| Symbol | Meaning |
|---|---|
| `B` | batch size, in sequences |
| `T` | query sequence length |
| `S` | key/value sequence length |
| `D` | model dimension (`d_model`) |
| `F` | feed-forward dimension (`d_ff`) |
| `N` | query heads |
| `K` | key/value heads |
| `H` | head dimension |
| `L` | layers |
| `V` | vocabulary size |
| `E` | experts per layer |
| `E_a` | experts activated per token |
| `C` | accelerator FLOPs/s |
| `β` | bandwidth (subscripted: `β_hbm`, `β_xgmi`, `β_net`) |

**Four collisions the source book lives with and we should not.** Fixing them
costs nothing today and would cost a rewrite later, so they are settled here:

1. **Bandwidth is `β`, not `W`.** The source book uses `W` for bandwidth, but
   `W` is also the universal name for a weight matrix, and Chapter 6's five-part
   treatment of each strategy is literally written in terms of `W_in` and
   `W_out`. Having `W_in` mean a weight matrix and `W_hbm` mean a bandwidth on
   the same page is a trap. Greek for the hardware constant, Roman for the
   tensor.
2. **Activated experts are `E_a`, not `k`.** `k` versus `K` (key/value heads) is
   a case-only distinction, which survives neither prose nor a reader's
   handwriting. `E` total and `E_a` activated also makes the sparsity ratio read
   correctly as `E / E_a`, which is the quantity that actually recurs.
3. **Times are lowercase: `t_math`, `t_comms`.** `T` is the query sequence
   length everywhere else in the book, so `T_math` invites exactly the wrong
   reading in a chapter that is about sequence-length scaling. Lowercase for
   durations, uppercase for shapes, no exceptions.
4. **Device and shard counts are mesh-axis sizes, never `N`.** `N` is the query
   head count, so any cost model written over `N` devices collides in exactly the
   chapters that need both at once. Axis sizes are written `|X|`, `|Y|`, `|Z|`,
   `|Ex|`, and the total device count is their product. This one is not
   hypothetical: the ragged all-to-all cost in Chapter 7 is `min(E_a / |Ex|, 1)`,
   and the version of that expression with `N` in it is unreadable in a chapter
   that also talks about attention heads.

Mesh axes are named: `X` for data/FSDP, `Y` for tensor, `Z` for pipeline, `Ex`
for expert. Sharded arrays are written `A[I_X, J_Y]`, meaning `A` is sharded on
its first axis over mesh axis `X` and its second over `Y`.

**Quantities that are also config keys keep the config key as their name.** No
symbol for capacity factor: write `capacity_factor`, because that is the field the
reader edits and because every single-letter candidate collides with something in
the table above. Same rule for anything else in Chapters 9 and 10 that maps
one-to-one onto a MaxText option.

The named-axis notation is the source book's best pedagogical invention and it
makes every later derivation readable. It costs most of Chapter 4 to teach,
which is worth it, and being JAX-only means it maps directly onto what the
reader writes: `A[I_X, J_Y]` is the `PartitionSpec` they pass to
`NamedSharding`. Introduce the two together so the notation never feels like
a parallel vocabulary invented for the book.

---

## Chapter 0 — the landing page

`index.md` · Worth planning rather than writing last, because it is the only page
most visitors will read, and because the source book's equivalent does real work.
Four things, in order:

- **The one-sentence thesis and why the reader should care.** Given a model and
  some number of MI300X-class GPUs, how do I run it in JAX so that adding GPUs adds
  throughput. Then the strong-scaling framing: adding GPUs cuts compute time but
  adds communication, and the whole book is about where those two cross.
- **Expected background**, stated plainly: comfortable with the Transformer
  architecture, some JAX, no assumed knowledge of GPUs, ROCm, collectives, or the
  TPU book. Also what is *not* assumed, because "you do not need to have read the
  TPU book" is a selling point for exactly the audience most likely to worry.
- **What this book adds**, in three lines: measurements on real AMD hardware
  rather than pure analysis, MoE as a first-class subject, and an honest account of
  what happens to the checkpoint after training. Also **what it is not**: the focus
  is training, and readers looking for a serving deployment guide should be pointed
  at vLLM on ROCm in the first hundred words rather than the tenth chapter.
- **Reading paths.** Thirteen chapters is enough that "read it in order" is not
  the only useful instruction, and offering routes is a cheap, genuine improvement
  on the source book, which just lists its chapters. Four routes:

  | Route | Chapters |
  |---|---|
  | I want to train a dense model efficiently | 1, 2, 3, 4, 5, 6, 8, 9 |
  | I want to train an MoE | 1, 2, 3, 4, 5, 7, 8, 10 |
  | I need to size and ship what I trained | 1, 2, 5, 11, 12 |
  | My profile looks wrong and I just need the tooling | 2, 3, 8 |

  Chapters 1, 2 and 5 appear in all four routes, which makes them the
  irreducible core: useful for the reader to know, and a useful constraint on us
  when deciding what belongs in them. Note that the two training routes carry
  Chapter 3 even though it is a tooling chapter, because Chapter 8 is unreadable
  without it; the honest version of these routes cannot drop profiling from a book
  whose spine is predict-then-measure. The fourth row is a reference route rather
  than a reading route, for the readers who arrive from a search result with a
  broken profile and no interest in the rest of the book. There will be a lot of
  them, and Chapter 3's limitations table is what they came for.

The notation table also lives here, repeated in each chapter's front matter as
[Notation](#notation) describes.

---

## Part I — Preliminaries

Everything a reader needs before the word "parallelism" means anything. All four
chapters are independent of MaxText and of any large model, which makes them the
publishable first wave.

### Chapter 1 — All About Rooflines

`rooflines.md` · **What actually limits how fast this runs?**

No AMD content at all, deliberately. Hardware-independent reasoning first, so
Chapter 2's constants have somewhere to land. Short: this is the gentlest
chapter in the book.

- **Three bounds**: compute, memory bandwidth, communication. `t_math` and
  `t_comms`, and why `max(t_math, t_comms)` is a decent lower bound while
  `t_math + t_comms` is the upper bound, with the observation that they differ
  by at most 2x so it rarely matters which you use.
- **Arithmetic intensity** as FLOPs per byte. The key algebraic move, which the
  whole book reuses: being compute-bound is exactly the statement that the
  algorithm's intensity exceeds the hardware's. Dot product as the hopeless
  case, matmul as the good case.
- **The critical batch size.** Matmul intensity is roughly `B` when `B << D, F`,
  so the ridge point falls at a per-device token batch size. Derive it
  symbolically here; Chapter 2 substitutes MI300X numbers and gets an actual
  figure. This is the single most-cited number in the book and it deserves a
  clean derivation.

  **Disambiguate the name in one sentence, here, the first time it is used.** In
  the training literature "critical batch size" means the batch past which more
  data stops buying convergence, which is a different quantity with a different
  value, and a training-focused reader arrives holding that meaning. This book
  always means the hardware ridge point. Chapter 6 needs both at once, when it
  works out that the DP degree times the per-device batch is capped from above by
  the convergence limit and from below by this one.
- **Communication rooflines.** A two-device sharded matmul where the crossover
  depends on `D` rather than `B`, to prime the reader that *which* variable
  controls the roofline changes per strategy.

**Worked problems.** Steal the source book's best exercise wholesale: hand the
reader an AMD spec sheet and make them notice that the headline bf16 number is
the sparsity-enabled figure, then have them redo the ridge point with the dense
one. Teaching spec-sheet skepticism as an exercise is cheap and sets up
Chapter 2's honesty about our own tooling. Also: an int8 and an fp8 variant, to
show the ridge point moving.

### Chapter 2 — How to Think About AMD GPUs

`amd-gpus.md` · **How fast should this run on an MI300X, and what in the
hardware decides that?**

Every constant the rest of the book uses. Resist making it a survey of CDNA; if
a fact does not feed an arithmetic prediction, cut it. The test for each
paragraph is "which later prediction breaks without this?"

- **What an MI300X is.** XCDs, compute units, SIMDs, MFMA matrix cores, register
  file and LDS per CU. The numbers that matter downstream are 304 CUs at roughly
  2.10 GHz, both of which our own traces report in the device plane. Mention
  SPX/NPS partitioning only insofar as it changes the CU count a process sees.
- **The memory hierarchy.** HBM3, Infinity Cache, L2, LDS, registers: capacity
  and bandwidth at each level. 192 GiB of HBM at 5.3 TB/s. Supplies the
  memory-bound side of every later roofline.
- **Peak FLOPs by dtype.** A table: fp32, tf32, bf16, fp16, fp8, int8. State
  plainly that the marketing numbers usually quote the sparsity-enabled figure
  and that we use dense throughout. Dense bf16 on MI300X is 1307 TFLOP/s, from
  304 CU x 2048 FLOP/clock/CU x 2.10 GHz, which is the figure to carry forward.
  Show that arithmetic rather than quoting the table, both because it lets the
  reader rebuild the number for any other part and because published tables
  disagree in the third digit: AMD's spec sheet says 1307.4, several OEM pages
  say 1305 from a slightly different clock, and a few third-party sites have it
  at 653 because they mistook the dense figure for the sparse one. Substitute
  into Chapter 1's ridge point and get the AMD critical batch size.

  **This is also where the numeric formats get introduced, once, for the whole
  book.** fp8 is 2x bf16 on MI300X, which halves the ridge point, and gfx942's
  fp8 is the NANOO/FNUZ variant rather than the OCP one that gfx950 implements.
  That distinction bites anyone moving a checkpoint between the two, so name it
  here and let Chapters 6, 11 and 12 spend it rather than re-explaining it.
- **Inside a node.** The 8-GPU OAM baseboard (UBB 2.0) is a *switchless, fully
  connected mesh*: seven xGMI links per GPU at 128 GB/s bidirectional each,
  896 GB/s aggregate, one hop to every peer, plus one PCIe Gen 5 x16 to the host.
  Contrast with an NVLink domain, which puts a switch in the middle and can
  therefore grow past eight. **The AMD scale-up domain cannot: eight is the
  ceiling, and the ninth GPU is over the NIC.** Enough to predict an all-reduce,
  and enough for Chapter 6 to reason about what must stay on the baseboard.

  **This chapter owns the wiring; Chapter 4 owns what it costs.** The topology is
  described once, here, in physical terms. Chapter 4 gets one line of recap and
  spends its space on the two things that are actually its own: that RCCL's chosen
  algorithm need not match the wiring, and what each collective therefore costs.
  Chapters 6, 7 and 12 cite the eight-GPU ceiling rather than restating it. Without
  that split this paragraph gets written four times, which is what the previous
  draft of this outline was heading for.

  Honesty note worth including rather than smoothing over: AMD's own MI300X data
  sheet says "seven Infinity Fabric links for full connectivity between eight
  GPUs **in a ring**," while the platform data sheet and ROCm architecture docs
  in the same family say fully meshed. The mesh is correct. This is a good early
  demonstration that vendor documentation needs cross-checking, and it costs one
  footnote.
- **Beyond a node.** Scale-out NICs, RoCE/Ethernet versus InfiniBand, per-node
  egress bandwidth, and the fabric topology of a typical MI300X cluster. **This
  is currently a hole in our knowledge and a hole in the source book, and it is
  where a lot of production performance is decided.** Chapters 6, 7 and 12 all
  need a node-egress number. Until we have one, say so in the text and mark the
  affected derivations **[analytical]**: see
  [Conventions](#conventions) on what we are allowed to promise.
- **Worked example: a 4096³ bf16 matmul.** The first full predict-then-measure.
  `2 * 4096³ / 1307e12` gives the expected time; Chapter 3 captures the profile
  and checks it. Same shape as `jax_matmul.py` so the two chapters line up.
- **What changes across the family.** Short, but with the counterintuitive parts
  stated plainly, because a reader substituting constants will get them wrong
  otherwise.

  MI325X first, in one sentence, because it is the cheapest thing to get wrong:
  same gfx942, same 304 CUs and therefore the same FLOP table as MI300X, but 256 GB
  of HBM3E at 6.0 TB/s. A reader on MI325X who carries the MI300X memory numbers
  gets both the capacity ceiling and every bandwidth-bound roofline wrong while
  getting the compute ones right, which is the confusing failure mode.

  Then MI355X, which changes almost everything. gfx950, CDNA4, 288 GB HBM3E at
  8.0 TB/s, and **256 CUs, which is
  fewer than MI300X's 304**: the generation got faster per CU and narrower, so
  anything the reader scaled by CU count breaks. Dense bf16 goes to roughly
  2.5 PFLOP/s and fp8 to 5.0, xGMI per-link rises to 153.6 GB/s while the
  seven-link mesh topology is unchanged. **LDS goes from 64 KB to 160 KB per CU,
  while the 512-entry-per-lane vector register file does not change**, which is the
  pairing to state explicitly because the common assumption is that CDNA4 doubled
  the registers too. Revised FLOP table including fp6/fp4,
  where the interesting entry is that **MXFP6 runs at the same rate as MXFP4**
  rather than half of it, so on gfx950 fp6 is a nearly free accuracy upgrade over
  fp4. That is a real scaling-arithmetic result and worth its own sentence.

  One paragraph, no more, on what comes after: the MI400 series (MI455X, CDNA5,
  HBM4) and rack-scale Helios landed in mid-2026, and CDNA5 renames the compute
  unit to a Work Group Processor, so the book's core unit of accounting changes
  name again. Point the reader at primary datasheets rather than quoting numbers
  we cannot check, and note that published per-GPU HBM4 bandwidth figures for
  MI455X currently disagree with each other.
- **A translation table**, if we can support it: MI300X against H100 and TPU
  v5e, unit for unit (CU vs SM vs Tensor Core, LDS vs SMEM vs VMEM, MFMA vs
  Tensor Core vs MXU). Readers arrive with NVIDIA vocabulary and this is the
  cheapest way to meet them. AMD's
  [occupancy math post](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html)
  has a compact CDNA-to-CUDA table that covers the compute-side rows well, including
  the one people trip on, that an AMD wavefront is 64 threads against NVIDIA's
  32-thread warp; use it as a source and add the TPU column, which is ours to write. We have same-workload traces on all three, which
  almost nobody else does. Caveat the empirical half honestly: the captures are
  8x MI300X, 4x H100 and a single v5e chip, so they support per-device
  comparisons and unit-vocabulary mapping but not scaling claims.

> **Verify before writing.** Every hardware number above needs checking against
> official specs, with the check dated in the text. See
> [Open questions](#open-questions) for the HBM bandwidth discrepancy, which is
> now resolved and should be written up *as an example* rather than quietly
> corrected: it is a perfect illustration of why you check the tool against the
> spec sheet.

### Chapter 3 — How to Profile AMD GPU Programs

`profiling.md` · **Your model is slower than the arithmetic says it should be.
Where did the time go?**

**Why the book stops here rather than pressing on to sharding.** Because
Chapter 2 ends on a cliffhanger. It predicts a time for a 4096³ matmul and has no
way to find out whether it was right. This chapter is the payoff, and framing it
that way turns what would otherwise read as a detour into the natural next step:
the reader wants to know if the number held. Say this explicitly in the opening
paragraph, and note that from Chapter 5 onward every claim is checked against a
profile, so the instrument has to be in hand before the experiments start.

Scoped to "capture a trace and read it." The escalation path to hardware
counters and ISA lives in Chapter 8, where the reader has a reason to care.

**Tell the impatient reader what the minimum is.** This is the one chapter a
scaling reader might resent, since it is tooling and it sits between them and any
parallelism content. So name the short path in the opening: the first-trace section
and the limitations table are what Chapter 4 onward actually depends on, the XProf
tool tour can be skimmed and returned to, and nothing after that is load-bearing
until Chapter 8. One line, and it removes the only real objection to the ordering.

**This chapter is still the biggest length risk in the book and should be
actively resisted while writing.** We have more material here than anywhere else,
which is exactly why it will bloat. Three specific economies, all applied below:
setup shrinks to a container pull, the profiler's internals collapse into the
limitations section that needs them, and the two "read the compiler's output"
sections merge into one. If a section does not help the reader answer "where did
my time go", it goes to an appendix or to Chapter 8.

- **A thousand-foot view of the stack.** JAX to StableHLO to HLO to LLVM/ROCDL
  to GCN ISA, with the passes that matter (fusion, layout assignment) called
  out. Then where the PJRT plugin sits and why ROCm ships two wheels rather than
  one. The `fw101` PJRT section and the two XLA pipeline diagrams drop in here
  almost unchanged.
- **Setup, in about fifteen lines.** Pull AMD's prebuilt JAX image, confirm the
  GPU with `rocm-smi --showproductname`, map gfx942 to MI300-class and gfx950 to
  MI350-class, then the verification snippet that asserts
  `jax.default_backend() == "rocm"` and runs a real computation end to end. That
  is the whole section. Installing the four wheels in the right order
  (`jax-rocm7-pjrt`, `jax-rocm7-plugin`, `jaxlib`, `jax`), the ROCm version
  matrix, and building from source all go to Appendix A: they are necessary,
  they are nobody's reason for reading the book, and they rot fastest.
  `gpu_profiling/readme.md` is the source for the appendix.
- **How a profile gets made.** `rocprofiler-sdk` to the ROCm collector to
  XPlane/XSpace protos to XProf. **Fold this into the limitations section rather
  than giving it its own top-level slot.** It exists to make the limitations
  legible: once you know the collector writes XStats that XProf later reads, the
  broken fields stop looking arbitrary and start looking like a specific missing
  write. On its own, ahead of any symptom, it is stack tourism. Keep it to a
  diagram and four paragraphs, immediately before the table. Source is
  `gpu_profiling/docs/xla-rocm-profiler-backend.md`.
- **Your first trace.** Purely mechanical: wrap `jax.profiler.trace` around the
  matmul, show the trace directory layout down to the `.xplane.pb` and
  `.trace.json.gz`, launch `xprof --logdir`, forward the port. Establish the
  convention of writing every trace under `/tmp/traces/<workload>/` so one
  `--logdir` picks them all up.
- **The XProf tools.** One at a time, each with a real screenshot: Trace Viewer,
  Graph Viewer, Op Profile, Kernel Stats, Memory Profile, Roofline. What each is
  for, what to look at first. Trace Viewer gets the most space, including the
  video-game navigation keys.
- **What works today.** The limitations table, with a stable anchor, because
  this is the section people will be linked to directly. Each row is a symptom
  the reader will observe, a one-line cause, and the workaround.

  **Pin it to a version and date it, exactly as the spec tables are pinned.** The
  book already insists on a "verified against" line for hardware numbers, which
  change once a generation. This table describes software behaviour, which changes
  every few months, and several of these rows have fixes in flight in our own
  repo. An undated table of broken things is a liability: it will be quoted back
  at us after the bug is fixed. So the heading carries the exact wheel and ROCm
  versions the observations were made against, and any row with a known fix in
  progress says so in a fourth column.

  Seven rows, curated down from the twenty in the internal audit. The selection
  rule is "a reader will hit this in their first week", which is why the row about
  NVIDIA-shaped capability structs does not appear:
  1. Overview and Input Pipeline read "No step time measured" — no step markers
     in the trace. **The reader fixes this**: add
     `jax.profiler.StepTraceAnnotation`.
  2. Roofline compute ceiling reads 0 GFLOP/s and the labels say "per
     TensorCore" — the peak-FLOP value is never computed for AMD. Do it by hand,
     per Chapter 2.
  3. Kernel Stats occupancy, registers per thread and shared memory all read 0 —
     the collector does not emit them. Use `rocprofv3` (Chapter 8). Point forward
     to Chapter 8's occupancy section too, not just to the tool: a reader who goes
     and fetches the number is about to misinterpret it, and the row costs nothing
     by saying so.
  4. "GPU TensorCore utilization" reads 0 on every row — the kernel-name
     classifier does not recognise MFMA kernels. Ignore the column.
  5. Multi-GPU op times are summed across devices, so a value on an 8-GPU node
     is roughly 8x the wall-clock figure. Divide by the device count.
  6. Device Compute Precisions reads 0% / 0% — depends on step markers, so it
     follows from row 1.
  7. HBM bandwidth in the device plane is understated by 2x, and displayed in
     binary units while labelled decimal. Cause and correct arithmetic in
     Chapter 2; this row is the pointer.

  Tone matters: this is a candid "here is the state of the tooling" section, not
  an apology. Most readers arriving at it have already hit one of these and
  assumed they had misconfigured something.
- **From an HLO op back to a Python line.** These were two sections in the
  previous outline and they are one skill: following the compiler's output back to
  the code you wrote. Split, they make the reader learn to parse an op, put the
  book down, and then learn separately what to do with it. Run it as one arc
  instead, in the direction the reader actually travels: a slow kernel row in
  Kernel Stats, to the HLO op that emitted it, to the `jax.named_scope`
  annotation, to the source line, plus what to do when the chain breaks.

  Along the way, teach the anatomy of the op. Do **not** port the source book's
  version; its worked example is TPU-specific and neither the `T(8,128)(2,1)`
  tiling nor the `S(1)` memory-space annotation exists on AMD. Write it fresh
  against a real op from the matmul trace: op name, output shape and dtype, the
  plain major-to-minor layout, operands. Then the AMD specifics that matter for
  the traversal: fusion kinds (`kLoop`, `kInput`, `kCustom`), custom calls into
  hipBLASLt and rocBLAS and how they surface as `Cijk_*` kernels in Kernel Stats,
  and async collective pairs. Follow the style guide's toy-case-first move. The
  `fw101` "tracing upwards from kernel to python" material feeds this.
- **The matmul, revisited.** Close the loop opened in Chapter 2. Expected time
  versus measured, stated plainly whether the model held. If it did not, this is
  the best possible place to explain why: warmup, autotuning, clock throttling.
- **A training step.** Forward, backward, optimizer. Where `StepTraceAnnotation`
  is introduced and where the Overview page comes alive. Show it both ways:
  capture without the annotation first, hit the "No step marker observed" wall,
  then add it and recapture. Most readers have already seen the broken version,
  so leading with it is the honest ordering.

**Worked problems.** Given a trace, identify which of two matmuls is
memory-bound and say why; find the kernel with the largest self-time and trace
it back to its Python line; explain a device-op-time figure that is 8x the wall
clock.

### Chapter 4 — Sharded Matrices and How to Multiply Them

`sharding.md` · **You split the matrix across eight GPUs. What does that cost?**

New chapter, and the keystone the previous outline was missing. Without it,
Chapter 6 has to introduce sharding notation, collective cost, *and* five
parallelism strategies at once, which is why the source book spends a whole
chapter here. Every inequality in Chapters 6, 7 and 11 is a substitution into the
cost model built here.

- **Notation.** Meshes, named axes, `A[I_X, J_Y]`, and the "unreduced / partial
  sums" annotation. Toy shapes first, per the style guide.
- **The four collectives.** AllGather, ReduceScatter, AllReduce, AllToAll: what
  each does to a sharding, and what each costs.

  **Derive against the mesh, in one recap line and then two consequences.**
  Chapter 2 established the wiring: switchless, fully connected, seven direct
  xGMI links per GPU. Recap it in a sentence and do not re-derive it. AMD sits in
  neither of the two positions the reader may arrive with, not a torus like a TPU
  pod and not a crossbar like an NVSwitch domain, so note where this differs from
  the torus model the TPU book assumes, since the TPU result (cost independent of
  axis size) does not carry over cleanly.

  Then the two consequences, which are this chapter's own material rather than
  Chapter 2's. **First, physical topology is not the collective algorithm.** The
  links form a mesh, but RCCL still chooses a ring or a tree schedule over it, so
  the achieved cost follows the algorithm rather than the wiring, and that is
  exactly why the measured sweep below matters more than the derivation. **Second,
  the scale-up domain stops hard at eight**, so every cost in this chapter has two
  regimes with a cliff between them rather than one smooth curve. That
  discontinuity drives placement decisions in Chapters 6, 7 and 12, and it is the
  reason those chapters can cite a single fact instead of re-arguing topology.
- **RCCL in practice.** How collectives appear in the trace and in HLO, matching
  `all-reduce-start` to `all-reduce-done`, and reading `replica_groups` to work
  out the sharding actually in use. Inferring the parallelism strategy purely
  from replica groups is a genuinely useful trick and worth reproducing.
- **Measured versus spec bandwidth.** The most valuable section in the chapter.
  Sweep message size and device count, plot achieved against theoretical,
  and state the message size at which RCCL reaches asymptotic bandwidth. The
  source book does exactly this for NCCL and finds 370 GB/s against a claimed
  450, and 150 GB/s at realistic LLM message sizes. Nobody has published the
  AMD equivalent. **This section alone justifies the chapter.**
- **The four sharded-matmul cases.** Which sharding of inputs and outputs
  requires which collective, and the resulting cost. This is the lookup table
  Chapter 6 indexes into.
- **One program, many processes.** New bullet, and the missing prerequisite for
  every placement argument later in the book. A multi-node JAX job is one SPMD
  program running in as many processes as there are hosts:
  `jax.distributed.initialize`, arrays that are globally shaped but locally
  backed, and `jax.process_index` for the things that genuinely differ per host.
  The reason it belongs in this chapter rather than in a capstone is that **the
  order of the axes in the `Mesh` decides which collective crosses the NIC**, and
  that is precisely the question Chapters 6 and 7 keep asking. A mesh laid out so
  that the expert axis straddles two baseboards is a one-line mistake with a large
  bill, and the reader cannot see it without this section. Keep it short and keep
  it honest: the mechanism is exact, the eight-GPU numbers are **[measured]**, and
  anything spanning hosts is **[analytical]** until we have a cluster.
- **Who inserts the collective: GSPMD or you?** New bullet, and the missing half
  of the JAX story. Annotating an array with a `NamedSharding` and letting the
  compiler derive the collectives is the default, and for the strategies in
  Chapter 6 it is usually the right one. `shard_map` hands you the per-device view
  and makes you write the collective yourself. The reader needs the distinction
  here, before Chapter 7, because MoE routing is the case where the compiler's
  choice is not good enough and hand-written all-to-all is the norm. Show the same
  sharded matmul both ways and diff the HLO: that comparison teaches more about
  what GSPMD is doing than any amount of explanation.
- **Is the collective overlapping?** Serialized versus overlapped is immediately
  visible in the Trace Viewer. Show the same workload with and without
  `--xla_gpu_enable_latency_hiding_scheduler=true` and
  `--xla_gpu_enable_highest_priority_async_stream=true`, both already set in
  `transformer_block.py`, and give the measured step-time difference.

  **This section owns overlap for the whole book.** Chapter 8's triage list opens
  with "are the collectives overlapping", which is the same question, and the two
  must not both teach it. Chapter 4 teaches how to see it and what the flags do;
  Chapter 8 gets one line and a cross-reference.

**Worked problems.** How long should an all-reduce of a 1 GB gradient buffer
take on one node, and on two; from a set of replica groups, name the parallelism
strategy; at what message size does RCCL stop being latency-bound.

---

## Part II — Training Transformers

The core of the book. Part I taught the hardware and the cost model; this part
spends them, on training.

**Inference waits until Part IV, and that is a deliberate change from the previous
draft**, which put an inference chapter here between MoE and the practice chapters.
It made the middle of the book alternate subject every chapter: train, train,
infer, train, infer. Now Parts II and III are one continuous training argument and
Part IV is one continuous serving argument, which costs nothing in dependencies and
reads far better. Say on the landing page that the serving arithmetic is in Part IV
so nobody hunts for it.

### Chapter 5 — All the Transformer Math You Need

`transformers.md` · **How many parameters, FLOPs and bytes, exactly?**

New chapter, and the notational backbone for everything after it. Reference-
dense rather than argumentative. **MoE accounting appears here alongside dense
accounting, not in an appendix**. Treating sparsity as a footnote is one of the
source book's clearer mistakes, given that most frontier models are sparse.

- **Counting rule.** For a contraction, FLOPs = 2 x (product of all dimensions,
  batch and contracting dims counted once). Forward versus backward, and the
  derivation of `6 * params * tokens`.
- **Per-layer accounting.** MLP (`3DF` params with a gated einsum, `18BTDF`
  training FLOPs), attention (`2D(N+K)H` for GQA, `24BTDNH + 12BT²NH`), norms,
  and the vocabulary projection. Notes on MHA/MQA/GQA and pre- versus post-norm.
- **When does attention matter?** The `T > 8D` result, which is the licence for
  the rest of the book to model a Transformer as a stack of MLPs. Give the
  crossover for real models rather than in the abstract.
- **MoE accounting.** Total versus activated parameters, sparsity as `E/E_a`,
  and what that does to arithmetic intensity: an MoE's effective intensity is
  `E_a/E` of the dense equivalent, so its critical batch size is `E/E_a` times
  larger. Derive the MoE critical batch size on MI300X here. Shared experts and
  fine-grained experts as modifiers on the count.
- **KV cache, and the attention variants that shrink it.** Shape
  `[2, S, L, K, H]`, size in bytes, and the observation that a handful of
  long-context sequences can exceed the parameter memory. Set up Chapter 11.

  **Multi-head latent attention belongs here, not in the MoE chapter.** The
  previous draft of this outline derived MLA in Chapter 7, which is wrong on
  reflection: MLA is an attention mechanism that happens to appear in a model
  that is also sparse. Its content is exactly the accounting this section already
  does, so it sits naturally as the last step in the sequence MHA to MQA to GQA
  to MLA: same question each time, how many bytes of cache per token, and what
  did you pay in parameters or quality to get there. Two things follow. Chapter 7
  gets to be purely about sparsity instead of becoming the DeepSeek chapter, and
  this section gets to finish its own KV story instead of deferring a third of it
  forward. The serving economics stay in Chapter 11.
- **Gradient checkpointing.** Two named policies with their FLOP costs, motivated
  by an actual activation-memory figure for a model we care about.
- **MFU, and why it is not hardware utilization.** New bullet, and it has to be
  here because every chapter from 7 onward quotes an MFU figure and no chapter
  currently defines one. Model FLOPs utilization is the `6 * params * tokens` count
  above, over elapsed time, over Chapter 2's dense peak. Hardware FLOPs utilization
  counts the FLOPs the device actually issued, which rematerialization inflates: a
  run with full remat can sit at 55% HFU and 40% MFU with nothing whatsoever wrong
  with it. Published figures rarely say which they are, so this book always says.
  It sits immediately after gradient checkpointing on purpose, because remat is
  exactly what separates the two numbers.
- **Summary table** covering dense and MoE side by side.

**Worked problems.** Back out achieved FLOPs utilization from a published
training cost (the DeepSeek v3 exercise is excellent and reusable); compute the
KV cache for Llama 3 70B at 128k context and say what that means for batch size;
compute the MoE critical batch size for a given `E` and `E_a`.

### Chapter 6 — How to Parallelize a Transformer for Training

`training.md` · **You added seven more GPUs and got four times the throughput.
Where did the rest go?**

Dense models only; expert parallelism is Chapter 7. The chapter's real product
is a decision procedure, and the structure is deliberately repetitive: each
strategy gets the same five-part treatment so the reader learns the *move*, not
five separate facts.

For each of data parallel, FSDP/ZeRO, tensor parallel, pipeline parallel and
context parallel:

1. **What it shards**, as a one-line sharding of In / W_in / W_out / Out.
2. **Why do this, why not do this.** Qualitative motivation before algebra.
3. **The algorithm**, as a numbered listing with each collective annotated
   *(on critical path)* or *(not on critical path, can be overlapped)*. This
   annotation is how the reader learns that DP's all-reduce is forgiving and
   TP's is not, and it is nearly free to write.
4. **The roofline.** Set `t_math > t_comms`, solve for a clean inequality, and
   substitute MI300X and Infinity Fabric constants to get a real number.
5. **Predict then measure.** Run it, show the trace, say whether the bound held.

**Settle the sequence-parallelism confusion where context parallelism is
introduced**, in one sentence, because readers arrive with the two words fused and
MaxText exposes both as separate axes. Sequence parallelism in the Megatron sense
is a companion to tensor parallelism that shards norms and residual activations
along the sequence axis to save activation memory; context parallelism shards
attention itself and needs a ring-style KV exchange. Different collectives,
different reasons, and a reader who conflates them writes a config that is silently
wrong rather than loudly broken.

Then the parts that only exist once you have all five:

- **How they compose.** Which combinations put which collective on which mesh
  axis, and the interaction with node topology: what must stay inside a node and
  what can cross. On AMD this is sharper than on TPU because intra-node and
  inter-node bandwidth differ by a large factor. Chapter 4's multi-process section
  is the mechanism; this is where it gets spent.
- **The optimal split.** FSDP comms grow with the data axis while TP comms
  shrink with it, so the worst case is minimized where they meet. Derive it.
- **Memory, not just time.** Optimizer state, activations, remat, and the
  parameter ceiling for pure data parallelism at 192 GiB per device. Several
  strategies are chosen for memory reasons and the time roofline never explains
  that. Gradient accumulation belongs here as the lever that decouples the two:
  it buys a large global batch without the memory of one, at the cost of more
  steps. It is also where the two meanings of "critical batch size" from Chapter 1
  finally meet, because the global batch is bounded below by the ridge point and
  above by convergence, and the DP degree has to fit between them.
- **Low precision, as a parallelism decision.** New section, and a real hole while
  quantization was treated as a purely inference-side topic.
  On MI300X fp8 is exactly 2x bf16, so training in fp8 halves `t_math` and
  therefore *moves every inequality in this chapter*: it doubles the critical
  batch size, makes TP go communication-bound sooner, and changes which strategy
  wins at a given scale. That is a scaling result, not a numerics footnote, and it
  belongs next to the inequalities it perturbs.

  Keep it tight: what fp8 does to each roofline above, the practical scaling
  recipes, and the gfx942-versus-gfx950 format split introduced in Chapter 2. It
  is also the best-supported thing in the whole book on the software side, since
  AMD's ROCm MaxText fork ships `nanoo_fp8` for MI300X and `fp8` for MI355X as
  documented benchmark configurations, so this section can be measured rather
  than derived. Chapter 11 handles the inference side: weight-only quantization,
  KV cache quantization, fp4 and fp6.
- **A decision procedure.** Explicit, as a flowchart or a short ordered
  checklist. The source book leaves this implicit and it is the single most
  common thing readers want. Cover the small-model, large-model, and
  large-batch regimes separately, and make precision an input to it rather than
  an afterthought.

**Pipeline parallelism gets full treatment here**, including a roofline, unlike
the source book which declines to derive one on the grounds that pipelining
matters less on TPU. On a scale-out Ethernet fabric it is a first-class
strategy and skipping it would be a real hole.

**Worked problems.** From a trace alone, determine the parallelism strategy and
per-device batch size; compute what fraction of step time *should* be the
all-reduce and compare; for a given model and 64 GPUs, pick a strategy and
justify it against the inequalities.

**End the chapter by pointing at Chapter 8.** A reader who has just chosen a
strategy is about to run it and find out they are at 22% MFU, and Chapter 8 is
readable from here: everything in it except the MoE-kernel section stands on this
chapter and Chapter 3. One line, so a dense-model reader does not have to get
through sparsity first to find the triage list.

### Chapter 7 — Mixture-of-Experts at Scale

`moe.md` · **Only a fraction of the parameters run per token, so why isn't it a
fraction of the time?**

The most technically distinctive chapter and the one with the least prior art.
Nearly everything here is a failure mode that does not exist in dense models.
Chapter 5 did the accounting; this chapter does the systems.

- **Why the MoE roofline is different.** A dense model's roofline predicts well;
  an MoE's does not, because effective FLOPs depend on routing decisions made at
  runtime. Set up the gap between the naive prediction and reality here, then
  spend the chapter closing it.
- **Routing.** What a router computes, token-choice versus expert-choice, the
  auxiliary load-balancing loss and what it is trading against. Enough
  mechanism that the reader can reason about imbalance rather than just observe
  it.

  **Two lines on router numerics, because they are cheap and the failure they
  prevent looks like a hardware fault.** The router runs in fp32 even when the
  rest of the layer is bf16 or fp8, and it usually carries a z-loss on the logits.
  A router that goes numerically unstable produces a loss spike or a NaN, and the
  reader's first instinct in a book full of hardware will be to suspect the
  hardware. Name it once here so that instinct is corrected before Chapter 10.
- **Load imbalance.** What it looks like in a trace: expert GEMMs of visibly
  unequal duration with the step gated on the slowest. How to quantify it from
  a profile, and how it varies with the data and over the course of training.
- **Capacity, dropping, padding, and going dropless.** The quiet FLOP thief. Fixed
  capacity means padding when an expert is underfull and dropped tokens when it
  overflows, both invisible in wall-clock time and very visible in achieved MFU.
  How to measure the waste from a profile and how capacity trades against
  quality.

  **Then the dropless alternative, which is where modern implementations have
  landed and which the previous draft of this outline left out.** Instead of
  padding to a fixed capacity you size each expert's matmul to whatever actually
  arrived (`capacity_factor: -1` in MaxText; confirm the field name when writing).
  No padding, no dropping, exact FLOPs, and in exchange the expert matmul becomes
  a ragged shape rather than a rectangular one. **That trade is the hinge of the
  whole chapter**: it converts a FLOP-efficiency problem into a kernel-availability
  problem, and on AMD in JAX the kernel is the part that is missing. Set that up
  here and let the next two bullets pay it off.
- **Three ways to implement an expert layer, and what each one costs.** New
  section, and the chapter's spine. The choice is made in one or two config fields
  and it moves the FLOP bill by a factor of `E / E_a`, which for a fine-grained
  model is an order of magnitude. Give the arithmetic for each rather than a
  ranking, because the ranking flips with `E`, `E_a` and the hardware:

  1. **Dense masked compute.** Every device runs every expert over every token and
     multiplies by a one-hot mask. No dispatch collective at all, and every matmul
     is a plain dense GEMM at full kernel efficiency, which is why toy
     implementations look deceptively good. It also does `E / E_a` times the
     activated FLOPs, which is precisely the sparsity you bought, handed back: 4x
     for Mixtral's 8 experts with 2 active, and 16x for a fine-grained model at 128
     and 8. Survivable in the first case, indefensible in the second.
  2. **One-hot dispatch at fixed capacity.** The GShard formulation: an einsum
     routes tokens into an `[E, capacity, D]` buffer, so the expert matmul is a
     dense GEMM of statically known shape and the compiler is happy. You pay the
     padding and dropping from the section above.
  3. **Sort and grouped GEMM.** Sort tokens by expert, then one ragged matmul over
     variable-sized groups. Dropless, no padding, and the FLOP count is exactly
     the activated one. The entire cost moves into needing a grouped or ragged GEMM
     kernel that stays fast on ragged shapes.

  Say which MaxText knobs select which, since that is what the reader edits, and
  note that a reader whose stack only offers (1) is not doing it wrong: they need
  to know what it costs, not that it is inelegant.
- **Which of the three you can actually get on AMD in JAX.** How expert matmuls
  execute, how those kernels appear in Kernel Stats, and why their efficiency
  depends on the token distribution.

  **The AMD specifics here are the most important finding in this outline and
  they need verifying before the chapter is written.** AMD's fast MoE kernels
  live in AITER: fused routing, block-scaled grouped GEMM, the FlyDSL work that
  is superseding hand-written Composable Kernel templates. All of it is reached
  from PyTorch. The JAX bridge (`ROCm/jax-aiter`, via XLA FFI) exposes attention
  and *dense* GEMM, and as far as we can tell **exposes no grouped or ragged MoE
  GEMM at all.** So implementation (3) above is the one a JAX user on ROCm cannot
  simply pick up: the expert matmuls are XLA-generated, or they are yours to write
  in Pallas or Triton, and Pallas on ROCm routes through the Triton backend and is
  labelled experimental (Mosaic GPU is NVIDIA-only).

  If that holds, it is a genuine result and the chapter should lead with it
  rather than bury it: the gap between AMD's best MoE kernels and what a JAX user
  can reach is the single largest performance factor in this chapter, and
  quantifying it is something nobody has published. Two measurements settle it, and
  they are the most valuable numbers in the chapter. First, XLA-generated expert
  GEMMs against the AITER figures AMD publishes, stated as a ratio. Second, the
  three implementations above against each other on the same model, at the same
  `E` and `E_a`, since a reader forced away from (3) needs to know whether (1) or
  (2) is the better consolation prize and the answer is not obvious. Treat CK as
  the legacy path it now is.
- **All-to-all dispatch and combine.** The two collectives that define MoE
  performance once experts are spread over devices. Cost derived from Chapter 4's
  model, including the top-`E_a` ragged variant where cost scales with
  `min(E_a / |Ex|, 1)`.

  **The AMD-specific result here is a good one, and the outline should not bury it
  under the bad one.** All-to-all is the collective a switchless full mesh is best
  suited to: every device has a direct link to every peer, so each of the seven
  links can carry one peer's share concurrently, with no switch to contend for and
  no multi-hop forwarding. A ring schedule lights a fraction of the links at any
  instant; an 8-way all-to-all can in principle light all seven at once, which puts
  the whole 896 GB/s of per-GPU egress in play. Predict it, then check it against
  Chapter 4's sweep, and be honest that RCCL's choice of schedule is what decides
  whether the prediction lands. **Inside a baseboard, MoE dispatch should be
  cheap**, and that is worth stating clearly because the reader expects the
  opposite.

  Then the cliff, which is the same fact from the other side. The ninth GPU is over
  the NIC, so an expert axis that crosses the baseboard trades a 128 GB/s direct
  link for a share of node egress. That is the central placement question of the
  chapter: **keep `Ex` inside the node and spend the slow axis on something that
  tolerates it.** Mark the inter-node arithmetic **[analytical]**.
- **Expert parallelism.** Full five-part treatment as in Chapter 6, then how EP
  composes with DP, FSDP, TP and PP, and which mesh axis all-to-all should land
  on given the node topology. **This is where most real MoE performance is won
  or lost.** This is also the chapter's best argument for `shard_map` over
  automatic partitioning, per Chapter 4: expert routing is the canonical case
  where you want to write the collective yourself.

  **Open with memory rather than with communication**, which is the opposite of how
  EP is usually introduced and the more honest motivation. An MoE has `E` times the
  MLP parameters at `E_a / E` of the MLP FLOPs, so relative to a dense model of the
  same quality it is memory-hungry and FLOP-light, and the optimizer state scales
  with the total parameter count rather than the activated one. That is why the
  expert axis exists at all: you shard by expert because the weights do not fit,
  and only then discover you have bought an all-to-all. The dispatch tensors are
  also large enough that remat policy interacts with routing, which is worth a
  sentence pointing back at Chapter 5.
- **Anatomy of three real models.** Mixtral 8x7B as the simple case, Qwen3
  30B-A3B as the fine-grained one, DeepSeek v3 as the elaborate one: shared
  experts that run for every token, and the published parallelism configuration
  with an explanation of why each degree was chosen. Choosing these three is not
  arbitrary: all three are in the pre-optimised model list for AMD's ROCm MaxText
  fork, so they are the MoE models we can actually run and measure. DeepSeek's
  MLA is accounted for in Chapter 5 and its serving consequences in Chapter 11;
  reference, don't re-derive.
- **The four numbers to log for every MoE run.** Short closing bullet, and the
  most reusable thing in the chapter: tokens per expert as a histogram rather than
  a max, the dropped-token fraction (or the ragged-shape distribution if dropless),
  the achieved efficiency of the expert GEMM against a dense GEMM of equivalent
  size, and the all-to-all share of step time. Each has a named source in a
  profile, each maps onto one of the failure modes above, and none of them is on
  by default. A reader who instruments these four can diagnose their own MoE
  without this chapter, which is the correct ambition for it.

**Scope boundary with Chapter 11, stated so the writing does not drift.** This
chapter owns MoE *mechanism and training*: routing, imbalance, capacity, dispatch
implementation, expert parallelism, all-to-all. Chapter 11 owns MoE *at decode*,
because the reader needs the two-regime model before MoE decode makes any sense.
The seam is the critical batch size: this chapter derives why sparsity inflates it,
Chapter 11 shows why that inflation is close to fatal when you are serving.

**Worked problems.** From a trace, estimate routing imbalance and its cost in
step time; determine whether a run is dropping tokens; decide whether a given
expert-parallel degree helps or hurts; compute the all-to-all cost crossing one
node boundary versus staying inside; for a given `E` and `E_a`, work out at what
grouped-GEMM efficiency the sort-and-group implementation stops beating dense
masked compute, which is the calculation a reader on a stack without a ragged
kernel actually has to do.

---

## Part III — Training in Practice

Part II says what performance you should get. Part III is about the gap between
that and what you actually get, and then about closing that gap twice on real
models. Chapter 8 is where a training run goes from 22% MFU to something
defensible, which is the book's whole promise cashed out; Chapters 9 and 10 apply
the entire book to one model each, dense then sparse, same method both times. Both
capstones are gated on MaxText only, which AMD supports; see
[Sequencing](#sequencing).

**The capstones live here rather than at the end, which is a change from the
previous draft and is recorded so it does not drift back.** They are training
chapters, so putting them inside the training arc keeps every training topic
contiguous and every "what happens next" pointer facing forward. The old ordering
put them last, after the serving material, which meant the sparse capstone had to
point *backwards* to tell the reader what to do with the checkpoint it had just
produced.

### Chapter 8 — Getting to Roofline

`getting-to-roofline.md` · **The prediction says 40% MFU. You measured 22%.**

New chapter, and one the source book has no equivalent of at all. It is where
the deep tooling material from the old profiling chapter belongs, because here
the reader has a specific question the tool answers.

**Everything except the MoE-kernel section is readable straight after Chapter 6.**
Say so in one line at the top, matching the pointer at the end of Chapter 6, so a
reader who is training a dense model right now and sitting at 22% MFU does not have
to get through sparsity first. The chapter sits after Chapter 7 because it needs
the sparse vocabulary to be *complete*, not to be useful.

- **A triage order.** Given a gap, what to check and in what sequence. Cheapest
  checks first: is the device even busy, are the collectives overlapping (one line,
  cross-referencing Chapter 4, which owns that question), is the kernel selection
  right, is anything falling back to a slow path, is there a bubble.

  **Put host starvation first, because it is common and nobody looks for it.** A
  step-time gap caused by the input pipeline looks nothing like a kernel problem
  and is invisible if you only read Kernel Stats, which is where everyone starts.
  It also has a nasty interaction with our own tooling: the XProf Input Pipeline
  page is one of the broken views from Chapter 3, so the reader cannot use the
  obvious instrument and needs the trace-viewer symptom instead, which is device
  rows going quiet while host rows stay busy. That combination, a frequent cause
  plus a broken detector, is exactly what this chapter exists for.
- **Kernel selection and tuning.** hipBLASLt heuristics and offline tuning,
  where rocBLAS still gets used, Triton on ROCm, and AITER. What to do when the
  autotuner picks badly. Real before-and-after numbers. Composable Kernel gets
  named as the legacy path rather than the future, since AMD is moving AITER off
  CK templates onto a Python DSL over MLIR; check the state of that before writing
  and do not present CK as the destination.
- **Fusion.** What XLA fuses and what it does not, how to read fusion decisions
  from HLO, and the cases worth forcing.
- **Attention kernels.** Flash-style attention on AMD, what the kernel looks
  like in a trace, and how to tell which implementation you actually got. This
  surprises people constantly. The concrete JAX-on-ROCm answer to check is
  `ROCm/jax-aiter`, which bridges AITER's flash attention into JAX over XLA FFI
  with a `custom_vjp` so gradients still flow: that is the one place where a JAX
  user on AMD can reach vendor kernels without going through PyTorch, and it is
  worth a measured before-and-after. It is also alpha, so date the claim.
- **When XProf isn't enough.** The escalation path, moved here from Chapter 3.
  `rocprofv3` and rocprof-compute for cache hit rates, MFMA utilization, LDS
  bank conflicts and memory coalescing. TraceLens for large multi-node
  timelines. Concrete handoff recipes: given a kernel name from Kernel Stats,
  the exact invocation that profiles just it, and how to extract its ISA. The
  `fw101` material feeds this directly.
- **What occupancy does and does not tell you.** New section, and it earns its
  place defensively: this is the number the reader is most likely to misread, and
  the misreading costs a week. Chapter 3 already told them XProf reports occupancy
  as 0 on AMD and to escalate to `rocprofv3`. This is where they arrive holding a
  real occupancy figure, and the useful thing to say is **that low occupancy is
  usually not the bug.**

  Keep it to three moves and resist the fourth. What the number *is*: resident
  waves per SIMD over the maximum of 8, set by whichever of four resources runs out
  first (VGPRs, SGPRs, LDS, workgroup and barrier slots), with the registers
  per-SIMD and the LDS per-CU, which is the unit mismatch that makes hand-computed
  occupancy disagree with the profiler. Then the result that matters: on a measured
  MXFP8 MFMA sweep, eight independent accumulator chains per wave hold roughly 97%
  of the matrix peak at **12% occupancy**, and beat a two-chain kernel running at
  96% occupancy. Occupancy and per-wave tile size compete for one 512-register
  file, so buying waves costs arithmetic intensity: **the same roofline argument as
  Chapter 1, one level down the stack.** Finally, what to read instead: `MfmaUtil`
  and `VALUBusy` tell you whether the engine is fed, which is the actual question.

  The fourth move, which we do not make, is teaching the reader to tune registers
  and tiles. That is kernel authoring and it is a stated non-goal. **The section's
  whole job is to stop a JAX user from chasing a number they cannot move**, since
  the tile shapes come from hipBLASLt or AITER or XLA, not from them. Frame it as
  triage: if the matrix core is already saturated, occupancy is a distraction, stop
  here.

  AMD published a genuinely good from-first-principles treatment of this in July
  2026, [Occupancy Math on the AMD MI355X GPU
  (CDNA4)](https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html),
  which is the right thing to link rather than reproduce. Its
  CDNA3-versus-CDNA4 worked example is worth citing directly: the
  *same* kernel is LDS-bound at 25% occupancy on MI300X and register-bound at 50% on
  MI355X, purely because LDS went from 64 KB to 160 KB per CU. That relocation of
  the bottleneck teaches more than either number does. Two cautions when using it:
  its constants are gfx950 throughout, so the MI300X reader needs the 64 KB LDS
  figure substituted, and its headline throughput is an issue ceiling measured with
  register-resident operands and no memory traffic, which the post says plainly and
  we should repeat.
- **When to write your own kernel**, and when not to. Mostly not to.

**Worked problems.** Given a profile at 22% MFU, produce a ranked list of
suspects; identify from Kernel Stats whether a matmul got the tuned kernel.

### Chapter 9 — Training Llama 3 on MI300X

`llama.md` · Config and parallelism strategy justified against Chapter 6's
inequalities, MaxText-on-ROCm setup, the capture recipe at production scale
(a few steps out of thousands, trace file sizes, multi-host capture and where
the per-host traces land), per-layer breakdown, and MFU against the Chapter 2
roofline. Establishes the method Chapter 10 reuses.

**This chapter is in better shape than it looks.** AMD's
ROCm MaxText fork ships Llama 3 8B and 70B as pre-optimised configurations, with
Llama 3.1 405B documented as a multi-node benchmark, so the config and launch
path are given rather than invented. Use their configuration as the starting
point and spend the chapter explaining *why* each degree was chosen against
Chapter 6, which is the part their documentation does not do and the part the
reader needs.

Three additions worth a section each, all of them things that decide whether a real
training run finishes and none owned by any other chapter. **Checkpointing:** what a
checkpoint costs in time and bytes at this scale, how often to take one, and how
long a restart takes, because at 405B on a shared cluster this is a first-order
throughput term and no chapter currently owns it. **The input pipeline at scale:**
how a sharded dataset gets fed to a multi-process mesh without every host reading
the same shard, and what a deterministic resume costs. It is checkpointing's sibling
and the other thing that stalls real runs; Chapter 8 taught the reader to *recognise*
host starvation in a trace, and this is where they see it prevented at production
scale rather than diagnosed after the fact. **The failure that is not a
performance problem:** AMD's own benchmark scripts disable an RCCL feature to avoid
NaN losses on MI355X, which is a perfect, real, and slightly uncomfortable example
of the kind of thing that no roofline predicts. One footnote, honestly told, is
worth more than a page of generalities about robustness.

### Chapter 10 — Training DeepSeek-V2-Lite on MI300X

`deepseek.md` · The sparse capstone, and the hardest thing in the book that we can
actually run end to end. Same method as Chapter 9, harder model: predict from
Chapters 5 and 7, measure, explain the gap. MoE routing and imbalance, expert
parallelism, all-to-all placement against the eight-GPU mesh ceiling, MLA's effect
on the memory profile, and achieved MFU against the Chapter 2 roofline. The four
numbers from Chapter 7 are the instrumentation, so this chapter is where they earn
their keep.

**Converted from a serving capstone to a training capstone, which retires the
book's largest risk.** The serving version needed `decode.py` working on ROCm plus a
JAX serving path, neither of which exists, and Chapter 12 now explains why chasing
them was the wrong call anyway. The training version needs a model that AMD's own
ROCm MaxText fork already lists as pre-optimised, which DeepSeek-V2-Lite is. So
this chapter goes from the riskiest in the book to roughly as safe as Chapter 9.

**And Part III is better for it**: a dense capstone followed by a sparse one, same
method both times, which makes the second chapter a genuine "now the hard version"
rather than a change of subject. It also means every measured number in the book
comes from a training run, which is a cleaner promise than half-training and
half-serving.

Qwen3 30B-A3B is the alternative if DeepSeek-V2-Lite disappoints, and Mixtral 8x7B
is the simpler fallback below that. All three are in the same pre-optimised list, so
switching costs a re-run rather than a rewrite. Close the chapter by pointing forward
at Part IV for what happens to the checkpoint next, which is now a forward pointer
rather than the backward one the previous ordering forced. Make no serving claims
here at all: this is a training chapter and Chapters 11 and 12 own that ground.

---

## Part IV — After Training

Everything that happens once the loss curve is acceptable: what the thing costs to
serve, how it leaves JAX, and where to read next. The two serving chapters are the
shortest treatment in the book of the largest subject in the industry, on purpose;
see [What this book is](#what-this-book-is) for why serving is covered honestly
rather than deeply. They are a pair and should be read as one: Chapter 11 is the
arithmetic a single request obeys, Chapter 12 is what a serving engine does about
the parts the arithmetic cannot fix. Chapter 13 then closes the book, and ending on
the handoff rather than on a capstone is the right note: the last thing the reader
is told is where to go next.

### Chapter 11 — How to Think About Inference

`inference.md` · **Serving is not training with `no_grad`.**

Promoted from a section to a chapter. Inference has a completely different
profile signature, a different roofline, and a different set of legal sharding
strategies, so the training intuitions built over the previous six chapters are
actively misleading here and the reader needs to be told which ones invert.

**Scope this chapter as arithmetic, not as a serving guide.** It answers "how much
memory do I need to serve this, at what batch size, and what will a token cost",
which the roofline settles cleanly and which every reader who trains a model
immediately needs. It does not teach a serving system: that is Chapter 12's much
smaller job, and the book's focus is training. Keeping this chapter analytical is
what lets it be short, standalone and correct without depending on a stack we do
not run. Everything in it is **[analytical]** and the opening line should say so
once rather than tagging every claim.

- **What are we optimizing?** Latency, throughput, TTFT, cost per token, and the
  fact that these conflict. Name the three workload shapes (offline batch, chat
  streaming, agentic/long-CoT) because they want different answers.
- **Cost per token, in dollars.** Short section, high value. Everything else in
  the book is measured in seconds and bytes, but the question a reader is usually
  being asked is what serving this costs, and the conversion is a one-liner:
  GPU-hour rate divided by achieved tokens per second. The source book's most
  quoted passages are the ones that answer "how much would this cost", and it is
  the natural unit for the latency-throughput tradeoff, since it makes the price
  of a latency target explicit instead of rhetorical. Carry the resulting figure
  into Chapter 12.
- **Two regimes.** Prefill is compute-bound and looks like training. Decode is
  memory-bandwidth-bound and dominated by weight loading. Ask the same
  compute-versus-memory question of linear ops and of attention, separately, for
  each regime, and fill in the 2x2. Include the arithmetic for why decode is
  bandwidth-bound: it is the single most useful mental model for inference
  performance.
- **The step-time formula.** Minimum decode step time as
  `(params + batch * KV cache) / β_hbm`. If the reader takes one thing from the
  chapter, this is it.
- **KV cache economics.** Growth with context, memory cost against 192 GiB, and
  the batch size at which you run out. GQA, MLA and quantized caches as the
  levers. Tables sweeping batch size, as the source book does, because the
  saturation and OOM points are the whole story.
- **Sharding for decode.** Why FSDP is actively harmful at decode time and why
  data parallelism is pointless. What remains is tensor parallelism, and the
  important twist that when you are bandwidth-bound rather than FLOPs-bound you
  can shard *past* the training bound to buy latency. KV cache sharding and the
  all-to-alls it costs.
- **MoE at decode.** Where Chapters 5 and 7 collide: `E/E_a` inflates
  the critical batch size to numbers that are hard to reach, and expert placement
  interacts with the memory ceiling. This is the hardest problem in the book and
  it deserves to be named as such.
- **Quantization for inference.** Chapter 2 owns the formats and Chapter 6 owns
  fp8 *training*, so this section is specifically the inference levers: weight-only
  quantization, KV cache quantization, and what each does to the step-time formula
  above rather than to the FLOP ceiling. That distinction is the whole point at
  decode: you are buying bandwidth, not FLOPs, which is why weight-only
  quantization helps decode far more than it helps prefill. MI355X's fp6/fp4 as the
  forward look, including the CDNA4 oddity from Chapter 2 that fp6 costs the same
  as fp4.
- **Speculative decoding.** Short, but present, because it is the cleanest
  illustration that the step-time formula above is the thing that governs: it wins
  precisely by converting spare FLOPs into fewer weight loads, which is the one
  trade a bandwidth-bound decode rewards. One derivation and an acceptance-rate
  sensitivity, then stop. The implementation lives in the serving engine, not in
  our stack.

**Then hand off to Chapter 12 by naming what this chapter cannot fix.** The
closing section should state the three problems that survive a perfect
single-request roofline: requests arrive at different times and in different shapes
(scheduling), the KV cache is allocated dynamically and fragments (memory
management), and prefill and decode want different hardware (placement). Those
three are also the honest reason serving engines exist and the reason we are not
writing one, so ending on them sets up the next chapter's "hand it to vLLM" argument
instead of leaving it to arrive as an anticlimax. The two chapters being adjacent is
what makes that handoff land; in the previous ordering a capstone sat between them.

**Worked problems.** Explain why decode throughput barely improves when you
double the FLOPs; compute the maximum batch size for a given model and context
at 192 GiB; decide whether fp8 weights help at a given batch size; convert a
measured tokens/sec into dollars per million tokens.

### Chapter 12 — Getting Your Model Into Production

`serving.md` · **You trained it in JAX. Now it has to serve traffic, and that
probably isn't JAX.**

New chapter, and the honest answer to a question the source book never asks: what
happens to the checkpoint after training finishes.

**The scope of this chapter is deliberately narrower than its previous draft, and
the reason is worth stating plainly at the top of the chapter itself.** Production
serving on AMD means exporting weights and running them under vLLM or SGLang. That
is not a compromise forced on us by JAX, it is what the whole industry does,
including the teams with the most AMD capacity. Four facts make it unambiguous:

- **There is no JAX serving engine for ROCm.** JetStream was archived on
  1 February 2026 and its functionality moved into `vllm-project/tpu-inference`,
  which is TPU-only by name and design.
- **MaxText's blessed inference path is itself an out-of-tree vLLM plugin.** Even
  on TPU, Google's own answer to serving MaxText is now vLLM.
- **AMD's MaxText fork is a training path** and documents no inference story.
- **vLLM and SGLang on ROCm are genuinely good**, not merely popular: AITER
  attention backends, MXFP4, and published disaggregated-serving results
  competitive with B200.

So this chapter does not build a JAX serving stack, and it does not pretend the
absence of one is a gap in the book. **It teaches the handoff, and it teaches
enough of the serving concepts that the reader can size and tune what they hand
off to.** That is a genuinely useful chapter and a much less risky one: nothing in
it is gated on `decode.py` working on ROCm.

- **The handoff, concretely.** This is the section nobody has written and the
  reason the chapter exists. Orbax checkpoint to a HuggingFace-format checkpoint,
  what MaxText's conversion scripts do and where they break, quantization on the
  way out (AMD's Quark toolkit, and the fp8 format question from Chapter 2 showing
  up again as a checkpoint compatibility problem), then loading it under vLLM on
  ROCm and confirming the outputs match. **Verify this path end to end before
  writing**; it is the chapter's spine and the one thing here that could surprise
  us. It is also the natural place to close the loop on the capstones: the
  checkpoint being exported is the one Chapter 9 or Chapter 10 just produced.
- **What the serving engine is doing with your weights.** Continuous batching,
  chunked prefill with its roofline, paged attention and KV cache management,
  prefix caching and its routing-affinity consequence. Taught as properties of the
  workload rather than as a library tour, which is what makes them transfer:
  these are the same arithmetic as Chapter 11, applied to many requests instead of
  one. Keep it tight. The goal is a reader who can read a vLLM configuration and
  predict what it will do, not a reader who could reimplement it.
- **Disaggregated prefill and decode.** Presented as the endpoint of an escalating
  argument: naive batching, then interleaved, then disaggregated, with the specific
  failure that motivates each step. The two real advantages are independent scaling
  and specialized sharding. The cost is moving KV cache over the network, which is
  a Chapter 2 bandwidth question, and it is the best single demonstration that the
  book's arithmetic still governs a stack we did not write.
- **What this costs and how it is operated.** Sizing a deployment against an SLO,
  using the dollars-per-token model from Chapter 11. Then briefly: what to autoscale
  on, load balancing under heterogeneous request shapes, and reliability. Short.
  This is the part of the chapter most likely to become filler, so cap it.

**Cut from the previous draft**, and recorded here so it does not creep back: the
prefill/generate/transfer thread anatomy of a JAX serving engine, multi-node
serving as its own section, and any attempt to measure a JAX decode path. The first
two are things a reader gets from vLLM's own documentation; the third is a porting
project, not a writing project.

**Worked problems.** Given a request trace and an SLO, size a deployment; compute
the KV transfer cost of disaggregation and say whether it pays; given a trained
checkpoint and a target quantization, work out the served memory footprint per
GPU.

### Chapter 13 — Conclusions and Further Reading

`conclusion.md` · Closing thoughts, what we got wrong, and pointers out:
ROCm documentation, the source book, the HuggingFace Ultra-Scale Playbook,
Stas Bekman's ML Engineering handbook. Acknowledgements naming the people
consulted, which is both correct and a credibility move.

**One level down is a real reading path and worth signposting.** This book stops at
the kernel boundary by design. For the reader who wants to cross it, the ROCm blog's
occupancy-math post is the best entry point we know of, along with the CDNA4 ISA
guide and the CDNA4 architecture whitepaper it cites. Framing them as "the next
level down" rather than "further reading" is more useful, because it tells the reader
what kind of question they answer.

**Point serving readers somewhere real.** Since the book deliberately stops at the
handoff, this is the place to list the vLLM and SGLang ROCm documentation, AMD's
inference optimization guides and AITER, as the genuine next step rather than as
consolation. A reader who trained a model with this book and then serves it well
with someone else's has been served correctly.

---

## Appendices

Two of them, both load-bearing, and neither one a chapter: they carry no
`section_number` and sit after Chapter 13 in the navbar. The previous draft
promised both in passing without giving either a file, which is how they end up
scattered through three chapters instead.

**Appendix A — Installing JAX on ROCm.** `appendix-install.md`. The four wheels in
the right order (`jax-rocm7-pjrt`, `jax-rocm7-plugin`, `jaxlib`, `jax`), the ROCm
version matrix, building from source, and the known-broken combinations. Chapter 3
gets a container pull and a pointer here. `gpu_profiling/readme.md` is the source.

**Appendix B — How We Measure.** `appendix-protocol.md`. The protocol from
[Conventions](#conventions) written out once: container tag, warmup and repeat
counts, median rather than mean, clock and power state, device count and
partitioning mode. Every **[measured]** number links here. So does every
**[analytical]** one, because the useful thing to say about an unmeasured claim is
what it would take to measure it.

Both are the fastest-rotting prose in the book, which is the argument for putting
them somewhere that can be revised without touching a chapter.

---

## Existing assets

What already exists and where it lands. **Note the shape of this table: almost
everything we have written feeds Chapters 2, 3 and 8.** Part II is close to
greenfield on the writing side. That is the honest scheduling picture and the
previous outline obscured it by making profiling the centre of the book. The
mitigating discovery is at the bottom of this section: for Parts II and III the
*runnable* material largely exists even though the prose does not.

| Asset | Feeds |
|---|---|
| `gpu_profiling/readme.md` | Ch 3 setup, first trace, trace layout |
| `gpu_profiling/docs/xla-rocm-profiler-backend.md` | Ch 3 "how a profile gets made" |
| `gpu_profiling/docs/writeup/xprof-for-amd.md` | Ch 3 limitations table (internal source, not linked) |
| `gpu_profiling/docs/writeup/fix-xprof-*.md`, `rocm-pm-sampler-wiring.md` | Ch 3 "fix in progress" column; Ch 8 counters |
| `gpu_profiling/traces/transformer_block/` (8x MI300X, 4x H100, 1x v5e) | Ch 2 translation table, Ch 4 collectives |
| `fw101` XLA pipeline diagrams (`xla-gpu-pipeline`, `xla-hlo-to-thunk`) | Ch 3 thousand-foot view |
| `fw101` PJRT and backend responsibilities (JAX/XLA-4) | Ch 3 thousand-foot view |
| `fw101` kernel-to-Python tracing | Ch 3 "from an HLO op back to a Python line" |
| `fw101` `rocprofv3`, rocprof-compute, ISA dumps | Ch 8 "when XProf isn't enough" |
| `scripts/jax_matmul.py` | Ch 2 worked example, Ch 3 first trace |
| `scripts/transformer_block.py` | Ch 3 training step, Ch 4 collectives and overlap |
| `scripts/transformer_block_H100_pm_counter.py` | Ch 2 translation table (H100 side) |
| `scripts/basic_training.py` | Ch 3 training step |
| `scripts/devlab-llm-scaling-talk-2025.py` | Ch 6 DP, FSDP, TP, `shard_map` stages |
| `tools/parse_xplane.py` | Extracting numbers for prose; not reader-facing |
| ROCm/maxtext + `rocm/jax-training` images | Ch 6 fp8, Ch 7 MoE models, Ch 9, Ch 10 |

Two corrections against the repository as it actually stands. **The internal
audit documents twenty issues, not seven**, so Chapter 3's table is a curated
subset and should say so rather than implying it is exhaustive. And **three of
those writeups are fixes in progress by our own team**, which is the strongest
possible argument for the version-pinning discipline in
[Conventions](#conventions): we are on both sides of that table.

**The three-platform traces are an underused asset, with one caveat.** The same
`transformer_block.py` workload captured on 8x MI300X, 4x H100 and a single v5e is
something almost nobody else has, and it makes the Chapter 2 translation table
empirical rather than a spec-sheet comparison. The caveat is that the device
counts differ, so the traces support per-device and vocabulary comparisons but not
scaling claims, and Chapter 2 must say which it is doing.

**And the biggest asset is one the previous table missed entirely:** AMD's ROCm
MaxText fork and its prebuilt images. They supply Llama 3 8B/70B and Llama 3.1
405B multi-node for Chapter 9, Mixtral 8x7B / DeepSeek-V2-Lite / Qwen3 30B-A3B
for Chapter 7, and documented fp8 configurations for Chapter 6. That materially
changes the scheduling picture for Parts II and III.

Scripts still needed, roughly in order of when they block a chapter: an RCCL
bandwidth sweep (Ch 4), a TP and a PP variant (Ch 6), an MoE block with
instrumented routing (Ch 7), a decode benchmark (Ch 11), and capture wrappers
around the ROCm MaxText configs (Ch 9 and Ch 10). All JAX.

**One addition to that list, and it is the one that decides whether Chapter 7 has a
result or an opinion:** the MoE block needs to implement all three dispatch
strategies from that chapter behind one flag, so the comparison is the same model
and the same tokens with only the implementation varying. Without that, the
kernel-reachability finding is an assertion. With it, it is a table.

One script the previous list had and this one does not: a JAX decode deployment
with a load generator. Chapter 12 no longer needs it. What Chapter 12 needs instead
is a checkpoint export and reload script, JAX out to vLLM on ROCm and back in
again, verifying that the outputs match. That is a much smaller piece of work and
it is the only new dependency the reframed chapter has.

**Screenshots are the largest unplanned cost.** Two dozen or more, and they all
invalidate when the XProf UI or the build changes. Decide before writing Chapter
3 whether they are captured by hand or scripted against XProf's data endpoints.
This is the most likely thing to stall the chapter.

---

## Sequencing

The book is large, so it ships in waves. Each wave is publishable on its own and
answers a question a reader actually has.

**Wave 1 — "What should this run at, what does splitting it cost, and how do I
check?"** Chapters 1, 2, 3, **4**. No MaxText, no large model, no cluster: one
8-GPU node does all of it, which we have.

**Chapter 4 belongs in Wave 1 rather than Wave 2.** Three reasons. It has no
dependency Wave 1 does not already satisfy, since a sharded matmul and an RCCL sweep
need a single baseboard and nothing else. It contains the book's single most novel
measurement, the RCCL bandwidth sweep that nobody has published, so shipping it
early puts the strongest evidence in the first release instead of the second. And
without it, Wave 1 is a profiling guide that does not deliver on the title, whereas
Wave 1 with Chapter 4 in it is a genuine, if short, scaling book.

Order within the wave: Chapter 2's hardware numbers first, since everything else
substitutes into them, verified and dated. Then Chapter 3's limitations table
before the rest of Chapter 3, because everything links to it and it is
independently useful the moment it exists.

**Wave 2 — "How do I parallelize a Transformer?"** Chapters 5, 6. Needs the TP
and PP variants and a Transformer-shaped workload. This is the wave that turns
rooflines into parallelism decisions, and it is the one most at risk of being
deferred indefinitely because the prose does not already exist. Protect it.

**Wave 3 — "What about sparse models, and why am I not at roofline?"** Chapters 7,
8. The differentiated content, and less risky than it appears: AMD's
ROCm MaxText fork lists Mixtral 8x7B, DeepSeek-V2-Lite and Qwen3 30B-A3B as
pre-optimised, so a runnable MoE in JAX on ROCm is close to given. Verify one of
them actually trains before the wave starts. Keep the hand-rolled expert block as
the fallback and consider writing it anyway: instrumenting your own router is
better teaching material than instrumenting MaxText's, so it may be worth having
both, the toy one to explain and the real one to measure. Chapter 8 pairs naturally
with 7 because its MoE-kernel section needs the sparse vocabulary, and because most
of its material already exists in the repository, which makes it the cheap half of
an otherwise expensive wave. If Wave 2 slips, Chapter 8 can ship ahead of Chapter 7
with the MoE section held back.

**Wave 4 — "Show me two real runs."** Chapters 9, 10. The two capstones are a
coherent release on their own: same method twice, dense then sparse, and the wave
where the book's central promise is either kept or visibly not. This is the wave
with the most hardware exposure, so it is also the one to start scheduling machine
time for during Wave 3.

**Wave 5 — "What will it cost to serve, and how do I ship it?"** Chapters 11, 12,
13. Deliberately last, and cheap: Chapter 11 is analytical throughout and needs no
hardware we lack, Chapter 12's only hard dependency is a checkpoint round-trip, and
Chapter 13 can be written whenever. Shipping the serving arc last is also the
schedule agreeing with the stated priorities rather than contradicting them.

Waves 1 and 2 together are a coherent standalone book, and Wave 1 alone is a
defensible short one.

---

## Open questions

**~~Who owns MaxText on ROCm and which models run?~~ Largely answered, and the
answer is good.** AMD maintains a ROCm fork of MaxText with a `rocm-main` branch
tracking upstream, prebuilt `rocm/jax-training` images, and ROCm documentation
covering MI300X, MI325X, MI350X and MI355X. The pre-optimised model list includes
Llama 2 7B/70B, Llama 3 8B/70B, Llama 3.1 8B/70B/405B (multi-node), Llama 3.3 70B,
**Mixtral 8x7B, DeepSeek-V2-Lite and Qwen3 30B-A3B**, with Flash Attention 3, GEMM
tuning, multi-node launch via Primus, and both NANOO fp8 (gfx942) and standard fp8
(gfx950). Three consequences, all already folded into the chapters above: Chapter 7
has three runnable MoE models, Chapter 6 can measure fp8 rather than derive it, and
Chapter 9's configuration is given rather than invented.

**~~Does JAX decode work on ROCm?~~ No longer on the critical path, because the
book stopped needing it to.** It very likely does not in any useful sense:
JetStream was archived on 1 February 2026 and its successor is TPU-only, MaxText's
documented inference path is an out-of-tree vLLM plugin with no ROCm extra, and
AMD's fork documents training only. Rather than treat that as a risk to manage,
Chapter 12 now teaches the export-to-vLLM handoff and Chapter 10 is a training
capstone, so nothing in the book is gated on the answer.

**The replacement question is much easier: does the checkpoint round-trip?** JAX
and Orbax out, HuggingFace format, quantized with Quark, loaded under vLLM on ROCm,
outputs matching. That is Chapter 12's only hard dependency and it is a day of work
to establish rather than a porting project. Do it well before Wave 5, because if
the conversion scripts are broken for a model we care about, that is worth knowing
early and is itself publishable.

**How much of AMD's kernel stack can a JAX user actually reach?** New question,
and it turns out to be a book-shaping one rather than a detail. AITER is where
AMD's fast kernels live, `ROCm/jax-aiter` bridges some of them into JAX over XLA
FFI (flash attention with gradients, dense and fp4 GEMM), Pallas on ROCm works
through the Triton backend and is labelled experimental, and Mosaic GPU is
NVIDIA-only. **Most importantly, we have found no grouped or ragged MoE GEMM
reachable from JAX**, which if true is the central performance fact of Chapter 7
and worth confirming carefully rather than assuming. Verify the whole list against
current wheels before Wave 3, and pin versions when writing it up, because this is
the fastest-moving area in the book.

Two sub-questions that Chapter 7's expansion adds, both cheap to settle and both
blocking a section rather than a sentence. **Which of the three dispatch
implementations does MaxText actually run on ROCm, and under which config fields?**
The names to confirm are `sparse_matmul`, `megablox` and `capacity_factor`, along
with whether the dropless path depends on a Pallas kernel that only exists for TPU,
because if it does then implementation (3) is unavailable rather than merely slow and
the chapter's framing changes. **And does RCCL actually schedule an 8-way all-to-all
across all seven links?** The full-mesh argument predicts it should, which would make
intra-node MoE dispatch cheap and is one of the better findings available to us, but
the prediction is about a schedule RCCL chooses rather than about wiring. Measure it
in the Chapter 4 sweep, since the script is needed there anyway.

**What is the scale-out fabric on the target cluster?** Chapters 2, 6, 7 and 12
all need a per-node egress bandwidth figure and a topology, and multi-node is
where MoE all-to-all either works or does not. We currently have no multi-node
data at all. This is the largest gap between what the book promises and what we
can measure, which is why the measurement promise in
[What this book is](#what-this-book-is) is now scoped to single-node claims. If a
multi-node allocation does become available, note that AMD documents Llama 3.1
405B as a multi-node MaxText benchmark, so there is a ready-made workload and we
would not be starting from a blank script.

Intra-node is settled and needs no further investigation: switchless full mesh,
seven xGMI links per GPU, 128 GB/s bidirectional each on MI300X and 153.6 GB/s on
MI355X.

**Screenshot workflow.** Manual or scripted against XProf's data endpoints.

**Is there a runnable companion?** The site is public but the notebook was
scoped as internally hosted, which external readers cannot reach. Cheapest
resolution is to publish the example scripts in a public repo and link them from
each chapter, keeping the internal notebook for AMD-side walkthroughs.

**~~The HBM bandwidth discrepancy.~~ Resolved.** Recorded here because Chapter 2
should write it up rather than quietly use the right number.

The ROCm collector computes HBM bandwidth at
`xla/xla/backends/profiler/gpu/device_tracer_rocm.cc:586` as
`2 * mem_clock_khz * 1000 * mem_bus_width_bits / 8`, carrying over a comment
from the CUDA path that reads "Times 2 because HBM is DDR memory; it gets two
data bits per each data lane." On MI300X, HIP reports a 1.3 GHz memory clock and
an 8192-bit bus, so that yields `2 * 1.3e9 * 8192 / 8 = 2662.4 GB/s`, which
XProf then renders in binary units as **2479.6**: the figure in our traces.

MI300X HBM3 runs at 5.2 Gbps per pin against that 1.3 GHz reference, which is
four transfers per clock, not two. With the correct multiplier,
`4 * 1.3e9 * 8192 / 8 = 5324.8 GB/s`, the 5.3 TB/s spec figure. So there are two
separate bugs stacked: a 2x undercount from the inherited DDR assumption, and a
decimal/binary unit mismatch in display. The same unit mismatch explains the
other oddity we noticed, that 192 GiB of HBM is reported as 206.1 GB.

**Carry 5.3 TB/s in the book.** Two follow-ups: file the collector bug upstream,
and check whether the same 2x applies on MI355X before quoting its number.
