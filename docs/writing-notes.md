# Writing notes: first drafting pass

Written alongside the first pass of prose over `index.md` and `pages/`, 4 August 2026.
Not published; `docs/` is excluded in `_config.yml`.

This document exists so the drafting decisions are reviewable without reading fifteen
diffs. It records what was written, what was deliberately left empty and why, every
factual claim that could not be verified in this pass, and every place the roadmap in
`structure.md` turned out to be wrong or ambiguous.

## Contents

- [How to review this](#how-to-review-this)
- [What got written and what did not](#what-got-written-and-what-did-not)
- [Conventions I chose, and where I departed from instructions](#conventions-i-chose-and-where-i-departed-from-instructions)
- [Corrections to the roadmap](#corrections-to-the-roadmap)
- [Unverified claims, by risk](#unverified-claims-by-risk)
- [Derivations worth a second pair of eyes](#derivations-worth-a-second-pair-of-eyes)
- [The blocker list, ordered by what it unblocks](#the-blocker-list-ordered-by-what-it-unblocks)
- [Ambiguities I resolved by choosing](#ambiguities-i-resolved-by-choosing)

---

## How to review this

Three things are worth your time, in this order.

**One: the numbers.** Every arithmetic result in the written chapters is mine, computed
from published specifications. [Derivations worth a second pair of eyes](#derivations-worth-a-second-pair-of-eyes)
lists the ones where an error would propagate furthest. The single highest-risk number is
the xGMI direction convention, because a factor of 2 there moves every collective cost in
the book.

**Two: the corrections.** [Corrections to the roadmap](#corrections-to-the-roadmap) lists
four places where `structure.md` says something the source does not support. Two of them
change what a chapter claims.

**Three: whether the blocked sections are blocked for the right reasons.** I was
deliberately strict: a section is blocked if writing it would require asserting a number I
did not compute, an artifact I have not seen, or a software behaviour I have not checked.
If you think I was too strict anywhere, the likely candidates are Chapter 3's XProf tool
tour (blocked only on screenshots) and Chapter 8's escalation recipes (blocked on
verification of commands that exist in `fw101`).

## What got written and what did not

| Page | State | What is missing |
|---|---|---|
| `index.md` | **Complete** | Nothing. Written despite being outside `pages/`; see below |
| `pages/1-rooflines.md` | **Complete** | Nothing. No hardware dependency by design |
| `pages/2-amd-gpus.md` | Substantially complete | The measured matmul (deferred to Ch 3 by design), and the three-platform comparison table |
| `pages/3-profiling.md` | Partial | Tool tour (screenshots), HLO traversal, matmul measurement, training step, problems |
| `pages/4-sharding.md` | Substantially complete | Our own RCCL sweep, the GSPMD/`shard_map` HLO diff, the overlap before-and-after |
| `pages/5-transformers.md` | **Complete** | Nothing. Pure accounting |
| `pages/6-training.md` | Substantially complete | Part 5 (measurement) of each of the five treatments; the fp8 measurement |
| `pages/7-moe.md` | Substantially complete | The two kernel measurements; the AITER reachability survey |
| `pages/8-getting-to-roofline.md` | Partial | Four of seven sections. Occupancy and triage written |
| `pages/9-llama.md` | **Not started** | Everything. Blocked on a training run |
| `pages/10-deepseek.md` | **Not started** | Everything. Blocked on a run *and* on Ch 7's kernel question |
| `pages/11-inference.md` | **Complete** | Nothing. Analytical by design |
| `pages/12-serving.md` | Substantially complete | The handoff section, which is the chapter's spine |
| `pages/13-conclusion.md` | Partial | "What we got wrong" (by construction) and acknowledgements (needs names) |
| `pages/a-appendix-install.md` | Substantially complete | More known-broken rows; a container tag matching our stack |
| `pages/b-appendix-protocol.md` | Substantially complete | A public container tag matching our measurement stack |

Rough shape: **five pages complete, six substantially complete, two partial, two not
started.** The unblocked half of the book is written.

## Conventions I chose, and where I departed from instructions

**Blocked sections are HTML comments, not visible blockquotes.** The instruction was
"commented blocks stating what needs to be done", so the scaffolding is in
`<!-- ... -->` and invisible on the site. Each comment states what the section has to
deliver, what specifically blocks it, and what would unblock it. `grep -rn 'BLOCKED' pages/`
is the work list.

**Pages with blocked sections carry one visible draft note at the top.** This is a
departure and worth objecting to if you disagree. My reasoning: a published page with
empty headings and no explanation reads as broken rather than as in progress, and the
book's own honesty convention argues for saying so. The notes are one or two sentences and
name what is missing. If you would rather the site look finished, deleting those
blockquotes is a one-line change per page.

**The old `> **To write.**` blockquotes are gone everywhere.** They were the skeleton's
scaffolding and they were visible on the site. Their content is either now prose or now in
a `BLOCKED` comment.

**Every page ends with a `## References` section, added to each page's `toc`.** Annotated
with one line on what each link is for, per the instruction to reference only directly
relevant material. Two rules I applied: primary sources (AMD data sheets, ROCm docs, the
paper that introduced the technique) over secondary, and nothing cited that the prose does
not use. `pages/9-llama.md` and `pages/10-deepseek.md` carry provisional reference lists marked
as such, since they have no prose to support yet.

**I wrote `index.md` even though it is not in `pages/`.** It is Chapter 0, it is first in
the notebook order the instruction specified, it has no blockers, and leaving the landing
page as a skeleton while every chapter has prose would be the worst of both. Say if you
would rather I had left it.

**No new figures or screenshots.** Several sections want them; all are blocked on the
same asset question. `assets/img/` has nothing in it and the two XLA pipeline diagrams
plus the profiler pipeline diagram exist only in `fw101` and `gpu_profiling` as source
material.

**Notation:** symbols in inline code (`` `t_math` ``), arithmetic as fenced blocks or
inline code with real numbers and units, per the style guide's calibration example. I did
not use KaTeX, even though the layout supports it, because the skeletons had established
the code-span convention and mixing the two would look worse than either.

**Two mechanical traps found the hard way, both worth adding to the three rules in
`structure.md`.**

**Pipes cannot appear in a table cell, and escaping them does not help.** Kramdown splits
table rows on `|` before inline parsing, and `\|` inside a code span renders as a literal
backslash-pipe rather than a pipe. So a mesh-axis size like `|Y|` simply cannot go in a
table cell: I renamed the columns instead ("Tensor-parallel degree"). Fine in prose and in
fenced code blocks.

**Liquid parses `{{` everywhere, including inside fenced code blocks.** An HLO snippet
containing `replica_groups={{0,1},{2,3}}` is a build failure, not a rendering glitch, and
the error names a line number in a different place. Wrap those in
`{% raw %}` / `{% endraw %}`. Three places in `sharding.md`.

## Corrections to the roadmap

**Four places where `structure.md` says something the source does not support.** Two are
material.

**1. MaxText's `z_loss_multiplier` is not a router z-loss.** The roadmap says the router
"usually carries a z-loss on the logits" and implies MaxText provides one. It does not:
`z_loss_multiplier` applies to the output vocabulary logits in the cross-entropy, not to
the router. `moe.md` now says this explicitly and tells the reader they would have to add
a router z-loss themselves. **Material**, because the roadmap's version would have told
readers they were protected when they are not.

**2. The dropless grouped-GEMM path is worse than "merely slow" on GPU.** The roadmap asks
whether "the dropless path depends on a Pallas kernel that only exists for TPU, because if
it does then implementation (3) is unavailable rather than merely slow and the chapter's
framing changes." It does, and the framing did change. MaxText's default
(`sparse_matmul: true`, `megablox: true`) selects a Pallas TPU kernel that runs in
**interpret mode** on a GPU mesh, silently, with no config validator objecting. The
platform-general path is `megablox: false`, which selects `jax.lax.ragged_dot`.
`moe.md` now leads its AMD section with that and gives four numbered pieces of advice.
**Material**, and the most useful finding of this pass.

**3. The MoE critical batch size claim in Chapter 1's worked problem needed splitting.**
The roadmap asks the reader to "watch the ridge point move" when going to fp8. It does not
move, in tokens: on MI300X fp8 doubles the FLOP rate and halves the bytes, so the token
ridge point is unchanged at 247 while the FLOPs-per-byte ridge point doubles. I rewrote
the problem to make that the answer, and added a part (b) on weight-only quantization,
where the token ridge point genuinely does halve. Better pedagogy and it sets up
Chapter 11.

**4. The MI300X "in a ring" discrepancy is worse than the roadmap says.** The roadmap
notes the GPU data sheet says "in a ring" while the platform data sheet says fully meshed.
In fact the *platform* data sheet says both, on the same page: "fully-meshed 128 GB/s
bidirectional" in the prose and "Ring of 8 aggregate bandwidth" in the spec table.
`amd-gpus.md` quotes both and explains why the mesh reading is correct.

**One thing the roadmap got right that I expected to be wrong:** `capacity_factor: -1` is
indeed the dropless sentinel, and it is indeed the default.

## Unverified claims, by risk

**Nothing in this list is asserted in the prose without a hedge or a blocked comment.**
This is the list of things a reviewer should check before publication.

**High risk, because a chapter's framing depends on it.**

- **Whether `ROCm/jax-aiter` exposes any grouped or ragged MoE GEMM.** The roadmap says it
  does not, which if true is Chapter 7's central performance fact. I did not verify it and
  the prose does not assert it: it is in a `BLOCKED` comment in `moe.md`. Check against
  current wheels and pin a version.
- **Whether AMD maintains a ROCm MaxText fork with the pre-optimised model list the
  roadmap describes.** I inspected upstream `AI-Hypercomputer/maxtext` at commit `9f9ac05`,
  which carries in-tree ROCm support (`run_rocm.py`, the `nanoo_fp8` quantization value)
  but is not AMD's fork. Chapters 9 and 10 are written against "the configuration
  published for this model" without naming a repository, and both carry a verification
  note. **This affects the scheduling assumption for Wave 4**, since the roadmap treats
  those configs as given.
- **Whether XLA:GPU lowers `jax.lax.ragged_dot` to anything efficient on ROCm.** Chapter 7
  says it is the only platform-general path and explicitly does not claim it is fast.

**Medium risk, because a specific sentence depends on it.**

- **The RCCL NaN workaround on MI355X** that Chapter 9 is supposed to tell as a footnote.
  I have not identified the flag or the affected models, so the section is blocked with a
  note saying to verify before writing. Getting this wrong would be worse than omitting it.
- **Pallas on ROCm working at all**, which Chapter 8 asserts is available-but-experimental.
  Untested on our stack. Flagged in a comment.
- **The `--shm-size` claim in Appendix A**, that the Docker default causes a multi-process
  JAX failure. Asserted from experience, not from a clean reproduction.
- **Which library XLA:ROCm actually dispatches a bf16 matmul to.** Chapter 3 mentions
  `Cijk_*` kernel names as a rocBLAS/Tensile signature; Chapter 8's kernel-selection
  section is blocked partly on getting this right.

**Low risk, verified this pass but worth spot-checking.**

- All MI300X, MI325X and MI355X hardware constants, against the AMD data sheets and ROCm
  microarchitecture docs linked in each chapter's references. I rebuilt every FLOP figure
  from `CUs * FLOPs-per-clock * clock` and they agree with the published tables.
- The fp8 FNUZ/OCP split, against the HIP low-precision-types documentation.
- JetStream's archival on 1 February 2026 and `vllm-project/tpu-inference` being TPU-only,
  against the repositories themselves.
- AMD's published realised xGMI figures (45-48 GB/s per link, 310-330 GB/s per GPU for
  RCCL), against the ROCm xGMI blog post.
- Every MaxText config field name, default and einsum equation quoted in Chapter 7, read
  from source at commit `9f9ac05`.

## Derivations worth a second pair of eyes

**These are mine and they propagate.** Each one is used by at least two chapters.

**The xGMI direction convention.** I treat 128 GB/s per link as bidirectional and use
**64 GB/s unidirectional** in all cost models, with per-GPU egress `7 * 64 = 448 GB/s`.
AMD's platform data sheet and the ROCm xGMI blog both support this reading, and the blog's
"448 GB/s (7x64 GB/s)" aggregate-unidirectional line is the clinching evidence. **If this
is wrong, every collective cost in Chapters 4, 6, 7 and 12 is wrong by 2x.**

**`β_g = 320 GB/s` as the standard per-GPU egress for predictions.** Derived from AMD's
published 310-330 GB/s realised RCCL figure, about 0.7 of spec. Chapters 4, 6 and 7 all
substitute it. The alternative would be to quote spec bandwidth throughout and derate at
the end; I chose to derate up front and say so, because it makes the predictions
comparable to future measurements without a footnote.

**All-reduce on a full mesh costs `2V/(nβ_link)`, falling with participant count.** This
is why Chapter 6's tensor-parallel threshold `F > 1816 * (|Y|-1)` has the degree in it at
all, and why the ring-versus-mesh distinction is a factor of 7 rather than a detail.
**Chapter 6 says explicitly that this is the most schedule-sensitive result in the
chapter** and that the pending RCCL sweep decides it.

**The MoE dense-masked win condition, `η_ragged > η_dense * E_a / E`.** Simple algebra,
and it produces the most useful table in Chapter 7 (a ragged kernel at 5% efficiency beats
dense masking for a fine-grained model). Worth checking because it is counterintuitive
enough that a reader will want to argue with it.

**MoE decode saturation at `B ≈ E / E_a`.** Chapter 11's claim that batching a sparse
model buys almost nothing below that batch size. The argument is that batch `B` touches
about `min(B * E_a, E)` distinct experts, so weight bytes grow linearly until every expert
is hit. I believe this is right and I have not seen it stated this way elsewhere, which is
either a reason to be pleased or a reason to check it carefully.

**The optimal FSDP-to-TP split, `|X|/|Y| = 4 * B_glob / (9 * F)`.** Derived by equating
each strategy's communication-to-compute ratio. It produces `|X| = 16, |Y| = 4` for a 70B
model on 64 GPUs, which matches what practitioners land on empirically, so the sanity check
passes.

**Activation memory of 4 MiB per token for an 8B-shaped model.** Counted as
`4D + (N+2K)H + 3F` elements per token per layer, assuming Flash-style attention so the
score matrix is never materialised. Chapters 5 and 6 both use it. It is a *rough* count and
the prose says so.

## The blocker list, ordered by what it unblocks

**Everything below needs one 8-GPU MI300X node except where noted.** Ordered by how many
sections each item unblocks.

**1. Capture the two existing scripts and take screenshots.** Unblocks: Chapter 3's tool
tour (6 subsections), the HLO traversal, the matmul measurement, the training step, and
Chapter 3's worked problems. Then Chapter 8's fusion section, which depends on the HLO
traversal existing. `scripts/jax_matmul.py` and `scripts/transformer_block.py` already
exist and are the right shapes. **Decision needed first: hand-captured or scripted
against XProf's data endpoints?** The roadmap calls this the largest unplanned cost in the
book and the most likely thing to stall Chapter 3, and it is still undecided.

**2. The RCCL bandwidth sweep.** Unblocks Chapter 4's central section, and *resolves*
Chapter 6's tensor-parallel roofline, which currently swings by 7x on an assumption, and
Chapter 7's claim that intra-node dispatch is cheap. Needs a script that does not exist.
**This is the highest-value single measurement in the book** and it is one node.

**3. Chapter 7's three-implementation comparison.** Unblocks Chapter 7's central section
and is a hard prerequisite for Chapter 10 being worth running at all. Needs an MoE block
implementing all three dispatch strategies behind one flag.

**4. TP, FSDP and PP variant scripts.** Unblocks part 5 of four of Chapter 6's five
treatments. The FSDP one is missing from `structure.md`'s script list and should be added.

**5. The fp8 measurement.** Unblocks Chapter 6's low-precision section, and it is the
easiest of these if AMD's documented fp8 configurations work as described.

**6. Verify the checkpoint round-trip.** Unblocks Chapter 12's spine. Roughly a day, per
the roadmap, and worth doing well before it is needed because a broken conversion script is
itself publishable.

**7. A Llama 3 8B training run.** Unblocks most of Chapter 9. 70B and 405B need a cluster.

**8. A DeepSeek-V2-Lite training run.** Unblocks Chapter 10, after item 3.

**9. Verify the library survey** (hipBLASLt dispatch, AITER reachability, Pallas on ROCm,
`rocprofv3` handoff recipes). Unblocks Chapter 8's remaining three sections. No hardware
needed beyond a shell on a node, but it needs care and version pinning.

**Not blocked by hardware at all**, and doable today: identifying a public `rocm/jax`
container tag that matches the measurement stack (Appendices A and B both want it), and
exporting the three existing diagrams from `fw101` and `gpu_profiling` into `assets/img/`.

## Ambiguities I resolved by choosing

**Where an instruction or the roadmap admitted more than one reading, this is what I did.**

**"Sequentially complete each page in the order they appear in the notebook."** I read
"notebook" as the book's navigation order, starting at `index.md` as Chapter 0 and ending
at Appendix B. Writing was strictly in that order, which is how the dependency chains
surfaced: Chapter 8's fusion section turned out to depend on Chapter 3's HLO traversal, and
Chapter 10 turned out to depend on Chapter 7's kernel question, neither of which is obvious
from `structure.md`.

**How strict to be about "blocked".** I used three tests, and a section is blocked if any
one fires: it would require asserting a performance number I did not compute or measure; it
would require describing an artifact (screenshot, HLO dump, trace) I have not seen; or it
would require stating a software behaviour I have not checked. The style guide's "never
assert a number you did not either compute or measure" is the first test, and applying it
literally is what makes Chapters 9 and 10 empty rather than plausible.

**Whether to use AMD's published measurements.** Yes, but never as ours. They appear in
Chapter 4 (xGMI and RCCL bandwidth) and Chapter 8 (the occupancy sweep), each with a
"these are AMD's measurements, not ours" note. This is a third category the roadmap's
two-tag scheme did not anticipate, so I added a paragraph naming it on the landing page and
in Appendix B. **Worth a decision from you: is a third tag warranted, or is inline
attribution enough?** I chose inline attribution because two tags are already at the limit
of what a reader will track.

**Whether Chapter 3's tool tour could be written without screenshots.** I decided no. It
would become exactly the feature-list prose the style guide forbids ("The Trace Viewer
provides comprehensive visibility into..."), and the guide's requirement of a real artifact
before each explanation is not satisfiable in prose. This is the section I am least certain
about blocking.

**Whether to quote spec or realised bandwidth in cost models.** Realised, at
`β_g = 320 GB/s`, with spec given alongside. See
[Derivations](#derivations-worth-a-second-pair-of-eyes).

**Whether Chapter 2 should carry the "beyond a node" section at all**, given that we have
no cluster data. I wrote it, using AMD's cluster reference architecture (eight 400 Gbps
NICs per node, rail-optimised leaf-spine), because Chapters 6, 7 and 12 all need a
`β_net` figure and leaving it out would mean each of them inventing one. It is marked
**[analytical]** and the section says plainly that this is the weakest part of the book.

**How much of Chapter 5's model-specific data to include.** I included full parameter
breakdowns for Llama 3 8B and 70B and expert configurations for five MoE models, all
sourced (the MoE ones from MaxText's config files, checked at a commit). The alternative
was to keep the chapter symbolic, which would have made it shorter and much less useful.

**Whether to name the `E/E_a` inflation of the ridge point in Chapter 5 or Chapter 7.**
Both, deliberately. Chapter 5 derives it as accounting and Chapter 7 spends it as a systems
constraint, which is the seam `structure.md` asks for between the two chapters.
