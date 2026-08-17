---
layout: distill
title: "Sharded Matrices and How to Multiply Them"
description: "You split the matrix across eight GPUs. What does that cost? Named-axis notation for sharded arrays, the four collectives and their prices on a switchless xGMI mesh, the multi-process program model, and who inserts the collective: the compiler or you."
date: 2026-08-04

section_number: 4

previous_section_url: "/pages/3-profiling"
previous_section_name: "Chapter 3: Profiling"

next_section_url: "/pages/5-transformers"
next_section_name: "Chapter 5: Transformer Math"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Notation for Sharded Arrays
  - name: The Four Collectives
  - name: RCCL in Practice
  - name: Measured Versus Spec Bandwidth
    subsections:
      - name: Achieved Bandwidth Against Message Size
      - name: Does RCCL Use All Seven Links?
      - name: The Number to Use
  - name: The Four Sharded-Matmul Cases
  - name: One Program, Many Processes
  - name: "Who Inserts the Collective: GSPMD or You?"
    subsections:
      - name: They Compile To The Same Thing
  - name: Is the Collective Overlapping?
    subsections:
      - name: The Measurement, And Why It Is A Two-By-Two
      - name: Why Combining Beats Overlapping
  - name: Worked Problems
  - name: References
---

> **Draft.** The measurements are in. What is still owed is figures: the trace-viewer
> screenshots for [Is the Collective Overlapping?](#is-the-collective-overlapping) and
> [RCCL in Practice](#rccl-in-practice), and the HLO diff in
> [Who Inserts the Collective](#who-inserts-the-collective-gspmd-or-you).

**Depends on:** [Chapter 1]({{ '/pages/1-rooflines' | relative_url }}) for rooflines,
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) for the xGMI mesh and its
bandwidths, and [Chapter 3]({{ '/pages/3-profiling' | relative_url }}) for reading a
trace, since half of this chapter is looking at collectives in one.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

One GPU holds 192 GB. A 70B-parameter model in bf16 needs 140 GB for weights alone,
and about a terabyte once you add gradients and Adam state, so **the interesting
question was never how fast one GPU is.** It is what happens when the array is split
across eight of them, and specifically what the splitting costs.

**This chapter builds the cost model that every inequality in
[Chapters 6]({{ '/pages/6-training' | relative_url }}),
[7]({{ '/pages/7-moe' | relative_url }}) and
[11]({{ '/pages/11-inference' | relative_url }}) is a substitution into.** Three things,
in order: a notation for saying how an array is split, the four collective operations
that move between splittings, and what each one costs on the specific fabric that
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) described. Then the part most
sharding tutorials skip, which is who writes the collective: you, or the compiler.

The costs here are **[analytical]** unless marked otherwise. Intra-node numbers are
derived from specification bandwidths and cross-checked against AMD's own published
measurements; anything crossing a host boundary is arithmetic only, because we have no
cluster.

## Notation for Sharded Arrays

**A mesh is a named grid of devices, and every sharding decision in this book is a
statement about which array axis lives on which mesh axis.** That is the whole idea,
and it is worth the page it costs to teach because it makes every later derivation
readable.

**Start with a toy.** Take `A`, a `[4, 8]` array, on four devices. Call the mesh axis
`X`, so `|X| = 4`. There are three interesting things you can do:

- **Replicate it.** Every device holds all 32 elements. Written `A[I, J]`, with no
  subscripts.
- **Shard the first axis over `X`.** Each device holds a `[1, 8]` slice. Written
  `A[I_X, J]`.
- **Shard the second axis over `X`.** Each device holds a `[4, 2]` slice. Written
  `A[I, J_X]`.

**The subscript says which mesh axis a given array axis is cut along.** With a
two-dimensional mesh, say `|X| = 2` and `|Y| = 4`, you can do both at once:
`A[I_X, J_Y]` means the first axis is cut in two over `X` and the second in four over
`Y`, so each of the eight devices holds a `[2, 2]` block.

**This maps one-to-one onto what you write in JAX, and the two should be learned
together so the notation never feels like a parallel vocabulary invented for a book.**

```python
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# 8 devices as a 2x4 grid with named axes.
mesh = Mesh(np.array(jax.devices()).reshape(2, 4), axis_names=("X", "Y"))

# A[I_X, J_Y]
sharding = NamedSharding(mesh, P("X", "Y"))

# A[I_X, J]  -- second axis replicated
sharding = NamedSharding(mesh, P("X", None))

# A[I, J]    -- fully replicated
sharding = NamedSharding(mesh, P(None, None))
```

The `PartitionSpec` *is* the subscript list. `P("X", "Y")` is `[I_X, J_Y]`.

**The mesh axes are named consistently for the whole book**, and the names are chosen
so that a sharding tells you the parallelism strategy at a glance: `X` for data and
FSDP, `Y` for tensor, `Z` for pipeline, `Ex` for expert. Axis sizes are written `|X|`,
`|Y|` and so on, and the total device count is their product.

**One more annotation, and it is the one that generates all the interesting
collectives: partial sums.** If you shard a matmul along its contracting dimension,
each device computes a piece of the sum and holds a full-shaped array that is *not the
answer*. Written `A[I, J]{U_Y}`, read as "unreduced over `Y`": every device has a
`[I, J]`-shaped array, and the true value is the elementwise sum of them across `Y`.

**An unreduced array is a debt, not a result.** You can carry it for a while, add
other unreduced arrays to it, even multiply it by a scalar, but the moment you need
the real values you have to pay an all-reduce. Half of parallel matmul design is
choosing *when* to pay.

## The Four Collectives

**Four operations move between shardings, and everything else is built from them.**
Each one has a job, a signature in the notation above, and a price.

| Collective | What it does to the sharding | Bytes out per device | Cost on `n` devices |
|---|---|---|---|
| **AllGather** | `A[I_X, J]` to `A[I, J]` | `V(n-1)/n` | `V(n-1)/n / β_g` |
| **ReduceScatter** | `A[I, J]{U_X}` to `A[I_X, J]` | `V(n-1)/n` | `V(n-1)/n / β_g` |
| **AllReduce** | `A[I, J]{U_X}` to `A[I, J]` | `2V(n-1)/n` | `2V(n-1)/n / β_g` |
| **AllToAll** | `A[I_X, J]` to `A[I, J_X]` | `(V/n)(n-1)/n` | `(V/n)(n-1)/n / β_g` |

`V` is the size in bytes of the full logical array, and `β_g` is the per-GPU egress
bandwidth: how fast one device can push bytes out across all of its links at once.

**Three observations that pay for the table.**

**An AllReduce is exactly twice a ReduceScatter, because that is how it is
implemented.** Reduce down to shards, then gather the shards back up. This matters
practically: if the consumer of the result only needs its own shard, ask for a
ReduceScatter and halve the bill. FSDP in
[Chapter 6]({{ '/pages/6-training' | relative_url }}) is precisely that observation
applied to gradients.

**AllToAll is `n` times cheaper than AllGather for the same array**, because it moves
each byte once to one destination instead of to everybody. That factor of `n` is the
reason Mixture-of-Experts dispatch is affordable at all, and
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) leans on it hard.

**None of the costs depend on how the array was shaped, only on how many bytes it
is.** Which is why this table is a lookup rather than a derivation, and why the rest
of the book can substitute into it without re-deriving anything.

**Now the AMD-specific part, which is where `β_g` comes from and where the interesting
argument lives.** [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) established the
wiring: eight GPUs on a baseboard, seven direct xGMI links each at 64 GB/s
unidirectional, no switch, and a hard ceiling at eight. Two consequences follow, and
they are this chapter's own material rather than a restatement of the topology.

**First: the physical topology is not the collective algorithm, and the gap between
them can be a factor of seven.** Consider an AllGather of a 1 GB array over 8 devices.

- **What the wiring permits.** Each device sends its 128 MB shard to seven peers, and
  because it has seven independent links it can send all seven copies at once. Time is
  `128e6 / 64e9 = 2.0 ms`, and per-GPU egress is the full `7 * 64 = 448 GB/s`. This is
  the bandwidth-optimal schedule: each device must receive `7/8` of the array, and it
  has ingress capacity of 448 GB/s, so 2.0 ms is a hard lower bound.
- **What a single ring would give.** Pass shards hand to hand around a ring of eight:
  seven steps, each moving 128 MB across one link. Time is
  `7 * 128e6 / 64e9 = 14.0 ms`, seven times worse, because at any instant one link per
  device is busy and six are idle.

**RCCL chooses the schedule, not the fabric.** It is a port of NCCL and it builds
ring and tree schedules over whatever topology it discovers, using multiple parallel
channels so that more than one link is lit at once. How close it lands to the 2.0 ms
bound is an empirical question about a library, which is exactly why the sweep in the
next section matters more than this derivation does.

**The answer, measured two sections from here, is 2.97 ms**: not the 2.0 ms the wiring
permits, but nowhere near the 14.0 ms a single ring would cost. **[measured]** RCCL lights
all seven links and loses about 26% to protocol overhead. If you want the short version of
this chapter, that is it.

**Second: the cost has two regimes with a cliff between them, not one smooth
curve.** Every number above assumes `n <= 8`, all on one baseboard. The ninth GPU is
over the NIC, at `β_net = 50 GB/s` per GPU against 448 GB/s intra-node, so a
collective that crosses a host boundary is roughly 9x more expensive per byte. There
is no gentle degradation: eight is cheap, nine is a different machine. Every placement
decision in [Chapters 6]({{ '/pages/6-training' | relative_url }}),
[7]({{ '/pages/7-moe' | relative_url }}) and
[12]({{ '/pages/12-serving' | relative_url }}) is downstream of that discontinuity, and
those chapters cite it rather than re-deriving it.

**A note for readers arriving from the TPU literature**, because the difference will
bite you. On a torus, collective cost is famously independent of how many chips are on
the axis, since each hop is to a neighbour and the ring is the natural schedule. That
result does not carry over. On a complete graph, cost *falls* with participant count
up to eight, because more participants means more links available, and AMD's own
guidance says so: a 2-GPU or 4-GPU collective can only use a fraction of the fabric.
Two GPUs talking have exactly one link between them. **Using half the node is worse
than half as good.**

## RCCL in Practice

**Collectives are visible in the compiled program before they are visible in a trace,
and reading them out of HLO is the fastest way to find out what sharding you actually
got.** Dump the optimised HLO:

```bash
XLA_FLAGS="--xla_dump_to=/tmp/hlo --xla_dump_hlo_pass_re=.*" python3 your_script.py
```

**Every collective appears as a start/done pair.** XLA splits an asynchronous
collective into two ops so that the scheduler can put other work between them:

{% raw %}
```
%all-reduce-start = f32[8192,4096] all-reduce-start(%add.3), replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=%add
%all-reduce-done  = f32[8192,4096] all-reduce-done(%all-reduce-start)
```
{% endraw %}

The gap between the two in the schedule is the overlap opportunity;
[Is the Collective Overlapping?](#is-the-collective-overlapping) is about whether
anything landed in it.

**`replica_groups` is the field to read, and it tells you the parallelism strategy for
free.** It lists which devices participate in each instance of the collective. On an
8-GPU node:

- {% raw %}`{{0,1,2,3,4,5,6,7}}`{% endraw %} is one group of eight: a collective over the whole node, so
  the axis it runs on has size 8. A gradient all-reduce here means 8-way data
  parallelism.
- {% raw %}`{{0,1},{2,3},{4,5},{6,7}}`{% endraw %} is four groups of two: an axis of size 2, with four
  independent copies. Combined with an all-reduce of activations rather than
  gradients, that is 2-way tensor parallelism inside 4-way data parallelism.
- {% raw %}`{{0,2,4,6},{1,3,5,7}}`{% endraw %} is two groups of four, *strided*. Stride tells you the mesh
  axis order: this collective runs on the slower-varying axis, which on a multi-node
  job is the one that crosses hosts. See
  [One Program, Many Processes](#one-program-many-processes) for why that is the most
  important thing on the page.

**Inferring the strategy from replica groups alone is a genuinely useful trick**, and
it is more reliable than reading the config, because it tells you what the compiler
did rather than what you asked for.

<!-- BLOCKED: the trace-viewer half of this section. What a collective looks like in
     XProf, including the RCCL kernel names on the device rows, the gap between
     all-reduce-start and all-reduce-done, and how to match the two ends up by
     correlation. Needs a captured trace of scripts/transformer_block.py, which
     already sets the latency-hiding flags, plus a screenshot. Same screenshot
     dependency as Chapter 3's tool tour. The HLO half above stands on its own. -->

## Measured Versus Spec Bandwidth

**Nobody sustains spec bandwidth, and the gap has two separate causes.** The first is
protocol overhead: CRC, framing, and flow control on the wire, a fixed derate that applies
no matter how large your messages are. The second is message size: below some threshold you
are paying per-collective latency rather than bandwidth, and the achieved figure falls off
a cliff.

**So we measured both.** **[measured]**

```bash
python -m bench.rccl_sweep --devices 2 4 8
```

Four collectives, message sizes from 1 KiB to 1 GiB in powers of two, at 2, 4 and 8
devices, bf16. Bandwidth conventions follow `nccl-tests` so the numbers are comparable with
anything published: `algbw = size / time`, and `busbw = algbw * 2(n-1)/n` for all-reduce or
`algbw * (n-1)/n` for the other three. **busbw is the per-GPU egress rate**, which is the
`β_g` in the cost model above.

**One methodological note, because it changes the numbers by an order of magnitude.**
Dispatching a collective to eight devices from Python costs about a millisecond, which is
longer than most of these collectives take. Timing the loop measures the loop. Every figure
here is the RCCL kernel's own duration, read out of an XPlane capture, and specifically the
*shortest* per-device kernel in each iteration: the device that arrives last does not wait
for anyone, so its kernel is the transfer and nothing else.

### Achieved Bandwidth Against Message Size

**Eight devices, per-GPU egress in GB/s:** **[measured]**

| Message | All-reduce | All-gather | Reduce-scatter | All-to-all |
|---|---|---|---|---|
| 64 KiB | 4.2 | 3.3 | 3.1 | 5.0 |
| 1 MiB | 30.5 | 30.5 | 36.6 | 56.0 |
| 4 MiB | 117.8 | 95.2 | 99.9 | 170.2 |
| 16 MiB | 212.0 | 206.1 | 196.9 | 247.9 |
| 64 MiB | 288.7 | 277.9 | 270.0 | 293.6 |
| 256 MiB | 317.1 | 309.0 | 298.8 | 298.1 |
| 1 GiB | **319.6** | **316.0** | **308.0** | **297.2** |

{% include figure.liquid path="assets/img/rccl-bandwidth-curve.png" class="img-fluid" caption="Per-GPU egress against message size, 8x MI300X, bf16, timed from the RCCL kernel in an XPlane capture. All four collectives converge on roughly 320 GB/s, and none of them is close to it below 16 MiB." %}

**All four converge on roughly 300 to 320 GB/s**, and they get there slowly. Half of peak
arrives at about 16 MiB; 90% of peak needs 64 to 128 MiB. **A collective smaller than a
megabyte is a latency measurement wearing a bandwidth costume.**

**The latency floor is 9 to 23 microseconds** depending on the collective, and it is flat
all the way from 4 KiB to about 256 KiB, which is what a flat region on this table means:
the wire is idle and you are paying for the round trip. All-to-all has the lowest floor at
9.5 microseconds and all-reduce the highest at 22.9, which is the expected ordering, since
all-reduce is the only one of the four that has to make two passes over the data.

### Does RCCL Use All Seven Links?

**Yes. This was the open question in
[The Four Collectives](#the-four-collectives) and the answer is unambiguous.**

The trick is to calibrate against the two-device case rather than against the data sheet. At
two devices a GPU has exactly one peer, so it has exactly one link, and whatever bandwidth
it achieves *is* the per-link rate on real hardware:

| Devices | Peers | All-reduce peak | Implied per-link | Links in use |
|---|---|---|---|---|
| 2 | 1 | 47.5 GB/s | 47.5 GB/s | 1.00 |
| 4 | 3 | 141.2 GB/s | 47.1 GB/s | 2.97 |
| 8 | 7 | 319.6 GB/s | 45.7 GB/s | 6.73 |

{% include figure.liquid path="assets/img/rccl-links-lit.png" class="img-fluid" caption="Peak per-GPU egress against participant count. The dashed line is the prediction if a GPU transmits on every link it has, at the per-link rate measured in the two-device case. All four collectives sit on it." %}

**The per-link rate barely moves: 45.7 to 47.5 GB/s, or about 74% of the 64 GB/s
specification.** And the number of links in use tracks the number of peers exactly. **A GPU
in an 8-way collective is transmitting on all seven of its xGMI links at once**, at the same
per-link efficiency it achieves talking to a single neighbour.

**A single-ring schedule would have capped at 47.5 GB/s.** It did not, by a factor of
6.7, so that hypothesis is dead. Every collective behaves the same way, including
**all-to-all at 8 devices, which reaches 297 GB/s, 6.3 links' worth**. That is the specific
question [Chapter 7]({{ '/pages/7-moe' | relative_url }}) flagged as unconfirmed, and it is
now confirmed: intra-node expert dispatch really is cheap.

**This also means the `(n-1)/n` factor in the cost model is doing double duty and you
should not double-count it.** The factor is already in the busbw definition; the topology
does not impose a *second* penalty for partial nodes. A 4-GPU collective gets 3/7 of the
fabric because it has 3 peers, not because RCCL is scheduling it badly.

### The Number to Use

**Use `β_g = 320 GB/s` for any 8-GPU intra-node prediction you intend to compare against a
measurement**, and 448 GB/s only when you want a hard lower bound on time. The derate is
about 0.71, it is consistent across all four collectives, and it holds only above about
64 MiB.

**Below that, use the table.** Quoting 320 GB/s for a 4 MiB all-reduce overstates the
bandwidth by 2.7x, and a per-layer gradient all-reduce in an 8B model is exactly in that
awkward region unless XLA combines it with its neighbours, which is what
`--xla_gpu_all_gather_combine_threshold_bytes` exists to make happen.

> **Corroboration:** AMD's
> [MI300X RCCL and xGMI](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html)
> blog post publishes 45 to 48 GB/s realised per link and 310 to 330 GB/s for an 8-GPU
> collective. **Our independent measurement lands inside both ranges**, which is reassuring
> for us and for them. Where this book previously quoted AMD's figures because it had none
> of its own, it now quotes its own and cites AMD's as agreement.

## The Four Sharded-Matmul Cases

**Every parallelism strategy in this book is one of four cases, and this is the lookup
table [Chapter 6]({{ '/pages/6-training' | relative_url }}) indexes into.** Take
`Out[B, F] = A[B, D] @ W[D, F]`, put it on a mesh axis `Y`, and ask which collective
the sharding forces.

**Case 1: nothing is sharded along the contracting dimension.** `A[B_Y, D] @ W[D, F]`
gives `Out[B_Y, F]`, with no communication at all. Each device has whole rows of `A`
and the whole of `W`, so it computes whole rows of the output. **This is data
parallelism**, and the fact that it needs no collective in the forward pass is why it
is the default everywhere.

**Case 2: the output dimension is sharded.** `A[B, D] @ W[D, F_Y]` gives
`Out[B, F_Y]`, again with no communication: each device owns a slice of columns of `W`
and produces the matching slice of output columns. **This is the first half of tensor
parallelism**, and it is free.

**Case 3: the contracting dimension is sharded on both operands.**
`A[B, D_Y] @ W[D_Y, F]` gives `Out[B, F]{U_Y}`, an unreduced array, which costs an
**AllReduce** of `B * F` elements to resolve. **This is the second half of tensor
parallelism**, and it is where the bill arrives. Note that you can defer it: if the
next operation is elementwise, or another matmul that contracts over `F`, you may be
able to push the reduction downstream.

**Case 4: the contracting dimension is sharded on one operand only.**
`A[B, D] @ W[D_Y, F]` does not typecheck as a local matmul: device `i` has all of `A`
but only rows `D_Y=i` of `W`. Two ways out, and the choice between them is the whole
of FSDP:

- **AllGather `W` first**, giving case 1 or 2, at a cost of `D * F` elements. Pay in
  communication proportional to *weight* size.
- **Shard `A` to match**, giving case 3, at a cost of an AllReduce over `B * F`. Pay
  in communication proportional to *activation* size.

**Which is cheaper depends entirely on whether `B` or `D` is larger**, and that single
comparison is why FSDP and tensor parallelism win in different regimes.
[Chapter 6]({{ '/pages/6-training' | relative_url }}) does the algebra properly.

| Sharding of `A` | Sharding of `W` | Output | Collective | Cost in elements |
|---|---|---|---|---|
| `[B_Y, D]` | `[D, F]` | `[B_Y, F]` | none | 0 |
| `[B, D]` | `[D, F_Y]` | `[B, F_Y]` | none | 0 |
| `[B, D_Y]` | `[D_Y, F]` | `[B, F]{U_Y}` | AllReduce | `2 * B * F * (n-1)/n` |
| `[B, D]` | `[D_Y, F]` | `[B, F]` | AllGather `W`, or reshard `A` | `D * F * (n-1)/n`, or the row above |

## One Program, Many Processes

**A multi-node JAX job is one program running in as many processes as there are
hosts, and this section is the prerequisite for every placement argument later in the
book.** It is short, and if you only take one sentence from it, take this one: **the
order of the axes in your `Mesh` decides which collective crosses the NIC.**

```python
import jax

# Once, before any other JAX call. Every process runs the same script.
jax.distributed.initialize()

print(jax.process_index(), "of", jax.process_count())   # 0..15 of 16
print(len(jax.local_devices()), len(jax.devices()))     # 8 local, 128 global
```

**Three facts about the model, and they are all mechanical rather than heuristic.**

**Arrays are globally shaped and locally backed.** An array with global shape
`[1024, 8192]` sharded over 128 devices exists as 128 pieces, and each process
physically holds only the pieces belonging to its eight local devices. Every process
sees the same global shape and the same sharding; `jax.Array` handles the bookkeeping.
Printing `x.shape` gives you the global shape on every process, which is
occasionally confusing and always correct.

**Every process runs the same program.** No rank-zero special case, no manual
scatter-gather. The collectives are in the compiled program, and all processes enter
them together.

**`jax.process_index()` is for the things that genuinely differ per host**: which data
shard to read, where to write logs, whether to print. Use it for I/O and nothing else.
Using it to branch on computation breaks the SPMD assumption and you will get a hang
rather than an error.

**Now the part that costs money.** `jax.devices()` returns devices ordered by process,
so the first eight are host 0's, the next eight are host 1's, and so on. When you
reshape that list into a mesh, the axis ordering decides which axis is inside a
baseboard and which spans hosts:

```python
devices = np.array(jax.devices()).reshape(16, 8)   # 16 hosts, 8 GPUs each

# X varies slowest: X spans hosts, Y is intra-node.
mesh = Mesh(devices, axis_names=("X", "Y"))
```

With that mesh, a collective on `Y` stays on one baseboard at 448 GB/s per GPU, and a
collective on `X` crosses the NIC at 50 GB/s per GPU. **Swap the two names and every
cost model in the book inverts.** The rule to internalise: **the last mesh axis is the
fast one**, and whichever parallelism strategy communicates most often belongs on it.

Applying that immediately:

- **Tensor parallelism** all-reduces activations several times per layer. It goes on
  the intra-node axis, always.
- **Expert parallelism** all-to-alls twice per MoE layer. Also intra-node, and
  [Chapter 7]({{ '/pages/7-moe' | relative_url }}) argues this is the single most
  important placement decision for a sparse model.
- **Data parallelism and FSDP** communicate once per step, on gradients, and can be
  overlapped with the backward pass. These are the ones that tolerate the slow axis.
- **Pipeline parallelism** sends activations between adjacent stages only, which is a
  small point-to-point transfer rather than a collective. Also fine across hosts.

**A mesh laid out so that the expert axis straddles two baseboards is a one-line
mistake with a large bill**, and it is invisible unless you know to look. The way to
check is the `replica_groups` reading from
[RCCL in Practice](#rccl-in-practice): if the groups on your fast axis are strided
rather than contiguous, the axis order is not what you think it is.

**Honesty note:** the mechanism above is exact and testable. The 8-GPU bandwidths are
specification-derived and calibrated against AMD's published measurements. Everything
about what happens across hosts is **[analytical]** until we have a cluster.

## Who Inserts the Collective: GSPMD or You?

**There are two ways to get a collective into your program, and the choice matters
enough that [Chapter 7]({{ '/pages/7-moe' | relative_url }}) turns on it.**

**The default is that you annotate arrays and the compiler works out the
collectives.** You say where the inputs and outputs live, XLA's GSPMD partitioner
propagates shardings through the graph, and it inserts an AllReduce or an AllGather
wherever the sharding it inferred does not match the sharding an operation needs.

```python
@jax.jit
def layer(x, w):
    return x @ w

x = jax.device_put(x, NamedSharding(mesh, P("X", None)))
w = jax.device_put(w, NamedSharding(mesh, P(None, "Y")))
out = layer(x, w)     # GSPMD decides what, if anything, has to be communicated
```

**For the strategies in [Chapter 6]({{ '/pages/6-training' | relative_url }}) this is
usually the right choice**, and it is a genuinely good compiler pass: it will fuse the
collective with neighbouring work, and it knows the four cases above better than you
do.

**`shard_map` is the other way: it hands you the per-device view and makes you write
the collective yourself.**

```python
from jax.experimental.shard_map import shard_map

@partial(shard_map, mesh=mesh, in_specs=(P(None, "Y"), P("Y", None)),
         out_specs=P(None, None))
def layer(x_shard, w_shard):
    partial_out = x_shard @ w_shard        # a [B, F] partial sum, per device
    return jax.lax.psum(partial_out, "Y")  # you pay the AllReduce, explicitly
```

Inside the body, shapes are *local*: `x_shard` really is a `[B, D/|Y|]` array, not a
global array in disguise. `jax.lax.psum`, `all_gather` and `all_to_all` are the
collectives, named and placed by you.

**Use GSPMD by default and reach for `shard_map` when you need a collective the
compiler will not choose.** The canonical case is Mixture-of-Experts routing: the
dispatch is a data-dependent all-to-all over a ragged set of tokens, GSPMD has no way
to infer it from shardings, and every serious implementation writes it by hand. That
is why this distinction is taught here rather than in
[Chapter 7]({{ '/pages/7-moe' | relative_url }}), where it would arrive as an
interruption.

### They Compile To The Same Thing

**The claim above is that these are two spellings of one program. That is checkable, so we
checked it.** Same matmul, `x[4096, 4096] @ w[4096, 14336]` with the contracting dimension
split eight ways, written both ways and dumped:

```bash
python -m bench.gspmd_vs_shardmap --diff
```

**Both compile to the same four instructions:** a `custom-call` into hipBLASLt for the local
partial product, a `get-tuple-element` to pull the result out, then an
`all-reduce-start` and `all-reduce-done` pair. **[measured]** Same entry layout, same
`channel_id`, same `use_global_device_ids`, same asynchronous split of the collective.
And they run at the same speed: 1.166 ms against 1.172 ms, which is half a percent apart
and therefore the same number.

**The diff is 40 lines and every one of them is cosmetic.** The two worth looking at:

{% raw %}
```
# GSPMD
%all-reduce-start = bf16[4096,14336]{1,0} all-reduce-start(%get-tuple-element),
    channel_id=1, replica_groups=mesh['axis_0'=1,'axis_1'=8] {'axis_1'},
    use_global_device_ids=true, to_apply=%add.clone

# shard_map
%all-reduce-start = bf16[4096,14336]{1,0} all-reduce-start(%get-tuple-element),
    channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}},
    use_global_device_ids=true, to_apply=%region_0
```
{% endraw %}

**`replica_groups` says the same thing twice.** GSPMD keeps the symbolic mesh form, naming
the axis it is reducing over; `shard_map` has already resolved it to an explicit list of
device ids. One group of eight either way.

**And the reduction function differs only in what its arguments are called**: GSPMD emits
`%add.clone` taking `x` and `y`, `shard_map` emits `%region_0` taking two parameters both
named `psum_invariant`. Both are a two-argument `add`.

**So the choice between them is not a performance decision.** It is about who has to know
the four cases in [The Four Sharded-Matmul Cases](#the-four-sharded-matmul-cases): with
GSPMD the compiler works out that a contracting-dimension split owes an all-reduce, and
with `shard_map` you write `jax.lax.psum` because you worked it out. Pick on the basis of
whether you want to be able to express something the partitioner would not infer, which is
the MoE case above, and not on the basis of expected speed.

**One caveat before you generalise from a four-instruction module.** This is the easy case,
where both spellings describe the same schedule. The interesting divergences show up when
GSPMD has a choice, for example when it can reorder a collective past other work or fold
it into a neighbouring op, and `shard_map` has pinned it in place. **Diff your own HLO
rather than trusting this result to transfer**; the command above is three lines and the
comparison takes a minute.

## Is the Collective Overlapping?

**A collective that overlaps with compute is nearly free, and one that does not is
pure added latency. The difference is a scheduler flag.** This section owns overlap for
the whole book: [Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }})'s
triage list refers back here rather than teaching it again.

Two flags do the work:

```bash
XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
           --xla_gpu_enable_highest_priority_async_stream=true"
```

**The first tells XLA's scheduler to move independent compute between an
`all-reduce-start` and its matching `all-reduce-done`.** Without it, the schedule is
free to place the `done` immediately after the `start`, and the device sits idle for
the whole transfer even though the collective was issued asynchronously. **The second
puts the collective's stream at the highest priority**, so the copy engine is not
starved by compute kernels queued ahead of it.

**In the Trace Viewer this is one of the easiest things in the book to see**, which is
why it is worth checking first: find the RCCL kernel on the device rows and look at
what is above it. Compute kernels running concurrently means the overlap is working.
A gap on the compute rows exactly as wide as the collective means it is not.

**You can also compute it rather than eyeball it**, which is what we did here: merge every
compute kernel on a device into a set of busy intervals, then ask what fraction of each
collective falls inside one. `tools/parse_xplane.py` does this, and the answer for a real
FSDP step turned out to be more interesting than the flag documentation suggests.

### The Measurement, And Why It Is A Two-By-Two

**We set out to measure one flag and found it does nothing, for a reason that is more
useful than the flag.** Four FSDP transformer blocks, 2048 tokens per device, eight GPUs,
toggling the latency-hiding scheduler:

| Latency hiding | Step time | Collective time hidden |
|---|---|---|
| On (as shipped) | 42.30 ms | 0.9% |
| Off | 42.52 ms | 0.2% |

**[measured]** Half a percent apart, which is noise. **Almost none of the communication is
overlapped either way**, and turning off the scheduler that exists to overlap it changes
nothing.

**The scheduler is not broken. It has nothing to work with.** The container ships with

```
--xla_gpu_all_gather_combine_threshold_bytes=8589934592
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
```

which is 8 GiB, comfortably more than this model's 1.75 GB of weights. **So XLA merges every
layer's all-gather into one collective at the top of the step.** A single collective with
nothing scheduled before it has nothing to hide behind, and no scheduler can fix that.

**Drop the threshold to 1 MiB so the collectives stay separate, and the flag springs to
life:**

| Combine threshold | Latency hiding | Step time | Collective time hidden | MFU |
|---|---|---|---|---|
| 8 GiB (as shipped) | On | **42.30 ms** | 0.9% | 20.1% |
| 8 GiB (as shipped) | Off | 42.52 ms | 0.2% | 20.0% |
| 1 MiB | On | 53.89 ms | **36.6%** | 15.8% |
| 1 MiB | Off | 55.85 ms | 0.1% | 15.2% |

**[measured]**

**Read the bottom two rows first, because that is the experiment the flag documentation
describes.** With many small collectives, the latency-hiding scheduler takes overlap from
0.1% to 36.6% and buys 3.5% of step time. It works exactly as advertised.

**Then read the first column, because that is the result that matters.** The configuration
with essentially no overlap is **27% faster** than the one with 36.6% overlap. Combining
the collectives wins by more than overlapping them does, and it wins by enough that giving
up all the overlap is still the right trade.

### Why Combining Beats Overlapping

**Because of the bandwidth curve in
[Measured Versus Spec Bandwidth](#measured-versus-spec-bandwidth).** A collective smaller
than a megabyte runs at under 10% of peak bandwidth and pays a 9 to 23 microsecond latency
floor. Split this model's communication into per-layer, per-tensor pieces and you issue
many of them, each in the worst part of that curve; the total collective time rises from
28.8% to 61.2% of device kernel time. **Overlapping 36.6% of a much larger number is a
losing trade.**

**The general rule, and it is the opposite of the folklore.** Overlap is the second thing
to reach for, not the first. **First get your collectives into the bandwidth-efficient part
of the curve, then overlap whatever is left.** A step with one big all-gather that does not
overlap can easily beat a step with twenty small ones that do.

**Two caveats on the numbers above, both of which cut the same way.** This is a four-layer
model, so there are only four all-gathers to combine and only three layers of compute to
hide them behind; a 32-layer model has more of both, and the balance may move. And the
1 MiB runs were noticeably jittery, with the mean running 30% above the median, which
[Appendix B]({{ '/pages/b-appendix-protocol' | relative_url }}) says to report rather than
smooth over. Many small collectives are less predictable as well as slower.

**What to actually do:** leave the combine thresholds where the container puts them, leave
latency hiding on since it costs nothing and starts mattering as soon as anything does not
combine, and check the overlap fraction rather than assuming the flag did something.

## Worked Problems

**Question 1:** How long should an all-reduce of a 1 GB bf16 gradient buffer take on
one 8-GPU node? Then on two nodes, 16 GPUs.

{% details Click here for the answer. %}

**One node.** AllReduce moves `2V(n-1)/n` bytes out of each device:

```
2 * 1e9 * (7/8) / 448e9 = 3.9 ms      at spec bandwidth
2 * 1e9 * (7/8) / 320e9 = 5.5 ms      at AMD's realised RCCL bandwidth
```

Quote the second one if you intend to compare against a stopwatch.

**Two nodes.** The sensible schedule is hierarchical: reduce-scatter inside each node,
all-reduce the resulting shards across nodes, all-gather inside each node again.

- Intra-node reduce-scatter: `1e9 * (7/8) / 320e9 = 2.7 ms`.
- Each GPU now owns a 125 MB shard. Cross-node all-reduce between two nodes, at
  `β_net = 50 GB/s` per GPU: each GPU sends its shard once for the reduce and once for
  the gather, so `2 * 125e6 * (1/2) / 50e9 = 2.5 ms`.
- Intra-node all-gather: another 2.7 ms.

Total about **7.9 ms**, against 5.5 ms on one node. **Crossing one host boundary costs
about 1.4x on this operation**, which is much better than the 9x bandwidth ratio might
suggest, and the reason is that hierarchical scheduling only sends `1/8` of the buffer
over the slow link. That is the general lesson: the cost of the slow axis depends
enormously on how much traffic you route onto it.

Mark the two-node figure **[analytical]**. We have not run it.

{% enddetails %}

**Question 2:** You dump the HLO of a training step on a 16-GPU, 2-node job and find
two collectives: an all-reduce with
{% raw %}`replica_groups={{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}}`{% endraw %}
on a `[8192, 4096]` buffer, and an all-reduce with
{% raw %}`replica_groups={{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}}`{% endraw %} on a
`[2048, 8192]` buffer. What is the parallelism strategy, and is the mesh the right way
round?

{% details Click here for the answer. %}

**Read the groups.** The first collective has two groups of eight contiguous devices,
which is exactly one baseboard each, so it runs on an axis of size 8 that is
intra-node. The second has eight groups of two, striding by 8, so it runs on an axis of
size 2 that pairs device `i` on host 0 with device `i` on host 1: an inter-node axis.

**So the mesh is `(2, 8)` with the size-2 axis spanning hosts**, which is the natural
layout and almost certainly what was intended.

**Now which strategy is on which axis.** The `[8192, 4096]` buffer is weight-shaped and
the `[2048, 8192]` one is activation-shaped, and activations get all-reduced by tensor
parallelism while gradients get all-reduced by data parallelism. So this is **8-way
tensor parallelism inside the node and 2-way data parallelism across nodes**, which is
backwards. Tensor parallelism all-reduces several times per layer and data parallelism
once per step, so the frequent collective is correctly on the fast axis, but 8-way TP
is a lot of TP: [Chapter 6]({{ '/pages/6-training' | relative_url }}) shows the roofline
usually breaks well before that degree. The mesh axis order is right; the degrees are
suspect.

**And a cheaper tell than any of this:** if the buffer being all-reduced is
activation-shaped, `B * F`, it is tensor parallelism. If it is weight-shaped, `D * F`,
it is data parallelism. You can classify collectives in a profile by their shape alone.

{% enddetails %}

**Question 3:** Your MoE model has its expert axis `Ex` of size 16 on a 2-node,
16-GPU job. The all-to-all for dispatch moves 512 MB per step. What does it cost, and
what would it cost if `Ex` were 8 and stayed inside a node?

{% details Click here for the answer. %}

**With `|Ex| = 16` spanning both nodes**, every device sends `(V/n)(n-1)/n` bytes,
and the bottleneck is the slowest link in the path, which is the NIC at 50 GB/s per
GPU:

```
(512e6 / 16) * (15/16) / 50e9 = 0.6 ms
```

**With `|Ex| = 8` inside one node**, the same total data per device, but at intra-node
bandwidth:

```
(512e6 / 8) * (7/8) / 320e9 = 0.18 ms
```

**Roughly 3x cheaper, and the sparse model got *more* experts per device rather than
fewer.** The comparison is not quite apples to apples, which is the point: keeping
`Ex` inside the node does not mean fewer experts, it means the other parallelism axis
carries the host boundary instead. **Keep `Ex` inside the node and spend the slow axis
on something that tolerates it**, which is data parallelism.
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) makes this the central placement
argument of the chapter.

Both figures are **[analytical]**, and the inter-node one especially so.

{% enddetails %}

## References

**JAX APIs.**

- [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
  (JAX). `Mesh`, `NamedSharding` and `PartitionSpec`, which are the notation in this
  chapter made executable.
- [shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html) (JAX). The
  manual-collective path, and the reference for `psum`, `all_gather` and `all_to_all`.
- [Multi-process JAX](https://docs.jax.dev/en/latest/multi_process.html) (JAX).
  `jax.distributed.initialize`, `process_index`, and the global-shape/local-data model.

**AMD interconnect and collectives.**

- [MI300X RCCL and xGMI](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html)
  (AMD). The realised per-link and per-GPU bandwidth figures this chapter calibrates
  against, and the 75%-of-peak protocol-overhead result.
- [MI300 and MI350 Series workload optimization](https://rocmdocs.amd.com/en/develop/how-to/rocm-for-ai/inference-optimization/workload.html)
  (AMD). The statement that collectives perform best with all eight GPUs
  participating, and the per-generation aggregate bandwidth table.
- [RCCL documentation](https://rocm.docs.amd.com/projects/rccl/en/latest/) (AMD). The
  collective library itself, including the environment variables worth knowing when a
  collective picks a bad algorithm.
- [AMD Instinct MI300X Platform data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf).
  The seven-link topology and the 128 GB/s bidirectional per-link figure.

**Compiler side.**

- [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663)
  (Xu et al., 2021). The partitioner that inserts your collectives when you do not.
- [How To Scale Your Model, Part 3: Sharded Matrices](https://jax-ml.github.io/scaling-book/sharding/)
  (Google DeepMind). The same four cases in TPU idiom, and the origin of the named-axis
  notation this book uses.
