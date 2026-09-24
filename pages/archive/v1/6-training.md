---
layout: distill
title: "How to Parallelize a Transformer for Training"
description: "You added seven more GPUs and got four times the throughput. Where did the rest go? Data, fully sharded data, tensor, pipeline and context parallelism, each with the same five-part treatment, then how they compose and how to choose."
date: 2026-08-04

section_number: 6

previous_section_url: "/pages/5-transformers"
previous_section_name: "Chapter 5: Transformer Math"

next_section_url: "/pages/7-moe"
next_section_name: "Chapter 7: Mixture-of-Experts"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: How to Read This Chapter
  - name: Data Parallelism
  - name: Fully Sharded Data Parallelism
  - name: Tensor Parallelism
  - name: Pipeline Parallelism
  - name: Context Parallelism
  - name: How They Compose
  - name: The Optimal Split
  - name: Memory, Not Just Time
  - name: Low Precision as a Parallelism Decision
    subsections:
      - name: The Measurement
  - name: A Decision Procedure
  - name: Worked Problems
  - name: References
---

> **Draft.** Every roofline in this chapter is derived, substituted and now measured.
> Three of the five predictions needed correcting and the corrections are in place: the
> tensor-parallel threshold uses a bandwidth its messages never reach, FSDP's collective
> does not overlap on this stack, and fp8 does not raise the FSDP threshold. The decision
> procedure at the end has not yet been reworked around those.

**Depends on:** [Chapter 4]({{ '/pages/4-sharding' | relative_url }}) for the collective
cost model and the sharding notation, and
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}) for the FLOP and byte counts
being sharded. Dense models only: expert parallelism is
[Chapter 7]({{ '/pages/7-moe' | relative_url }}).

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

You have a model that fits on one GPU and takes a month to train. You have 64 GPUs.
**The naive expectation is 64x, the naive outcome is more like 20x, and this chapter is
about the difference.**

There are five ways to split a Transformer across devices, they fail for five different
reasons, and each one has a threshold you can compute in advance. That is the useful
part: **you do not have to run four configurations to find out which one works**, you
have to substitute two numbers into an inequality.

## How to Read This Chapter

**Each strategy gets the same five-part treatment, deliberately, so that you learn the
move rather than five separate facts.**

1. **What it shards**, as a one-line sharding of the inputs, the two MLP weight
   matrices and the output, in
   [Chapter 4]({{ '/pages/4-sharding' | relative_url }})'s notation.
2. **Why do this, why not do this.** Motivation before algebra.
3. **The algorithm**, as a numbered list with every collective annotated *(on critical
   path)* or *(overlappable)*. That annotation is how you learn that data parallelism's
   all-reduce is forgiving and tensor parallelism's is not.
4. **The roofline.** Set `t_math > t_comms`, solve for a clean inequality, and
   substitute MI300X constants to get a real number.
5. **Predict, then measure.** Run it, read the trace, say whether the bound held.

**The constants substituted throughout**, from
[Chapters 2]({{ '/pages/2-amd-gpus' | relative_url }}) and
[4]({{ '/pages/4-sharding' | relative_url }}):

| Symbol | Value | Source |
|---|---|---|
| `C` | 1307.4e12 FLOP/s | MI300X dense bf16 peak |
| `β_g` | 320 GB/s | Per-GPU egress, 8-way collective, AMD-published realised RCCL |
| `β_net` | 50 GB/s | Per-GPU share of node egress, one 400 Gbps NIC |
| `w` | 2 bytes | bf16 |

**We model a Transformer layer as an MLP with `3DF` parameters and `18*B_tok*D*F`
training FLOPs**, per
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}), where `B_tok` is tokens per
device. That throws away attention, which is about 40% of layer FLOPs at 8k context, so
every threshold below is conservative by roughly that much. It keeps the algebra
readable, which is worth more than the last 40% here.

## Data Parallelism

**1. What it shards.**

```
In[B_X, D]  @  W_in[D, F]  ->  hidden[B_X, F]  @  W_out[F, D]  ->  Out[B_X, D]
```

The batch is split; every device holds every parameter.

**2. Why do this, why not do this.**

Do it because it is nearly free and it is the only strategy that requires no change to
the model code at all. The forward and backward passes are entirely local; the only
communication is a gradient all-reduce, once per step, which can be overlapped with the
backward pass that produces it.

Do not do it when the model does not fit. Every device holds a full copy of the
parameters, the gradients and the optimizer state, and
[Memory, Not Just Time](#memory-not-just-time) shows that ceiling arrives at about 12B
parameters on a 192 GB MI300X. Also do not do it past the point where the global batch
exceeds the convergence limit from
[Chapter 1]({{ '/pages/1-rooflines' | relative_url }}), because then you are adding GPUs
to process tokens that do not help.

**3. The algorithm.**

1. Forward pass, entirely local. *(no communication)*
2. Backward pass, entirely local, producing per-device gradients.
3. All-reduce the gradients across `X`. *(overlappable: the gradient for layer `i` is
   ready while layer `i-1` is still being differentiated, so a well-scheduled backward
   pass hides the whole thing)*
4. Optimizer update, local and identical on every device.

**4. The roofline.**

Per layer, per step, compute is `18*B_tok*D*F / C` and the all-reduce moves
`2 * w * 3*D*F * (|X|-1)/|X|` bytes out of each device. Compute-bound means:

```
18*B_tok*D*F / C  >  6*w*D*F*(|X|-1)/(|X| * β_g)
  =>  B_tok  >  w * C * (|X|-1) / (3 * |X| * β_g)
```

**`D` and `F` cancel, so the threshold does not depend on the model at all.** In the
large-`|X|` limit:

```
B_tok > 2 * 1307.4e12 / (3 * 320e9) = 2724 tokens per device
```

**About 2700 tokens per device, which at 8-way is 2400 and at spec bandwidth would be
1950.** Above that, the gradient all-reduce hides completely behind the backward pass
and data parallelism is free. Below it, you are waiting on the network every step.

**Two observations that matter more than the number.** The threshold is independent of
the device count, so data parallelism scales indefinitely as far as *bandwidth* is
concerned; what limits it is memory and the convergence limit. And 2700 tokens is small:
two sequences of 2048 per device clears it. **Data parallelism is almost always
bandwidth-fine and almost always memory-blocked**, which is why the next section exists.

**5. The measurement.**

```bash
python -m bench.transformer_block --strategy dp --seq-len 1024 \
    --tokens 1024 2048 3072 4096 5120 6144 8192 --trace
```

Four Llama-3-8B-shaped blocks on 8x MI300X, sweeping per-device tokens across the
threshold. The column that matters is the last one: the share of the step spent inside a
collective **with no compute running underneath it**, measured by merging the kernel
intervals in the trace. **[measured]**

| Tokens per device | Step time | MFU | All-reduce hidden | Step time exposed to it |
|---|---|---|---|---|
| 1024 | 45.0 ms | 9.3% | 37.5% | 34.2% |
| 2048 | 47.0 ms | 17.8% | 57.1% | 19.9% |
| 3072 | 50.5 ms | 24.8% | 70.5% | 11.9% |
| **4096** | 59.2 ms | 28.2% | **88.0%** | **3.9%** |
| 5120 | 68.7 ms | 30.4% | 89.9% | 3.0% |
| 6144 | 79.5 ms | 31.5% | 90.2% | 2.8% |
| 8192 | 98.1 ms | 34.1% | 92.3% | 1.9% |

**The claim holds: above the threshold the gradient all-reduce really does disappear behind
the backward pass.** By 4096 tokens per device, 88% of the collective has compute running
concurrently and the step pays under 4% for it. That is the scheduler question the
arithmetic could not answer, and the answer is yes.

**The threshold is in the right place and about 35% optimistic.** Predicted 2724 tokens;
the measured crossover, taking "exposed" falling below 5% as the criterion, is between 3072
and 4096. **The gap is arrival skew**, which the model does not include: the collective's
wall time per device is about twice its transfer time, because devices do not enter the
all-reduce together and the ones that arrive early wait inside it.

**Two things worth noticing in the table beyond the threshold.** The all-reduce takes
about the same 20 milliseconds per step at every batch size, exactly as the model says it
should, since gradient volume depends on the parameter count and not on the batch. So the
exposed fraction falls for the ordinary reason that the numerator is fixed and the
denominator is growing. And MFU is still climbing at 8192 tokens, which means the
communication stopped being the constraint well before the arithmetic started being
efficient. **Clearing the communication threshold is necessary and nowhere near
sufficient.**

## Fully Sharded Data Parallelism

**1. What it shards.**

```
In[B_X, D]  @  W_in[D_X, F]  ->  ...  ->  Out[B_X, D]
```

The batch *and* the parameters, gradients and optimizer state. Weights are gathered
just in time for the layer that needs them and thrown away afterwards.

**2. Why do this, why not do this.**

Do it because it is the strategy that makes large models fit. Sharding the optimizer
state `|X|` ways is a 16-bytes-per-parameter saving divided by the data-parallel degree,
which is the difference between a 70B model being impossible and being comfortable.

Do not do it if you already fit, because it costs 1.5x data parallelism's communication
and buys nothing else. And be careful crossing hosts: FSDP's all-gathers are on the
critical path in a way that data parallelism's all-reduce is not, because you cannot
start layer `i` until you have layer `i`'s weights.

**3. The algorithm.**

Also known as ZeRO stage 3. Per layer, in the forward pass:

1. All-gather `W_in` and `W_out` for this layer across `X`. *(on critical path, but
   prefetchable: gather layer `i+1` while computing layer `i`)*
2. Compute the layer locally.
3. Discard the gathered weights, keeping only your shard.

In the backward pass:

4. All-gather the layer's weights again. *(on critical path, same prefetch trick)*
5. Compute the gradients.
6. Reduce-scatter the gradients so each device keeps only the shard it owns.
   *(overlappable)*

**4. The roofline.**

Three collectives per layer per step instead of one: two all-gathers of `w*3DF` and one
reduce-scatter of the same size, each costing `V*(|X|-1)/|X| / β_g`.

```
18*B_tok*D*F / C  >  9*w*D*F*(|X|-1)/(|X| * β_g)
  =>  B_tok  >  w * C * (|X|-1) / (2 * |X| * β_g)
```

```
B_tok > 2 * 1307.4e12 / (2 * 320e9) = 4086 tokens per device
```

**About 4100 tokens per device, exactly 1.5x data parallelism**, which is the ratio of
three collectives to two. Again independent of the model.

**Note the asymmetry in what the collectives cost you.** The reduce-scatter overlaps as
happily as data parallelism's all-reduce. The two all-gathers are structurally
different: they block the layer that needs them. Prefetching one layer ahead fixes this
in principle, and in practice it means FSDP needs about a layer's worth of spare
memory and a scheduler that cooperates. When FSDP underperforms this roofline, failed
prefetch is the first thing to check.

**5. The measurement.**

```bash
python -m bench.transformer_block --strategy fsdp --seq-len 1024 \
    --tokens 1024 2048 3072 4096 5120 6144 8192 --trace
```

The same sweep, the same four blocks, the same eight devices, with the weights sharded.
**[measured]**

| Tokens per device | Step time | MFU | All-gather hidden | Step time exposed to it |
|---|---|---|---|---|
| 1024 | 48.9 ms | 8.5% | 11.9% | 57.1% |
| 2048 | 40.5 ms | 20.6% | 1.1% | 29.6% |
| 3072 | 48.6 ms | 25.8% | 1.4% | 21.8% |
| 4096 | 59.3 ms | 28.2% | 1.3% | 20.2% |
| 5120 | 69.3 ms | 30.1% | 1.5% | 15.6% |
| 6144 | 80.2 ms | 31.3% | 1.3% | 13.8% |
| 8192 | 99.0 ms | 33.8% | 1.5% | 12.5% |

**Compare the "hidden" column against data parallelism's and the difference is the whole
result. FSDP's communication is not overlapped at all, at any batch size.** Data
parallelism reached 92% hidden; this reaches 1.5% and stays there.

**The prefetch we told you to look for is not happening, and the reason is not FSDP's
fault.** The check was to find `all-gather-start` and `all-gather-done` pairs spanning the
previous layer's compute. There are no such pairs, because there is only one all-gather:
the container's `--xla_gpu_all_gather_combine_threshold_bytes=8589934592` merges all four
layers' weight gathers into a single collective at the top of the step, and a collective
with nothing scheduled before it cannot overlap anything.
[Chapter 4]({{ '/pages/4-sharding' | relative_url }})'s overlap section takes that apart,
including the awkward finding that **combining is still the faster choice**: splitting the
collectives back up buys 36% overlap and costs 27% of step time, because the pieces land in
the latency-bound part of the bandwidth curve.

**So the exposed fraction falls in this table for a different and much weaker reason than
in the DP table.** It is not that the collective got hidden; it is that the step got
longer around a collective of constant size. At 8192 tokens per device FSDP is still paying
12.5% of its step in exposed communication where DP pays 1.9%.

**Which makes the 1.5x ratio between the two thresholds the wrong thing to take away.**
The arithmetic that produces 4086 against 2724 is correct as arithmetic, and if the
all-gather overlapped the way the model assumes, the crossover would land there. On this
stack it does not overlap, so **the honest statement is that FSDP costs about 12% of step
time at a batch where DP costs about 2%, and you should budget for that rather than for a
threshold.** The trade is still usually worth taking, because the alternative to FSDP is
not DP, it is not fitting the model at all.

## Tensor Parallelism

**1. What it shards.**

```
In[B, D]  @  W_in[D, F_Y]  ->  hidden[B, F_Y]  @  W_out[F_Y, D]  ->  Out[B, D]{U_Y}
```

The feed-forward dimension, so each device holds a slice of every layer's hidden
dimension. The output is a partial sum and needs reducing.

**2. Why do this, why not do this.**

Do it because it shards *activations* as well as weights, and it does so without
increasing the global batch. That makes it the only strategy that reduces per-device
memory at a fixed batch size, which is exactly what you need when the model is too big
and the batch is already at the convergence limit.

Do not do it across hosts, ever, if you can avoid it. Tensor parallelism all-reduces
twice per layer, which at 80 layers is 160 collectives per step, each on the critical
path. On the intra-node mesh that is affordable. Over a NIC it is not.

**3. The algorithm.**

The Megatron pairing: shard the first matrix by columns and the second by rows, so that
only one reduction is needed for the pair.

1. `In` is replicated. Compute `hidden[B, F_Y] = In[B, D] @ W_in[D, F_Y]`.
   *(no communication: [Chapter 4]({{ '/pages/4-sharding' | relative_url }})'s case 2)*
2. Apply the nonlinearity elementwise, which is local because `F` is the sharded axis.
3. Compute `Out[B, D]{U_Y} = hidden[B, F_Y] @ W_out[F_Y, D]`. *(produces partial sums)*
4. All-reduce `Out` across `Y`. **(on critical path, and nothing else is available to
   overlap it with, because the next layer needs the result)**
5. In the backward pass, one further all-reduce of the input gradient per layer.
   *(on critical path)*

**Sequence parallelism is a companion to this, and it is not context parallelism.**
Settle the confusion here, because the two words arrive fused and MaxText exposes both
as separate mesh axes. Sequence parallelism in the Megatron sense shards the norms and
residual activations along the sequence axis *within* a tensor-parallel group, purely to
save activation memory; it converts the pair of all-reduces above into a
reduce-scatter and an all-gather of the same total volume, so it is free in bandwidth
and saves memory. **Context parallelism shards attention itself and needs a ring
exchange of keys and values.** Different collectives, different reasons. A reader who
conflates them writes a config that is silently wrong rather than loudly broken.

**4. The roofline.**

Compute per device per layer is `18*B_tok*D*F / (C * |Y|)`, since the FLOPs are
divided. Communication is two all-reduces of `B_tok * D` elements, so
`4 * w * B_tok * D * (|Y|-1) / (|Y| * β_g)` bytes out per device.

```
18*B_tok*D*F / (C*|Y|)  >  4*w*B_tok*D*(|Y|-1) / (|Y| * β_g)
  =>  F  >  2 * w * C * (|Y|-1) / (9 * β_g)
```

**`B_tok` cancels, so tensor parallelism does not care about the batch size at all.**
It cares about `F`, and about the degree:

```
F > 2 * 2 * 1307.4e12 * (|Y|-1) / (9 * 320e9) = 1816 * (|Y|-1)
```

| Tensor-parallel degree | `F` required | Llama 3 8B (`F` = 14336) | Llama 3 70B (`F` = 28672) |
|---|---|---|---|
| 2 | 1816 | fine | fine |
| 4 | 5448 | fine | fine |
| 8 | 12712 | marginal | fine |
| 16 | 27240 | breaks | marginal |

**8-way tensor parallelism inside a node is defensible for a 70B model and marginal for
an 8B one**, and 16-way is out of the question on this hardware because there is no
16-GPU scale-up domain to put it in.

**This inequality is the most schedule-sensitive result in the chapter and it deserves
a warning.** It uses `β_g = 320 GB/s`, AMD's published realised per-GPU RCCL bandwidth,
which implies RCCL is lighting most of the seven links. If RCCL instead ran a single
ring, per-GPU egress would be one link at 64 GB/s and the requirement at 8-way would be
`F > 63,553`, which no model in this book satisfies. **The difference between those two
worlds is a factor of seven and it decides whether 8-way tensor parallelism is a good
idea.** [Chapter 4]({{ '/pages/4-sharding' | relative_url }})'s pending bandwidth sweep is
what settles it; until then, treat the table above as the optimistic case and check the
all-reduce share of step time in your own trace before committing to a high degree.

**5. The measurement.**

```bash
for N in 2 4 8; do
  python -m bench.transformer_block --strategy tp --devices $N \
      --seq-len 2048 --tokens 2048 --trace
done
```

Fixed global batch of 2048 tokens, so the only thing changing is how many ways the
feed-forward dimension is split. **[measured]**

| Tensor-parallel degree | Step time | Collective time per step | Step time exposed to it | MFU |
|---|---|---|---|---|
| 2 | 22.8 ms | 1.55 ms | 4.2% | 18.7% |
| 4 | 17.8 ms | 3.40 ms | 14.1% | 11.9% |
| 8 | **28.5 ms** | 13.37 ms | **43.1%** | 3.7% |

**Eight-way tensor parallelism is slower in wall clock than four-way, on a model whose `F`
the threshold says should clear eight-way.** `F = 14336` against a requirement of
`1816 * 7 = 12712`. It clears by 13%, the table above calls that marginal, and the
measurement says marginal was optimistic.

**First, the good news, because it settles the 7x question.** The activation all-reduce is
not running at single-link speed. Each layer's collective is `2048 * 4096 * 2 = 16.8 MB`,
there are 15 of them per step, and
[Chapter 4]({{ '/pages/4-sharding' | relative_url }})'s sweep measures **212 GB/s of
per-GPU egress at that message size**, which is 4.5 links' worth rather than one. **The
optimistic branch of the roofline was the correct one**: `F > 12712` is the right form of
the requirement, not `F > 63553`.

**Now the bad news, which is that 212 GB/s is not 320 GB/s.** The threshold was derived
with the asymptotic figure, and tensor parallelism never sends asymptotic-sized messages:
its collectives are activation-sized, and activations are megabytes where gradients are
gigabytes. Substituting the bandwidth that a 16.8 MB message actually gets:

```
F > 2 * 2 * 1307.4e12 * (|Y|-1) / (9 * 212e9) = 2741 * (|Y|-1)
  =>  8-way needs F > 19191, not 12712
```

**`F = 14336` fails that, and the measurement agrees.** This is the correction to make:

| `\|Y\|` | Requirement at 320 GB/s | Requirement at the bandwidth these messages get | Llama 3 8B, `F` = 14336 |
|---|---|---|---|
| 2 | 1816 | 2741 | fine |
| 4 | 5448 | 8223 | fine |
| 8 | 12712 | 19191 | **breaks** |

**The general lesson is worth more than the specific number.** A threshold derived from
peak bandwidth is only valid if the collective is big enough to reach peak bandwidth, and
whether it is depends on which strategy you are using. Data parallelism's gradient
all-reduce is gigabytes and gets the asymptotic figure. Tensor parallelism's activation
all-reduce is megabytes and gets about two thirds of it. **Check your message size against
the curve before you trust a threshold.**

## Pipeline Parallelism

**1. What it shards.**

```
Layers 0..19   on Z=0     Layers 20..39  on Z=1    ...
```

Not a tensor axis at all: the *layers* are split, and microbatches flow through the
stages.

**2. Why do this, why not do this.**

Do it because it is by far the cheapest strategy in bandwidth terms. The only
communication is the activation tensor at each stage boundary, point to point, and
there are `|Z|-1` boundaries rather than a collective per layer. **That makes pipeline
parallelism the right way to cross a slow link**, which on an Ethernet-connected AMD
cluster is a first-class consideration rather than the afterthought it is on a TPU pod.

Do not do it if you cannot supply enough microbatches, because the cost is not bandwidth
but the bubble: stages idle while the pipeline fills and drains. And do not underestimate
the implementation complexity; pipeline schedules are the most intricate code in any
training framework.

**3. The algorithm.**

With `m` microbatches and 1F1B scheduling:

1. Stage 0 runs the forward pass for microbatch 0 and sends its output activations to
   stage 1. *(point to point, small, overlappable with microbatch 1's forward)*
2. Each stage alternates one forward and one backward as inputs arrive.
3. Gradients accumulate locally across microbatches; there is no gradient collective on
   the pipeline axis at all.
4. Stages idle during fill and drain.

**4. The roofline, which has two terms rather than one.**

**The bubble.** With `m` microbatches over `|Z|` stages, the fraction of time a stage
spends idle is:

```
bubble = (|Z| - 1) / (m + |Z| - 1)
```

For the bubble to cost less than 10% you need `m > 9 * (|Z| - 1)`, so 8 stages needs 63
microbatches. **That is the real constraint on pipeline parallelism** and it is a
constraint on batch size, since `m` microbatches of `B_micro` each is the global batch.

**The wire, which turns out not to matter.** Per stage boundary per microbatch, the
transfer is `w * M_tok * D` bytes where `M_tok` is tokens per microbatch, against
`18 * M_tok * D * F * L / |Z| / C` seconds of stage compute. The ratio is:

```
comms / compute = w * C * |Z| / (18 * F * L * β)
```

Substituting Llama 3 70B, 8 stages, and the *inter-node* bandwidth `β_net = 50 GB/s`:

```
2 * 1307.4e12 * 8 / (18 * 28672 * 80 * 50e9) = 0.010
```

**One percent of stage compute, over the slow link.** Pipeline activation transfer is
essentially free, which is the whole reason to reach for this strategy when you run out
of baseboard. Note that the ratio improves with more layers and larger `F`, so it gets
better on bigger models, and degrades with more stages, which is the same bubble
pressure arriving in a different form.

**5. The measurement.**

**Read this one knowing what it is not.** Pipelining exists to cross host boundaries, and
we have one host, so what follows is a bubble demo on eight GPUs that happen to be in the
same box. **The multi-node case, which is the only reason anyone reaches for pipelining,
is not validated here and this chapter does not claim otherwise.**

```bash
python -m bench.pipeline --stages 8 --layers-per-stage 2 --microbatches 1 2 4 8 16 32
```

Eight stages of two layers each, one stage per device, microbatches dispatched without
blocking so that JAX's asynchronous dispatch does the pipelining. One stage in isolation
takes 3.645 ms, which is the unit everything below is quoted in. **[measured]**

| Microbatches | Elapsed | Predicted bubble | Measured bubble | Stage-times elapsed, measured against `m + \|Z\| - 1` |
|---|---|---|---|---|
| 1 | 32.2 ms | 87.5% | 88.7% | 8.8 against 8 |
| 2 | 39.5 ms | 77.8% | 81.6% | 10.8 against 9 |
| 4 | 48.2 ms | 63.6% | 69.8% | 13.2 against 11 |
| 8 | 65.2 ms | 46.7% | 55.2% | 17.9 against 15 |
| 16 | 99.0 ms | 30.4% | 41.1% | 27.2 against 23 |
| 32 | 166.9 ms | 17.9% | 30.1% | 45.8 against 39 |

**The pipeline pipelines.** That was not guaranteed: a naive implementation that blocked
between stages would have taken `m * |Z|` stage-times, which at `m = 32` would be 256
rather than the 45.8 measured. Asynchronous dispatch is doing the overlapping with no
scheduling code at all.

**The bubble follows the formula, with a consistent and growing overhead.** The measured
fraction runs above the prediction everywhere, by 1 point at `m = 1` and 12 points at
`m = 32`, and the last column shows where that comes from: the elapsed time is always
about `m + |Z| - 1` stage-times **plus roughly 0.2 stage-times per microbatch**.

**That per-microbatch overhead is the stage-boundary transfer, and it is not invisible.**
0.2 stage-times is 0.69 ms, against a 16.8 MB activation crossing one xGMI link, which at
the 47.5 GB/s [Chapter 4]({{ '/pages/4-sharding' | relative_url }}) measures per link is
353 microseconds of transfer plus dispatch. **Budget about 20% of a stage-time per
microbatch for handing activations along**, and note that this is the intra-node case where
the hop is xGMI. Over a NIC it is the dominant term rather than a correction, which is the
part we cannot measure.

**The practical reading:** the formula is good enough to plan with, `m` needs to be several
times `|Z|` before the bubble stops dominating, and even at `m = 32` against `|Z| = 8` you
are giving up 30% of the machine. **Pipelining is what you do when the alternative is not
fitting, not when you want throughput.**

## Context Parallelism

**1. What it shards.**

```
In[B, T_Y, D]
```

The sequence axis. Each device holds a contiguous slice of tokens for every sequence.

**2. Why do this, why not do this.**

Do it when the sequence is so long that a single sequence's activations do not fit on
one device. At 128k context an 8B model's activations are about 512 GB per sequence
before remat, so this is not an exotic case, it is the normal case for long-context
training.

Do not do it at ordinary context lengths, where data parallelism achieves the same
thing more cheaply: splitting 8 sequences across 8 devices needs no collective at all,
while splitting one sequence 8 ways needs a ring exchange per layer.

**3. The algorithm.**

The MLP shards trivially, because it is pointwise in the sequence axis. Attention does
not: every query needs every key.

1. MLP and norms compute locally on the local token slice. *(no communication)*
2. For attention, each device computes partial attention against its own keys and
   values.
3. Pass the local key/value shard to the next device in a ring, accumulate the next
   partial, repeat `|Y|-1` times. *(on critical path, but overlappable with the partial
   attention computation itself, which is what makes ring attention work)*
4. Combine the partials with the usual online-softmax rescaling.

**4. The roofline.**

Per layer, each device sends its key/value shard around the ring: `2 * w * K * H * T *
(|Y|-1) / |Y|` bytes. Attention compute per device is
`12 * B * (T/|Y|) * T * N * H / C`. **Communication grows linearly in `T` while
attention compute grows quadratically, so context parallelism gets cheaper the longer
the context**, which is the opposite of every other strategy in this chapter and the
reason it works at all.

Concretely, Llama 3 70B, one sequence at 128k context, 8-way:

- Key/value traffic per layer: `2 * 2 * 8 * 128 * 131072 * (7/8) = 470 MB`, so
  `470e6 / 320e9 = 1.5 ms`.
- Attention compute per layer per device:
  `12 * 16384 * 131072 * 64 * 128 / 1307.4e12 = 161 ms`.

**Under 1% of the attention time.** At 8k context the same arithmetic gives about 15%,
and at 2k it is over 50%, which is the crossover: **context parallelism is free above
roughly 32k tokens and expensive below 4k.**

**5. The measurement.**

```bash
for S in 2048 8192 32768; do
  python -m bench.transformer_block --strategy cp --devices 8 --seq-len $S --tokens $S --trace
done
```

One sequence split eight ways, at three context lengths. **[measured]**

| Sequence length | Tokens per device | Step time | Collective time hidden | Step time exposed to it |
|---|---|---|---|---|
| 2048 | 256 | 48.8 ms | 15.9% | 94.9% |
| 8192 | 1024 | 53.0 ms | 40.8% | 45.1% |
| 32768 | 4096 | 76.2 ms | 59.5% | 8.4% |

**Context parallelism is unusable at 2k and reasonable at 32k, and the crossover is not
really about context length.** Look at the second column: what changes across these rows
is tokens per device, and the exposed fraction falls the same way it does for data
parallelism over the same range. **Splitting a sequence eight ways is splitting a batch
eight ways as far as the arithmetic is concerned**; the difference is that attention's
communication is quadratic in the sequence while everything else is linear, which is why
the hidden fraction improves rather than staying flat.

**A 2048-token sequence over eight devices leaves 256 tokens per device and 95% of the
step exposed to communication.** That is the number to remember, because it is the mistake
people make: reaching for context parallelism at a context length that does not need it.
**Below about 8k, shard the batch instead.**

**Two caveats.** The 32768 row is a single layer rather than four, because four did not
fit, so its MFU is not comparable with the rows above it; only the exposed fraction is.
And this is XLA's attention rather than a ring-attention implementation, so the
communication pattern is whatever GSPMD infers from a sequence-sharded input, not the
overlapped ring the section describes. **A real ring attention would do better than this,
and the crossover is therefore an upper bound.** Nothing in Part III runs long context, so
we have left it there.

## How They Compose

**All five compose, they land on different mesh axes, and the placement question is
always the same: which axis crosses a host boundary.**
[Chapter 4]({{ '/pages/4-sharding' | relative_url }})'s section on the multi-process
program model is the mechanism; this is where it gets spent.

| Strategy | Mesh axis | Collectives per step | Placement |
|---|---|---|---|
| Data parallel | `X` | 1 all-reduce, overlappable | Slow axis is fine |
| FSDP | `X` | `3L` collectives, partly overlappable | Prefers fast, tolerates slow |
| Tensor parallel | `Y` | `2L` all-reduces, critical path | **Intra-node only** |
| Pipeline parallel | `Z` | One point-to-point send per stage boundary | Slow axis is ideal |
| Context parallel | shares `Y` or its own | `L` ring exchanges | Intra-node, prefers long context |

**The standard recipe for a multi-node AMD cluster follows directly from that table**,
and it is worth stating as a default rather than making the reader derive it: **tensor
parallelism inside the baseboard, data parallelism or FSDP across nodes, pipeline
parallelism if you need to cross more nodes than the batch supports.**

```python
# 16 hosts x 8 GPUs. X varies slowest, so X spans hosts and Y is intra-node.
devices = np.array(jax.devices()).reshape(16, 8)
mesh = Mesh(devices, axis_names=("X", "Y"))   # X: FSDP across hosts. Y: TP in node.
```

**The interaction that surprises people is between tensor parallelism and the eight-GPU
ceiling.** On a TPU pod you can run 16-way or 32-way tensor parallelism because the
torus keeps going. On AMD you cannot: `|Y| <= 8`, full stop, and if the tensor-parallel
roofline says you need more than 8-way to fit the model, you need a different strategy
rather than a bigger `Y`. That constraint is doing more work in AMD parallelism
decisions than any bandwidth number.

## The Optimal Split

**FSDP's communication cost per device grows with `|X|` and tensor parallelism's grows
with `|Y|`, so with a fixed device count there is a split that minimises the total.**
This is the calculation the source book leaves implicit and it is the one people
actually want.

Fix the total device count `n = |X| * |Y|` and the global token batch `B_glob`, so
per-device tokens are `B_tok = B_glob / |X|`. Express each strategy's communication as a
fraction of its own compute:

```
FSDP:   R_x = 9*w*D*F/β_g  /  (18*B_glob*D*F/(|X|*C))  =  w*C*|X| / (2*β_g*B_glob)
TP:     R_y = 4*w*B_tok*D*|Y|/(|Y|*β_g)  /  (18*B_tok*D*F/(C*|Y|))  ≈  2*w*C*|Y| / (9*F*β_g)
```

Setting them equal gives the split where neither dominates:

```
|X| / |Y|  =  4 * B_glob / (9 * F)
```

**Substituting a real configuration:** Llama 3 70B (`F = 28672`) on 64 GPUs with a
global batch of 256k tokens:

```
|X|/|Y| = 4 * 262144 / (9 * 28672) = 4.06
|X| * |Y| = 64   =>   |X| = 16, |Y| = 4
```

**16-way FSDP with 4-way tensor parallelism**, which is what practitioners land on by
experiment. The formula is worth having because it tells you which way to move when a
configuration is wrong: **more tensor parallelism when `F` is large or the batch is
small, more FSDP when the batch is large.** And note that `|Y| = 4` leaves half the
baseboard for something else, which on a sparse model is where the expert axis goes.

## Memory, Not Just Time

**Several strategies are chosen for memory reasons and the time roofline never explains
that.** Here is the memory ledger, per device, for mixed-precision Adam training:

| Item | Bytes per parameter | Sharded by |
|---|---|---|
| bf16 parameters | 2 | FSDP, TP, PP |
| bf16 gradients | 2 | FSDP, TP, PP |
| fp32 master parameters | 4 | FSDP |
| fp32 Adam first moment | 4 | FSDP |
| fp32 Adam second moment | 4 | FSDP |
| **Total** | **16** | |

**Sixteen bytes per parameter is the number to remember**, and it produces the single
most useful ceiling in this chapter:

```
192e9 / 16 = 12e9 parameters
```

**Pure data parallelism on MI300X tops out at about 12B parameters, before any
activations.** With activations and workspace, call it 8B. That is why Llama 3 8B
trains happily under plain data parallelism and Llama 3 70B does not: 70.6B parameters
needs 1130 GB of state, which is six MI300X worth of memory for the state alone.

**FSDP divides the whole ledger by `|X|`.** 70B over 8 devices is 141 GB per device,
which fits but leaves nothing for activations; over 64 devices it is 17.6 GB, which is
comfortable. That is the memory argument for FSDP, and it is a stronger argument than
anything in the timing section.

**Activations are usually larger than all of the above**, per
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}): 4 MiB per token for an 8B
model, which at 16384 tokens per device is 64 GiB against 128 GB of parameter state.
The levers are remat, which costs 33% of your FLOPs for a 16x memory reduction, and
tensor parallelism, which shards activations by `|Y|` at no FLOP cost but with an
all-reduce per layer.

**Gradient accumulation is the lever that decouples batch size from memory**, and it
belongs here because it is what you reach for when the two constraints conflict. Run
`m` microbatches, accumulate gradients, and step once: the global batch is `m` times
the per-device batch, while activation memory is that of a single microbatch. The cost
is that you do `m` sequential forward and backward passes, so the gradient all-reduce
is amortised over more compute, which *improves* the data-parallel roofline. Gradient
accumulation makes data parallelism look better and makes wall-clock time worse.

**This is where [Chapter 1]({{ '/pages/1-rooflines' | relative_url }})'s two meanings of
critical batch size finally meet.** The global batch is bounded below by the hardware
ridge point, because each device needs enough tokens to be compute-bound, and above by
the convergence limit, because past it extra tokens stop improving the model. The
data-parallel degree has to fit between them:

```
|X|  <=  B_convergence / B_ridge
```

**With MI300X's 247-token ridge point and a convergence limit of a few million tokens,
that ratio is in the thousands for a dense model**, so data parallelism is not the
binding constraint. For a sparse model with `E/E_a = 32` the ridge point is 7904 tokens
and the ratio drops by that factor, which is the first hint of why
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) is harder than this chapter.

## Low Precision as a Parallelism Decision

**fp8 is exactly 2x bf16 on MI300X, so it halves `t_math` and therefore moves every
inequality in this chapter.** That is a scaling result, not a numerics footnote, which
is why it sits next to the inequalities it perturbs rather than in an appendix.

**The mechanism, stated precisely.** The compute term has `C` in the denominator and the
communication term does not, so doubling `C` doubles every threshold that has the form
"you need at least this much work per byte communicated". And **the collectives usually
stay in bf16 or fp32 even when the matmuls are fp8**, for numerical reasons: gradient
all-reduces in fp8 lose too much. So the bytes do not halve while the FLOPs double.

| Threshold | bf16 | fp8 matmuls, bf16 collectives |
|---|---|---|
| DP tokens per device | 2724 | **5448** |
| FSDP tokens per device | 4086 | **8172** |
| TP requirement on `F`, 8-way | 12712 | **25424** |

**The FSDP row of that table turns out not to hold, and
[The Measurement](#the-measurement) below has the correction:** MaxText gathers fp8
weights when the matmuls are fp8, so the bytes halve along with the FLOPs and the FSDP
threshold stays roughly where it was. The DP and TP rows stand, because their collectives
carry gradients and activations rather than weights.

**Every threshold doubles, and the tensor-parallel one is the painful entry.** 8-way
tensor parallelism on Llama 3 70B is comfortable in bf16 (`F = 28672` against a
requirement of 12712) and marginal in fp8 (against 25424). **Training faster makes
your communication relatively more expensive, and the strategy that was fine at bf16
can be the wrong one at fp8.** Practitioners discover this as "fp8 gave us 1.3x, not
2x", and the reason is in that table.

**The practical recipe, and the format trap from
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}).** Keep the master weights and the
optimizer in fp32, cast to fp8 for the forward and backward matmuls, keep norms,
softmax and the loss in higher precision, and use per-tensor or finer scaling to keep
the fp8 dynamic range usable. In MaxText this is one config value, and **it is a
different value on each generation**: `nanoo_fp8` for gfx942 (MI300X, MI325X), which
selects the FNUZ format, and `fp8` for gfx950 (MI355X), which selects OCP. Setting the
gfx950 value on a gfx942 part is not a portability warning, it is wrong numerics.

[Chapter 11]({{ '/pages/11-inference' | relative_url }}) handles the inference side:
weight-only quantization, KV cache quantization, fp4 and fp6.

### The Measurement

**Llama 3 8B under MaxText on 8x MI300X, same batch, same everything, one config value
changed.** **[measured]**

```bash
python -m bench.maxtext_run --model llama3-8b --trace --tag bf16
python -m bench.maxtext_run --model llama3-8b --trace --tag fp8 --set quantization=nanoo_fp8
```

| | bf16 | `nanoo_fp8` | Ratio |
|---|---|---|---|
| Step time | 3.896 s | 3.011 s | **0.77** |
| TFLOP/s per device | 432.8 | 560.1 | 1.29 |
| MFU against the data sheet | 33.1% | 42.8% | 1.29 |
| MFU against the sustained clock | 43.7% | 56.6% | 1.29 |
| Collective time per step per device | 0.273 s | 0.216 s | 0.79 |
| Of which exposed | 0.213 s | 0.118 s | 0.55 |
| Exposed as a share of the step | 5.5% | 3.9% | |

**1.29x, which is the "1.3x, not 2x" this section predicted, arrived at for a different
reason than the one it gave.**

**The prediction above was that fp8 makes communication relatively more expensive. It does
not, at least not here.** Exposed communication fell from 5.5% to 3.9% of the step, in
absolute terms almost halving. **The "every threshold doubles" table is too pessimistic
about FSDP**, and the trace says why: the collective kernels come in the same numbers in
both runs, but two of the three largest families roughly halve their duration while the
third does not move. That is the signature of **the weight all-gather carrying fp8 while
the gradient all-reduce stays in higher precision**. FSDP gathers what the matmul will
consume, and if the matmul consumes fp8, so does the wire.

**So split the table's claim in two.** The DP threshold does double, because the gradient
all-reduce is genuinely unchanged. **The FSDP threshold does not**, because the weight
gather shrinks along with the compute. Tensor parallelism sits with DP: its collectives
carry activations, and MaxText did not quantize those here.

**Then where did the missing 0.7x go? Attention, which fp8 does not touch.** The fused
attention kernel is 15.4% of device kernel time in the bf16 run and 19.4% in the fp8 run,
and since the fp8 run's total is 23% smaller, that is the same absolute time in both.
**Halving the cost of the matmuls cannot help the 15% of the step that is not matmuls**,
and Amdahl's arithmetic on a 15% fixed remainder caps the speedup near 1.3x almost exactly
where we landed.

**The practical reading, which is more encouraging than the table suggested.** fp8 is worth
taking: 1.29x for one config value, with the communication picture improving rather than
degrading. What it will not do is approach 2x, and the thing to look at when it does not is
**what fraction of your step is already not matmul**, not whether you have become
communication-bound. The
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) triage order applies
unchanged.

**One confirmation worth recording, since Chapter 2 warns about the format trap.** The
`nanoo_fp8` run's kernels appear in the trace as `NANOOFp8DotGeneralOp_0` through `_6`, so
the FNUZ path really is the one that ran on gfx942, rather than a silent fall back to bf16.
If you set this and see no such kernels, the config did not take.

## A Decision Procedure

**Explicit, in order, and it takes about ten minutes with a calculator.**

**Step 1: does the parameter state fit?** Compute `16 * params` bytes. If it is under
about 150 GB, plain data parallelism is available and you should probably use it. If
not, you need FSDP, tensor parallelism, pipeline parallelism, or some combination, and
the rest of this procedure is about which.

**Step 2: pick the per-device token batch.** It must clear the ridge point from
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}), 247 tokens for dense bf16, and it
must clear whichever communication threshold applies: 2700 tokens for data parallelism,
4100 for FSDP. Multiply by the device count and check the result against the
convergence limit. If those constraints do not have a common solution, you need
gradient accumulation or fewer devices.

**Step 3: do activations fit at that batch?** `4 MiB per token` for an 8B-shaped model,
scaling roughly with `L * F`. If not, in this order: turn on the matmul-output remat
policy, add tensor parallelism, then full remat. Full remat last, because 33% of your
FLOPs is a lot to pay for memory you might get another way.

**Step 4: choose the tensor-parallel degree from `F`.** `|Y| - 1 < F / 1816` in bf16,
and half that in fp8. Cap it at 8, because that is the scale-up domain. If `F` cannot
support the degree you need to fit the model, go to step 5.

**Step 5: add pipeline parallelism only if steps 1 to 4 do not have a solution.** It
costs a bubble of `(|Z|-1)/(m + |Z|-1)` and demands `m > 9(|Z|-1)` microbatches for a
10% bubble. Put it on the axis that crosses hosts.

**Step 6: lay out the mesh so that the frequent collectives are intra-node.** Tensor
and expert axes last, data and pipeline axes first. Verify by reading `replica_groups`
out of the HLO rather than trusting the config, per
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}).

**Step 7: run it and find out you are at 22% MFU.** That is not a failure of this
procedure, it is the normal outcome, and
[Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) is the triage list.
**Everything in that chapter except the MoE-kernel section is readable from here**, so
if you are training a dense model right now you do not need to get through sparsity
first.

**Three regimes worth naming, because the procedure resolves differently in each.**

- **Small model, many GPUs.** Data parallel until the convergence limit, then stop
  adding GPUs. The failure mode is a batch so large the model stops learning, and no
  amount of systems work fixes it.
- **Large model, enough GPUs.** FSDP across hosts, tensor parallelism inside them, remat
  as needed. The failure mode is memory, and the lever is `|X|`.
- **Large batch, long context.** Context parallelism inside the node, since it is nearly
  free above 32k tokens, with data parallelism outside. The failure mode is activation
  memory, and attention FLOPs dominate, so
  [Chapter 5]({{ '/pages/5-transformers' | relative_url }})'s crossover matters more than
  anything in this chapter.

## Worked Problems

**Question 1:** You have 64 MI300X (8 nodes) and want to train Llama 3 70B in bf16.
Pick a parallelism configuration and justify each degree against the inequalities.

{% details Click here for the answer. %}

**Step 1, memory.** `16 * 70.6e9 = 1130 GB` of parameter state. Across 64 devices that
is 17.6 GB each if fully sharded, so FSDP over all 64 would fit easily, but FSDP across
hosts puts `3L` collectives on the slow axis. Better to shard within the node too.

**Step 4, tensor parallelism.** `F = 28672`, so `|Y| - 1 < 28672 / 1816 = 15.8`, which
permits up to 16-way. The scale-up domain caps it at 8. Take `|Y| = 4` or `8`.

**And [The Optimal Split](#the-optimal-split).** With a global batch of, say, 4M tokens (2048 sequences of
2048), `|X|/|Y| = 4 * 4.19e6 / (9 * 28672) = 65`, which with `|X| * |Y| = 64` wants
`|Y| = 1`. That is the formula telling you the batch is large enough that tensor
parallelism is not needed for *bandwidth* reasons at all. So the tensor-parallel degree
should be chosen purely for memory.

**A defensible answer:** `|X| = 16` (FSDP, spanning 8 hosts and 2 GPUs within each) and
`|Y| = 4` (tensor parallel, intra-node).

- Parameter state: `1130 / 64 = 17.6 GB` per device. Comfortable.
- Per-device tokens at a 4M global batch: `4.19e6 / 16 = 262144`, which is 64x the FSDP
  threshold of 4100. The gradient collectives will hide completely.
- Activations: `4 MiB per token` scaled to 70B's shape is roughly 11 MiB per token,
  divided by `|Y| = 4` for the tensor-sharded part. At 262144 tokens per device that is
  still far too much, so you need gradient accumulation: run 32 microbatches of 8192
  tokens each. Microbatch activations are then about 22 GB with the matmul-output remat
  policy.
- Mesh order: `X` first so FSDP spans hosts, `Y` last so tensor parallelism stays inside
  the baseboard.

**Sanity-check the answer against the alternative.** `|Y| = 8, |X| = 8` also works and
halves activation memory again, at the cost of moving tensor parallelism to 8-way where
the `F` requirement is 12712 against 28672, still fine. Either is defensible; the one to
avoid is any configuration where `|Y| > 8`.

{% enddetails %}

**Question 2:** A trace from an 8-GPU job shows a step time of 420 ms, of which one
all-reduce of a `[8192, 28672]` bf16 buffer takes 95 ms. What strategy is this, what
fraction of the step is communication, and does it match the roofline?

{% details Click here for the answer. %}

**The buffer is weight-shaped**, `D` by `F` for Llama 3 70B, so this is a gradient
all-reduce and the strategy is data parallelism (or the reduce-scatter half of FSDP, but
an all-reduce rather than a reduce-scatter says plain DP).

**Fraction of step:** `95 / 420 = 23%`. That is bad: the whole point of the DP
all-reduce is that it hides behind the backward pass, so 23% on the critical path means
overlap is not happening. Check the two scheduler flags from
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}) before looking at anything else.

**Does the 95 ms itself make sense?** The buffer is `8192 * 28672 * 2 = 470 MB`, and an
all-reduce moves `2V(n-1)/n` per device:

```
2 * 470e6 * (7/8) / 320e9 = 2.6 ms
```

**95 ms against a prediction of 2.6 ms is a factor of 36, so this is not a bandwidth
problem at all.** Something else is going on: the collective is serialising against
compute, or it is being issued at a low stream priority, or the buffer is being copied
to host and back. The lesson is that the roofline is most useful when it *fails*: a 36x
gap is not a tuning opportunity, it is a bug, and knowing the difference is the point of
computing the number first.

{% enddetails %}

**Question 3:** Your 8B model trains at 38% MFU in bf16. You switch to fp8 and get 47%
MFU. Did fp8 help?

{% details Click here for the answer. %}

**Almost certainly yes, and the MFU numbers are close to meaningless as stated, because
they are computed against different peaks.** MFU divides by `C_peak`, and `C_peak` for
fp8 is 2614.9 TFLOP/s against bf16's 1307.4.

- bf16: `0.38 * 1307.4 = 497 TFLOP/s` achieved.
- fp8: `0.47 * 2614.9 = 1229 TFLOP/s` achieved.

**That is a 2.47x throughput improvement**, which is more than the 2x the hardware
offers, so one of the two figures is wrong or the comparison is not like for like.
Plausible explanations, in the order worth checking: the fp8 run used a larger batch,
or the bf16 run was communication-bound and the fp8 run was not, or the MFU
denominators were not what you assumed.

**The general habit: never compare MFU figures across precisions.** Convert to achieved
FLOP/s, or better, compare step times at a fixed batch. This is the same error as
comparing MFU against HFU from
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}), and it is just as common.

{% enddetails %}

## References

**Strategies, in the order the chapter covers them.**

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
  (Rajbhandari et al., 2019). The three stages, and the source of FSDP's cost model.
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
  (Shoeybi et al., 2019). The column-then-row tensor-parallel pairing used above.
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
  (Korthikanti et al., 2022). Sequence parallelism in the Megatron sense, which is the
  one people confuse with context parallelism.
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
  (Huang et al., 2018). The bubble formula.
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)
  (Liu et al., 2023). Context parallelism's ring exchange and its overlap argument.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
  (Narayanan et al., 2021). Composing the strategies, and the 1F1B schedule.

**Low precision.**

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (Micikevicius et
  al., 2022). The OCP fp8 formats, which gfx950 implements and gfx942 does not.
- [ROCm precision support](https://rocm.docs.amd.com/en/docs-6.4.3/reference/precision-support.html)
  (AMD). The FNUZ variants, and which architectures have which.
- [MaxText quantization configuration](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/configs/base.yml)
  (AI-Hypercomputer). The `quantization` field, including the `nanoo_fp8` value for
  MI300 and MI325 and `fp8` for newer parts. Read against commit `9f9ac05`,
  4 August 2026.

**Reference treatments.**

- [How To Scale Your Model, Part 5: Training](https://jax-ml.github.io/scaling-book/training/)
  (Google DeepMind). The same five strategies with TPU constants, and the origin of the
  five-part treatment structure used here.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
  (Hugging Face). A complementary, more empirical walk through the same decisions on
  NVIDIA hardware.
