---
layout: distill
title: "How to Think About Inference"
description: "Serving is not training with no_grad. Prefill is compute-bound and decode is bandwidth-bound, which inverts most of the intuitions the training chapters built. How much memory do you need, at what batch size, and what will a token cost?"
date: 2026-08-04

section_number: 11

previous_section_url: "/pages/10-deepseek"
previous_section_name: "Chapter 10: DeepSeek-V2-Lite"

next_section_url: "/pages/12-serving"
next_section_name: "Chapter 12: Production"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: What Are We Optimizing?
  - name: Cost Per Token, in Dollars
  - name: Two Regimes
  - name: The Step-Time Formula
  - name: KV Cache Economics
  - name: Sharding for Decode
  - name: MoE at Decode
  - name: Quantization for Inference
  - name: Speculative Decoding
  - name: What This Chapter Cannot Fix
  - name: Worked Problems
  - name: References
---

**Depends on:** [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) for HBM bandwidth and the
numeric formats, [Chapter 5]({{ '/pages/5-transformers' | relative_url }}) for the KV cache and
the attention variants, and [Chapter 7]({{ '/pages/7-moe' | relative_url }}) for the sparsity
that makes decode hard.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

**Everything in this chapter is [analytical].** We do not run a serving stack, so there
are no measurements here and the chapter says so once rather than tagging every sentence.
That is a deliberate scoping decision rather than an omission: the questions this chapter
answers, how much memory you need, at what batch size, and what a token costs, are settled
cleanly by arithmetic, and keeping the chapter analytical is what lets it be short,
standalone and correct without depending on a stack we do not have.

**Serving is not training with `no_grad`, and the most useful thing this chapter can do is
tell you which of your training intuitions to throw away.** Three of them, up front:

- **Bigger batches are not obviously better.** In training, more tokens per device is
  free throughput. At decode, more tokens per device costs memory you may not have and
  latency your users will notice.
- **Sharding more is not obviously worse.** In training, tensor parallelism has a
  bandwidth threshold you must respect. At decode you can shard *past* it, because you are
  buying bandwidth rather than FLOPs.
- **FLOPs are not the resource.** At decode, the machine is a memory-bandwidth engine
  with some arithmetic units attached, and MI300X's 1307 TFLOP/s is almost irrelevant.

## What Are We Optimizing?

**Four metrics, and they conflict.**

- **TTFT** (time to first token) is how long the user waits before anything happens. It
  is dominated by prefill, so it grows with prompt length.
- **TPOT** (time per output token) is the streaming rate after that, and it is what makes
  a response feel fast or slow. Its reciprocal is what people quote as tokens per second.
- **Throughput** is total tokens per second across all concurrent requests, which is what
  decides how many GPUs you need.
- **Cost per token** is throughput converted into money, and it is usually the metric the
  question was really about.

**Throughput and latency trade against each other through the batch size, and that is the
single knob most serving decisions come down to.** A bigger batch amortises the weight
read across more sequences, so throughput goes up; but every sequence in the batch waits
for the whole batch, so TPOT goes up too.

**Three workload shapes want different answers, and it is worth naming them because a
configuration that is right for one is wrong for the others.**

- **Offline batch.** Summarise a million documents overnight. Latency is irrelevant,
  throughput is everything, use the largest batch that fits.
- **Chat streaming.** A human is reading the output, so TPOT below about 50 ms per token
  is imperceptible and there is no reward for going faster. Batch up to the point where
  TPOT reaches your budget.
- **Agentic and long chain-of-thought.** Thousands of output tokens that nobody reads until
  the end, so it behaves like offline batch on latency but has chat's memory profile,
  with long contexts and unpredictable lengths. The hardest of the three to serve and the
  fastest-growing.

## Cost Per Token, in Dollars

**Everything else in this book is measured in seconds and bytes, and the question you are
usually being asked is what serving this costs.** The conversion is one line:

```
dollars per token = (GPU-hour rate * GPUs) / (3600 * tokens per second)
```

**Worked, for Llama 3 70B in bf16 on one 8-GPU MI300X node** at 8k context and batch 32,
using the step time derived in [The Step-Time Formula](#the-step-time-formula) below:

- Per-GPU bytes read per decode step: 17.6 GB of weights plus 32 sequences worth of
  sharded KV cache, `17.6 + 32 * 0.335 = 28.3 GB`.
- Step time: `28.3e9 / 5.3e12 = 5.3 ms`, producing 32 tokens.
- Throughput: `32 / 5.3e-3 = 5993` tokens per second.
- At $2 per GPU-hour, so $16 per node-hour: `16 / (3600 * 5993) = 7.4e-7` dollars per
  token, or **$0.74 per million output tokens.**

**Carry that figure into [Chapter 12]({{ '/pages/12-serving' | relative_url }})**, which sizes
a deployment against it.

**Two things this number is good for and one thing it is not.** It makes the price of a
latency target explicit: halve the batch to halve TPOT and the cost per token roughly
doubles, which converts an argument about user experience into a line item. And it lets
you sanity-check a build-versus-buy decision against published API prices, which for
70B-class models sit in the same range, meaning the arithmetic is roughly right and the
providers are not making much margin on this class of model.

**What it is not is a total cost of ownership.** It ignores prefill entirely, which for
short outputs and long prompts can dominate; it assumes 100% utilization, which nobody
achieves; and it assumes the batch is always full.

## Two Regimes

**Prefill and decode are different workloads running the same weights, and almost every
statement about inference performance is only true of one of them.**

**Prefill processes the whole prompt at once.** A 2048-token prompt is 2048 tokens through
every matmul, which by
[Chapter 1]({{ '/pages/1-rooflines' | relative_url }})'s ridge point of 247 tokens is
comfortably compute-bound. **Prefill looks exactly like training**, minus the backward
pass, and everything in [Chapter 6]({{ '/pages/6-training' | relative_url }}) applies.

**Decode processes one token per sequence.** A batch of 32 sequences is 32 tokens through
a matmul that reads gigabytes of weights, which is 8x *below* the ridge point.
**Decode is memory-bandwidth-bound and there is nothing you can do about it**, because the
bound is structural: you must read every weight to produce one token.

Here is the arithmetic, because it is the single most useful mental model for inference
performance. At batch `B`, one token per sequence, the linear layers do `2 * B * params`
FLOPs and read `w * params` bytes. Intensity is `2B / w`, which in bf16 is `B`. Against
MI300X's 247 FLOPs per byte:

```
B < 247   =>  memory-bound  (every realistic decode batch)
B > 247   =>  compute-bound (needs 247 concurrent sequences and the memory for their caches)
```

**Ask the same question of attention separately, because it has a different answer.**

| | Linear layers | Attention |
|---|---|---|
| **Prefill** | Compute-bound. `T` tokens through the weights | Compute-bound, and `T^2` FLOPs, so it dominates at long prompts |
| **Decode** | Memory-bound. `B` tokens, `w * params` bytes | Memory-bound, reading `B * S` of KV cache to do `B * S` worth of work |

**The bottom-right cell is the one that surprises people.** Attention at decode has an
arithmetic intensity of about `1`, independent of everything: each cached key is read once
and used for one multiply-accumulate. It is *never* compute-bound, at any batch size, and
increasing the batch does not help because each sequence has its own cache. That is why
long-context decode is expensive in a way that long-context prefill is not.

## The Step-Time Formula

**If you take one thing from this chapter, take this.** The minimum time for one decode
step is everything you must read from HBM, divided by HBM bandwidth:

```
t_decode  >=  (w * params + B * KV_bytes_per_sequence) / β_hbm
```

**Both terms matter and which one dominates tells you what to fix.**

**Llama 3 70B in bf16 on one MI300X**, ignoring that it barely fits:

```
141e9 / 5.3e12 = 26.6 ms per token  =>  37.6 tokens/s, at any batch size that fits
```

**Across 8 GPUs with 8-way tensor parallelism**, where each GPU holds an eighth of the
weights and reads only its own shard:

```
17.6e9 / 5.3e12 = 3.3 ms per token  =>  300 tokens/s
```

**Tensor parallelism at decode buys latency almost linearly**, which is the exact opposite
of its behaviour in training, and
[Sharding for Decode](#sharding-for-decode) is about why.

**Now add the cache term**, at 8k context where Llama 3 70B costs 2.68 GB per sequence,
sharded 8 ways to 335 MB per GPU:

| Batch | Bytes per GPU per step | Step time | Tokens/s (node) | Per-sequence TPOT |
|---|---|---|---|---|
| 1 | 17.9 GB | 3.4 ms | 296 | 3.4 ms |
| 8 | 20.3 GB | 3.8 ms | 2088 | 3.8 ms |
| 32 | 28.3 GB | 5.3 ms | 5993 | 5.3 ms |
| 128 | 60.5 GB | 11.4 ms | 11220 | 11.4 ms |
| 256 | 103.4 GB | 19.5 ms | 13128 | 19.5 ms |

**Read the shape of that table rather than the numbers.** Throughput improves 44x going
from batch 1 to batch 256, and TPOT degrades 5.7x. The knee is where the cache term
overtakes the weight term, at batch 53 here, and past the knee you are trading latency for
throughput at a much worse rate. **The knee is the batch size to serve at**, and it moves
with context length: at 128k context the cache is 5.36 GB per GPU per sequence and the knee
is at batch 3.

## KV Cache Economics

**The cache is what decides your batch size, and therefore your throughput, and therefore
your cost per token.** Everything above followed from it.

**Llama 3 70B in bf16 on an 8-GPU node**, 1536 GB total, 141 GB of weights, so 1395 GB for
cache and workspace. At 327.7 KB per token from
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}):

| Context | Cache per sequence | Max concurrent sequences | Compute-bound? |
|---|---|---|---|
| 2k | 671 MB | 2078 | Yes, 8x over the ridge point |
| 8k | 2.68 GB | 520 | Yes, 2x over |
| 32k | 10.7 GB | 130 | No, half the ridge point |
| 128k | 42.9 GB | 32 | No, 8x under |
| 1M | 335 GB | **0** | Does not fit on the node |

**Two conclusions, and the second is the load-bearing one.**

**Long context and high throughput are directly opposed**, and the exchange rate is
linear: 16x the context is 1/16 the batch. There is no configuration trick that avoids
this, only compression.

**At long context you cannot reach the ridge point at all**, so the machine is
memory-bound no matter what you do, and the compute you paid for is idle. That is not a
tuning failure, it is what serving long context *is*. The correct response is to spend
money on bandwidth and capacity rather than on FLOPs, which is a procurement decision as
much as an engineering one.

**Three levers, in increasing order of how much they buy.**

**GQA**, which you already have: Llama 3 70B with full multi-head attention would be
2.6 MB per token and 128k context would need 343 GB per sequence, which does not fit on
any single GPU. Sharing key/value heads 8 ways is the difference between servable and not.

**Quantizing the cache** to fp8 halves it, to fp4 quarters it. This is the cheapest large
win available and [Quantization for Inference](#quantization-for-inference) covers what it
costs.

**MLA**, if you get to choose the model: DeepSeek-V3 caches 70.3 KB per token against
Llama 3 70B's 327.7 KB, for a model ten times the size. At 128k context that is 9.2 GB per
sequence rather than 42.9, so 151 concurrent sequences on a node rather than 32. **MLA is
worth more than any systems optimisation in this chapter.**

## Sharding for Decode

**Two of the strategies from [Chapter 6]({{ '/pages/6-training' | relative_url }}) are
actively wrong at decode, one is pointless, and one inverts.**

**FSDP is harmful.** Its whole mechanism is to all-gather each layer's weights just in
time and discard them afterwards, which is a good trade when a layer's compute is
milliseconds of matmul. At decode, a layer's compute is microseconds and the all-gather is
the entire step: you have replaced a memory read with a memory read *plus* a collective.
Never serve with FSDP.

**Data parallelism is pointless for latency and fine for throughput**, with a caveat. Two
independent replicas serve two independent batches, which doubles throughput and does
nothing for TPOT. The caveat is that splitting a batch of 64 into two replicas of 32 moves
you *down* the throughput curve in the table above, so if the batch is small, replicating
is worse than batching.

**Pipeline parallelism works and nobody likes it.** Stages are cheap to communicate
between, and the bubble is filled by having many requests in flight, which serving has
naturally. The problem is that each token traverses every stage, so TPOT includes
`|Z|` stage latencies, and it complicates scheduling enormously.

**Tensor parallelism is what remains, and here is the twist.** In training,
[Chapter 6]({{ '/pages/6-training' | relative_url }}) gave a hard threshold: `F > 1816 *
(|Y|-1)`, past which tensor parallelism goes communication-bound and stops paying. **At
decode you can shard past that bound**, because the thing you are buying is not FLOPs.

The reason is in the step-time formula. Tensor parallelism divides the weight bytes each
GPU must read by `|Y|`, so it cuts the memory-bound step time linearly, while the
all-reduce it adds moves only `B * D` elements, which at decode batch sizes is kilobytes
rather than megabytes. **You are trading a large bandwidth saving for a small latency
cost**, and that trade keeps paying well past the point where the FLOP-based inequality
says stop.

What eventually stops it is *latency*, not bandwidth: each all-reduce has a fixed cost of
some microseconds regardless of size, there are two per layer, and at 80 layers that is
160 fixed costs per token. When the per-collective latency times 160 approaches the step
time you have saturated, and on an 8-GPU baseboard you will typically hit the eight-GPU
scale-up ceiling first.

**KV cache sharding follows the attention heads.** With `|Y|`-way tensor parallelism the
`K` key/value heads distribute across the tensor axis, so each GPU caches `K / |Y|` heads.
That works cleanly while `|Y| <= K`, and Llama 3's `K = 8` on an 8-GPU node is exactly the
boundary: at 8-way you have one KV head per GPU and no further to go. Past that you must
either replicate the cache, wasting memory, or shard along the sequence axis, which costs
an all-to-all per attention operation.

## MoE at Decode

**This is where [Chapters 5]({{ '/pages/5-transformers' | relative_url }}) and
[7]({{ '/pages/7-moe' | relative_url }}) collide, and it is the hardest problem in this
book.** It deserves to be named as such rather than presented as one more consideration.

**The ridge point is `E / E_a` times higher, and decode is already far below it.**
DeepSeek-V3 needs 7904 tokens per device on MI300X to be compute-bound, against a dense
model's 247. A decode batch of 256 is 30x below that. **A sparse model at decode is
memory-bound by a wide margin and no batch size you can afford fixes it.**

**But there is a genuinely interesting wrinkle, and it works in your favour at small
batch.** At batch 1, a token activates `E_a` of `E` experts, so you read only
`E_a / E` of the expert weights. DeepSeek-V3 at batch 1 reads 8 experts out of 256: the
step reads roughly the activated 37B parameters, not the total 671B. **Sparse models decode
*faster* than dense models of the same total size, at batch 1.**

**And then it stops working, at a batch size you can compute.** With batch `B`, the number
of distinct experts touched is about `min(B * E_a, E)`, so the weight bytes you read grow
linearly with batch until every expert is hit:

```
B_saturate ≈ E / E_a
```

For DeepSeek-V3 that is 32. **Past batch 32 you are reading all 671B parameters every
step**, and further batching adds cache bytes without adding weight bytes, so the curve
flattens.

**That gives sparse models the opposite batch-scaling behaviour to dense ones**, which is
the fact to take away:

| | Dense | Sparse |
|---|---|---|
| Weight bytes per step | Constant in `B` | Grows linearly to `B = E/E_a`, then constant |
| Throughput gain from batching, small `B` | Nearly linear | **Nearly none** |
| Throughput gain from batching, large `B` | Sublinear | Nearly linear |
| Batch to be compute-bound | 247 | `247 * E/E_a` |

**Below `B = E/E_a`, batching a sparse model buys you almost nothing**, because each new
sequence brings its own new experts to read. That is the opposite of the entire economic
logic of serving, which is that batching amortises the weight read. It is the reason
sparse models are harder to serve efficiently than their activated parameter count
suggests, and the reason serving engines put so much effort into expert-aware routing of
requests.

**Expert placement then interacts with the memory ceiling.** 671B parameters in fp8 is
671 GB, which needs at least four MI300X for weights alone and realistically a full node
once you add cache. With experts spread across 8 GPUs, every decode step needs an
all-to-all dispatch of a handful of tokens: tiny in bytes, so entirely latency-bound, and
there are two per layer per token. At 61 layers that is 122 latency-bound collectives per
token, and this is the term that dominates sparse decode on real systems.

## Quantization for Inference

**[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) owns the formats and
[Chapter 6]({{ '/pages/6-training' | relative_url }}) owns fp8 training, so this section is
specifically the inference levers, and the distinction that matters is that at decode you
are buying bandwidth, not FLOPs.**

**Weight-only quantization is the big one, and it helps decode far more than prefill.**
Store weights in fp8 or fp4 and dequantize on the way into the matmul. The FLOP rate does
not change, so prefill barely improves; the bytes read halve or quarter, so decode step
time halves or quarters. In [Chapter 1]({{ '/pages/1-rooflines' | relative_url }})'s terms,
it moves the ridge point *down*, from 247 tokens to 123 for fp8 weights, making more of
the batch range compute-bound.

Llama 3 70B, batch 1, 8-way tensor parallel:

| Weight format | Bytes per GPU | Step time | Tokens/s |
|---|---|---|---|
| bf16 | 17.6 GB | 3.3 ms | 300 |
| fp8 | 8.8 GB | 1.7 ms | 600 |
| fp4 | 4.4 GB | 0.8 ms | 1200 |

**KV cache quantization is the second lever and it buys batch rather than latency.**
Halving the cache doubles the number of sequences that fit, which moves you up the
throughput curve. The two levers multiply:
[Chapter 5]({{ '/pages/5-transformers' | relative_url }})'s second worked problem shows fp8
weights plus fp8 cache taking a 128k-context deployment from one sequence per GPU to five.

**What quantization costs, stated plainly.** Weight-only fp8 with per-channel scaling is
close to free in quality for most models. fp4 needs finer-grained scaling and careful
calibration, and it is where quality regressions start showing up in evaluations rather
than in perplexity. KV cache quantization to fp8 is generally safe; to fp4 it is not,
because the cache is read many times and errors compound over the sequence.

**And the format trap from [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) shows up
here as a checkpoint problem.** gfx942 implements FNUZ fp8 and gfx950 implements OCP fp8,
so an fp8 checkpoint quantized on MI300X does not load correctly on MI355X, and vice
versa. This is why vLLM and SGLang carry a load-time conversion keyed on the architecture
string containing `gfx94`.
[Chapter 12]({{ '/pages/12-serving' | relative_url }}) meets it again on the way out of JAX.

**The forward look, and one genuine oddity.** MI355X adds fp6 and fp4 at
10066 TFLOP/s dense, and **MXFP6 runs at the same rate as MXFP4** rather than half of it.
So on gfx950, fp6 is a nearly free accuracy upgrade over fp4: same throughput, one more
bit of mantissa. For weight-only quantization, where you are buying bytes rather than
FLOPs, fp6 weights are 1.5x the bytes of fp4, so the trade is not free on the bandwidth
side. But for anything compute-bound, prefer fp6.

## Speculative Decoding

**Short, but present, because it is the cleanest illustration that the step-time formula
governs everything.** Speculative decoding wins precisely by converting spare FLOPs into
fewer weight loads, which is the one trade a bandwidth-bound decode rewards.

**The mechanism.** A small draft model proposes `k` tokens cheaply. The large model
verifies all `k` in a single forward pass, which costs one weight read rather than `k`. You
keep the longest accepted prefix.

**The arithmetic.** With per-token acceptance probability `α`, the expected number of
tokens accepted per verification pass is:

```
E[tokens] = (1 - α^(k+1)) / (1 - α)
```

At `α = 0.7` and `k = 4`, that is `(1 - 0.168) / 0.3 = 2.77` tokens per verification. If
the draft model costs a fraction `c` of the target per token, the total cost of a
verification cycle is `1 + k*c` target-equivalent passes, so:

```
speedup = E[tokens] / (1 + k*c) = 2.77 / (1 + 4*0.1) = 1.98x
```

**Roughly 2x, and the sensitivity to `α` is the thing to know.** At `α = 0.5` the same
configuration gives `1.94 / 1.4 = 1.39x`; at `α = 0.9` it gives `4.10 / 1.4 = 2.93x`.
**Acceptance rate is everything**, and it depends on how well the draft model matches the
target on your actual traffic, which is an empirical question rather than an architectural
one.

**Why it works at all is worth stating in the terms of this chapter.** Verifying `k`
tokens is `k` tokens through the same weight read, so it moves you `k`x up the arithmetic
intensity axis for free. At decode batch 1 you are 247x below the ridge point, so there is
an enormous amount of unused compute lying around, and speculation is the cleanest way to
spend it. **It also stops helping as the batch grows**, because a large batch is already
using that headroom: at batch 64 with `k = 4` you are effectively at 256 tokens per weight
read, which is at the ridge point, and further speculation buys nothing.

The implementation lives in the serving engine, not in your training stack.
[Chapter 12]({{ '/pages/12-serving' | relative_url }}) is where it becomes a configuration
flag.

## What This Chapter Cannot Fix

**Three problems survive a perfect single-request roofline, and they are the reason
serving engines exist.**

**Requests arrive at different times and in different shapes.** Everything above assumed a
batch of identical sequences stepping in lockstep. Real traffic has a request arriving
mid-step, a 200-token prompt next to a 100k-token one, and generations that finish at
unpredictable times. Naive batching makes every request wait for the slowest, which
destroys the latency numbers above. **Scheduling is a systems problem and no amount of
arithmetic solves it.**

**The KV cache is allocated dynamically and fragments.** Every table in this chapter
assumed you can use all 1395 GB of non-weight memory for cache. In reality sequences grow
one token at a time to unknown final lengths, so a contiguous allocator either
over-reserves for the maximum length, wasting most of the memory, or fragments. **Paged
allocation is the fix and it is not something you can derive.**

**Prefill and decode want different hardware.** Prefill is compute-bound and rewards
FLOPs; decode is bandwidth-bound and rewards HBM. Running both on the same GPU means one
of them is always using the wrong machine, and interleaving them means a long prefill
stalls every decode in flight.

**Those three are the honest reason this book stops here**, and they are what
[Chapter 12]({{ '/pages/12-serving' | relative_url }}) is about: continuous batching, paged
attention, and disaggregated prefill and decode are the three answers, in the same order.
They are also the reason we are not writing a serving engine.

## Worked Problems

**Question 1:** Someone proposes upgrading from MI300X to a hypothetical part with twice
the FLOPs and the same memory bandwidth, to make decode faster. What do you tell them?

{% details Click here for the answer. %}

**It will not make decode faster at all.** Decode step time is
`(w * params + B * KV) / β_hbm`, and there is no `C` in that expression. Doubling the FLOP
rate changes nothing.

**What it would help is prefill**, which is compute-bound, so TTFT would roughly halve on
long prompts. Whether that matters depends on your workload: for chat with short prompts
and long outputs, almost not at all; for a retrieval-augmented application with 32k-token
prompts and 200-token answers, substantially.

**And the corollary, which is the useful part of the answer:** the upgrade you want for
decode is bandwidth and capacity. MI355X is instructive here, because it delivers both:
1.9x the FLOPs, which does little for decode, and 1.5x the bandwidth with 1.5x the
capacity, which does a lot. Say which of the three numbers you are buying.

{% enddetails %}

**Question 2:** You are serving Llama 3 70B in bf16 on an 8-GPU MI300X node at 8k context.
Your SLO is 40 ms per output token. What batch size should you run, what throughput does
that give, and what does a million tokens cost at $2 per GPU-hour?

{% details Click here for the answer. %}

**Start from the step-time formula.** Per GPU, weights are 17.6 GB and each sequence's
sharded cache at 8k context is 335 MB. The 40 ms budget allows:

```
0.040 * 5.3e12 = 212 GB of reads per step
(212e9 - 17.6e9) / 0.335e9 = 580 sequences
```

**So bandwidth is not your constraint: memory is.** From
[KV Cache Economics](#kv-cache-economics), 8k context on this node fits about 520
sequences, so take **batch 512**.

**Throughput and step time at batch 512:**

```
bytes per GPU = 17.6 + 512 * 0.335 = 189 GB
step time     = 189e9 / 5.3e12 = 35.7 ms
throughput    = 512 / 0.0357 = 14342 tokens/s
```

**Cost:** `16 / (3600 * 14342) = 3.1e-7` dollars per token, or **$0.31 per million
tokens.**

**Three things worth noticing.** The SLO turned out not to bind, which is common at short
context and is why people over-provision GPUs for latency they were never going to
violate. Throughput at batch 512 is 2.4x that at batch 32, but cost per token fell by 2.4x
too, so the large batch is strictly better if you can fill it. And this is a theoretical
ceiling: it assumes every one of those 512 slots is occupied by an active sequence at all
times, which is exactly what
[What This Chapter Cannot Fix](#what-this-chapter-cannot-fix) says you cannot count on.
Halve it for scheduling reality and you are still at $0.62 per million.

{% enddetails %}

**Question 3:** Why does doubling the decode batch from 8 to 16 nearly double throughput
for Llama 3 70B, but barely change it for DeepSeek-V3?

{% details Click here for the answer. %}

**For the dense model, the weight read is shared.** At batch 8 and batch 16 you read the
same 17.6 GB per GPU of weights, and the cache term is small at 8k context, so step time
goes from 20.3 GB to 22.9 GB of reads, a 13% increase, while producing twice the tokens.
**Throughput improves by about 1.8x.** That is the whole economic logic of batched serving.

**For the sparse model, each new sequence brings new experts.** With `E = 256` and
`E_a = 8`, batch 8 touches at most 64 distinct experts and batch 16 touches at most 128, so
the expert weight bytes read roughly *double* along with the token count. Step time
doubles, throughput is flat. **Batching buys nothing until you pass `B = E / E_a = 32`,
after which every expert is being read anyway and the dense logic resumes.**

**The practical consequences of that, in order of how much they matter.** A sparse model
wants either a very small batch, where it is genuinely cheap per token, or a batch well
past `E / E_a`, where amortisation restarts; the middle is the worst of both. Expert-aware
request routing, grouping requests likely to activate the same experts, is worth real
effort for sparse models and nothing at all for dense ones. And the activated parameter
count is a good guide to sparse decode cost at batch 1 and a badly misleading one at batch
64, which is the single most common error in reasoning about serving MoE models.

{% enddetails %}

## References

**The arithmetic.**

- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (Pope et
  al., 2022). The two-regime model, the decode step-time formula, and the sharding
  analysis this chapter's decode section follows.
- [How To Scale Your Model, Part 7: Inference](https://jax-ml.github.io/scaling-book/inference/)
  (Google DeepMind). The same arithmetic in TPU idiom, including the batch-size sweeps
  this chapter's tables are modelled on.

**Techniques.**

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
  (Leviathan et al., 2022). The acceptance-rate formula used above.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
  (Chen et al., 2023). The complementary treatment, with the rejection-sampling
  correctness argument.
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
  (Kwon et al., 2023). The fragmentation problem named in
  [What This Chapter Cannot Fix](#what-this-chapter-cannot-fix), and vLLM's answer to it.
- [KV Cache quantization in vLLM](https://docs.vllm.ai/en/latest/features/quantization/)
  (vLLM). What the levers in this chapter look like as configuration.

**Hardware.**

- [AMD Instinct MI300X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
  and [MI355X GPU brochure](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf).
  The bandwidth and capacity figures every table here divides by, and the fp6/fp4 rates.
- [ROCm precision support](https://rocm.docs.amd.com/en/docs-6.4.3/reference/precision-support.html)
  (AMD). The FNUZ versus OCP fp8 split that makes quantized checkpoints
  generation-specific.
