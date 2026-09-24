---
layout: distill
title: "All the Transformer Math You Need"
description: "How many parameters, FLOPs and bytes, exactly? A counting rule you can apply to any einsum, per-layer accounting for dense and Mixture-of-Experts layers side by side, when attention starts to matter, what a KV cache costs, and what MFU actually measures."
date: 2026-08-04

section_number: 5

previous_section_url: "/pages/4-sharding"
previous_section_name: "Chapter 4: Sharding"

next_section_url: "/pages/6-training"
next_section_name: "Chapter 6: Training"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: The Counting Rule
  - name: Per-Layer Accounting
  - name: When Does Attention Matter?
  - name: MoE Accounting
  - name: KV Cache and the Attention Variants That Shrink It
  - name: Gradient Checkpointing
  - name: MFU, and Why It Is Not Hardware Utilization
  - name: Summary Table
  - name: Worked Problems
  - name: References
---

**Depends on:** [Chapter 1]({{ '/pages/1-rooflines' | relative_url }}) for arithmetic
intensity and [Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) for the constants
that turn a symbolic ridge point into a number. Assumes you have seen a Transformer
before; this chapter counts one rather than explaining one.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

**Every performance argument in the rest of this book starts by counting something**,
and this is the chapter that does the counting. How many parameters, how many FLOPs
per token, how many bytes of activations, how many bytes of KV cache. It is
reference-dense rather than argumentative, and it is the chapter you will come back to
with a specific question rather than read twice.

**Mixture-of-Experts accounting is here, next to the dense accounting, not in an
appendix.** Most frontier models are sparse, and treating sparsity as a modifier you
apply later gets the arithmetic intensity wrong in a way that is hard to recover from.
Sparse models have a different critical batch size, and the difference is a factor of
`E / E_a`, which for a fine-grained model is more than an order of magnitude.

Everything here is **[analytical]** by nature: it is counting, not measuring. The
counts are exact; the approximations are flagged where they appear.

## The Counting Rule

**One rule covers every matmul in a Transformer.** For a contraction, written as an
einsum:

> FLOPs = 2 x (product of every distinct dimension in the expression), where batch and
> contracting dimensions are counted once.

The factor of 2 is one multiply plus one add per term.

**Toy case first.** `jnp.einsum("bd,df->bf", A, W)` with `A` of shape `[4, 512]` and
`W` of shape `[512, 1024]`. The distinct dimensions are `b = 4`, `d = 512`, `f = 1024`,
so:

```
2 * 4 * 512 * 1024 = 4.19e6 FLOPs
```

**And the real case, which is the same idea with more letters.** Attention scores are
`jnp.einsum("bnth,bnsh->bnts", Q, K)`, with a batch of `B`, `N` heads, query length
`T`, key length `S` and head dimension `H`. The distinct dimensions are `B`, `N`, `T`,
`S`, `H`, so it is `2 * B * N * T * S * H`. Note that `b` and `n` appear on both inputs
and the output, so they are batch dimensions and get counted once; `h` appears on both
inputs but not the output, so it is the contracting dimension and also gets counted
once. The rule does not care which is which.

**The backward pass costs exactly twice the forward pass, and it is worth knowing
why.** For `Out = X @ W`, the gradients are:

```
dX = dOut @ W.T      # same shape contraction as the forward, same FLOP count
dW = X.T @ dOut      # likewise
```

Two matmuls of the same cost as the one forward matmul, so a training step is `3x` a
forward pass. Combined with the observation that a forward pass does `2` FLOPs per
parameter per token, that gives the rule everyone quotes:

```
training FLOPs ≈ 6 * params * tokens
```

**Two caveats on that rule, and they are the reason the rest of this chapter exists.**
It counts only operations that have a weight matrix, so it misses the attention score
computation entirely, which has no parameters and grows as `T^2`. And it assumes every
parameter is used by every token, which is false for a Mixture-of-Experts model. Both
corrections are below.

## Per-Layer Accounting

**A modern decoder layer is two blocks, and the parameter count of each is worth
memorising.** We will use a Llama-style layer throughout: pre-norm, grouped-query
attention, and a gated MLP.

**The MLP has `3DF` parameters.** A gated MLP (SwiGLU and relatives) has three
matrices rather than two: a gate projection and an up projection, both `[D, F]`, and a
down projection `[F, D]`.

```
params      = 3 * D * F
forward     = 6 * B * T * D * F        # three matmuls, 2*B*T*D*F each
training    = 18 * B * T * D * F       # times 3 for the backward pass
```

**Attention has `2 * D * (N + K) * H` parameters**, where `N` is the number of query
heads and `K` the number of key/value heads. Four projections: `W_q` is
`[D, N*H]`, `W_k` and `W_v` are `[D, K*H]`, and `W_o` is `[N*H, D]`.

```
params      = 2 * D * (N + K) * H
projections = 12 * B * T * D * (N + K) * H       # training FLOPs
scores      = 12 * B * T * S * N * H             # training FLOPs, no parameters
```

The score term is the `QK^T` product and the `AV` product together, each
`2 * B * N * T * S * H` in the forward pass.

**Three variants of the same block, and the only thing that changes is `K`.**

- **MHA** (multi-head attention): `K = N`. Every query head has its own key and value
  head.
- **MQA** (multi-query): `K = 1`. All query heads share one key/value head.
- **GQA** (grouped-query): `1 < K < N`. Llama 3 uses `K = 8` at every size.

`K` barely affects the FLOP count and enormously affects the KV cache, which is the
whole reason GQA exists and is covered
[below](#kv-cache-and-the-attention-variants-that-shrink-it).

**Norms and the vocabulary projection.** Two RMSNorms per layer at `D` parameters
each, which is negligible in both counts and which we will ignore from here. The
vocabulary projection is `D * V` parameters and `6 * B * T * D * V` training FLOPs,
which is *not* negligible: for a small model with a large vocabulary it can be 10% of
the total.

**Put it together for two real models**, because the abstract form is much less useful
than the numbers.

| | Llama 3 8B | Llama 3 70B |
|---|---|---|
| `L` layers | 32 | 80 |
| `D` model dim | 4096 | 8192 |
| `F` feed-forward dim | 14336 | 28672 |
| `N` query heads | 32 | 64 |
| `K` KV heads | 8 | 8 |
| `H` head dim | 128 | 128 |
| `V` vocabulary | 128256 | 128256 |
| MLP params per layer | 176.2M | 704.6M |
| Attention params per layer | 41.9M | 151.0M |
| All layers | 6.98B | 68.5B |
| Embeddings and output head | 1.05B | 2.10B |
| **Total** | **8.03B** | **70.6B** |

**Notice that the MLP is 81% of the parameters in both.** That ratio is stable across
dense Transformers and it is the licence for a lot of the modelling in
[Chapter 6]({{ '/pages/6-training' | relative_url }}): when we reason about a
Transformer as a stack of MLPs, we are throwing away about a fifth of the parameters
and, as the next section shows, rather more of the FLOPs.

## When Does Attention Matter?

**The score computation is the only part of a Transformer that grows quadratically in
sequence length, so there is a context length past which it dominates everything
else.** Knowing where that is tells you whether you can ignore it.

Compare the score FLOPs to the MLP FLOPs, with `S = T`:

```
scores / MLP = 12*B*T*T*N*H / (18*B*T*D*F) = (2 * T * N * H) / (3 * D * F)
```

**In every model in the table above, `N * H = D` exactly**, because the query
projection is square. That is a design convention rather than a law, but it holds
widely enough to simplify the expression to:

```
scores / MLP = 2*T / (3*F)
```

**So the crossover is at `T = 1.5 * F`**, and for a Llama-shaped model where
`F ≈ 3.5 * D` that is `T ≈ 5 * D`. Concretely:

| Model | Crossover `T` | Score FLOPs at `T = 8192` | At `T = 131072` |
|---|---|---|---|
| Llama 3 8B | 21504 | 38% of MLP | 6.1x MLP |
| Llama 3 70B | 43008 | 19% of MLP | 3.0x MLP |

**Two conclusions, and the second is the one people get wrong.**

**At training context lengths, attention scores are a correction, not the main
term.** At 8k context on a 70B model they are 19% of the MLP FLOPs, and attention
projections add another 21%, so the "stack of MLPs" model of a Transformer
underestimates layer FLOPs by roughly 40% and gets the *scaling* right. That is
accurate enough for parallelism decisions and not accurate enough for an MFU figure,
which is why MFU is always computed from the full `6 * params * tokens` count plus the
score term.

**At long context it inverts completely, and it inverts sooner for smaller models.**
The 8B model crosses over at 21.5k tokens and the 70B at 43k, because the crossover
depends on `F` and bigger models have bigger `F`. A reader who learned "attention is
cheap" on a 70B model at 8k will be badly wrong about an 8B model at 128k, where
scores are six times the MLP cost.

## MoE Accounting

**A Mixture-of-Experts layer replaces one MLP with `E` of them and routes each token to
`E_a` of them.** The accounting consequence is that two parameter counts exist and
they differ by a factor of tens:

```
total params per MoE layer     = E * 3 * D * F
activated params per token     = E_a * 3 * D * F
sparsity                       = E / E_a
```

**FLOPs follow the activated count, and memory follows the total count.** That
sentence is the whole chapter as far as sparse models are concerned, and everything
below is a consequence.

Training FLOPs for the expert layer are `18 * B * T * D * F * E_a`, so
`6 * params * tokens` still holds *if you use the activated parameter count*. Weight
memory, optimizer state and gradient buffers all scale with the total count. So an MoE
is FLOP-light and memory-heavy relative to a dense model of similar quality, which is
why [Chapter 7]({{ '/pages/7-moe' | relative_url }}) introduces expert parallelism as a
memory strategy before it is a communication strategy.

**Now the arithmetic intensity, which is the result that changes how you run these
models.** For a batch large enough that every expert receives at least some tokens,
the layer reads all `E` experts' weights from HBM but does only `E_a` experts' worth of
arithmetic per token. Redo
[Chapter 1]({{ '/pages/1-rooflines' | relative_url }})'s derivation with that asymmetry:

```
intensity = 2 * B_tok * E_a / (w * E)
```

against the dense `2 * B_tok / w`. **An MoE's effective arithmetic intensity is `E_a /
E` of the dense equivalent, so its ridge point is `E / E_a` times higher.** On MI300X
in bf16, where the dense ridge point is 247 tokens per device:

| Model | `E` | `E_a` | Sparsity | Ridge point, tokens per device |
|---|---|---|---|---|
| Dense | — | — | 1 | 247 |
| Mixtral 8x7B | 8 | 2 | 4 | 988 |
| DeepSeek-V2-Lite | 64 | 6 | 10.7 | 2634 |
| Qwen3 30B-A3B | 128 | 8 | 16 | 3952 |
| DeepSeek-V3 | 256 | 8 | 32 | 7904 |

> Expert counts read from the MaxText model configurations for each of these models
> (`num_experts` and `num_experts_per_tok`), checked 4 August 2026.

**Nearly 8000 tokens per device to saturate an MI300X on DeepSeek-V3 is a demanding
requirement, and it is reachable in training and mostly not in serving.** A training
step with 8 sequences of 2048 tokens per device clears it twice over. A decode step at
batch 64 has 64 tokens. **That gap is the single hardest problem in this book** and
[Chapter 11]({{ '/pages/11-inference' | relative_url }}) is where it gets confronted.

**Two modifiers on the count that real models use.**

**Shared experts** run for every token in addition to the routed ones. DeepSeek-V3 has
one, so its activated count is 8 routed plus 1 shared. Shared experts reduce the
effective sparsity, which helps the ridge point, and they give the router something to
fall back on, which helps quality.

**Fine-grained experts** are the trend the table shows: rather than 8 experts of width
`F`, use 128 experts of width `F/16` and activate 8. Total and activated parameters can
be held constant while `E / E_a` goes up, which improves quality per activated FLOP and
makes every systems problem in
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) harder. Note the `base_moe_mlp_dim`
column in a MaxText config: Qwen3 30B-A3B runs `F = 768` per expert against Mixtral's
14336.

## KV Cache and the Attention Variants That Shrink It

**At inference you cache the keys and values for every token you have already
processed, and the size of that cache decides your batch size.** Per sequence:

```
KV bytes = 2 * S * L * K * H * w
```

The leading 2 is keys and values; `w` is bytes per element.

**Per token it is a fixed cost that depends only on the model**, which is the useful
way to hold it:

```
KV bytes per token = 2 * L * K * H * w
```

For Llama 3 70B in bf16: `2 * 80 * 8 * 128 * 2 = 327.7 KB per token`. At 128k context
that is **42.9 GB for a single sequence**, against 141 GB for the weights themselves.
On a 192 GB MI300X, weights plus one maximum-length sequence is 184 GB and you are
done: **one sequence.** This is not a corner case, it is the normal state of affairs
for long-context serving, and it is why
[Chapter 11]({{ '/pages/11-inference' | relative_url }}) spends its time on memory rather
than on FLOPs.

**Now the sequence of attention variants, which is one question asked four times: how
many bytes of cache per token, and what did you pay to get there?**

| Variant | Cache per token, per layer | Llama 3 70B equivalent | Cost |
|---|---|---|---|
| **MHA**, `K = N` | `2 * N * H * w` | 2.6 MB per token | Baseline; nothing shared |
| **MQA**, `K = 1` | `2 * H * w` | 41 KB per token | Quality loss; all heads share one KV |
| **GQA**, `K = 8` | `2 * K * H * w` | 328 KB per token | Small quality loss, 8x saving |
| **MLA** | `(c_kv + r) * w` | 70 KB per token (DeepSeek-V3) | Extra projections, more FLOPs |

**GQA is the free lunch and it is why every current model uses it.** Llama 3 70B with
64 KV heads would need 2.6 MB per token, and 128k context would be 343 GB, which does
not fit on any single GPU made. Sharing key/value heads eight ways costs very little
quality and cuts the cache by exactly 8x.

**Multi-head latent attention is the more aggressive move, and it belongs here rather
than in the MoE chapter.** MLA is an attention mechanism that happens to appear in a
model that is also sparse, and its content is exactly the accounting this section is
already doing. Instead of caching `K` and `V` per head, MLA caches a single
low-rank latent vector per token, of dimension `c_kv`, plus a small decoupled key for
rotary embeddings, of dimension `r`. Keys and values are reconstructed from the latent
on the fly.

For DeepSeek-V3, `c_kv = 512` and `r = 64`, so the cache is
`(512 + 64) * 2 bytes = 1152 bytes per layer per token`, and across 61 layers that is
**70.3 KB per token**. Compare that to Llama 3 70B's 327.7 KB per token, for a model
with nearly ten times the parameters. **MLA is the reason a 671B model is servable at
long context at all.**

What you pay: extra up-projection matmuls to reconstruct `K` and `V`, which is FLOPs
you did not previously spend, and a considerably more complicated attention kernel.
The serving economics are in
[Chapter 11]({{ '/pages/11-inference' | relative_url }}); the memory-profile consequences
during training are in
[Chapter 10]({{ '/pages/10-deepseek' | relative_url }}).

## Gradient Checkpointing

**Activation memory is usually the thing that actually stops you, and it is much
larger than people expect.** Count it for Llama 3 8B.

Everything the backward pass needs, per token per layer, stored in bf16: the layer
input (`D`), the query, key and value projections (`(N + 2K) * H`), the attention
output (`D`), the two MLP projections and their product (`3F`), and the block output
(`D`). Roughly:

```
4*D + (N + 2*K)*H + 3*F
  = 4*4096 + 48*128 + 3*14336
  = 65536 elements = 128 KiB per token per layer
```

Across 32 layers that is **4 MiB per token**. At a per-device batch of 16384 tokens,
which is the sort of number
[Chapter 6]({{ '/pages/6-training' | relative_url }}) will want, that is **64 GiB of
activations** for an 8B model whose weights are 16 GB. Activations are the dominant
term by a factor of four, and they are why the answer to "why am I out of memory" is
almost never "the weights".

**Note:** this count assumes the attention score matrix is never materialised. A naive
attention implementation stores `B * N * T * S` scores, which at `T = S = 2048` and 32
heads is another 256 MB per sequence per layer, and is the reason Flash-style attention
is not optional. [Chapter 8]({{ '/pages/8-getting-to-roofline' | relative_url }}) covers
finding out which implementation you actually got.

**Rematerialization trades that memory for FLOPs, and there are two policies worth
knowing.**

**Full remat, keeping only layer boundaries.** Store the input to each layer and
recompute everything inside it during the backward pass. Memory drops to
`L * D * w = 256 KiB` per token, a **16x reduction** to 4 GiB for the case above. The
cost is one extra forward pass through each layer, so training FLOPs go from
`6 * params * tokens` to `8 * params * tokens`: **33% more arithmetic.**

**Save the matmul outputs.** Keep the results of the expensive contractions and
recompute only the cheap elementwise work around them. Memory saving is smaller,
typically 2-4x, and the FLOP cost is a few percent rather than 33%. In JAX this is
`jax.checkpoint` with a policy such as `jax.checkpoint_policies.dots_saveable`.

**The decision procedure is short.** If you fit without remat, do not use it. If you do
not fit, use the matmul-output policy first, because 33% is a lot of throughput to give
away. Reach for full remat when you need the memory for something that buys more than
33%, which in practice means a larger batch that gets you over the ridge point, or an
expert axis that would not otherwise fit.

## MFU, and Why It Is Not Hardware Utilization

**Every chapter from here on quotes an MFU figure and no chapter has yet defined
one.** Model FLOPs utilization is:

```
MFU = (6 * params * tokens) / (elapsed_seconds * C_peak)
```

with `C_peak` the dense peak from
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}), summed over the devices in the
run. **It measures how much useful modelling work you got per unit of hardware
capability**, and it is deliberately blind to how you got there.

**Hardware FLOPs utilization counts the FLOPs the device actually issued**, which
includes every recomputed activation. The two differ by exactly the remat overhead:

```
HFU = MFU * (issued FLOPs / model FLOPs)
```

**A run with full remat sitting at 55% HFU and 41% MFU has nothing whatsoever wrong
with it.** That is `8/6 = 1.33`, exactly the recompute factor from the previous
section. The hardware is 55% busy and the model is getting 41% of the machine, and
both numbers are true.

**This distinction sits here, immediately after gradient checkpointing, because remat
is what separates the two.** Published figures rarely say which one they are quoting,
and the difference is large enough to change a conclusion: an MFU of 41% and an HFU of
55% describe the same run, and comparing your MFU against someone else's HFU will send
you looking for a 14-point gap that does not exist.

**This book always says which**, and it always means MFU unless the word HFU appears.
Two further conventions worth stating once:

- **MFU includes the attention score FLOPs**, using the `12 * B * T * S * N * H` term
  from above, because at long context excluding them understates utilization
  substantially. Some published figures exclude them. Say which you are doing.
- **For an MoE, `params` in the MFU formula is the *activated* count.** Using the total
  count produces a number that can exceed 100%, which is a good sanity check that you
  used the wrong one.

## Summary Table

Everything above, in one place. `B_tok` is tokens per device, `w` is bytes per element.

| Quantity | Dense layer | MoE layer |
|---|---|---|
| MLP parameters | `3*D*F` | `E * 3*D*F` |
| MLP params used per token | `3*D*F` | `E_a * 3*D*F` |
| Attention parameters | `2*D*(N+K)*H` | same |
| MLP training FLOPs | `18*B*T*D*F` | `18*B*T*D*F*E_a` |
| Attention projection FLOPs | `12*B*T*D*(N+K)*H` | same |
| Attention score FLOPs | `12*B*T*S*N*H` | same |
| Training FLOPs, all in | `6 * params * tokens` + scores | `6 * activated_params * tokens` + scores |
| Arithmetic intensity | `2*B_tok / w` | `2*B_tok*E_a / (w*E)` |
| Ridge point, MI300X bf16 | 247 tokens | `247 * E/E_a` tokens |
| KV cache per token | `2*L*K*H*w` | same |
| Activation memory per token | `L * (4*D + (N+2*K)*H + 3*F) * w` | same, plus dispatch buffers |
| Full remat cost | `+33%` FLOPs | `+33%` FLOPs |

## Worked Problems

**Question 1:** DeepSeek published that pre-training V3 took 2.664M H800 GPU-hours for
14.8T tokens, with 37B activated parameters out of 671B total. What FLOPs utilization
did they achieve? An H800 does 989.5 TFLOP/s dense bf16 and 1979 TFLOP/s dense fp8.

{% details Click here for the answer. %}

**Model FLOPs**, using the activated parameter count:

```
6 * 37e9 * 14.8e12 = 3.29e24 FLOPs
```

**GPU-seconds:**

```
2.664e6 * 3600 = 9.59e9
```

**Achieved rate per GPU:**

```
3.29e24 / 9.59e9 = 343 TFLOP/s
```

**And now the question that makes this exercise worth doing: divided by which peak?**

- Against dense **bf16**: `343 / 989.5 = 34.6%`.
- Against dense **fp8**: `343 / 1979 = 17.3%`.

**DeepSeek-V3 was trained in fp8, so 17.3% is the honest MFU** and 34.6% is a
"bf16-equivalent" figure. Both numbers get quoted in public and they differ by 2x. If
you are comparing your own fp8 run against a published bf16-equivalent figure you will
conclude you are doing twice as well as you are.

**Two more sanity checks worth doing on any number like this.** Using the *total* 671B
parameters instead of the activated 37B gives 6.2 PFLOP/s per GPU, which is more than
three times the hardware's fp8 peak: an impossible answer is how you find out you used
the wrong parameter count. And 2.664M GPU-hours is the pre-training figure only;
DeepSeek's total including context extension and post-training is 2.788M, which would
move the answer by 4%.

{% enddetails %}

**Question 2:** You want to serve Llama 3 70B in bf16 at 128k context on one 192 GB
MI300X. How many concurrent sequences fit? What changes if you quantize the KV cache
to fp8, and what changes if you quantize the weights instead?

{% details Click here for the answer. %}

**Weights:** 70.6B parameters at 2 bytes is **141 GB**. That leaves 51 GB, and you
should reserve a few GB for activations and workspace, so call it 48 GB usable.

**KV cache per sequence** at 128k context:

```
2 * 131072 * 80 * 8 * 128 * 2 = 42.9 GB
```

**So one sequence.** `48 / 42.9 = 1.1`. You have a 192 GB accelerator, a model that
fits comfortably, and a batch size of one, which
[Chapter 1]({{ '/pages/1-rooflines' | relative_url }}) tells you is 247x below the ridge
point.

**Quantize the KV cache to fp8** and it halves to 21.5 GB per sequence, so you fit
**two**. Doubling the batch at decode roughly doubles throughput, because decode is
bandwidth-bound and the weight read is shared across the batch. This is the highest-
leverage single change available.

**Quantize the weights to fp8** and they drop to 70.6 GB, leaving 118 GB for cache, so
you fit **two** as well, by a different route. Do both and weights are 70.6 GB, cache
is 21.5 GB per sequence, and you fit **five**.

**The lesson is that these two levers multiply and are usually mistaken for
alternatives.** [Chapter 11]({{ '/pages/11-inference' | relative_url }}) works the
throughput consequences properly, including why the answer would be different on a
model with MLA.

{% enddetails %}

**Question 3:** You are choosing between two MoE configurations with the same total and
activated parameter count: 8 experts with 2 active, or 128 experts with 8 active. On
an 8-GPU MI300X node, what per-device token batch does each need to be compute-bound,
and what does that mean for the global batch size?

{% details Click here for the answer. %}

**Ridge points.** The dense MI300X bf16 figure is 247 tokens per device, scaled by
`E / E_a`:

- 8 experts, 2 active: `E/E_a = 4`, so **988 tokens per device**.
- 128 experts, 8 active: `E/E_a = 16`, so **3952 tokens per device**.

**Global batch, at 8-way data parallelism**, which is the simplest thing to run on one
node:

- Coarse-grained: `988 * 8 = 7904` tokens, which is 4 sequences of 2048.
- Fine-grained: `3952 * 8 = 31616` tokens, which is 15 sequences of 2048.

**Both are entirely reachable on one node, and that is the point worth taking away.**
The fine-grained model needs 4x the batch to saturate the same hardware, which is
free in training and is exactly what you cannot have in serving. Push the same
comparison to 64 GPUs and the fine-grained model needs a global batch of 253k tokens,
which starts to press against the convergence limit from
[Chapter 1]({{ '/pages/1-rooflines' | relative_url }}).

**One thing this arithmetic does not capture, and it is the reason
[Chapter 7]({{ '/pages/7-moe' | relative_url }}) exists:** it assumes every expert
receives an equal share of tokens. Real routers do not balance perfectly, so the
effective batch per expert is smaller than `B_tok * E_a / E` for the busy experts and
larger for the idle ones, and the step waits for the slowest. The ridge point above is
the optimistic bound.

{% enddetails %}

## References

**Model architectures, for the numbers used above.**

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Meta, 2024). Layer
  counts, model and feed-forward dimensions, and the GQA configuration for the 8B and
  70B models in the tables.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek, 2024).
  The 671B/37B parameter split, the 256-expert configuration with one shared expert,
  the MLA dimensions (`c_kv = 512`, `r = 64`), and the GPU-hour figure in Question 1.
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (DeepSeek, 2024). Where MLA is
  introduced and derived, and the source for DeepSeek-V2-Lite's expert configuration.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Mistral, 2024). The 8-expert,
  2-active configuration that is the simple case throughout this book.
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (Alibaba, 2025). The
  fine-grained 128-expert, 8-active configuration.

**Techniques.**

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
  (Ainslie et al., 2023). Grouped-query attention, and the quality-versus-cache
  trade-off quantified.
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
  (Shazeer, 2019). Multi-query attention, the origin of the whole line of work.
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
  (Chen et al., 2016). Gradient checkpointing, and the `sqrt(L)` memory result.
- [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022). Why the
  attention score matrix is never materialised, which the activation-memory count above
  assumes.
- [jax.checkpoint documentation](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html)
  (JAX). The remat policies, including `dots_saveable`, referenced above.

**Utilization conventions.**

- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
  (Pope et al., 2022). Where the MFU definition this book uses comes from, along with
  the model-versus-hardware FLOPs distinction.
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
  (Korthikanti et al., 2022). The activation-memory accounting this section's per-token
  count follows, and the selective-recompute policy.
