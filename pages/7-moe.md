---
layout: distill
title: "Mixture-of-Experts at Scale"
description: "Only a fraction of the parameters run per token, so why isn't it a fraction of the time? Routing and imbalance, capacity and dropping, the three ways to implement an expert layer and what each costs in FLOPs, and which of them a JAX user on ROCm can actually reach."
date: 2026-08-04

section_number: 7

previous_section_url: "/pages/6-training"
previous_section_name: "Chapter 6: Training"

next_section_url: "/pages/8-getting-to-roofline"
next_section_name: "Chapter 8: Getting to Roofline"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Why the MoE Roofline Is Different
  - name: Routing
  - name: Load Imbalance
  - name: Capacity, Dropping, Padding, and Going Dropless
  - name: Three Ways to Implement an Expert Layer
  - name: Which of the Three You Can Get on AMD in JAX
    subsections:
      - name: The Missing Mesh Axis
      - name: The Implementations, Measured
      - name: Working the Win Condition
      - name: What AMD Ships, And Why
      - name: AITER Has No Grouped MoE GEMM
  - name: All-to-All Dispatch and Combine
  - name: Expert Parallelism
    subsections:
      - name: But The Lever Points The Other Way
  - name: Anatomy of Three Real Models
  - name: The Four Numbers to Log for Every MoE Run
  - name: Worked Problems
  - name: References
---

> **Draft.** The arithmetic, the three implementations and the config mapping are
> written and checked against MaxText source. The two measurements that would turn the
> central finding from a source-code reading into a table are not in yet, and are marked
> where they belong.

**Depends on:** [Chapter 4]({{ '/pages/4-sharding' | relative_url }}) for collective costs
and `shard_map`, [Chapter 5]({{ '/pages/5-transformers' | relative_url }}) for MoE
parameter and FLOP accounting, and
[Chapter 6]({{ '/pages/6-training' | relative_url }}) for the five-part treatment that
expert parallelism reuses.

{% details Notation used in this chapter %}

{% include notation.liquid %}

{% enddetails %}

**A Mixture-of-Experts model activates a fraction of its parameters per token, and
somehow does not run in a fraction of the time.** DeepSeek-V3 activates 37B of 671B
parameters, so an eighteenth of the arithmetic of the dense equivalent, and nobody gets
anything close to eighteen times the throughput.

**The gap has one root cause and about five symptoms.** The root cause is that the
FLOPs an MoE layer performs are decided at runtime by a router, and every layer of the
software stack below that decision prefers shapes known at compile time. Everything in
this chapter is a consequence: padding to fixed capacity, dropping tokens that overflow,
imbalance that idles most of the device, dispatch collectives that would not exist in a
dense model, and matmul kernels that are not the ones you would have chosen.

**The chapter's spine is a question nobody has answered in public: of the three ways to
implement an expert layer, which one can a JAX user on ROCm actually run fast, and what
does the answer cost them in FLOPs.** That question has a partial answer already, from
reading MaxText's source, and it is not a comfortable one.

## Why the MoE Roofline Is Different

**For a dense model, the naive roofline predicts well.** Count the parameters, multiply
by six and by the token count, divide by the peak FLOP rate, and you have a step time
that is usually within 2x of reality and often within 20%. Everything in
[Chapter 6]({{ '/pages/6-training' | relative_url }}) is built on that being true.

**For an MoE, the same procedure can be wrong by an order of magnitude, and it is wrong
in the optimistic direction.** Four multiplicative factors sit between the activated
FLOP count and the FLOPs your device actually issues:

```
issued FLOPs = activated FLOPs
             * (implementation factor: 1 to E/E_a)
             * (capacity factor: 1 to ~1.25, or a dropped-token discount below 1)
             * (imbalance factor: max tokens per expert / mean)
             / (kernel efficiency: 1 down to whatever a ragged GEMM achieves)
```

**The first factor is the big one, and it is chosen by a config field.** Depending on
how the expert layer is implemented, you either do exactly the activated FLOPs or you do
`E / E_a` times as many, which for DeepSeek-V3 is 32x. **The sparsity you bought is
handed straight back.**

**The second and third are the ones that hide.** Padding and imbalance both cost time
without showing up as anything obviously wrong: the step is slower, every kernel looks
healthy, and MFU is bad. Chapter
[8]({{ '/pages/8-getting-to-roofline' | relative_url }})'s triage list will not find them
unless you know to look, which is why
[The Four Numbers to Log](#the-four-numbers-to-log-for-every-moe-run) exists.

**And the last one is where AMD specifically bites**, because the kernels that make the
efficient implementation efficient are not reachable from JAX on ROCm today.

**One more thing that is different, from
[Chapter 5]({{ '/pages/5-transformers' | relative_url }}) and worth restating because it
governs everything downstream: the ridge point is `E / E_a` times higher.** DeepSeek-V3
needs 7904 tokens per device on MI300X to be compute-bound, against 247 for a dense
model. In training that is reachable. In serving it is not, and
[Chapter 11]({{ '/pages/11-inference' | relative_url }}) is where that becomes the hardest
problem in the book.

## Routing

**The router is the smallest matmul in the model and it decides the cost of everything
after it.** Per token, it is a `[D, E]` projection producing one logit per expert, a
softmax, and a top-`E_a` selection:

```python
logits = x @ w_gate                      # [tokens, E]
probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
weights, indices = jax.lax.top_k(probs, k=num_experts_per_tok)
```

FLOPs are `2 * B_tok * D * E` per layer, which for DeepSeek-V3 is `2 * D * 256` per
token against `2 * D * F * 3 * 8` for the experts: about 0.2% of the layer. **The router
is free and it is entirely in charge.**

**Two routing disciplines, and the difference is which side gets to choose.**

**Token-choice routing** is what everything in this chapter assumes: each token picks
its top `E_a` experts. Simple, and it is what every model in
[Anatomy of Three Real Models](#anatomy-of-three-real-models) does. The problem is that
nothing constrains how many tokens pick a given expert, so the load is whatever the
data says it is.

**Expert-choice routing** inverts it: each expert picks its top `c` tokens. Load is
perfectly balanced by construction, which eliminates the entire
[Load Imbalance](#load-imbalance) section, at the cost of some tokens being processed by
no expert at all and a routing decision that depends on the whole batch, which
complicates autoregressive inference.

**The auxiliary load-balancing loss is how token-choice routing is persuaded to
behave.** Add a term to the loss that penalises unevenness, typically the product of the
fraction of tokens routed to each expert and the mean router probability for that
expert, summed over experts and scaled by `E^2`:

```python
# MaxText's formulation, layers/moe.py
loss = jnp.mean(density * density_prob) * (num_experts ** 2) * load_balance_loss_weight
```

**What it trades against is model quality, and the trade is real.** The loss is asking
the router to make choices it would not otherwise make; too much weight and you have a
uniform router that has stopped routing. In MaxText `load_balance_loss_weight` defaults
to `0.0`, which means load balancing is *off* unless you turn it on, and a run with
severe imbalance is often a run where nobody set it.

**A second, newer approach avoids the loss entirely**: keep a learnable per-expert bias
on the routing scores and nudge it away from overloaded experts between steps. MaxText
exposes this as `routed_bias` with `routed_bias_update_rate`, and DeepSeek-V3 uses it in
preference to an auxiliary loss precisely because it does not perturb the gradient.

**Two lines on router numerics, because the failure they prevent looks like a hardware
fault.** The router should be computed in fp32 even when the rest of the layer is bf16
or fp8. The softmax over `E` logits is where a bf16 router goes unstable, and the
observable symptom is a loss spike or a NaN, which in a book full of hardware the reader
will instinctively blame on the hardware.

Concretely, in MaxText: `float32_gate_logits` controls whether the gate *matmul* runs in
fp32 and **defaults to false**, while the softmax and the top-`k` renormalisation are
cast to fp32 unconditionally. So the risky part is the projection, and turning that
config value on is cheap insurance.

**One correction to a piece of common advice, because we checked.** The literature
recommends a router z-loss, which penalises large logits to keep the softmax in range.
MaxText's `z_loss_multiplier` is *not* that: it applies to the output vocabulary logits
in the cross-entropy, not to the router. If you want a router z-loss you have to add
it. Worth knowing before you conclude that you already have one.

## Load Imbalance

**A dense matmul takes as long as it takes. An expert matmul takes as long as the
busiest expert.** That is the whole of load imbalance, and it costs exactly the ratio
you would expect:

```
imbalance factor = (max tokens routed to any expert) / (mean tokens per expert)
```

**With perfectly balanced routing the mean is `B_tok * E_a / E` tokens per expert.** For
DeepSeek-V2-Lite at 4096 tokens per device, that is `4096 * 6 / 64 = 384` tokens each.
An imbalance factor of 2 means the busiest expert sees 768, the step waits for it, and
half of your expert compute capacity is idle.

**What it looks like in a trace** is a row of expert GEMMs of visibly unequal duration,
with the step gated on the longest. In a grouped-GEMM implementation it is subtler: one
kernel with internally uneven groups, so the durations look fine and the achieved FLOP
rate is bad. **This is the failure mode that most often gets misdiagnosed as a slow
kernel.**

**How to quantify it from a profile:** you cannot, directly, and that is the point of
logging it yourself. The token counts per expert are a tensor inside your program, not
something the profiler can see. `jnp.bincount` on the routing indices costs nothing and
gives you the histogram; see
[The Four Numbers to Log](#the-four-numbers-to-log-for-every-moe-run).

**Two things about how imbalance behaves over time**, both of which affect how you
interpret a single measurement.

It **improves over training**. Early in training the router is close to random, which is
close to balanced. Then it specialises, which makes it useful and unbalanced, and the
auxiliary loss fights back. A run measured at step 100 tells you very little about the
run at step 10000.

It **varies with the data**, sharply. Experts specialise by domain, so a batch of code
and a batch of prose have different imbalance, and a curriculum that switches domains
mid-training will move your step time with no code change at all.

## Capacity, Dropping, Padding, and Going Dropless

**If you want statically shaped expert matmuls, you have to decide in advance how many
tokens each expert will process. That decision is the quiet FLOP thief.**

**Capacity is the per-expert token budget**, and it is computed from a factor:

```python
# MaxText, layers/moe.py
tokens_per_batch = seq_len * num_experts_per_tok
expert_capacity_per_batch = ceil(tokens_per_batch / num_experts) * capacity_factor
```

So `capacity_factor = 1.0` gives each expert exactly the balanced share, and 1.25 gives
it 25% headroom.

**Then two failure modes, in opposite directions, and both invisible in wall-clock
time.**

**Underfull experts get padded.** An expert that received 200 tokens against a capacity
of 384 still runs a matmul sized for 384. The padding does real FLOPs on zeros. Cost is
`capacity_factor / (utilisation)`, and at `capacity_factor = 1.25` with typical
imbalance you are doing 25-60% more expert arithmetic than the token count justifies.

**Overflowing experts drop tokens, silently.** MaxText's dense path builds a one-hot mask
and truncates it:

```python
trunc_expert_mask = expert_mask * jnp.less_equal(expert_token_count, expert_capacity_per_batch)
```

A token whose chosen expert is full contributes nothing from that expert. No error, no
warning, and the loss curve degrades a little. **This is the single most under-monitored
thing in MoE training.**

**Note the perverse accounting: dropping makes your MFU look better.** Dropped tokens
are FLOPs you did not do, and MFU's numerator is the FLOPs you *should* have done, so a
run that drops 8% of tokens reports about 8% higher MFU than the same run that does
not. If your MFU improved when you lowered `capacity_factor`, check the drop rate
before celebrating.

**Now the dropless alternative, which is where modern implementations have landed.**
Instead of padding to a fixed capacity, size each expert's matmul to whatever actually
arrived. No padding, no dropping, exactly the activated FLOP count. In exchange, **the
expert matmul becomes a ragged shape rather than a rectangular one**: `E` matmuls of
different sizes whose sizes are data, not compile-time constants.

In MaxText, dropless is not one flag but a combination, and the sentinel is a negative
factor:

| Configuration | Behaviour |
|---|---|
| `sparse_matmul: False`, `capacity_factor: -1` | Dropless, dense masked compute |
| `sparse_matmul: False`, `capacity_factor: > 0` | Fixed capacity, drops on overflow |
| `sparse_matmul: True`, `ragged_buffer_factor: -1` | Dropless, grouped GEMM, worst-case buffer |
| `sparse_matmul: True`, `ragged_buffer_factor: > 0` | Grouped GEMM sized for balance, drops beyond it |

> Field names and defaults read from MaxText `src/maxtext/configs/base.yml` at commit
> `9f9ac05`, 4 August 2026. `capacity_factor` defaults to `-1.0`, `sparse_matmul` to
> `true` and `ragged_buffer_factor` to `-1.0`, so **the shipped default is dropless
> grouped GEMM**.

**That trade is the hinge of the whole chapter.** Going dropless converts a
FLOP-efficiency problem into a kernel-availability problem: you stop wasting arithmetic
and start needing a matmul kernel that stays fast on ragged shapes. **On AMD in JAX, the
kernel is the part that is missing.** The next two sections are that story.

## Three Ways to Implement an Expert Layer

**The choice is made in one or two config fields and it moves the FLOP bill by a factor
of `E / E_a`.** Here is each one, with the arithmetic, because the ranking flips with
`E`, `E_a` and the hardware and a ranking would be useless.

**1. Dense masked compute.** Every device runs every expert over every token, then
multiplies by the router weights, which are zero for the `E - E_a` experts a token did
not choose.

```
"BSM,EMH -> BSEH"     # every token through every expert's up-projection
"BSEH,EHM -> BSEM"    # and every expert's down-projection
"BSEM,BSE -> BSM"     # combine, with zeros doing the masking
```

**No dispatch collective at all, and every matmul is a plain dense GEMM at full kernel
efficiency**, which is why toy implementations look deceptively good. And it does
`E / E_a` times the activated FLOPs:

| Model | `E` | `E_a` | Dense-masked FLOP multiplier |
|---|---|---|---|
| Mixtral 8x7B | 8 | 2 | **4x** |
| DeepSeek-V2-Lite | 64 | 6 | **10.7x** |
| Qwen3 30B-A3B | 128 | 8 | **16x** |
| DeepSeek-V3 | 256 | 8 | **32x** |

**Survivable for Mixtral, indefensible for anything fine-grained.** A 4x FLOP penalty
on a kernel running at 80% efficiency still beats a 1x FLOP count on a kernel running at
15%. A 32x penalty does not beat anything.

**2. One-hot dispatch at fixed capacity.** The GShard formulation: an einsum routes
tokens into a `[E, capacity, D]` buffer, so every expert matmul has a statically known
shape and the compiler is happy.

```
"BSM,BSEC -> EBCM"    # dispatch: scatter tokens into per-expert capacity slots
"EBCM,EMH -> EBCH"    # expert up-projection, static shapes
"EBCH,EHM -> EBCM"    # expert down-projection
"EBCM,BSEC -> BSM"    # combine: gather results back to token order
```

**You pay the padding and the dropping from the previous section, and you also pay for
the dispatch einsum itself, which is the cost nobody mentions.** That first einsum does
`2 * B * S * D * E * C` FLOPs, and with `C` proportional to `S`, it is *quadratic in
sequence length*. Its cost relative to the expert matmuls is `S / F`, so at 2048 tokens
against Mixtral's `F = 14336` it is 14% and at 32k tokens it is 2.3x the expert compute.
**One-hot dispatch does not scale to long context**, which is a good reason it is not
the default anywhere any more.

**3. Sort and grouped GEMM.** Sort the tokens by chosen expert, count how many landed in
each group, and do one ragged matmul over variable-sized groups.

```python
# MaxText's permute(), layers/moe.py
sorted_selected_experts = jnp.argsort(flatten_selected_experts)
sorted_inputs = _sort_activations(replicated_inputs_2d, sorted_selected_experts)
group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)
# then: gmm(sorted_inputs, expert_weights, group_sizes)
```

**Dropless, no padding, exactly the activated FLOP count.** The entire cost moves into
needing a grouped or ragged GEMM kernel that stays fast when the groups are uneven and
their sizes are runtime values. Plus a sort, which is cheap but not free, and a custom
backward pass for it: MaxText's `use_custom_sort_vjp` defaults to `true` because the
naive gradient of a sort is expensive.

**The calculation that decides between them, and it is the one a reader on a stack
without a ragged kernel actually has to do.** Sort-and-group wins when:

```
1 / η_ragged  <  (E / E_a) / η_dense
  =>  η_ragged  >  η_dense * E_a / E
```

With a dense GEMM achieving `η_dense = 0.8`:

| Model | `E_a / E` | Ragged efficiency needed to win |
|---|---|---|
| Mixtral 8x7B | 0.25 | 20% |
| DeepSeek-V2-Lite | 0.094 | 7.5% |
| Qwen3 30B-A3B | 0.0625 | 5% |
| DeepSeek-V3 | 0.031 | 2.5% |

**For a fine-grained model, an appallingly bad ragged kernel still wins**, and that is
the most useful number in this section. It also means a reader whose stack only offers
dense masked compute is not doing it wrong: they need to know what it costs, which for
Mixtral is 4x and is survivable, and for Qwen3 is 16x and is not.

## Which of the Three You Can Get on AMD in JAX

**This is the section that matters most and the one with the least prior art.** Here is
what we can establish by reading source, followed by what still has to be measured.

**Selecting the implementation in MaxText.** Three config fields, in the order they are
consulted:

| `sparse_matmul` | `megablox` | `use_tokamax_gmm` | Path taken |
|---|---|---|---|
| `false` | — | — | Dense masked, or one-hot capacity if `capacity_factor > 0` |
| `true` | `true` | `false` | Megablox Pallas grouped GEMM **(default)** |
| `true` | `false` | `false` | `jax.lax.ragged_dot` |
| `true` | — | `true` | Tokamax `ragged_dot`, `implementation="mosaic"` |

**And here is the problem: two of those three grouped-GEMM backends are TPU kernels.**

**Megablox, the default, is a Pallas TPU kernel.** Its module docstring says so
directly, and it imports `jax.experimental.pallas.tpu`, using `pltpu.PrefetchScalarGridSpec`
and `pltpu.VMEM`. On a non-TPU mesh, MaxText does not fall back to a GPU kernel: it runs
the Pallas kernel in *interpret mode*.

```python
# MaxText layers/moe.py, at the gmm() call site
megablox_interpret = self.mesh.devices.flat[0].platform != "tpu"
```

**Interpret mode is a reference interpreter, not a kernel.** It exists so Pallas kernels
can be debugged on CPU. There is no error, no warning, and no config validator
objecting: a GPU user who accepts the defaults gets the dropless path, semantically
correct, executing through an interpreter.

**Tokamax is the same story**, calling `tokamax.ragged_dot(..., implementation="mosaic")`,
where Mosaic is the TPU lowering path. Mosaic GPU exists but is NVIDIA-only.

**That leaves `jax.lax.ragged_dot`, selected by `megablox: false`, as the only
platform-general grouped-GEMM path** and therefore the one a JAX user on ROCm has to
use. It is a real JAX primitive with an XLA lowering rather than a TPU kernel, so it will
run. Whether it runs *fast* on ROCm is the open question, and it is the difference
between implementation 3 being available and being nominal.

> **Verified against:** MaxText at commit `9f9ac05` (`AI-Hypercomputer/maxtext`, `main`),
> read 4 August 2026. This is upstream MaxText, which carries in-tree ROCm support
> (`run_rocm.py`, the `nanoo_fp8` quantization value); we have not separately audited
> AMD's fork.

**The practical advice that follows, and it is the most actionable thing in this
chapter.** If you are running MaxText MoE on ROCm:

1. **Do not accept the default.** `megablox: true` with `sparse_matmul: true` selects
   an interpreted TPU kernel.
2. **Set `megablox: false`** to get `jax.lax.ragged_dot`, and check in a profile that
   the resulting kernel is a real GEMM rather than a pile of elementwise work.
3. **If that is slow, fall back to `sparse_matmul: false` with `capacity_factor: -1`**,
   dense masked compute, and compute your `E / E_a` penalty from the table above so you
   know what you are paying.
4. **Log the four numbers** from
   [the closing section](#the-four-numbers-to-log-for-every-moe-run), because none of
   this is visible otherwise.

### The Missing Mesh Axis

**Before the implementation comparison, a configuration bug, because it is larger than
most of the differences the comparison finds and it is invisible in a profile.**

MaxText's `base.yml` maps logical axes onto physical mesh axes, and one of those lines
governs where MoE activations live:

```yaml
['activation_batch_moe', ['data', 'fsdp', 'fsdp_transpose', 'expert']],
```

**Two upstream refactors dropped `'expert'` from that list.** Nothing breaks: on a
multi-axis mesh the activations still shard over `data` or `fsdp`. But on a single node
running expert parallelism alone, which is exactly the AMD recipe with
`ici_expert_parallelism: -1` and everything else at 1, `expert` is the *only* active axis.
**Remove it and the MoE activations are replicated on every device.**

We put the line back the way it was and measured both, on Mixtral 8x7B, eight MI300X,
4096-token sequences: **[measured]**

| Per-device batch | With `'expert'` | Without | |
|---|---|---|---|
| 1 | 174.7 TFLOP/s | 224.0 TFLOP/s | unsharded wins |
| 2 | — | 239.0 TFLOP/s | |
| 4 | 291.0 TFLOP/s | 255.2 TFLOP/s | sharded wins |
| 12 | **344.0 TFLOP/s** | **out of memory** | |

**The headline is the last row, and it is a memory result before it is a throughput
result.** Unsharded MoE activations are eight times larger per device, so the recipe's
batch of 12 does not fit, and the largest batch that does fit reaches 255 TFLOP/s. The
sharded configuration runs the intended batch and reaches 344. **Best against best, the
line is worth 1.35x.**

**At a batch where both fit, the gap is smaller and at batch 1 it inverts.** Sharding the
MoE activation over eight experts when there is one sequence per device leaves too little
work per expert to pay for the dispatch, and the replicated version is 1.28x faster. This
is the same tokens-per-device story as
[Chapter 6]({{ '/pages/6-training' | relative_url }}), arriving in an unexpected place.

**Two things to take from this.** The upstream commit that restores the line describes a
2.5x regression; we measure 1.14x at matched batch and 1.35x best-against-best on this
hardware, so **check the number on your own configuration rather than inheriting ours or
theirs**. And more usefully: **a logical-to-physical axis mapping that silently degrades
when a mesh axis is inactive is a failure mode you will not find by reading a profile**,
because the profile of the slow configuration looks entirely reasonable. It is only
obvious when you have the fast one to compare against.

> **Verified against:** the container this book pins, whose MaxText already carries the
> restored line at `src/maxtext/configs/base.yml:541`. **You are unlikely to hit this on
> `rocm/jax-training:maxtext-v26.5`**; we reproduced it by removing the line deliberately.
> Check yours before assuming, because the fix is recent and the refactors that caused it
> are upstream.

### The Implementations, Measured

**We ran all five paths on two models, eight MI300X, expert parallelism only, same tokens,
same everything except the dispatch.** Mixtral 8x7B has coarse experts
(`E / E_a = 4`, `F_moe = 14336`); Qwen3 30B-A3B has fine ones
(`E / E_a = 16`, `F_moe = 768`). **[measured]**

| Implementation | Config | Mixtral | Qwen3 |
|---|---|---|---|
| One-hot capacity | `sparse_matmul: false`, `capacity_factor: 1.25` | **254.4 TFLOP/s** | **66.9 TFLOP/s** |
| Dense masked | `sparse_matmul: false`, `capacity_factor: -1` | 88.1, 2.9x slower | 27.0, 2.5x slower |
| `ragged_dot` | `sparse_matmul: true`, `megablox: false` | 21.5, 11.8x slower | 11.9, 5.6x slower |
| megablox | `sparse_matmul: true`, `megablox: true` | **fails to compile** | **fails to compile** |
| tokamax | `use_tokamax_gmm: true` | **refuses to run** | **refuses to run** |

**One-hot capacity dispatch wins on both models, and it is not close.** The three-way
comparison this chapter set up turns out to have a fourth answer that beats all three of
them, and it is the one AMD's own recipes already use.

**Three of the four things this section predicted are wrong, and the fourth is more wrong
than predicted.**

**megablox does not silently interpret. It fails, loudly, at compile time.**

```
File "jax/_src/pallas/core.py", line 161, in get_lowering_rule
ValueError: Compiler params for platform tpu cannot be used for gpu lowering.
```

**This is much better news than the interpret-mode reading above suggested**, and the
section's warning needs softening rather than strengthening: a GPU user who accepts
`megablox: true` does not get a silently interpreted kernel producing correct answers
slowly. They get a stack trace before the first step. **The failure is loud, immediate and
unambiguous**, which is the failure mode you want.

**tokamax refuses too, and just as clearly:**

```
File "tokamax/_src/ops/op.py", line 197
NotImplementedError: Not supported on AMD Instinct MI300X.
```

**So `jax.lax.ragged_dot` really is the only grouped-GEMM path that runs**, exactly as this
section argued. **And it is unusable.** 60.7 seconds against 5.1 for the one-hot path is
not a tuning gap, it is a different order of magnitude, and no amount of `E / E_a` saved
recovers a factor of twelve.

### Working the Win Condition

**Substituting the measured numbers into the inequality from
[Three Ways to Implement an Expert Layer](#three-ways-to-implement-an-expert-layer).**
MaxText reports *useful* model FLOP/s, so dense masked's figure already has its
`E / E_a` waste divided out; the step-time ratio and the throughput ratio agree at 2.89,
which is how you can tell. Recovering the kernel efficiencies against the 990 TFLOP/s
roofline [Chapter 3]({{ '/pages/3-profiling' | relative_url }}) measures:

```
eta_dense  = 88.1 * (E / E_a) / 990 = 88.1 * 4 / 990 = 35.6%
eta_ragged = 21.5 / 990                              =  2.2%
```

**For Mixtral, `E / E_a = 4`, and the condition fails by a factor of four:**

```
eta_ragged > eta_dense * E_a / E
      2.2% > 35.6% * 0.25 = 8.9%     FALSE
```

**For Qwen3, `E / E_a = 16`, which should be `ragged_dot`'s best case. It still fails:**

```
eta_dense  = 27.0 * 16 / 990 = 43.6%
eta_ragged = 11.9 / 990      =  1.2%
      1.2% > 43.6% / 16 = 2.7%       FALSE
```

**The prediction this section made was that `ragged_dot` wins easily for fine-grained
models even at poor efficiency. It does not win at all.** Sixteen-way sparsity closes the
gap from 4x to 2.3x and no further, because `jax.lax.ragged_dot` gets worse as the experts
get narrower at almost the same rate the sparsity argument helps. **The derivation assumed
6% efficiency for the ragged path; the measurement is 1.2% to 2.2%.**

**And notice what dense masked does on the fine-grained model, because it is the
counterintuitive part.** Qwen3 runs sixteen times the expert FLOPs in only 2.5 times the
step time, which means its kernel efficiency is **higher** than Mixtral's: 43.6% against
35.6%. **Making the experts narrow makes every per-expert GEMM worse, and the dense path is
the one that does not have per-expert GEMMs.** Fine-grained MoE punishes exactly the
implementations that try to exploit the sparsity.

### What AMD Ships, And Why

**The Primus MI300X recipes for both Mixtral and DeepSeek-V2-Lite set
`sparse_matmul: false` with a finite `capacity_factor`**, which contradicts upstream
MaxText's defaults and, before this measurement, looked like a conservative choice. It is
not conservative. **It is the only configuration of the five that is both fast and
runnable**, and it beats the nearest alternative by 2.9x.

**So the advice earlier in this section needs reordering, and here it is corrected:**

1. **Set `sparse_matmul: false` and a finite `capacity_factor`.** This is the fast path on
   ROCm today, and it is what AMD's own recipes use.
2. **Do not set `megablox: true`.** It will not run. If you inherited a config that sets
   it, you already know, because nothing started.
3. **Do not reach for `ragged_dot` on the strength of its `E / E_a` argument.** The
   argument is sound and the implementation is twelve times too slow for it to matter.
4. **Re-check all of this when you upgrade.** Every row of that table is a statement about
   one container, and the grouped-GEMM situation is the fastest-moving thing in this
   chapter.

### AITER Has No Grouped MoE GEMM

**The comparison this section wanted to make against AMD's own kernels cannot be made,
and the reason is the most important fact in the chapter.**

[`ROCm/jax-aiter`](https://github.com/ROCm/jax-aiter) bridges AITER into JAX over XLA FFI,
needs no PyTorch at runtime, and requires Python 3.12 with ROCm 7.2 or newer, so it would
install against this book's container. Its published operator list, read on
**5 August 2026**, is:

> FP4 GEMM (training and low-level), MXFP4 quantizer and cast, **BF16 GEMM**, **flash
> attention** and its variable-length form, RMSNorm, fused add-and-RMSNorm, and
> SiLU-and-multiply.

**There is no grouped GEMM, no ragged GEMM, and no MoE dispatch operator in that list.**
AITER's fast kernels cover attention and dense matmul, which are the parts of a transformer
that were never the MoE problem. **So a JAX user on ROCm has no vendor-optimised expert
GEMM to fall back to**, and the 254 TFLOP/s the one-hot path reaches is not a floor with a
better kernel above it; it is the ceiling.

**That reframes the whole chapter.** The question is not "which grouped-GEMM implementation
should I pick", because the good ones are TPU kernels and the portable one is twelve times
too slow. **The question is how to make dense, batched matmuls do MoE work**, and one-hot
capacity dispatch is the answer that is actually available: it turns routing into a
gather, a dense GEMM per expert slot, and a scatter, all of which hipBLASLt is good at.

> **Verified against:** `rocm/jax-training:maxtext-v26.5`, MaxText `release/v26.5` at
> `a7c6c7e5`, `tokamax` 0.0.12, on 8x MI300X, **5 August 2026**. `jax-aiter` is not
> published on PyPI and was read from its repository rather than installed. **Pin the date
> on all of this**; it is the fastest-moving material in the book.

<!-- BLOCKED: MEASUREMENT 2 as originally scoped is not possible and the section above
     says why: AITER exposes no grouped MoE GEMM through jax-aiter, so there is no
     published AMD grouped-GEMM figure to state a ratio against. What could still be
     done, and would be worth doing:
       - Compare our one-hot capacity TFLOP/s against jax-aiter's published dense BF16
         GEMM figures, to bound how much of the gap is dispatch rather than GEMM. That
         needs jax-aiter built from source, which we have not done because it is a
         source build that would perturb the pinned container.
       - Revisit if AITER ever ships a grouped GEMM. Check the operator table linked
         above rather than assuming. -->

## All-to-All Dispatch and Combine

**Once experts live on different devices, every token has to travel to its experts and
its results have to come back. Those two collectives define MoE performance.**

The pattern is an all-to-all in each direction, per MoE layer:

- **Dispatch:** each device sends each of its tokens to the `E_a` devices holding the
  experts it chose. Bytes out per device are `w * B_tok * E_a * D * (|Ex|-1)/|Ex|`.
- **Combine:** the same volume in reverse, carrying the expert outputs back.

**When `E_a` exceeds the expert-axis size, the ragged all-to-all saturates**, because a
token bound for three experts that all live on the same device is one transfer, not
three. The general factor is `min(E_a / |Ex|, 1)` applied to the per-peer volume, and the
practical consequence is that dispatch cost stops growing once `|Ex| < E_a`.

**Here is the good AMD result, and it should not be buried under the bad one.**
**All-to-all is the collective a switchless full mesh is best suited to.** Every device
has a direct link to every peer, so all seven links carry one peer's share concurrently,
with no switch to contend for and no multi-hop forwarding. A ring schedule lights a
fraction of the links at any instant; an 8-way all-to-all can in principle light all
seven at once, which puts the whole 448 GB/s of per-GPU egress in play.

**Substituting real numbers, for DeepSeek-V2-Lite shapes** (`D = 2048`, `E_a = 6`,
`F_moe = 1408`) at 4096 tokens per device with `|Ex| = 8`:

- Dispatch bytes out per device: `2 * 4096 * 6 * 2048 * (7/8) = 88 MB`.
- At `β_g = 320 GB/s`: **0.275 ms**, and the same again for the combine, so 0.55 ms per
  layer.
- Expert compute per device per layer:
  `18 * 4096 * 2048 * 1408 * 6 / 1307.4e12 = 0.98 ms`.

**So dispatch and combine together are about 36% of the expert compute, inside the
node.** That is a real cost and it is affordable. **Inside a baseboard, MoE dispatch
should be cheap**, and it is worth saying clearly because the reader expects the
opposite.

**Now the cliff, which is the same fact from the other side.** Run the same arithmetic
with the expert axis crossing a host boundary, at `β_net = 50 GB/s`:

- Dispatch: `88e6 / 50e9 = 1.76 ms`, and the same for combine: **3.5 ms per layer**.
- Against 0.98 ms of expert compute, that is **3.6x the compute**, twice per layer,
  every layer.

**An expert axis that crosses the baseboard trades a 128 GB/s direct link for a share of
node egress, and it turns a 36% overhead into a 360% one.** That is the central
placement question of this chapter, and it has a one-line answer: **keep `Ex` inside the
node and spend the slow axis on something that tolerates it**, which per
[Chapter 6]({{ '/pages/6-training' | relative_url }}) means data parallelism.

**Two honest caveats.** The inter-node arithmetic is **[analytical]**; we have no
cluster. And the intra-node figure assumes RCCL actually schedules an 8-way all-to-all
across all seven links rather than decomposing it into rings. The topology says it
should; the library decides.
[Chapter 4]({{ '/pages/4-sharding' | relative_url }})'s pending sweep is what settles it,
and AMD's published per-GPU RCCL figure of 310-330 GB/s says the answer is probably yes.

## Expert Parallelism

**1. What it shards.**

```
W_expert[E_Ex, D, F]
```

The expert dimension. Each device holds `E / |Ex|` complete experts, and tokens travel
to them.

**2. Why do this, why not do this.**

**Open with memory, which is the opposite of how expert parallelism is usually
introduced and the more honest motivation.** An MoE has `E` times the MLP parameters at
`E_a / E` of the MLP FLOPs, and the optimizer state scales with the *total* parameter
count, not the activated one. DeepSeek-V2-Lite is a 16B-parameter model, which sounds
small, and at 16 bytes per parameter that is **256 GB of state: more than one MI300X
holds.** A model whose activated size is 2.4B does not fit on a 192 GB device.

**So you shard by expert because the weights do not fit, and only then discover you have
bought an all-to-all.** That ordering matters for how you reason about the trade: the
communication is not a cost you chose in exchange for speed, it is the price of the
model existing on your hardware at all.

Do not use it when FSDP alone gets the weights to fit, because FSDP's collectives
overlap and expert parallelism's do not, and do not push `|Ex|` past a node boundary
unless you have exhausted every alternative.

**3. The algorithm.**

Per MoE layer:

1. Compute the router locally on your own tokens. *(no communication)*
2. All-to-all dispatch: send each token to the devices holding its chosen experts.
   **(on critical path)**
3. Run the local experts over whatever arrived. Shapes are ragged.
4. All-to-all combine: send outputs back to the devices that own the tokens.
   **(on critical path)**
5. Weighted-sum the `E_a` expert outputs per token, locally.

**This is the canonical case for `shard_map` over automatic partitioning**, per
[Chapter 4]({{ '/pages/4-sharding' | relative_url }}). The dispatch is a data-dependent
all-to-all over a ragged token set, GSPMD has no way to infer it from shardings, and
every serious implementation writes it by hand.

**4. The roofline.**

From the section above: dispatch and combine cost
`2 * w * B_tok * E_a * D * (|Ex|-1)/|Ex| / β_g` per layer, and expert compute is
`18 * B_tok * D * F_moe * E_a / C`. Compute-bound means:

```
18 * F_moe / C  >  4 * w * (|Ex|-1) / (|Ex| * β_g)
  =>  F_moe  >  2 * w * C * (|Ex|-1) / (9 * |Ex| * β_g)
```

Substituting MI300X constants at `|Ex| = 8`:

```
F_moe > 2 * 2 * 1307.4e12 * (7/8) / (9 * 320e9) = 1589
```

| Model | `F_moe` | Compute-bound at 8-way EP? |
|---|---|---|
| Mixtral 8x7B | 14336 | Comfortably, 9x margin |
| DeepSeek-V3 | 2048 | Yes, 1.3x margin |
| DeepSeek-V2-Lite | 1408 | **No, marginally under** |
| Qwen3 30B-A3B | 768 | **No, 2x under** |

**Fine-grained models are communication-bound on expert dispatch, and this is the
structural reason fine-grained MoE is hard.** The whole point of fine-grained experts is
to make each one narrow, and `F_moe` is precisely the quantity the roofline needs to be
large. Note that this is not a fixable inefficiency: it is what the architecture asks
for.

The levers, in order of how much they help: keep `Ex` intra-node so `β_g` is 320 rather
than 50; overlap dispatch with the previous layer's compute, which is what MaxText's
`num_moe_token_chunks` and ring-of-experts path exist for; and reduce `|Ex|` by combining
expert parallelism with FSDP so that fewer devices participate in the all-to-all.

**5. Predict, then measure.**

**Eight MI300X, one-hot capacity dispatch, `|Ex|` moved from 2 to 8 with FSDP taking up
whatever slack is left, so the device count is fixed at 8 throughout.** That is the real
decision: not "how much expert parallelism", but "how much of the node do I spend on it
instead of on FSDP". **[measured]**

**Mixtral 8x7B, `F_moe = 14336`, nine times the threshold:**

| `\|Ex\|` | FSDP | Step time | TFLOP/s per device | Collectives, share of device time |
|---|---|---|---|---|
| 2 | 4 | 7.66 s | 170.6 | 45.0% |
| 4 | 2 | 6.25 s | 209.0 | 35.8% |
| 8 | 1 | **5.14 s** | **253.9** | **19.0%** |

**Qwen3 30B-A3B, `F_moe = 768`, half the threshold:**

| `\|Ex\|` | FSDP | Step time | TFLOP/s per device | Collectives, share of device time |
|---|---|---|---|---|
| 2 | 4 | 3.10 s | 27.3 | 71.5% |
| 4 | 2 | 2.17 s | 39.1 | 65.7% |
| 8 | 1 | **1.31 s** | **64.6** | **45.8%** |

**The threshold predicts the right thing.** At 8-way expert parallelism the coarse-grained
model spends 19% of its device time in collectives and the fine-grained one spends 46%,
which is the difference between comfortable and dispatch-bound. A nineteen-fold difference
in `F_moe` produces a 2.4-fold difference in communication share, and the model that is
below the threshold is the one that is communication-dominated. **`F_moe > 1589` is a
usable rule.**

**The 36% prediction lands in the right place too, for the model it was derived for.** The
worked example above computes dispatch and combine at 36% of expert compute for a
DeepSeek-V2-Lite-shaped layer with `F_moe = 1408`. Qwen3, at `F_moe = 768`, is about half
that width and measures 46% of *total* device time in collectives, which is the same
statement seen through a wider denominator. **Nothing here contradicts the arithmetic.**

### But The Lever Points The Other Way

**One piece of the advice above is wrong and the table says so plainly.** The levers
listed at the end of part 4 include "reduce `|Ex|` by combining expert parallelism with
FSDP so that fewer devices participate in the all-to-all". **Do not do that.** Every row
of both tables gets worse as `|Ex|` falls: Mixtral loses 1.49x going from 8-way to 2-way,
Qwen3 loses 2.37x, and the collective share *rises* in both cases rather than falling.

**The reason is that the all-to-all is not the expensive collective.** On a fixed
eight-device node, the devices you take away from the expert axis go to FSDP, and FSDP's
job is to all-gather the expert weights, which for an MoE model are almost all of the
parameters. **Trading a dispatch of activations for a gather of experts is a bad trade,
and it gets worse the more of the model lives in the experts.**

**So the corrected lever list, in order:**

1. **Keep `Ex` intra-node**, which is the one piece of the original advice that holds
   and is the most important of the three.
2. **Then make `|Ex|` as large as the node allows**, rather than as small as you can get
   away with. On eight devices with an MoE model, `ici_expert_parallelism: -1` and nothing
   else is the configuration to beat, which is exactly what AMD's recipes ship.
3. **Overlap what remains.** Both models leave most of their collective time exposed:
   only 11.5% of Mixtral's and 10.2% of Qwen3's has compute running underneath it at
   8-way. That is the largest remaining gap in both, and
   [Chapter 4]({{ '/pages/4-sharding' | relative_url }})'s combine-threshold finding is
   the first thing to check.

**What this does not tell you is what happens across hosts**, which is where the 360%
figure above comes from and where pipelining the dispatch actually matters. That remains
**[analytical]**: we have one node.

**How expert parallelism composes with everything else**, which is where most real MoE
performance is won or lost:

| Axis | Strategy | Placement |
|---|---|---|
| `Ex` | Expert parallel | **Intra-node, always.** 8-way at most |
| `Y` | Tensor parallel | Intra-node, competing with `Ex` for the same eight GPUs |
| `X` | Data parallel or FSDP | Across hosts |
| `Z` | Pipeline parallel | Across hosts, if the batch supports the bubble |

**`Ex` and `Y` compete for the same scarce resource**, which is the eight GPUs on the
baseboard, and `|Ex| * |Y| <= 8` is the constraint that makes sparse-model configuration
harder than dense. Expert parallelism usually wins that argument, because it is solving
a memory problem that tensor parallelism only partly addresses.

**One pointer back to [Chapter 5]({{ '/pages/5-transformers' | relative_url }}):** the
dispatch tensors are large, `B_tok * E_a * D` per layer, so remat policy interacts with
routing. Rematerializing an expert layer means re-running the router and the dispatch,
including the all-to-all, which is usually the wrong trade. Check that your remat policy
saves the dispatch output.

## Anatomy of Three Real Models

**Three models, chosen because all three are in MaxText's model configurations and are
therefore models we can actually run and measure.**

| | Mixtral 8x7B | Qwen3 30B-A3B | DeepSeek-V3 |
|---|---|---|---|
| Total parameters | 46.7B | 30.5B | 671B |
| Activated | 12.9B | 3.3B | 37B |
| `L` | 32 | 48 | 61 |
| `D` | 4096 | 2048 | 7168 |
| `E` routed | 8 | 128 | 256 |
| `E_a` | 2 | 8 | 8 |
| Shared experts | 0 | 0 | 1 |
| `F_moe` per expert | 14336 | 768 | 2048 |
| Sparsity `E/E_a` | 4 | 16 | 32 |
| Attention | GQA | GQA | MLA |
| MI300X ridge point (tokens) | 988 | 3952 | 7904 |
| Dense-masked penalty | 4x | 16x | 32x |
| 8-way EP compute-bound? | yes | no | yes |

> Expert configurations from MaxText `configs/models/`, read at commit `9f9ac05`.

**Mixtral 8x7B is the simple case and the one to develop against.** Eight fat experts,
two active, no shared expert, ordinary GQA attention. Its `F_moe = 14336` makes it
comfortably compute-bound on dispatch, and its 4x dense-masked penalty means even the
worst implementation choice is survivable. **If a technique does not work on Mixtral it
will not work anywhere.**

**Qwen3 30B-A3B is the fine-grained case and the hard one.** 128 experts of width 768,
eight active. Every systems problem in this chapter is worse: 16x dense-masked penalty,
2x under the dispatch roofline at 8-way expert parallelism, and a ridge point of nearly
4000 tokens per device. It is also 3.3B activated parameters, so it is *fast* when it
works, which is the whole appeal.

**DeepSeek-V3 is the elaborate one, and the interesting thing is which parts are hard.**
256 experts plus one shared expert that runs for every token, MLA instead of GQA, a
learnable routing bias instead of an auxiliary loss, and 61 layers of which the first
three are dense. Its `F_moe = 2048` is large enough to clear the dispatch roofline, so
despite being 32x sparse it is *not* dispatch-bound at 8-way expert parallelism: the
fine-grained model with a tenth of the parameters is the harder one to run well.

**The shared expert is worth a sentence** because it does more than it looks. Running
one expert for every token adds dense FLOPs, which raises arithmetic intensity and pulls
the ridge point down, and it gives the router permission to specialise harder because
there is always a generalist path. It is a quality feature that happens to improve the
systems picture.

**MLA is accounted for in
[Chapter 5]({{ '/pages/5-transformers' | relative_url }})** and its serving consequences
are in [Chapter 11]({{ '/pages/11-inference' | relative_url }}). It is an attention
mechanism that happens to appear in a sparse model, not an MoE technique, and this
chapter does not re-derive it.

## The Four Numbers to Log for Every MoE Run

**None of these is on by default, all four are nearly free, and a reader who logs them
can diagnose their own MoE without this chapter.** That is the correct ambition for a
closing section.

**1. Tokens per expert, as a histogram, not a maximum.** One `bincount` on the routing
indices, logged every N steps.

```python
counts = jnp.bincount(expert_indices.reshape(-1), length=num_experts)
```

The maximum tells you the step time; the shape of the distribution tells you whether
the auxiliary loss is working, whether a few experts are dead, and whether the imbalance
is drifting. Maps directly onto [Load Imbalance](#load-imbalance).

**2. The dropped-token fraction, or the ragged-shape distribution if dropless.** With
fixed capacity, the fraction of tokens whose chosen expert was full. With a dropless
grouped GEMM, the distribution of group sizes, which is the same information in a
different currency. **This one has no proxy**: nothing else you can measure tells you
that tokens are being silently discarded, and MFU moves in the wrong direction when it
happens.

**3. The achieved efficiency of the expert GEMM against a dense GEMM of equivalent
size.** Time a dense matmul with the same total FLOP count and take the ratio. This
number is what the win condition in
[Three Ways to Implement an Expert Layer](#three-ways-to-implement-an-expert-layer)
needs, and on AMD in JAX it is the number most likely to be surprising, per
[the section above](#which-of-the-three-you-can-get-on-amd-in-jax). Source in a profile:
Kernel Stats, comparing the expert kernel's duration against its FLOP count.

**4. The all-to-all share of step time.** From the trace: total duration of dispatch and
combine collectives, over step time. Compare against the 36% intra-node prediction from
[All-to-All Dispatch and Combine](#all-to-all-dispatch-and-combine). If it is much
higher, check whether `Ex` crossed a host boundary, which is the single most expensive
configuration mistake available to you.

## Worked Problems

**Question 1:** Your MoE run logs a token histogram whose busiest expert sees 1180
tokens and whose mean is 384. Expert compute is 60% of step time. What is imbalance
costing you, and what is the best case if you fix it?

{% details Click here for the answer. %}

**The imbalance factor is `1180 / 384 = 3.07`.** The step waits for the busiest expert,
so the expert-compute portion of the step is taking about 3.07x as long as a perfectly
balanced version would.

If expert compute is 60% of a 100 ms step, that is 60 ms, of which a balanced version
would need `60 / 3.07 = 19.5 ms`. **Fixing imbalance perfectly would take the step from
100 ms to 59.5 ms, a 1.68x speedup.**

**Then the caveats that make this an estimate rather than a calculation.** Perfect
balance is not achievable, and pushing the auxiliary loss weight up far enough to get
close costs model quality. A realistic target is an imbalance factor of 1.2 to 1.5,
which recovers most but not all of that 40 ms. And if the implementation is dense masked
compute, imbalance costs you *nothing*, because every expert already processes every
token: check which implementation you are on before spending a week on the router.

**First thing to check:** whether `load_balance_loss_weight` is still at its default of
`0.0`.

{% enddetails %}

**Question 2:** You are running Qwen3 30B-A3B on 16 GPUs across two nodes with
`|Ex| = 16`. Someone suggests `|Ex| = 8` with 2-way data parallelism instead. Which is
faster, and by how much?

{% details Click here for the answer. %}

**Take the dispatch cost per layer in each configuration.** Qwen3 30B-A3B has
`D = 2048`, `E_a = 8`, `F_moe = 768`. Say 4096 tokens per device.

**Dispatch bytes out per device**, which barely changes between the two:

```
|Ex| = 16:  2 * 4096 * 8 * 2048 * (15/16) = 126 MB
|Ex| = 8:   2 * 4096 * 8 * 2048 * (7/8)   = 117 MB
```

**But the bandwidth changes by a factor of 6.4.** With `|Ex| = 16` the all-to-all spans
both nodes, so it is limited by `β_net = 50 GB/s`; with `|Ex| = 8` it is entirely
intra-node at `β_g = 320 GB/s`:

```
|Ex| = 16:  2 * 126e6 / 50e9  = 5.0 ms per layer (dispatch + combine)
|Ex| = 8:   2 * 117e6 / 320e9 = 0.73 ms per layer
```

**Expert compute per device per layer** is the same in both:

```
18 * 4096 * 2048 * 768 * 8 / 1307.4e12 = 0.71 ms
```

**So the 16-way configuration spends 5.0 ms communicating for every 0.71 ms of expert
compute, and the 8-way spends 0.73 ms.** The MoE layers go from being 88%
communication to about 51%, and the whole step improves by roughly a factor of 3.

**The 8-way configuration also has to pay a gradient all-reduce across nodes** for the
2-way data parallelism, which the 16-way expert-parallel version does not. That is one
collective per step, overlappable, against `L` all-to-alls per step on the critical
path. It is not close.

Both figures are **[analytical]**, and the inter-node one especially.

{% enddetails %}

**Question 3:** Your stack has no working ragged GEMM, so you are choosing between dense
masked compute and one-hot dispatch at `capacity_factor = 1.25`. Your model is Mixtral
8x7B at 2048-token sequences. Which is cheaper, and does the answer change at 32k
context?

{% details Click here for the answer. %}

**Dense masked compute** does `E / E_a = 4x` the activated expert FLOPs, at full dense
GEMM efficiency, and needs no dispatch einsum at all. Call the activated expert cost 1
unit: **dense masked costs 4 units.**

**One-hot dispatch at fixed capacity** does `capacity_factor = 1.25x` the activated
expert FLOPs, plus the dispatch and combine einsums, whose cost relative to the expert
matmuls is `S / F`:

```
At S = 2048, F = 14336:   1.25 + 2 * (2048/14336) * 1.25 = 1.25 + 0.36 = 1.61 units
At S = 32768, F = 14336:  1.25 + 2 * (32768/14336) * 1.25 = 1.25 + 5.71 = 6.96 units
```

**At 2048 tokens, one-hot dispatch wins by 2.5x.** At 32k it *loses* by 1.7x, because
the dispatch einsum is quadratic in sequence length and the dense-masked penalty is not.

**Two things this calculation deliberately ignores, and both favour dense masked
compute.** The one-hot path drops tokens, and the FLOPs it saves by dropping are not a
saving. And its expert matmuls, while statically shaped, are `E` separate GEMMs of
capacity `C` rather than one large one, so they are smaller and may achieve lower
efficiency than the dense-masked version's full-batch GEMMs.

**The honest summary: at ordinary context lengths and low sparsity, one-hot capacity
wins; at long context or high sparsity, dense masked wins; and both lose badly to a
working ragged GEMM.** Which is the argument for
[fixing the kernel situation](#which-of-the-three-you-can-get-on-amd-in-jax) rather than
optimising the consolation prizes.

{% enddetails %}

## References

**Architecture and routing.**

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
  (Shazeer et al., 2017). The origin of the gated MoE layer and the auxiliary
  load-balancing loss.
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
  (Lepikhin et al., 2020). The one-hot dispatch formulation and the capacity-factor
  mechanism, implementation 2 above.
- [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021). Top-1
  routing, and the load-balance loss in the form MaxText implements.
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)
  (Zoph et al., 2022). Router numerics, including the router z-loss and the fp32
  recommendation.
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
  (Zhou et al., 2022). Expert-choice routing, the alternative that makes imbalance
  structurally impossible.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) (Dai et al., 2024). Fine-grained
  experts and shared experts, the two modifiers used by most current models.
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)
  (Wang et al., 2024). The learnable routing bias that DeepSeek-V3 uses, exposed in
  MaxText as `routed_bias`.

**Implementation and kernels.**

- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
  (Gale et al., 2022). Dropless MoE via block-sparse grouped GEMM, implementation 3
  above, and the origin of the `megablox` name in MaxText.
- [MaxText MoE configuration reference](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/reference/core_concepts/moe_configuration.md)
  (AI-Hypercomputer). The decision tree between dropless and dropping paths, and the
  meaning of `sparse_matmul`, `megablox` and `capacity_factor`.
- [MaxText layers/moe.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/layers/moe.py)
  (AI-Hypercomputer). The three dispatch implementations as code, the einsum equations
  quoted above, and the `interpret` flag that decides what a GPU user actually runs.
  Read at commit `9f9ac05`.
- [AITER](https://github.com/ROCm/aiter) (AMD). Where AMD's fast MoE kernels live,
  including fused routing and block-scaled grouped GEMM. PyTorch-facing.
- [ROCm/jax-aiter](https://github.com/ROCm/jax-aiter) (AMD). The JAX bridge into AITER
  over XLA FFI, and the place to check whether a grouped GEMM has become reachable from
  JAX.
- [jax.lax.ragged_dot](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_dot.html)
  (JAX). The platform-general grouped-GEMM primitive, and the only one of the three
  MaxText backends that is not TPU-specific.

**Models used in the anatomy table.**

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Mistral, 2024).
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (Alibaba, 2025).
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek, 2024).
