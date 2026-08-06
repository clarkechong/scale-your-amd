"""A Llama-3-8B-shaped transformer block, sharded five ways.

The workhorse behind Chapters 3, 4 and 6. One block, real shapes
(`D=4096, F=14336, N=32, K=8, H=128`), a forward and backward pass, and a
`--strategy` flag that switches which collective the step is paying for.

    python -m bench.transformer_block --strategy fsdp --tokens 2048 --trace
    python -m bench.transformer_block --strategy dp --tokens 512 2048 8192
    python -m bench.transformer_block --strategy tp --no-annotate --trace

Every sub-block carries a `jax.named_scope`, so Kernel Stats rows say which
part of the block they came from rather than just `dot_general`. Step markers
are on by default and `--no-annotate` turns them off, which is how Chapter 3
shows the broken Overview page before fixing it.
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._harness import MI300X, Run, base_parser, configure_environment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))


@dataclass(frozen=True)
class BlockConfig:
    """Llama 3 8B's per-layer shape."""

    d_model: int = 4096
    d_ff: int = 14336
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def params(self) -> int:
        d, f = self.d_model, self.d_ff
        return 2 * d * self.q_dim + 2 * d * self.kv_dim + 3 * d * f

    def flops_per_token(self, seq_len: int) -> float:
        """Forward FLOPs for one token, causal attention included."""
        d, f = self.d_model, self.d_ff
        projections = 2 * (d * self.q_dim + 2 * d * self.kv_dim + self.q_dim * d)
        mlp = 2 * 3 * d * f
        # Scores and the value-weighted sum, halved for causality.
        attention = 2 * 2 * self.q_dim * seq_len / 2
        return projections + mlp + attention


# Which mesh axis each strategy shards over, and what it does to the step.
STRATEGIES = ("dp", "fsdp", "tp", "cp", "pp")


def build_parser():
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--strategy", default="fsdp", choices=STRATEGIES)
    p.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 2048, 8192],
        help="per-device tokens; Chapter 6's DP threshold is 2724",
    )
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument(
        "--layers",
        type=int,
        default=4,
        help="a single layer gives a collective nothing to overlap with",
    )
    p.add_argument("--devices", type=int, default=8)
    p.add_argument(
        "--no-annotate",
        action="store_true",
        help="drop StepTraceAnnotation, which breaks the Overview page on purpose",
    )
    p.add_argument(
        "--latency-hiding",
        default=None,
        choices=["true", "false"],
        help="override the container's latency-hiding scheduler setting",
    )
    p.add_argument(
        "--microbatches",
        type=int,
        default=8,
        help="pipeline microbatches, for --strategy pp",
    )
    p.add_argument(
        "--xla",
        action="append",
        default=[],
        metavar="FLAG=VALUE",
        help="override any XLA flag, repeatable",
    )
    return p


def init_params(cfg: BlockConfig, layers: int, dtype: Any, key: Any) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    def glorot(k, shape):
        return jax.random.normal(k, shape, dtype) * math.sqrt(1.0 / shape[0])

    keys = jax.random.split(key, 7 * layers)
    params = []
    for i in range(layers):
        k = keys[7 * i : 7 * (i + 1)]
        params.append(
            {
                "wq": glorot(k[0], (cfg.d_model, cfg.q_dim)),
                "wk": glorot(k[1], (cfg.d_model, cfg.kv_dim)),
                "wv": glorot(k[2], (cfg.d_model, cfg.kv_dim)),
                "wo": glorot(k[3], (cfg.q_dim, cfg.d_model)),
                "wg": glorot(k[4], (cfg.d_model, cfg.d_ff)),
                "wu": glorot(k[5], (cfg.d_model, cfg.d_ff)),
                "wd": glorot(k[6], (cfg.d_ff, cfg.d_model)),
            }
        )
    return {"layers": params}


def rms_norm(x, eps: float = 1e-6):
    import jax.numpy as jnp

    return x * (1.0 / jnp.sqrt(jnp.mean(x.astype(jnp.float32) ** 2, -1, keepdims=True) + eps)).astype(
        x.dtype
    )


def block_forward(params, x, cfg: BlockConfig):
    """One transformer block. Every sub-block is named so the trace is readable."""
    import jax
    import jax.numpy as jnp

    b, s, _ = x.shape

    with jax.named_scope("attention_norm"):
        h = rms_norm(x)

    with jax.named_scope("qkv_proj"):
        q = (h @ params["wq"]).reshape(b, s, cfg.n_heads, cfg.head_dim)
        k = (h @ params["wk"]).reshape(b, s, cfg.n_kv_heads, cfg.head_dim)
        v = (h @ params["wv"]).reshape(b, s, cfg.n_kv_heads, cfg.head_dim)

    with jax.named_scope("attention"):
        # Grouped-query attention: repeat each KV head across its query group.
        groups = cfg.n_heads // cfg.n_kv_heads
        k = jnp.repeat(k, groups, axis=2)
        v = jnp.repeat(v, groups, axis=2)
        attn = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        attn = attn.reshape(b, s, cfg.q_dim)

    with jax.named_scope("out_proj"):
        x = x + attn @ params["wo"]

    with jax.named_scope("mlp_norm"):
        h = rms_norm(x)

    with jax.named_scope("mlp"):
        gate = jax.nn.silu(h @ params["wg"])
        up = h @ params["wu"]
        x = x + (gate * up) @ params["wd"]

    return x


def loss_fn(params, x, cfg: BlockConfig):
    import jax
    import jax.numpy as jnp

    h = x
    for i, layer in enumerate(params["layers"]):
        with jax.named_scope(f"layer_{i}"):
            h = block_forward(layer, h, cfg)
    return jnp.mean(h.astype(jnp.float32) ** 2)


def adam_update(params, grads, state, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.95):
    """A real optimizer, so the trace has the third phase of a training step in it.

    Chapter 3 wants forward, backward and optimizer visible in the Trace Viewer,
    and an update is also where the FSDP reduce-scatter lands, so leaving it out
    would understate the communication.
    """
    import jax
    import jax.numpy as jnp

    with jax.named_scope("optimizer"):
        def one(p, g, m, v):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g.astype(jnp.float32) ** 2)
            return (p - lr * (m / (jnp.sqrt(v) + 1e-8)).astype(p.dtype), m, v)

        new_p, new_m, new_v = [], [], []
        for p_l, g_l, m_l, v_l in zip(params["layers"], grads["layers"], state["m"], state["v"]):
            np_, nm_, nv_ = {}, {}, {}
            for k in p_l:
                np_[k], nm_[k], nv_[k] = one(p_l[k], g_l[k], m_l[k], v_l[k])
            new_p.append(np_)
            new_m.append(nm_)
            new_v.append(nv_)
        return {"layers": new_p}, {"m": new_m, "v": new_v}


def shardings(strategy: str, mesh, cfg: BlockConfig):
    """Where the batch lives and where the weights live, per strategy.

    These are GSPMD constraints rather than explicit collectives: we say what is
    sharded and let the partitioner insert the communication, which is how
    MaxText does it and what Chapter 4 argues for.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P

    axis = {"dp": "data", "fsdp": "fsdp", "tp": "tensor", "cp": "context", "pp": "stage"}[strategy]

    if strategy == "dp":
        # Batch split, weights replicated: gradients pay an all-reduce.
        x_spec = P(axis, None, None)
        w_spec = {k: P(None, None) for k in ("wq", "wk", "wv", "wo", "wg", "wu", "wd")}
    elif strategy == "fsdp":
        # Batch split and weights split: an all-gather per layer in the forward
        # pass, a reduce-scatter on the way back.
        x_spec = P(axis, None, None)
        w_spec = {k: P(axis, None) for k in ("wq", "wk", "wv", "wo", "wg", "wu", "wd")}
    elif strategy == "tp":
        # Megatron layout: column-split going in, row-split coming out, so each
        # of attention and MLP ends in one all-reduce.
        x_spec = P(None, None, None)
        w_spec = {
            "wq": P(None, axis),
            "wk": P(None, axis),
            "wv": P(None, axis),
            "wo": P(axis, None),
            "wg": P(None, axis),
            "wu": P(None, axis),
            "wd": P(axis, None),
        }
    elif strategy == "cp":
        # Sequence split, weights replicated: attention has to cross devices.
        x_spec = P(None, axis, None)
        w_spec = {k: P(None, None) for k in ("wq", "wk", "wv", "wo", "wg", "wu", "wd")}
    else:  # pp
        # Pipelining is not a sharding, it is a schedule, so it does not fit the
        # in_specs/out_specs shape of the other four. bench/pipeline.py owns it.
        raise SystemExit(
            "pipeline parallelism is a schedule rather than a sharding; "
            "use `python -m bench.pipeline` instead"
        )

    return (
        axis,
        NamedSharding(mesh, x_spec),
        lambda: {"layers": [{k: NamedSharding(mesh, v) for k, v in w_spec.items()}]},
    )


def pipeline_bubble(microbatches: int, stages: int) -> float:
    """GPipe's bubble fraction, `(|Z| - 1) / (m + |Z| - 1)`."""
    return (stages - 1) / (microbatches + stages - 1)


def main() -> int:
    args = build_parser().parse_args()

    flags = {}
    if args.latency_hiding is not None:
        flags["xla_gpu_enable_latency_hiding_scheduler"] = args.latency_hiding
    for override in args.xla:
        key, _, value = override.partition("=")
        flags[key.lstrip("-")] = value
    configure_environment(xla_flags=flags or None)

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    from parse_xplane import (  # noqa: E402
        collective_timing,
        find_xplanes,
        kernels,
        overlap_analysis,
        phase_breakdown,
        scope_breakdown,
    )

    n = min(args.devices, jax.device_count())
    cfg = BlockConfig()
    dtype = jnp.bfloat16
    annotate = not args.no_annotate

    axis = {"dp": "data", "fsdp": "fsdp", "tp": "tensor", "cp": "context", "pp": "stage"}[
        args.strategy
    ]
    mesh = Mesh(jax.devices()[:n], axis_names=(axis,))
    _, x_sharding, w_shardings = shardings(args.strategy, mesh, cfg)

    tag = args.tag or f"{args.strategy}{'' if annotate else '-noannot'}"
    with Run("transformer-block", tag=tag, root=args.root, freeze=not args.no_freeze) as run:
        run.note("strategy", args.strategy)
        run.note("devices", n)
        run.note("config", cfg.__dict__)
        run.note("params_per_layer", cfg.params())
        run.note("layers", args.layers)
        run.note("seq_len", args.seq_len)
        run.note("annotated", annotate)
        run.note("latency_hiding_override", args.latency_hiding)
        if args.strategy == "pp":
            run.note("microbatches", args.microbatches)
            run.note("predicted_bubble_fraction", pipeline_bubble(args.microbatches, n))

        key = jax.random.key(0)
        params = init_params(cfg, args.layers, dtype, key)
        w_sharding = w_shardings()
        w_sharding = {"layers": w_sharding["layers"] * args.layers}
        params = jax.device_put(params, w_sharding)

        # Adam moments live wherever the weights do, which is what makes the
        # optimizer state sharded under FSDP and replicated under DP.
        opt_state = jax.device_put(
            {
                "m": [{k: jnp.zeros_like(v) for k, v in layer.items()} for layer in params["layers"]],
                "v": [
                    {k: jnp.zeros(v.shape, jnp.float32) for k, v in layer.items()}
                    for layer in params["layers"]
                ],
            },
            {"m": w_sharding["layers"], "v": w_sharding["layers"]},
        )

        grad_fn = jax.value_and_grad(loss_fn)

        def train_step(p, st, xx):
            loss, grads = grad_fn(p, xx, cfg)
            p, st = adam_update(p, grads, st)
            return loss, p, st

        # Donated buffers are invalidated by each call, so the carry has to live
        # outside the token loop rather than be rebuilt from the originals.
        carried = {"p": params, "st": opt_state}

        for tokens in args.tokens:
            if tokens % args.seq_len == 0:
                batch, seq = tokens // args.seq_len, args.seq_len
            else:
                batch, seq = 1, tokens
            global_batch = batch * n if args.strategy in ("dp", "fsdp") else batch
            if global_batch == 0:
                continue

            x = jax.device_put(
                jnp.zeros((global_batch, seq, cfg.d_model), dtype), x_sharding
            )

            step = jax.jit(train_step, donate_argnums=(0, 1))
            counter = itertools.count()

            def fn(xx=x, jitted=step, c=carried):
                if annotate:
                    with jax.profiler.StepTraceAnnotation("train", step_num=next(counter)):
                        out = jitted(c["p"], c["st"], xx)
                else:
                    out = jitted(c["p"], c["st"], xx)
                c["p"], c["st"] = out[1], out[2]
                return out

            tokens_total = global_batch * seq
            fwd_flops = cfg.flops_per_token(seq) * tokens_total * args.layers
            step_flops = 3.0 * fwd_flops  # forward plus backward

            m = run.measure(
                fn,
                name=f"{args.strategy}-{tokens}tok",
                trace=f"{tokens}" if args.trace else None,
                warmup=args.warmup,
                repeats=args.repeats,
                flops=step_flops,
                meta={
                    "tokens_per_device": tokens,
                    "global_tokens": tokens_total,
                    "batch": global_batch,
                    "seq_len": seq,
                },
            )
            mfu = (step_flops / m.median_s) / (n * MI300X["bf16_flops"])
            m.meta["mfu"] = mfu
            print(f"    MFU {mfu:.1%} of {n} x 1307 TFLOP/s")

            if args.trace:
                paths = find_xplanes(m.meta["trace_dir"])
                timing = collective_timing(paths)
                m.meta["collective_timing"] = timing
                rows = kernels(paths)
                total_us = sum(r["total_duration_us"] for r in rows)
                comm_us = sum(
                    r["total_duration_us"] for r in rows if "nccl" in str(r["name"]).lower()
                )
                m.meta["kernel_total_us"] = total_us
                m.meta["collective_total_us"] = comm_us
                m.meta["collective_share"] = comm_us / total_us if total_us else 0.0

                # Communication time that is not hidden. The step pays wall clock
                # for the collective only where it failed to overlap compute.
                busy_per_step = timing.get("busy_s_median", 0.0) or 0.0
                m.meta["collective_exposed_share"] = (
                    busy_per_step / m.median_s if m.median_s else 0.0
                )
                print(
                    f"    collectives {comm_us / total_us:.1%} of device kernel time, "
                    f"{busy_per_step * 1e6:.0f} us on the wire per step "
                    f"({busy_per_step / m.median_s:.1%} of step time if fully exposed)"
                )

                overlap = overlap_analysis(paths)
                m.meta["overlap"] = overlap
                phases = phase_breakdown(paths)
                m.meta["phases"] = phases
                print(
                    f"    overlap: {overlap['hidden_fraction']:.1%} of collective time has "
                    f"compute underneath it, {overlap['exposed_s'] * 1e3:.0f} ms exposed"
                )
                print(
                    "    phases: "
                    + "  ".join(f"{k} {v:.1%}" for k, v in phases["shares"].items() if v > 0.001)
                )

                scopes = scope_breakdown(paths)
                m.meta["scopes"] = scopes
                for s in scopes[:8]:
                    print(
                        f"      {s['share']:>6.1%}  {s['total_duration_us'] / 1e3:>8.2f} ms  "
                        f"{s['scope'][:40]}"
                    )
            del x

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
