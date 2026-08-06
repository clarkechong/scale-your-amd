"""Pipeline parallelism, and the bubble it cannot avoid.

Chapter 6 predicts a bubble fraction of `(|Z| - 1) / (m + |Z| - 1)` for a
pipeline of `|Z|` stages fed `m` microbatches: the first stage waits for nobody
and the last stage waits for everybody, and the fill and drain cost `|Z| - 1`
stage-times no matter how fast the stages are.

    python -m bench.pipeline --stages 8 --microbatches 1 2 4 8 16 32

**This is a bubble demo and the chapter says so.** Pipelining exists to cross host
boundaries, where the alternative is sending activations over a NIC, and we have
one host. What a single node can still show is that the bubble is real, that it
shrinks the way the formula says, and that the stage-boundary transfers hide.

Layers are split across devices and microbatches are dispatched without blocking,
so JAX's asynchronous dispatch is what actually does the pipelining. If it did
not, the measured time would be `m * |Z|` stage-times rather than
`m + |Z| - 1`, which is itself worth knowing.
"""

from __future__ import annotations

import time
from typing import Any

from ._harness import MI300X, Run, base_parser, configure_environment
from .transformer_block import BlockConfig, block_forward, init_params


def build_parser():
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--stages", type=int, default=8)
    p.add_argument("--layers-per-stage", type=int, default=2)
    p.add_argument("--microbatches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--microbatch-tokens", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=1024)
    return p


def bubble_fraction(microbatches: int, stages: int) -> float:
    return (stages - 1) / (microbatches + stages - 1)


def main() -> int:
    args = build_parser().parse_args()
    configure_environment()

    import jax
    import jax.numpy as jnp
    from jax.sharding import SingleDeviceSharding

    stages = min(args.stages, jax.device_count())
    devices = jax.devices()[:stages]
    cfg = BlockConfig()
    dtype = jnp.bfloat16

    seq = min(args.seq_len, args.microbatch_tokens)
    batch = max(1, args.microbatch_tokens // seq)

    with Run("pipeline", tag=args.tag, root=args.root, freeze=not args.no_freeze) as run:
        run.note("stages", stages)
        run.note("layers_per_stage", args.layers_per_stage)
        run.note("microbatch", {"batch": batch, "seq": seq, "tokens": batch * seq})
        run.note("note", "forward only; this is a fill-and-drain demo, not a training schedule")

        # One stage's worth of weights per device, and a jitted forward that
        # produces its output on the same device it consumed its input from.
        stage_params = []
        stage_fns = []
        for i, device in enumerate(devices):
            params = init_params(cfg, args.layers_per_stage, dtype, jax.random.key(i))
            placement = SingleDeviceSharding(device)
            stage_params.append(jax.device_put(params, placement))

            def stage(p, x, _cfg=cfg):
                h = x
                for j, layer in enumerate(p["layers"]):
                    with jax.named_scope(f"stage_layer_{j}"):
                        h = block_forward(layer, h, _cfg)
                return h

            stage_fns.append(jax.jit(stage, out_shardings=placement))

        x0 = jax.device_put(jnp.zeros((batch, seq, cfg.d_model), dtype), SingleDeviceSharding(devices[0]))

        def run_pipeline(m: int):
            """Push m microbatches through, blocking only at the very end."""
            outputs = []
            for _ in range(m):
                h = x0
                for i, fn in enumerate(stage_fns):
                    if i:
                        h = jax.device_put(h, SingleDeviceSharding(devices[i]))
                    h = fn(stage_params[i], h)
                outputs.append(h)
            return outputs

        # One stage alone, to get the stage-time the prediction is expressed in.
        single = run.measure(
            lambda: stage_fns[0](stage_params[0], x0),
            name="one-stage",
            warmup=args.warmup,
            repeats=args.repeats,
            meta={"stages": 1, "microbatches": 1},
        )
        stage_s = single.median_s
        run.note("stage_time_s", stage_s)

        print(f"\n  one stage: {stage_s * 1e3:.3f} ms, {stages} stages, ideal is m stage-times")
        print(
            f"  {'microbatches':>13}{'measured':>11}{'ideal':>10}{'predicted':>11}"
            f"{'measured':>11}{'stage-times':>13}"
        )
        print(f"  {'':>13}{'':>11}{'m*t':>10}{'bubble':>11}{'bubble':>11}{'elapsed/t':>13}")

        for m in args.microbatches:
            meas = run.measure(
                lambda mm=m: run_pipeline(mm),
                name=f"pipeline-m{m}",
                warmup=max(1, args.warmup // 2),
                repeats=max(3, args.repeats // 2),
                meta={"stages": stages, "microbatches": m},
            )
            ideal_s = m * stage_s
            predicted = bubble_fraction(m, stages)
            measured_bubble = 1.0 - ideal_s / meas.median_s if meas.median_s else 0.0
            meas.meta |= {
                "ideal_s": ideal_s,
                "predicted_bubble": predicted,
                "measured_bubble": measured_bubble,
                "elapsed_in_stage_times": meas.median_s / stage_s,
                "predicted_stage_times": m + stages - 1,
            }
            print(
                f"  {m:>13}{meas.median_s * 1e3:>10.2f}m{ideal_s * 1e3:>9.2f}m"
                f"{predicted:>11.1%}{measured_bubble:>11.1%}"
                f"{meas.median_s / stage_s:>8.1f} / {m + stages - 1}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
