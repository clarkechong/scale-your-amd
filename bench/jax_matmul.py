"""The single-device bf16 matmul Chapter 2 predicts and Chapter 3 measures.

Chapter 2 ends by predicting 105 microseconds for a 4096-cubed bf16 matmul on
one MI300X, from `2 * 4096**3 / 1307.4e12`. This closes that loop.

Runs on one GPU deliberately: an 8-GPU capture sums op times across devices,
per Chapter 3's limitations table, and this number has to be readable without
dividing by anything.

    python -m bench.jax_matmul --trace
    python -m bench.jax_matmul --trace --autotune 4 --tag autotune4

The second form is Chapter 8's kernel-selection A/B. The container ships
`--xla_gpu_autotune_level=0`, so the default arm is autotuning off.
"""

from __future__ import annotations

import json
from typing import Any

from ._harness import MI300X, Run, base_parser, configure_environment, sustained_clocks

SIZES = (1024, 2048, 4096, 8192)


def build_parser():
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--autotune",
        type=int,
        default=None,
        help="override --xla_gpu_autotune_level (container ships 0)",
    )
    p.add_argument("--device", default="0", help="which physical GPU to pin to")
    p.add_argument(
        "--sustain",
        type=float,
        default=0.0,
        help="seconds of back-to-back matmul to find the steady-state clock",
    )
    return p


def peak_flops(dtype: str) -> float:
    return {
        "bfloat16": MI300X["bf16_flops"],
        "float16": MI300X["fp16_flops"],
        "float32": MI300X["fp32_flops"],
    }[dtype]


def kernel_summary(trace_dir: Any) -> dict[str, Any]:
    """The device-side truth for this matmul, straight out of the capture.

    Wall-clock timing at these sizes is mostly dispatch overhead, so the number
    the book quotes is the kernel duration, not the loop time.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from parse_xplane import find_xplanes, kernels  # noqa: E402

    try:
        paths = find_xplanes(trace_dir)
    except SystemExit:
        return {"error": "no xplane written"}
    rows = kernels(paths)
    if not rows:
        return {"error": "no kernels in trace"}

    # Pick the kernel the dot_general lowered to rather than the longest row.
    # Autotuner probes and allocator kernels can outweigh it in a capture that
    # caught any compilation.
    dots = [r for r in rows if "dot" in str(r["op_name"]).lower()]
    top = dots[0] if dots else rows[0]
    return {
        "kernel_name": top["name"],
        "op_name": top["op_name"],
        "occurrences": top["occurrences"],
        "avg_duration_us": top["avg_duration_us"],
        "min_duration_us": top["min_duration_us"],
        "max_duration_us": top["max_duration_us"],
        "total_duration_us": top["total_duration_us"],
        "grid": top["grid"],
        "block": top["block"],
        "selected_by": "op_name contains dot" if dots else "longest kernel",
        "distinct_kernels": len(rows),
        "all_kernels": [
            {
                "name": r["name"],
                "op_name": r["op_name"],
                "occurrences": r["occurrences"],
                "avg_us": r["avg_duration_us"],
            }
            for r in rows[:5]
        ],
    }


def main() -> int:
    args = build_parser().parse_args()

    flags = {}
    if args.autotune is not None:
        flags["xla_gpu_autotune_level"] = str(args.autotune)
    configure_environment(xla_flags=flags or None, visible_devices=args.device)

    import jax
    import jax.numpy as jnp

    if jax.device_count() != 1:
        raise SystemExit(
            f"expected 1 device, saw {jax.device_count()}. "
            "Chapter 3's op-time summing makes a multi-device capture unreadable here."
        )

    dtype = getattr(jnp, args.dtype)
    peak = peak_flops(args.dtype)

    with Run("matmul", tag=args.tag, root=args.root, freeze=not args.no_freeze) as run:
        run.note("device", str(jax.devices()[0]))
        run.note("device_kind", jax.devices()[0].device_kind)
        run.note("dtype", args.dtype)
        run.note("peak_flops", peak)
        run.note("autotune_level_override", args.autotune)

        matmul = jax.jit(lambda a, b: a @ b)

        for n in args.sizes:
            a = jnp.ones((n, n), dtype)
            b = jnp.ones((n, n), dtype)
            flops = 2.0 * n**3
            # Three n^2 matrices moved once each, if nothing stays resident.
            byte_count = 3.0 * n * n * jnp.dtype(dtype).itemsize
            predicted = flops / peak

            m = run.measure(
                lambda a=a, b=b: matmul(a, b),
                name=f"matmul-{n}",
                trace=f"n{n}" if args.trace else None,
                warmup=args.warmup,
                repeats=args.repeats,
                flops=flops,
                bytes=byte_count,
                predicted_s=predicted,
                meta={"n": n, "arithmetic_intensity": flops / byte_count},
            )

            if args.trace:
                summary = kernel_summary(m.meta["trace_dir"])
                m.meta["kernel"] = summary
                if "min_duration_us" in summary:
                    kernel_s = summary["min_duration_us"] * 1e-6
                    m.meta["kernel_time_s"] = kernel_s
                    m.meta["kernel_tflops"] = flops / kernel_s / 1e12
                    m.meta["kernel_fraction_of_peak"] = flops / kernel_s / peak
                    m.meta["kernel_vs_prediction"] = kernel_s / predicted
                    m.meta["dispatch_overhead_s"] = m.median_s - kernel_s
                    print(
                        f"    device kernel {summary['min_duration_us']:.2f} us "
                        f"({flops / kernel_s / 1e12:.1f} TFLOP/s, "
                        f"{flops / kernel_s / peak:.0%} of peak, "
                        f"{kernel_s / predicted:.2f}x prediction)"
                    )
                    print(f"      {str(summary['kernel_name'])[:96]}")

            if args.sustain:
                clocks = sustained_clocks(
                    lambda a=a, b=b: matmul(a, b), seconds=args.sustain
                )
                m.meta["sustained_clocks"] = clocks
                if clocks.get("achievable_bf16_flops"):
                    achievable = clocks["achievable_bf16_flops"]
                    m.meta["achievable_flops_at_measured_clock"] = achievable
                    kernel_s = m.meta.get("kernel_time_s")
                    if kernel_s:
                        m.meta["kernel_fraction_of_achievable"] = flops / kernel_s / achievable
                    print(
                        f"    sustained {clocks['sclk_mhz_median']:.0f} MHz "
                        f"({clocks['sclk_fraction_of_boost']:.0%} of the 2100 MHz boost), "
                        f"{clocks.get('power_w_median', 0):.0f} W  ->  "
                        f"achievable peak {achievable / 1e12:.0f} TFLOP/s"
                    )
                    if kernel_s:
                        print(
                            f"      kernel reaches {flops / kernel_s / achievable:.0%} "
                            f"of that achievable peak"
                        )
            del a, b

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
