"""Collective bandwidth on one MI300X node, measured through JAX.

This is the measurement Chapter 4's cost model is calibrated against and the one
Chapter 6's tensor-parallelism threshold swings on. The open question it exists
to settle: during an 8-way all-reduce, does a GPU push data out over all seven
of its xGMI links at once, or does RCCL build one ring and use a single link at
a time? The answer moves Chapter 6's minimum profitable feed-forward dimension
by roughly 7x.

    python -m bench.rccl_sweep --devices 8
    python -m bench.rccl_sweep --devices 2 4 8 --max-bytes 268435456

There is no `rccl-tests` binary in this container, and a JAX-level number is
what the book wants anyway: it includes whatever XLA does around the collective,
which is what a real model pays.

**Timed from the trace, not the wall clock.** Dispatching to eight devices from
Python costs about a millisecond, which is longer than most of these collectives
take; a timing loop here measures the loop. The RCCL kernel's own duration comes
out of the XPlane capture instead.

Two times are reported for every point, and they answer different questions:

  * `busy` is the shortest per-device kernel in an iteration. The device that
    arrives last does not wait for anyone, so its kernel is the transfer alone.
    **This is the wire rate.**
  * `span` runs from the first device entering the collective to the last one
    leaving, so it includes arrival skew. This is closer to what a step pays.

Bandwidth conventions follow `nccl-tests` so the figures are comparable with
published numbers: `algbw = size / time`, and `busbw = algbw * 2(n-1)/n` for
all-reduce or `algbw * (n-1)/n` for the other three. **busbw is the per-GPU
egress rate**, to be compared against the 448 GB/s of aggregate xGMI a single
MI300X has, or the 64 GB/s of one link.
"""

from __future__ import annotations

import math
import sys
from functools import partial
from pathlib import Path
from typing import Any, Callable

from ._harness import MI300X, Run, base_parser, configure_environment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

DEFAULT_MIN_BYTES = 1 << 10
DEFAULT_MAX_BYTES = 1 << 30
COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")


def build_parser():
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--devices", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--collectives", nargs="+", default=list(COLLECTIVES))
    p.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    p.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--step",
        type=int,
        default=1,
        help="powers of two to skip; 2 gives powers of four",
    )
    return p


def bus_factor(collective: str, n: int) -> float:
    """The nccl-tests correction from application rate to wire rate."""
    if n <= 1:
        return 0.0
    return 2.0 * (n - 1) / n if collective == "all_reduce" else (n - 1) / n


def sizes_between(lo: int, hi: int, step: int = 1) -> list[int]:
    return [1 << k for k in range(int(math.log2(lo)), int(math.log2(hi)) + 1, step)]


def _sharded_ones(shape, dtype, mesh, spec):
    """Allocate directly onto the mesh.

    Building the array on one device and resharding it would put a scatter in
    front of every measurement, and it shows up in the trace as `_multi_slice`.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding

    return jax.jit(
        partial(jnp.ones, shape, dtype),
        out_shardings=NamedSharding(mesh, spec),
    )()


def make_collective(
    collective: str, mesh: Any, axis: str, n: int, nbytes: int, dtype: Any
) -> tuple[Callable[[], Any], int] | None:
    """Build a jitted single-collective callable and the size to report.

    Returns None when the requested size cannot be expressed on this many
    devices, which happens at the small end once a buffer no longer divides.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    itemsize = jnp.dtype(dtype).itemsize
    elements = nbytes // itemsize
    if elements < n:
        return None

    if collective == "all_reduce":
        per_device = elements
        x = _sharded_ones((n, per_device), dtype, mesh, P(axis, None))

        @jax.jit
        @jax.shard_map(mesh=mesh, in_specs=P(axis, None), out_specs=P(axis, None))
        def fn(v):
            return jax.lax.psum(v, axis)

        return (lambda: fn(x)), per_device * itemsize

    if collective == "all_gather":
        # nccl-tests reports the output size, so each device contributes 1/n.
        per_device = elements // n
        if per_device == 0:
            return None
        x = _sharded_ones((n * per_device,), dtype, mesh, P(axis))

        @jax.jit
        @jax.shard_map(mesh=mesh, in_specs=P(axis), out_specs=P(axis))
        def fn(v):
            return jax.lax.all_gather(v, axis, tiled=True)

        return (lambda: fn(x)), per_device * n * itemsize

    if collective == "reduce_scatter":
        # nccl-tests reports the input size; the output is 1/n of it.
        if elements % n:
            return None
        x = _sharded_ones((n * elements,), dtype, mesh, P(axis))

        @jax.jit
        @jax.shard_map(mesh=mesh, in_specs=P(axis), out_specs=P(axis))
        def fn(v):
            return jax.lax.psum_scatter(v, axis, tiled=True)

        return (lambda: fn(x)), elements * itemsize

    if collective == "all_to_all":
        if elements % n:
            return None
        x = _sharded_ones((n * elements,), dtype, mesh, P(axis))

        @jax.jit
        @jax.shard_map(mesh=mesh, in_specs=P(axis), out_specs=P(axis))
        def fn(v):
            return jax.lax.all_to_all(v, axis, 0, 0, tiled=True)

        return (lambda: fn(x)), elements * itemsize

    raise ValueError(f"unknown collective {collective}")


def main() -> int:
    args = build_parser().parse_args()
    configure_environment()

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    from parse_xplane import collective_timing, find_xplanes  # noqa: E402

    available = jax.device_count()
    device_counts = [n for n in args.devices if n <= available]
    if not device_counts:
        raise SystemExit(f"asked for {args.devices} devices, only {available} present")

    dtype = getattr(jnp, args.dtype)
    sizes = sizes_between(args.min_bytes, args.max_bytes, args.step)
    link = MI300X["xgmi_link_unidir_bytes_per_s"]

    with Run("rccl-sweep", tag=args.tag, root=args.root, freeze=not args.no_freeze) as run:
        run.note("dtype", args.dtype)
        run.note("device_counts", device_counts)
        run.note("sizes_bytes", sizes)
        run.note("xgmi_aggregate_egress_bytes_per_s", MI300X["xgmi_egress_unidir_bytes_per_s"])
        run.note("xgmi_single_link_bytes_per_s", link)

        for n in device_counts:
            mesh = Mesh(jax.devices()[:n], axis_names=("x",))
            for collective in args.collectives:
                factor = bus_factor(collective, n)
                print(f"\n=== {collective}, {n} devices (busbw factor {factor:.3f}) ===")
                print(
                    f"  {'bytes':>14}  {'busy':>10}  {'busbw':>11}  {'links':>6}  "
                    f"{'span':>10}  {'skew':>6}"
                )
                for nbytes in sizes:
                    built = make_collective(collective, mesh, "x", n, nbytes, dtype)
                    if built is None:
                        continue
                    fn, reported = built
                    try:
                        m = run.measure(
                            fn,
                            name=f"{collective}-n{n}-{reported}B",
                            trace=f"{collective}/n{n}/{reported}",
                            warmup=args.warmup,
                            repeats=args.repeats,
                            meta={
                                "collective": collective,
                                "devices": n,
                                "bytes": reported,
                                "bus_factor": factor,
                            },
                        )
                    except Exception as exc:  # running out of memory at the top is expected
                        print(f"  {reported:>14,}  skipped: {type(exc).__name__}: {exc}")
                        break

                    timing = collective_timing(find_xplanes(m.meta["trace_dir"]))
                    m.meta["collective_timing"] = timing
                    if "error" in timing:
                        print(f"  {reported:>14,}  {timing['error']}")
                        continue

                    busy, span = timing["busy_s_median"], timing["span_s_median"]
                    busbw = reported / busy * factor
                    m.meta |= {
                        "busy_s": busy,
                        "span_s": span,
                        "algbw_bytes_per_s": reported / busy,
                        "busbw_bytes_per_s": busbw,
                        "busbw_span_bytes_per_s": reported / span * factor,
                        "equivalent_xgmi_links": busbw / link,
                        "dispatch_overhead_s": m.median_s - span,
                    }
                    print(
                        f"  {reported:>14,}  {busy * 1e6:>9.1f}u  {busbw / 1e9:>8.1f} GB/s  "
                        f"{busbw / link:>6.2f}  {span * 1e6:>9.1f}u  "
                        f"{timing['skew_fraction']:>5.0%}"
                    )

        _report_verdict(run, link)

    return 0


def _report_verdict(run: Run, link: float) -> None:
    """Answer the two questions this benchmark exists for.

    Counting links against the 64 GB/s data sheet figure understates them,
    because no link runs at its rated speed. Calibrate on the two-device case
    instead, where a GPU has exactly one peer and therefore exactly one link, and
    the ratio at higher device counts is then a real count of links in use.
    """
    peaks: dict[tuple[str, int], Any] = {}
    for m in run.measurements:
        if "busbw_bytes_per_s" not in m.meta:
            continue
        key = (m.meta["collective"], m.meta["devices"])
        best = peaks.get(key)
        if best is None or m.meta["busbw_bytes_per_s"] > best.meta["busbw_bytes_per_s"]:
            peaks[key] = m

    two_device = [m.meta["busbw_bytes_per_s"] for (_, n), m in peaks.items() if n == 2]
    per_link = max(two_device) if two_device else link
    run.note("per_link_achieved_bytes_per_s", per_link)
    run.note("per_link_efficiency", per_link / link)
    if two_device:
        print(
            f"\nper-link rate, calibrated on the 2-device case: {per_link / 1e9:.1f} GB/s "
            f"= {per_link / link:.0%} of the {link / 1e9:.0f} GB/s xGMI spec"
        )

    print("\n=== peak bus bandwidth (per-GPU egress, wire rate) ===")
    summary = []
    for (collective, n), m in sorted(peaks.items()):
        busbw = m.meta["busbw_bytes_per_s"]
        row = {
            "collective": collective,
            "devices": n,
            "at_bytes": m.meta["bytes"],
            "busbw_bytes_per_s": busbw,
            "peers": n - 1,
            "links_lit": busbw / per_link,
            "per_link_bytes_per_s": busbw / (n - 1) if n > 1 else busbw,
        }
        summary.append(row)
        print(
            f"  {collective:<15} n={n}  {busbw / 1e9:>7.1f} GB/s at {m.meta['bytes']:>13,} B"
            f"  {row['links_lit']:>5.2f} of {n - 1} links"
        )
    run.note("peak_busbw", summary)

    for collective, question in (
        ("all_reduce", "8-way all-reduce"),
        ("all_to_all", "8-way all-to-all"),
    ):
        m = peaks.get((collective, 8))
        if not m:
            continue
        busbw = m.meta["busbw_bytes_per_s"]
        lit = busbw / per_link
        verdict = (
            "one ring, a single link at a time"
            if lit < 1.5
            else f"{lit:.1f} of 7 links, a partial mesh"
            if lit < 6.0
            else "all seven links concurrently"
        )
        run.note(f"{collective}_8way_busbw_bytes_per_s", busbw)
        run.note(f"{collective}_8way_links_lit", lit)
        run.note(f"{collective}_8way_verdict", verdict)
        print(f"\n{question}: {busbw / 1e9:.1f} GB/s, {lit:.2f} links -> {verdict}")


if __name__ == "__main__":
    raise SystemExit(main())
