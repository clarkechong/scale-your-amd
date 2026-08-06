#!/usr/bin/env python3
"""Generate the book's plotted figures from `runs/`.

Three of the figures in the screenshot checklist are plots rather than captures,
so they are built here and regenerate whenever the underlying run does. Anything
in `assets/img/` produced by this script can be deleted and rebuilt; the
hand-captured screenshots cannot, which is the reason for the split.

    python tools/make_figures.py
    python tools/make_figures.py --only rccl-bandwidth-curve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "assets" / "img"

XGMI_LINK = 64e9
BF16_PEAK = 1307.4e12

PRETTY = {
    "all_reduce": "all-reduce",
    "all_gather": "all-gather",
    "reduce_scatter": "reduce-scatter",
    "all_to_all": "all-to-all",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def load(name: str) -> dict:
    path = RUNS / name / "results.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run the benchmark first")
    return json.loads(path.read_text())


def rccl_bandwidth_curve(run: str) -> Path:
    """Busbw against message size, four collectives, eight devices."""
    data = load(run)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for collective, label in PRETTY.items():
        points = sorted(
            (m["meta"]["bytes"], m["meta"]["busbw_bytes_per_s"] / 1e9)
            for m in data["measurements"]
            if m["meta"].get("collective") == collective
            and m["meta"].get("devices") == 8
            and "busbw_bytes_per_s" in m["meta"]
        )
        if points:
            ax.plot(*zip(*points), marker="o", markersize=2.5, linewidth=1.4, label=label)

    ax.axhline(320, color="0.35", linestyle="--", linewidth=1)
    ax.text(2**18, 328, "320 GB/s, the figure to use", fontsize=8, color="0.35")
    ax.axhline(XGMI_LINK / 1e9, color="0.6", linestyle=":", linewidth=1)
    ax.text(1.5e3, 70, "one xGMI link, 64 GB/s", fontsize=8, color="0.6")
    ax.set_ylim(-15, 355)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("message size (bytes)")
    ax.set_ylabel("per-GPU egress, busbw (GB/s)")
    ax.set_title("Collective bandwidth on 8x MI300X", loc="left", fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = OUT / "rccl-bandwidth-curve.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def rccl_links_lit(run: str) -> Path:
    """Peak busbw against device count, against the all-links-lit prediction."""
    data = load(run)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    peaks: dict[tuple[str, int], float] = {}
    for m in data["measurements"]:
        meta = m["meta"]
        if "busbw_bytes_per_s" not in meta:
            continue
        key = (meta["collective"], meta["devices"])
        peaks[key] = max(peaks.get(key, 0.0), meta["busbw_bytes_per_s"])

    per_link = max(v for (c, n), v in peaks.items() if n == 2)
    counts = sorted({n for _, n in peaks})

    for collective, label in PRETTY.items():
        ys = [peaks.get((collective, n), 0.0) / 1e9 for n in counts]
        ax.plot(counts, ys, marker="o", markersize=4, linewidth=1.4, label=label)

    predicted = [(n - 1) * per_link / 1e9 for n in counts]
    ax.plot(
        counts,
        predicted,
        color="0.3",
        linestyle="--",
        linewidth=1.2,
        label=f"(n-1) x {per_link / 1e9:.1f} GB/s per link",
    )

    ax.set_xticks(counts)
    ax.set_xlabel("devices in the collective")
    ax.set_ylabel("peak per-GPU egress (GB/s)")
    ax.set_title("RCCL lights every link it has", loc="left", fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = OUT / "rccl-links-lit.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def matmul_size_sweep(sustained_clock_mhz: float = 1590.0) -> Path:
    """Achieved TFLOP/s against n, against both rooflines.

    Reads the isolated runs, one process per size, since measuring a sweep in a
    single process lets the clock drop as it goes and bends the curve.
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.5))

    achievable = 304 * 2048 * sustained_clock_mhz * 1e6

    arms = {"autotuning off, as shipped": "at0", "autotuning on (level 4)": "at4"}
    for label, suffix in arms.items():
        points = []
        for directory in sorted((RUNS / "isolated").glob(f"*-matmul-n*-{suffix}")):
            results = directory / "results.json"
            if not results.exists():
                continue
            for m in json.loads(results.read_text())["measurements"]:
                tflops = m["meta"].get("kernel_tflops")
                if tflops:
                    points.append((m["meta"]["n"], tflops))
        if points:
            ax.plot(*zip(*sorted(points)), marker="o", markersize=4, linewidth=1.4, label=label)

    ax.axhline(BF16_PEAK / 1e12, color="0.3", linestyle="--", linewidth=1.2)
    ax.text(1050, BF16_PEAK / 1e12 + 25, "1307 TFLOP/s at the 2.10 GHz boost clock", fontsize=8)
    ax.axhline(achievable / 1e12, color="0.55", linestyle=":", linewidth=1.2)
    ax.text(
        1050,
        achievable / 1e12 + 25,
        f"{achievable / 1e12:.0f} TFLOP/s at the {sustained_clock_mhz:.0f} MHz it holds",
        fontsize=8,
        color="0.4",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 2048, 4096, 8192])
    ax.set_xticklabels(["1024", "2048", "4096", "8192"])
    ax.set_ylim(0, BF16_PEAK / 1e12 * 1.12)
    ax.set_xlabel("n, for an n x n x n bf16 matmul")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title("One MI300X, and two rooflines", loc="left", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = OUT / "matmul-size-sweep.png"
    fig.savefig(path)
    plt.close(fig)
    return path


FIGURES = {
    "rccl-bandwidth-curve": lambda: rccl_bandwidth_curve("20260805-rccl-sweep-full"),
    "rccl-links-lit": lambda: rccl_links_lit("20260805-rccl-sweep-full"),
    "matmul-size-sweep": matmul_size_sweep,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", choices=sorted(FIGURES), default=None)
    args = ap.parse_args()

    style()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only or sorted(FIGURES):
        try:
            path = FIGURES[name]()
        except SystemExit as exc:
            print(f"  skipped {name}: {exc}")
            continue
        print(f"  wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
