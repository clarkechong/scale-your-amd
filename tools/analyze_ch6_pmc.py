#!/usr/bin/env python3
"""Summarize the Chapter 6 one-layer attention PMC captures.

The four counter groups were collected in separate rocprofv3 processes. This
script therefore aggregates counters within each pass and never compares their
timestamps or durations. FETCH_SIZE and WRITE_SIZE are KiB. The gfx950 profiler
definition converts memory-controller requests to bytes and divides by 1024.
SQ_INSTS_VALU_MFMA_MOPS_BF16 counts BF16 matrix math operations divided by
512, so multiplying by 512 gives the profiler's BF16 MFMA operation count.

Run from the repository root:

    python tools/analyze_ch6_pmc.py
    python tools/analyze_ch6_pmc.py --list-kernels
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path("/tmp/archive/llama7b/rocprof-pmc-20260910-1layer")
BACKENDS = ("xla", "te", "aiter", "triton")
REMAT_POLICIES = ("none", "minimal", "full")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def counter_values(path: Path) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in read_rows(path):
        name = row["Counter_Name"]
        totals[name] = totals.get(name, 0.0) + float(row["Counter_Value"])
    return totals


def summarize_variant(root: Path, prefix: str) -> dict[str, float]:
    groups = {
        group: counter_values(root / f"{prefix}-{group}_counter_collection.csv")
        for group in ("compute", "fetch", "cache", "write")
    }
    hits = groups["cache"]["TCC_HIT_sum"]
    misses = groups["cache"]["TCC_MISS_sum"]
    return {
        "mfma_tflop": groups["compute"]["SQ_INSTS_VALU_MFMA_MOPS_BF16"] * 512 / 1e12,
        "waves_m": groups["compute"]["SQ_WAVES"] / 1e6,
        "fetch_gib": groups["fetch"]["FETCH_SIZE"] / 1024**2,
        "write_gib": groups["write"]["WRITE_SIZE"] / 1024**2,
        "l2_hit_pct": 100 * hits / (hits + misses),
    }


def compact_kernel_name(name: str) -> str:
    if len(name) <= 88:
        return name
    mt = re.search(r"_MT(\d+x\d+x\d+)", name)
    mi = re.search(r"_MI(\d+x\d+x\d+)", name)
    if mt or mi:
        parts = [part.group(1) for part in (mt, mi) if part]
        return "Tensile " + " / ".join(parts)
    return name[:85] + "..."


def list_kernels(root: Path, backend: str) -> None:
    rows = read_rows(root / f"attention-{backend}-compute_kernel_trace.csv")
    counts = Counter(row["Kernel_Name"] for row in rows)
    first = {}
    for row in rows:
        first.setdefault(row["Kernel_Name"], row)

    print(f"\n[{backend}] {len(rows)} dispatches, {len(counts)} unique kernels")
    for name, count in counts.items():
        row = first[name]
        grid = "x".join(row[f"Grid_Size_{axis}"] for axis in "XYZ")
        workgroup = "x".join(row[f"Workgroup_Size_{axis}"] for axis in "XYZ")
        resources = (
            f"LDS={row['LDS_Block_Size']} B "
            f"VGPR={row['VGPR_Count']} "
            f"AccVGPR={row['Accum_VGPR_Count']}"
        )
        print(
            f"{count:2d}  grid={grid:>16}  wg={workgroup:>11}  "
            f"{resources:<34}  {compact_kernel_name(name)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--list-kernels", action="store_true")
    args = parser.parse_args()

    print(
        "backend  BF16 MFMA TFLOP  waves (M)  HBM fetch (GiB)  "
        "HBM write (GiB)  L2 hit (%)"
    )
    for backend in BACKENDS:
        row = summarize_variant(args.root, f"attention-{backend}")
        print(
            f"{backend:7}  {row['mfma_tflop']:16.3f}  {row['waves_m']:9.3f}  "
            f"{row['fetch_gib']:15.3f}  {row['write_gib']:15.3f}  "
            f"{row['l2_hit_pct']:10.2f}"
        )

    print("\nremat policy  BF16 MFMA TFLOP  waves (M)  HBM fetch (GiB)  HBM write (GiB)  L2 hit (%)")
    for policy in REMAT_POLICIES:
        row = summarize_variant(args.root, f"remat-{policy}")
        print(
            f"{policy:12}  {row['mfma_tflop']:16.3f}  {row['waves_m']:9.3f}  "
            f"{row['fetch_gib']:15.3f}  {row['write_gib']:15.3f}  "
            f"{row['l2_hit_pct']:10.2f}"
        )

    if args.list_kernels:
        for backend in BACKENDS:
            list_kernels(args.root, backend)


if __name__ == "__main__":
    main()
