#!/usr/bin/env python3
"""Pull numbers out of an XPlane capture without opening a browser.

XProf's UI is the right way to *read* a profile and the wrong way to get a
number into prose: you cannot diff a screenshot, and a figure retyped from one
is a figure nobody can reproduce. This wraps the same converters the UI calls,
so the tables in the book come out of the trace file itself.

    python tools/parse_xplane.py kernels runs/20260805-matmul/
    python tools/parse_xplane.py limitations runs/20260805-transformer-block/
    python tools/parse_xplane.py collectives runs/.../ --json

Not reader-facing; this is our extraction tool.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

# Names XLA gives the collective ops, used to pick them out of a kernel table.
COLLECTIVE_RE = re.compile(
    r"(all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute|ncclDevKernel|rccl)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def find_xplanes(target: str | Path) -> list[str]:
    """Accept a run directory, a trace directory, or a single .xplane.pb."""
    p = Path(target)
    if p.is_file():
        return [str(p)]
    hits = sorted(p.rglob("*.xplane.pb"))
    if not hits:
        raise SystemExit(f"no *.xplane.pb found under {p}")
    return [str(h) for h in hits]


def tool_data(paths: Sequence[str], tool: str, **params: Any) -> Any:
    """Call one XProf converter and parse whatever comes back."""
    from xprof.convert import raw_to_tool_data

    data, _content_type = raw_to_tool_data.xspace_to_tool_data(list(paths), tool, params)
    if data is None:
        return None
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError:
            return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data


def available_tools(paths: Sequence[str]) -> list[str]:
    from xprof.convert import raw_to_tool_data

    return raw_to_tool_data.xspace_to_tool_names(list(paths))


# --------------------------------------------------------------------------
# Shape normalisation
#
# XProf returns two shapes depending on the tool: a gviz DataTable
# ({cols, rows}) for the table views, and plain nested JSON for the rest.
# --------------------------------------------------------------------------


def as_rows(data: Any) -> list[dict[str, Any]]:
    """Flatten whatever a converter returned into a list of dict rows."""
    if data is None:
        return []
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and {"cols", "rows"} <= set(data[0]):
            return as_rows(data[0])
        return [d for d in data if isinstance(d, dict)]
    if not isinstance(data, dict):
        return []
    if {"cols", "rows"} <= set(data):
        # Prefer the column id: labels carry units and unicode ("Total Duration
        # (\u03bcs)") and drift between XProf builds.
        labels = [c.get("id") or c.get("label") for c in data["cols"]]
        out = []
        for row in data["rows"]:
            cells = row.get("c", row) if isinstance(row, dict) else row
            values = [c.get("v") if isinstance(c, dict) else c for c in cells]
            out.append(dict(zip(labels, values)))
        return out
    for key in ("kernelReports", "opMetrics", "rows", "data", "table"):
        if key in data:
            return as_rows(data[key])
    return []


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _find_key(row: dict[str, Any], *candidates: str) -> Any:
    """Match a column by fuzzy name, since labels drift between XProf builds."""
    lowered = {str(k).lower().replace(" ", "").replace("_", ""): v for k, v in row.items()}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in lowered:
            return lowered[key]
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        for k, v in lowered.items():
            if key in k:
                return v
    return None


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------


# Columns XProf 2.23 emits for kernel_stats. Durations are microseconds.
KERNEL_ZERO_COLUMNS = (
    "registers_per_thread",
    "shmem_bytes",
    "occupancy_pct",
    "is_op_tensor_core_eligible",
    "is_kernel_using_tensor_core",
)


def kernels(paths: Sequence[str]) -> list[dict[str, Any]]:
    """Kernel Stats, sorted by total duration."""
    rows = as_rows(tool_data(paths, "kernel_stats"))
    out = []
    for r in rows:
        out.append(
            {
                "name": _find_key(r, "kernel_name"),
                "op_name": _find_key(r, "op_name"),
                "occurrences": _num(_find_key(r, "occurrences")),
                "total_duration_us": _num(_find_key(r, "total_duration_us")),
                "avg_duration_us": _num(_find_key(r, "avg_duration_us")),
                "min_duration_us": _num(_find_key(r, "min_duration_us")),
                "max_duration_us": _num(_find_key(r, "max_duration_us")),
                "grid": _find_key(r, "grid_dim"),
                "block": _find_key(r, "block_dim"),
                "registers_per_thread": _find_key(r, "registers_per_thread"),
                "shmem_bytes": _find_key(r, "shmem_bytes"),
                "occupancy_pct": _find_key(r, "occupancy_pct"),
                "is_op_tensor_core_eligible": _find_key(r, "is_op_tensor_core_eligible"),
                "is_kernel_using_tensor_core": _find_key(r, "is_kernel_using_tensor_core"),
            }
        )
    out.sort(key=lambda k: k["total_duration_us"], reverse=True)
    return out


def collectives(paths: Sequence[str]) -> list[dict[str, Any]]:
    """The RCCL side of a kernel table."""
    return [k for k in kernels(paths) if k["name"] and COLLECTIVE_RE.search(str(k["name"]))]


DEVICE_PLANE_RE = re.compile(r"/device:(GPU|TPU|XLA):(\d+)", re.IGNORECASE)


def device_events(paths: Sequence[str], pattern: str) -> dict[str, list[dict[str, Any]]]:
    """Every matching event on every device plane, with start and duration.

    Kernel Stats aggregates to min/avg/max over all instances on all devices,
    which is not enough for a collective: it cannot separate "this device waited
    for its neighbours" from "the transfer took this long". The raw events can.
    """
    from xprof.profile_data import ProfileData

    out: dict[str, list[dict[str, Any]]] = {}
    matcher = re.compile(pattern, re.IGNORECASE)
    for path in paths:
        data = ProfileData.from_file(path)
        try:
            for plane in data.planes:
                m = DEVICE_PLANE_RE.search(plane.name or "")
                if not m:
                    continue
                device = f"{m.group(1).upper()}:{m.group(2)}"
                bucket = out.setdefault(device, [])
                for line in plane.lines:
                    for event in line.events:
                        if matcher.search(event.name or ""):
                            bucket.append(
                                {
                                    "name": event.name,
                                    "line": line.name,
                                    "start_ns": event.start_ns,
                                    "duration_ns": event.duration_ns,
                                }
                            )
        finally:
            data.close()
    for events in out.values():
        events.sort(key=lambda e: e["start_ns"])
    return out


def collective_timing(paths: Sequence[str], pattern: str = "ncclDevKernel") -> dict[str, Any]:
    """Time a collective properly, separating transfer from arrival skew.

    Two numbers come out, and the book needs both:

      * `span`, from the first device entering the collective to the last one
        leaving. This is what the model pays.
      * `busy`, the shortest per-device kernel duration in that iteration. The
        device that arrives last does not wait, so its kernel is closest to the
        transfer alone. This is the wire rate.

    A large gap between the two is arrival skew, not bandwidth.
    """
    per_device = device_events(paths, pattern)
    per_device = {d: e for d, e in per_device.items() if e}
    if not per_device:
        return {"error": f"no events matching {pattern!r}", "devices": 0}

    counts = {d: len(e) for d, e in per_device.items()}
    iterations = min(counts.values())
    spans_ns: list[float] = []
    busy_ns: list[float] = []
    for i in range(iterations):
        starts = [per_device[d][i]["start_ns"] for d in per_device]
        durs = [per_device[d][i]["duration_ns"] for d in per_device]
        ends = [s + x for s, x in zip(starts, durs)]
        spans_ns.append(max(ends) - min(starts))
        busy_ns.append(min(durs))

    import statistics

    return {
        "devices": len(per_device),
        "iterations": iterations,
        "events_per_device": counts,
        "span_s_median": statistics.median(spans_ns) / 1e9,
        "span_s_min": min(spans_ns) / 1e9,
        "busy_s_median": statistics.median(busy_ns) / 1e9,
        "busy_s_min": min(busy_ns) / 1e9,
        "skew_fraction": (
            1.0 - statistics.median(busy_ns) / statistics.median(spans_ns)
            if statistics.median(spans_ns)
            else 0.0
        ),
        "kernel_name": next(iter(per_device.values()))[0]["name"],
    }


JIT_PREFIX_RE = re.compile(r"^jit\([^)]*\)/")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
# Wrappers autodiff and batching add around a scope name, which are not scopes.
AUTODIFF_WRAPPERS = frozenset({"transpose", "jvp", "vmap", "custom_jvp_call", "custom_vjp_call",
                               "remat", "checkpoint", "pjit", "jit", "scan", "while", "cond"})


def scope_path(op_name: str) -> list[str]:
    """Recover the `jax.named_scope` stack from an XLA op name.

    Op names survive autodiff, but not cleanly: the backward pass of a scope
    called `mlp` arrives as `transpose(jvp(mlp))`, and a fused op concatenates
    several names with a colon. Unwrapping both puts the forward and backward
    time for a sub-block on the same row.
    """
    op = str(op_name or "").split(":")[0].lstrip("(").rstrip("/")
    op = JIT_PREFIX_RE.sub("", op)
    if not op:
        return []
    parts: list[str] = []
    for component in op.split("/"):
        # The innermost identifier that is not a wrapper is the scope name.
        names = [n for n in IDENT_RE.findall(component) if n not in AUTODIFF_WRAPPERS]
        if names:
            parts.append(names[-1])
    # The final component is the primitive (dot_general, mul, ...), not a scope.
    return parts[:-1]


def scope_breakdown(paths: Sequence[str], *, level: str = "leaf") -> list[dict[str, Any]]:
    """Group device kernel time by named scope.

    `level` picks how to collapse nested scopes: "leaf" is the innermost, so
    `layer_0/mlp` counts as `mlp` and the sub-blocks of every layer add up;
    "root" is the outermost, which separates the layers; "full" keeps the path.
    """
    rows = kernels(paths)
    total = sum(r["total_duration_us"] for r in rows) or 1.0
    buckets: dict[str, dict[str, Any]] = {}
    for r in rows:
        path = scope_path(r["op_name"])
        if not path:
            # Two different failures worth telling apart: an op that carries a
            # name but no scope, and a kernel XLA emitted with no metadata at
            # all. Collectives and post-scheduling fusions land in the latter,
            # and no amount of annotating will recover them.
            scope = "<unnamed kernel>" if not str(r["op_name"] or "").strip() else "<unscoped>"
        elif level == "root":
            scope = path[0]
        elif level == "full":
            scope = "/".join(path)
        else:
            scope = path[-1]
        b = buckets.setdefault(
            scope, {"scope": scope, "total_duration_us": 0.0, "kernels": 0, "collective_us": 0.0}
        )
        b["total_duration_us"] += r["total_duration_us"]
        b["kernels"] += 1
        if COLLECTIVE_RE.search(str(r["name"] or "")):
            b["collective_us"] += r["total_duration_us"]
    for b in buckets.values():
        b["share"] = b["total_duration_us"] / total
    return sorted(buckets.values(), key=lambda b: b["total_duration_us"], reverse=True)


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _covered(span: tuple[float, float], merged: list[tuple[float, float]]) -> float:
    """How much of `span` is inside an already-merged interval list."""
    total = 0.0
    for start, end in merged:
        if end <= span[0]:
            continue
        if start >= span[1]:
            break
        total += min(end, span[1]) - max(start, span[0])
    return total


def overlap_analysis(paths: Sequence[str], pattern: str = "nccl|rccl") -> dict[str, Any]:
    """How much of the collective time has compute running underneath it.

    Chapter 4's claim is that an overlapped collective is nearly free. That is a
    statement about wall-clock concurrency on the device, so it needs the raw
    event intervals: merge every compute kernel into a set of busy intervals, then
    ask what fraction of each collective falls inside one.
    """
    from xprof.profile_data import ProfileData

    comm_re = re.compile(pattern, re.IGNORECASE)
    per_device: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for path in paths:
        data = ProfileData.from_file(path)
        try:
            for plane in data.planes:
                m = DEVICE_PLANE_RE.search(plane.name or "")
                if not m:
                    continue
                device = f"{m.group(1).upper()}:{m.group(2)}"
                bucket = per_device.setdefault(device, {"comm": [], "compute": []})
                for line in plane.lines:
                    # Derived timelines repeat the same events under summary rows.
                    if any(w in (line.name or "").lower() for w in ("steps", "xla modules", "xla ops", "source")):
                        continue
                    for event in line.events:
                        span = (event.start_ns, event.start_ns + event.duration_ns)
                        key = "comm" if comm_re.search(event.name or "") else "compute"
                        bucket[key].append(span)
        finally:
            data.close()

    results = {}
    total_comm = total_hidden = 0.0
    for device, buckets in per_device.items():
        merged_compute = _merge(buckets["compute"])
        merged_comm = _merge(buckets["comm"])
        comm_ns = sum(e - s for s, e in merged_comm)
        hidden_ns = sum(_covered(span, merged_compute) for span in merged_comm)
        total_comm += comm_ns
        total_hidden += hidden_ns
        results[device] = {
            "comm_ns": comm_ns,
            "hidden_ns": hidden_ns,
            "hidden_fraction": hidden_ns / comm_ns if comm_ns else None,
        }

    return {
        "per_device": results,
        "total_comm_s": total_comm / 1e9,
        "total_hidden_s": total_hidden / 1e9,
        "hidden_fraction": total_hidden / total_comm if total_comm else None,
        "exposed_s": (total_comm - total_hidden) / 1e9,
    }


BACKWARD_RE = re.compile(r"\btranspose\(")


def phase_breakdown(paths: Sequence[str]) -> dict[str, Any]:
    """Split device kernel time into forward, backward, optimizer and collective.

    Autodiff leaves a usable marker: JAX names the reverse pass
    `transpose(jvp(scope))`, so an op name containing `transpose(` is backward
    work. Everything scoped and not transposed is forward.
    """
    rows = kernels(paths)
    total = sum(r["total_duration_us"] for r in rows) or 1.0
    phases = {k: 0.0 for k in ("forward", "backward", "optimizer", "collective", "unattributed")}
    for r in rows:
        op = str(r["op_name"] or "")
        us = r["total_duration_us"]
        if COLLECTIVE_RE.search(str(r["name"] or "")):
            phases["collective"] += us
        elif "optimizer" in op:
            phases["optimizer"] += us
        elif BACKWARD_RE.search(op):
            phases["backward"] += us
        elif op.strip():
            phases["forward"] += us
        else:
            phases["unattributed"] += us
    return {
        "total_us": total,
        "phases": phases,
        "shares": {k: v / total for k, v in phases.items()},
        "backward_over_forward": (
            phases["backward"] / phases["forward"] if phases["forward"] else None
        ),
    }


def hlo_stats(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows = as_rows(tool_data(paths, "hlo_stats"))
    out = []
    for r in rows:
        out.append(
            {
                "hlo_op": _find_key(r, "hlo_op_name", "hlo_op", "name"),
                "category": _find_key(r, "hlo_category", "category"),
                "occurrences": _num(_find_key(r, "occurrences")),
                "total_time_us": _num(_find_key(r, "total_time_in_us", "total_self_time")),
                "avg_time_us": _num(_find_key(r, "avg_time_in_us", "time_in_us")),
                "pct_total_time": _num(_find_key(r, "total_time_as_percentage", "percentage")),
                "flops": _num(_find_key(r, "measured_flop_rate", "flops")),
                "bytes_accessed": _num(_find_key(r, "measured_memory_bw", "bytes_accessed")),
                "expression": _find_key(r, "hlo_op_expression", "expression"),
                "raw": r,
            }
        )
    out.sort(key=lambda h: h["total_time_us"], reverse=True)
    return out


def overview(paths: Sequence[str]) -> dict[str, Any]:
    return tool_data(paths, "overview_page") or {}


def op_profile(paths: Sequence[str]) -> dict[str, Any]:
    return tool_data(paths, "op_profile", group_by="program") or {}


def roofline(paths: Sequence[str]) -> Any:
    return tool_data(paths, "roofline_model")


def memory_profile(paths: Sequence[str]) -> Any:
    # This converter handles one host at a time.
    return tool_data(paths[:1], "memory_profile")


def device_op_time_s(paths: Sequence[str]) -> float:
    return sum(k["total_duration_us"] for k in kernels(paths)) / 1e6


def _walk(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def op_profile_summary(paths: Sequence[str]) -> dict[str, Any]:
    """Op Profile's root totals plus the IDLE child.

    The IDLE row is the one Chapter 3 calls misleading without step markers, so
    it is worth pulling out explicitly rather than eyeballing the tree.
    """
    prof = op_profile(paths)
    root = prof.get("byProgram") or prof.get("byCategory") or {}
    root_metrics = root.get("metrics", {})
    total_ps = _num(root_metrics.get("rawTime"))

    idle_ps = 0.0
    children = []
    for child in root.get("children", []):
        name = child.get("name", "")
        raw = _num(child.get("metrics", {}).get("rawTime"))
        children.append({"name": name, "raw_time_ps": raw})
        if str(name).upper() == "IDLE":
            idle_ps = raw

    return {
        "total_raw_time_ps": total_ps,
        "idle_raw_time_ps": idle_ps,
        "idle_fraction": idle_ps / total_ps if total_ps else None,
        "raw_flops": _num(root_metrics.get("rawFlops")),
        "bf16_flops": _num(root_metrics.get("bf16Flops")),
        "bandwidth_utils": root_metrics.get("bandwidthUtils"),
        "raw_bytes_accessed": root_metrics.get("rawBytesAccessedArray"),
        "children": sorted(children, key=lambda c: c["raw_time_ps"], reverse=True),
    }


def idle_fraction(paths: Sequence[str]) -> float | None:
    return op_profile_summary(paths)["idle_fraction"]


# --------------------------------------------------------------------------
# The limitations table
# --------------------------------------------------------------------------


def limitations(paths: Sequence[str], *, wall_seconds: float | None = None) -> dict[str, Any]:
    """Re-verify each row of Chapter 3's limitations table on this stack.

    The table was measured against ROCm 7.2.4 and the book is being re-pinned,
    so every row is a claim about a different build until it is re-run.
    """
    report: dict[str, Any] = {"xplanes": list(paths)}
    report["available_tools"] = available_tools(paths)

    ks = kernels(paths)
    report["kernel_count"] = len(ks)

    # Rows 3 and 4: which Kernel Stats columns come back empty on AMD.
    zero_cols = {}
    for col in KERNEL_ZERO_COLUMNS:
        values = [k[col] for k in ks if k[col] is not None]
        nonzero = [v for v in values if _num(v) != 0]
        zero_cols[col] = {
            "present": bool(values),
            "nonzero_count": len(nonzero),
            "total": len(values),
            "reads_zero": bool(values) and not nonzero,
        }
    report["kernel_stats_columns"] = zero_cols

    # Row 5: device op time summed across devices against wall clock.
    total_s = device_op_time_s(paths)
    report["device_op_time_s"] = total_s
    if wall_seconds:
        report["wall_seconds"] = wall_seconds
        report["device_over_wall"] = total_s / wall_seconds

    # Row 1: the Roofline peak-compute ceiling.
    try:
        rl = roofline(paths)
        peaks = []
        for node in _walk(rl):
            for key, val in node.items():
                if "peak" in str(key).lower() and ("flop" in str(key).lower() or "gflop" in str(key).lower()):
                    peaks.append({key: val})
        report["roofline_peaks"] = peaks or "<no peak fields found>"
    except Exception as exc:
        report["roofline_peaks"] = f"<error: {type(exc).__name__}: {exc}>"

    # The IDLE row.
    try:
        report["op_profile"] = op_profile_summary(paths)
    except Exception as exc:
        report["op_profile"] = f"<error: {type(exc).__name__}: {exc}>"

    # HBM bandwidth as the device plane reports it. Chapter 3 says the device
    # plane reads less than half the data sheet; this is where that is checked.
    try:
        ov = overview(paths)
        bw = []
        for node in _walk(ov):
            for key, val in node.items():
                k = str(key).lower()
                if "bandwidth" in k or "memorybw" in k or "bwutil" in k:
                    bw.append({key: val})
        report["overview_bandwidth_fields"] = bw or "<none>"
        report["device_peaks"] = [
            {k: v for k, v in node.items() if "peak" in str(k).lower()}
            for node in _walk(ov)
            if any("peak" in str(k).lower() for k in node)
        ]
    except Exception as exc:
        report["overview_bandwidth_fields"] = f"<error: {type(exc).__name__}: {exc}>"

    return report


# --------------------------------------------------------------------------
# Trace fallback
# --------------------------------------------------------------------------


def trace_json(target: str | Path) -> dict[str, Any]:
    """Fallback path: read the Chrome trace directly.

    Used when a converter fails, which happens often enough on AMD traces to be
    worth keeping around.
    """
    p = Path(target)
    hits = [p] if p.is_file() else sorted(p.rglob("*.trace.json.gz"))
    if not hits:
        raise SystemExit(f"no *.trace.json.gz under {p}")
    with gzip.open(hits[0], "rt") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _print_kernels(rows: list[dict[str, Any]], limit: int) -> None:
    if not rows:
        print("no kernels found")
        return
    width = max(len(str(r["name"])[:78]) for r in rows[:limit])
    print(f"{'kernel':<{width}}  {'count':>7}  {'total ms':>10}  {'avg us':>10}  {'min us':>10}")
    for r in rows[:limit]:
        print(
            f"{str(r['name'])[:78]:<{width}}  {int(r['occurrences']):>7}  "
            f"{r['total_duration_us'] / 1e3:>10.3f}  {r['avg_duration_us']:>10.2f}  "
            f"{r['min_duration_us']:>10.2f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "view",
        choices=[
            "tools",
            "kernels",
            "collectives",
            "scopes",
            "hlo",
            "overview",
            "ops",
            "roofline",
            "memory",
            "limitations",
            "dump",
        ],
    )
    ap.add_argument("path", help="run directory, trace directory, or .xplane.pb")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--limit", type=int, default=25, help="rows to show")
    ap.add_argument("--wall", type=float, default=None, help="wall seconds, for the 8x check")
    ap.add_argument(
        "--level",
        default="leaf",
        choices=["leaf", "root", "full"],
        help="how to collapse nested scopes, for the scopes view",
    )
    ap.add_argument("--out", default=None, help="also write JSON here")
    args = ap.parse_args(argv)

    paths = find_xplanes(args.path)
    print(f"# {len(paths)} xplane file(s) from {args.path}", file=sys.stderr)

    result: Any
    if args.view == "tools":
        result = available_tools(paths)
    elif args.view == "kernels":
        result = kernels(paths)
    elif args.view == "collectives":
        result = collectives(paths)
    elif args.view == "scopes":
        result = scope_breakdown(paths, level=args.level)
    elif args.view == "hlo":
        result = hlo_stats(paths)
    elif args.view == "overview":
        result = overview(paths)
    elif args.view == "ops":
        result = op_profile(paths)
    elif args.view == "roofline":
        result = roofline(paths)
    elif args.view == "memory":
        result = memory_profile(paths)
    elif args.view == "limitations":
        wall = args.wall
        if wall is None:
            rj = Path(args.path) / "results.json"
            if rj.exists():
                wall = json.loads(rj.read_text()).get("wall_seconds")
        result = limitations(paths, wall_seconds=wall)
    else:
        result = {t: tool_data(paths, t) for t in available_tools(paths)}

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str))
        print(f"# wrote {args.out}", file=sys.stderr)

    if args.json or args.view in {"overview", "ops", "roofline", "memory", "limitations", "tools", "dump"}:
        print(json.dumps(result, indent=2, default=str))
    elif args.view in {"kernels", "collectives"}:
        _print_kernels(result, args.limit)
    elif args.view == "scopes":
        print(f"{'scope':<32}{'share':>8}{'total ms':>11}{'collective ms':>15}{'kernels':>9}")
        for b in result[: args.limit]:
            print(
                f"{b['scope'][:31]:<32}{b['share']:>7.1%}{b['total_duration_us'] / 1e3:>11.2f}"
                f"{b['collective_us'] / 1e3:>15.2f}{b['kernels']:>9}"
            )
    elif args.view == "hlo":
        for r in result[: args.limit]:
            print(f"{r['pct_total_time']:>6.2f}%  {r['total_time_us']:>12.1f} us  {r['category']:<20} {r['hlo_op']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
