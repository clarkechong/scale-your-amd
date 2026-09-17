#!/usr/bin/env python3
"""Build Chapter 3's Mixtral attribution ledger from a raw XPlane capture.

The instrumented MaxText source adds metadata-only `jax.named_scope` labels
around the semantic blocks in the chapter's forward and backward maps. This
script classifies every Kernel Stats row exactly once, separates rematerialized
replay from true gradient work, and reconciles summed HLO self-time with raw
per-device compute and communication interval unions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from parse_xplane import COLLECTIVE_RE, _covered, _merge, find_xplanes, kernels


COLLECTIVE_OPS = (
    "all-gather",
    "reduce-scatter",
    "all-reduce",
    "all-to-all",
    "collective-permute",
)
MODEL_PHASES = ("forward", "backward", "remat")
COMMUNICATION_PHASES = (
    "forward_collective",
    "backward_collective",
    "remat_collective",
    "other_collective",
)


def _is_collective(row: dict[str, Any]) -> bool:
    kernel_name = str(row["name"] or "")
    op_name = str(row["op_name"] or "").lower()
    return bool(COLLECTIVE_RE.search(kernel_name)) or any(term in op_name for term in COLLECTIVE_OPS)


def _phase(row: dict[str, Any]) -> str:
    op_name = str(row["op_name"] or "")
    if _is_collective(row):
        if "rematted_computation" in op_name:
            return "remat_collective"
        if "transpose(jvp(model_forward))" in op_name:
            return "backward_collective"
        if "jvp(model_forward)" in op_name:
            return "forward_collective"
        return "other_collective"
    if "rematted_computation" in op_name:
        return "remat"
    if "transpose(jvp(model_forward))" in op_name:
        return "backward"
    if "jvp(model_forward)" in op_name:
        return "forward"
    if "loss_" in op_name:
        return "loss"
    if "optimizer_update" in op_name or "gradient_clipping" in op_name:
        return "optimizer"
    if "gradient_accumulation_update" in op_name:
        return "gradient_accumulation"
    return "unattributed"


def _component(row: dict[str, Any]) -> str:
    op_name = str(row["op_name"] or "").lower()
    named_components = (
        ("embedding", "/embedding/"),
        ("final_norm", "final_norm"),
        ("lm_head", "lm_head"),
        ("pre_attention_norm", "pre_attention_norm"),
        ("post_attention_norm", "post_attention_norm"),
        ("attention_residual", "attention_residual"),
        ("moe_residual", "moe_residual"),
        ("router_gate", "router_gate"),
        ("router_topk", "router_topk"),
        ("router_weights", "router_weights"),
        ("router_masks", "router_masks"),
        ("dispatch", "/dispatch/"),
        ("expert_up", "/wi_0/"),
        ("expert_up", "/wi_1/"),
        ("expert_activation", "ffn_act"),
        ("expert_down", "/wo/"),
        ("combine", "/combine/"),
        ("combine", "weight_sum"),
        ("loss", "loss_"),
        ("optimizer", "optimizer_update"),
        ("gradient_clipping", "gradient_clipping"),
        ("gradient_accumulation", "gradient_accumulation_update"),
    )
    for component, marker in named_components:
        if marker in op_name:
            return component
    if "/attention/" in op_name:
        return "attention"
    if any(
        marker in op_name
        for marker in (
            "dynamic_index_in_dim",
            "dynamic_update_index_in_dim",
            "gradient_accumulation_scan",
            "/while/body",
        )
    ):
        return "scan_and_layout"
    return "other"


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _interval_summary(paths: list[str]) -> dict[str, Any]:
    from xprof.profile_data import ProfileData

    per_device: list[dict[str, Any]] = []
    for path in paths:
        data = ProfileData.from_file(path)
        try:
            for plane in data.planes:
                match = re.search(r"/device:GPU:(\d+)", plane.name or "")
                if not match:
                    continue
                compute: list[tuple[float, float]] = []
                communication: list[tuple[float, float]] = []
                for line in plane.lines:
                    if not (line.name or "").startswith("Stream"):
                        continue
                    for event in line.events:
                        interval = (event.start_ns, event.start_ns + event.duration_ns)
                        target = (
                            communication
                            if re.search(r"nccl|rccl", event.name or "", re.IGNORECASE)
                            else compute
                        )
                        target.append(interval)

                compute_union = _merge(compute)
                communication_union = _merge(communication)
                busy_union = _merge(compute + communication)
                compute_ns = sum(end - start for start, end in compute_union)
                communication_ns = sum(end - start for start, end in communication_union)
                hidden_ns = sum(_covered(interval, compute_union) for interval in communication_union)
                busy_ns = sum(end - start for start, end in busy_union)
                event_span_ns = (
                    max(end for _, end in busy_union) - min(start for start, _ in busy_union)
                    if busy_union
                    else 0.0
                )
                per_device.append(
                    {
                        "gpu": int(match.group(1)),
                        "compute_union_s": compute_ns / 1e9,
                        "communication_union_s": communication_ns / 1e9,
                        "communication_hidden_s": hidden_ns / 1e9,
                        "communication_exposed_s": (communication_ns - hidden_ns) / 1e9,
                        "busy_union_s": busy_ns / 1e9,
                        "event_span_s": event_span_ns / 1e9,
                        "idle_within_event_span_s": (event_span_ns - busy_ns) / 1e9,
                    }
                )
        finally:
            data.close()

    if not per_device:
        raise ValueError("the XPlane contains no GPU stream events")
    metric_names = [name for name in per_device[0] if name != "gpu"]
    return {
        "per_device": sorted(per_device, key=lambda row: row["gpu"]),
        "mean": {name: _mean([row[name] for row in per_device]) for name in metric_names},
        "min": {name: min(row[name] for row in per_device) for name in metric_names},
        "max": {name: max(row[name] for row in per_device) for name in metric_names},
    }


def summarize(
    target: Path,
    provenance_path: Path | None,
    archive_path: Path | None,
) -> dict[str, Any]:
    paths = find_xplanes(target)
    rows = kernels(paths)
    device_ids = set()
    interval_summary = _interval_summary(paths)
    for row in interval_summary["per_device"]:
        device_ids.add(row["gpu"])
    devices = len(device_ids)

    buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"time_ms": 0.0, "occurrences_per_gpu": 0.0}
    )
    for row in rows:
        key = (_phase(row), _component(row))
        buckets[key]["time_ms"] += row["total_duration_us"] / devices / 1000.0
        buckets[key]["occurrences_per_gpu"] += row["occurrences"] / devices

    def aggregate(phase: str, *components: str) -> dict[str, float]:
        return {
            "time_ms": sum(buckets[(phase, component)]["time_ms"] for component in components),
            "occurrences_per_gpu": sum(
                buckets[(phase, component)]["occurrences_per_gpu"] for component in components
            ),
        }

    component_names = sorted({component for _, component in buckets})
    by_phase = {
        phase: {
            component: buckets[(phase, component)]
            for component in component_names
            if buckets[(phase, component)]["time_ms"]
        }
        for phase in sorted({phase for phase, _ in buckets})
    }
    phase_summary_ms = {
        phase: sum(bucket["time_ms"] for bucket in components.values())
        for phase, components in by_phase.items()
    }

    forward_map = {
        "embedding": aggregate("forward", "embedding"),
        "pre_attention_norm": aggregate("forward", "pre_attention_norm"),
        "attention": aggregate("forward", "attention"),
        "attention_residual": aggregate("forward", "attention_residual"),
        "post_attention_norm": aggregate("forward", "post_attention_norm"),
        "router": aggregate(
            "forward",
            "router_gate",
            "router_topk",
            "router_weights",
            "router_masks",
        ),
        "dispatch": aggregate("forward", "dispatch"),
        "expert_mlp": aggregate(
            "forward",
            "expert_up",
            "expert_activation",
            "expert_down",
        ),
        "combine": aggregate("forward", "combine"),
        "moe_residual": aggregate("forward", "moe_residual"),
        "scan_and_layout": aggregate("forward", "scan_and_layout", "other"),
        "final_norm": aggregate("forward", "final_norm"),
        "lm_head": aggregate("forward", "lm_head"),
    }
    backward_map = {
        "embedding_vjp": aggregate("backward", "embedding"),
        "lm_head_vjp": aggregate("backward", "lm_head"),
        "final_norm_vjp": aggregate("backward", "final_norm"),
        "attention_backward": aggregate("backward", "attention", "attention_residual"),
        "layer_norm_vjps": aggregate(
            "backward",
            "pre_attention_norm",
            "post_attention_norm",
        ),
        "moe_backward": aggregate(
            "backward",
            "router_gate",
            "router_topk",
            "router_weights",
            "router_masks",
            "dispatch",
            "expert_up",
            "expert_activation",
            "expert_down",
            "combine",
            "moe_residual",
        ),
        "scan_and_layout": aggregate("backward", "scan_and_layout", "other"),
    }
    remat_map = {
        "attention_replay": aggregate(
            "remat",
            "attention",
            "attention_residual",
        ),
        "layer_norm_replay": aggregate(
            "remat",
            "pre_attention_norm",
            "post_attention_norm",
        ),
        "moe_replay": aggregate(
            "remat",
            "router_gate",
            "router_topk",
            "router_weights",
            "router_masks",
            "dispatch",
            "expert_up",
            "expert_activation",
            "expert_down",
            "combine",
            "moe_residual",
        ),
        "scan_and_layout": aggregate("remat", "scan_and_layout", "other"),
    }
    communication = {
        phase: {
            "time_ms": phase_summary_ms.get(phase, 0.0),
            "components": by_phase.get(phase, {}),
        }
        for phase in COMMUNICATION_PHASES
    }

    checks = (
        (sum(row["time_ms"] for row in forward_map.values()), phase_summary_ms["forward"]),
        (sum(row["time_ms"] for row in backward_map.values()), phase_summary_ms["backward"]),
        (sum(row["time_ms"] for row in remat_map.values()), phase_summary_ms["remat"]),
        (
            sum(row["time_ms"] for row in communication.values()),
            sum(phase_summary_ms.get(phase, 0.0) for phase in COMMUNICATION_PHASES),
        ),
    )
    if not all(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9) for actual, expected in checks):
        raise ValueError("component buckets do not reconcile with phase totals")

    provenance = json.loads(provenance_path.read_text()) if provenance_path else {}
    total_self_time_ms = sum(phase_summary_ms.values())
    return {
        "capture": {
            "model": "Mixtral 8x22B",
            "accelerators": "8 x MI355X",
            "maxtext_revision": provenance.get("maxtext_revision"),
            "observed_profiled_step": provenance.get("observed_profiled_step"),
            "reported_step_seconds": provenance.get("step_times_seconds", {}).get(
                str(provenance.get("observed_profiled_step"))
            ),
            "command": provenance.get("command"),
            "instrumentation_artifact_sha256": provenance.get("artifact_sha256"),
        },
        "provenance": (
            {
                "path": str(provenance_path),
                "sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            }
            if provenance_path
            else None
        ),
        "sources": [
            {
                "path": path,
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for path in paths
        ],
        "portable_xplane_archive": (
            {
                "path": str(archive_path),
                "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            }
            if archive_path
            else None
        ),
        "time_basis": (
            "component tables use mean summed Kernel Stats HLO self-time per GPU-step; "
            "interval summary uses unions of raw stream events"
        ),
        "device_count": devices,
        "kernel_stats_rows": len(rows),
        "total_summed_self_time_ms": total_self_time_ms,
        "phase_summary_ms": phase_summary_ms,
        "phase_summary_share": {
            phase: time_ms / total_self_time_ms for phase, time_ms in phase_summary_ms.items()
        },
        "forward_map": forward_map,
        "backward_map": backward_map,
        "remat_map": remat_map,
        "communication": communication,
        "interval_summary": interval_summary,
        "notes": [
            "Each Kernel Stats row is assigned to one phase and one component.",
            "Parameter-gradient kernels remain in their originating attention, MoE, norm, embedding, or head VJP.",
            "Residual additions that disappeared into adjacent fusions have no standalone duration.",
            "Summed component self-times are not wall time; use the raw interval unions for overlap reconciliation.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xplane", type=Path, help="XPlane file or directory containing it")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--archive", type=Path, help="compressed XPlane retained with the book")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = summarize(args.xplane, args.provenance, args.archive)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
