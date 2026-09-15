#!/usr/bin/env python3
"""Capture real HLO fixture graphs and render them as SVG snippets.

Every arm runs in a fresh process with XLA's text and DOT dumpers enabled. The
script retains only the named fixture module, its debug options, and provenance;
temporary compiler products are discarded. The SVGs are rendered directly from
XLA's DOT output with Graphviz.

These are real compiler artifacts for small explanatory fixtures. They are not
case-study measurements.

    python tools/capture_hlo_feature_svgs.py
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "hlo-fixtures"
IMAGES = ROOT / "pages" / "img"


@dataclass(frozen=True)
class Fixture:
    feature: str
    variant: str
    module: str
    stage: str
    devices: str = "0"
    extra_flags: tuple[str, ...] = ()
    extra_env: tuple[tuple[str, str], ...] = ()

    @property
    def image_name(self) -> str:
        return f"hlo-{self.feature}-{self.variant}.svg"


FIXTURES = (
    Fixture("scan", "unrolled", "scan_unrolled", "before_optimizations"),
    Fixture("scan", "scanned", "scan_scanned", "before_optimizations"),
    Fixture("remat", "none", "remat_none_loss", "before_optimizations"),
    Fixture("remat", "full", "remat_full_loss", "before_optimizations"),
    Fixture(
        "accumulation",
        "direct",
        "accumulation_direct",
        "before_optimizations",
    ),
    Fixture(
        "accumulation",
        "scanned",
        "accumulation_scanned",
        "before_optimizations",
    ),
    Fixture("sharding", "dp", "sharding_dp", "after_spmd_partitioner", "0,1,2,3,4,5,6,7"),
    Fixture("sharding", "fsdp", "sharding_fsdp", "after_spmd_partitioner", "0,1,2,3,4,5,6,7"),
    Fixture("sharding", "tp", "sharding_tp", "after_spmd_partitioner", "0,1,2,3,4,5,6,7"),
    Fixture(
        "lhs",
        "off",
        "lhs_off",
        "gfx950_gpu_after_optimizations",
        "0,1,2,3,4,5,6,7",
        ("--xla_gpu_enable_latency_hiding_scheduler=false",),
    ),
    Fixture(
        "lhs",
        "on",
        "lhs_on",
        "gfx950_gpu_after_optimizations",
        "0,1,2,3,4,5,6,7",
        ("--xla_gpu_enable_latency_hiding_scheduler=true",),
    ),
    Fixture("attention", "xla", "attention_xla", "before_optimizations"),
    Fixture(
        "attention",
        "te",
        "attention_te",
        "before_optimizations",
        extra_env=(
            ("NVTE_FUSED_ATTN_CK", "1"),
            ("NVTE_FUSED_ATTN_AOTRITON", "0"),
        ),
    ),
    Fixture("moe", "dense-masked", "moe_dense_masked", "before_optimizations"),
    Fixture("moe", "fixed-capacity", "moe_fixed_capacity", "before_optimizations"),
    Fixture(
        "moe",
        "ragged-padded",
        "moe_ragged_padded",
        "gfx950_gpu_after_optimizations",
        extra_flags=(
            "--xla_gpu_enable_cublaslt=true",
            "--xla_gpu_experimental_use_ragged_dot_grouped_gemm=false",
        ),
    ),
    Fixture(
        "moe",
        "ragged-grouped",
        "moe_ragged_grouped",
        "gfx950_gpu_after_optimizations",
        extra_flags=(
            "--xla_gpu_enable_cublaslt=true",
            "--xla_gpu_experimental_use_ragged_dot_grouped_gemm=true",
        ),
    ),
)


def _module_file(temp: Path, fixture: Fixture, extension: str) -> Path:
    suffix = f".{fixture.stage}.{extension}"
    candidates = sorted(
        path
        for path in temp.glob(f"*jit_{fixture.module}*{suffix}")
        if path.is_file()
    )
    if len(candidates) != 1:
        found = "\n".join(str(path.name) for path in candidates) or "(none)"
        raise RuntimeError(
            f"{fixture.feature}/{fixture.variant}: expected one module matching "
            f"*jit_{fixture.module}*{suffix}, found:\n{found}"
        )
    return candidates[0]


def _copy_debug_options(temp: Path, selected: Path, destination: Path) -> None:
    stem = selected.name.split(f".{selected.name.split('.')[-2]}")[0]
    candidates = sorted(temp.glob(f"{stem}.debug_options"))
    if candidates:
        shutil.copy2(candidates[0], destination / "debug_options.txt")


def _scheduled_entry_operations(hlo_text: str) -> list[str]:
    in_entry = False
    operations: list[str] = []
    for line in hlo_text.splitlines():
        if line.startswith("ENTRY "):
            in_entry = True
            continue
        if not in_entry:
            continue
        if line == "}":
            break
        stripped = line.strip()
        if not stripped.startswith(("%", "ROOT %")):
            continue
        if not any(
            marker in stripped
            for marker in ("async-start(", "async-done(", " fusion(")
        ):
            continue
        literal = re.split(r", metadata=|, backend_config=", stripped, maxsplit=1)[0]
        operations.append(literal)
    return operations


def _render_schedule_excerpt(hlo_text: str, svg: Path, title: str) -> None:
    operations = _scheduled_entry_operations(hlo_text)
    if not operations:
        raise RuntimeError(f"{title}: no scheduled operations found")

    lines = [
        "digraph schedule {",
        "rankdir=TB;",
        'graph [bgcolor="white", pad="0.2", nodesep="0.35", ranksep="0.4"];',
        'node [shape=box, style="rounded,filled", fontname="DejaVu Sans Mono", fontsize=10];',
        'edge [color="#666666", penwidth=1.2, arrowsize=0.7, fontname="DejaVu Sans", fontsize=9];',
        f'label=<{html.escape(title)}<br/><font point-size="10">literal operation order from is_scheduled=true HLO</font>>;',
        "labelloc=t;",
    ]
    for index, operation in enumerate(operations):
        if "async-" in operation:
            fill, color = "#ead8c3", "#8c6d46"
        elif "gemm" in operation or "dot_general" in operation:
            fill, color = "#dce9f5", "#3f76ab"
        else:
            fill, color = "#eeeeee", "#777777"
        wrapped = "<br/>".join(
            html.escape(part)
            for part in textwrap.wrap(operation, width=58, break_long_words=False)
        )
        lines.append(
            f'n{index} [label=<{wrapped}>, fillcolor="{fill}", color="{color}"];'
        )
        if index:
            lines.append(f'n{index - 1} -> n{index} [label="schedule"];')
    lines.append("}")

    schedule_dot = svg.with_suffix(".schedule.dot")
    schedule_dot.write_text("\n".join(lines) + "\n")
    subprocess.run(["dot", "-Tsvg", str(schedule_dot), "-o", str(svg)], check=True)
    schedule_dot.unlink()


def capture(fixture: Fixture) -> None:
    destination = ARTIFACTS / fixture.feature / fixture.variant
    destination.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"hlo-{fixture.feature}-{fixture.variant}-") as raw:
        temp = Path(raw)
        flags = (
            f"--xla_dump_to={temp}",
            "--xla_dump_hlo_as_text",
            "--xla_dump_hlo_as_dot",
            *fixture.extra_flags,
        )
        env = os.environ.copy()
        env.update(
            {
                "HIP_VISIBLE_DEVICES": fixture.devices,
                "JAX_PLATFORMS": "rocm",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_FLAGS": " ".join(flags),
            }
        )
        env.update(dict(fixture.extra_env))
        command = [
            shutil.which("python3") or "python3",
            "-m",
            "bench.hlo_feature_fixtures",
            fixture.feature,
            fixture.variant,
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{fixture.feature}/{fixture.variant} failed\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )

        dot_source = _module_file(temp, fixture, "dot")
        text_source = _module_file(temp, fixture, "txt")
        shutil.copy2(dot_source, destination / f"{fixture.stage}.dot")
        shutil.copy2(text_source, destination / f"{fixture.stage}.txt")
        _copy_debug_options(temp, text_source, destination)

        svg = IMAGES / fixture.image_name
        if fixture.feature == "lhs":
            _render_schedule_excerpt(
                text_source.read_text(),
                svg,
                f"jit_{fixture.module}: LHS {fixture.variant}",
            )
            renderer = "literal scheduled-HLO entry excerpt"
        else:
            subprocess.run(
                ["dot", "-Tsvg", str(dot_source), "-o", str(svg)],
                cwd=ROOT,
                check=True,
            )
            renderer = "XLA DOT rendered by Graphviz"

        additional_artifacts: dict[str, str] = {}
        if fixture.feature == "sharding":
            before_fixture = replace(fixture, stage="before_optimizations")
            before_dot = _module_file(temp, before_fixture, "dot")
            before_text = _module_file(temp, before_fixture, "txt")
            shutil.copy2(before_dot, destination / "before_optimizations.dot")
            shutil.copy2(before_text, destination / "before_optimizations.txt")
            if fixture.variant == "fsdp":
                before_svg = IMAGES / f"hlo-sharding-{fixture.variant}-before.svg"
                subprocess.run(
                    ["dot", "-Tsvg", str(before_dot), "-o", str(before_svg)],
                    cwd=ROOT,
                    check=True,
                )
                additional_artifacts["global_before_svg"] = str(
                    before_svg.relative_to(ROOT)
                )

        provenance = {
            "kind": "explanatory-hlo-fixture",
            "measured": False,
            "feature": fixture.feature,
            "variant": fixture.variant,
            "fixture_module": fixture.module,
            "retained_stage": fixture.stage,
            "command": command,
            "environment": {
                "HIP_VISIBLE_DEVICES": fixture.devices,
                "JAX_PLATFORMS": "rocm",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_FLAGS": " ".join(flags),
                **dict(fixture.extra_env),
            },
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "raw_dot": dot_source.name,
            "raw_text": text_source.name,
            "rendered_svg": str(svg.relative_to(ROOT)),
            "renderer": renderer,
            "additional_artifacts": additional_artifacts,
        }
        (destination / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"captured {fixture.feature}/{fixture.variant}: "
            f"{svg.relative_to(ROOT)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixtures",
        nargs="*",
        help="Optional feature/variant filters, for example lhs/on",
    )
    args = parser.parse_args()
    if shutil.which("dot") is None:
        raise SystemExit("Graphviz 'dot' is required")
    selected = FIXTURES
    if args.fixtures:
        wanted = set(args.fixtures)
        selected = tuple(
            fixture
            for fixture in FIXTURES
            if f"{fixture.feature}/{fixture.variant}" in wanted
        )
        missing = wanted - {
            f"{fixture.feature}/{fixture.variant}" for fixture in selected
        }
        if missing:
            raise SystemExit(f"unknown fixture filters: {', '.join(sorted(missing))}")
    for fixture in selected:
        capture(fixture)


if __name__ == "__main__":
    main()
