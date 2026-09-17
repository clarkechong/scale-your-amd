#!/usr/bin/env python3
"""Generate the original diagrams and HLO excerpts for Chapter 5.

The PNGs are explanatory, hand-placed diagrams. The SVG labels are pruned from
literal XLA DOT nodes under ``artifacts/hlo-fixtures/sharding``; dependency
arrows contract paths through omitted copy or tuple nodes. Graphviz performs
the SVG rendering, and this script does not synthesize HLO operations.

    python tools/make_ch5_sharding_figures.py
"""

from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pages" / "img"
HLO = ROOT / "artifacts" / "hlo-fixtures" / "sharding"

FRAMEWORK = ("#dce9f5", "#3f76ab")
COMPILER = ("#dff0e4", "#4e8a5c")
COMM = ("#f7e6c8", "#b0842f")
DEVICE = ("#f9d8d6", "#bf5b57")
NEUTRAL = ("#f3f3f3", "#777777")
INK = "#3f3f3f"


def rbox(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    palette=NEUTRAL,
    *,
    fontsize: float = 8.5,
    weight: str = "normal",
    zorder: int = 3,
) -> None:
    """Draw a rounded box centered at ``x, y``."""
    fill, edge = palette
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.0,
            zorder=zorder,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=INK,
        zorder=zorder + 1,
    )


def arrow(ax, start, end, *, color="#666666", label="", label_offset=(0, 0.12)) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color=color,
            zorder=2,
        )
    )
    if label:
        x = (start[0] + end[0]) / 2 + label_offset[0]
        y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(x, y, label, ha="center", va="bottom", fontsize=7.5, color=color)


def _striped_matrix(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    rows: int,
    label: str,
) -> None:
    colors = [
        "#dce9f5",
        "#e7eef6",
        "#f7e6c8",
        "#f9eeda",
        "#dff0e4",
        "#e9f4ec",
        "#e9e3f2",
        "#f1edf6",
    ]
    row_h = height / rows
    for row in range(rows):
        ax.add_patch(
            Rectangle(
                (x - width / 2, y - height / 2 + row * row_h),
                width,
                row_h,
                facecolor=colors[row % len(colors)],
                edgecolor="white",
                linewidth=0.7,
            )
        )
        ax.text(
            x - width / 2 + 0.10,
            y - height / 2 + (row + 0.5) * row_h,
            str(row),
            ha="left",
            va="center",
            fontsize=6.5,
            color="#666666",
        )
    ax.add_patch(
        Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            facecolor="none",
            edgecolor=FRAMEWORK[1],
            linewidth=1.1,
        )
    )
    ax.text(x, y + height / 2 + 0.20, label, ha="center", va="bottom", fontsize=8.5)


def global_to_local_arrays() -> Path:
    """Show one global FSDP matmul and the physical buffers on a rank."""
    fig, ax = plt.subplots(figsize=(11.5, 5.0))

    _striped_matrix(
        ax,
        1.35,
        3.15,
        1.65,
        2.20,
        8,
        "global x  f16[1024,512]\nP('mesh', None)",
    )
    _striped_matrix(
        ax,
        3.65,
        3.15,
        1.65,
        2.20,
        8,
        "global w  f16[512,512]\nP('mesh', None)",
    )

    ax.text(
        2.50,
        0.86,
        "JAX reports the global shapes.\nEach color is one device's row shard.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#555555",
    )

    rbox(ax, 6.45, 4.00, 2.10, 0.75, "x shard\nf16[128,512]", DEVICE)
    rbox(ax, 6.45, 2.88, 2.10, 0.75, "w shard\nf16[64,512]", DEVICE)
    ax.text(
        6.45, 4.63, "one device", ha="center", va="bottom", fontsize=9.0, weight="bold"
    )

    rbox(ax, 9.20, 2.88, 2.25, 0.85, "AllGather(mesh)\nw  f16[512,512]", COMM)
    rbox(ax, 9.20, 4.00, 2.25, 0.75, "local dot\nf16[128,512]", COMPILER)
    rbox(ax, 11.55, 4.00, 1.65, 0.75, "y shard\nf16[128,512]", DEVICE)

    arrow(ax, (4.55, 3.45), (5.35, 3.90), label="select row 3")
    arrow(ax, (4.55, 2.85), (5.35, 2.90), label="select row 3", label_offset=(0, -0.30))
    arrow(ax, (7.52, 2.88), (8.05, 2.88), color=COMM[1])
    arrow(ax, (9.20, 3.32), (9.20, 3.60), color=COMM[1])
    arrow(ax, (7.52, 4.00), (8.05, 4.00), color=COMPILER[1])
    arrow(ax, (10.35, 4.00), (10.70, 4.00), color=COMPILER[1])

    ax.text(
        9.20,
        1.67,
        "The compiler changes global operations into\n"
        "device-local shapes plus communication.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#555555",
    )

    ax.set_xlim(0.20, 12.55)
    ax.set_ylim(0.25, 5.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Global array semantics and device-local execution", loc="left", fontsize=11
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ch5-global-to-local-arrays.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def fsdp_ep_meshes() -> Path:
    """Draw the four logical FSDP/EP meshes that use one eight-GPU node."""
    meshes = ((1, 8), (2, 4), (4, 2), (8, 1))
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.8))
    ep_colors = [
        "#dce9f5",
        "#f7e6c8",
        "#dff0e4",
        "#e9e3f2",
        "#f9d8d6",
        "#e5edf0",
        "#f4e2c1",
        "#e6eed9",
    ]

    for ax, (fsdp, ep) in zip(axes, meshes):
        cell = min(0.82 / max(ep, 1), 0.82 / max(fsdp, 1))
        grid_w = ep * cell
        grid_h = fsdp * cell
        x0 = 0.52 - grid_w / 2
        y0 = 0.52 - grid_h / 2
        rank = 0
        for f in range(fsdp):
            for e in range(ep):
                ax.add_patch(
                    FancyBboxPatch(
                        (x0 + e * cell, y0 + (fsdp - 1 - f) * cell),
                        cell * 0.90,
                        cell * 0.90,
                        boxstyle="round,pad=0,rounding_size=0.02",
                        facecolor=ep_colors[e],
                        edgecolor=FRAMEWORK[1],
                        linewidth=0.8,
                    )
                )
                label = str(rank) if cell > 0.11 else f"{rank}"
                ax.text(
                    x0 + (e + 0.45) * cell,
                    y0 + (fsdp - 1 - f + 0.45) * cell,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                )
                rank += 1

        ax.annotate(
            "",
            xy=(x0 + grid_w, y0 - 0.075),
            xytext=(x0, y0 - 0.075),
            arrowprops={"arrowstyle": "-|>", "color": COMM[1], "linewidth": 1.0},
        )
        ax.text(
            0.52,
            y0 - 0.14,
            "expert",
            ha="center",
            va="top",
            fontsize=7.5,
            color=COMM[1],
        )
        ax.annotate(
            "",
            xy=(x0 - 0.075, y0),
            xytext=(x0 - 0.075, y0 + grid_h),
            arrowprops={"arrowstyle": "-|>", "color": COMPILER[1], "linewidth": 1.0},
        )
        ax.text(
            x0 - 0.13,
            0.52,
            "fsdp",
            ha="right",
            va="center",
            rotation=90,
            fontsize=7.5,
            color=COMPILER[1],
        )
        ax.set_title(f"FSDP={fsdp}, EP={ep}\nshape ({fsdp}, {ep})", fontsize=9.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        "Four logical meshes, the same eight physical MI355X devices",
        x=0.04,
        ha="left",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.035,
        "Grid adjacency is a logical coordinate system. It does not replace or describe the UBB's physical full mesh.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.84, bottom=0.15, wspace=0.10)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ch5-fsdp-ep-mesh-cells.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def maxtext_sharding_flow() -> Path:
    """Trace MaxText configuration into compiler-visible shardings."""
    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    columns = (
        (2.0, "configuration", FRAMEWORK),
        (5.7, "mesh and array layout", COMPILER),
        (9.4, "compilation", COMM),
    )
    for x, label, palette in columns:
        ax.add_patch(
            FancyBboxPatch(
                (x - 1.55, 0.45),
                3.10,
                6.15,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                facecolor=palette[0],
                edgecolor=palette[1],
                linewidth=0.8,
                alpha=0.24,
                zorder=0,
            )
        )
        ax.text(
            x,
            6.36,
            label,
            ha="center",
            va="center",
            fontsize=9.0,
            weight="bold",
            color=palette[1],
        )

    rbox(
        ax,
        2.0,
        5.55,
        2.55,
        0.82,
        "base.yml + model YAML\nici_fsdp_parallelism\nici_expert_parallelism",
        FRAMEWORK,
        fontsize=8,
    )
    rbox(
        ax,
        2.0,
        4.15,
        2.55,
        0.82,
        "configs/types.py\nparse, validate,\nresolve -1 axes",
        FRAMEWORK,
        fontsize=8,
    )
    rbox(
        ax,
        2.0,
        2.75,
        2.55,
        0.82,
        "ICI vector per slice\nDCN vector across slices",
        FRAMEWORK,
        fontsize=8,
    )

    rbox(
        ax,
        5.7,
        5.55,
        2.70,
        0.82,
        "create_device_mesh\ncreate_device_mesh or\ncreate_hybrid_device_mesh",
        COMPILER,
        fontsize=8,
    )
    rbox(
        ax,
        5.7,
        4.15,
        2.70,
        0.82,
        "Mesh(devices, mesh_axes)\n'fsdp', 'expert', …",
        COMPILER,
        fontsize=8,
    )
    rbox(
        ax,
        5.7,
        2.75,
        2.70,
        0.95,
        "layer logical names\n('exp', 'embed_moe',\n'mlp_moe')",
        COMPILER,
        fontsize=8,
    )
    rbox(
        ax,
        5.7,
        1.28,
        2.70,
        0.95,
        "logical_axis_rules\n→ PartitionSpec\n→ NamedSharding tree",
        COMPILER,
        fontsize=8,
    )

    rbox(
        ax,
        9.4,
        4.85,
        2.55,
        0.88,
        "jax.jit\nin_shardings\nout_shardings",
        COMM,
        fontsize=8,
    )
    rbox(
        ax,
        9.4,
        3.05,
        2.55,
        0.88,
        "Shardy propagation\nexplicit reshards\npartitioning",
        COMM,
        fontsize=8,
    )
    rbox(
        ax,
        9.4,
        1.28,
        2.55,
        0.88,
        "device-local HLO\ncollectives → RCCL\nGPU kernels",
        DEVICE,
        fontsize=8,
    )

    arrow(ax, (2.0, 5.12), (2.0, 4.58), color=FRAMEWORK[1])
    arrow(ax, (2.0, 3.72), (2.0, 3.18), color=FRAMEWORK[1])
    arrow(ax, (3.30, 2.75), (4.30, 5.30), color=COMPILER[1])
    arrow(ax, (5.7, 5.12), (5.7, 4.58), color=COMPILER[1])
    arrow(ax, (5.7, 3.72), (5.7, 3.23), color=COMPILER[1])
    arrow(ax, (5.7, 2.26), (5.7, 1.76), color=COMPILER[1])
    arrow(ax, (7.08, 1.45), (8.12, 4.55), color=COMM[1])
    arrow(ax, (7.08, 4.15), (8.12, 4.65), color=COMM[1])
    arrow(ax, (9.4, 4.40), (9.4, 3.50), color=COMM[1])
    arrow(ax, (9.4, 2.60), (9.4, 1.73), color=DEVICE[1])

    ax.text(
        2.0,
        1.12,
        "axis sizes",
        ha="center",
        va="center",
        fontsize=7.5,
        color=FRAMEWORK[1],
    )
    ax.text(
        7.62,
        4.53,
        "mesh",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=COMM[1],
    )
    ax.text(
        7.57,
        2.45,
        "array shardings",
        ha="center",
        va="bottom",
        rotation=72,
        fontsize=7.5,
        color=COMM[1],
    )

    ax.set_xlim(0.25, 11.15)
    ax.set_ylim(0.25, 6.85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "MaxText v26.6: configuration to device-local program", loc="left", fontsize=11
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ch5-maxtext-sharding-flow.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def _dot_nodes_and_edges(dot_text: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Extract literal XLA node statements and numeric edges."""
    lines = dot_text.splitlines()
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        node_match = re.match(r"^\s*(\d+) \[label=", line)
        if node_match:
            statement = [line]
            while not statement[-1].rstrip().endswith("];"):
                index += 1
                statement.append(lines[index])
            nodes[node_match.group(1)] = "\n".join(statement)
        else:
            edge_match = re.match(r"^\s*(\d+) -> (\d+) ", line)
            if edge_match:
                edges.append((edge_match.group(1), edge_match.group(2)))
        index += 1
    return nodes, edges


def _node_name(statement: str) -> str | None:
    match = re.search(r"<b>([^<]+)</b>", statement)
    return match.group(1) if match else None


def _clean_node(statement: str) -> str:
    """Remove source-stack tooltips while preserving XLA's visible node label."""
    statement = re.sub(
        r', tooltip=".*?", style=',
        ', tooltip=" ", style=',
        statement,
        flags=re.DOTALL,
    )
    # Retain the literal SDY value while dropping two redundant wrapper names
    # that make the before-partitioning parameters twice as wide on the page.
    statement = re.sub(
        r"frontend_attributes=\{xla\.sdy\.sharding=&quot;"
        r"#sdy\.sharding&lt;(.*?)&gt;&quot;\}",
        r"sdy.sharding=&lt;\1&gt;",
        statement,
    )
    statement = re.sub(
        r"frontend_attributes=\{xla\.sdy\.sharding=&quot;"
        r"#sdy\.sharding_per_value&lt;\[&lt;(.*?)&gt;\]&gt;&quot;\}",
        r"result sharding=&lt;\1&gt;",
        statement,
    )
    statement = statement.replace("<br/>custom_call_has_side_effect=true", "")
    return statement


def _contract_edges(
    selected: set[str],
    edges: list[tuple[str, str]],
) -> set[tuple[str, str]]:
    """Connect retained literal nodes across omitted layout-only nodes."""
    adjacency: dict[str, list[str]] = {}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)

    contracted: set[tuple[str, str]] = set()
    for source in selected:
        queue = list(adjacency.get(source, ()))
        visited: set[str] = set()
        while queue:
            target = queue.pop(0)
            if target in visited:
                continue
            visited.add(target)
            if target in selected:
                contracted.add((source, target))
            else:
                queue.extend(adjacency.get(target, ()))
    return contracted


def render_hlo_excerpt(
    source: Path,
    destination: Path,
    wanted_names: set[str],
    *,
    title: str,
    stage: str,
) -> Path:
    """Render selected literal nodes from one XLA DOT graph."""
    nodes, edges = _dot_nodes_and_edges(source.read_text())
    selected = {
        node_id: statement
        for node_id, statement in nodes.items()
        if _node_name(statement) in wanted_names
    }
    found = {_node_name(statement) for statement in selected.values()}
    missing = wanted_names - found
    if missing:
        raise RuntimeError(f"{source}: missing HLO nodes {sorted(missing)}")

    lines = [
        "digraph G {",
        "rankdir=TB;",
        'graph [bgcolor="white", pad="0.2", nodesep="0.25", ranksep="0.38"];',
        'node [fontname="Roboto", fontsize=11];',
        'edge [color="#666666", penwidth=1.2, arrowsize=0.7];',
        (
            f'label=<{title}<br/><font point-size="10">'
            f"{stage}; representative subgraph from literal XLA DOT"
            "</font>>;"
        ),
        "labelloc=t;",
    ]
    for node_id in sorted(selected, key=int):
        lines.append(_clean_node(selected[node_id]))
    for source_id, target_id in sorted(_contract_edges(set(selected), edges)):
        lines.append(f"{source_id} -> {target_id};")
    lines.append("}")

    completed = subprocess.run(
        ["dot", "-Tsvg", "-o", str(destination)],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Graphviz failed for {source}:\n{completed.stderr}")
    return destination


def hlo_figures() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = [
        render_hlo_excerpt(
            HLO / "fsdp" / "before_optimizations.dot",
            OUT / "ch5-hlo-fsdp-before.svg",
            {"Parameter 0", "Parameter 1", "dot_general.2", "dot_general.3"},
            title="FSDP-style matmul: global program",
            stage="before_optimizations",
        ),
        render_hlo_excerpt(
            HLO / "fsdp" / "after_spmd_partitioner.dot",
            OUT / "ch5-hlo-fsdp-after.svg",
            {"Parameter 0", "Parameter 1", "all-gather", "dot"},
            title="FSDP-style matmul: device-local program",
            stage="after_spmd_partitioner",
        ),
        render_hlo_excerpt(
            HLO / "tp" / "before_optimizations.dot",
            OUT / "ch5-hlo-tp-before.svg",
            {"dot_general.1", "psum_invariant.5"},
            title="TP-style manual body before downstream partitioning",
            stage="before_optimizations",
        ),
        render_hlo_excerpt(
            HLO / "tp" / "after_spmd_partitioner.dot",
            OUT / "ch5-hlo-tp-after.svg",
            {"Parameter 0", "Parameter 1", "dot_general.3", "psum_invariant.7"},
            title="TP-style manual computation: device-local program",
            stage="after_spmd_partitioner",
        ),
    ]
    return outputs


def main() -> None:
    outputs = [
        global_to_local_arrays(),
        fsdp_ep_meshes(),
        maxtext_sharding_flow(),
        *hlo_figures(),
    ]
    for output in outputs:
        print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
