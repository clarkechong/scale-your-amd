#!/usr/bin/env python3
"""Generate the mechanism and literal-HLO figures for Chapter 6.

The mechanism figures are original explanatory drawings. HLO figures are
Graphviz-rendered excerpts of checked-in literal gfx950 HLO fixtures. Every HLO
node label is asserted against the retained text before rendering.

Run from the repository root:

    python tools/make_ch6_figures.py
"""

from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pages" / "img"
HLO = ROOT / "artifacts" / "hlo-fixtures"
PMC = Path("/tmp/archive/llama7b/rocprof-pmc-20260910-1layer")

INK = "#27313b"
MUTED = "#65717d"
LINE = "#8a959e"
BLUE = ("#dce9f5", "#3f76ab")
GREEN = ("#dff0e4", "#4e8a5c")
GOLD = ("#f7e6c8", "#b0842f")
PURPLE = ("#e9e3f2", "#6d5b9e")
RED = ("#f9d8d6", "#bf5b57")
GRAY = ("#f5f7f8", "#8a959e")


def box(ax, x, y, w, h, text, palette=GRAY, fontsize=8.2, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.07",
            facecolor=palette[0],
            edgecolor=palette[1],
            linewidth=1.1,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=INK, weight=weight)


def arrow(
    ax,
    x0,
    y0,
    x1,
    y1,
    *,
    color=LINE,
    dashed=False,
    label=None,
    connectionstyle="arc3",
):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=10,
            color=color,
            linewidth=1.05,
            linestyle=(0, (4, 3)) if dashed else "-",
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.13, label, ha="center", fontsize=7.2, color=MUTED)


def panel(ax, x, y, w, h, title):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            facecolor="#fcfcfc",
            edgecolor="#c9d0d6",
            linewidth=1,
            zorder=-2,
        )
    )
    ax.text(x + 0.16, y + h - 0.28, title, ha="left", va="center", fontsize=10.2, weight="bold", color=INK)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path.relative_to(ROOT))


def remat_mechanism():
    fig, ax = plt.subplots(figsize=(14, 6.1))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.1)
    ax.axis("off")
    ax.text(0.2, 5.8, "What crosses the forward–backward boundary", fontsize=14, weight="bold", color=INK)
    ax.text(
        0.2,
        5.48,
        "A policy saves selected residuals in HBM; backward recomputes the omitted prerequisites.",
        fontsize=9.2,
        color=MUTED,
    )

    rows = [
        ("No explicit remat", 4.45, "all AD residuals", RED, "little deliberate replay"),
        ("Named selective policy", 2.9, "named dot outputs", GOLD, "replay omitted operations"),
        ("Full remat", 1.35, "layer input only", GREEN, "replay complete layer body"),
    ]
    for title, y, saved, palette, replay in rows:
        ax.text(0.25, y + 0.47, title, fontsize=9.5, weight="bold", color=INK)
        box(ax, 1.25, y, 1.45, 0.62, "layer input", BLUE)
        box(ax, 3.35, y, 1.8, 0.62, "attention + MLP", PURPLE)
        box(ax, 5.45, y, 1.45, 0.62, "layer output", BLUE)
        arrow(ax, 1.99, y, 2.42, y)
        arrow(ax, 4.27, y, 4.71, y)

        ax.plot([6.55, 6.55], [y - 0.58, y + 0.58], color="#b9c1c8", linewidth=1.2)
        ax.text(6.55, y + 0.72, "AD boundary", ha="center", fontsize=7.3, color=MUTED)
        box(ax, 8.25, y, 2.15, 0.7, f"HBM checkpoint\n{saved}", palette)
        arrow(ax, 5.99, y, 7.15, y, label="save")
        box(ax, 11.05, y, 2.1, 0.7, f"backward VJP\n{replay}", PURPLE)
        arrow(ax, 9.34, y, 9.97, y)
        box(ax, 13.1, y, 1.45, 0.62, "gradients", BLUE)
        arrow(ax, 12.12, y, 12.35, y)

        if replay.startswith("replay"):
            arrow(ax, 8.25, y - 0.37, 10.45, y - 0.37, color=palette[1], dashed=True, label="recompute")

    ax.text(
        7.0,
        0.35,
        "Saved bytes depend on local sharding. Replayed FLOPs and collectives depend on what the rematerialized region contains.",
        ha="center",
        fontsize=8.6,
        color=MUTED,
    )
    save(fig, "ch6-remat-saved-vs-recomputed.png")


def attention_mechanism():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(0.2, 6.68, "Attention forward and backward dependencies", fontsize=14, weight="bold", color=INK)

    panel(ax, 0.2, 3.72, 13.6, 2.5, "Forward")
    boxes = [
        (1.15, "Q, K", BLUE),
        (3.15, r"$S=QK^T/\sqrt{d}$", PURPLE),
        (5.35, "causal mask", GRAY),
        (7.35, r"$P=\mathrm{softmax}(S)$", GOLD),
        (9.65, r"$O=PV$", PURPLE),
        (12.25, "O + row LSE", GREEN),
    ]
    for x, text, palette in boxes:
        box(ax, x, 4.77, 1.72 if x not in (7.35, 12.25) else 1.95, 0.68, text, palette)
    for a, b in zip(boxes, boxes[1:]):
        arrow(ax, a[0] + 0.9, 4.77, b[0] - 0.9, 4.77)
    box(ax, 9.65, 5.65, 1.2, 0.48, "V", BLUE)
    arrow(ax, 9.65, 5.39, 9.65, 5.13)

    panel(ax, 0.2, 0.55, 13.6, 2.8, "Backward")
    box(ax, 1.0, 1.82, 1.25, 0.62, "dO", BLUE)
    box(ax, 3.55, 2.65, 1.85, 0.62, r"$dV=P^T dO$", PURPLE)
    box(ax, 3.55, 1.82, 1.85, 0.72, r"$dP=dO\,V^T$", PURPLE)
    box(ax, 6.15, 1.82, 2.35, 0.82, r"$dS=P\odot(dP-\Sigma(dP\odot P))$", GOLD, fontsize=7.7)
    box(ax, 9.15, 2.35, 1.7, 0.64, r"$dQ=dS\,K$", PURPLE)
    box(ax, 9.15, 1.28, 1.7, 0.64, r"$dK=dS^TQ$", PURPLE)
    box(ax, 13.05, 1.82, 1.25, 0.62, "dQ,dK,dV", GREEN)
    arrow(ax, 1.64, 1.98, 2.62, 2.55)
    arrow(ax, 1.64, 1.82, 2.61, 1.82)
    arrow(ax, 4.48, 1.82, 4.96, 1.82)
    arrow(ax, 7.34, 1.96, 8.29, 2.24)
    arrow(ax, 7.34, 1.68, 8.29, 1.37)
    arrow(ax, 10.02, 2.35, 12.41, 1.99)
    arrow(ax, 10.02, 1.28, 12.41, 1.64)
    arrow(
        ax,
        4.48,
        2.68,
        12.41,
        2.06,
        connectionstyle="arc3,rad=-0.16",
    )
    ax.text(
        7.0,
        0.83,
        "Standard attention can save P or S. Flash-style kernels save compact row statistics and reconstruct tiles during backward.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "ch6-attention-forward-backward.png")


def _trace_rows(backend):
    path = PMC / f"attention-{backend}-compute_kernel_trace.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _one(rows, needle):
    matches = [row for row in rows if needle in row["Kernel_Name"]]
    if len(matches) != 1:
        raise RuntimeError(f"expected one kernel containing {needle!r}, got {len(matches)}")
    return matches[0]


def _geometry(row):
    grid = "×".join(row[f"Grid_Size_{axis}"] for axis in "XYZ")
    workgroup = "×".join(row[f"Workgroup_Size_{axis}"] for axis in "XYZ")
    return f"grid {grid} · WG {workgroup}\nLDS {int(row['LDS_Block_Size']) // 1024} KiB · VGPR {row['VGPR_Count']} + Acc {row['Accum_VGPR_Count']}"


def observed_attention_kernels():
    te = _trace_rows("te")
    aiter = _trace_rows("aiter")
    triton = _trace_rows("triton")
    routes = [
        (
            "Transformer Engine → CK/AITER",
            [
                ("forward", _one(te, "aiter::fmha_fwd_hd128_bf16_causal"), "aiter::fmha_fwd_hd128_bf16_causal"),
                ("backward helper", _one(te, "aiter::fmha_bwd_hd128_odo_bf16"), "aiter::fmha_bwd_hd128_odo_bf16"),
                ("backward main", _one(te, "aiter::fmha_bwd_hd128_bf16_causal"), "aiter::fmha_bwd_hd128_bf16_causal_a32_psskddv"),
                ("gradient convert", _one(te, "aiter::fmha_bwd_hd128_dq_convert_bf16"), "aiter::fmha_bwd_hd128_dq_convert_bf16"),
            ],
        ),
        (
            "Direct JAX-AITER FFI",
            [
                ("forward", _one(aiter, "aiter::fmha_fwd_hd128_bf16_causal"), "aiter::fmha_fwd_hd128_bf16_causal"),
                ("backward helper", _one(aiter, "FmhaBwdOGradDotOKernel"), "ck_tile::FmhaBwdOGradDotOKernel"),
                ("backward main", _one(aiter, "FmhaBwdDQDKDVKernel"), "ck_tile::FmhaBwdDQDKDVKernel"),
                ("gradient convert", _one(aiter, "FmhaBwdConvertQGradKernel"), "ck_tile::FmhaBwdConvertQGradKernel"),
            ],
        ),
        (
            "Tokamax Pallas–Triton",
            [
                (
                    "forward",
                    [row for row in triton if row["Kernel_Name"] == "pallas_flash_attention"][0],
                    "pallas_flash_attention",
                ),
                ("backward setup", _one(triton, "pallas_flash_attention_fwd_res"), "pallas_flash_attention_fwd_res"),
                ("backward", _one(triton, "pallas_flash_attention_vjp"), "pallas_flash_attention_vjp"),
            ],
        ),
    ]

    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.text(0.2, 6.88, "Observed production-shape attention kernel geometry", fontsize=14, weight="bold", color=INK)
    ax.text(
        0.2,
        6.55,
        "BF16, B=4, S=4096, 32 heads, head dimension 128; one-layer selected-step rocprof capture.",
        fontsize=9.2,
        color=MUTED,
    )
    colors = (GREEN, GOLD, PURPLE)
    for column, ((title, kernels), palette) in enumerate(zip(routes, colors)):
        x0 = 0.25 + 4.55 * column
        panel(ax, x0, 0.75, 4.25, 5.35, title)
        for index, (phase, row, name) in enumerate(kernels):
            y = 4.95 - 1.18 * index
            ax.text(x0 + 0.25, y + 0.42, phase, fontsize=7.7, weight="bold", color=MUTED)
            box(ax, x0 + 2.12, y, 3.65, 0.78, f"{name}\n{_geometry(row)}", palette, fontsize=6.8)
            if index:
                arrow(ax, x0 + 2.12, y + 0.76, x0 + 2.12, y + 0.42)
    ax.text(
        7,
        0.25,
        "Kernel names and launch metadata are literal CSV fields. They describe the selected shape, not a backend-wide fixed tile.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "ch6-attention-observed-kernels.png")


def moe_routing():
    fig, ax = plt.subplots(figsize=(14, 6.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.text(0.2, 6.3, "Uneven routing creates four different matrix workloads", fontsize=14, weight="bold", color=INK)
    ax.text(0.2, 5.98, "Illustrative 12-assignment layer: expert counts [6, 3, 1, 2].", fontsize=9.2, color=MUTED)

    sections = [
        (0.25, "Dense masked", "12 rows × every expert", RED),
        (3.7, "Fixed capacity C=6", "pad to 24 slots", GOLD),
        (7.15, "Ragged, dense-padded", "runtime groups → padded batch", PURPLE),
        (10.6, "Ragged, grouped GEMM", "groups [6,3,1,2]", GREEN),
    ]
    counts = [6, 3, 1, 2]
    for x0, title, subtitle, palette in sections:
        panel(ax, x0, 0.75, 3.15, 4.75, title)
        ax.text(x0 + 1.575, 4.9, subtitle, ha="center", fontsize=8.2, color=MUTED)
        for expert, count in enumerate(counts):
            y = 4.25 - expert * 0.82
            ax.text(x0 + 0.2, y, f"E{expert}", ha="left", va="center", fontsize=8, color=INK)
            if "Dense masked" in title:
                used, total = 12, 12
            elif "Fixed" in title:
                used, total = count, 6
            elif "dense-padded" in title:
                used, total = count, 8
            else:
                used, total = count, count
            cell_w = min(0.18, 2.35 / max(total, 1))
            for cell in range(total):
                ax.add_patch(
                    Rectangle(
                        (x0 + 0.62 + cell * cell_w, y - 0.16),
                        cell_w - 0.01,
                        0.32,
                        facecolor=palette[0] if cell < used else "#eceff1",
                        edgecolor=palette[1] if cell < used else "#b6bec5",
                        linewidth=0.65,
                    )
                )
        if "grouped" in title:
            ax.text(x0 + 1.58, 1.15, "one runtime M dimension per expert", ha="center", fontsize=7.6, color=MUTED)
        elif "dense-padded" in title:
            ax.text(x0 + 1.58, 1.15, "mask + batched dense dot + reduce", ha="center", fontsize=7.6, color=MUTED)
        elif "Fixed" in title:
            ax.text(x0 + 1.58, 1.15, "unused slots still occupy the tensor", ha="center", fontsize=7.6, color=MUTED)
        else:
            ax.text(x0 + 1.58, 1.15, "regular, but every expert executes", ha="center", fontsize=7.6, color=MUTED)
    ax.text(
        7,
        0.25,
        "The grouped path removes model-level capacity padding; MFMA tiles can still have shape tails.",
        ha="center",
        fontsize=8.6,
        color=MUTED,
    )
    save(fig, "ch6-moe-routing-and-padding.png")


@dataclass(frozen=True)
class HloNode:
    name: str
    shape: str
    op: str
    palette: tuple[str, str] = GRAY
    detail: str = ""


def _assert_nodes(source: Path, nodes: list[HloNode]):
    text = source.read_text()
    for node in nodes:
        pattern = rf"%(?:{re.escape(node.name)})\s*=\s*{re.escape(node.shape)}\s+{re.escape(node.op)}"
        if not re.search(pattern, text):
            raise RuntimeError(f"{source}: literal HLO node not found: {node}")


def _dot_label(node: HloNode):
    detail = f"<br/><font point-size='9'>{node.detail}</font>" if node.detail else ""
    return (
        f"<<b>{node.name}</b><br/>{node.op}<br/>"
        f"<font face='DejaVu Sans Mono' point-size='9'>{node.shape}</font>{detail}>"
    )


def _render_hlo(name, title, clusters, edges, *, rankdir="LR"):
    dot = [
        "digraph G {",
        f"rankdir={rankdir};",
        'graph [bgcolor="white", pad="0.2", nodesep="0.28", ranksep="0.48"];',
        'node [shape=box, style="rounded,filled", fontname="DejaVu Sans", fontsize=10];',
        'edge [color="#6f7880", penwidth=1.1, arrowsize=0.7];',
        f'label=<{title}<br/><font point-size="9">pruned from literal gfx950 HLO; names and shapes retained</font>>;',
        "labelloc=t;",
    ]
    seen = {}
    for cluster_index, (cluster_title, source, nodes) in enumerate(clusters):
        _assert_nodes(source, nodes)
        dot.append(f"subgraph cluster_{cluster_index} {{")
        dot.append(f'label="{cluster_title}"; color="#c9d0d6"; style="rounded";')
        for node in nodes:
            node_id = f"n{len(seen)}"
            seen[(cluster_index, node.name)] = node_id
            dot.append(
                f'{node_id} [label={_dot_label(node)}, fillcolor="{node.palette[0]}", '
                f'color="{node.palette[1]}"];'
            )
        dot.append("}")
    for c0, n0, c1, n1, label, dashed in edges:
        attrs = [f'label="{label}"'] if label else []
        if dashed:
            attrs.append('style="dashed"')
        dot.append(f"{seen[(c0, n0)]} -> {seen[(c1, n1)]} [{', '.join(attrs)}];")
    dot.append("}")
    dot_path = OUT / f"{name}.dot"
    svg_path = OUT / f"{name}.svg"
    dot_path.write_text("\n".join(dot) + "\n")
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    dot_path.unlink()
    print(svg_path.relative_to(ROOT))


def remat_hlo():
    none = HLO / "remat" / "none" / "before_optimizations.txt"
    full = HLO / "remat" / "full" / "before_optimizations.txt"
    clusters = [
        (
            "none: forward residual is reused",
            none,
            [
                HloNode("dot_general.6", "bf16[64,128]{1,0}", "dot(", BLUE),
                HloNode("tanh.3", "bf16[64,128]{1,0}", "tanh(", GREEN, "forward residual"),
                HloNode("sub.3", "bf16[64,128]{1,0}", "subtract(", GOLD),
                HloNode("mul.3", "bf16[64,128]{1,0}", "multiply(", PURPLE),
                HloNode("add_any.3", "bf16[64,128]{1,0}", "add(", PURPLE),
                HloNode("dot_general.7", "bf16[128,128]{1,0}", "dot(", BLUE, "weight gradient"),
            ],
        ),
        (
            "full: backward reconstructs the residual",
            full,
            [
                HloNode(
                    "remat2.10",
                    "(bf16[64,128]{1,0}, bf16[128,128]{1,0}, bf16[64,128]{1,0})",
                    "tuple(",
                    GREEN,
                    "checkpoint tuple",
                ),
                HloNode(
                    "remat2.11",
                    "(bf16[64,128]{1,0}, bf16[128,128]{1,0}, bf16[64,128]{1,0})",
                    "opt-barrier(",
                    GREEN,
                ),
                HloNode("dot_general.9", "bf16[64,128]{1,0}", "dot(", RED, "checkpoint/rematted_computation"),
                HloNode("tanh.6", "bf16[64,128]{1,0}", "tanh(", RED, "checkpoint/rematted_computation"),
                HloNode("add_any.3", "bf16[64,128]{1,0}", "add(", PURPLE),
                HloNode("dot_general.10", "bf16[128,128]{1,0}", "dot(", BLUE, "weight gradient"),
            ],
        ),
    ]
    edges = [
        (0, "dot_general.6", 0, "tanh.3", "", False),
        (0, "tanh.3", 0, "sub.3", "saved", False),
        (0, "sub.3", 0, "mul.3", "", False),
        (0, "mul.3", 0, "add_any.3", "", False),
        (0, "add_any.3", 0, "dot_general.7", "", False),
        (1, "remat2.10", 1, "remat2.11", "", False),
        (1, "remat2.11", 1, "dot_general.9", "", False),
        (1, "dot_general.9", 1, "tanh.6", "recompute", True),
        (1, "tanh.6", 1, "add_any.3", "", False),
        (1, "add_any.3", 1, "dot_general.10", "", False),
    ]
    _render_hlo(
        "ch6-hlo-remat-none-full",
        "Rematerialization changes backward dataflow",
        clusters,
        edges,
        rankdir="TB",
    )


def moe_hlo():
    dense = HLO / "moe" / "dense-masked" / "before_optimizations.txt"
    fixed = HLO / "moe" / "fixed-capacity" / "before_optimizations.txt"
    clusters = [
        (
            "dense masked",
            dense,
            [
                HloNode("tokens.1", "bf16[64,128]{1,0}", "parameter(", BLUE),
                HloNode("experts.1", "bf16[4,128,128]{2,1,0}", "parameter(", BLUE),
                HloNode("dot_general.2", "bf16[64,4,128]{2,1,0}", "dot(", RED, "all experts"),
                HloNode("routing.1", "bf16[64,4]{1,0}", "parameter(", GOLD),
                HloNode("dot_general.3", "bf16[64,128]{1,0}", "dot(", GREEN, "combine"),
            ],
        ),
        (
            "fixed capacity",
            fixed,
            [
                HloNode("tokens.1", "bf16[64,128]{1,0}", "parameter(", BLUE),
                HloNode("dispatch.1", "bf16[64,4,16]{2,1,0}", "parameter(", GOLD),
                HloNode("dot_general.3", "bf16[4,16,128]{2,1,0}", "dot(", PURPLE, "dispatch"),
                HloNode("experts.1", "bf16[4,128,128]{2,1,0}", "parameter(", BLUE),
                HloNode("dot_general.4", "bf16[4,16,128]{2,1,0}", "dot(", RED, "expert GEMM"),
                HloNode("dot_general.5", "bf16[64,128]{1,0}", "dot(", GREEN, "combine"),
            ],
        ),
    ]
    edges = [
        (0, "tokens.1", 0, "dot_general.2", "", False),
        (0, "experts.1", 0, "dot_general.2", "", False),
        (0, "dot_general.2", 0, "dot_general.3", "", False),
        (0, "routing.1", 0, "dot_general.3", "", False),
        (1, "tokens.1", 1, "dot_general.3", "", False),
        (1, "dispatch.1", 1, "dot_general.3", "", False),
        (1, "dot_general.3", 1, "dot_general.4", "", False),
        (1, "experts.1", 1, "dot_general.4", "", False),
        (1, "dot_general.4", 1, "dot_general.5", "", False),
        (1, "dispatch.1", 1, "dot_general.5", "", False),
    ]
    _render_hlo("ch6-hlo-moe-dense-fixed", "Dense and fixed-capacity expert execution", clusters, edges)

    padded = HLO / "moe" / "ragged-padded" / "gfx950_gpu_after_optimizations.txt"
    grouped = HLO / "moe" / "ragged-grouped" / "gfx950_gpu_after_optimizations.txt"
    clusters = [
        (
            "ragged_dot → dense-padded fusions",
            padded,
            [
                HloNode("tokens.1", "f16[64,128]{1,0}", "parameter(", BLUE),
                HloNode("group_sizes.1", "s32[4]{0}", "parameter(", GOLD),
                HloNode("loop_transpose_fusion", "f16[64,4,128]{2,1,0}", "fusion(", PURPLE, "mask and expand"),
                HloNode("gemm_fusion_dot.2", "f32[8,64,128]{2,1,0}", "fusion(", RED, "__triton_nested_gemm_fusion"),
                HloNode("loop_reduce_fusion", "f16[64,128]{1,0}", "fusion(", GREEN, "reduce groups"),
            ],
        ),
        (
            "ragged_dot → hipBLASLt grouped GEMM",
            grouped,
            [
                HloNode("tokens.1", "f16[64,128]{1,0}", "parameter(", BLUE),
                HloNode("experts.1", "f16[4,128,128]{2,1,0}", "parameter(", BLUE),
                HloNode("group_sizes.1", "s32[4]{0}", "parameter(", GOLD),
                HloNode(
                    "custom-call.1",
                    "(f16[64,128]{1,0}, s8[784]{0})",
                    "custom-call(",
                    RED,
                    'custom_call_target="__cublas$lt$groupedMatmul"',
                ),
                HloNode("get-tuple-element.1", "f16[64,128]{1,0}", "get-tuple-element(", GREEN),
            ],
        ),
    ]
    edges = [
        (0, "tokens.1", 0, "loop_transpose_fusion", "", False),
        (0, "group_sizes.1", 0, "loop_transpose_fusion", "", False),
        (0, "loop_transpose_fusion", 0, "gemm_fusion_dot.2", "", False),
        (0, "gemm_fusion_dot.2", 0, "loop_reduce_fusion", "", False),
        (1, "tokens.1", 1, "custom-call.1", "", False),
        (1, "experts.1", 1, "custom-call.1", "", False),
        (1, "group_sizes.1", 1, "custom-call.1", "", False),
        (1, "custom-call.1", 1, "get-tuple-element.1", "", False),
    ]
    _render_hlo("ch6-hlo-moe-ragged-lowerings", "The same ragged frontend, two gfx950 lowerings", clusters, edges)


def attention_hlo():
    for variant in ("xla", "te"):
        source = HLO / "attention" / variant / "representative.dot"
        destination = OUT / f"ch6-hlo-attention-{variant}.svg"
        subprocess.run(["dot", "-Tsvg", str(source), "-o", str(destination)], check=True)
        print(destination.relative_to(ROOT))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    remat_mechanism()
    attention_mechanism()
    observed_attention_kernels()
    moe_routing()
    remat_hlo()
    moe_hlo()
    attention_hlo()


if __name__ == "__main__":
    main()
