#!/usr/bin/env python3
"""Generate the custom diagrams still needed by Chapter 3.

The generic roofline and XSpace figures come from cited external sources.
This script draws the ROCm profiler flow and Mixtral attribution maps, then
renders the retained literal XLA DOT fixture with Graphviz.

Run from the repository root:

    python tools/make_ch3_profiling_figures.py
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pages" / "img"

INK = "#27313b"
MUTED = "#65717d"
LINE = "#8a959e"
FRAMEWORK = ("#e8eff5", "#496a82")
XLA = FRAMEWORK
ROCM = FRAMEWORK
EVIDENCE = ("#f1f3f4", "#7a858e")
EXECUTION = EVIDENCE
NEUTRAL = ("#f5f7f8", "#8a959e")


def box(
    ax,
    cx: float,
    cy: float,
    width: float,
    height: float,
    text: str,
    palette: tuple[str, str] = NEUTRAL,
    *,
    fontsize: float = 8.5,
    weight: str = "normal",
    linestyle: str | tuple = "-",
    zorder: int = 3,
) -> None:
    face, edge = palette
    ax.add_patch(
        FancyBboxPatch(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            boxstyle="round,pad=0.035,rounding_size=0.09",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.1,
            linestyle=linestyle,
            zorder=zorder,
        )
    )
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=INK,
        linespacing=1.25,
        zorder=zorder + 1,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    dashed: bool = False,
    connectionstyle: str = "arc3",
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            color=color,
            linewidth=1.05,
            linestyle=(0, (4, 3)) if dashed else "-",
            connectionstyle=connectionstyle,
            shrinkA=3,
            shrinkB=3,
            zorder=zorder,
        )
    )


def panel(ax, x: float, y: float, width: float, height: float, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.045,rounding_size=0.11",
            facecolor="#fcfcfc",
            edgecolor="#c9d0d6",
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(
        x + 0.18,
        y + height - 0.28,
        title,
        ha="left",
        va="center",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )


def title(ax, text: str, subtitle: str = "") -> None:
    ax.text(0.25, 9.05, text, fontsize=15, weight="bold", color=INK, ha="left")
    if subtitle:
        ax.text(0.25, 8.72, subtitle, fontsize=9.2, color=MUTED, ha="left")


def save(fig, ax, name: str, *, xlim=(0, 14), ylim=(0.25, 9.35)) -> Path:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def jax_profiler_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(14.0, 4.6))
    ax.text(
        0.25,
        4.15,
        "How ROCm events become an XSpace profile",
        fontsize=15,
        weight="bold",
        color=INK,
        ha="left",
    )
    ax.text(
        0.25,
        3.82,
        "OpenXLA separates event capture from organization and storage.",
        fontsize=9.2,
        color=MUTED,
        ha="left",
    )

    box(
        ax,
        1.45,
        2.55,
        2.35,
        0.72,
        "ROCprofiler-SDK\nHIP · kernels · copies",
        NEUTRAL,
        fontsize=8.4,
    )
    box(
        ax,
        1.45,
        1.35,
        2.35,
        0.72,
        "JAX / XLA annotations\nnames · correlation IDs",
        NEUTRAL,
        fontsize=8.4,
    )
    box(
        ax,
        4.35,
        1.95,
        2.35,
        0.92,
        "RocmTracer\nreceives event records",
        FRAMEWORK,
        fontsize=9.0,
        weight="bold",
    )
    box(
        ax,
        7.35,
        1.95,
        2.45,
        0.92,
        "RocmTraceCollector\nbuilds XPlanes",
        FRAMEWORK,
        fontsize=9.0,
        weight="bold",
    )
    box(
        ax,
        10.25,
        1.95,
        2.10,
        0.92,
        "XSpace\nhost + GPU planes",
        FRAMEWORK,
        fontsize=9.0,
        weight="bold",
    )
    box(
        ax,
        12.75,
        1.95,
        1.65,
        0.92,
        "XProf\nviews",
        EVIDENCE,
        fontsize=9.0,
        weight="bold",
    )

    arrow(ax, (2.64, 2.43), (3.17, 2.12))
    arrow(ax, (2.64, 1.47), (3.17, 1.78))
    arrow(ax, (5.55, 1.95), (6.10, 1.95))
    arrow(ax, (8.59, 1.95), (9.18, 1.95))
    arrow(ax, (11.32, 1.95), (11.91, 1.95))

    ax.text(
        7.35,
        0.55,
        "Tracer: capture records    Collector: normalize and organize    XSpace: serialize the profile",
        ha="center",
        fontsize=8.7,
        color=MUTED,
    )
    return save(
        fig,
        ax,
        "ch3-jax-profiler-pipeline.png",
        xlim=(0, 14),
        ylim=(0.2, 4.45),
    )


def mixtral_forward() -> Path:
    fig, ax = plt.subplots(figsize=(9.6, 12.4))
    ax.text(
        0.35,
        13.05,
        "Mixtral 8x22B forward attribution map",
        fontsize=15,
        weight="bold",
        color=INK,
        ha="left",
    )
    ax.text(
        0.35,
        12.68,
        "Expand one decoder layer, then repeat it 56 times.",
        fontsize=9.5,
        color=MUTED,
        ha="left",
    )

    box(
        ax,
        5.0,
        11.92,
        8.6,
        0.62,
        "Step context: embedding  →  decoder layer × 56  →  final norm  →  LM head  →  loss",
        NEUTRAL,
        fontsize=9.0,
        weight="bold",
    )
    panel(ax, 0.65, 1.05, 8.7, 10.25, "One MixtralDecoderLayer")

    nodes = [
        (10.35, "Layer input  h", NEUTRAL, 2.8, 0.56),
        (9.45, "Pre-attention RMSNorm", NEUTRAL, 3.2, 0.56),
        (
            8.20,
            "Attention\nQ/K/V projections  →  RoPE + fused core  →  output projection\nscope: self_attention",
            FRAMEWORK,
            6.6,
            1.02,
        ),
        (6.92, "Residual add with h", NEUTRAL, 3.3, 0.58),
        (5.98, "Post-attention RMSNorm", NEUTRAL, 3.2, 0.56),
        (
            4.88,
            "Router and top-2 selection\ngate logits  →  expert indices + route weights\nscope: MoeBlock_0/gate",
            FRAMEWORK,
            5.6,
            0.80,
        ),
        (
            3.80,
            "Dispatch / token permutation\nscope: dispatch",
            FRAMEWORK,
            4.4,
            0.66,
        ),
        (
            2.62,
            "Expert MLP\nwi_0 + wi_1  →  SiLU × gate  →  wo\nscopes: wi_0 · wi_1 · ffn_act · wo",
            FRAMEWORK,
            5.9,
            0.92,
        ),
        (
            1.48,
            "Combine, restore token order, and add residual\nscopes: combine · weight_sum",
            FRAMEWORK,
            5.4,
            0.70,
        ),
    ]

    cx = 5.0
    for cy, text, palette, width, height in nodes:
        box(ax, cx, cy, width, height, text, palette, fontsize=9.0)
    for current, following in zip(nodes, nodes[1:]):
        current_y, _, _, _, current_h = current
        next_y, _, _, _, next_h = following
        arrow(
            ax,
            (cx, current_y - current_h / 2),
            (cx, next_y + next_h / 2),
        )

    ax.text(
        0.95,
        0.45,
        "The labels inside each compute block are the names to search in HLO and XProf.",
        fontsize=8.6,
        color=MUTED,
        ha="left",
    )
    return save(
        fig,
        ax,
        "ch3-mixtral-forward.png",
        xlim=(0, 10),
        ylim=(0.2, 13.35),
    )


def mixtral_backward() -> Path:
    fig, ax = plt.subplots(figsize=(10.2, 11.2))
    ax.text(
        0.35,
        11.55,
        "Mixtral 8x22B backward attribution map",
        fontsize=15,
        weight="bold",
        color=INK,
        ha="left",
    )
    ax.text(
        0.35,
        11.18,
        "Follow activation gradients down the main path and parameter gradients along the side rail.",
        fontsize=9.3,
        color=MUTED,
        ha="left",
    )

    box(
        ax,
        5.0,
        10.42,
        8.8,
        0.62,
        "Step context: loss VJP  →  LM-head VJP  →  final-norm VJP  →  decoder-layer VJP × 56",
        NEUTRAL,
        fontsize=8.8,
        weight="bold",
    )
    panel(ax, 0.55, 0.85, 6.15, 8.95, "One decoder-layer VJP")
    panel(ax, 7.05, 2.05, 2.40, 6.55, "Parameter path")

    main_nodes = [
        (8.95, "d(layer output)", NEUTRAL, 2.75, 0.56),
        (8.02, "Residual gradient split\nidentity + MoE branch", NEUTRAL, 3.45, 0.68),
        (
            6.70,
            "MoE backward\ncombine VJP  →  expert GEMM VJPs  →  reverse dispatch\nrouter gradient · scopes: combine, wo, wi_0, wi_1, gate",
            FRAMEWORK,
            5.30,
            1.18,
        ),
        (5.35, "Post-attention RMSNorm VJP", NEUTRAL, 3.50, 0.60),
        (4.37, "Residual gradient split\nidentity + attention branch", NEUTRAL, 3.60, 0.68),
        (
            2.98,
            "Attention backward\noutput-projection VJP  →  fused attention dQ/dK/dV\n→  Q/K/V projection VJPs · scope: self_attention",
            FRAMEWORK,
            5.30,
            1.20,
        ),
        (1.58, "Pre-attention RMSNorm VJP\n→  d(layer input)", NEUTRAL, 3.65, 0.72),
    ]

    main_x = 3.62
    for cy, text, palette, width, height in main_nodes:
        box(ax, main_x, cy, width, height, text, palette, fontsize=8.8)
    for current, following in zip(main_nodes, main_nodes[1:]):
        current_y, _, _, _, current_h = current
        next_y, _, _, _, next_h = following
        arrow(
            ax,
            (main_x, current_y - current_h / 2),
            (main_x, next_y + next_h / 2),
        )

    side_x = 8.25
    box(
        ax,
        side_x,
        7.30,
        1.92,
        0.98,
        "Parameter gradients\nLM head · attention\nrouter · experts · norms",
        FRAMEWORK,
        fontsize=8.3,
        weight="bold",
    )
    box(
        ax,
        side_x,
        5.55,
        1.92,
        0.84,
        "Gradient collective\nReduceScatter / AllReduce",
        FRAMEWORK,
        fontsize=8.3,
    )
    box(
        ax,
        side_x,
        3.88,
        1.92,
        0.90,
        "Accumulate × 2\nthen AdamW once",
        NEUTRAL,
        fontsize=8.5,
        weight="bold",
    )
    arrow(ax, (side_x, 6.80), (side_x, 5.98))
    arrow(ax, (side_x, 5.12), (side_x, 4.34))
    arrow(
        ax,
        (6.30, 6.70),
        (7.26, 7.08),
        dashed=True,
        connectionstyle="arc3,rad=-0.12",
    )
    arrow(
        ax,
        (6.30, 2.98),
        (7.26, 6.95),
        dashed=True,
        connectionstyle="arc3,rad=-0.18",
    )

    ax.text(
        0.85,
        0.35,
        "Solid arrows follow activation gradients. Dashed arrows collect parameter-gradient leaves.",
        fontsize=8.5,
        color=MUTED,
        ha="left",
    )
    return save(
        fig,
        ax,
        "ch3-mixtral-backward.png",
        xlim=(0, 10),
        ylim=(0.15, 11.85),
    )


def render_hlo_fixture() -> Path:
    dot = (
        ROOT
        / "artifacts"
        / "hlo-fixtures"
        / "moe"
        / "ragged-grouped"
        / "gfx950_gpu_after_optimizations.dot"
    )
    provenance = dot.with_name("provenance.json")
    if not dot.exists() or not provenance.exists():
        raise FileNotFoundError("The retained ragged-grouped HLO fixture is incomplete")
    graphviz = shutil.which("dot")
    if graphviz is None:
        raise RuntimeError("Graphviz 'dot' is required to render the HLO fixture")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ch3-hlo-ragged-grouped.svg"
    render_dot = re.sub(
        r', tooltip=".*?", style=',
        ', tooltip=" ", style=',
        dot.read_text(),
        flags=re.DOTALL,
    )
    subprocess.run(
        [graphviz, "-Tsvg", "-o", str(path)],
        input=render_dot,
        text=True,
        check=True,
    )
    return path


if __name__ == "__main__":
    outputs = [
        jax_profiler_pipeline(),
        mixtral_forward(),
        mixtral_backward(),
        render_hlo_fixture(),
    ]
    for output in outputs:
        print(f"  wrote {output.relative_to(ROOT)}")
