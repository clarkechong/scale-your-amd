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
from matplotlib.font_manager import FontProperties, fontManager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

for _font in (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
):
    fontManager.addfont(_font)

# Liberation Sans is the Arial-metric face available on this host.
ARIAL = FontProperties(fname="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
ARIAL_BOLD = FontProperties(fname="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")

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
    """Vertical jax.profiler → XSpace → XProf pipeline, white on black."""
    white = "#ffffff"
    black = "#000000"
    fig, ax = plt.subplots(figsize=(5.8, 11.2), facecolor=black)
    ax.set_facecolor(black)

    def node(cx, cy, w, h, text, *, fontsize=11.5, weight="normal"):
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="square,pad=0",
                facecolor=black,
                edgecolor=white,
                linewidth=1.35,
                zorder=3,
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
            color=white,
            linespacing=1.35,
            family="DejaVu Sans",
            parse_math=False,
            zorder=4,
        )

    def down_arrow(y0, y1, label=""):
        ax.annotate(
            "",
            xy=(0, y1),
            xytext=(0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=white,
                lw=1.25,
                mutation_scale=12,
            ),
            zorder=2,
        )
        if label:
            ax.text(
                0.28,
                (y0 + y1) / 2,
                label,
                ha="left",
                va="center",
                fontsize=9.5,
                color=white,
                family="DejaVu Sans",
                parse_math=False,
                zorder=4,
            )

    w = 3.85
    cx = 0.0
    node(cx, 9.55, w, 0.95, "jax.profiler", fontsize=12.5, weight="bold")
    down_arrow(9.07, 8.48)
    node(cx, 7.90, w, 1.15, "XLA Profiler\nBackend", fontsize=12.0, weight="bold")
    down_arrow(7.32, 6.62, "populates")
    node(cx, 6.00, w, 1.15, "XSpace (.pb)\nTrace Schema", fontsize=12.0, weight="bold")
    down_arrow(5.42, 4.72, "parsed by")
    node(cx, 4.10, w, 1.15, "XProf\nParser + Backend", fontsize=12.0, weight="bold")
    down_arrow(3.52, 2.92)
    node(
        cx,
        1.70,
        w,
        2.35,
        "XProf\nTimeline\nRoofline\nKernel Statistics\nFramework Ops",
        fontsize=11.5,
        weight="bold",
    )

    ax.set_xlim(-2.55, 2.75)
    ax.set_ylim(0.25, 10.25)
    ax.axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ch3-jax-profiler-pipeline.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.28, facecolor=black)
    plt.close(fig)
    return path


def rocm_profiler_pipeline() -> Path:
    """Horizontal ROCprofiler-SDK to XProf frontend pipeline, white on black."""
    white = "#ffffff"
    black = "#000000"
    dim = "#a5abb3"
    faint = "#484d52"

    stages = [
        ("ROCprofiler-SDK", "HIP API · kernel dispatch\nmemory copies"),
        ("RocmTracer", "receives runtime\nevent records"),
        ("RocmTraceCollector", "normalizes timestamps\nbuilds XPlanes"),
        ("XSpace (.pb)", "host and GPU planes\nlines · events · stats"),
        ("XProf parser", "reads the\ntrace schema"),
        ("XProf database", "derived per-event\nand per-op tables"),
        ("XProf frontend", "timeline · roofline\nkernel and op views"),
    ]
    groups = [
        (0, 0, "ROCm"),
        (1, 2, "XLA profiler backend"),
        (3, 3, "trace format"),
        (4, 6, "XProf"),
    ]

    width, pitch = 3.30, 5.05
    height, mid = 1.30, 2.00
    centers = [width / 2 + 0.35 + i * pitch for i in range(len(stages))]

    fig, ax = plt.subplots(figsize=(21.0, 3.6), facecolor=black)
    ax.set_facecolor(black)
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    def text(x, y, body, *, size, color, weight="normal", ha="center", va="center"):
        ax.text(
            x,
            y,
            body,
            ha=ha,
            va=va,
            fontsize=size,
            weight=weight,
            color=color,
            linespacing=1.35,
            fontproperties=ARIAL_BOLD if weight == "bold" else ARIAL,
            parse_math=False,
            zorder=5,
        )

    for start, end, label in groups:
        x0 = centers[start] - width / 2 - 0.30
        x1 = centers[end] + width / 2 + 0.30
        ax.add_patch(
            FancyBboxPatch(
                (x0, mid - height / 2 - 0.34),
                x1 - x0,
                height + 0.68,
                boxstyle="round,pad=0,rounding_size=0.10",
                facecolor=black,
                edgecolor=faint,
                linewidth=1.0,
                linestyle=(0, (5, 4)),
                zorder=1,
            )
        )
        text(
            (x0 + x1) / 2,
            mid + height / 2 + 0.60,
            label.upper(),
            size=9.5,
            color=white,
        )

    for (title_text, subtitle), cx in zip(stages, centers):
        ax.add_patch(
            FancyBboxPatch(
                (cx - width / 2, mid - height / 2),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor="#0d0d0d",
                edgecolor=white,
                linewidth=1.45,
                zorder=3,
            )
        )
        text(cx, mid + 0.26, title_text, size=12.5, color=white, weight="bold")
        text(cx, mid - 0.25, subtitle, size=9.0, color=dim)

    for cx in centers[:-1]:
        x0, x1 = cx + width / 2 + 0.12, cx + pitch - width / 2 - 0.12
        ax.annotate(
            "",
            xy=(x1, mid),
            xytext=(x0, mid),
            arrowprops=dict(arrowstyle="-|>", color=white, lw=1.3, mutation_scale=13),
            zorder=4,
        )

    ax.set_xlim(0, centers[-1] + width / 2 + 0.55)
    ax.set_ylim(mid - height / 2 - 0.55, mid + height / 2 + 0.95)
    ax.axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ch3-rocm-profiler-pipeline.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.30, facecolor=black)
    plt.close(fig)
    return path


def python_to_kernels() -> Path:
    """Python train_step to JAX ops and kernels, white on black."""
    white = "#ffffff"
    black = "#000000"
    dim = "#c4c8cc"
    fill = "#0d0d0d"

    fig, ax = plt.subplots(figsize=(11.4, 9.4), facecolor=black)
    ax.set_facecolor(black)
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    def text(x, y, body, *, size, color, weight="normal"):
        ax.text(
            x,
            y,
            body,
            ha="center",
            va="center",
            fontsize=size,
            weight=weight,
            color=color,
            linespacing=1.2,
            fontproperties=ARIAL_BOLD if weight == "bold" else ARIAL,
            parse_math=False,
            zorder=5,
        )

    def node(cx, cy, w, h, title_text, subtitle=""):
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.07",
                facecolor=fill,
                edgecolor=white,
                linewidth=1.35,
                zorder=3,
            )
        )
        if subtitle:
            text(cx, cy + 0.15, title_text, size=11.5, color=white, weight="bold")
            text(cx, cy - 0.16, subtitle, size=9.5, color=dim)
        else:
            text(cx, cy, title_text, size=11.5, color=white, weight="bold")

    def v_arrow(x, y0, y1):
        ax.annotate(
            "",
            xy=(x, y1),
            xytext=(x, y0),
            arrowprops=dict(arrowstyle="-|>", color=white, lw=1.25, mutation_scale=12),
            zorder=4,
        )

    def stem(x0, y0, x1, y1):
        ax.plot([x0, x1], [y0, y1], color=white, lw=1.25, solid_capstyle="butt", zorder=2)

    mid = 6.0
    top_w, top_h = 3.35, 0.82
    op_w, op_h = 1.72, 0.78
    k_w, k_h = 1.42, 0.48
    op_xs = (2.40, 6.00, 9.60)
    k_spread = 0.92
    k_y = 3.28

    node(mid, 7.20, top_w, top_h, "Python function", "train_step()")
    v_arrow(mid, 6.79, 6.33)
    text(mid + 0.48, 6.56, "jax.jit", size=9.5, color=dim)
    node(mid, 5.92, top_w, top_h, "JAX module", "compiled computation")

    split_y = 5.17
    stem(mid, 5.51, mid, split_y)
    stem(op_xs[0], split_y, op_xs[2], split_y)
    for x in op_xs:
        v_arrow(x, split_y, 4.73)

    ops = (("JAX op", "dot"), ("JAX op", "softmax"), ("JAX op", "rms_norm"))
    op_y = 4.34
    for x, (title_text, subtitle) in zip(op_xs, ops):
        node(x, op_y, op_w, op_h, title_text, subtitle)

    kernel_groups = [
        (op_xs[0], [op_xs[0] - k_spread, op_xs[0] + k_spread], ["Kernel A", "Kernel B"]),
        (op_xs[1], [op_xs[1] - k_spread, op_xs[1] + k_spread], ["Kernel C", "Kernel D"]),
        (op_xs[2], [op_xs[2]], ["Kernel E"]),
    ]
    k_split = 3.74
    for parent_x, xs, names in kernel_groups:
        stem(parent_x, op_y - op_h / 2, parent_x, k_split)
        if len(xs) > 1:
            stem(xs[0], k_split, xs[-1], k_split)
        for x in xs:
            v_arrow(x, k_split, k_y + k_h / 2)
        for x, name in zip(xs, names):
            node(x, k_y, k_w, k_h, name)

    row2_y = k_y - 0.92
    row3_y = row2_y - (k_h + (0.92 - k_h) / 3)
    text(mid, (k_y + row2_y) / 2, "...", size=16, color=white)
    extra_rows = [
        (
            row2_y,
            [
                (1.48, 1.42, "Kernel 1"),
                (3.28, 1.10, "Kernel 2"),
                (6.00, 3.10, "Kernel 3"),
                (8.42, 1.20, "Kernel 4"),
                (10.05, 1.30, "Kernel 5"),
            ],
        ),
        (
            row3_y,
            [
                (2.40, 2.20, "Kernel X"),
                (5.20, 1.20, "Kernel Y"),
                (7.55, 2.40, "Kernel Z"),
                (9.95, 1.30, "Kernel K"),
            ],
        ),
    ]
    for cy, boxes in extra_rows:
        for cx, w, name in boxes:
            node(cx, cy, w, k_h, name)

    ax.set_xlim(0.4, 11.6)
    ax.set_ylim(1.35, 7.82)
    ax.axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ch3-python-to-kernels.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.28, facecolor=black)
    plt.close(fig)
    return path


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
        rocm_profiler_pipeline(),
        python_to_kernels(),
        mixtral_forward(),
        mixtral_backward(),
        render_hlo_fixture(),
    ]
    for output in outputs:
        print(f"  wrote {output.relative_to(ROOT)}")
