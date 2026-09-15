#!/usr/bin/env python3
"""Draw the dtype-attribution diagram for the mixed-precision chapter.

The figure is an illustrative BF16-surrounded FP8/MX training recipe. It keeps
the model graph, the inside of one eligible GEMM, and optimizer state in
separate panels so that unlike roles are not presented as one model-wide dtype.

Run from the repository root:

    python tools/make_mixed_precision_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pages" / "img"

INK = "#27313b"
MUTED = "#65717d"
LINE = "#8a959e"
SURROUND = ("#dce9f5", "#3f76ab")
LOW = ("#f7e6c8", "#b0842f")
ACCUM = ("#e9e3f2", "#6d5b9e")
HIGH = ("#dff0e4", "#4e8a5c")
LOSS = ("#f9d8d6", "#bf5b57")
NEUTRAL = ("#f5f7f8", "#8a959e")


def rounded_box(
    ax,
    cx: float,
    cy: float,
    width: float,
    height: float,
    text: str,
    palette: tuple[str, str],
    *,
    fontsize: float = 8.5,
    weight: str = "normal",
    zorder: int = 3,
) -> None:
    face, edge = palette
    ax.add_patch(
        FancyBboxPatch(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.1,
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
        color=INK,
        weight=weight,
        zorder=zorder + 1,
    )


def arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = LINE,
    dashed: bool = False,
    zorder: int = 5,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=10,
            color=color,
            linewidth=1.0,
            linestyle=(0, (4, 3)) if dashed else "-",
            shrinkA=2,
            shrinkB=2,
            zorder=zorder,
        )
    )


def panel(ax, x: float, y: float, width: float, height: float, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor="#fcfcfc",
            edgecolor="#c9d0d6",
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(x + 0.18, y + height - 0.28, title, ha="left", va="center", fontsize=10.5, weight="bold", color=INK)


def model_graph(ax) -> None:
    panel(ax, 0.25, 0.58, 6.25, 7.55, "Example Llama forward graph")

    cx, width, height = 3.35, 3.82, 0.43
    ys = [1.10, 1.71, 2.32, 2.93, 3.54, 4.15, 4.76, 5.37, 5.98, 6.68, 7.38]
    nodes = [
        ("Embeddings\nBF16 activations · FP32 master parameters", SURROUND),
        ("RMSNorm\nBF16 input/output · FP32 variance", HIGH),
        ("Q / K / V projections\nlow-precision operands · FP32 accumulate · BF16 output", LOW),
        ("RoPE + attention core\nBF16 tensors · FP32 softmax/reductions", ACCUM),
        ("Output projection\nlow-precision operands · FP32 accumulate · BF16 output", LOW),
        ("Residual add\nBF16", SURROUND),
        ("RMSNorm\nBF16 input/output · FP32 variance", HIGH),
        ("Gate / up / down projections\nlow-precision operands · FP32 accumulate · BF16 output", LOW),
        ("Residual output\nBF16", SURROUND),
        ("Final RMSNorm\nBF16 input/output · FP32 variance", HIGH),
        ("LM head + loss\nrecipe-specific head · FP32 logits/loss", LOSS),
    ]

    ax.add_patch(
        Rectangle(
            (0.94, 1.43),
            4.82,
            4.84,
            facecolor="none",
            edgecolor="#a4adb5",
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            zorder=1,
        )
    )
    ax.text(5.90, 6.02, "N×", ha="left", va="center", fontsize=10, color=MUTED)

    for i, ((text, palette), y) in enumerate(zip(nodes, ys)):
        rounded_box(ax, cx, y, width, height, text, palette, fontsize=7.65)
        if i:
            arrow(ax, cx, ys[i - 1] + height / 2, cx, y - height / 2)

def gemm_panel(ax) -> None:
    panel(ax, 6.82, 4.26, 6.93, 3.87, "Inside one eligible projection")

    rounded_box(ax, 7.90, 6.95, 1.60, 0.55, "FP32 master\nweight", HIGH, fontsize=8.0)
    rounded_box(ax, 7.90, 5.66, 1.60, 0.55, "BF16\nactivation", SURROUND, fontsize=8.0)

    rounded_box(ax, 10.02, 6.95, 1.73, 0.55, "quantize / scale\nFP8 or MX", LOW, fontsize=8.0)
    rounded_box(ax, 10.02, 5.66, 1.73, 0.55, "quantize / scale\nFP8 or MX", LOW, fontsize=8.0)
    arrow(ax, 8.71, 6.95, 9.14, 6.95)
    arrow(ax, 8.71, 5.66, 9.14, 5.66)

    rounded_box(ax, 12.02, 6.30, 1.72, 0.82, "scaled MFMA\nlow × low\nFP32 accumulate", ACCUM, fontsize=8.0)
    arrow(ax, 10.90, 6.89, 11.15, 6.52)
    arrow(ax, 10.90, 5.72, 11.15, 6.08)

    rounded_box(ax, 13.06, 5.03, 1.18, 0.55, "BF16\noutput", SURROUND, fontsize=8.0)
    arrow(ax, 12.35, 5.88, 12.82, 5.31)

    rounded_box(ax, 10.10, 4.92, 2.10, 0.48, "scale metadata\nE8M0 or amax history", NEUTRAL, fontsize=7.6)
    arrow(ax, 10.78, 5.18, 11.53, 5.88, dashed=True)

    ax.text(
        10.28,
        4.46,
        "The API output can be BF16 even when the matrix operands were lower precision.",
        ha="center",
        va="center",
        fontsize=7.8,
        color=MUTED,
    )


def state_panel(ax) -> None:
    panel(ax, 6.82, 0.58, 6.93, 3.36, "Training state outside the forward block diagram")

    rounded_box(ax, 7.95, 2.74, 1.75, 0.64, "Backward GEMMs\nrecipe-specific", LOW, fontsize=8.0)
    rounded_box(ax, 10.12, 2.74, 1.72, 0.64, "Gradient +\ncollective dtype", SURROUND, fontsize=8.0)
    rounded_box(ax, 12.42, 2.74, 1.86, 0.64, "Adam update\nusually FP32", HIGH, fontsize=8.0)
    arrow(ax, 8.84, 2.74, 9.24, 2.74)
    arrow(ax, 10.99, 2.74, 11.48, 2.74)

    rounded_box(ax, 12.42, 1.53, 1.86, 0.60, "Master parameters\nusually FP32", HIGH, fontsize=8.0)
    arrow(ax, 12.42, 2.40, 12.42, 1.85)
    arrow(ax, 11.48, 1.53, 8.84, 2.45, dashed=True)

    rounded_box(ax, 9.16, 1.37, 2.20, 0.50, "Adam moments\nusually FP32", HIGH, fontsize=8.0)
    arrow(ax, 11.48, 2.50, 10.17, 1.66, dashed=True)

    ax.text(
        10.28,
        0.84,
        "Gradient, collective, moment, and master-weight dtypes\n"
        "are independent recipe choices.",
        ha="center",
        va="center",
        fontsize=7.8,
        color=MUTED,
    )


def legend(ax) -> None:
    items = [
        (SURROUND, "surrounding BF16"),
        (LOW, "low-precision operands"),
        (ACCUM, "mixed operation"),
        (HIGH, "high-precision state/reduction"),
        (LOSS, "loss boundary"),
    ]
    x = 0.50
    for palette, label in items:
        face, edge = palette
        ax.add_patch(Rectangle((x, 8.47), 0.24, 0.16, facecolor=face, edgecolor=edge, linewidth=0.9))
        ax.text(x + 0.32, 8.55, label, ha="left", va="center", fontsize=8.0, color=INK)
        x += 2.50


def mixed_precision_attribution() -> Path:
    fig, ax = plt.subplots(figsize=(15.0, 9.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0.30, 9.20)
    ax.axis("off")

    ax.text(
        0.25,
        9.02,
        "Mixed precision assigns dtypes to roles, not to the whole model",
        fontsize=15,
        weight="bold",
        color=INK,
        ha="left",
    )
    ax.text(
        0.25,
        8.76,
        "Illustrative BF16-surrounded FP8/MX recipe; exact coverage and dtypes are backend-specific.",
        fontsize=9.3,
        color=MUTED,
        ha="left",
    )

    legend(ax)
    model_graph(ax)
    gemm_panel(ax)
    state_panel(ax)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "mixed-precision-llama-attribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


if __name__ == "__main__":
    output = mixed_precision_attribution()
    print(f"  wrote {output.relative_to(ROOT)}")
