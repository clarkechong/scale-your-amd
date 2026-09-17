#!/usr/bin/env python3
"""Generate the custom implementation-path figure retained by Chapter 4.

Run from the repository root:

    python tools/make_ch4_precision_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pages" / "img"

INK = "#27313b"
MUTED = "#65717d"
LINE = "#8a959e"
BLUE = ("#e8eff5", "#496a82")
GOLD = BLUE
PURPLE = BLUE
GREEN = BLUE
RED = BLUE
NEUTRAL = ("#f5f7f8", "#8a959e")


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    palette: tuple[str, str] = NEUTRAL,
    *,
    fontsize: float = 8.5,
    weight: str = "normal",
    radius: float = 0.06,
    zorder: int = 3,
) -> None:
    face, edge = palette
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.025,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.05,
            zorder=zorder,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
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
    connectionstyle: str = "arc3",
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
            connectionstyle=connectionstyle,
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
            boxstyle="round,pad=0.04,rounding_size=0.09",
            facecolor="#fcfcfc",
            edgecolor="#c9d0d6",
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(
        x + 0.17,
        y + height - 0.25,
        title,
        ha="left",
        va="center",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )


def implementation_paths() -> Path:
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(
        0.25,
        6.90,
        "A MaxText precision flag changes the JAX program before XLA compiles it",
        fontsize=14.5,
        weight="bold",
        color=INK,
        ha="left",
    )

    panel(ax, 0.25, 0.55, 2.55, 5.93, "Configuration")
    configs = [
        ("dtype: bfloat16\nquantization: \"\"", BLUE),
        ("te_fp8_delayedscaling", GOLD),
        ("te_mxfp8", PURPLE),
        ("aiter_fp4\nuse_jax_aiter: true", GREEN),
    ]
    config_y = [5.24, 4.15, 3.06, 1.97]
    for (text, palette), y in zip(configs, config_y):
        box(ax, 0.56, y, 1.94, 0.68, text, palette, fontsize=8.0)

    panel(ax, 3.05, 0.55, 3.04, 5.93, "MaxText layer construction")
    box(ax, 3.40, 5.17, 2.34, 0.80, "configure_quantization\nselects a recipe object", NEUTRAL, fontsize=8.2)
    box(ax, 3.40, 3.75, 2.34, 0.80, "DenseGeneral\nreceives quant=recipe", BLUE, fontsize=8.2)
    box(ax, 3.40, 2.33, 2.34, 0.80, "_compute_dot_general\nchooses the callable", PURPLE, fontsize=8.2)
    for y in config_y:
        arrow(ax, 2.51, y + 0.34, 3.39, 5.57, connectionstyle="arc3,rad=-0.10")
    arrow(ax, 4.57, 5.16, 4.57, 4.56)
    arrow(ax, 4.57, 3.74, 4.57, 3.14)

    panel(ax, 6.34, 0.55, 3.20, 5.93, "Traced JAX / HLO")
    box(ax, 6.68, 5.18, 2.53, 0.78, "standard lax.dot_general\n→ HLO dot", BLUE, fontsize=8.2)
    box(
        ax,
        6.68,
        3.70,
        2.53,
        0.96,
        "TE quantize primitives\n→ te_dbias_quantize_ffi\n→ te_gemm_v2_ffi",
        GOLD,
        fontsize=7.8,
    )
    box(
        ax,
        6.68,
        2.03,
        2.53,
        1.10,
        "JAX-AITER custom_vjp\n→ CastMxfp4DualJA\n→ GemmFp4FwdJA",
        GREEN,
        fontsize=7.8,
    )
    arrow(ax, 5.75, 2.73, 6.67, 5.57, connectionstyle="arc3,rad=-0.18")
    arrow(ax, 5.75, 2.73, 6.67, 4.18, connectionstyle="arc3,rad=-0.08")
    arrow(ax, 5.75, 2.73, 6.67, 2.58, connectionstyle="arc3,rad=0.08")

    panel(ax, 9.79, 0.55, 3.96, 5.93, "ROCm execution")
    box(ax, 10.18, 5.18, 3.17, 0.78, "XLA GPU lowering\nhipBLASLt or generated kernel", BLUE, fontsize=8.2)
    box(ax, 10.18, 3.70, 3.17, 0.96, "Transformer Engine handler\nquantization + GEMM backend", GOLD, fontsize=8.2)
    box(ax, 10.18, 2.03, 3.17, 1.10, "JAX-AITER typed FFI handler\nfused casts + AITER FP4 ASM", GREEN, fontsize=8.2)
    arrow(ax, 9.22, 5.57, 10.17, 5.57)
    arrow(ax, 9.22, 4.18, 10.17, 4.18)
    arrow(ax, 9.22, 2.58, 10.17, 2.58)

    ax.text(
        7.02,
        0.84,
        "A custom_call is the HLO boundary where XLA hands buffers and the GPU stream to a registered runtime implementation.",
        ha="center",
        va="center",
        fontsize=8.4,
        color=MUTED,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ch4-precision-implementation-paths.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for output in (implementation_paths(),):
        print(f"  wrote {output.relative_to(ROOT)}")
