#!/usr/bin/env python3
"""Generate Chapter 4's precision-format and implementation-path figures.

Run from the repository root:

    python tools/make_ch4_precision_figures.py
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
BLUE = ("#dce9f5", "#3f76ab")
GOLD = ("#f7e6c8", "#b0842f")
PURPLE = ("#e9e3f2", "#6d5b9e")
GREEN = ("#dff0e4", "#4e8a5c")
RED = ("#f9d8d6", "#bf5b57")
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


def bit_field(
    ax,
    x: float,
    y: float,
    total_width: float,
    height: float,
    fields: list[tuple[str, int, tuple[str, str]]],
) -> None:
    total_bits = sum(bits for _, bits, _ in fields)
    cursor = x
    for label, bits, palette in fields:
        width = total_width * bits / total_bits
        face, edge = palette
        ax.add_patch(
            Rectangle(
                (cursor, y),
                width,
                height,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.0,
            )
        )
        ax.text(
            cursor + width / 2,
            y + height / 2,
            f"{label}\n{bits} bit{'s' if bits != 1 else ''}",
            ha="center",
            va="center",
            fontsize=7.9,
            color=INK,
        )
        cursor += width


def precision_formats() -> Path:
    fig, ax = plt.subplots(figsize=(14.5, 8.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(
        0.25,
        7.73,
        "Scalar formats spend bits locally; MX formats share range across 32 values",
        fontsize=14.5,
        weight="bold",
        color=INK,
        ha="left",
    )

    panel(ax, 0.25, 3.72, 6.45, 3.65, "Scalar floating-point layouts")
    rows = [
        ("BF16", [("S", 1, RED), ("exponent", 8, BLUE), ("fraction", 7, GREEN)], "max 3.39e38 · ε=2⁻⁷"),
        ("FP16", [("S", 1, RED), ("exponent", 5, BLUE), ("fraction", 10, GREEN)], "max 65,504 · ε=2⁻¹⁰"),
        ("FP8 E4M3", [("S", 1, RED), ("exponent", 4, BLUE), ("fraction", 3, GREEN)], "max 448 · ε=2⁻³"),
        ("FP8 E5M2", [("S", 1, RED), ("exponent", 5, BLUE), ("fraction", 2, GREEN)], "max 57,344 · ε=2⁻²"),
    ]
    y_positions = [6.47, 5.70, 4.93, 4.16]
    for (name, fields, note), y in zip(rows, y_positions):
        ax.text(0.52, y + 0.23, name, ha="left", va="center", fontsize=9.2, weight="bold", color=INK)
        bit_field(ax, 1.65, y, 3.65, 0.47, fields)
        ax.text(5.42, y + 0.23, note, ha="left", va="center", fontsize=7.1, color=MUTED)

    panel(ax, 6.95, 3.72, 6.80, 3.65, "One OCP MX block")
    box(ax, 7.24, 6.28, 1.20, 0.56, "BF16 / FP32\nsource", BLUE, fontsize=8.0)
    box(ax, 8.80, 6.28, 1.38, 0.56, "block amax\n+ power-of-2 scale", PURPLE, fontsize=7.7)
    box(ax, 10.56, 6.28, 1.20, 0.56, "E8M0\n8-bit scale", GOLD, fontsize=8.0)
    arrow(ax, 8.45, 6.56, 8.79, 6.56)
    arrow(ax, 10.19, 6.56, 10.55, 6.56)
    arrow(ax, 11.77, 6.56, 12.62, 5.88, connectionstyle="arc3,rad=-0.10")

    start_x, cell_w, cell_y = 7.30, 0.185, 5.30
    for i in range(32):
        ax.add_patch(
            Rectangle(
                (start_x + i * cell_w, cell_y),
                cell_w,
                0.48,
                facecolor=GOLD[0] if i % 2 == 0 else "#f3d9ad",
                edgecolor=GOLD[1],
                linewidth=0.48,
            )
        )
        if i in (0, 1, 30, 31):
            ax.text(
                start_x + (i + 0.5) * cell_w,
                cell_y + 0.24,
                str(i),
                ha="center",
                va="center",
                fontsize=6.0,
                color=INK,
            )
    ax.text(
        10.26,
        5.00,
        "One E8M0 scale multiplies the 32 decoded element values.",
        ha="center",
        va="center",
        fontsize=8.0,
        color=MUTED,
    )

    storage_rows = [
        ("MXFP8", "32 × 8-bit elements + 8-bit scale", "264 bits = 33 bytes", BLUE),
        ("MXFP6", "32 × 6-bit elements + 8-bit scale", "200 bits = 25 bytes", PURPLE),
        ("MXFP4", "32 × 4-bit elements + 8-bit scale", "136 bits = 17 bytes", GREEN),
    ]
    for i, (name, expression, total, palette) in enumerate(storage_rows):
        y = 4.58 - i * 0.30
        ax.text(7.30, y, name, ha="left", va="center", fontsize=7.8, weight="bold", color=palette[1])
        ax.text(8.27, y, expression, ha="left", va="center", fontsize=7.4, color=INK)
        ax.text(13.48, y, total, ha="right", va="center", fontsize=7.4, color=MUTED)

    panel(ax, 0.25, 0.35, 13.50, 3.08, "The operation still has three precision decisions")
    box(ax, 0.63, 2.14, 2.08, 0.64, "operand A\nFP8 or MX block", GOLD, fontsize=8.4)
    box(ax, 0.63, 1.25, 2.08, 0.64, "operand B\nFP8 or MX block", GOLD, fontsize=8.4)
    box(ax, 3.53, 1.43, 2.42, 1.12, "CDNA 4 matrix instruction\nscaled low × low\nFP32 partial sums", PURPLE, fontsize=8.7)
    box(ax, 7.05, 1.62, 1.74, 0.74, "stored output\nusually BF16", BLUE, fontsize=8.4)
    box(ax, 10.10, 1.62, 1.72, 0.74, "master state\nusually FP32", GREEN, fontsize=8.4)
    arrow(ax, 2.72, 2.46, 3.52, 2.24)
    arrow(ax, 2.72, 1.57, 3.52, 1.75)
    arrow(ax, 5.96, 1.99, 7.04, 1.99)
    arrow(ax, 8.80, 1.99, 10.09, 1.99, dashed=True)
    ax.text(
        6.75,
        0.93,
        "Element format, scale granularity, accumulator type, and stored output are separate recipe choices.",
        ha="center",
        va="center",
        fontsize=8.7,
        color=MUTED,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ch4-precision-format-layouts.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


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
    for output in (precision_formats(), implementation_paths()):
        print(f"  wrote {output.relative_to(ROOT)}")
