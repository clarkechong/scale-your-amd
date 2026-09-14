#!/usr/bin/env python3
"""Draw the GEMM-to-MFMA diagrams used by the MI355X hardware chapter.

The figures are conceptual rather than measurements. They intentionally keep
backend-dependent tile sizes symbolic while showing the stable hierarchy:

    JAX projection -> dot_general -> GEMM kernel -> macrotile -> wave MFMA

Run from the repository root:

    python tools/make_mfma_diagrams.py
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
GRID = "#9ba6af"
MODEL = ("#dce9f5", "#3f76ab")
IR = ("#dff0e4", "#4e8a5c")
KERNEL = ("#f7e6c8", "#b0842f")
TILE = ("#e9e3f2", "#6d5b9e")
MFMA = ("#f9d8d6", "#bf5b57")
A_COLOR = "#4c86b9"
B_COLOR = "#d39a36"
C_COLOR = "#8a69a8"


def rounded_box(ax, cx, cy, width, height, text, palette, fontsize=10.0) -> None:
    face, edge = palette
    ax.add_patch(
        FancyBboxPatch(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.3,
            zorder=2,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=MUTED, linewidth=1.2) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=11,
            color=color,
            linewidth=linewidth,
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        )
    )


def grid(
    ax,
    x,
    y,
    width,
    height,
    rows,
    cols,
    *,
    face="#ffffff",
    edge=GRID,
    highlight: tuple[int, int] | None = None,
    highlight_color="#f3c87c",
) -> None:
    cell_w, cell_h = width / cols, height / rows
    for row in range(rows):
        for col in range(cols):
            color = highlight_color if highlight == (row, col) else face
            ax.add_patch(
                Rectangle(
                    (x + col * cell_w, y + (rows - 1 - row) * cell_h),
                    cell_w,
                    cell_h,
                    facecolor=color,
                    edgecolor=edge,
                    linewidth=0.65,
                    zorder=2,
                )
            )


def gemm_to_mfma() -> Path:
    fig, ax = plt.subplots(figsize=(14.0, 5.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(
        0.25,
        4.78,
        "One Transformer projection, from model operation to matrix instruction",
        fontsize=14,
        weight="bold",
        color=INK,
        ha="left",
    )
    ax.text(
        0.25,
        4.43,
        "Each level fixes a different part of the performance problem.",
        fontsize=9.5,
        color=MUTED,
        ha="left",
    )

    cy = 2.92
    rounded_box(ax, 1.25, cy, 2.05, 1.22, r"Llama projection" "\n" r"$X[M,K]\,W[K,N]$", MODEL)
    rounded_box(ax, 3.75, cy, 1.72, 1.22, "StableHLO\n" r"$\mathtt{dot\_general}$", IR)
    rounded_box(ax, 6.05, cy, 2.05, 1.22, "GEMM kernel\nselected route", KERNEL)

    arrow(ax, 2.30, cy, 2.87, cy)
    arrow(ax, 4.62, cy, 5.00, cy)
    arrow(ax, 7.08, cy, 7.52, cy)

    ax.text(6.05, 2.12, "hipBLASLt · Triton · XLA", ha="center", fontsize=8.5, color=MUTED)

    grid(ax, 7.70, 2.18, 1.40, 1.48, 5, 5, face="#f8fafb", highlight=(1, 3))
    ax.text(8.40, 3.90, r"output $C[M,N]$", ha="center", fontsize=9.5, color=INK)
    ax.text(8.40, 1.88, "workgroup grid", ha="center", fontsize=8.5, color=MUTED)

    arrow(ax, 9.12, cy, 9.67, cy)
    grid(ax, 9.88, 2.18, 1.40, 1.48, 4, 4, face=TILE[0], edge=TILE[1], highlight=(2, 1))
    ax.text(10.58, 3.90, r"macrotile $B_M\times B_N$", ha="center", fontsize=9.5, color=INK)
    ax.text(10.58, 1.88, r"repeat over $K$ blocks", ha="center", fontsize=8.5, color=MUTED)

    arrow(ax, 11.30, cy, 11.83, cy)
    rounded_box(
        ax,
        12.85,
        cy,
        1.92,
        1.48,
        "wave-level MFMA\n"
        r"$D=A_fB_f+C$"
        "\n"
        r"$16\times16\times K_f$",
        MFMA,
        fontsize=9.5,
    )
    ax.text(12.85, 1.88, "64 lanes · register fragments", ha="center", fontsize=8.5, color=MUTED)

    labels = [
        (1.25, "model shape"),
        (3.75, "portable contraction"),
        (6.05, "backend choice"),
        (8.40, "parallel work"),
        (10.58, "data reuse"),
        (12.85, "matrix issue"),
    ]
    for x, label in labels:
        ax.text(x, 0.82, label, ha="center", va="center", fontsize=9, color=INK)
        ax.plot([x - 0.48, x + 0.48], [1.05, 1.05], color="#d5dce1", linewidth=1.0)

    ax.text(
        7.0,
        0.28,
        "For a compute-bound GEMM, tiling and memory movement are arranged to sustain useful MFMA issue.",
        ha="center",
        fontsize=10,
        color=INK,
        weight="bold",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gemm-to-mfma.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def colored_grid(
    ax,
    x,
    y,
    rows,
    cols,
    cell,
    *,
    base="#f8fafb",
    row_highlight: int | None = None,
    col_highlight: int | None = None,
    row_color="#dce9f5",
    col_color="#f7e6c8",
) -> None:
    for row in range(rows):
        for col in range(cols):
            color = base
            if row_highlight == row:
                color = row_color
            if col_highlight == col:
                color = col_color
            if row_highlight == row and col_highlight == col:
                color = "#cdb9de"
            ax.add_patch(
                Rectangle(
                    (x + col * cell, y + (rows - 1 - row) * cell),
                    cell,
                    cell,
                    facecolor=color,
                    edgecolor=GRID,
                    linewidth=0.55,
                )
            )


def gemm_inner_outer() -> Path:
    fig, ax = plt.subplots(figsize=(13.0, 5.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(
        0.25,
        4.62,
        "Inner products and outer-product updates are two views of the same GEMM",
        fontsize=14,
        weight="bold",
        color=INK,
        ha="left",
    )

    # Panel 1: whole GEMM.
    ax.text(1.85, 4.00, "Whole contraction", ha="center", fontsize=10.5, weight="bold", color=INK)
    cell = 0.22
    colored_grid(ax, 0.45, 2.05, 6, 7, cell, row_highlight=2, col_highlight=3)
    ax.text(1.22, 1.78, r"$A[M,K]$", ha="center", fontsize=9.5, color=A_COLOR)
    ax.text(2.22, 2.70, "@", fontsize=13, color=INK, ha="center")
    colored_grid(ax, 2.62, 2.05, 7, 6, cell, row_highlight=3, col_highlight=4)
    ax.text(3.28, 1.78, r"$B[K,N]$", ha="center", fontsize=9.5, color=B_COLOR)
    ax.text(4.22, 2.70, "=", fontsize=13, color=INK, ha="center")
    colored_grid(ax, 4.57, 2.05, 6, 6, cell, row_highlight=2, col_highlight=4)
    ax.text(5.23, 1.78, r"$C[M,N]$", ha="center", fontsize=9.5, color=C_COLOR)

    ax.plot([6.15, 6.15], [1.34, 4.05], color="#d5dce1", linewidth=1.1)

    # Panel 2: one output element as an inner product.
    ax.text(7.63, 4.00, "One output element", ha="center", fontsize=10.5, weight="bold", color=INK)
    ax.text(7.63, 3.63, "inner-product view", ha="center", fontsize=9.5, color=MUTED)
    for col in range(7):
        ax.add_patch(
            Rectangle(
                (6.70 + col * 0.22, 2.78),
                0.22,
                0.22,
                facecolor="#dce9f5",
                edgecolor=GRID,
                linewidth=0.55,
            )
        )
        ax.add_patch(
            Rectangle(
                (8.52, 1.66 + col * 0.22),
                0.22,
                0.22,
                facecolor="#f7e6c8",
                edgecolor=GRID,
                linewidth=0.55,
            )
        )
    ax.text(7.47, 2.46, r"$A[i,:]$", ha="center", fontsize=9, color=A_COLOR)
    ax.text(8.63, 3.35, r"$B[:,j]$", ha="center", fontsize=9, color=B_COLOR)
    ax.text(8.38, 2.89, r"$\cdot$", fontsize=15, color=INK, ha="center")
    arrow(ax, 8.78, 2.43, 9.41, 2.43)
    rounded_box(ax, 9.82, 2.43, 0.70, 0.62, r"$C_{ij}$", TILE, fontsize=10)
    ax.text(8.25, 1.30, r"$C_{ij}=\sum_k A_{ik}B_{kj}$", ha="center", fontsize=11, color=INK)

    ax.plot([10.45, 10.45], [1.34, 4.05], color="#d5dce1", linewidth=1.1)

    # Panel 3: one K slice as an outer update.
    ax.text(11.67, 4.00, "One K slice", ha="center", fontsize=10.5, weight="bold", color=INK)
    ax.text(11.67, 3.63, "outer-update view", ha="center", fontsize=9.5, color=MUTED)
    for row in range(6):
        ax.add_patch(
            Rectangle(
                (10.67, 1.96 + row * 0.22),
                0.22,
                0.22,
                facecolor="#dce9f5",
                edgecolor=GRID,
                linewidth=0.55,
            )
        )
    for col in range(6):
        ax.add_patch(
            Rectangle(
                (11.20 + col * 0.22, 3.02),
                0.22,
                0.22,
                facecolor="#f7e6c8",
                edgecolor=GRID,
                linewidth=0.55,
            )
        )
    ax.text(10.55, 2.62, r"$A[:,k]$", ha="right", fontsize=9, color=A_COLOR)
    ax.text(11.86, 3.35, r"$B[k,:]$", ha="center", fontsize=9, color=B_COLOR)
    colored_grid(
        ax,
        11.20,
        1.42,
        6,
        6,
        0.22,
        base="#e9e3f2",
        row_color="#e9e3f2",
        col_color="#e9e3f2",
    )
    arrow(ax, 10.98, 2.60, 11.12, 2.35)
    ax.text(
        11.86,
        1.18,
        r"$C\leftarrow C+A[:,k]B[k,:]$",
        ha="center",
        fontsize=10.5,
        color=INK,
    )

    ax.text(
        6.5,
        0.48,
        r"Blocked kernel update:  $C_{IJ}\leftarrow C_{IJ}+A_{IK_b}B_{K_bJ}$",
        ha="center",
        fontsize=11,
        color=INK,
        weight="bold",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gemm-inner-outer.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for output in (gemm_to_mfma(), gemm_inner_outer()):
        print(f"  wrote {output.relative_to(ROOT)}")
