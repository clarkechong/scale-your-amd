#!/usr/bin/env python3
"""Option A: Google's scaling book vs this presentation, with a knowledge gap."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties, fontManager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

for _font in (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
):
    fontManager.addfont(_font)

ARIAL = FontProperties(fname="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
ARIAL_BOLD = FontProperties(fname="/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pages" / "img"

WHITE = "#ffffff"
BLACK = "#000000"
FILL = "#0d0d0d"
DIM = "#c4c8cc"
BLUE = "#4c8bf5"
RED = "#ed1c24"


def option_a() -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 8.6), facecolor=BLACK)
    ax.set_facecolor(BLACK)
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
            linespacing=1.28,
            fontproperties=ARIAL_BOLD if weight == "bold" else ARIAL,
            parse_math=False,
            zorder=5,
        )

    def v_arrow(x, y0, y1, color):
        ax.annotate(
            "",
            xy=(x, y1),
            xytext=(x, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, mutation_scale=12),
            zorder=4,
        )

    def card(cx, cy, w, h, title, bullets, color):
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.07",
                facecolor=FILL,
                edgecolor=color,
                linewidth=1.45,
                zorder=3,
            )
        )
        top = cy + h / 2 - 0.20
        text(cx, top, title, size=12.5, color=color, weight="bold")
        for i, item in enumerate(bullets):
            line, line_color = item if isinstance(item, tuple) else (item, WHITE)
            text(cx, top - 0.30 - i * 0.24, line, size=10.0, color=line_color)

    def card_h(n):
        return 0.20 + 0.22 + 0.10 + n * 0.24 + 0.14

    cx = 3.55
    w = 4.05
    gap = 0.38
    y = 8.70

    layers = [
        (
            "First-principles derivations",
            ["•  Roofline analysis", "•  Memory accounting", "•  Accelerator theory"],
            BLUE,
        ),
        (
            "Training concepts",
            ["•  FSDP", "•  Quantization", "•  Rematerialization", "•  Parallelism"],
            BLUE,
        ),
        "line",
        (
            "Implementation layer",
            [("•  JAX", BLUE), "•  XLA", "•  MaxText"],
            RED,
        ),
        (
            "AMD GPU ecosystem",
            ["•  ROCm libraries", "•  Kernel backends", "•  Profiling on AMD GPUs", "•  Performance tuning"],
            RED,
        ),
    ]

    prev_bottom = None
    prev_color = None
    for layer in layers:
        if layer == "line":
            y -= 0.22
            ax.plot([cx - w / 2, cx + w / 2], [y, y], color=WHITE, lw=1.35, zorder=2)
            y -= 0.22
            prev_bottom = None
            continue
        title, bullets, color = layer
        h = card_h(len(bullets))
        if prev_bottom is not None:
            v_arrow(cx, prev_bottom, y, prev_color)
        cy = y - h / 2
        card(cx, cy, w, h, title, bullets, color)
        prev_bottom = cy - h / 2
        prev_color = color
        y = prev_bottom - gap

    ax.set_xlim(1.05, 6.05)
    ax.set_ylim(y + gap - 0.15, 8.95)
    ax.axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "option-a-knowledge-gap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.32, facecolor=BLACK)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(option_a())
