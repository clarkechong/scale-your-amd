#!/usr/bin/env python3
"""Draw the JAX/XLA/ROCm stack for the software chapter.

A hand-placed diagram, like `make_topology_diagram.py` and unlike the measured
plots in `make_figures.py`. Pass names and the IR on each edge follow the
XLA:GPU pipeline as documented at https://openxla.org/xla/gpu_architecture,
with the NVIDIA leg of that diagram replaced by its ROCm counterpart and the
whole thing straightened into one vertical spine.

    python tools/make_stack_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pages" / "img"

SPINE_CX, SPINE_W, BOX_H, PITCH = 6.90, 2.90, 0.50, 0.90
BOTTOM_CY = 1.05
LEFT_CX, LEFT_W = 2.10, 3.10

# Widen the two gaps the XLA:GPU boundary runs through, so that the dashed line
# does not strike through the edge label sitting in the same gap.
EXTRA_DROP = {12: 0.30}
EXTRA_RAISE = {2: 0.25}

FRAMEWORK = ("#dce9f5", "#3f76ab")
XLA = ("#dff0e4", "#4e8a5c")
THIRD_PARTY = ("#e9e3f2", "#6d5b9e")
ROCM = ("#f7e6c8", "#b0842f")
EXECUTION = ("#f9d8d6", "#bf5b57")

EDGE = "0.45"

KEY = [
    (FRAMEWORK, "JAX frontend"),
    (XLA, "XLA:GPU pipeline"),
    (ROCM, "XLA ROCm backend"),
    (THIRD_PARTY, "Triton (third party)"),
    (EXECUTION, "Execution"),
]

# Top to bottom: (name, palette, label on the edge leaving this box).
SPINE = [
    ("Python", FRAMEWORK, "jit-decorated function"),
    ("JAX", FRAMEWORK, "jaxpr, lowered to MLIR"),
    ("StableHLO", FRAMEWORK, "HLO (w/ shardings)"),
    ("SPMD Partitioner", XLA, "HLO (w/ collectives)"),
    ("Algebraic Rewrites", XLA, "HLO (w/ above)"),
    ("Layout Assignment", XLA, "HLO (w/ layouts & above)"),
    ("Fusion", XLA, "HLO (w/ fusions & above)"),
    ("Scheduling", XLA, "Schedule"),
    ("Bufferization", XLA, "Buffer Assignment"),
    ("Codegen", XLA, ""),
    ("JitRT", XLA, "RuntimeIR / Object File"),
    ("Runtime", XLA, "Buffers"),
    ("PJRT Executable", EXECUTION, "kernel dispatch"),
    ("CDNA4 Hardware (gfx950)", EXECUTION, ""),
]

FUSION = 6
CODEGEN = 9
JITRT = 10
RUNTIME = 11

# Left column, each entry (name, palette, centre y, height).
AUTOTUNING = ("Autotuning", XLA, 7.85, 0.50)
COST_MODEL = ("Cost Model", XLA, 7.05, 0.50)
TRITON = ("Triton", THIRD_PARTY, 5.55, 0.50)
ROCM_LLVM = ("ROCm LLVM", ROCM, 4.45, 0.50)
AMDGPU_CODEGEN = ("AMDGPU CodeGen", ROCM, 3.55, 0.50)
LIBRARIES = ("hipBLAS, rocBLAS, CK,\nMIOpen, RCCL", ROCM, 2.60, 0.68)
HIP = ("HIP / HSA API", ROCM, 1.70, 0.50)


def spine_cy(i: int) -> float:
    base = BOTTOM_CY + (len(SPINE) - 1 - i) * PITCH
    base -= sum(drop for first, drop in EXTRA_DROP.items() if i >= first)
    return base + sum(raise_ for last, raise_ in EXTRA_RAISE.items() if i <= last)


def rbox(ax, cx, cy, w, h, text, palette, fontsize=8.5) -> None:
    fc, ec = palette
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.1,
            zorder=3,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=4)


def group(ax, x0, y0, x1, y1, label, fc="none", ec="0.6", label_side="right") -> None:
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            zorder=0,
        )
    )
    if not label:
        return
    if label_side == "outside-top":
        ax.text(x0 - 0.10, y1 - 0.04, label, ha="right", va="top", fontsize=8, color="0.4")
    else:
        ax.text(x1 - 0.12, y0 + 0.10, label, ha="right", va="bottom", fontsize=8, color="0.4")


def arrow(ax, start, end, label="", side="right") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": EDGE,
            "linewidth": 1.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=2,
    )
    if not label:
        return
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    if side == "right":
        ax.text(mx + 0.14, my, label, ha="left", va="center", fontsize=7.5, color="0.45")
    elif side == "above":
        ax.text(mx, my + 0.09, label, ha="center", va="bottom", fontsize=7.0, color="0.45")
    else:
        ax.text(mx, my - 0.09, label, ha="center", va="top", fontsize=7.0, color="0.45")


def draw_spine(ax) -> None:
    for i, (name, palette, edge_label) in enumerate(SPINE):
        cy = spine_cy(i)
        rbox(ax, SPINE_CX, cy, SPINE_W, BOX_H, name, palette)
        if i + 1 < len(SPINE):
            arrow(
                ax,
                (SPINE_CX, cy - BOX_H / 2),
                (SPINE_CX, spine_cy(i + 1) + BOX_H / 2),
                edge_label,
            )

    group(
        ax,
        SPINE_CX - SPINE_W / 2 - 0.20,
        spine_cy(RUNTIME) - BOX_H / 2 - 0.20,
        9.95,  # wide enough to take in the edge labels, as the OpenXLA diagram does
        spine_cy(3) + BOX_H / 2 + 0.18,
        "XLA:GPU",
        fc="#f7fbf8",
        label_side="outside-top",
    )


def draw_left(ax) -> None:
    for name, palette, cy, h in (
        AUTOTUNING,
        COST_MODEL,
        TRITON,
        ROCM_LLVM,
        AMDGPU_CODEGEN,
        LIBRARIES,
        HIP,
    ):
        rbox(ax, LEFT_CX, cy, LEFT_W, h, name, palette, fontsize=8.0)

    left_edge, right_edge = LEFT_CX - LEFT_W / 2 - 0.18, LEFT_CX + LEFT_W / 2
    group(
        ax,
        left_edge,
        COST_MODEL[2] - 0.43,
        LEFT_CX + LEFT_W / 2 + 0.18,
        AUTOTUNING[2] + 0.43,
        "",
    )
    group(
        ax,
        left_edge,
        HIP[2] - 0.62,
        LEFT_CX + LEFT_W / 2 + 0.18,
        ROCM_LLVM[2] + 0.42,
        "ROCm",
        fc="#fdf9f1",
    )

    spine_left = SPINE_CX - SPINE_W / 2
    fusion_cy = spine_cy(FUSION)
    arrow(ax, (right_edge, AUTOTUNING[2]), (spine_left, fusion_cy + 0.10))
    arrow(ax, (spine_left, fusion_cy - 0.10), (right_edge, COST_MODEL[2]))

    codegen_cy = spine_cy(CODEGEN)
    arrow(ax, (spine_left, codegen_cy), (right_edge, TRITON[2]), "TTIR", side="above")
    arrow(ax, (spine_left, codegen_cy), (right_edge, ROCM_LLVM[2]), "LLVM IR", side="above")
    arrow(
        ax,
        (LEFT_CX, TRITON[2] - 0.25),
        (LEFT_CX, ROCM_LLVM[2] + 0.25),
        "LLVM IR",
        side="right",
    )
    arrow(
        ax,
        (LEFT_CX, ROCM_LLVM[2] - 0.25),
        (LEFT_CX, AMDGPU_CODEGEN[2] + 0.25),
        "amdgcn ISA",
        side="right",
    )
    arrow(
        ax,
        (right_edge, AMDGPU_CODEGEN[2]),
        (spine_left, spine_cy(JITRT)),
        "hsaco",
        side="below",
    )

    runtime_cy = spine_cy(RUNTIME)
    arrow(ax, (spine_left, runtime_cy), (right_edge, LIBRARIES[2]))
    arrow(ax, (spine_left, runtime_cy), (right_edge, HIP[2]))


def draw_key(ax) -> None:
    swatch_cx, text_x, top_cy, step = 0.70, 1.05, 13.15, 0.42
    for i, ((fc, ec), label) in enumerate(KEY):
        cy = top_cy - i * step
        ax.add_patch(
            FancyBboxPatch(
                (swatch_cx - 0.18, cy - 0.11),
                0.36,
                0.22,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=fc,
                edgecolor=ec,
                linewidth=0.9,
            )
        )
        ax.text(text_x, cy, label, ha="left", va="center", fontsize=8.5)

    ax.add_patch(
        Rectangle(
            (0.35, top_cy - (len(KEY) - 1) * step - 0.28),
            2.85,
            (len(KEY) - 1) * step + 0.56,
            facecolor="none",
            edgecolor="0.78",
            linewidth=0.9,
            zorder=0,
        )
    )


def jax_rocm_stack() -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 10.3))
    draw_spine(ax)
    draw_left(ax)
    draw_key(ax)

    ax.set_xlim(0.20, 10.30)
    ax.set_ylim(0.30, 13.60)
    ax.set_aspect("equal")
    ax.axis("off")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "jax-rocm-stack.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return path


def rocm_device_stack() -> Path:
    """Draw the short device-discovery stack used in the hardware chapter."""
    stack = [
        ("JAX Device  rocm:0", FRAMEWORK, "backend discovery"),
        ("ROCm PJRT plugin", XLA, "device, buffer, executable API"),
        ("XLA StreamExecutor", XLA, "streams, events, allocation"),
        ("HIP runtime", ROCM, "runtime calls"),
        ("ROCr / HSA runtime", ROCM, "queues, memory, code objects"),
        ("amdgpu + KFD", EXECUTION, "kernel-driver submission"),
        ("MI355X  (gfx950)", EXECUTION, ""),
    ]
    cx, w, h, pitch = 4.55, 3.40, 0.56, 1.02
    top = 6.65

    fig, ax = plt.subplots(figsize=(9.2, 6.2))

    groups = [
        (0, 2, "JAX / XLA", "#f7fbf8"),
        (3, 4, "ROCm userspace", "#fdf9f1"),
        (5, 5, "Linux kernel", "#fdf4f3"),
        (6, 6, "Hardware", "#fdf4f3"),
    ]
    for first, last, label, fc in groups:
        y_top = top - first * pitch + h / 2 + 0.17
        y_bottom = top - last * pitch - h / 2 - 0.17
        group(
            ax,
            cx - w / 2 - 0.24,
            y_bottom,
            cx + w / 2 + 0.24,
            y_top,
            label,
            fc=fc,
            label_side="outside-top",
        )

    for i, (name, palette, edge_label) in enumerate(stack):
        cy = top - i * pitch
        rbox(ax, cx, cy, w, h, name, palette, fontsize=9.0)
        if i + 1 < len(stack):
            arrow(
                ax,
                (cx, cy - h / 2),
                (cx, top - (i + 1) * pitch + h / 2),
                edge_label,
            )

    note_x, note_y, note_w, note_h = 8.60, 3.52, 2.70, 1.78
    ax.add_patch(
        FancyBboxPatch(
            (note_x - note_w / 2, note_y - note_h / 2),
            note_w,
            note_h,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor="#fcfcfc",
            edgecolor="#8a8a8a",
            linewidth=1.0,
            zorder=3,
        )
    )
    ax.text(
        note_x,
        note_y + 0.58,
        "What changes enumeration",
        ha="center",
        va="center",
        fontsize=8.5,
        weight="bold",
        zorder=4,
    )
    ax.text(
        note_x,
        note_y + 0.08,
        "container device mapping\n"
        "HIP_VISIBLE_DEVICES\n"
        "SPX / DPX / QPX / CPX",
        ha="center",
        va="center",
        fontsize=8.0,
        linespacing=1.55,
        zorder=4,
    )
    ax.set_xlim(1.75, 10.20)
    ax.set_ylim(0.15, 7.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("The ROCm device stack", loc="left", fontsize=11.0)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rocm-device-stack.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for output in (jax_rocm_stack(), rocm_device_stack()):
        print(f"  wrote {output.relative_to(ROOT)}")
