#!/usr/bin/env python3
"""Draw the multi-node topology schematic for the hardware chapter.

Nothing here is measured, unlike `make_figures.py`: it is a hand-placed diagram
of what the Multi-node Architecture and Topology section describes. It lives as
code so the labelled bandwidths sit in one place and can be corrected when the
platform they describe changes.

    python tools/make_topology_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pages" / "img"

GPUS_PER_NODE = 8
SPACING = 0.55

GPU_W, GPU_H, GPU_CY = 0.42, 0.40, 1.95
NIC_W, NIC_H, NIC_CY = 0.42, 0.30, 2.60
LEAF_W, LEAF_H, LEAF_CY = 0.46, 0.40, 4.30
SPINE_W, SPINE_H, SPINE_CY = 0.90, 0.40, 5.60

NODE_ORIGIN = (0.55, 5.75)  # centre of GPU 0 in each node
NODE_Y0, NODE_Y1 = 0.35, 3.00
SPINE_CX = (4.60, 5.60)

GPU_FC, GPU_EC = "#dce9f5", "#3f76ab"
NIC_FC, NIC_EC = "#f7e6c8", "#b0842f"
SW_FC, SW_EC = "#e9e9e9", "#6a6a6a"
NODE_FC, NODE_EC = "#fcfcfc", "#8a8a8a"
MESH, WIRE, RAIL, CROSS = "#3f76ab", "#8a8a8a", "#b8352b", "#d9862a"

# Depth of the deepest mesh arc, as a fraction of the widest GPU pair spacing.
MESH_DEPTH = 0.34


def gpu_cx(node: int, i: int) -> float:
    return NODE_ORIGIN[node] + i * SPACING


def leaf_cx(i: int) -> float:
    return (NODE_ORIGIN[0] + NODE_ORIGIN[1]) / 2 + i * SPACING


def rbox(ax, cx, cy, w, h, text, fc, ec, fontsize=7.0, weight="normal") -> None:
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.07",
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.9,
            zorder=3,
        )
    )
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, weight=weight, zorder=4)


def wire(ax, points, color=WIRE, lw=0.8, ls="-", alpha=1.0, zorder=2) -> None:
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder)


def draw_node(ax, node: int) -> None:
    """One scale-up domain: 8 GPUs, a full mesh beneath, a NIC above each GPU."""
    x0, x1 = gpu_cx(node, 0) - 0.30, gpu_cx(node, GPUS_PER_NODE - 1) + 0.30
    ax.add_patch(
        FancyBboxPatch(
            (x0, NODE_Y0),
            x1 - x0,
            NODE_Y1 - NODE_Y0,
            boxstyle="round,pad=0,rounding_size=0.10",
            facecolor=NODE_FC,
            edgecolor=NODE_EC,
            linewidth=1.1,
            zorder=1,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        NODE_Y0 + 0.27,
        f"Node {node}",
        ha="center",
        va="center",
        fontsize=8.5,
        weight="bold",
        zorder=4,
    )

    for i in range(GPUS_PER_NODE):
        for j in range(i + 1, GPUS_PER_NODE):
            xi, xj = gpu_cx(node, i), gpu_cx(node, j)
            span = xj - xi
            ax.add_patch(
                Arc(
                    ((xi + xj) / 2, GPU_CY - GPU_H / 2),
                    width=span,
                    height=2 * MESH_DEPTH * span,
                    theta1=180,
                    theta2=360,
                    edgecolor=MESH,
                    linewidth=0.7,
                    alpha=0.30,
                    zorder=2,
                )
            )

    for i in range(GPUS_PER_NODE):
        cx = gpu_cx(node, i)
        rbox(ax, cx, GPU_CY, GPU_W, GPU_H, f"GPU{i}", GPU_FC, GPU_EC, fontsize=6.0)
        rbox(ax, cx, NIC_CY, NIC_W, NIC_H, "NIC", NIC_FC, NIC_EC, fontsize=5.5)
        wire(ax, [(cx, GPU_CY + GPU_H / 2), (cx, NIC_CY - NIC_H / 2)], lw=0.9)
        wire(ax, [(cx, NIC_CY + NIC_H / 2), (leaf_cx(i), LEAF_CY - LEAF_H / 2)], lw=0.8)


def draw_fabric(ax) -> None:
    for i in range(GPUS_PER_NODE):
        rbox(ax, leaf_cx(i), LEAF_CY, LEAF_W, LEAF_H, str(i), SW_FC, SW_EC, fontsize=7.5)
        for sx in SPINE_CX:
            wire(
                ax,
                [(leaf_cx(i), LEAF_CY + LEAF_H / 2), (sx, SPINE_CY - SPINE_H / 2)],
                lw=0.6,
                alpha=0.30,
            )
    for sx in SPINE_CX:
        rbox(ax, sx, SPINE_CY, SPINE_W, SPINE_H, "spine", SW_FC, SW_EC, fontsize=7.0)


def draw_paths(ax) -> None:
    """A same-rail transfer, and a cross-rail one that has to climb the spine."""
    gpu_top, nic_bot, nic_top = GPU_CY + GPU_H / 2, NIC_CY - NIC_H / 2, NIC_CY + NIC_H / 2
    leaf_bot, leaf_top = LEAF_CY - LEAF_H / 2, LEAF_CY + LEAF_H / 2

    rail = 3
    a, b, leaf = gpu_cx(0, rail), gpu_cx(1, rail), leaf_cx(rail)
    for segment in (
        [(a, gpu_top), (a, nic_bot)],
        [(a, nic_top), (leaf, leaf_bot)],
        [(leaf, leaf_bot), (b, nic_top)],
        [(b, nic_bot), (b, gpu_top)],
    ):
        wire(ax, segment, color=RAIL, lw=2.0, zorder=5)

    src, dst = 1, 6
    a, b = gpu_cx(0, src), gpu_cx(1, dst)
    for segment in (
        [(a, gpu_top), (a, nic_bot)],
        [(a, nic_top), (leaf_cx(src), leaf_bot)],
        [(leaf_cx(src), leaf_top), (SPINE_CX[0], SPINE_CY - SPINE_H / 2)],
        [(SPINE_CX[0], SPINE_CY - SPINE_H / 2), (leaf_cx(dst), leaf_top)],
        [(leaf_cx(dst), leaf_bot), (b, nic_top)],
        [(b, nic_bot), (b, gpu_top)],
    ):
        wire(ax, segment, color=CROSS, lw=1.6, ls=(0, (4, 2)), zorder=5)

    ax.annotate(
        "same rail, one switch hop",
        xy=(leaf_cx(rail) - LEAF_W / 2, LEAF_CY),
        xytext=(0.75, 5.15),
        color=RAIL,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": RAIL, "linewidth": 0.8},
    )
    ax.annotate(
        "crossing rails, up to the spine",
        xy=(5.55, 4.94),
        xytext=(7.05, 4.90),
        color=CROSS,
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": CROSS, "linewidth": 0.8},
    )


def draw_key(ax) -> None:
    """Right-hand column, each entry level with the layer it describes."""
    swatch_x, text_x = 10.20, 10.55
    entries = [
        (SPINE_CY, "box", SW_FC, SW_EC, "Spine, crossed only by cross-rail traffic"),
        (LEAF_CY, "box", SW_FC, SW_EC, "Rail leaves: rail i serves GPU i of every node"),
        (NIC_CY, "box", NIC_FC, NIC_EC, "Pollara 400 AI NIC, 400Gb/s = 50GB/s per direction"),
        (2.24, "line", WIRE, WIRE, "PCIe Gen5 x16, 128GB/s, RDMA straight into HBM"),
        (1.30, "line", MESH, MESH, "Infinity Fabric mesh, 153.6GB/s per link, 1 hop"),
        (0.62, "box", NODE_FC, NODE_EC, "One node, one scale-up domain: 2.3TB coherent HBM"),
    ]
    for y, kind, fc, ec, label in entries:
        if kind == "box":
            ax.add_patch(
                Rectangle(
                    (swatch_x - 0.11, y - 0.08),
                    0.22,
                    0.16,
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=0.9,
                )
            )
        else:
            ax.plot(
                [swatch_x - 0.13, swatch_x + 0.13], [y, y], color=ec, linewidth=1.2, alpha=0.8
            )
        ax.text(text_x, y, label, ha="left", va="center", fontsize=8)


def multinode_topology() -> Path:
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    for node in (0, 1):
        draw_node(ax, node)
    draw_fabric(ax)
    draw_paths(ax)
    draw_key(ax)

    ax.set_xlim(0.0, 13.6)
    ax.set_ylim(0.0, 6.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Scale-up inside the node, scale-out between them",
        loc="left",
        fontsize=10.5,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "multinode-topology.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(f"  wrote {multinode_topology().relative_to(ROOT)}")
