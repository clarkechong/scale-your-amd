#!/usr/bin/env python3
"""Generate the original diagrams used by Chapter 3.

The figures are explanatory, not benchmark plots. The roofline uses the
published MI355X BF16 peak and HBM bandwidth from Chapter 1. The HLO SVG is
rendered from the retained literal XLA DOT fixture and its provenance under
``artifacts/hlo-fixtures/moe/ragged-grouped``.

Run from the repository root:

    python tools/make_ch3_profiling_figures.py
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "pages" / "img"

INK = "#27313b"
MUTED = "#65717d"
LINE = "#8a959e"
FRAMEWORK = ("#dce9f5", "#3f76ab")
XLA = ("#dff0e4", "#4e8a5c")
ROCM = ("#f7e6c8", "#b0842f")
EVIDENCE = ("#e9e3f2", "#6d5b9e")
EXECUTION = ("#f9d8d6", "#bf5b57")
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


def two_capture_workflow() -> Path:
    fig, ax = plt.subplots(figsize=(15.2, 9.4))
    title(
        ax,
        "One configuration, two independent profiler captures",
        "Keep unprofiled timing separate; join evidence by HLO and kernel identity.",
    )

    box(
        ax,
        7.0,
        7.98,
        4.7,
        0.72,
        "Resolved MaxText configuration\nsoftware revisions · mesh · shapes · dtypes · XLA_FLAGS",
        NEUTRAL,
        fontsize=8.7,
        weight="bold",
    )
    box(
        ax,
        7.0,
        6.82,
        4.25,
        0.66,
        "Warm up outside capture\ncompile · autotune · fill caches · synchronize",
        EXECUTION,
        fontsize=8.6,
    )
    arrow(ax, (7.0, 7.61), (7.0, 7.17))

    panel(ax, 0.40, 1.88, 6.25, 4.28, "Capture A — JAX/XLA attribution")
    panel(ax, 7.35, 1.88, 6.25, 4.28, "Capture B — ROCm execution evidence")
    # Enter the first capture boxes from their inner edges so the branch
    # arrows do not cross either panel title.
    arrow(ax, (6.45, 6.49), (5.98, 5.45), connectionstyle="arc3,rad=0.08")
    arrow(ax, (7.55, 6.49), (8.02, 5.45), connectionstyle="arc3,rad=-0.08")

    left = [
        (3.55, 5.45, "MaxText  profiler=xplane\nshort steady-state window", FRAMEWORK),
        (3.55, 4.46, "XSpace + HLO sidecars\noptimized HLO dump", XLA),
        (3.55, 3.47, "XProf\nsteps · framework/HLO ops · memory · kernels", EVIDENCE),
        (3.55, 2.48, "Answers: which model component?\nwhich HLO and source scope?", NEUTRAL),
    ]
    right = [
        (10.45, 5.45, "MaxText  profiler=\"\"\nrun under rocprofv3", FRAMEWORK),
        (10.45, 4.46, "rocpd trace → Perfetto\nHIP · queues · kernels · copies · RCCL", ROCM),
        (10.45, 3.47, "Separate filtered PMC run\nper-dispatch hardware counters", ROCM),
        (10.45, 2.48, "Answers: which kernel ran?\nhow long, what traffic, what resources?", NEUTRAL),
    ]
    for column in (left, right):
        for i, (cx, cy, text, palette) in enumerate(column):
            box(ax, cx, cy, 4.85, 0.66, text, palette, fontsize=8.25)
            if i:
                arrow(ax, (cx, column[i - 1][1] - 0.34), (cx, cy + 0.34))

    box(
        ax,
        7.0,
        0.96,
        7.55,
        0.78,
        "Correlation ledger\nJAX name stack → HLO op/custom call → emitted kernel name → trace dispatch → counters",
        EVIDENCE,
        fontsize=8.8,
        weight="bold",
    )
    arrow(ax, (3.55, 2.12), (5.80, 1.36), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (10.45, 2.12), (8.20, 1.36), connectionstyle="arc3,rad=0.08")
    return save(fig, ax, "ch3-two-capture-workflow.png")


def jax_profiler_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(15.2, 9.4))
    title(
        ax,
        "How a JAX ROCm profile becomes XSpace",
        "Solid arrows carry timed events; dashed arrows carry names or compiler sideband data.",
    )

    panel(ax, 0.35, 1.02, 3.08, 7.10, "Event and metadata sources")
    panel(ax, 3.68, 1.02, 6.55, 7.10, "OpenXLA profiler session")
    panel(ax, 10.48, 1.02, 3.12, 7.10, "Export and analysis")

    sources = [
        (1.88, 7.34, "TraceMe / StepTraceAnnotation\nhost ranges", FRAMEWORK),
        (1.88, 5.92, "jax.named_scope\nJAX name stack", FRAMEWORK),
        (1.88, 4.50, "XLA executable metadata\nHLO proto + source info", XLA),
        (1.88, 2.72, "rocprofiler-sdk records\nHIP runtime · dispatch · memcpy", ROCM),
        (1.88, 1.60, "ROCTx calls made by app\nhost marker ranges", ROCM),
    ]
    for cx, cy, text, palette in sources:
        box(ax, cx, cy, 2.55, 0.72, text, palette, fontsize=7.8)

    box(
        ax,
        6.96,
        7.15,
        4.30,
        0.72,
        "jax.profiler.start_trace → ProfilerSession\ncreates registered ProfilerInterface implementations",
        XLA,
        fontsize=8.35,
        weight="bold",
    )
    box(ax, 5.34, 5.96, 2.55, 0.84, "HostTracer\ncollects host TraceMe events", FRAMEWORK, fontsize=8.1)
    box(ax, 8.58, 5.96, 2.55, 0.84, "ROCm GpuTracer\nstarts/stops RocmTracer", ROCM, fontsize=8.1)
    box(
        ax,
        8.58,
        4.48,
        2.55,
        0.92,
        "RocmTraceCollector\njoins correlation/name data;\nbuilds host and GPU XPlanes",
        ROCM,
        fontsize=7.8,
    )
    box(
        ax,
        5.34,
        3.15,
        2.55,
        0.88,
        "Compiler sideband\nHLO proto · cost/source metadata",
        XLA,
        fontsize=7.9,
    )
    box(
        ax,
        6.96,
        1.72,
        4.38,
        0.84,
        "ProfilerSession.CollectData(XSpace*)\nstops collectors, merges planes, normalizes timestamps",
        XLA,
        fontsize=8.15,
        weight="bold",
    )

    arrow(ax, (6.96, 6.77), (5.55, 6.39))
    arrow(ax, (6.96, 6.77), (8.38, 6.39))
    arrow(ax, (3.18, 7.34), (4.02, 6.16))
    arrow(ax, (3.18, 2.72), (7.28, 5.00))
    arrow(ax, (8.58, 5.52), (8.58, 4.96))
    arrow(ax, (3.18, 1.60), (7.28, 4.32))
    arrow(ax, (3.18, 5.92), (7.28, 4.62), dashed=True)
    arrow(ax, (3.18, 4.50), (4.05, 3.24), dashed=True)
    arrow(ax, (5.34, 5.50), (6.15, 2.14))
    arrow(ax, (8.58, 3.98), (7.78, 2.14))
    arrow(ax, (5.34, 2.69), (6.13, 2.08), dashed=True)

    box(ax, 12.04, 6.74, 2.52, 0.86, "XSpace protobuf\nhost + device + sideband planes", EVIDENCE, fontsize=8.2)
    box(ax, 12.04, 5.12, 2.52, 0.78, "TensorBoard profile directory\n.xplane.pb + HLO protos", EVIDENCE, fontsize=8.0)
    box(ax, 12.04, 3.54, 2.52, 0.78, "XProf converters\ntrace, op, kernel, memory stats", EVIDENCE, fontsize=8.0)
    box(ax, 12.04, 1.90, 2.52, 0.86, "Views and exports\nTrace Viewer · Kernel Stats\nRoofline · CSV", EVIDENCE, fontsize=8.0)
    arrow(ax, (9.17, 1.72), (10.78, 6.55), connectionstyle="arc3,rad=-0.18")
    arrow(ax, (12.04, 6.28), (12.04, 5.53))
    arrow(ax, (12.04, 4.71), (12.04, 3.95))
    arrow(ax, (12.04, 3.13), (12.04, 2.35))

    return save(fig, ax, "ch3-jax-profiler-pipeline.png")


def xspace_structure() -> Path:
    fig, ax = plt.subplots(figsize=(15.2, 9.4))
    title(
        ax,
        "XSpace stores parallel timelines and shared metadata",
        "Schema view based on OpenXLA's xplane.proto; examples are illustrative.",
    )

    box(
        ax,
        2.00,
        7.35,
        3.10,
        0.92,
        "XSpace\nrepeated planes · hostnames\nerrors · warnings",
        EVIDENCE,
        fontsize=8.7,
        weight="bold",
    )

    planes = [
        (1.38, 5.68, "/host:CPU\nthreads and JAX ranges", FRAMEWORK),
        (4.32, 5.68, "/device:GPU:0\nstreams and kernels", ROCM),
        (7.26, 5.68, "sideband planes\ntask env · scope tree", XLA),
    ]
    for cx, cy, text, palette in planes:
        box(ax, cx, cy, 2.52, 0.86, text, palette, fontsize=8.0)
        arrow(ax, (2.00, 6.86), (cx, 6.12), connectionstyle="arc3,rad=0.05")

    box(
        ax,
        4.32,
        4.07,
        3.20,
        0.96,
        "XPlane\nid · name · plane stats\nmaps: event_metadata, stat_metadata\nrepeated XLine",
        ROCM,
        fontsize=8.1,
        weight="bold",
    )
    arrow(ax, (4.32, 5.22), (4.32, 4.58))

    box(
        ax,
        4.32,
        2.34,
        3.20,
        0.98,
        "XLine\nid · display_id · name\ntimestamp_ns · duration_ps\nrepeated XEvent",
        NEUTRAL,
        fontsize=8.1,
        weight="bold",
    )
    arrow(ax, (4.32, 3.56), (4.32, 2.85))

    box(
        ax,
        4.32,
        0.86,
        3.20,
        0.84,
        "XEvent\nmetadata_id · offset_ps · duration_ps\nrepeated XStat",
        EXECUTION,
        fontsize=8.1,
        weight="bold",
    )
    arrow(ax, (4.32, 1.83), (4.32, 1.30))

    panel(ax, 8.55, 0.78, 5.05, 7.30, "How one kernel event is represented")
    box(ax, 11.08, 6.92, 3.92, 0.70, "XPlane  /device:GPU:0", ROCM, fontsize=8.5)
    box(ax, 11.08, 5.66, 3.92, 0.70, "XLine  Stream 2 / Queue 1", NEUTRAL, fontsize=8.5)
    box(
        ax,
        11.08,
        4.22,
        3.92,
        0.88,
        "XEvent  hipBLASLt GEMM\nstart = line timestamp + offset\nduration = end − start",
        EXECUTION,
        fontsize=8.2,
    )
    box(
        ax,
        11.08,
        2.68,
        3.92,
        0.92,
        "XStats on the event\ncorrelation_id · framework/HLO op\nkernel details · ROCTx range",
        EVIDENCE,
        fontsize=8.15,
    )
    box(
        ax,
        11.08,
        1.28,
        3.92,
        0.76,
        "Metadata maps deduplicate names and keys;\nevents refer to them by integer ID.",
        XLA,
        fontsize=8.1,
    )
    for y0, y1 in ((6.56, 6.02), (5.30, 4.68), (3.76, 3.16), (2.20, 1.68)):
        arrow(ax, (11.08, y0), (11.08, y1))

    return save(fig, ax, "ch3-xspace-structure.png")


def roofline() -> Path:
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    intensity = np.logspace(-1, 5, 500)
    peak_tflops = 2516.6
    hbm_tb_s = 8.0
    performance = np.minimum(peak_tflops, intensity * hbm_tb_s)
    ridge = peak_tflops / hbm_tb_s

    ax.loglog(intensity, performance, color=XLA[1], linewidth=2.5, label="BF16 roofline")
    ax.loglog(intensity, intensity * hbm_tb_s, color=ROCM[1], linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axhline(peak_tflops, color=FRAMEWORK[1], linewidth=1.2, linestyle="--", alpha=0.8)
    ax.axvline(ridge, color=MUTED, linewidth=1.0, linestyle=":")
    ax.scatter([ridge], [peak_tflops], color=INK, s=30, zorder=5)
    ax.text(ridge * 1.12, peak_tflops * 0.78, "ridge ≈ 315 FLOP/byte", fontsize=9, color=INK)
    ax.text(0.16, 1.75, "HBM ceiling: 8 TB/s", fontsize=9, color=ROCM[1], rotation=35)
    ax.text(2.2e3, peak_tflops * 1.10, "BF16 matrix peak: 2.5166 PFLOP/s", fontsize=9, color=FRAMEWORK[1])
    ax.text(1.1, 600, "memory-bound", fontsize=10, color=MUTED)
    ax.text(2.0e3, 1.0e3, "compute-bound", fontsize=10, color=MUTED)
    ax.set_xlabel("Arithmetic intensity (FLOP/byte at HBM)", fontsize=10)
    ax.set_ylabel("Attainable performance (TFLOP/s)", fontsize=10)
    ax.set_title(
        "Theoretical MI355X BF16/HBM roofline",
        loc="left",
        y=1.08,
        fontsize=14,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.0,
        1.025,
        "Published peaks only; no measured kernel points.",
        transform=ax.transAxes,
        fontsize=9,
        color=MUTED,
        va="bottom",
    )
    ax.grid(True, which="both", color="#dfe3e6", linewidth=0.6, alpha=0.7)
    ax.set_xlim(1e-1, 1e5)
    ax.set_ylim(5e-1, 5e3)
    ax.spines[["top", "right"]].set_visible(False)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ch3-roofline-mi355x.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def mixtral_forward() -> Path:
    fig, ax = plt.subplots(figsize=(16.0, 9.5))
    title(
        ax,
        "Mixtral 8x22B forward attribution map",
        "One decoder layer is expanded; the bracketed block repeats 56 times.",
    )

    box(ax, 1.00, 7.82, 1.50, 0.62, "token IDs", NEUTRAL, fontsize=8.4)
    box(ax, 2.85, 7.82, 1.65, 0.62, "embedding\n[·, ·, 6144]", FRAMEWORK, fontsize=8.1)
    arrow(ax, (1.76, 7.82), (2.00, 7.82))

    panel(ax, 0.45, 1.46, 13.10, 5.72, "MixtralDecoderLayer × 56")
    y_top, y_bottom = 6.45, 2.20
    nodes = [
        (1.25, 5.55, 1.55, "pre-attn\nRMSNorm", NEUTRAL),
        (3.10, 5.55, 1.65, "Q / K / V\nprojections", FRAMEWORK),
        (5.05, 5.55, 1.65, "RoPE + fused\nattention core", EVIDENCE),
        (7.00, 5.55, 1.65, "output\nprojection", FRAMEWORK),
        (8.95, 5.55, 1.65, "attention\nresidual", NEUTRAL),
        (10.90, 5.55, 1.65, "post-attn\nRMSNorm", NEUTRAL),
        (12.62, 5.55, 1.25, "hidden\nstates", NEUTRAL),
        (12.62, 3.93, 1.25, "gate\nlogits", FRAMEWORK),
        (10.90, 3.93, 1.65, "top-2 +\nroute weights", EVIDENCE),
        (8.95, 3.93, 1.65, "dispatch mask /\ntoken movement", ROCM),
        (7.00, 3.93, 1.65, "wi_0 · wi_1\nexpert GEMMs", FRAMEWORK),
        (5.05, 3.93, 1.65, "ffn_act\nSiLU × linear", NEUTRAL),
        (3.10, 3.93, 1.65, "wo\nexpert GEMM", FRAMEWORK),
        (1.25, 3.93, 1.55, "combine +\nroute weights", ROCM),
        (3.10, 2.48, 1.65, "MoE\nresidual", NEUTRAL),
    ]
    for cx, cy, width, text, palette in nodes:
        box(ax, cx, cy, width, 0.68, text, palette, fontsize=7.7)

    top_chain = nodes[:7]
    for left_node, right_node in zip(top_chain, top_chain[1:]):
        arrow(ax, (left_node[0] + left_node[2] / 2, left_node[1]), (right_node[0] - right_node[2] / 2, right_node[1]))
    arrow(ax, (12.62, 5.20), (12.62, 4.28))
    lower_chain = nodes[7:14]
    for right_node, left_node in zip(lower_chain, lower_chain[1:]):
        arrow(ax, (right_node[0] - right_node[2] / 2, right_node[1]), (left_node[0] + left_node[2] / 2, left_node[1]))
    arrow(ax, (1.25, 3.58), (2.32, 2.70), connectionstyle="arc3,rad=-0.08")

    arrow(ax, (2.85, 7.50), (1.25, 5.91), connectionstyle="arc3,rad=0.12")
    arrow(ax, (2.85, 7.50), (8.95, 5.91), dashed=True, connectionstyle="arc3,rad=-0.18")
    arrow(ax, (8.95, 5.20), (3.10, 2.83), dashed=True, connectionstyle="arc3,rad=0.22")

    box(ax, 5.62, 0.78, 2.05, 0.62, "final RMSNorm", NEUTRAL, fontsize=8.1)
    box(ax, 8.05, 0.78, 2.05, 0.62, "LM head", FRAMEWORK, fontsize=8.1)
    box(ax, 10.48, 0.78, 2.05, 0.62, "cross-entropy loss", EXECUTION, fontsize=8.1)
    arrow(ax, (3.94, 2.29), (5.35, 1.10), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (6.66, 0.78), (7.01, 0.78))
    arrow(ax, (9.09, 0.78), (9.44, 0.78))

    ax.text(13.78, y_top, "Attribution anchors", fontsize=9.2, weight="bold", color=INK, ha="left")
    ax.text(
        13.78,
        y_top - 0.38,
        "Flax/module paths\n"
        "pre_self_attention_layer_norm\n"
        "self_attention\n"
        "post_self_attention_layer_norm\n"
        "MoeBlock_0\n\n"
        "Existing JAX scopes\n"
        "dispatch · wi_0 · wi_1\n"
        "ffn_act · wo · combine\n"
        "weight_sum",
        fontsize=7.7,
        color=MUTED,
        ha="left",
        va="top",
        linespacing=1.45,
    )
    ax.add_patch(
        Rectangle(
            (0.72, y_bottom - 0.20),
            12.45,
            y_top - y_bottom + 0.42,
            facecolor="none",
            edgecolor="#a4adb5",
            linestyle=(0, (4, 3)),
            linewidth=0.9,
            zorder=1,
        )
    )
    return save(fig, ax, "ch3-mixtral-forward.png", xlim=(0, 16.5))


def mixtral_backward() -> Path:
    fig, ax = plt.subplots(figsize=(16.0, 9.5))
    title(
        ax,
        "Mixtral 8x22B backward attribution map",
        "The reverse pass creates input gradients, parameter gradients, and communication around sharded state.",
    )

    box(ax, 1.02, 7.75, 1.55, 0.68, "loss gradient", EXECUTION, fontsize=8.2)
    box(ax, 3.05, 7.75, 1.75, 0.68, "LM head VJP\n+ weight grad", FRAMEWORK, fontsize=7.9)
    box(ax, 5.22, 7.75, 1.75, 0.68, "final norm VJP", NEUTRAL, fontsize=8.0)
    arrow(ax, (1.80, 7.75), (2.15, 7.75))
    arrow(ax, (3.94, 7.75), (4.33, 7.75))

    panel(ax, 0.45, 1.68, 13.18, 5.38, "Reverse through MixtralDecoderLayer × 56")
    nodes = [
        (12.55, 6.18, "residual grad\nsplit + sum", NEUTRAL),
        (10.43, 6.18, "combine /\nunpermute VJP", ROCM),
        (8.31, 6.18, "wo VJP\ndX + dW", FRAMEWORK),
        (6.19, 6.18, "SwiGLU VJP", NEUTRAL),
        (4.07, 6.18, "wi_0 / wi_1 VJPs\ndX + dW", FRAMEWORK),
        (1.72, 6.18, "reverse dispatch\n+ router gradient", ROCM),
        (1.72, 3.60, "post-attn\nRMSNorm VJP", NEUTRAL),
        (4.07, 3.60, "residual grad\nsplit + sum", NEUTRAL),
        (6.19, 3.60, "output projection VJP\ndX + dW", FRAMEWORK),
        (8.31, 3.60, "attention backward\ndQ · dK · dV", EVIDENCE),
        (10.43, 3.60, "Q / K / V VJPs\ndX + dW", FRAMEWORK),
        (12.55, 3.60, "pre-attn\nRMSNorm VJP", NEUTRAL),
    ]
    widths = [1.75, 1.75, 1.75, 1.65, 1.92, 1.92, 1.82, 1.75, 1.92, 1.82, 1.92, 1.75]
    for (cx, cy, text, palette), width in zip(nodes, widths):
        box(ax, cx, cy, width, 0.76, text, palette, fontsize=7.55)
    for i in range(5):
        left = nodes[i]
        right = nodes[i + 1]
        arrow(ax, (left[0] - widths[i] / 2, left[1]), (right[0] + widths[i + 1] / 2, right[1]))
    arrow(ax, (1.72, 5.79), (1.72, 3.99))
    for i in range(6, 11):
        left = nodes[i]
        right = nodes[i + 1]
        arrow(ax, (left[0] + widths[i] / 2, left[1]), (right[0] - widths[i + 1] / 2, right[1]))

    arrow(ax, (5.22, 7.40), (12.28, 6.56), connectionstyle="arc3,rad=-0.11")
    box(
        ax,
        6.98,
        2.36,
        4.12,
        0.74,
        "parameter-gradient leaves\nattention · router · experts · norms",
        EVIDENCE,
        fontsize=8.2,
        weight="bold",
    )
    for cx in (3.05, 4.07, 6.19, 8.31, 10.43):
        arrow(ax, (cx, 3.20 if cx >= 6.19 else 5.78), (6.98, 2.74), dashed=True, connectionstyle="arc3,rad=0.05")

    box(ax, 4.75, 0.83, 2.62, 0.72, "gradient collectives\nby mesh/sharding", ROCM, fontsize=8.1)
    box(ax, 8.18, 0.83, 2.62, 0.72, "AdamW update\nmoments + parameters", XLA, fontsize=8.1)
    box(ax, 11.61, 0.83, 2.62, 0.72, "next-step state", EXECUTION, fontsize=8.1)
    arrow(ax, (6.98, 1.97), (5.19, 1.20), connectionstyle="arc3,rad=0.08")
    arrow(ax, (6.08, 0.83), (6.86, 0.83))
    arrow(ax, (9.51, 0.83), (10.29, 0.83))

    ax.text(13.88, 6.60, "Read the backward trace by", fontsize=9.0, weight="bold", color=INK)
    ax.text(
        13.88,
        6.18,
        "• VJP/custom-VJP names\n"
        "• transpose/gradient GEMM shapes\n"
        "• fused-attention bwd target\n"
        "• reverse token movement\n"
        "• gradient collectives\n"
        "• optimizer fusions\n\n"
        "Dashed arrows collect\ngradient leaves; they are\nnot timeline dependencies.",
        fontsize=7.8,
        color=MUTED,
        ha="left",
        va="top",
        linespacing=1.45,
    )
    return save(fig, ax, "ch3-mixtral-backward.png", xlim=(0, 16.6))


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
    subprocess.run([graphviz, "-Tsvg", str(dot), "-o", str(path)], check=True)
    return path


if __name__ == "__main__":
    outputs = [
        roofline(),
        two_capture_workflow(),
        jax_profiler_pipeline(),
        xspace_structure(),
        mixtral_forward(),
        mixtral_backward(),
        render_hlo_fixture(),
    ]
    for output in outputs:
        print(f"  wrote {output.relative_to(ROOT)}")
