#!/usr/bin/env python3
"""Summarize the Chapter 6 Llama rematerialization artifacts.

Timing summaries use synchronized steps 10--29. Memory values preserve the
runner's historical ``GB`` label, although the implementation divides by
1024**3 and therefore reports GiB.

Run from the repository root:

    python tools/analyze_ch6_results.py
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

ONE_GPU = Path("/tmp/ch6-remat-1gpu-20260916")
FSDP8 = Path("/tmp/archive/llama7b/fsdp8-remat-sweep-20260911")
POLICIES = ("none", "minimal", "full")

STEP = re.compile(
    r"completed step: (?P<step>\d+), seconds: (?P<seconds>[0-9.]+), "
    r"TFLOP/s/device: (?P<tflops>[0-9.]+), "
    r"Tokens/s/device: (?P<tokens>[0-9.]+)"
)
MEMORY = re.compile(
    r"Total memory size: (?P<total>[0-9.]+) GB, Output size: (?P<output>[0-9.]+) GB, "
    r"Temp size: (?P<temp>[0-9.]+) GB"
)
PEAK = re.compile(r"Peak \(GB\) (?P<peak>[0-9.]+) on rocm:0")
SHAPE = re.compile(r"\b(bf16|f16|f32|s32|u32|pred|s8|u8)\[([0-9,]+)\]")
WIDTH = {"bf16": 2, "f16": 2, "f32": 4, "s32": 4, "u32": 4, "pred": 1, "s8": 1, "u8": 1}
COLLECTIVE_START = re.compile(r"\s(all-gather|all-reduce|reduce-scatter|collective-permute)-start\(")


def summarize_log(path: Path) -> dict[str, float]:
    text = path.read_text()
    steps = [
        {key: float(value) for key, value in match.groupdict().items()}
        for match in STEP.finditer(text)
    ]
    steady = [row for row in steps if row["step"] >= 10]
    if len(steady) != 20:
        raise RuntimeError(f"{path}: expected 20 steady steps, found {len(steady)}")
    memory = MEMORY.search(text)
    peaks = list(PEAK.finditer(text))
    if memory is None or not peaks:
        raise RuntimeError(f"{path}: missing memory fields")
    seconds = [row["seconds"] for row in steady]
    return {
        "median_s": statistics.median(seconds),
        "mean_s": statistics.mean(seconds),
        "stdev_s": statistics.stdev(seconds),
        "tokens_s": 16384 / statistics.median(seconds),
        "compiled_gib": float(memory.group("total")),
        "temp_gib": float(memory.group("temp")),
        "peak_gib": float(peaks[-1].group("peak")),
    }


def result_paths() -> dict[str, dict[str, Path]]:
    return {
        "1 GPU": {policy: ONE_GPU / f"{policy}.log" for policy in POLICIES},
        "FSDP-8": {
            "none": FSDP8 / "sweep-remat-none.log",
            "minimal": FSDP8 / "sweep-remat-minimal.log",
            "full": FSDP8 / "sweep-remat-full.log",
        },
    }


def shape_bytes(dtype: str, dimensions: str) -> int:
    elements = 1
    for dimension in dimensions.split(","):
        elements *= int(dimension)
    return elements * WIDTH[dtype]


def collective_summary(path: Path) -> tuple[int, float, int]:
    count = 0
    output_bytes = 0
    rematted = 0
    for line in path.read_text().splitlines():
        if COLLECTIVE_START.search(line) is None:
            continue
        count += 1
        result = line.split(" all-", 1)[0]
        shapes = SHAPE.findall(result)
        # Async start returns ((operands), (results)); charge the result half.
        if len(shapes) % 2 == 0:
            shapes = shapes[len(shapes) // 2 :]
        output_bytes += sum(shape_bytes(dtype, dims) for dtype, dims in shapes)
        rematted += "checkpoint/rematted_computation" in line
    return count, output_bytes / 1024**3, rematted


def main() -> None:
    layers, batch, sequence = 32, 4, 4096
    model_width, mlp_width = 4096, 11008
    heads, head_dim, vocab = 32, 128, 32000
    named_gib = (
        layers * batch * sequence * 2 * (6 * model_width + 2 * mlp_width) / 1024**3
    )
    projection_flops = (
        (
            6 * batch * sequence * model_width * mlp_width
            + 2 * batch * sequence * model_width * (3 * heads) * head_dim
            + 2 * batch * sequence * model_width * heads * head_dim
        )
        * layers
        / 1e12
    )
    attention_flops = (
        4 * batch * sequence**2 * heads * head_dim / 2 * layers / 1e12
    )
    vocab_flops = 2 * batch * sequence * model_width * vocab / 1e12
    print(
        f"prediction named_residual_GiB={named_gib:.3f} "
        f"full_replay_projection_TFLOP={projection_flops:.3f} "
        f"full_replay_attention_TFLOP={attention_flops:.3f} "
        f"excluded_vocab_head_TFLOP={vocab_flops:.3f}\n"
    )

    print("scope    policy    median_s  std_s   tokens/s/GPU  compiled_GiB  temp_GiB  peak_GiB")
    for scope, paths in result_paths().items():
        for policy, path in paths.items():
            row = summarize_log(path)
            print(
                f"{scope:7}  {policy:7}  {row['median_s']:8.3f}  {row['stdev_s']:6.4f}  "
                f"{row['tokens_s']:12.1f}  {row['compiled_gib']:12.1f}  "
                f"{row['temp_gib']:8.1f}  {row['peak_gib']:8.2f}"
            )

    print("\nFSDP scheduled HLO static collective starts")
    for policy, filename in (
        ("none", "hlo-none.txt"),
        ("minimal", "hlo-minimal_with_context.txt"),
        ("full", "hlo-full.txt"),
    ):
        count, output_gib, rematted = collective_summary(FSDP8 / filename)
        print(
            f"{policy:7} starts={count:2d} summed_result_GiB={output_gib:7.3f} "
            f"inside_checkpoint/rematted_computation={rematted}"
        )


if __name__ == "__main__":
    main()
