"""Run a MaxText pretraining job and extract the numbers the book quotes.

Everything in Part III is a MaxText run, and the awkward part is not launching
it but getting a comparable number out afterwards. MaxText prints a step time per
step to stdout and writes an XPlane capture; this wraps both, applies Appendix
B's convention of discarding early steps, and lands the result in the same
`results.json` shape as the microbenchmarks.

    python -m bench.maxtext_run --model llama3-8b --tag bf16
    python -m bench.maxtext_run --model llama3-8b --tag fp8 --set quantization=nanoo_fp8

Two defaults worth knowing about, both of which bite:

  * `base.yml` sets `hardware: tpu`, so `hardware=gpu` is not optional.
  * `dataset_type` defaults to `tfds` and there is no dataset in this container,
    so synthetic is the default here.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ._harness import MI300X, Run, base_parser

MAXTEXT = Path("/workspace/maxtext")
CONFIG = MAXTEXT / "src/maxtext/configs/base.yml"

# MaxText logs one of these per step.
STEP_RE = re.compile(
    r"completed step:\s*(\d+),\s*seconds:\s*([\d.]+),\s*TFLOP/s/device:\s*([\d.]+)"
    r"(?:,\s*Tokens/s/device:\s*([\d.]+))?",
    re.IGNORECASE,
)


def build_parser():
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--model", default="llama3-8b")
    p.add_argument(
        "--config",
        default=str(CONFIG),
        help="base config to start from; used to A/B a change to base.yml itself",
    )
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--skip-steps", type=int, default=10, help="steps discarded before measuring")
    p.add_argument("--per-device-batch-size", type=float, default=4.0)
    p.add_argument("--max-target-length", type=int, default=8192)
    p.add_argument("--remat-policy", default="minimal_flash")
    p.add_argument("--ici", default="fsdp=8", help="comma-separated, e.g. fsdp=4,tensor=2")
    p.add_argument(
        "--attention",
        default="cudnn_flash_te",
        help="the only setting that runs here; see the attention A/B in Chapter 8",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="any additional MaxText config override, repeatable",
    )
    p.add_argument("--timeout", type=int, default=3600)
    return p


def ici_flags(spec: str) -> list[str]:
    """Turn `fsdp=4,tensor=2` into MaxText's ici_*_parallelism flags."""
    out = []
    for part in spec.split(","):
        if not part.strip():
            continue
        axis, _, degree = part.partition("=")
        out.append(f"ici_{axis.strip()}_parallelism={degree.strip()}")
    return out


def parse_steps(log: str) -> list[dict[str, float]]:
    steps = []
    for m in STEP_RE.finditer(log):
        steps.append(
            {
                "step": int(m.group(1)),
                "seconds": float(m.group(2)),
                "tflops_per_device": float(m.group(3)),
                "tokens_per_second_per_device": float(m.group(4)) if m.group(4) else None,
            }
        )
    return steps


def summarise(steps: list[dict[str, float]], skip: int, devices: int) -> dict[str, Any]:
    import statistics

    measured = [s for s in steps if s["step"] >= skip]
    if not measured:
        return {"error": f"no steps past {skip}", "steps_seen": len(steps)}
    times = [s["seconds"] for s in measured]
    tflops = [s["tflops_per_device"] for s in measured]
    median = statistics.median(times)
    mean = statistics.fmean(times)
    return {
        "steps_measured": len(measured),
        "first_measured_step": measured[0]["step"],
        "step_s_median": median,
        "step_s_mean": mean,
        "step_s_min": min(times),
        "step_s_max": max(times),
        "median_mean_gap_pct": (mean - median) / median * 100.0 if median else 0.0,
        "tflops_per_device_median": statistics.median(tflops),
        # MaxText's own TFLOP/s figure, against the sustained-clock roofline
        # Chapter 3 measures rather than the data sheet's boost-clock one.
        "mfu_vs_boost_clock": statistics.median(tflops) * 1e12 / MI300X["bf16_flops"],
        "mfu_vs_sustained_clock": statistics.median(tflops) * 1e12 / 990e12,
        "devices": devices,
    }


def _run_child(cmd: list[str], *, timeout: int) -> tuple[int, str]:
    """Run MaxText in its own process group and never leave it behind.

    A MaxText process that outlives its parent holds every GPU's memory and the
    next run fails with `HIP_ERROR_OutOfMemory` from a completely unrelated
    place. Putting the child in its own session means we can signal the whole
    group, and the `finally` means we do it even when we are killed ourselves.
    """
    import os
    import signal

    proc = subprocess.Popen(
        cmd,
        cwd=str(MAXTEXT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env={**os.environ, "JAX_PLATFORMS": "rocm"},
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out or ""
    except subprocess.TimeoutExpired:
        out = "<timed out>"
        return 124, out
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def main() -> int:
    args = build_parser().parse_args()

    import jax  # noqa: F401  (only to count devices before we hand off)

    devices = jax.device_count()

    tag = args.tag or args.model
    with Run("maxtext", tag=tag, root=args.root, freeze=not args.no_freeze) as run:
        out_dir = run.dir / "maxtext"
        out_dir.mkdir(parents=True, exist_ok=True)

        overrides = [
            f"model_name={args.model}",
            "hardware=gpu",
            "dataset_type=synthetic",
            "enable_checkpointing=false",
            f"steps={args.steps}",
            f"per_device_batch_size={args.per_device_batch_size}",
            f"max_target_length={args.max_target_length}",
            f"remat_policy={args.remat_policy}",
            f"run_name={tag}",
            f"base_output_directory={out_dir}",
            *ici_flags(args.ici),
        ]
        if args.attention:
            overrides.append(f"attention={args.attention}")
            # TransformerEngine attention refuses to configure without this and
            # the error names pydantic rather than the cause.
            if "te" in args.attention and not any(
                s.startswith("max_segments_per_seq") for s in args.set
            ):
                overrides.append("max_segments_per_seq=1")
        if args.trace:
            overrides += [
                "profiler=xplane",
                f"skip_first_n_steps_for_profiler={args.skip_steps}",
                "profiler_steps=5",
            ]
        overrides += args.set

        # The child runs with cwd=/workspace/maxtext, so a relative config path
        # would resolve against the wrong tree.
        config = str(Path(args.config).resolve())
        cmd = [sys.executable, "-m", "maxtext.trainers.pre_train.train", config, *overrides]
        run.note("config", config)
        run.note("command", " ".join(cmd))
        run.note("model", args.model)
        run.note("ici", args.ici)
        run.note("overrides", overrides)
        print("  " + " ".join(cmd[3:]))

        started = time.time()
        returncode, log = _run_child(cmd, timeout=args.timeout)
        elapsed = time.time() - started
        (run.dir / "maxtext.log").write_text(log)
        proc = subprocess.CompletedProcess(cmd, returncode, "", "")

        steps = parse_steps(log)
        summary = summarise(steps, args.skip_steps, devices)
        run.note("returncode", proc.returncode)
        run.note("wall_seconds", elapsed)
        run.note("steps", steps)
        run.note("summary", summary)

        if proc.returncode != 0:
            tail = "\n".join(log.strip().splitlines()[-25:])
            run.note("failure_tail", tail)
            print(f"\n  FAILED (exit {proc.returncode}) after {elapsed:.0f}s\n{tail}")
            return proc.returncode

        if "error" in summary:
            print(f"  {summary}")
            return 1

        print(
            f"  step {summary['step_s_median'] * 1e3:.1f} ms median over "
            f"{summary['steps_measured']} steps, "
            f"{summary['tflops_per_device_median']:.1f} TFLOP/s/device, "
            f"MFU {summary['mfu_vs_boost_clock']:.1%} of the data sheet, "
            f"{summary['mfu_vs_sustained_clock']:.1%} of the clock it holds"
        )

        if args.trace:
            _report_trace(run, out_dir)

    return 0


def _report_trace(run: Run, out_dir: Path) -> None:
    """Pull the collective share and the per-scope breakdown out of the capture."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from parse_xplane import (  # noqa: E402
        find_xplanes,
        overlap_analysis,
        phase_breakdown,
        scope_breakdown,
    )

    try:
        paths = find_xplanes(out_dir)
    except SystemExit:
        print("  no xplane written; check profiler settings")
        return

    overlap = overlap_analysis(paths)
    phases = phase_breakdown(paths)
    scopes = scope_breakdown(paths)
    run.note("overlap", overlap)
    run.note("phases", phases)
    run.note("scopes", scopes[:25])
    print(
        f"  collectives {phases['shares']['collective']:.1%} of device kernel time, "
        f"{overlap['hidden_fraction']:.1%} of it hidden"
    )
    for s in scopes[:8]:
        print(f"      {s['share']:>6.1%}  {s['scope'][:48]}")


if __name__ == "__main__":
    raise SystemExit(main())
