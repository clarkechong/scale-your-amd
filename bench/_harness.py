"""The measurement protocol from Appendix B, as code.

Three iterations discarded, ten measured, median reported, `block_until_ready`
on every call. Writing it once here means no benchmark in `bench/` can quietly
disagree with the appendix, and every result lands in the same JSON shape so
`tools/` can turn a directory of runs into a table.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from . import _env

WARMUP = 3
REPEATS = 10

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "runs"

# Chapter 2's hardware constants, in one place so a prediction in any benchmark
# is derived rather than retyped. Dense peaks, SPX, all 304 CUs.
MI300X = {
    "cus": 304,
    "clock_ghz": 2.10,
    "bf16_flops": 1307.4e12,
    "fp16_flops": 1307.4e12,
    "fp8_flops": 2614.9e12,
    "fp32_flops": 163.4e12,
    "hbm_bytes_per_s": 5.3e12,
    "hbm_bytes": 192 * 1024**3,
    "xgmi_links": 7,
    "xgmi_link_unidir_bytes_per_s": 64e9,
    "xgmi_egress_unidir_bytes_per_s": 448e9,
    # AMD's published realised RCCL figure, about 0.7 of spec. Chapter 4 uses
    # this for anything meant to be compared against a measurement.
    "rccl_realised_bytes_per_s": 320e9,
}


@dataclass
class Measurement:
    """One benchmarked callable, timed to Appendix B's protocol."""

    name: str
    warmup: int
    repeats: int
    times_s: list[float]
    median_s: float
    mean_s: float
    stdev_s: float
    min_s: float
    max_s: float
    median_mean_gap_pct: float
    flops: float | None = None
    bytes: float | None = None
    achieved_flops_per_s: float | None = None
    achieved_bytes_per_s: float | None = None
    predicted_s: float | None = None
    ratio_to_prediction: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def noisy(self) -> bool:
        """Appendix B: a median/mean gap over a few percent is worth reporting."""
        return abs(self.median_mean_gap_pct) > 3.0

    def summary(self) -> str:
        parts = [f"{self.name}: median {_fmt_time(self.median_s)}"]
        if self.noisy:
            parts.append(f"(mean {_fmt_time(self.mean_s)}, gap {self.median_mean_gap_pct:+.1f}%)")
        if self.achieved_flops_per_s:
            parts.append(f"{self.achieved_flops_per_s / 1e12:.1f} TFLOP/s")
        if self.achieved_bytes_per_s:
            parts.append(f"{self.achieved_bytes_per_s / 1e9:.1f} GB/s")
        if self.predicted_s:
            parts.append(f"predicted {_fmt_time(self.predicted_s)} ({self.ratio_to_prediction:.2f}x)")
        return "  ".join(parts)


def _fmt_time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.3f} s"


def warm(fn: Callable[[], Any], iterations: int = WARMUP) -> None:
    """Burn the compile, the autotune and the clock ramp.

    Kept separate from `measure` so a caller can warm up *outside* a profiler
    capture. That is not a stylistic preference: with autotuning on, the first
    call runs the autotuner, whose probe kernels land in the trace and outweigh
    the kernel you meant to measure.
    """
    import jax

    for _ in range(iterations):
        jax.block_until_ready(fn())


def measure(
    fn: Callable[[], Any],
    *,
    name: str,
    warmup: int = WARMUP,
    repeats: int = REPEATS,
    flops: float | None = None,
    bytes: float | None = None,
    predicted_s: float | None = None,
    meta: dict[str, Any] | None = None,
) -> Measurement:
    """Time `fn` to the book's protocol.

    `fn` takes no arguments and returns a pytree of arrays. The harness blocks
    on the result before stopping the clock, because JAX dispatches
    asynchronously and a loop without it measures Python.

    Nothing else runs during the loop, deliberately. An earlier version polled
    `rocm-smi` here to record the clock, which costs half a second per call and
    turned a 2 ms measurement into a 10 ms one. Clocks are sampled around the
    run in `env.json` and characterised properly by `sustained_clocks`.
    """
    import jax

    warm(fn, warmup)

    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - start)

    meta = dict(meta or {})
    median = statistics.median(times)
    mean = statistics.fmean(times)
    return Measurement(
        name=name,
        warmup=warmup,
        repeats=repeats,
        times_s=times,
        median_s=median,
        mean_s=mean,
        stdev_s=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_s=min(times),
        max_s=max(times),
        median_mean_gap_pct=(mean - median) / median * 100.0 if median else 0.0,
        flops=flops,
        bytes=bytes,
        achieved_flops_per_s=flops / median if flops and median else None,
        achieved_bytes_per_s=bytes / median if bytes and median else None,
        predicted_s=predicted_s,
        ratio_to_prediction=median / predicted_s if predicted_s else None,
        meta=meta,
    )


class Run:
    """One run directory: provenance in, results out.

    Layout is `runs/<date>-<workload>[-<tag>]/` holding `env.json`,
    `results.json`, any `*.xplane.pb` under `trace/`, and HLO dumps under
    `hlo/`.
    """

    def __init__(
        self,
        workload: str,
        *,
        tag: str | None = None,
        root: str | Path | None = None,
        notes: dict[str, Any] | None = None,
        freeze: bool = True,
    ):
        self.workload = workload
        self.tag = tag
        stamp = datetime.now().strftime("%Y%m%d")
        name = f"{stamp}-{workload}" + (f"-{tag}" if tag else "")
        self.dir = Path(root or RUNS_ROOT) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir = self.dir / "trace"
        self.hlo_dir = self.dir / "hlo"
        self.measurements: list[Measurement] = []
        self.notes: dict[str, Any] = dict(notes or {})
        self._started = time.time()
        self._freeze = freeze
        _env.write(self.dir, extra={"workload": workload, "tag": tag}, freeze=freeze)
        print(f"[run] {self.dir}")

    def add(self, m: Measurement) -> Measurement:
        self.measurements.append(m)
        print(f"  {m.summary()}")
        return m

    def measure(
        self,
        fn: Callable[[], Any],
        *,
        trace: str | None = None,
        warmup: int = WARMUP,
        **kw: Any,
    ) -> Measurement:
        """Measure, optionally capturing only the measured iterations.

        When `trace` is a subdirectory name, warmup happens before the capture
        opens, so the profile holds the ten measured iterations and nothing
        else. This is the ordering Appendix B specifies.
        """
        if trace is None:
            return self.add(measure(fn, warmup=warmup, **kw))

        warm(fn, warmup)
        with self.trace(trace) as tdir:
            m = measure(fn, warmup=0, **kw)
        m.warmup = warmup
        m.meta["trace_dir"] = str(tdir)
        return self.add(m)

    def note(self, key: str, value: Any) -> None:
        self.notes[key] = value

    @contextlib.contextmanager
    def trace(self, subdir: str = "") -> Iterator[Path]:
        """Capture an XPlane profile around the block."""
        import jax

        target = self.trace_dir / subdir if subdir else self.trace_dir
        target.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(target)):
            yield target

    def finish(self) -> Path:
        """Write results.json and re-read the GPU state for the clocks section."""
        payload = {
            "workload": self.workload,
            "tag": self.tag,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.time() - self._started,
            "protocol": {"warmup": WARMUP, "repeats": REPEATS, "statistic": "median"},
            "constants": MI300X,
            "notes": self.notes,
            "measurements": [asdict(m) for m in self.measurements],
        }
        path = self.dir / "results.json"
        path.write_text(json.dumps(payload, indent=2, default=str))

        env_path = self.dir / "env.json"
        if env_path.exists():
            env = json.loads(env_path.read_text())
            env["gpu_state_after"] = _env.gpu_state()
            env_path.write_text(json.dumps(env, indent=2, default=str))

        print(f"[run] wrote {path}")
        return path

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()


def sustained_clocks(
    fn: Callable[[], Any],
    *,
    seconds: float = 25.0,
    gpu: int = 0,
    interval: float = 3.0,
) -> dict[str, Any]:
    """Run `fn` back to back and watch what the clock settles to.

    Separate from `measure` because the two answer different questions. Ten
    iterations of a 170 microsecond kernel last under two milliseconds, which is
    not long enough for the power controller to react, so the clock during a
    measured loop is whatever the ramp happened to leave behind. The number that
    explains a roofline gap is the steady-state clock under continuous work, and
    that takes tens of seconds to reach.
    """
    import jax

    jax.block_until_ready(fn())
    sampler = _env.ClockSampler(gpu=gpu, interval=interval).start()
    start = time.perf_counter()
    iterations = 0
    while time.perf_counter() - start < seconds:
        jax.block_until_ready(fn())
        iterations += 1
    summary = sampler.stop()
    summary["seconds"] = time.perf_counter() - start
    summary["iterations"] = iterations
    if summary.get("sclk_mhz_median"):
        # The roofline the book quotes assumes 2100 MHz. Rescale it to the clock
        # the device actually held.
        summary["achievable_bf16_flops"] = (
            MI300X["cus"] * 2048 * summary["sclk_mhz_median"] * 1e6
        )
    return summary


_RELAUNCH_MARKER = "SCALE_YOUR_AMD_RELAUNCHED"


def configure_environment(
    *,
    xla_flags: dict[str, str] | None = None,
    visible_devices: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Set backend-init environment, re-execing if the process already started.

    XLA reads `XLA_FLAGS` once, when the backend initialises, and `jax` may
    already be imported by the time a script has parsed its arguments. Rather
    than ask every caller to remember an env prefix, re-exec ourselves with the
    right environment and a marker so it happens exactly once. Being able to say
    `--autotune 4` instead of composing an XLA_FLAGS string by hand is the
    difference between an A/B that gets run and one that does not.
    """
    if os.environ.get(_RELAUNCH_MARKER):
        return
    if not (xla_flags or visible_devices or env):
        return

    if xla_flags:
        existing = os.environ.get("XLA_FLAGS", "")
        for key, value in xla_flags.items():
            key = key if key.startswith("--") else f"--{key}"
            # Replace in place when present so the container's own flags keep
            # their position and only the arm under test changes.
            pattern = f"{key}="
            tokens = shlex.split(existing) if existing else []
            tokens = [t for t in tokens if not t.startswith(pattern) and t != key]
            tokens.append(f"{key}={value}")
            existing = " ".join(tokens)
        os.environ["XLA_FLAGS"] = existing

    if visible_devices is not None:
        os.environ["ROCR_VISIBLE_DEVICES"] = visible_devices
    for key, value in (env or {}).items():
        os.environ[key] = value

    os.environ[_RELAUNCH_MARKER] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "rocm")
    # orig_argv preserves `-m package.module`, which plain argv loses, but its
    # argv[0] is whatever the user typed rather than a path worth exec'ing.
    orig = list(getattr(sys, "orig_argv", []))
    argv = [sys.executable] + (orig[1:] if orig else sys.argv)
    os.execv(sys.executable, argv)


def base_parser(description: str) -> argparse.ArgumentParser:
    """The flags every benchmark in `bench/` shares."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--warmup", type=int, default=WARMUP, help="iterations discarded")
    p.add_argument("--repeats", type=int, default=REPEATS, help="iterations measured")
    p.add_argument("--tag", default=None, help="suffix for the run directory, e.g. an A/B arm")
    p.add_argument("--root", default=None, help="override runs/ root")
    p.add_argument("--trace", action="store_true", help="capture an XPlane profile")
    p.add_argument(
        "--no-freeze",
        action="store_true",
        help="skip pip freeze in env.json (faster for sweeps)",
    )
    return p
