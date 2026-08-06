"""Environment provenance for every measurement in the book.

Appendix B promises that each [measured] number carries the stack it was taken
on. This module makes that promise mechanical: every benchmark writes one
`env.json` next to its results, and nothing has to be remembered.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Prefixes worth recording in full. The container bakes in XLA_FLAGS and a
# large NVTE_/HSA_ set that silently shape every number taken inside it.
ENV_PREFIXES = (
    "XLA_",
    "JAX_",
    "NVTE_",
    "HSA_",
    "HIP_",
    "GPU_",
    "ROCM_",
    "ROCR_",
    "ROCPROFILER_",
    "NCCL_",
    "RCCL_",
    "TF_",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
)

_MANIFEST = Path("/workspace/.manifest")


def _run(cmd: list[str] | str, timeout: int = 60) -> str | None:
    """Best-effort command capture. Returns None when the tool is absent."""
    exe = cmd[0] if isinstance(cmd, list) else cmd.split()[0]
    if shutil.which(exe) is None:
        return None
    try:
        out = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"<failed: {type(exc).__name__}: {exc}>"
    return (out.stdout + out.stderr).strip()


def rocm_version() -> str | None:
    p = Path("/opt/rocm/.info/version")
    return p.read_text().strip() if p.exists() else None


def rccl_version() -> str | None:
    """Pull the version banner out of the librccl binary.

    The container ships a custom RCCL build rather than the one in the ROCm
    tarball, so the package version is not the whole story.
    """
    for cand in ("/opt/rocm/lib/librccl.so", "/opt/rocm/lib/librccl.so.1"):
        if Path(cand).exists():
            out = _run(f"strings {cand} | grep -m1 '^RCCL version'")
            if out:
                return out.strip()
    return None


def container() -> dict[str, Any]:
    """Read the image build manifest, which is the real reproducibility anchor."""
    info: dict[str, Any] = {"tag": os.environ.get("CONTAINER_TAG", "rocm/jax-training:maxtext-v26.5")}
    ver = _MANIFEST / "training_docker_version"
    if ver.exists():
        info["training_docker_version"] = ver.read_text().strip()
    info["manifest_present"] = _MANIFEST.exists()
    return info


def gpu_state() -> dict[str, Any]:
    """Clocks, power and throttle status.

    Appendix B does not lock clocks and says so; this is what lets a reader see
    whether a gap is thermal. Call before and after a long measurement.
    """
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "clocks": _run(["rocm-smi", "--showclocks"]),
        "power": _run(["rocm-smi", "--showpower"]),
        "temperature": _run(["rocm-smi", "--showtemp"]),
        "perf_level": _run(["rocm-smi", "--showperflevel"]),
    }


# The level indicator is a digit under load and "S" when the GPU is asleep, so
# an idle sample reads `sclk clock level: S: (132Mhz)`.
_SCLK_RE = re.compile(r"GPU\[(\d+)\][^\n]*?\bsclk clock level:\s*\w+:\s*\((\d+)Mhz\)", re.I)
_POWER_RE = re.compile(r"GPU\[(\d+)\].*?Power \(W\):\s*([\d.]+)", re.I)


def sample_clocks(gpu: int = 0) -> dict[str, float] | None:
    """One (sclk, power) reading for a GPU, in MHz and watts."""
    out = _run(["rocm-smi", "--showclocks", "--showpower"], timeout=15)
    if not out:
        return None
    sclk = {int(g): float(v) for g, v in _SCLK_RE.findall(out)}
    power = {int(g): float(v) for g, v in _POWER_RE.findall(out)}
    if gpu not in sclk and gpu not in power:
        return None
    return {"sclk_mhz": sclk.get(gpu, 0.0), "power_w": power.get(gpu, 0.0)}


class ClockSampler:
    """Poll clocks in the background for the duration of a measurement.

    Appendix B says to watch clocks alongside a long measurement rather than
    only afterwards, and the reason is concrete: MI300X is power-limited at
    750 W under a dense bf16 matmul and settles well below its 2100 MHz boost
    clock, so a before-and-after reading taken at idle tells you nothing about
    the clock the kernel actually ran at.
    """

    def __init__(self, gpu: int = 0, interval: float = 1.0):
        self.gpu = gpu
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        # Sample once straight away. A ten-iteration loop over a 170 us kernel
        # lasts under 2 ms and rocm-smi takes about half a second per call, so
        # without this a short measurement records no clock at all.
        first = sample_clocks(self.gpu)
        if first:
            self.samples.append(first)
        while not self._stop.wait(self.interval):
            s = sample_clocks(self.gpu)
            if s:
                self.samples.append(s)

    def start(self) -> "ClockSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"samples": 0}
        sclk = [s["sclk_mhz"] for s in self.samples if s["sclk_mhz"]]
        power = [s["power_w"] for s in self.samples if s["power_w"]]
        out: dict[str, Any] = {"samples": len(self.samples), "gpu": self.gpu}
        if sclk:
            out |= {
                "sclk_mhz_median": statistics.median(sclk),
                "sclk_mhz_min": min(sclk),
                "sclk_mhz_max": max(sclk),
                # What the roofline should have used, against the 2100 MHz boost
                # figure the data sheet quotes.
                "sclk_fraction_of_boost": statistics.median(sclk) / 2100.0,
            }
        if power:
            out |= {"power_w_median": statistics.median(power), "power_w_max": max(power)}
        return out

    def __enter__(self) -> "ClockSampler":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def hardware() -> dict[str, Any]:
    """Product, partitioning mode and topology.

    Partitioning is load-bearing: a CPX device invalidates every FLOP figure in
    Chapter 2 by exactly the fraction of CUs it holds.
    """
    return {
        "product": _run(["rocm-smi", "--showproductname"]),
        "compute_partition": _run(["rocm-smi", "--showcomputepartition"]),
        "memory_partition": _run(["rocm-smi", "--showmemorypartition"]),
        "vram": _run(["rocm-smi", "--showmeminfo", "vram"]),
        "topology": _run(["rocm-smi", "--showtopo"]),
    }


def jax_state() -> dict[str, Any]:
    """JAX's own view. Imported lazily so provenance works without a backend."""
    try:
        import jax
        import jaxlib
    except ImportError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    try:
        devices = [
            {
                "id": d.id,
                "kind": d.device_kind,
                "platform": d.platform,
                "repr": str(d),
            }
            for d in jax.devices()
        ]
    except Exception as exc:  # a backend failure is itself worth recording
        devices = [{"error": f"{type(exc).__name__}: {exc}"}]
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "default_backend": jax.default_backend(),
        "device_count": len(devices),
        "devices": devices,
    }


def packages(names: tuple[str, ...] = ()) -> dict[str, str]:
    """Installed versions. Defaults to the set the book's claims depend on."""
    wanted = names or (
        "jax",
        "jaxlib",
        "jax-rocm7-pjrt",
        "jax-rocm7-plugin",
        "libtpu",
        "flax",
        "optax",
        "orbax-checkpoint",
        "xprof",
        "tensorboard-plugin-profile",
        "tokamax",
        "transformer_engine_rocm_jax",
        "maxtext",
        "aqtp",
        "qwix",
        "grain",
        "transformers",
        "datasets",
        "numpy",
        "ml_dtypes",
    )
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return {}
    out = {}
    for name in wanted:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "<absent>"
        except Exception as exc:
            out[name] = f"<error: {type(exc).__name__}>"
    return out


def pip_freeze() -> list[str]:
    out = _run([sys.executable, "-m", "pip", "freeze"], timeout=120)
    return out.splitlines() if out else []


def relevant_env() -> dict[str, str]:
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(ENV_PREFIXES)
    }


def xla_flags() -> dict[str, str | bool]:
    """Parse XLA_FLAGS into a dict so an A/B diff is readable.

    An empty value is meaningful here: the container ships
    `--xla_gpu_enable_command_buffer=''`, which is Appendix A's workaround and
    is not the same thing as `false`.
    """
    raw = os.environ.get("XLA_FLAGS", "")
    parsed: dict[str, str | bool] = {}
    for tok in re.findall(r"--[\w.]+(?:=(?:'[^']*'|\"[^\"]*\"|\S*))?", raw):
        if "=" in tok:
            key, _, val = tok.partition("=")
            parsed[key.lstrip("-")] = val.strip("'\"")
        else:
            parsed[tok.lstrip("-")] = True
    return parsed


def snapshot(extra: dict[str, Any] | None = None, *, freeze: bool = True) -> dict[str, Any]:
    """The full provenance record written alongside every result."""
    snap: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "container": container(),
        "rocm_version": rocm_version(),
        "rccl_version": rccl_version(),
        "hip_version": _run(["hipconfig", "--version"]),
        "packages": packages(),
        "jax": jax_state(),
        "hardware": hardware(),
        "gpu_state_before": gpu_state(),
        "env": relevant_env(),
        "xla_flags": xla_flags(),
        "argv": sys.argv,
    }
    if freeze:
        snap["pip_freeze"] = pip_freeze()
    if extra:
        snap["extra"] = extra
    return snap


def write(run_dir: str | Path, extra: dict[str, Any] | None = None, *, freeze: bool = True) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "env.json"
    path.write_text(json.dumps(snapshot(extra, freeze=freeze), indent=2, default=str))
    return path


if __name__ == "__main__":
    print(json.dumps(snapshot(freeze=False), indent=2, default=str))
