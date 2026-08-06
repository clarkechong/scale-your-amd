"""What you can actually write a custom kernel with, on ROCm, in JAX, today.

Chapter 8's escalation ladder ends with "write the kernel yourself", and that
advice is only worth giving if the rungs exist. This checks them, one smoke test
each, and reports what is reachable rather than what is documented.

    python -m bench.kernels_survey
    python -m bench.kernels_survey --only pallas fusion

Each check returns a verdict and, where it fails, the actual error, because on
this platform the error message is usually the useful part.
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from ._harness import Run, base_parser, configure_environment


def build_parser():
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--only", nargs="+", default=None, help="run a subset of the checks")
    return p


def _verdict(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"ok": True, **fn()}
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc().splitlines()[-6:],
        }


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_libraries() -> dict[str, Any]:
    """Which kernel-authoring packages are installed at all."""
    from importlib import import_module
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, Any] = {}
    for name, module in (
        ("triton", "triton"),
        ("jax-triton", "jax_triton"),
        ("aiter", "aiter"),
        ("tokamax", "tokamax"),
        ("transformer_engine_rocm_jax", "transformer_engine.jax"),
    ):
        entry: dict[str, Any] = {}
        try:
            entry["version"] = version(name)
        except PackageNotFoundError:
            entry["version"] = None
        try:
            import_module(module)
            entry["importable"] = True
        except Exception as exc:
            entry["importable"] = False
            entry["import_error"] = type(exc).__name__
        out[name] = entry
    return {"packages": out}


def check_pallas() -> dict[str, Any]:
    """A trivial Pallas kernel, both ways round.

    Pallas on GPU has two lowering backends and the default one is wrong here,
    so the interesting output is which of the two works rather than whether
    Pallas works.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    def add_one_kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0

    x = jnp.arange(256, dtype=jnp.float32).reshape(16, 16)
    results: dict[str, Any] = {}

    for label, params in (("default", None), ("triton", pltriton.CompilerParams())):
        kwargs = {"compiler_params": params} if params is not None else {}
        try:
            fn = jax.jit(
                lambda a, kw=kwargs: pl.pallas_call(
                    add_one_kernel,
                    out_shape=jax.ShapeDtypeStruct(a.shape, a.dtype),
                    **kw,
                )(a)
            )
            out = jax.block_until_ready(fn(x))
            results[label] = {"ok": True, "correct": bool(jnp.allclose(out, x + 1.0))}
        except Exception as exc:
            results[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}

    return {"backends": results, "device": str(jax.devices()[0])}


def check_pallas_mosaic_gpu() -> dict[str, Any]:
    """Whether the Mosaic GPU lowering path exists here.

    Relevant because MaxText's megablox kernels are Mosaic kernels, and Chapter 7
    found they will not lower on this platform.
    """
    from jax._src.pallas import core as pallas_core

    available = {}
    for platform in ("gpu", "tpu"):
        try:
            from jax.experimental.pallas import mosaic_gpu  # noqa: F401

            available["mosaic_gpu_importable"] = True
        except Exception as exc:
            available["mosaic_gpu_importable"] = False
            available["mosaic_gpu_error"] = f"{type(exc).__name__}: {exc}"
        break
    available["registered_lowering_platforms"] = sorted(
        getattr(pallas_core, "_LOWERING_RULES", {}).keys()
    ) or "<not introspectable>"
    return available


def check_ffi() -> dict[str, Any]:
    """An XLA FFI custom call, which is how jax-aiter bridges AITER into JAX.

    We do not compile C++ here; what matters for the chapter is whether the
    registration and call machinery is present and reachable, since that is the
    rung a reader would build on.
    """
    import jax
    import jax.extend as jex

    api = {
        "jax.ffi": hasattr(jax, "ffi"),
        "register_ffi_target": hasattr(getattr(jax, "ffi", None), "register_ffi_target"),
        "ffi_call": hasattr(getattr(jax, "ffi", None), "ffi_call"),
        "jex.ffi": hasattr(jex, "ffi"),
    }
    registrations = {}
    try:
        existing = jax.ffi.registrations()  # type: ignore[attr-defined]
        registrations = {
            "platforms": sorted(existing.keys()),
            "rocm_target_count": len(existing.get("ROCM", existing.get("rocm", {}))),
        }
    except Exception as exc:
        registrations = {"error": f"{type(exc).__name__}: {exc}"}
    return {"api": api, "registrations": registrations}


def check_rocprofv3() -> dict[str, Any]:
    """rocprofv3's availability and the counters Chapter 8's occupancy section needs."""
    import shutil

    if shutil.which("rocprofv3") is None:
        return {"present": False}
    version = subprocess.run(
        ["rocprofv3", "--version"], capture_output=True, text=True, check=False
    )
    listing = subprocess.run(
        ["rocprofv3", "--list-avail"], capture_output=True, text=True, check=False, timeout=180
    )
    text = listing.stdout + listing.stderr
    wanted = (
        "OccupancyPercent",
        "MfmaUtil",
        "SQ_INSTS_MFMA",
        "TCC_HIT_sum",
        "TCC_MISS_sum",
        "VALUBusy",
        "MemUnitStalled",
    )
    return {
        "present": True,
        "version": (version.stdout + version.stderr).strip().splitlines()[:3],
        "counters_found": {c: (c in text) for c in wanted},
        "total_counter_lines": sum(1 for line in text.splitlines() if "Name:" in line),
    }


def check_fusion(run: Run) -> dict[str, Any]:
    """One fusion XLA makes and one it declines, with the measured effect.

    The elementwise chain fuses into a single pass; a matmul between two
    elementwise ops does not, because a library GEMM call is opaque.
    """
    import jax
    import jax.numpy as jnp

    from ._harness import measure

    n = 8192
    x = jnp.ones((n, n), jnp.float32)
    w = jnp.ones((n, n), jnp.bfloat16)

    @jax.jit
    def fusible(a):
        return jnp.tanh(a) * 2.0 + 1.0

    @jax.jit
    def unfusible(a, b):
        # A GEMM between the elementwise work blocks the fusion.
        return jnp.tanh(a.astype(jnp.bfloat16) @ b).astype(jnp.float32) * 2.0 + 1.0

    a = measure(lambda: fusible(x), name="elementwise-chain", warmup=3, repeats=10)
    b = measure(lambda: unfusible(x, w), name="elementwise-around-gemm", warmup=3, repeats=10)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from parse_xplane import find_xplanes, kernels  # noqa: E402

    counts = {}
    for name, fn in (("fusible", lambda: fusible(x)), ("unfusible", lambda: unfusible(x, w))):
        with run.trace(f"fusion/{name}") as tdir:
            for _ in range(5):
                jax.block_until_ready(fn())
        rows = kernels(find_xplanes(tdir))
        counts[name] = {
            "distinct_kernels": len(rows),
            "kernel_names": [str(r["name"])[:70] for r in rows[:6]],
        }

    return {
        "elementwise_chain_s": a.median_s,
        "elementwise_around_gemm_s": b.median_s,
        "kernels": counts,
    }


CHECKS: dict[str, Any] = {
    "libraries": check_libraries,
    "pallas": check_pallas,
    "pallas_mosaic": check_pallas_mosaic_gpu,
    "ffi": check_ffi,
    "rocprofv3": check_rocprofv3,
}


def main() -> int:
    args = build_parser().parse_args()
    configure_environment(visible_devices="0")

    wanted = args.only or [*CHECKS, "fusion"]
    with Run("kernels-survey", tag=args.tag, root=args.root, freeze=not args.no_freeze) as run:
        for name in wanted:
            if name == "fusion":
                continue
            result = _verdict(CHECKS[name])
            run.note(name, result)
            status = "ok" if result["ok"] else result["error"][:80]
            print(f"  {name:<16} {status}")
            if result["ok"]:
                print(f"      {json.dumps({k: v for k, v in result.items() if k != 'ok'})[:400]}")

        if "fusion" in wanted:
            result = _verdict(lambda: check_fusion(run))
            run.note("fusion", result)
            if result["ok"]:
                print(
                    f"  {'fusion':<16} chain {result['elementwise_chain_s'] * 1e3:.2f} ms in "
                    f"{result['kernels']['fusible']['distinct_kernels']} kernel(s); "
                    f"around a GEMM {result['elementwise_around_gemm_s'] * 1e3:.2f} ms in "
                    f"{result['kernels']['unfusible']['distinct_kernels']} kernel(s)"
                )
                for label, info in result["kernels"].items():
                    for kn in info["kernel_names"]:
                        print(f"      {label:<10} {kn}")
            else:
                print(f"  {'fusion':<16} {result['error'][:100]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
