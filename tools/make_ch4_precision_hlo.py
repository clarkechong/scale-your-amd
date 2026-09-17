#!/usr/bin/env python3
"""Capture and render the real Chapter 4 precision HLO fixtures.

The script runs a MaxText-style BF16 dot and two
``TransformerEngineQuantization`` recipes in the installed JAX/ROCm
environment. It preserves XLA HLO text and literal DOT output under
``artifacts/hlo-fixtures/precision``. Graphviz renders the literal XLA DOT
without a hand-authored intermediate graph.

Run from the repository root:

    HIP_VISIBLE_DEVICES=0 JAX_PLATFORMS=rocm \
      NVTE_ROCM_ENABLE_MXFP8=1 \
      PYTHONPATH=/workspace/maxtext/src \
      python tools/make_ch4_precision_hlo.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

os.environ.setdefault("NVTE_ROCM_ENABLE_MXFP8", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import linen as nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = ROOT / "artifacts" / "hlo-fixtures" / "precision"
IMAGE_ROOT = ROOT / "pages" / "img"

M = 128
K = 256
N = 256


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def ready(tree):
    return jax.tree.map(
        lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value,
        tree,
    )


def bf16_lowering():
    from maxtext.layers.linears import _compute_dot_general

    activation = jnp.ones((M, K), dtype=jnp.bfloat16)
    master_weight = jnp.ones((K, N), dtype=jnp.float32)

    def maxtext_bf16_dot(weight, x):
        compute_weight = weight.astype(jnp.bfloat16)
        return _compute_dot_general(
            x,
            compute_weight,
            (),
            (1,),
            (0,),
            "default",
            None,
        )

    lowered = jax.jit(maxtext_bf16_dot).lower(master_weight, activation)
    ready(lowered.compile()(master_weight, activation))
    return lowered, {
        "entry": "maxtext.layers.linears._compute_dot_general",
        "config": {"dtype": "bfloat16", "quantization": ""},
        "variables": {
            "activation": f"bf16[{M},{K}]",
            "master_weight": f"f32[{K},{N}]",
            "output": f"bf16[{M},{N}]",
        },
    }


def te_lowering(recipe_name: str):
    from maxtext.layers.quantizations import TransformerEngineQuantization

    activation = jnp.ones((M, K), dtype=jnp.bfloat16)
    config = SimpleNamespace(
        quantization=recipe_name,
        use_te_comm_gemm_overlap=False,
    )
    quantization = TransformerEngineQuantization(config)
    dot_general_cls = quantization.dot_general_cls(mesh_axes=())
    layer = nn.Dense(
        N,
        use_bias=False,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        dot_general_cls=dot_general_cls,
    )
    variables = layer.init(jax.random.key(0), activation)
    mutable = sorted(set(variables.keys()) - {"params"})

    def apply_layer(state, x):
        if mutable:
            output, updated = layer.apply(state, x, mutable=mutable)
            return output, updated
        return layer.apply(state, x)

    lowered = jax.jit(apply_layer).lower(variables, activation)
    ready(lowered.compile()(variables, activation))
    return lowered, {
        "entry": "MaxText TransformerEngineQuantization.dot_general_cls",
        "config": {
            "dtype": "bfloat16",
            "quantization": recipe_name,
            "use_te_comm_gemm_overlap": False,
        },
        "variables": jax.tree.map(
            lambda value: {"shape": list(value.shape), "dtype": str(value.dtype)},
            variables,
        ),
        "activation": f"bf16[{M},{K}]",
        "output": f"bf16[{M},{N}]",
    }


def validate_capture(kind: str, hlo_text: str) -> None:
    required = {
        "bf16": ["dot(", f"bf16[{M},{N}]"],
        "fp8": [
            'custom_call_target="te_dbias_quantize_ffi"',
            'custom_call_target="te_gemm_v2_ffi"',
            f"f8e4m3fn[{M},{K}]",
            f"bf16[{M},{N}]",
        ],
        "mxfp8": [
            'custom_call_target="te_dbias_quantize_ffi"',
            'custom_call_target="te_gemm_v2_ffi"',
            "f8e8m0fnu",
            "scaling_mode = 2",
            f"bf16[{M},{N}]",
        ],
    }[kind]
    missing = [token for token in required if token not in hlo_text]
    if missing:
        raise RuntimeError(f"{kind} HLO is missing expected evidence: {missing}")


def capture(kind: str, lowered, details: dict) -> Path:
    computation = lowered.compiler_ir(dialect="hlo")
    hlo_text = computation.as_hlo_text()
    xla_dot = computation.as_hlo_dot_graph()
    validate_capture(kind, hlo_text)
    render_dot = re.sub(
        r', tooltip=".*?", style=',
        ', tooltip=" ", style=',
        xla_dot,
        flags=re.DOTALL,
    )

    artifact_dir = ARTIFACT_ROOT / kind
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "before_optimizations.txt").write_text(hlo_text)
    (artifact_dir / "before_optimizations.dot").write_text(xla_dot)

    targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo_text)))
    provenance = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": kind,
        "description": details,
        "capture_script": "tools/make_ch4_precision_hlo.py",
        "compiler_output": "jax.jit(...).lower(...).compiler_ir(dialect='hlo')",
        "raw_dot": "before_optimizations.dot",
        "raw_hlo": "before_optimizations.txt",
        "rendering": (
            "Literal XLA DOT rendered with Graphviz after removing source-path "
            "tooltips; nodes, edges, and visible labels are unchanged."
        ),
        "executed_on_device": True,
        "shape": {"M": M, "K": K, "N": N},
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "jax_rocm7_plugin": package_version("jax-rocm7-plugin"),
        "jax_rocm7_pjrt": package_version("jax-rocm7-pjrt"),
        "rocm": package_version("rocm"),
        "rocm_sdk_core": package_version("rocm-sdk-core"),
        "transformer_engine": package_version("transformer_engine"),
        "maxtext_revision": git_revision(Path("/workspace/maxtext")),
        "platform": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "custom_call_targets": targets,
        "environment": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", ""),
            "NVTE_ROCM_ENABLE_MXFP8": os.environ.get("NVTE_ROCM_ENABLE_MXFP8", ""),
        },
    }
    (artifact_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    output = IMAGE_ROOT / f"hlo-precision-{kind}.svg"
    subprocess.run(
        ["dot", "-Tsvg", "-o", str(output)],
        input=render_dot,
        text=True,
        check=True,
    )
    return output


def main() -> None:
    captures = {
        "bf16": bf16_lowering(),
        "fp8": te_lowering("te_fp8_delayedscaling"),
        "mxfp8": te_lowering("te_mxfp8"),
    }
    for kind, (lowered, details) in captures.items():
        output = capture(kind, lowered, details)
        print(f"  wrote {output.relative_to(ROOT)}")
        print(f"  preserved {ARTIFACT_ROOT.relative_to(ROOT) / kind}")


if __name__ == "__main__":
    main()
