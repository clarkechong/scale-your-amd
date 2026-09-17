#!/usr/bin/env python3
"""Capture and render the real Chapter 4 precision HLO fixtures.

The script runs a MaxText-style BF16 dot and two
``TransformerEngineQuantization`` recipes in the installed JAX/ROCm
environment. It preserves XLA HLO text and literal DOT output under
``artifacts/hlo-fixtures/precision``. The chapter SVGs are small, manually
pruned views rendered from those captures.

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

INK = "#27313b"
BLUE = ("#dce9f5", "#3f76ab")
GOLD = ("#f7e6c8", "#b0842f")
PURPLE = ("#e9e3f2", "#6d5b9e")
GREEN = ("#dff0e4", "#4e8a5c")
NEUTRAL = ("#f5f7f8", "#8a959e")


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


def extract_op_name(hlo_text: str, target: str, occurrence: int = 0) -> str:
    matches = re.findall(
        rf"^\s*([A-Za-z0-9_.-]+)\s*=.*custom-call\(.*custom_call_target=\"{re.escape(target)}\"",
        hlo_text,
        flags=re.MULTILINE,
    )
    if len(matches) <= occurrence:
        raise RuntimeError(f"Could not find occurrence {occurrence} of {target!r} in HLO")
    return matches[occurrence]


def extract_dot_name(hlo_text: str) -> str:
    matches = re.findall(
        r"^\s*(?:ROOT\s+)?([A-Za-z0-9_.-]+)\s*=\s*bf16\[[^\]]+\].*\sdot\(",
        hlo_text,
        flags=re.MULTILINE,
    )
    if not matches:
        raise RuntimeError("Could not find BF16 dot in HLO")
    return matches[0]


def q(value: str) -> str:
    return '"' + value.replace('"', '\\"').replace("\n", "\\n") + '"'


def node(
    node_id: str,
    label: str,
    palette: tuple[str, str],
    *,
    shape: str = "box",
) -> str:
    face, edge = palette
    return (
        f"  {node_id} [label={q(label)}, shape={shape}, style=\"rounded,filled\", "
        f"fillcolor={q(face)}, color={q(edge)}, fontcolor={q(INK)}, "
        'fontname="Helvetica", fontsize=10, margin="0.12,0.08"];\n'
    )


def graph_header(title: str) -> str:
    return (
        "digraph G {\n"
        '  graph [rankdir=LR, bgcolor="transparent", pad=0.10, nodesep=0.35, '
        f'ranksep=0.52, label={q(title)}, labelloc="t", fontsize=14, '
        f'fontname="Helvetica", fontcolor={q(INK)}];\n'
        '  edge [color="#8a959e", penwidth=1.1, arrowsize=0.7, fontname="Helvetica", '
        'fontsize=8, fontcolor="#65717d"];\n'
    )


def bf16_representative(hlo_text: str) -> str:
    dot_name = extract_dot_name(hlo_text)
    graph = graph_header("BF16 MaxText linear: ordinary HLO dot")
    graph += node("master", f"master weight\\nf32[{K},{N}]", GREEN)
    graph += node("cast", f"convert_element_type.1\\nbf16[{K},{N}]", NEUTRAL)
    graph += node("activation", f"activation\\nbf16[{M},{K}]", BLUE)
    graph += node("dot", f"{dot_name}\\ndot, contract K={K}\\nbf16[{M},{N}]", BLUE)
    graph += "  master -> cast;\n  cast -> dot [label=\"rhs\"];\n  activation -> dot [label=\"lhs\"];\n}\n"
    return graph


def fp8_representative(hlo_text: str) -> str:
    quant_x = extract_op_name(hlo_text, "te_dbias_quantize_ffi", 0)
    quant_w = extract_op_name(hlo_text, "te_dbias_quantize_ffi", 1)
    gemm = extract_op_name(hlo_text, "te_gemm_v2_ffi")
    graph = graph_header("FP8 delayed scaling: typed FFI quantize and GEMM calls")
    graph += node("activation", f"activation\\nbf16[{M},{K}]", BLUE)
    graph += node("xstate", "delayed x scale\\nf32\\nhistory updated separately", PURPLE)
    graph += node(
        "quantx",
        f"{quant_x}\\nte_dbias_quantize_ffi\\nf8e4m3fn[{M},{K}] + f32 scale",
        GOLD,
    )
    graph += node("master", f"master weight\\nf32[{K},{N}]", GREEN)
    graph += node("cast", f"compute weight\\nbf16[{K},{N}]", NEUTRAL)
    graph += node("wstate", "delayed weight scale\\nf32\\nhistory updated separately", PURPLE)
    graph += node(
        "quantw",
        f"{quant_w}\\nte_dbias_quantize_ffi\\nf8e4m3fn[{K},{N}] + f32 scale",
        GOLD,
    )
    graph += node(
        "gemm",
        f"{gemm}\\nte_gemm_v2_ffi\\nper-tensor scaled FP8\\n→ bf16[{M},{N}]",
        PURPLE,
    )
    graph += (
        "  activation -> quantx;\n"
        "  xstate -> quantx [style=dashed];\n"
        "  master -> cast;\n"
        "  cast -> quantw;\n"
        "  wstate -> quantw [style=dashed];\n"
        "  quantx -> gemm [label=\"FP8 + scale\"];\n"
        "  quantw -> gemm [label=\"FP8 + scale\"];\n"
        "}\n"
    )
    return graph


def mxfp8_representative(hlo_text: str) -> str:
    quant_x = extract_op_name(hlo_text, "te_dbias_quantize_ffi", 0)
    quant_w = extract_op_name(hlo_text, "te_dbias_quantize_ffi", 1)
    gemm = extract_op_name(hlo_text, "te_gemm_v2_ffi")
    graph = graph_header("MXFP8 block scaling: element arrays and E8M0 scales enter GEMM")
    graph += node("activation", f"activation\\nbf16[{M},{K}]", BLUE)
    graph += node(
        "quantx",
        f"{quant_x}\\nte_dbias_quantize_ffi\\nrow f8e4m3fn[{M},{K}]\\nE8M0 scale [{M},{K // 32}]",
        GOLD,
    )
    graph += node("master", f"master weight\\nf32[{K},{N}]", GREEN)
    graph += node("cast", f"compute weight\\nbf16[{K},{N}]", NEUTRAL)
    graph += node(
        "quantw",
        f"{quant_w}\\nte_dbias_quantize_ffi\\ncolumn f8e4m3fn[{K},{N}]\\nE8M0 scale [{K // 32},{N}]",
        GOLD,
    )
    graph += node(
        "gemm",
        f"{gemm}\\nte_gemm_v2_ffi\\nMXFP8_1D_SCALING\\n→ bf16[{M},{N}]",
        PURPLE,
    )
    graph += (
        "  activation -> quantx;\n"
        "  master -> cast;\n"
        "  cast -> quantw;\n"
        "  quantx -> gemm [label=\"E4M3 + E8M0\"];\n"
        "  quantw -> gemm [label=\"E4M3 + E8M0\"];\n"
        "}\n"
    )
    return graph


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

    representative_builders = {
        "bf16": bf16_representative,
        "fp8": fp8_representative,
        "mxfp8": mxfp8_representative,
    }
    representative_dot = representative_builders[kind](hlo_text)

    artifact_dir = ARTIFACT_ROOT / kind
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "before_optimizations.txt").write_text(hlo_text)
    (artifact_dir / "before_optimizations.dot").write_text(xla_dot)
    (artifact_dir / "representative.dot").write_text(representative_dot)

    targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo_text)))
    provenance = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": kind,
        "description": details,
        "capture_script": "tools/make_ch4_precision_hlo.py",
        "compiler_output": "jax.jit(...).lower(...).compiler_ir(dialect='hlo')",
        "raw_dot": "before_optimizations.dot",
        "raw_hlo": "before_optimizations.txt",
        "representative_dot": "representative.dot",
        "representative_note": (
            "Manually pruned view of the literal XLA DOT. Operation names, "
            "targets, dtypes, and shapes are asserted against the preserved HLO."
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
        ["dot", "-Tsvg", str(artifact_dir / "representative.dot"), "-o", str(output)],
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
