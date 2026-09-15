#!/usr/bin/env python3
"""Small JAX programs used to capture literal compiler-delta HLO graphs.

Each invocation compiles exactly one fixture arm. XLA dump flags must be set by
the caller before this module imports JAX. The fixtures are explanatory compiler
inputs, not performance benchmarks or substitutes for case-study captures.

Examples:

    python -m bench.hlo_feature_fixtures scan unrolled
    python -m bench.hlo_feature_fixtures sharding fsdp
"""

from __future__ import annotations

import argparse
import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _compile_and_run(fn, *args, donate_argnums=()):
    compiled = jax.jit(fn, donate_argnums=donate_argnums).lower(*args).compile()
    result = compiled(*args)
    jax.block_until_ready(result)
    return result


def _layer(x, weight):
    return jnp.tanh(x @ weight)


def scan_fixture(variant: str):
    key = jax.random.key(0)
    weights = jax.random.normal(key, (4, 128, 128), dtype=jnp.bfloat16)
    x = jax.random.normal(key, (32, 128), dtype=jnp.bfloat16)

    if variant == "unrolled":

        def scan_unrolled(weights, x):
            for i in range(4):
                x = x @ weights[i]
            return x

        return _compile_and_run(scan_unrolled, weights, x)

    if variant == "scanned":

        def scan_scanned(weights, x):
            def body(carry, weight):
                return carry @ weight, None

            return jax.lax.scan(body, x, weights)[0]

        return _compile_and_run(scan_scanned, weights, x)

    raise ValueError(f"unknown scan variant: {variant}")


def remat_fixture(variant: str):
    key = jax.random.key(1)
    weights = jax.random.normal(key, (2, 128, 128), dtype=jnp.bfloat16)
    x = jax.random.normal(key, (64, 128), dtype=jnp.bfloat16)

    def remat_none_loss(weights, x):
        for i in range(2):
            x = _layer(x, weights[i])
        return jnp.sum(x.astype(jnp.float32))

    if variant not in {"none", "full"}:
        raise ValueError(f"unknown remat variant: {variant}")

    def remat_full_loss(weights, x):
        remat_layer = jax.checkpoint(_layer)
        for i in range(2):
            x = remat_layer(x, weights[i])
        return jnp.sum(x.astype(jnp.float32))

    loss = remat_none_loss if variant == "none" else remat_full_loss
    fn = jax.value_and_grad(loss)
    return _compile_and_run(fn, weights, x)


def accumulation_fixture(variant: str):
    key = jax.random.key(4)
    weight = jax.random.normal(key, (128, 128), dtype=jnp.bfloat16)
    microbatches = jax.random.normal(key, (4, 32, 128), dtype=jnp.bfloat16)

    def microbatch_loss(weight, batch):
        return jnp.sum(jnp.tanh(batch @ weight).astype(jnp.float32))

    if variant == "direct":

        def accumulation_direct(weight, microbatches):
            batch = jnp.reshape(microbatches, (128, 128))
            return jax.grad(microbatch_loss)(weight, batch)

        return _compile_and_run(accumulation_direct, weight, microbatches)

    if variant == "scanned":

        def accumulation_scanned(weight, microbatches):
            def body(gradient, microbatch):
                gradient_i = jax.grad(microbatch_loss)(weight, microbatch)
                return gradient + gradient_i, None

            initial = jnp.zeros_like(weight)
            return jax.lax.scan(body, initial, microbatches)[0]

        return _compile_and_run(accumulation_scanned, weight, microbatches)

    raise ValueError(f"unknown accumulation variant: {variant}")


def _mesh() -> Mesh:
    devices = np.asarray(jax.devices())
    if devices.size < 8:
        raise RuntimeError(f"sharding fixtures require 8 devices, found {devices.size}")
    return Mesh(devices[:8], ("mesh",))


def sharding_fixture(variant: str):
    mesh = _mesh()
    x_host = np.ones((1024, 512), dtype=np.float16)
    w_host = np.ones((512, 512), dtype=np.float16)

    if variant == "dp":
        x_s = NamedSharding(mesh, P("mesh", None))
        w_s = NamedSharding(mesh, P(None, None))
        y_s = NamedSharding(mesh, P("mesh", None))
        x = jax.device_put(x_host, x_s)
        w = jax.device_put(w_host, w_s)

        def sharding_dp(x, w):
            return x @ w

        compiled = jax.jit(
            sharding_dp,
            in_shardings=(x_s, w_s),
            out_shardings=y_s,
        ).lower(x, w).compile()
        out = compiled(x, w)
        jax.block_until_ready(out)
        return out

    if variant == "fsdp":
        x_s = NamedSharding(mesh, P("mesh", None))
        w_s = NamedSharding(mesh, P("mesh", None))
        y_s = NamedSharding(mesh, P("mesh", None))
        x = jax.device_put(x_host, x_s)
        w = jax.device_put(w_host, w_s)

        def sharding_fsdp(x, w):
            return x @ w

        compiled = jax.jit(
            sharding_fsdp,
            in_shardings=(x_s, w_s),
            out_shardings=y_s,
        ).lower(x, w).compile()
        out = compiled(x, w)
        jax.block_until_ready(out)
        return out

    if variant == "tp":
        x_s = NamedSharding(mesh, P(None, "mesh"))
        w_s = NamedSharding(mesh, P("mesh", None))
        x = jax.device_put(x_host, x_s)
        w = jax.device_put(w_host, w_s)

        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P(None, "mesh"), P("mesh", None)),
            out_specs=P(None, None),
        )
        def sharding_tp(x_local, w_local):
            partial = x_local @ w_local
            return jax.lax.psum(partial, "mesh")

        compiled = jax.jit(sharding_tp).lower(x, w).compile()
        out = compiled(x, w)
        jax.block_until_ready(out)
        return out

    raise ValueError(f"unknown sharding variant: {variant}")


def lhs_fixture(variant: str):
    if variant not in {"off", "on"}:
        raise ValueError(f"unknown lhs variant: {variant}")
    mesh = _mesh()

    x = jax.device_put(
        np.ones((2048, 128), dtype=np.float16),
        NamedSharding(mesh, P("mesh", None)),
    )
    w = jax.device_put(
        np.ones((128, 256), dtype=np.float16),
        NamedSharding(mesh, P(None, None)),
    )
    a = jax.device_put(
        np.ones((2048, 512), dtype=np.float16),
        NamedSharding(mesh, P(None, None)),
    )
    b = jax.device_put(
        np.ones((512, 256), dtype=np.float16),
        NamedSharding(mesh, P(None, None)),
    )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(
            P("mesh", None),
            P(None, None),
            P(None, None),
            P(None, None),
        ),
        out_specs=P(None, None),
        check_vma=False,
    )
    def lhs_body(x_local, w_local, a_local, b_local):
        x_full = jax.lax.all_gather(x_local, "mesh", axis=0, tiled=True)
        independent = a_local @ b_local
        dependent = x_full @ w_local
        return dependent + independent

    def lhs_off(x, w, a, b):
        return lhs_body(x, w, a, b)

    def lhs_on(x, w, a, b):
        return lhs_body(x, w, a, b)

    fn = lhs_on if variant == "on" else lhs_off
    compiled = jax.jit(fn).lower(x, w, a, b).compile()
    out = compiled(x, w, a, b)
    jax.block_until_ready(out)
    return out


class _TEAttention(nn.Module):
    @nn.compact
    def __call__(self, q, k, v):
        from transformer_engine.jax.flax import DotProductAttention

        return DotProductAttention(
            head_dim=64,
            num_attention_heads=8,
            num_gqa_groups=8,
            attn_mask_type="causal",
            attn_bias_type="no_bias",
            attention_dropout=0.0,
            qkv_layout="BSHD_BSHD_BSHD",
            scale_factor=1.0 / 8.0,
            transpose_batch_sequence=False,
            max_segments_per_seq=1,
            name="te_attention",
        )(q, k, v, deterministic=True)


def attention_fixture(variant: str):
    key = jax.random.key(2)
    shape = (1, 128, 8, 64)
    q = jax.random.normal(key, shape, dtype=jnp.bfloat16)
    k = jax.random.normal(key, shape, dtype=jnp.bfloat16)
    v = jax.random.normal(key, shape, dtype=jnp.bfloat16)

    if variant == "xla":

        def attention_xla(q, k, v):
            return jax.nn.dot_product_attention(
                q,
                k,
                v,
                scale=1.0 / 8.0,
                is_causal=True,
                implementation="xla",
            )

        return _compile_and_run(attention_xla, q, k, v)

    if variant == "te":
        module = _TEAttention()
        variables = module.init(key, q, k, v)

        def attention_te(q, k, v):
            return module.apply(variables, q, k, v)

        return _compile_and_run(attention_te, q, k, v)

    raise ValueError(f"unknown attention variant: {variant}")


def moe_fixture(variant: str):
    key = jax.random.key(3)

    if variant == "dense-masked":
        tokens = jax.random.normal(key, (64, 128), dtype=jnp.bfloat16)
        experts = jax.random.normal(key, (4, 128, 128), dtype=jnp.bfloat16)
        routing = jax.nn.one_hot(jnp.arange(64) % 4, 4, dtype=jnp.bfloat16)

        def moe_dense_masked(tokens, experts, routing):
            all_outputs = jnp.einsum("mk,gkn->mgn", tokens, experts)
            return jnp.einsum("mgn,mg->mn", all_outputs, routing)

        return _compile_and_run(moe_dense_masked, tokens, experts, routing)

    if variant == "fixed-capacity":
        tokens = jax.random.normal(key, (64, 128), dtype=jnp.bfloat16)
        experts = jax.random.normal(key, (4, 128, 128), dtype=jnp.bfloat16)
        dispatch = jax.nn.one_hot(jnp.arange(64) % 16, 16, dtype=jnp.bfloat16)
        dispatch = dispatch[:, None, :] * jax.nn.one_hot(
            jnp.arange(64) % 4, 4, dtype=jnp.bfloat16
        )[:, :, None]

        def moe_fixed_capacity(tokens, experts, dispatch):
            expert_inputs = jnp.einsum("mk,mgc->gck", tokens, dispatch)
            expert_outputs = jnp.einsum("gck,gkn->gcn", expert_inputs, experts)
            return jnp.einsum("gcn,mgc->mn", expert_outputs, dispatch)

        return _compile_and_run(moe_fixed_capacity, tokens, experts, dispatch)

    if variant in {"ragged-padded", "ragged-grouped"}:
        tokens = jax.random.normal(key, (64, 128), dtype=jnp.float16)
        experts = jax.random.normal(key, (4, 128, 128), dtype=jnp.float16)
        group_sizes = jnp.asarray([16, 16, 16, 16], dtype=jnp.int32)

        def moe_ragged_padded(tokens, experts, group_sizes):
            return jax.lax.ragged_dot(tokens, experts, group_sizes)

        def moe_ragged_grouped(tokens, experts, group_sizes):
            return jax.lax.ragged_dot(tokens, experts, group_sizes)

        fn = moe_ragged_grouped if variant == "ragged-grouped" else moe_ragged_padded
        return _compile_and_run(fn, tokens, experts, group_sizes)

    raise ValueError(f"unknown MoE variant: {variant}")


FIXTURES = {
    "accumulation": accumulation_fixture,
    "scan": scan_fixture,
    "remat": remat_fixture,
    "sharding": sharding_fixture,
    "lhs": lhs_fixture,
    "attention": attention_fixture,
    "moe": moe_fixture,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", choices=sorted(FIXTURES))
    parser.add_argument("variant")
    args = parser.parse_args()
    FIXTURES[args.fixture](args.variant)
    print(
        f"compiled fixture={args.fixture} variant={args.variant} "
        f"backend={jax.default_backend()} devices={jax.device_count()}"
    )


if __name__ == "__main__":
    main()
