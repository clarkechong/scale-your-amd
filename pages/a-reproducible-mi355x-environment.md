---
layout: distill
title: "Reproducible MI355X Environment"
description: "Pinned containers, source revisions, launch flags, and verification checks for the book's MI355X experiments."
date: 2026-09-13

section_label: "Appendix A"

previous_section_url: "/pages/13-mixtral-8x22b-sharding-meshes-and-moe-optimizations"
previous_section_name: "Chapter 13: Sharding Meshes and MoE Optimizations"

next_section_url: "/pages/b-measurement-and-convergence-protocol"
next_section_name: "Appendix B: Protocol"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Baseline Container
  - name: Preserved Version Pins
  - name: Launch the Container
  - name: Verify the Runtime
  - name: Optional Source Variants
  - name: Compatibility Sources
---

## Baseline Container

All current MI355X experiment repositories start from:

```bash
docker pull docker.io/rocm/jax-training:maxtext-v26.6
```

Record the immutable digest after pulling:

{% raw %}
```bash
IMAGE_TAG=docker.io/rocm/jax-training:maxtext-v26.6
IMAGE_DIGEST=$(docker image inspect \
  --format '{{index .RepoDigests 0}}' \
  "$IMAGE_TAG")
printf '%s\n' "$IMAGE_DIGEST"
```
{% endraw %}

The tag is the shared baseline. The digest, package inventory, effective `XLA_FLAGS`, and
source commits in the run manifest decide whether two runs used the same environment.

## Preserved Version Pins

| Component | Pin used by the current repositories | Scope |
|---|---|---|
| Container | `docker.io/rocm/jax-training:maxtext-v26.6` | All three cases |
| MaxText | `release/v26.6`, commit `b47d74bf` | Stock path in `/workspace/maxtext` |
| JAX and `jaxlib` | `0.11.0` | Llama 7B experiment contract |
| ROCm plugin and PJRT | `0.11.0` | Llama 7B experiment contract |
| Transformer Engine | README: `fix/jax-gfx950-mxfp8-workspace`; installer: `experiment/v2.17-gfx950-mxfp8-workspace` | MXFP8; branch conflict and exact commit must be resolved |
| ROCm MaxText | `feature/jax-aiter-mxfp4-v26.6`, commit `b437942a5f33704f8438deb948488ad08164285c` | MXFP4 only |
| JAX-AITER | `release/v0.1.0-alpha2`, commit `35b7175c763153ddb5da50c47d33dec436d5f191` | Direct AITER and MXFP4 |
| AITER submodule | commit `31350226161346314b3d8882c8085bd31dce6a34` | MXFP4 FFI build |

The Transformer Engine row is not reproducible: the README and installer name different
branches, neither is commit-pinned, and no built-wheel hash is recorded. Resolve all three
before publishing an MXFP8 result. Do not use either branch head as evidence.

## Launch the Container

{% raw %}
```bash
HOST_WORKSPACE=/home/clchong/work
IMAGE_TAG=docker.io/rocm/jax-training:maxtext-v26.6
IMAGE_DIGEST=$(docker image inspect \
  --format '{{index .RepoDigests 0}}' \
  "$IMAGE_TAG")

docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$HOST_WORKSPACE":/workspace/book \
  "$IMAGE_DIGEST"
```
{% endraw %}

The explicit host path makes `/workspace/book/llama7b`, `/workspace/book/llama70b`,
and `/workspace/book/mixtral8-22b` resolve regardless of the shell's current
directory. `--ipc=host` uses the host's shared-memory namespace; do not combine it
with `--shm-size`, which controls a private IPC namespace. If host IPC is disallowed,
omit `--ipc=host` and set a tested private size such as `--shm-size=64G`.

Host networking is not required for the single-node commands in this book. Add
`--network=host` only when the tested multi-node coordinator or profiling setup
requires it, and record that choice. `SYS_PTRACE` and the seccomp override allow the
profiling workflow. If local policy requires different privileges, record the
replacement in the manifest.

Use the repositories' checked-in flag file as the starting point:

```text
--xla_gpu_autotune_level=4
--xla_gpu_enable_triton_gemm=true
--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_all_gather_combine_threshold_bytes=8589934592
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
--xla_gpu_all_reduce_combine_threshold_bytes=8589934592
--xla_gpu_enable_command_buffer=
```

An empty command-buffer assignment is intentional. Do not rewrite it as `false`.

## Verify the Runtime

Capture this inventory before a run:

```bash
python3 -m pip freeze | sort
rocm-smi --showproductname
rocm-smi --showcomputepartition
rocm-smi --showmemorypartition
rocminfo
```

Then execute a real device computation:

```python
import jax
import jax.numpy as jnp

devices = jax.devices()
assert devices, "JAX found no devices"
assert "rocm" in devices[0].client.platform_version.lower()
x = jnp.arange(8, dtype=jnp.float32)
assert float(jax.jit(lambda value: (value * value).sum())(x).block_until_ready()) == 140.0
print(jax.__version__, devices[0].device_kind, devices[0].client.platform_version)
```

`jax.default_backend()` returns the generic value `gpu` on ROCm, so it does not distinguish
ROCm from CUDA. Use `platform_version` and the device inventory.

Set `JAX_PLATFORMS=rocm` to avoid probing unrelated backends:

```bash
export JAX_PLATFORMS=rocm
```

## Optional Source Variants

Use only the variant required by the experiment:

```bash
# Llama 7B direct JAX-AITER attention
cd /workspace/book/llama7b
bash scripts/setup/setup_aiter.sh

# Llama 70B patched MXFP8 workspace
cd /workspace/book/llama70b
bash scripts/setup/install_te_mxfp8.sh

# Llama 70B MXFP4 MaxText and JAX-AITER path
bash scripts/setup/setup_mxfp4.sh
```

The runners accept:

```text
MAXTEXT_ROOT
MAXTEXT_MXFP4_ROOT
JAX_AITER_ROOT
DATA_ROOT
OUTPUT_ROOT
```

Changing one of these paths changes the source tree in use. Record its commit and dirty diff
in [Appendix F]({{ '/pages/f-case-study-artifact-schema' | relative_url }}).

## Compatibility Sources

Use AMD's live matrices instead of copying a fast-rotting table:

- [JAX on ROCm compatibility](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/jax-compatibility.html)
- [Install JAX for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html)
- [ROCm JAX releases](https://github.com/ROCm/rocm-jax/releases)
- [ROCm JAX containers](https://hub.docker.com/r/rocm/jax-training/tags)

Record observed mismatches and workarounds in
[Appendix E]({{ '/pages/e-compatibility-and-negative-results' | relative_url }}).
