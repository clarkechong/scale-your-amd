---
layout: distill
title: "Appendix A: Installing JAX on ROCm"
description: "The container path, the wheel path, the version matrix, and the combinations known to be broken. Kept out of the chapters because it is necessary, it is nobody's reason for reading the book, and it rots faster than anything else here."
date: 2026-08-04

section_label: "Appendix A"

previous_section_url: "/pages/13-conclusion"
previous_section_name: "Chapter 13: Conclusions"

next_section_url: "/pages/b-appendix-protocol"
next_section_name: "Appendix B: How We Measure"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: The Container Path
  - name: The Wheel Path
  - name: The ROCm Version Matrix
  - name: Building From Source
  - name: Known-Broken Combinations
---

**Depends on:** nothing. This is a reference page and readers arrive at it directly from
[Chapter 3]({{ '/pages/3-profiling' | relative_url }}) or from a search result.

> **Verified against:** the versions and commands below were checked on **5 August 2026**
> against AMD's ROCm JAX installation documentation and against the container this book
> measures in, `rocm/jax-training:maxtext-v26.5` (ROCm 7.14.0, `jax` 0.10.0).
> **This appendix exists precisely because this material rots**, so treat anything here
> older than a couple of ROCm releases as a hint rather than an instruction, and follow the
> links to the current version.

**Use the container.** If you take one thing from this page, take that. The ROCm JAX stack
is four packages whose versions have to agree with each other and with the ROCm install on
the host, and the container is the one artifact where somebody else has already made them
agree.

## The Container Path

**AMD publishes prebuilt JAX images on Docker Hub with ROCm, JAX, jaxlib and both plugin
wheels installed and pinned.**

```bash
docker pull rocm/jax:latest

docker run -it -d --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 64G \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $(pwd):/workspace \
  --name rocm_jax rocm/jax:latest /bin/bash

docker attach rocm_jax
```

**The flags are not optional and each one is there for a reason.** `--device=/dev/kfd` and
`--device=/dev/dri` expose the GPU driver interfaces; without them JAX sees no devices at
all. `--group-add video` grants access to them. `--shm-size 64G` matters because the
default 64 MB of shared memory is far too small for multi-process JAX and the failure
looks like an unrelated crash. `--cap-add=SYS_PTRACE` and the seccomp setting are what let
the profiler attach, so drop them and
[Chapter 3]({{ '/pages/3-profiling' | relative_url }}) stops working.

**Three image families, and the difference matters for reproducibility.**
`rocm/jax` images are validated and released by AMD alongside ROCm releases, quarterly.
`rocm/jax-community` images track upstream JAX releases against the latest available ROCm
and are not put through the same testing. `rocm/jax-training` images add MaxText, its
dependencies and AMD's tuning environment on top, which is what makes them the right
starting point for anything in Part III. **Pin a version-specific tag rather than
`latest`**, since `latest` is not a reproducible reference.

**This book measures inside one specific tag**, and it is the training image rather than the
plain one:

```bash
docker pull rocm/jax-training:maxtext-v26.5
```

That single string pins ROCm 7.14.0, `jax` 0.10.0, both plugin wheels, RCCL 2.30.4, XProf
2.23.0 and a MaxText checkout, which is the entire stack table in
[Appendix B]({{ '/pages/b-appendix-protocol' | relative_url }}). It also presets `XLA_FLAGS`
in ways that change measured numbers, which that appendix spells out. **If you want to
reproduce a number from this book, start here rather than from the wheel path below.**

## The Wheel Path

**Four packages, and the order matters less than the version agreement.** Inside a
container with a matching ROCm already present, or on a bare-metal host with ROCm
installed:

```bash
pip3 install jax==0.8.2
pip3 install jax-rocm7-pjrt==0.8.2
pip3 install jax-rocm7-plugin==0.8.2
pip3 install https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.8.2/jaxlib-0.8.2+rocm7-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

Then check that all four agree:

```bash
pip3 freeze | grep jax
# jax==0.8.2
# jax-rocm7-pjrt==0.8.2
# jax-rocm7-plugin==0.8.2
# jaxlib==0.8.2
```

**The version above is the one AMD's documentation currently gives as its example, not the
one this book's measurements were taken with**, which is 0.10.0 on ROCm 7.14.0. Substitute
the version you want and check it against the compatibility matrix below; the pattern is
what to copy, not the number.

**What each package is**, since the split confuses people:

- **`jax`** is the pure-Python frontend. No compiled code, no hardware knowledge.
- **`jaxlib`** is the compiled core, including XLA. The ROCm build is distributed from the
  `ROCm/rocm-jax` releases page rather than PyPI, which is why that line is a URL.
- **`jax-rocm7-pjrt`** is the PJRT plugin: the shared library implementing the device API
  against HIP.
- **`jax-rocm7-plugin`** is the Python registration package that tells JAX the plugin
  exists.

**There is also a convenience extra**, `pip install --upgrade "jax[rocm7-local]"`, which
installs the plugin and PJRT packages on top of an existing ROCm. Note the `-local`: JAX
ships no extra that installs ROCm itself, so ROCm has to be present on the host or in the
container already.

**Verify with a real computation rather than an import.** The failure mode you are checking
for is a silent fallback to CPU:

```python
import jax, jax.numpy as jnp

device = jax.devices()[0]
assert "rocm" in device.client.platform_version, f"not ROCm: {device.client.platform_version}"
x = jnp.arange(8.0)
y = jax.jit(lambda a: (a * a).sum())(x)
y.block_until_ready()
assert float(y) == 140.0
print("OK:", len(jax.devices()), "devices,", device.device_kind)
```

**Do not assert on `jax.default_backend()`, which is the obvious thing to write and is
wrong.** On a correctly installed ROCm stack it returns `"gpu"`, not `"rocm"`, and so does
CUDA, so the assertion both fails on a working install and would not have caught the thing
you were worried about. `device.client.platform_version` is the field that actually names
the backend; on this book's stack it reads `PJRT C API` followed by `rocm 71400`. The device
repr is a decent second-best, since ROCm devices print as `rocm:0` while CUDA prints
`cuda:0`.

**Tip:** set `JAX_PLATFORMS=rocm`. The training images ship `libtpu` as a transitive
dependency, and it initialises on import, so every JAX program prints a paragraph of TPU
warnings about `TPU_ACCELERATOR_TYPE` and `TPU_WORKER_HOSTNAMES` before running perfectly
well on ROCm. It is noise rather than a symptom, but it is alarming noise in a log, and
naming the platform explicitly skips it entirely.

## The ROCm Version Matrix

**Each JAX ROCm plugin release targets one specific ROCm version, and mismatches fail in
unhelpful ways.** The `rocm7` plugin packages require a ROCm 7.x install.

**We are deliberately not reproducing the matrix here.** It changes every few months, and a
stale copy on this page is worse than a link:

- **[JAX on ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/jax-compatibility.html)**
  (AMD) is authoritative for which ROCm version each JAX release needs.
- **[Install JAX for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html)**
  (AMD) carries the validated container tags for the current ROCm release.

**Check the architecture target too**, not just the version. `rocm-smi --showproductname`
gives you the `gfx` string: `gfx942` is MI300-class (MI300X, MI325X) and `gfx950` is
MI350-class (MI355X). A wheel built for one will not run kernels on the other, and per
[Chapter 2]({{ '/pages/2-amd-gpus' | relative_url }}) the fp8 numerics differ between them
even when everything installs cleanly.

## Building From Source

**Only necessary if you are changing XLA or the plugin.** If you are reading this book to
train models, use the container; this section is for people fixing the profiler.

The ROCm forks, and which branch to build from:

| Repository | Branch to build | Notes |
|---|---|---|
| [ROCm/xla](https://github.com/ROCm/xla) | `main` | The fork's full XLA source tree |
| [ROCm/jax](https://github.com/ROCm/jax) | `amd-main` | The ROCm JAX integration branch |

Point JAX's build at your local XLA checkout, in `~/work/jax/.jax_configure.bazelrc`:

```
build --override_repository=xla="/root/work/xla"
build --override_module=xla="/root/work/xla"
```

**Absolute paths only**; `~` is not expanded here.

Then build the two plugin wheels, which is the common case since most changes do not need
`jaxlib` recompiled:

```bash
cd ~/work/jax
python3 build/build.py build \
    --wheels="jax-rocm-plugin,jax-rocm-pjrt" \
    --bazel_startup_options="--bazelrc=build/rocm/rocm.bazelrc" \
    --rocm_path=/opt/rocm \
    --rocm_amdgpu_targets=gfx942 \
    --python_version=3.12 \
    --local_xla_path=/root/work/xla \
    --bazel_options=--config=rocm_clang_hermetic \
    --output_path=/root/work/jax/dist \
    --verbose
```

And install them, forcing reinstall because pip skips same-version replacements:

```bash
pip install dist/jax_rocm7_pjrt-*.whl --force-reinstall
pip install dist/jax_rocm7_plugin-*.whl --force-reinstall
```

**Three things that will waste your afternoon.** `--rocm_amdgpu_targets` must match the
node's GPU or the kernels will not load. The wheels sometimes do not land in `dist/`,
because `build.py`'s copy glob can miss what Bazel produced, in which case find them under
`bazel-bin` and copy them yourself. And the pure-Python `jax` package is not built by
`build.py` at all, so if you changed the frontend you need `pip install -e .` separately.

## Known-Broken Combinations

> **Verified against:** `rocm/jax-training:maxtext-v26.5` (ROCm 7.14.0, `jax` 0.10.0,
> plugin wheels 0.10.0+rocm7.14.0), on 8x MI300X (gfx942) in SPX/NPS1, observed
> **5 August 2026**. The command-buffer row below was first observed on ROCm 7.2.4 with
> `jax` 0.11.0 in **July 2026**.

**HIP command-buffer capture segfaults on some embedding-gradient paths.** Symptom is a
crash inside the HIP runtime during a training step with an embedding layer, reproducible
at a high rate rather than intermittently. Workaround is to disable command buffers by
setting the flag to an empty value:

```bash
XLA_FLAGS="--xla_gpu_enable_command_buffer="
```

**That is an empty assignment, not a `false`**, and it disables all command-buffer
categories. The cost is losing whatever launch-overhead reduction command buffers were
providing, which for large kernels is negligible.

**Note:** `rocm/jax-training:maxtext-v26.5` already applies this workaround in the image's
own `XLA_FLAGS`, so if you are working inside that container you have it whether you meant
to or not. That is worth knowing before you conclude the bug is fixed.

**`libtpu` initialises on import and prints TPU errors on a ROCm-only box.** The training
images carry `libtpu` 0.0.40 as a transitive dependency. Every JAX program emits
`could not determine TPU accelerator type` and an `INVALID_ARGUMENT` about
`TPU_WORKER_HOSTNAMES` before running correctly on ROCm. **This is cosmetic**, and
`JAX_PLATFORMS=rocm` removes it:

```bash
JAX_PLATFORMS=rocm python your_script.py
```

**`jax.default_backend()` returns `"gpu"`, not `"rocm"`.** This is not a bug, but it breaks
the obvious installation check and it is the single most common way people convince
themselves a working install is broken. See the verification snippet above for what to
assert instead.

**Everything about the profiler's broken views is in
[Chapter 3]({{ '/pages/3-profiling' | relative_url }})** rather than here, because a reader
hitting those symptoms is reading a profile rather than installing software.

<!-- BLOCKED: two rows still owed, both needing a clean reproduction rather than a
     recollection. The container-tag question that used to be here is resolved: the book is
     now pinned to rocm/jax-training:maxtext-v26.5 and both this page and Appendix B quote
     it.

     Still to check:
       - Whether the four-package version-mismatch failure produces a usable error message
         or a silent CPU fallback. The verification snippet above assumes silent fallback;
         confirm and document the actual symptom. Needs deliberately installing a
         mismatched pair in a scratch container, which we have not done because it means
         breaking the working environment.
       - Whether the shm-size default really does cause a multi-process JAX failure, and
         what the failure looks like. Asserted above from experience, not from a clean
         reproduction. Note our own container runs with /dev/shm at 1008 GB, so we cannot
         observe the failure without deliberately constraining it. -->

## References

- [Install JAX for ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html)
  (AMD). The authoritative container tags, the four-package wheel install, and the
  bare-metal path.
- [JAX on ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/jax-compatibility.html)
  (AMD). Which ROCm version each JAX release requires, and the release cadence for the
  validated images.
- [ROCm/rocm-jax](https://github.com/ROCm/rocm-jax) (AMD). Sources for the ROCm JAX plugin,
  the `jaxlib` release assets, and the Dockerfiles behind the `rocm/jax` images.
- [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) (JAX). The
  `jax[rocm7-local]` extra, and the upstream statement that JAX ships no extra which
  installs ROCm itself.
- [rocm/jax on Docker Hub](https://hub.docker.com/r/rocm/jax/tags),
  [rocm/jax-community](https://hub.docker.com/r/rocm/jax-community/tags) and
  [rocm/jax-training](https://hub.docker.com/r/rocm/jax-training/tags). The tag lists, for
  pinning something other than `latest`. The last of these holds
  `maxtext-v26.5`, the tag every measurement in this book was taken in.
