#!/usr/bin/env python3
"""Capture one instrumented Mixtral 8x22B step for Chapter 3 attribution."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
MAXTEXT = Path(
    os.environ.get(
        "MAXTEXT_ROOT",
        WORKSPACE / "maxtext-ch3-attribution",
    )
)
CONFIG = REPO / "configs" / "mixtral8-22b.yml"
FLAG_FILE = REPO / "configs" / "flags" / "rocm.txt"

timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
tag = os.environ.get("ATTRIBUTION_TAG", f"{timestamp}-mixtral8x22b-ch3-xprof")
output = Path(
    os.environ.get(
        "ATTRIBUTION_OUTPUT_ROOT",
        WORKSPACE / "scale-your-amd" / "runs" / tag,
    )
)
output.mkdir(parents=True, exist_ok=False)
(output / "maxtext").mkdir()

flags = " ".join(
    line.split("#", 1)[0].strip()
    for line in FLAG_FILE.read_text().splitlines()
    if line.split("#", 1)[0].strip()
)
env = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}
env.update(
    {
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "JAX_PLATFORMS": "rocm",
        "NVTE_FRAMEWORK": "jax",
        "PYTHONPATH": f"{MAXTEXT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":"),
        "XLA_FLAGS": flags,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.97",
    }
)

recipe = [
    "dtype=bfloat16",
    "quantization=",
    "ici_fsdp_parallelism=4",
    "ici_expert_parallelism=2",
    "sparse_matmul=false",
    "megablox=false",
    "capacity_factor=1.0",
    "ragged_buffer_factor=-1.0",
    "moe_dispatch_no_expert_sharding=true",
    "num_experts=8",
    "num_experts_per_tok=2",
    "use_custom_sort_vjp=true",
    "use_ragged_sort=false",
    "use_ring_of_experts=false",
    "use_tokamax_gmm=false",
    "profiler=xplane",
    "skip_first_n_steps_for_profiler=3",
    "profiler_steps=1",
    "profile_cleanly=true",
    "steps=5",
    f"base_output_directory={output / 'maxtext'}",
    f"run_name={tag}",
]
command = [
    sys.executable,
    "-m",
    "maxtext.trainers.pre_train.train",
    str(CONFIG),
    *recipe,
    *sys.argv[1:],
]

provenance = {
    "tag": tag,
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "maxtext_root": str(MAXTEXT),
    "maxtext_revision": subprocess.check_output(
        ["git", "-C", str(MAXTEXT), "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "instrumentation_diff": subprocess.check_output(
        ["git", "-C", str(MAXTEXT), "diff", "--stat"],
        text=True,
    ).strip(),
    "command": command,
    "xla_flags": flags,
    "configured_skip_steps": 3,
    "profiled_steps": 1,
}
(output / "instrumentation.patch").write_text(
    subprocess.check_output(
        ["git", "-C", str(MAXTEXT), "diff", "--binary"],
        text=True,
    )
)
(output / "runner.py").write_text(Path(__file__).read_text())
(output / "config.yml").write_text(CONFIG.read_text())
(output / "xla_flags.txt").write_text(flags + "\n")
(output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

with (output / "train.log").open("w") as log:
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return_code = process.wait()

provenance["return_code"] = return_code
provenance["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
log_text = (output / "train.log").read_text()
profile_match = re.search(
    r"Profiler session started.*?completed step: (\d+)",
    log_text,
    re.DOTALL,
)
provenance["observed_profiled_step"] = int(profile_match.group(1)) if profile_match else None
provenance["step_times_seconds"] = {
    int(step): float(seconds)
    for step, seconds in re.findall(r"completed step: (\d+), seconds: ([0-9.]+)", log_text)
}
(output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
raise SystemExit(return_code)
