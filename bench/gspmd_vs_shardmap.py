"""The same sharded matmul written twice, then diffed at the HLO level.

Chapter 4 argues that GSPMD and `shard_map` are two ways of saying the same
thing, with the difference being who writes the collective. That is a claim
about compiler output, so it should be checked against compiler output rather
than asserted.

The computation, both ways: `x[B, D] @ w[D, F]` with the contracting dimension D
split over the mesh. That is the case that needs an all-reduce, because every
device produces a partial sum of the whole output.

    python -m bench.gspmd_vs_shardmap
    python -m bench.gspmd_vs_shardmap --diff

Both optimised modules land in the run directory, so the diff is reproducible
rather than a screenshot.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from ._harness import Run, base_parser, configure_environment

# Differences that carry no meaning: the compiler numbers instructions in order
# of creation, and the two spellings create them in a different order.
NOISE = (
    (re.compile(r"fingerprint_before_lhs=\"[0-9a-f]+\""), 'fingerprint="..."'),
    (re.compile(r"scheduling_name=\"[^\"]*\""), 'scheduling_name="..."'),
    (re.compile(r"metadata=\{[^}]*\}"), "metadata={...}"),
    (re.compile(r"\.\d+\b"), ""),
    (re.compile(r"\bjit_\w+\b"), "jit_FN"),
)

COLLECTIVE_OP_RE = re.compile(
    r"=\s*\S+\s+(all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute)"
    r"(-start|-done)?\("
)


def build_parser():
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--d-ff", type=int, default=14336)
    p.add_argument("--devices", type=int, default=8)
    p.add_argument("--diff", action="store_true", help="print the unified diff")
    return p


def normalise(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("//", "FileNames", "FunctionNames", "FileLocations", "StackFrames")):
            continue
        for pattern, replacement in NOISE:
            line = pattern.sub(replacement, line)
        out.append(line)
    return out


def find_module(dump_dir: Path, fn_name: str) -> tuple[str, str]:
    hits = sorted(dump_dir.glob(f"*jit_{fn_name}*after_optimizations.txt"))
    if not hits:
        return "", ""
    return hits[-1].name, hits[-1].read_text()


def collectives_in(text: str) -> list[str]:
    return [
        m.group(1) + (m.group(2) or "")
        for line in text.splitlines()
        if (m := COLLECTIVE_OP_RE.search(line))
    ]


def main() -> int:
    args = build_parser().parse_args()

    # XLA reads the dump path when the backend starts, so it has to be set
    # before JAX comes up. Both variants dump into the same directory and are
    # told apart by module name, which is why the two functions below are named
    # rather than lambdas.
    stamp = Path(args.root or (Path(__file__).resolve().parent.parent / "runs"))
    dump_dir = stamp / "_hlo_gspmd_vs_shardmap"
    configure_environment(xla_flags={"xla_dump_to": str(dump_dir)})

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    n = min(args.devices, jax.device_count())
    mesh = Mesh(jax.devices()[:n], axis_names=("Y",))
    dtype = jnp.bfloat16
    B, D, F = args.batch, args.d_model, args.d_ff

    x_sharding = NamedSharding(mesh, P(None, "Y"))
    w_sharding = NamedSharding(mesh, P("Y", None))
    out_sharding = NamedSharding(mesh, P(None, None))

    # GSPMD: say where the data lives, let the partitioner work out the rest.
    def gspmd(a, b):
        return a @ b

    # shard_map: local shapes, and the all-reduce written by hand.
    def shardmap(a, b):
        @jax.shard_map(
            mesh=mesh, in_specs=(P(None, "Y"), P("Y", None)), out_specs=P(None, None)
        )
        def inner(a_shard, b_shard):
            return jax.lax.psum(a_shard @ b_shard, "Y")

        return inner(a, b)

    with Run("gspmd-vs-shardmap", tag=args.tag, root=args.root, freeze=not args.no_freeze) as run:
        run.note("shapes", {"B": B, "D": D, "F": F, "devices": n})
        run.note("sharded_dimension", "contracting (D), so both spellings owe an all-reduce")

        x = jax.device_put(jnp.ones((B, D), dtype), x_sharding)
        w = jax.device_put(jnp.ones((D, F), dtype), w_sharding)
        flops = 2.0 * B * D * F

        compiled = {
            "gspmd": jax.jit(
                gspmd, in_shardings=(x_sharding, w_sharding), out_shardings=out_sharding
            ),
            "shardmap": jax.jit(shardmap),
        }

        for name, fn in compiled.items():
            run.measure(
                lambda f=fn: f(x, w),
                name=name,
                warmup=args.warmup,
                repeats=args.repeats,
                flops=flops,
            )

        modules = {}
        for name in compiled:
            filename, text = find_module(dump_dir, name)
            modules[name] = text
            run.note(f"{name}_module_file", filename or "<not dumped>")
            run.note(f"{name}_collectives", collectives_in(text))
            if text:
                (run.hlo_dir).mkdir(parents=True, exist_ok=True)
                (run.hlo_dir / f"{name}.txt").write_text(text)

        print("\n=== collectives the compiler emitted ===")
        for name in compiled:
            print(f"  {name:<10} {collectives_in(modules[name]) or ['<none>']}")

        a, b = normalise(modules["gspmd"]), normalise(modules["shardmap"])
        diff = list(
            difflib.unified_diff(a, b, fromfile="gspmd", tofile="shardmap", lineterm="", n=1)
        )
        identical = bool(a) and not diff
        run.note("optimised_hlo_identical", identical)
        run.note("diff_lines", len(diff))
        run.note("gspmd_instructions", len(a))
        run.note("shardmap_instructions", len(b))
        run.hlo_dir.mkdir(parents=True, exist_ok=True)
        (run.hlo_dir / "diff.txt").write_text("\n".join(diff) or "<identical>")

        print(
            f"\n{len(a)} vs {len(b)} normalised lines; optimised HLO "
            f"{'is identical' if identical else f'differs in {len(diff)} diff lines'}"
        )
        if args.diff and diff:
            print("\n".join(diff[:150]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
