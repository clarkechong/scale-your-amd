# Literal HLO figure fixtures

These directories contain real HLO text and DOT output captured from
`bench/hlo_feature_fixtures.py` on MI355X/gfx950 with the JAX/ROCm stack recorded
in each `provenance.json`.

Regenerate the complete set with:

```bash
python3 tools/capture_hlo_feature_svgs.py
```

The capture tool runs every arm in a fresh process, retains only the named fixture
module and debug options, and renders XLA's DOT output directly to
`pages/img/hlo-*.svg`.

The LHS figures are derived from literal operation lines in scheduled HLO rather
than from the dependency DOT graph. Their arrows mean file/schedule order. The raw
scheduled HLO and XLA DOT graph are retained beside the provenance.

These fixtures are explanatory compiler artifacts. They are not timing
benchmarks, full-model captures, or `[measured]` case-study evidence.
