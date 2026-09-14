---
layout: distill
title: "Case-Study Artifact Schema"
description: "The directory layout, manifest fields, checksums, and publication rules required for measured claims."
date: 2026-09-13

section_label: "Appendix F"

previous_section_url: "/pages/e-compatibility-and-negative-results"
previous_section_name: "Appendix E: Compatibility"

next_section_url: ""
next_section_name: "End of the book"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Bundle Layout
  - name: Manifest Schema
  - name: Result Records
  - name: Required Files by Claim
  - name: Checksums and Publication
---

A **[measured]** claim points to one immutable bundle. Configs and scripts explain intent;
the manifest records what the process actually used.

## Bundle Layout

```text
artifacts/<case_id>/<run_id>/
├── manifest.yaml
├── run-contract.yaml
├── series.yaml
├── checksums.sha256
├── config/
│   ├── effective.yaml
│   ├── command.txt
│   ├── environment.txt
│   ├── xla_flags.txt
│   ├── source-revisions.yaml
│   └── patches/
├── hardware/
│   ├── devices.txt
│   ├── partitions.txt
│   ├── topology.txt
│   ├── numa.txt
│   ├── network.txt
│   ├── rail-map.csv
│   └── rank-affinity.csv
├── data/
│   ├── dataset-manifest.json
│   ├── tokenizer-manifest.json
│   └── sample-audit.jsonl
├── logs/
│   ├── stdout.log
│   ├── stderr.log
│   ├── steps.jsonl
│   └── rank-<id>.jsonl
├── results/
│   ├── metrics.json
│   ├── timing-samples.csv
│   └── acceptance.json
├── ledgers/
│   ├── model.json
│   ├── training-state.json
│   ├── mla.json
│   ├── moe.json
│   └── mtp.json
├── components/
│   ├── correctness.json
│   └── test-vectors/
├── monitoring/
│   ├── events.jsonl
│   ├── failures.jsonl
│   └── recovery.jsonl
├── hlo/
│   ├── before-optimizations.txt
│   ├── optimized.txt
│   └── memory-analysis.txt
├── profiles/
│   ├── xprof/
│   ├── rocprof/
│   └── rocprof-compute/
├── checkpoints/
│   ├── inventory.json
│   └── restore-tests.jsonl
└── fallback/
    ├── effective.yaml
    └── command.txt
```

Omit a directory only when the claim does not need it. Record the omission and reason in the
manifest.

## Manifest Schema

`artifacts/schema.json` is the machine-validatable JSON Schema for this
manifest. YAML manifests must deserialize to the same data model before
validation.

`manifest.yaml` uses this minimum schema:

```yaml
schema_version: 1
case_id: llama70b-precision
run_id: 20260913T210000Z-bf16-001
evidence_label: measured
claim_ids: [ch12-bf16-baseline]
bundle_status: complete
run_outcome: pass
started_at_utc: 2026-09-13T21:00:00Z
completed_at_utc: 2026-09-13T21:20:00Z

hardware:
  accelerator: MI355X
  architecture: gfx950
  nodes: 1
  devices_per_node: 8
  compute_partition: SPX
  memory_partition: NPS1
  device_inventory_file: hardware/devices.txt
  topology_file: hardware/topology.txt
  network_file: hardware/network.txt
  rail_map_file: hardware/rail-map.csv
  rank_affinity_file: hardware/rank-affinity.csv

software:
  container: docker.io/rocm/jax-training:maxtext-v26.6
  container_digest: required
  rocm: required
  python: required
  jax: required
  jaxlib: required
  pjrt_plugin: required
  rccl: required
  xprof: required
  repositories:
    maxtext: {commit: required, dirty: false}
    transformer_engine: {commit: null, dirty: false}
    jax_aiter: {commit: null, dirty: false}
    aiter: {commit: null, dirty: false}

workload:
  model_name: required
  model_config_hash: required
  precision_recipe: required
  sequence_length: required
  per_device_batch_size: required
  global_sequences_per_step: required
  global_nonpadding_tokens_per_step: required
  gradient_accumulation_steps: required
  steps: required
  dataset_kind: synthetic_or_manifest
  dataset_manifest_sha256: null
  tokenizer_manifest_sha256: null

launch:
  command_file: config/command.txt
  environment_file: config/environment.txt
  xla_flags_file: config/xla_flags.txt
  process_count: required
  device_visibility: required
  initialization_timeout: required
  heartbeat_timeout_seconds: required
  shutdown_timeout_seconds: required
  coordinator_bind_address: required

mesh:
  ici: required
  dcn: required
  logical_axis_rules_hash: required

protocol:
  warmup_steps: 10
  measured_steps: 20
  synchronization: block_until_ready
  statistic: median
  timing_profiled: false
  cache_policy: required
  clock_policy: required

ledgers:
  model: ledgers/model.json
  training_state: ledgers/training-state.json
  mla: null
  moe: null
  mtp: null

series:
  series_id: null
  point_index: null
  fixed_contract_hash: null

monitoring:
  events_file: null
  failures_file: null
  recovery_file: null

fallback:
  config_file: null
  command_file: null

artifacts:
  checksum_file: checksums.sha256
  omissions: []
```

Use `null` only when a component is not imported by the run. `required` is a placeholder in
this reference, not a valid published value.

`bundle_status` is `complete` or `blocked`. `run_outcome` is `pass`, `fail`,
`aborted`, or `not_run`. A retained negative result can therefore have
`bundle_status: complete` and `run_outcome: fail`. Missing evidence is
`bundle_status: blocked`; it is not a failed run.

## Result Records

`results/metrics.json` contains values and definitions:

```yaml
step_time:
  unit: seconds
  statistic: median
  median: required
  minimum: required
  maximum: required
  mean: required
  median_absolute_deviation: required
  samples_file: results/timing-samples.csv
throughput:
  aggregate_tokens_per_second: required
  tokens_per_second_per_gpu: required
  token_definition: global_nonpadding_training_tokens
memory:
  unit: bytes
  peak_per_device: required
compile:
  first_compile_seconds: required
  compile_count: required
collectives:
  total_seconds: null
  exposed_seconds: null
quality:
  train_loss: required
  validation_loss_file: null
kernel_route:
  expected: required
  observed: required
  proof_file: required
```

JSON files use the same keys. Store integer byte and token counts without rounded display
units. Derived tables and plots are outputs; raw samples remain the source.

`results/acceptance.json` records each predeclared gate:

```yaml
gates:
  - id: kernel-route
    criterion: expected backend appears in HLO and trace
    status: pass
    evidence: [hlo/optimized.txt, profiles/rocprof/kernels.db]
  - id: convergence
    criterion: declared before run
    status: not_applicable
    evidence: []
```

Allowed statuses are `pass`, `fail`, `blocked`, and `not_applicable`. A failed gate does not
delete the bundle.

## Required Files by Claim

| Claim | Minimum additional evidence |
|---|---|
| Step time or tokens/s | Unprofiled timing samples, synchronization proof, batch/token definition |
| Peak memory | Compiled analysis plus runtime high-water mark from the same config |
| Kernel/backend route | Optimized HLO plus kernel trace |
| Collective placement | HLO replica groups, trace, rank-affinity map |
| Profiler attribution | Native XProf or lossless `rocpd`, not only a screenshot |
| Hardware-counter result | Raw PMC passes, counter list, tool version, analysis output |
| Convergence | Dataset/tokenizer manifests, all train/eval metrics, seed, comparison script |
| Checkpoint/restart | Checkpoint inventory, interrupted-write test, restore log, resumed sample audit |
| Negative result | Inputs, complete error/hang/fallback evidence, workaround test |
| Multi-node scaling | One bundle per point plus a series manifest fixing workload and gates |
| Multi-node operation | Run contract, network/rail map, rank monitoring, failure and recovery records |
| DeepSeek component | Machine-readable MLA, MoE, MTP, and training-state ledgers plus test vectors |
| DeepSeek capstone | Full-model fit proof, component gates, scaling series, quality smoke test, fallback config |

A figure in the book links to the bundle and names the script that generated it.

## Checksums and Publication

Create checksums after the bundle is complete:

```bash
cd artifacts/<case_id>/<run_id>
find . -type f ! -name checksums.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum > checksums.sha256
sha256sum -c checksums.sha256
```

Before publication:

1. remove credentials, tokens, private hostnames, and user data;
2. keep rank ordering and topology relationships intact when anonymizing;
3. verify checksums from a clean checkout;
4. run the analysis script from the bundle without unpublished files;
5. mark incomplete bundles with `bundle_status: blocked`; record a retained failed
   run as `bundle_status: complete` and `run_outcome: fail`; and
6. give the bundle a content-addressed or immutable release URL.

If a result is superseded, retain its bundle and add the replacement run ID. Do not edit an
old bundle in place.
