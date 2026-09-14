---
layout: distill
title: "Operating Multi-Node MI355X Training"
description: "The acceptance specification for launching, placing, feeding, checkpointing, restarting, and measuring a multi-node JAX training job."
date: 2026-09-13

section_number: 14

previous_section_url: "/pages/13-mixtral8-22b"
previous_section_name: "Chapter 13: Mixtral 8x22B"

next_section_url: "/pages/15-deepseek-v3"
next_section_name: "Chapter 15: DeepSeek V3"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: Status and Exit Criteria
  - name: Freeze the Run Contract
  - name: Launch Contract
  - name: Topology and Affinity
  - name: Dataset Contract
  - name: Checkpoint and Restart Contract
  - name: Monitoring and Failure Records
  - name: Scaling Gates
  - name: Required Outputs
---

> **Future acceptance specification.** This chapter contains no multi-node performance
> result. It becomes a measured chapter only after every gate below has a retained artifact
> bundle conforming to [Appendix F]({{ '/pages/f-appendix-artifacts' | relative_url }}).

**Depends on:** [Chapter 7]({{ '/pages/7-sharding' | relative_url }}) for mesh axes and
collectives, [Chapter 10]({{ '/pages/10-flags' | relative_url }}) for runtime controls,
and [Chapter 13]({{ '/pages/13-mixtral8-22b' | relative_url }}) for the last single-node
baseline.

## Status and Exit Criteria

The current repositories run on one MI355X node. They do not establish multi-process launch,
cross-node placement, dataset uniqueness, checkpoint recovery, or scaling. Until those are
tested, inter-node behavior is **unverified**, not **[measured]**.

This chapter exits future status when:

1. the same immutable container digest and recorded source revisions start cleanly
   on one and then multiple nodes;
2. every process reports the expected global device set and mesh;
3. GPU, CPU NUMA domain, and NIC placement are recorded for every rank;
4. real-data sample assignment has no unexplained duplication or omission;
5. a checkpoint restores model, optimizer, step, and data position after a forced stop;
6. monitoring identifies the failed rank and last completed global step;
7. each scaling stage passes its predeclared correctness and throughput gate; and
8. configs, logs, traces, manifests, and checksums are published together.

Passing launch alone is not enough. A job that starts but cannot resume deterministically is
not accepted.

## Freeze the Run Contract

Create `run-contract.yaml` before allocating the cluster. It must record:

| Field | Required value |
|---|---|
| Hardware | Node count, MI355X count per node, partition mode, host CPU/NUMA layout |
| Network | NIC model, firmware, link state, MTU, RDMA device, switch/rail map |
| Software | Container tag and digest, ROCm/JAX/plugin/PJRT/RCCL versions, all source commits |
| Workload | Model config, dtype roles, sequence length, batch vocabulary, optimizer, step count |
| Process model | Processes per host, visible devices per process, rank assignment |
| Mesh | ICI and DCN axis names, sizes, and physical placement |
| Data | Dataset manifest hash, tokenizer hash, shuffle seed, host-sharding rule |
| Recovery | Checkpoint path, cadence, retention, restore mode, data-cursor policy |
| Protocol | Warmup, measured window, synchronization, statistic, profiler-free timing command |
| Gates | Numerical tolerances and scaling thresholds declared before the run |

Do not infer this information from a log after the run. The contract is an input and the
runtime manifest in [Appendix F]({{ '/pages/f-appendix-artifacts' | relative_url }}) is the
record of what actually ran. A difference between them fails the configuration gate.

## Launch Contract

Use one explicit rank assignment. Do not let hostnames, scheduler order, and device
visibility each define a different ordering.

The launcher must provide these values to every process:

```text
coordinator_address
num_processes
process_id
local_device_ids
node_id
local_process_id
initialization_timeout
heartbeat_timeout_seconds
shutdown_timeout_seconds
coordinator_bind_address
```

Initialize JAX before creating arrays, importing code that queries devices, or constructing
the mesh:

```python
jax.distributed.initialize(
    coordinator_address=coordinator_address,
    num_processes=num_processes,
    process_id=process_id,
    local_device_ids=local_device_ids,
    initialization_timeout=initialization_timeout,
    heartbeat_timeout_seconds=heartbeat_timeout_seconds,
    shutdown_timeout_seconds=shutdown_timeout_seconds,
    coordinator_bind_address=coordinator_bind_address,
)
```

The process model is a decision, not an assumption. Record whether one host process controls
all local GPUs or each process controls a subset. The selected MaxText revision and launcher
must support that model. Do not mix the two within one run.

Each rank writes a startup record containing:

```text
hostname, pid, process_id, local_process_id
HIP_VISIBLE_DEVICES
jax.process_count(), jax.process_index()
jax.device_count(), jax.local_device_count()
device id, device kind, platform version
coordinator address and resolved IP
```

The launch gate passes only if process IDs are unique and contiguous, all ranks agree on the
global device count, each intended device appears exactly once in the ownership map, and a
cross-rank barrier completes. The launcher passes initialization, heartbeat, shutdown, and
coordinator-bind settings to JAX; the runtime enforces them. Record their effective values
in the run contract. Application retry policy belongs in the launcher rather than the
training loop.

## Topology and Affinity

Capture topology before running a collective:

```bash
rocm-smi --showproductname
rocm-smi --showcomputepartition
rocm-smi --showmemorypartition
rocm-smi --showtopo
numactl --hardware
lspci -tv
rdma link
ibdev2netdev
```

Store the raw output per node. If a command is unavailable in the pinned image, record that
as a compatibility result rather than substituting an undocumented command.

Build one mapping with these columns:

| Rank | Host | CPU set | NUMA node | Local GPU | PCI address | RDMA device | NIC/rail |
|---|---|---|---|---|---|---|---|

Pin the host process to the CPU and NUMA domain chosen for its device and NIC. Record the
actual affinity after launch. A requested binding that the scheduler silently ignores does
not pass.

Mesh placement is accepted only when:

- ICI axes remain within the intended node boundaries;
- DCN axes cross the intended hosts and rails;
- compiled HLO replica groups match the physical rank map;
- a trace contains the expected collective type and participant count; and
- no rank reaches an inter-node peer through an accidental local-GPU relay unless that route
  is part of the declared topology plan.

The collective sweep needed to choose between legal placements is a prerequisite to a
performance recommendation. This chapter does not predict its result.

## Dataset Contract

Synthetic reused data is allowed for launch and compiler gates. It cannot pass the
real-data gate.

For a real dataset, publish:

- immutable file names, sizes, and hashes;
- tokenizer files and hashes;
- train/evaluation split definitions;
- packing and maximum-segment settings;
- shuffle seed and epoch construction;
- the rule mapping `(epoch, process_id, data_process_index, optimizer_step,
  accumulation_microstep, batch_slot, packed_segment)` to input records; and
- the checkpointed data cursor or the deterministic rule used to reconstruct it.

Instrument a bounded audit window with sample IDs before tokenization and after packing.
Across all data-consuming processes, the merged audit must show exactly the duplication and
dropping policy declared in the run contract. The audit must be repeated after restart.

Measure host readiness separately from device step time. A trace should contain named input,
transfer, and train-step ranges. Unexplained gaps before a train step fail the input gate even
if the model produces finite loss.

## Checkpoint and Restart Contract

Use a shared path whose durability and visibility are tested from every host. Record whether
Orbax writes sharded state directly or stages data elsewhere.

A checkpoint is complete only when it contains or references:

- model parameters;
- optimizer state and loss-scaling state;
- global step and learning-rate schedule position;
- random-number state required by the recipe;
- mesh and sharding metadata needed for restore;
- dataset epoch, shard, and record position, or a deterministic equivalent;
- the exact config, environment manifest, and source revisions; and
- a completion marker written after all shards and metadata are durable.

Run three recovery tests:

1. clean save and restore with the original allocation;
2. forced termination during training, followed by restore from the last complete
   checkpoint; and
3. termination during a checkpoint write, proving that the incomplete checkpoint is rejected.

After restore, compare the first resumed batch IDs, step number, schedule value, parameter
tree, optimizer tree, and the next-step loss against the declared tolerance. A run that
restores weights but restarts its data order or schedule does not pass.

Checkpoint cadence is chosen from measured save duration, restore duration, storage budget,
and expected interruption rate. Record those inputs; do not copy a cadence from a
single-node run.

## Monitoring and Failure Records

Every rank emits structured records with synchronized wall time, monotonic time, process ID,
host, global step, compile state, input wait, step duration, loss, gradient norm, tokens
processed, checkpoint state, and last collective entered.

The cluster view must include:

- process liveness and restart count;
- GPU memory, power, clocks, utilization, and throttling state;
- NIC link state, errors, drops, and traffic per interface;
- filesystem capacity and checkpoint write state;
- compile count and cache hits;
- step-time distribution by rank; and
- router and per-expert load for MoE runs.

Classify each failure as launch, rendezvous, device, network, collective, input, filesystem,
checkpoint, compile, out-of-memory, or numerical. Add the exact stack and workaround to
[Appendix E]({{ '/pages/e-appendix-compatibility' | relative_url }}). Do not relabel a hang as
a slow run without evidence of forward progress.

Availability-adjusted throughput is reported separately from steady-state throughput:

$$
\text{available tokens/s}
=
\frac{\text{accepted training tokens}}
{\text{allocation wall time, including recovery}}.
$$

This is a required output, not a claim about the future system.

## Scaling Gates

Use the same model, global batch definition, precision recipe, data manifest, optimizer, and
measured-step protocol at every stage. If memory forces a batch or mesh change, start a new
scaling series.

| Gate | Run | Must be true before advancing |
|---|---|---|
| A | One process, one GPU | Reference loss and gradients pass; artifact capture works |
| B | Declared process model, one node | Device ownership and mesh are exact; results match Gate A within the declared tolerance |
| C | Two nodes, synthetic data | Rendezvous, cross-node collectives, HLO groups, and rank-local metrics are complete |
| D | Two nodes, real data | Sample audit, checkpoint, forced-stop restart, and monitoring tests pass |
| E | Each larger allocation | Aggregate tokens/s increases over the prior stage and the predeclared efficiency floor is met |
| F | Sustained run | No unexplained loss divergence, repeated compilation, data starvation, or unrecovered rank failure |

Declare the efficiency floor and noise band before Gate E. Report aggregate tokens/s,
tokens/s/GPU, median step time, dispersion, exposed collective time, and
availability-adjusted throughput. A failed gate remains a **[measured] negative result** and
must not be removed from the scaling series.

## Required Outputs

Publication requires:

- scheduler submission file and fully expanded launch command;
- rank, GPU, CPU, NUMA, NIC, and rail map;
- run contract and runtime manifest;
- dataset and tokenizer manifests;
- mesh config plus optimized HLO showing replica groups;
- one unprofiled timing log per scaling point;
- synchronized per-rank logs and monitoring export;
- XProf and lossless `rocprofv3` traces for the selected points;
- complete and interrupted checkpoint records plus restore logs;
- scaling and availability result files with raw samples;
- all negative results and workarounds; and
- checksums for every published artifact.

Chapter 15 may use this infrastructure only after Gates A through D pass with a smaller
workload.
