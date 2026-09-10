---
layout: distill
title: "Profiling"
description: "Roofline analysis, XProf, rocprofv3, rocprof-compute, and rocgdb, from a train_step down to a single kernel."
date: 2026-09-10

section_number: 4

previous_section_url: "/pages/3-dl-methods"
previous_section_name: "Chapter 3: Deep Learning Methods"

next_section_url: "/pages/5-llama7b"
next_section_name: "Chapter 5: Llama 7B"

authors:
  - name: Clarke Chong
    url: "https://github.com/clarkechong"

toc:
  - name: "Roofline Analysis"
  - name: "JAX and ROCm Profiling Tool Stack"
    subsections:
      - name: "Trace Analysis with XProf"
      - name: "Trace Analysis with rocprofv3 and ROCTx range markers"
      - name: "Hardware Counter Analysis with rocprof-compute"
  - name: "Debugging with rocgdb"
  - name: "End-to-End Performance"
---
## Roofline Analysis

TODO

- theoretical roofline first: peak FLOP/s per datatype and peak HBM bandwidth, both from the hardware chapter
- the ridge point, ie. the FLOPs per byte you need to saturate the Matrix Cores (288 for BF16 on MI355X)
- arithmetic intensity worked for two cases: a large matmul (compute-bound) and an elementwise chain (bandwidth-bound)
- how the datatype moves the answer: fp32 vs bf16 vs fp8/MXFP4 raise the compute ceiling but not the HBM ceiling, so the ridge point moves right
- memory-bound vs compute-bound as a decision rather than a label: reuse, fusion and tiling in one case, larger tiles and better MFMA patterns in the other
- theoretical vs empirical roofline, ie. why a measured HBM line lands well below spec, using the `rocprof-compute` roofline below
- multi-level rooflines (LDS, L1, L2, Infinity Cache) and what a single HBM line hides about cache reuse
- one worked example carried end to end: predict, measure, state whether the model held

START:

Roofline analysis is a useful top-level analysis methods for high performance computing workloads. It measures the achieved hardware utilization (FLOPS) of the workload in proportion to its peak theoretical value. Roofline analysis can be applied hierarchically. For example, lets consider an XLA module `train_step` as the top-level workload we want to analyze. If we perform roofline analysis on this module and find it operates far below the roofline, we can decompose it into ops, then into kernels, and perform roofline analysis at each level. The goal is to identify specific optimisation opportunities and target improvement efforts accordingly.

So how does roofline analysis work?

First we should understand that a workload tends to be either compute bound or memory bound, hence any roofline value should be determined by a combination of these two factors. We already know from Section 1 (Hardware) that the GPU spec provides peak FLOPS and peak HBM bandwidth. We quantify this balance using the concept of Arithmetic Intensity (AI), measured in FLOPs per byte. In other words, how much computation is performed per byte of data moved to or from memory.

Lets understand this through 2 example workloadS: 1) a `memcopy` kernel which simply moves data over HBM with no calculations, and 2) a large GEMM kernel.

We should expect the `memcopy` kernel to be memory-bound as the kernel will saturate the HBM bandwidth and this is the limiting factor. The speed of this kernel is determined solely by the peak HBM bandwidth. From the perspective of AI, FLOPs per byte, this kernel has 0 FLOPs per byte of data moved. But remember our end metric is hardware utilization in terms of FLOPS. Hence, a `memcopy` kernel that inherently performs no FLOPs, has a roofline value of 0 FLOPs!

Now consider the large GEMM kernel. For a square GEMM, the FLOPs complexity is O(n^3), and the memory traffic complexity is O(n^2). Which means AI for a square GEMM is O(n). For our hypothetical large N, we have a large AI and are compute bound. Here, our roofline value in FLOPs is the peak FLOPs of the GPU. 

 where the time complexity grows O(N^2) or O(N^3) for a batched GEMM. At large N these kernels tend towards being purely compute bound, in which case the roofline is the peak FLOPs of the hardware. Here, the speed of this kernel is determined by the peak FLOPs for the datatype being used.
 increasing the peak FLOPS (for that datatype specifically) increases the kernel speed. we have N^2 or N^3 flops per byte (a very large arithmetic intensity for large GEMMs!). 

In other words, the AI determines which roofline applies to our workload. 

<roofline diagram: higher AI means we are bound by peak FLOPS (right side of the ridge point), low AI means we are bound by the HBM bandwidth. >

- notice the labelled ridge point. 
- to the left of this we are memory bound. as the AI increases from 0, the roofline for that workload is the max theoretical flops required by the workload. flops/byte(AI) * bytes/s = flop/s. up until this value saturates to the hardware ceiling (which happens at the ridge point and hence flattens out) this increases linearly with AI
- to the right of the ridge point we are flat at peak flops of the hardware
- at the ridge point we are at the exact peak of the hardware without saturating. any more AI and we are limited by peak flops compute, and any less we are bound by HBM bandwidth

this makes sense but we also know from 1-hardware that we have differnet flops per datatype AND different mem bandwidth per HBM vs l1 cache vs within node infinity fabric bandwidth.

how does this affect the roofline? lower FP formats have higher max FLOPS throughput so they require more flops/byte to saturate. the ridge point moves right relative to AI.

how about memory bandwidths? within-node has a lower memory bandwidth. (less bytes/s). Since memory is slower, you need a higher arithmetic intensity before compute becomes the bottleneck.

<visual diagram: conveyer belt -> matrix unit>
1. memory bound: slow conveyer belt, matrix unit is instantly churning through, no backlog
2. compute bound: rapid conveyer belt, matrix unit cant process faster than data arriving. 2
3. ridge point: matched speed, data arrives on conveyer belt at same rate matrix unit processes it.
4. AI: 0 AI means matrix unit spends 0 time processing (passes straight through). increase AI means matrix unit spends more time (more processing, more FLOPS) per unit of data.
knobs: HBM speed, peak FLOPS, AI of workload

so what are the takeaways? the final thing that affects your actual final FLOPs (utilization) are 
1. how well your kernel performs relative to its AI roofline (of course, in reality we arent always hitting the theoretical peak of our workload due to execution factors such as: cache misses due to mem access patterns, ... ETC LIST MORE)
2. the design of the algorithm in terms of AI itself (if you can improve the AI of an algorithm to push it to to compute bound region, overall flops increases) but this is not always possible eg. a memcopy kernel inherently has no compute. but a chain of low AI operations can be FUSED together into one larger AI (and hence greater roofline) kernel (KERNEL FUSION!), or an algorithm can inherently have better access patterns (tiling, data reuse).


see <https://jax-ml.github.io/scaling-book/roofline/> for a more concrete mathematical view now that you have a baseline understanding. this walks thorugh a computed example for particular kernels.

but remember, roofline analysis is just one component of profiling. it can identify which kernels are performing relatively poorly but it doesnt tell you why. For that you need to look into the trace/timeline of actual GPU events and even hardware counters which preserve details about actually achieved hardware utilization.

---

## JAX and ROCm Profiling Tool Stack

there are different profiling tools for working at different abstraction levels. for example at early stage profiling (still identifying phase), framework level profiler is ideal. ideally we would liek unified profiler to see from jax level to kernel level (which xprof would like to be) but current limitations means that low level kernel details (such as what?) are not visible to jax profiler. so we use jax profiler for high level roofline analysis and trace identification, and as we identify items down the stack we go to lower level profiling tools. 

<diagram here showing stack: python, framework(jax), framework(xla){hlo modules}{hlo ops}, kernel library, gpu stream, and showing where each profiler operates in the stack>


### Trace Analysis with XProf

XProf is visualizer for the JAX profiler. Capture with `jax.profiler`:

```python
import jax

with jax.profiler.trace("/tmp/trace"):
    for _ in range(5):
        train_step(...)
```

```bash
xprof --logdir /tmp/trace
```

You get an `.xplane.pb` as the native output format, with a lossy timeline-only trace `.gz.json` which can be fed into Perfetto or other generic trace viewers.

The `.xplane.pb` is formatted as such:

![]({{ '/pages/img/what-is-an-xspace.png' | relative_url }})

- **XSpace** is the entire trace/capture.
- **XPlane** is one device or host component within it, e.g. `/device:GPU:0` or `/host:CPU`. An 8 GPU node would contain 8 GPU XPlanes + 1 host XPlane
- **XLane** is a lane within the plane. A GPU XPlane will typically contain a `Stream` lane which records stream-level events (ie. actual kernels that ran), along with framework-level lanes which correlate a framework-level op down to its stream-level event.
- **XEvent** is an event on the lane: a kernel dispatch, an op, an API call.
- **XStats** are the key/value pairs hanging off an event. It contains metadata, e.g. `correlation_id`, `hlo_op`, `hlo_module`, and `kernel_details` carrying `regs`, `grid`, `block` and `occ_pct`.

The correlation between framework-level ops (JAX, XLA) to their GPU kernels is the main benefit of profiling through XLA as oppose to profiling with proprietary AMD tools such as `rocprof`. However, AMDs proprietary profiling tools will likely provide a more detailed and accurate result especially from a lower-level perspective (FLOPs, utilization, roofline)

XProf offers various tools which are best used for a top-level profiling view:

- `trace_viewer` for the timeline, ie. what ran when, on which stream, and what overlapped
- `op_profile` and `framework_op_stats` for time by op, and `hlo_stats` for the same by HLO instruction
- `memory_profile` and `memory_viewer` for allocation over time and the peak-memory breakdown
- `kernel_stats` for per-kernel duration and launch geometry
- `graph_viewer` for the HLO graph, which is the same DAG we dumped earlier

So, use XProf for the timeline, for op and HLO attribution, and for memory. For accurate and precise FLOPS/hardware utilization metrics per a specific identified kernel, you should use `rocprof` or `rocprof-compute`.

---

### Trace Analysis with rocprofv3 and ROCTx range markers

`rocprofv3` sits a layer below XProf. On its own, it knows nothing about the framework level (HLO or JAX), only about the ROCm runtime and the device.

!! ROCTx range markers allow manual ranges to be inserted and rocprof will autocorrelate these roctx into the trace (--kernel-rename to align kernel with range, otherwise range is seperate timeline). however this has to be done manually! There is work in progress to bring automatic roctx range attribution to XLA (ie patch xla ops/modules as roctx ranges so they can be picked up by rocprof).

```bash
rocprofv3 <collection-modes> -- <command-to-profile>
```

It collects device-side timelines (traces) and, separately, hardware counters via `--pmc`. Tracing has individual types:

- `--kernel-trace` kernel dispatches
- `--memory-copy-trace` / `--memory-allocation-trace` copies and allocations (address, size, agent)
- `--marker-trace` Marker (ROCTx) ranges
- `--hip-runtime-trace` the HIP API (`hip*`), `--hsa-core-trace` the HSA API (`hsa_*`)
- `--rccl-trace` RCCL collectives
- `--att`/`--advanced-thread-trace` instruction-level thread trace, separate and heavyweight

and aggregate bundles:

- `--hip-trace` HIP API only, ie. NOT kernels or memcpys
- `-r`/`--runtime-trace` HIP runtime + marker + RCCL + memory ops + kernel dispatches
- `-s`/`--sys-trace` everything, including the HSA API

`-f` sets the output format independently of the collection mode:

- `pftrace` a Perfetto trace, viewable at `ui.perfetto.dev`. Convenient but lossy: a flattened timeline that drops the relational schema, static kernel metadata and structured counters.
- `csv` plain tables, one file per domain. `json` the same, structured.
- `rocpd` a sqlite `.db`, the lossless form
- `otf2` Open Trace Format 2, for HPC viewers such as Vampir or TAU

You can pass several at once:

```bash
rocprofv3 --kernel-trace -f pftrace rocpd -d prof -o vadd_trace -- /tmp/vadd
```

![]({{ '/pages/img/rocprofv3-perfetto-trace.png' | relative_url }})

There is also a `--stats` mode which aggregates the kernel trace into a `top_kernels` view in the sqlite output:

```bash
rocprofv3 --kernel-trace --stats -d /tmp/prof -o jaxcnn -- python3 cnn.py
```

A simple SQL query script:
```python
import sqlite3, sys

for d, c, p, n in sqlite3.connect(sys.argv[1]).execute(
    "SELECT total_duration,total_calls,percentage,name "
    "FROM top_kernels ORDER BY total_duration DESC LIMIT 10"
):
    print(f"{d:10.0f} {c:7d} {p:6.1f}  {n[:60]}")
```

![]({{ '/pages/img/rocprofv3-kernel-stats.png' | relative_url }})

See [AMD's rocprofv3 documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) for further reading

---

### Hardware Counter Analysis with rocprof-compute

Hardware counters give the most accurate and detailed breakdown of a specific kernels execution. `rocprof-compute` collects them and derives micro-architectural metrics: cache hit rates, occupancy, an empirical roofline, and speed-of-light against peak.

Three subcommands:

```bash
rocprof-compute profile -n vadd -- /tmp/vadd          # run the workload, collect base counters
rocprof-compute analyze -p workloads/vadd/MI300X_A1   # derive metrics into CLI tables
```

INSERT DIAGRAM EG. KERNEL -(PROFILE)-> GENERATE BASE COUNTER DATA -(ANALYZE)-> FEW DATA TO MANY DERIVED METRICS

- `profile` replays the workload many times to collect every counter set, and writes a workload directory: `pmc_perf.csv` holds the merged counters that `analyze` reads, alongside `sysinfo.csv` and the raw per-pass output.
- `analyze` derives the metrics. Add `-b` to select only the sections you want, and `--gui` for a dash app on `localhost:8050` instead of text tables.

CLI table:
![]({{ '/pages/img/rocprof-compute-cli.png' | relative_url }})

GUI:
![]({{ '/pages/img/rocprof-compute-speed-of-light.png' | relative_url }})
![]({{ '/pages/img/rocprof-compute-workflow.png' | relative_url }})
![]({{ '/pages/img/rocprof-compute-roofline.png' | relative_url }})

Notice the roofline model here is more detailed than the XProf equivalent. 

See [AMD's rocprof-compute documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/use.html) for further reading.

---

## Debugging with rocgdb

`rocgdb` is ROCm's fork of gdb. It debugs host x86 and additionally understands AMDGPU device code, exposing wavefronts as threads and adding `info agents` / `info queues` / `info dispatches`.

Take note of two different workflows you may run into:

- If you are debugging issues within XLA, e.g. debugging the effect of an XLA flag, or the lowering of an op, you will be working with a host-side regular python traceback.
    - The binary needs debug symbols, ie. a debug build rather than the release wheel, and the system needs to be dumping core files if you want to work from a crash after the fact rather than under the debugger.
    
- If you are debugging device-side (GPU) execution behaviour, you will see the device-side thread stack in gdb.
    - For device-side debugging, breakpoints inside a kernel need that kernel compiled with `-ggdb`, after which `info threads` lists wavefronts and you can step through GCN. That is realistic for a HIP kernel you wrote and much less so for an XLA-generated fusion, which is generated, fused and optimised before it ever reaches the device. Counter tools do the device-side work instead.

<need images and examples of each, either create faulty gpu execution intentionally and debug>

You can also debug with the logs:

```bash
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_MAX_VLOG_LEVEL=2 python3 train.py
```

- `TF_CPP_MIN_LOG_LEVEL` is the lowest severity to print, `0` for everything down to INFO, `2` for ERROR and FATAL only
- `TF_CPP_MAX_VLOG_LEVEL` is verbosity of the debug stream, `0` none, `2` a great deal, including per-pass and per-module timing

![]({{ '/pages/img/xla-vlog-unimplemented.png' | relative_url }})

See [AMD's ROCgdb documentation](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/how-to/quick-start.html) for further reading.

---

## End-to-End Performance

START:

ultimately the end goal with profiling is to identify where improvement opportunities are in relation to the objective we are optimizing for. here we focus on LLMs.
often the focus is tokens/sec (a measure of end to end latency), or max memory usage (so that a model can actually train on a system).

- tokens/sec as the headline, defined per-GPU and per-node so that it scales, plus samples/sec for non-LLM work
- MFU, ie. measured throughput against the datatype's roofline from the first chapter, and why it is the honest version of "utilisation"
- peak memory against the 288GB budget: parameters, optimiser state, activations, and where the buffer-assignment plan and the memory viewer disagree with the napkin figure
- achieved HBM bandwidth against 8TB/s, per kernel and aggregated over a step
- collective time as a fraction of step time, and how much of it is overlapped rather than exposed, which is where the scale-up/scale-out ladder from the first chapter starts to bite
- compile time and autotuning cost, amortised over the length of the run
- convergence: loss against a reference curve, and why a faster step that changes numerics is not a win
- a reporting checklist, ie. hardware, ROCm and JAX versions, XLA flags, batch and sharding config, and which numbers were measured against which were computed
