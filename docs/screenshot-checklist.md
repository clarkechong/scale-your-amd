# Screenshot checklist

Working document, not book content. Twenty-four hand-captured shots, each naming the run
to load, the view to open, and what has to be in frame. Everything listed here already
exists in `runs/`; nothing needs re-running.

## Setting up

```bash
cd /root/work/scale-your-amd
xprof --logdir=runs --port=6006
```

Port 6006 forwards automatically under VS Code Remote and dev containers. The run picker
lists 280 entries because the RCCL sweep captures one trace per point; the **Run** strings
below are exact, so paste rather than scroll.

## Conventions

- Save as `assets/img/<slug>.png`, using the slug in each entry.
- **Capture at a window width of at least 1600 px.** Several of these are wide tables and
  the columns that matter are on the right.
- **Light theme.** The book's figures are read on white.
- Crop to the panel, not the browser. No URL bar, no OS chrome.
- Where an entry says *annotate*, add a single red box or arrow in post. Nothing else.

---

## Chapter 3: Your First Trace

### 1. `xprof-tool-picker`

- **Run:** `20260805-transformer-block-onegpu/trace/2048/2026_08_05_12_44_57`
- **View:** any; the target is the **Tools** dropdown, opened.
- **In frame:** the full dropdown list, so a reader can see the eleven tools this stack
  produces and match them against the section headings that follow.
- **Goes in:** *Your First Trace*, after the `xprof --logdir` command.

## Chapter 3: The XProf Tools

### 2. `trace-viewer-overview`

- **Run:** `20260805-transformer-block-onegpu/trace/2048/2026_08_05_12_44_57`
- **View:** Trace Viewer, zoomed to show **two or three complete steps**.
- **In frame:** the `XLA Ops` row and the rows beneath it, with the row labels on the left
  legible. The point of the shot is the contrast between the hardware timeline and the
  approximate one built from `jax.named_scope`.
- **Goes in:** *Trace Viewer*.

### 3. `trace-viewer-one-layer`

- **Run:** same as 2.
- **View:** Trace Viewer, zoomed to **one layer** of one step.
- **In frame:** the nesting `layer_0` > `mlp` / `attention` / `qkv_proj`, with at least
  `qkv_proj`, `attention` and `mlp` labelled. This is the shot the
  *From an HLO Op Back to a Python Line* walk refers back to.
- **Annotate:** box the `mlp` span.
- **Goes in:** *Trace Viewer*.

### 4. `trace-viewer-selection-detail`

- **Run:** same as 2.
- **View:** Trace Viewer with **one GEMM kernel selected**, so the detail pane is open.
- **In frame:** the detail pane showing duration and the op name. Pick a kernel inside
  `mlp`.
- **Goes in:** *Trace Viewer*, alongside the w/a/s/d navigation note.

### 5. `graph-viewer-fusion`

- **Run:** `20260805-transformer-block-onegpu/trace/2048/2026_08_05_12_44_57`
- **View:** Graph Viewer, op name `input_reduce_fusion` or any `*_fusion` node.
- **In frame:** the fusion node with its operands, and the **op name search box** at the
  top with the search term still in it.
- **Goes in:** *Graph Viewer*.

### 6. `op-profile-tree`

- **Run:** `20260805-transformer-block-fsdp-sweep/trace/2048/2026_08_05_12_37_25`
- **View:** Op Profile, tree expanded one level under the root.
- **In frame:** the **IDLE row** and the top three or four named entries, with their
  percentages readable. IDLE reads **10.9%** on this run.
- **Goes in:** *Op Profile*.

### 7. `op-profile-idle-contrast`

- **Run:** `isolated/20260805-matmul-n4096-at0/trace/n4096/2026_08_05_12_01_56`
- **View:** Op Profile, same level as 6.
- **In frame:** the IDLE row, which reads **32.1%** here against 10.9% in shot 6 for a
  busier workload. The two shots together make the section's argument, which is that IDLE
  describes the capture window rather than the code.
- **Goes in:** *Op Profile*, immediately after 6.

### 8. `kernel-stats-working-columns`

- **Run:** `isolated/20260805-matmul-n4096-at0/trace/n4096/2026_08_05_12_01_56`
- **View:** Kernel Stats.
- **In frame:** the single `Cijk_Ailk_Bljk_BBS_BH_UserArgs_MT256x224x64_...` row with
  **Occurrences 10**, and the duration columns. Widen the Kernel Name column enough that
  `MT256x224x64` is visible; that token is referenced in the prose.
- **Goes in:** *Kernel Stats*, and reused by *The Matmul, Revisited*.

### 9. `kernel-stats-zero-columns`

- **Run:** same as 8.
- **View:** Kernel Stats, scrolled left so the leading columns show.
- **In frame:** **Registers per thread, Shared Mem bytes, Theoretical Occupancy %, Op is
  TensorCore eligible and Kernel uses TensorCore, all reading 0 or false.**
- **Annotate:** box those five columns.
- **Goes in:** *Kernel Stats*, and referenced from the limitations table.

### 10. `memory-profile-peak`

- **Run:** `20260805-transformer-block-onegpu/trace/2048/2026_08_05_12_44_57`
- **View:** Memory Profile.
- **In frame:** the allocation-over-time plot **and** the largest-buffers table beneath it,
  with at least the top three buffers' shapes readable.
- **Goes in:** *Memory Profile*.

### 11. `roofline-zero-ceiling`

- **Run:** `isolated/20260805-matmul-n4096-at0/trace/n4096/2026_08_05_12_01_56`
- **View:** Roofline Model.
- **In frame:** the **0 GFLOP/s peak-compute ceiling** and the "per TensorCore" axis
  labels. This is the evidence for the limitations-table row, so the zero has to be
  legible.
- **Goes in:** *Roofline*, and the limitations table links to it.

## Chapter 3: What Works Today

### 12. `overview-no-step-time-annotated`

- **Run:** `20260805-transformer-block-fsdp-sweep/trace/2048/2026_08_05_12_37_25`
- **View:** Overview Page.
- **In frame:** the sentence **"No step time measured. Therefore we cannot tell where the
  performance bottleneck is."**
- **Why this run specifically:** it is the arm **with** `StepTraceAnnotation`. The shot only
  makes its point if the caption can say the annotation was present, so keep the run name
  in the caption.
- **Goes in:** *The Step Markers, And Why We Are Not Going To Fix The Overview Page*.

### 13. `overview-device-compute-precisions`

- **Run:** same as 12.
- **View:** Overview Page, the summary block.
- **In frame:** `device_compute_16bit_percent` and `device_compute_32bit_percent` both
  reading **0.0%** on a run that is entirely bf16.
- **Goes in:** the limitations table row for Device Compute Precisions.

## Chapter 3: From an HLO Op Back to a Python Line

### 14. `hlo-walk-step1-kernel-stats`

- **Run:** `20260805-transformer-block-onegpu/trace/2048/2026_08_05_12_44_57`
- **View:** Kernel Stats, sorted by Total Duration.
- **In frame:** a `Cijk_...` row **with its Op Name column visible**, reading something of
  the form `jit(train_step)/layer_0/mlp/dot_general`. The Op Name column is the whole point
  of the shot; if it is cropped the figure is useless.
- **Annotate:** box the Op Name cell.
- **Goes in:** *Step 1: The Kernel Stats Row*.

### 15. `hlo-walk-step2-graph-viewer`

- **Run:** same as 14.
- **View:** Graph Viewer, searched for the op name from shot 14.
- **In frame:** the `custom-call` node, with `__cublas$lt$matmul` visible in the node label
  or the detail pane. The CUDA-flavoured name on an AMD GPU is the thing the prose calls
  out.
- **Goes in:** *Step 2: The HLO Op*.

## Chapter 3: A Training Step

### 16. `training-step-three-phases`

- **Run:** `20260805-transformer-block-onegpu/trace/2048/2026_08_05_12_44_57`
- **View:** Trace Viewer, zoomed to exactly **one step**, boundary to boundary.
- **In frame:** the whole step, with the forward region, the longer backward region and the
  short optimizer tail all visible.
- **Annotate:** three brackets above the timeline labelled forward, backward, optimizer.
  This is the chapter's anchor figure and it is worth the extra care.
- **Goes in:** *What Is In A Step*.

### 17. `training-step-eight-devices`

- **Run:** `20260805-transformer-block-fsdp-sweep/trace/2048/2026_08_05_12_37_25`
- **View:** Trace Viewer, all eight device rows collapsed to one row each.
- **In frame:** eight `/device:GPU:N` rows stacked, so the reader can see the same step
  running eight times over. Pairs with the "summed op time" row of the limitations table.
- **Goes in:** *One GPU Versus Eight*.

## Chapter 4: RCCL in Practice

### 18. `rccl-kernel-in-trace`

- **Run:** `20260805-rccl-sweep-full/trace/all_reduce/n8/1073741824/2026_08_05_12_21_20`
- **View:** Trace Viewer, zoomed to two or three of the ten iterations.
- **In frame:** the `ncclDevKernel_Generic_2` spans on all eight device rows, showing the
  staggered starts and the common finish. That stagger is the arrival skew the
  bandwidth section explains.
- **Goes in:** *RCCL in Practice*.

### 19. `rccl-kernel-detail`

- **Run:** same as 18.
- **View:** Trace Viewer with one `ncclDevKernel_Generic_2` span selected.
- **In frame:** the detail pane with the duration, so the reader can read a per-device
  collective time off the tool rather than taking ours.
- **Goes in:** *RCCL in Practice*.

## Chapter 4: Is the Collective Overlapping?

**These three are a set and must be captured at the same zoom and the same window width.**
The comparison is the figure; three shots at three scales prove nothing.

### 20. `overlap-combined-not-overlapped`

- **Run:** `20260805-transformer-block-combine-8589934592/trace/2048/2026_08_05_12_52_23`
- **View:** Trace Viewer, one step.
- **In frame:** the **single large all-gather** at the top of the step with **no compute
  running underneath it**. Measured overlap here is 1.1%.
- **Goes in:** *The Measurement, And Why It Is A Two-By-Two*, first of the three.

### 21. `overlap-split-overlapped`

- **Run:** `20260805-transformer-block-combine-1048576/trace/2048/2026_08_05_12_51_13`
- **View:** Trace Viewer, one step, **same zoom as 20**.
- **In frame:** many small collectives with compute kernels running concurrently above
  them. Measured overlap 36.6%.
- **Goes in:** second of the three.

### 22. `overlap-split-serialised`

- **Run:** `20260805-transformer-block-combine-small-lhs-off/trace/2048/2026_08_05_12_55_00`
- **View:** Trace Viewer, one step, **same zoom as 20 and 21**.
- **In frame:** the same many small collectives, now with compute **gaps** exactly as wide
  as each collective. Measured overlap 0.1%. This is the shot that shows the latency-hiding
  scheduler doing its job by its absence.
- **Goes in:** third of the three.

## Chapter 4: Who Inserts the Collective

### 23. `gspmd-vs-shardmap-graph`

- **Run:** load `runs/20260805-gspmd-vs-shardmap/hlo/gspmd.txt` and `shardmap.txt` in an
  editor rather than XProf.
- **In frame:** the two `all-reduce-start` lines side by side, so the only visible
  difference is `replica_groups` notation and the reduction function's name.
- **Note:** a side-by-side editor screenshot is fine and probably clearer than Graph
  Viewer here. `runs/20260805-gspmd-vs-shardmap/hlo/diff.txt` has the full diff.
- **Goes in:** *They Compile To The Same Thing*.

## Chapter 8: Occupancy

### 24. `kernel-stats-small-matmul-occupancy`

- **Run:** `isolated/20260805-matmul-n1024-at0/trace/n1024/2026_08_05_11_55_14`
- **View:** Kernel Stats.
- **In frame:** the `MT128x128x64` kernel with **Grid dim 64,1,1**, next to shot 8's
  **Grid dim 304,1,1** for the 4096 case. Sixty-four workgroups on 304 compute units is the
  occupancy story in one number, and it is the one column Kernel Stats gets right.
- **Annotate:** box the Grid dim cell.
- **Goes in:** Chapter 8's occupancy section, and referenced from *The Size Sweep*.

---

## Not screenshots

Three figures are plots and are generated rather than captured, by
`python tools/make_figures.py`:

- `rccl-bandwidth-curve` — busbw against message size, four collectives, 8 devices.
- `rccl-links-lit` — per-link rate against device count, showing links used tracks peers.
- `matmul-size-sweep` — achieved TFLOP/s against n, with the boost-clock and
  sustained-clock rooflines drawn in.

## Still owed, and not capturable from these runs

- The two XLA pipeline diagrams and the profiler-backend diagram in Chapter 3. These were
  sourced from `fw101` and `gpu_profiling`, neither of which is on this filesystem. They
  need relocating or redrawing; no capture will produce them.
