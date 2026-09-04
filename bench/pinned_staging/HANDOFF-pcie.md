# Handoff: measure pageable, pinned and staged host-to-device copies on a PCIe machine

This is a self-contained brief for reproducing the `bench/pinned_staging`
experiment on a different architecture, specifically an x86 host with a GPU on
PCIe. Everything needed is in this directory; the Sirius repository, pixi, cudf
and cucascade are not required.

## 1. The question to answer

Sirius pins most of host RAM at startup so that host-to-device copies run from
pinned memory. On a Grace Blackwell (GB300, NVLink-C2C) box we measured three
alternatives and found that on that platform:

- a **pinned** source and a **pageable** source copied directly with
  `cudaMemcpyAsync` are equally fast (about 360 GB/s at 32 MiB or larger
  buffers), so the pinned pool buys nothing there;
- **staging** pageable data through a small preallocated pinned ring
  (pageable → CPU memcpy → pinned slot → DMA → device, double buffered) caps at
  about 100 GB/s because the CPU memcpy competes with the DMA for host memory
  bandwidth.

On PCIe the balance is expected to be different: the link is 25–64 GB/s rather
than 370, one or two CPU cores can fill it, and pageable copies are usually much
slower than pinned because the driver stages them through its own small pinned
buffers. The hypotheses to test on PCIe are:

1. Does a pageable source reach the pinned ceiling with direct `cudaMemcpyAsync`?
   (Expected: no.)
2. Does the staged double-buffered path reach the pinned ceiling, with how many
   threads, what buffer size, and what pinned footprint? (Expected: yes, with
   2–4 threads and a ring of a few hundred MiB.)
3. How much does pinning cost per GiB at startup on that machine, and how does
   NUMA placement (GPU-local node vs remote node) change all of the above?

If 2 holds, Sirius could replace the startup pinning of most of RAM with a small
preallocated ring on PCIe machines.

## 2. What you receive

The directory `bench/pinned_staging/`:

| File | Purpose |
| --- | --- |
| `pinned_staging_bench.cu` | the benchmark (modes `pinned`, `pageable`, `memcpy`, `staged`, `alloc`; `--sched ring\|pool`) |
| `CMakeLists.txt` | standalone CMake project, needs only the CUDA toolkit |
| `run_grid.sh` | buffer-size × thread-count grid for the three strategies (what the chart plots) |
| `run_sweep.sh` | the wider sweep (alloc cost, memcpy-only, slots, pool scheduler, huge pages) |
| `profile.sh`, `analyze_nsys.py` | low-overhead nsys captures and a sqlite summarizer |
| `plot_results.py` | merges result CSVs into the SVG chart and markdown tables (no dependencies) |
| `README.md` | the full GB300 write-up and result tables |
| `results/` | the GB300 CSVs, chart and nsys summaries, for comparison |

Get it either from the Sirius repo, branch `claude/sirius-pinned-memory-buffering-785a25`
(commit `ae7d3ed9` or later), or as a tarball made with:

```bash
git archive --format=tar.gz -o pinned_staging.tar.gz HEAD bench/pinned_staging
```

## 3. Prerequisites on the target machine

- Linux, x86-64 (or any), with an NVIDIA GPU and a driver matching the toolkit.
- CUDA toolkit 12.x or 13.x with `nvcc`, `cuda_profiler_api.h` and
  `nvtx3/nvToolsExt.h` (both ship in the toolkit's `include/`).
- CMake 3.24 or newer, Ninja (or GNU make), a C++20 host compiler (GCC 11+).
- Python 3.8+ (plotting; standard library only).
- Optional: Nsight Systems (`nsys`) for the profiles, `numactl`, `flock`
  (util-linux, used by the scripts to serialize GPU runs).
- Memory: the benchmark allocates `--total-bytes` of pageable host memory, the
  same amount of device memory, and for `--mode pinned` the same amount of
  pinned host memory. 16 GiB is a good default on a PCIe box (at 25 GB/s one
  pass is 0.7 s); use 8 GiB if RAM or device memory is tight, and keep the
  value constant across every run you compare.

## 4. Record the machine first

Paste the output of these into the report:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current --format=csv
```

```bash
nvidia-smi topo -m
```

```bash
lscpu | grep -E "Model name|Socket|NUMA|Thread|Core"; numactl -H | head -20; free -g; getconf PAGESIZE; cat /sys/kernel/mm/transparent_hugepage/enabled; cat /proc/sys/kernel/perf_event_paranoid; uname -r; nvcc --version | tail -1
```

The PCIe generation and width give the theoretical ceiling: Gen4 x16 is
32 GB/s raw (about 25 GB/s achievable), Gen5 x16 is 64 GB/s raw (about 50 GB/s
achievable). `nvidia-smi topo -m` shows which CPU cores and NUMA node are local
to the GPU; that matters more on PCIe than anything else below.

## 5. Build and smoke test

From the directory that contains `bench/`:

```bash
cmake -S bench/pinned_staging -B build/bench/pinned_staging -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build/bench/pinned_staging
```

If CMake cannot find `nvtx3/nvToolsExt.h`, pass
`-DNVTX3_INCLUDE_DIR=/usr/local/cuda/include` (or wherever the toolkit lives).

Smoke test every mode at 2 GiB with verification (about 30 s total):

```bash
B=build/bench/pinned_staging/pinned_staging_bench; $B --csv-header --csv --mode pinned --total-bytes 2G --chunk-bytes 16M --threads 2 --iters 2 --verify && $B --csv --mode pageable --total-bytes 2G --chunk-bytes 16M --threads 2 --iters 2 --verify && $B --csv --mode staged --total-bytes 2G --chunk-bytes 16M --threads 2 --slots 2 --iters 2 --verify && $B --csv --mode staged --sched pool --slots 16 --streams 4 --total-bytes 2G --chunk-bytes 16M --threads 2 --iters 2 --verify && $B --csv --mode memcpy --total-bytes 2G --chunk-bytes 16M --threads 2 --iters 1 && $B --csv --mode alloc --total-bytes 2G
```

Every `--verify` line must end in `ok`. The human-readable progress goes to
stderr; the CSV line goes to stdout.

## 6. NUMA placement

On a multi-socket x86 box run everything bound to the GPU's NUMA node, then
repeat the headline configurations bound to a remote node. Find the node:

```bash
cat /sys/bus/pci/devices/$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | sed 's/^0000//' | tr 'A-F' 'a-f')/numa_node
```

Then run every benchmark under `numactl --cpunodebind=N --membind=N`. For the
scripts, set `RUN_PREFIX="numactl --cpunodebind=N --membind=N"`; each benchmark
invocation is launched through that prefix. For ad-hoc runs put the prefix in
front of the binary. Report both placements. `--pin-cpus` pins worker `t` to
CPU `t`, which on x86 with SMT may land on sibling hyperthreads; prefer
`numactl` over `--pin-cpus`.

## 7. The experiment matrix

The chart's grid, all three strategies, buffer sizes 1–64 MiB, threads 1–16
(the GB300 needed up to 64 threads only because its link is 370 GB/s; on PCIe
the staged path should saturate the link long before 16). About 25–40 minutes
at 16 GiB:

```bash
TOTAL=16G THREADS="1 2 4 8 16" RUN_PREFIX="numactl --cpunodebind=0 --membind=0" bench/pinned_staging/run_grid.sh
```

The wider sweep (startup cost of pinning, memcpy-only ceiling, 1/2/4 slots, the
shared-pool scheduler, huge pages, pinning). About 30–60 minutes at 16 GiB; the
`ALLOC_SIZES` probe pins that much memory briefly, so keep it below free RAM:

```bash
TOTAL=16G ALLOC_SIZES="8G 32G 64G" bench/pinned_staging/run_sweep.sh
```

Both scripts write one CSV under `build/bench/pinned_staging/` and print its
path; each benchmark run is wrapped in `flock` on
`/tmp/sirius-pinned-bench.gpu.lock` so two sessions on the same box do not
corrupt each other. If the box is shared, check `nvidia-smi` for other users'
processes before starting and do not run while they are measuring.

Ad-hoc single runs, for example to find the smallest ring that still matches
pinned:

```bash
B=build/bench/pinned_staging/pinned_staging_bench; for s in 1 2 4; do for t in 1 2 4; do $B --csv --mode staged --total-bytes 16G --chunk-bytes 16M --threads $t --slots $s; done; done
```

The pinned footprint of a staged run is `threads × slots × chunk` (ring) or
`slots × chunk` (pool); the CSV `pinned_bytes` column records it.

## 8. Profiles (optional but valuable)

```bash
TOTAL=16G bench/pinned_staging/profile.sh
```

This captures pinned, pageable and staged runs with `--trace=cuda,nvtx`, no CPU
sampling, only the measured passes, and prints an overhead table (plain vs
profiled GB/s). Reports land in `build/bench/pinned_staging/nsys/`: open the
`.nsys-rep` in the GUI, or read the `*.analysis.txt` summaries, which give per
pass the DMA busy fraction, the peak number of concurrent copies, per-copy
bandwidth, and how worker time splits between memcpy, waiting and issue. On the
GB300 the interesting facts were "one copy in flight at a time" and "per-copy
DMA bandwidth falls while CPU threads memcpy"; check whether either holds on
PCIe. CPU sampling needs `kernel.perf_event_paranoid` ≤ 2.

## 9. Chart and tables

```bash
python3 bench/pinned_staging/plot_results.py build/bench/pinned_staging/grid-*.csv --threads 1,2,4,8,16 --cols 3 -o pcie-throughput-vs-buffer.svg
```

```bash
python3 bench/pinned_staging/plot_results.py build/bench/pinned_staging/grid-*.csv --threads 1,2,4,8,16 --table > pcie-tables.md
```

Change the subtitle with `--subtitle "…"` to name the machine and link.

## 10. What to send back

1. The machine record from section 4.
2. The CSV files from sections 7 and 8 (`grid-*.csv`, `sweep-*.csv`,
   `nsys/overhead.csv`, `nsys/*.analysis.txt`). The CSV columns are:
   `mode, pinned_alloc, total_bytes, chunk_bytes, threads, slots, pinned_bytes,
   pinned_alloc_ms, pinned_free_ms, median_gbps, best_gbps, memcpy_busy_frac,
   wait_frac, issue_frac, verify, sched, streams, huge, pin_cpus`
   (GB/s are decimal, bytes/1e9/s, median of `--iters` passes after one warmup).
3. The chart and tables from section 9, for the GPU-local and the remote NUMA
   placement.
4. Answers to the three questions in section 1, in this form:

| Question | Metric | Value |
| --- | --- | --- |
| pinned ceiling | best pinned GB/s, buffer size and threads that reach it | |
| pageable vs pinned | pageable/pinned ratio at the ceiling buffer size, 1 and 8 threads | |
| staged vs pinned | smallest threads × slots × chunk that reaches ≥ 95% of pinned, and its GB/s | |
| staged bottleneck | `memcpy_busy_frac` and `wait_frac` at that configuration | |
| pinning cost | `pinned_alloc_ms` per GiB for `cudaHostAlloc` and for `register` | |
| NUMA | same three numbers on the remote node | |

Decision rule: if staged reaches ≥ 95% of pinned with a ring under about 1 GiB,
the startup pool can be replaced by a preallocated ring on that architecture;
if pageable also reaches ≥ 95%, no ring is needed at all (as on Grace).

## 11. GB300 reference numbers (32 GiB per pass)

| Strategy, 64 MiB buffers | 1 thread | 8 threads | 32 threads |
| --- | --- | --- | --- |
| pinned | 359 | 357 | 368 |
| pageable, direct | 338 | 347 | 347 |
| staged, 2 slots per thread | 13.7 | 77.9 | 98.3 |

Buffer-size sensitivity of the direct copies on GB300: 1 MiB 170, 4 MiB 290,
8 MiB 326, 32 MiB 362 GB/s (pinned; pageable within 5% from 8 threads up).
Pinning cost: 40–45 ms per GiB. Staged is flat across buffer sizes and bound by
CPU memcpy at about 14 GB/s per core with the aggregate saturating near 135 GB/s;
nsys showed one host-to-device copy in flight at a time and per-copy DMA
bandwidth dropping from 372 to 187 GB/s while 32 threads memcpy. The full
write-up is `README.md`.

## 12. Pitfalls seen so far

- Compare only runs with the same `--total-bytes`; run-to-run drift is about
  3–5%, so repeat the headline configurations and interleave baseline and
  candidate rather than trusting a single pass.
- `cudaMemcpyAsync` from pageable memory blocks the calling thread for the whole
  transfer; a 1-thread pageable number is a property of that blocking, not of
  the link.
- Small buffers (1 MiB) are issue-bound on both pinned and pageable sources;
  do not judge the strategies at 1 MiB only.
- The `alloc` probe of 64 GiB pins that much memory for a few seconds; make sure
  nothing else on the box needs it.
- Older result CSVs in `results/` have fewer columns; `plot_results.py` handles
  both, but do not mix machines in one chart.
