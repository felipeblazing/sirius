/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// pinned_staging_bench — host->device copy throughput microbenchmark.
//
// Hypothesis under test: instead of pinning most of host RAM at startup (what
// Sirius does today through cucascade's numa_region_pinned_host_memory_resource),
// keep the data in *pageable* host memory and stream it to the GPU through a
// small ring of pinned staging slots:
//
//   pageable src --(CPU memcpy, worker thread)--> pinned slot
//   pinned slot  --(cudaMemcpyAsync, DMA)-------> device dst
//
// Each worker owns a private stream and a private ring of R slots. While the
// DMA engine drains slot k, the worker is already filling slot k+1, so with
// R >= 2 the two stages overlap (classic double buffering). The worker waits on
// a slot's event only when it comes back around to that slot.
//
// Modes:
//   pinned    source already pinned; chunked cudaMemcpyAsync (the ceiling)
//   pageable  source pageable; cudaMemcpyAsync straight from it (driver staging)
//   memcpy    CPU-only stage A: pageable -> pinned ring, no GPU copies
//   staged    the hypothesis: pageable -> pinned ring -> device
//   alloc     time cudaHostAlloc / cudaHostRegister + free for --total-bytes
//
// Throughput is reported in decimal GB/s (bytes / 1e9 / seconds) over the whole
// dataset, measured from worker launch to the last stream synchronize.

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cuda_profiler_api.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                 \
  do {                                                   \
    cudaError_t err_ = (call);                           \
    if (err_ != cudaSuccess) {                           \
      std::fprintf(stderr,                               \
                   "CUDA error: %s\n  at %s:%d\n  %s\n", \
                   cudaGetErrorString(err_),             \
                   __FILE__,                             \
                   __LINE__,                             \
                   #call);                               \
      std::exit(2);                                      \
    }                                                    \
  } while (0)

using bench_clock = std::chrono::steady_clock;

double seconds_since(bench_clock::time_point t0)
{
  return std::chrono::duration<double>(bench_clock::now() - t0).count();
}

// NVTX push/pop range, enabled by --nvtx so nsys can attribute worker time to
// memcpy / wait / issue. Without an injection library (no profiler attached)
// the NVTX calls are cheap no-ops; without --nvtx they are not made at all.
struct nvtx_scope {
  bool on;
  nvtx_scope(bool enabled, const char* name) : on(enabled)
  {
    if (on) nvtxRangePushA(name);
  }
  ~nvtx_scope()
  {
    if (on) nvtxRangePop();
  }
  nvtx_scope(const nvtx_scope&)            = delete;
  nvtx_scope& operator=(const nvtx_scope&) = delete;
};

enum class bench_mode { alloc, pinned, pageable, memcpy_only, staged };
enum class pinned_alloc_kind { host_alloc, host_register };
// How staged mode hands pinned slots to the DMA:
//   ring  each worker owns --slots slots and one stream (double buffering)
//   pool  --slots shared slots, --streams shared streams, a reaper thread
//         recycles slots as their events complete; up to --slots copies queued
enum class sched_kind { ring, pool };

const char* to_string(sched_kind s) { return s == sched_kind::ring ? "ring" : "pool"; }

const char* to_string(bench_mode m)
{
  switch (m) {
    case bench_mode::alloc: return "alloc";
    case bench_mode::pinned: return "pinned";
    case bench_mode::pageable: return "pageable";
    case bench_mode::memcpy_only: return "memcpy";
    case bench_mode::staged: return "staged";
  }
  return "?";
}

const char* to_string(pinned_alloc_kind k)
{
  return k == pinned_alloc_kind::host_alloc ? "hostalloc" : "register";
}

struct options {
  bench_mode mode{bench_mode::staged};
  pinned_alloc_kind pinned_alloc{pinned_alloc_kind::host_alloc};
  std::size_t total_bytes{32ULL << 30};
  std::size_t chunk_bytes{64ULL << 20};
  int threads{8};
  int slots{2};
  sched_kind sched{sched_kind::ring};
  int streams{0};  // pool scheduling only; 0 = same as threads
  int iters{3};
  int device{0};
  bool verify{false};
  bool huge{false};
  bool pin_cpus{false};
  bool nvtx{false};
  bool csv{false};
  bool csv_header{false};
};

void usage(const char* argv0)
{
  std::fprintf(
    stderr,
    "usage: %s [options]\n"
    "  --mode {alloc|pinned|pageable|memcpy|staged}   (default staged)\n"
    "  --pinned-alloc {hostalloc|register}   how pinned memory is obtained (default hostalloc)\n"
    "  --total-bytes N[K|M|G|T]              dataset size, rounded down to whole chunks (32G)\n"
    "  --chunk-bytes N[K|M|G]                bytes per chunk / staging slot (64M)\n"
    "  --threads T                           worker threads, one stream each (8)\n"
    "  --slots R                             ring: pinned slots per worker, 1 = no overlap;\n"
    "                                        pool: total shared slots (2)\n"
    "  --sched {ring|pool}                   staged slot scheduling (ring)\n"
    "  --streams K                           pool: shared DMA streams (threads)\n"
    "  --iters N                             measured passes after one warmup (3)\n"
    "  --device D                            CUDA device (0)\n"
    "  --verify                              check device contents after the last pass\n"
    "  --huge                                madvise(MADV_HUGEPAGE) the pageable source\n"
    "  --pin-cpus                            pin worker t to cpu t\n"
    "  --nvtx                                emit NVTX ranges (pass, memcpy, wait, issue)\n"
    "  --csv                                 print one CSV result line\n"
    "  --csv-header                          print the CSV header line first\n",
    argv0);
}

std::size_t parse_bytes(const std::string& text)
{
  char* end    = nullptr;
  double value = std::strtod(text.c_str(), &end);
  std::string suffix(end);
  for (auto& ch : suffix)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (suffix.size() >= 2 && suffix.compare(suffix.size() - 2, 2, "ib") == 0) {
    suffix.resize(suffix.size() - 2);
  } else if (!suffix.empty() && suffix.back() == 'b') {
    suffix.pop_back();
  }
  double mult = 1.0;
  if (suffix == "k") {
    mult = 1ULL << 10;
  } else if (suffix == "m") {
    mult = 1ULL << 20;
  } else if (suffix == "g") {
    mult = 1ULL << 30;
  } else if (suffix == "t") {
    mult = 1ULL << 40;
  } else if (!suffix.empty()) {
    std::fprintf(stderr, "bad byte size: %s\n", text.c_str());
    std::exit(1);
  }
  return static_cast<std::size_t>(value * mult);
}

options parse_args(int argc, char** argv)
{
  options opt;
  auto need = [&](int& i) -> std::string {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", argv[i]);
      usage(argv[0]);
      std::exit(1);
    }
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mode") {
      std::string v = need(i);
      if (v == "alloc") {
        opt.mode = bench_mode::alloc;
      } else if (v == "pinned") {
        opt.mode = bench_mode::pinned;
      } else if (v == "pageable") {
        opt.mode = bench_mode::pageable;
      } else if (v == "memcpy") {
        opt.mode = bench_mode::memcpy_only;
      } else if (v == "staged") {
        opt.mode = bench_mode::staged;
      } else {
        std::fprintf(stderr, "unknown mode %s\n", v.c_str());
        std::exit(1);
      }
    } else if (a == "--pinned-alloc") {
      std::string v = need(i);
      if (v == "hostalloc") {
        opt.pinned_alloc = pinned_alloc_kind::host_alloc;
      } else if (v == "register") {
        opt.pinned_alloc = pinned_alloc_kind::host_register;
      } else {
        std::fprintf(stderr, "unknown --pinned-alloc %s\n", v.c_str());
        std::exit(1);
      }
    } else if (a == "--total-bytes") {
      opt.total_bytes = parse_bytes(need(i));
    } else if (a == "--chunk-bytes") {
      opt.chunk_bytes = parse_bytes(need(i));
    } else if (a == "--threads") {
      opt.threads = std::stoi(need(i));
    } else if (a == "--slots") {
      opt.slots = std::stoi(need(i));
    } else if (a == "--sched") {
      std::string v = need(i);
      if (v == "ring") {
        opt.sched = sched_kind::ring;
      } else if (v == "pool") {
        opt.sched = sched_kind::pool;
      } else {
        std::fprintf(stderr, "unknown --sched %s\n", v.c_str());
        std::exit(1);
      }
    } else if (a == "--streams") {
      opt.streams = std::stoi(need(i));
    } else if (a == "--iters") {
      opt.iters = std::stoi(need(i));
    } else if (a == "--device") {
      opt.device = std::stoi(need(i));
    } else if (a == "--verify") {
      opt.verify = true;
    } else if (a == "--huge") {
      opt.huge = true;
    } else if (a == "--pin-cpus") {
      opt.pin_cpus = true;
    } else if (a == "--nvtx") {
      opt.nvtx = true;
    } else if (a == "--csv") {
      opt.csv = true;
    } else if (a == "--csv-header") {
      opt.csv_header = true;
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage(argv[0]);
      std::exit(1);
    }
  }
  if (opt.threads < 1 || opt.slots < 1 || opt.iters < 1 || opt.chunk_bytes < 8 ||
      opt.chunk_bytes % 8 != 0) {
    std::fprintf(stderr, "invalid arguments (threads/slots/iters >= 1, chunk multiple of 8)\n");
    std::exit(1);
  }
  return opt;
}

// ---------------------------------------------------------------------------
// Data pattern: a cheap per-word function of the word index, so the device
// side can verify the whole destination without a reference copy.
// ---------------------------------------------------------------------------

__host__ __device__ inline std::uint64_t pattern_word(std::uint64_t i)
{
  return (i * 0x9E3779B97F4A7C15ULL) ^ 0xD1B54A32D192ED03ULL;
}

// Parallel fill; also pre-faults every page of a fresh mapping.
void fill_pattern(std::uint64_t* words, std::size_t n)
{
  unsigned nt     = std::max(1u, std::thread::hardware_concurrency());
  std::size_t per = (n + nt - 1) / nt;
  std::vector<std::thread> ts;
  for (unsigned t = 0; t < nt; ++t) {
    std::size_t b = static_cast<std::size_t>(t) * per;
    std::size_t e = std::min(n, b + per);
    if (b >= e) break;
    ts.emplace_back([=] {
      for (std::size_t i = b; i < e; ++i)
        words[i] = pattern_word(i);
    });
  }
  for (auto& t : ts)
    t.join();
}

__global__ void verify_kernel(const std::uint64_t* __restrict__ words,
                              std::size_t n,
                              unsigned long long* mismatches)
{
  std::size_t stride       = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  unsigned long long local = 0;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    local += (words[i] != pattern_word(i)) ? 1ULL : 0ULL;
  }
  if (local != 0) atomicAdd(mismatches, local);
}

// ---------------------------------------------------------------------------
// Host memory helpers
// ---------------------------------------------------------------------------

void* map_pageable(std::size_t bytes, bool huge)
{
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    std::perror("mmap");
    std::exit(2);
  }
  if (huge && madvise(p, bytes, MADV_HUGEPAGE) != 0) { std::perror("madvise(MADV_HUGEPAGE)"); }
  return p;
}

struct pinned_block {
  void* ptr{nullptr};
  std::size_t bytes{0};
  pinned_alloc_kind kind{pinned_alloc_kind::host_alloc};
  double alloc_ms{0};
  double free_ms{0};
};

// Mirrors cucascade's numa_region_pinned_host_allocator: either
// cudaHostAlloc(Portable|Mapped) or a fresh (untouched) anonymous mapping
// followed by cudaHostRegister(Portable|Mapped). The register variant is timed
// on unpopulated pages on purpose: that is what registering a fresh
// numa_alloc_onnode() region costs at Sirius startup.
pinned_block allocate_pinned(std::size_t bytes, pinned_alloc_kind kind, bool huge)
{
  pinned_block b;
  b.bytes = bytes;
  b.kind  = kind;
  auto t0 = bench_clock::now();
  if (kind == pinned_alloc_kind::host_alloc) {
    CUDA_CHECK(cudaHostAlloc(&b.ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
  } else {
    b.ptr = map_pageable(bytes, huge);
    CUDA_CHECK(cudaHostRegister(b.ptr, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped));
  }
  b.alloc_ms = seconds_since(t0) * 1e3;
  return b;
}

void free_pinned(pinned_block& b)
{
  if (b.ptr == nullptr) return;
  auto t0 = bench_clock::now();
  if (b.kind == pinned_alloc_kind::host_alloc) {
    CUDA_CHECK(cudaFreeHost(b.ptr));
  } else {
    CUDA_CHECK(cudaHostUnregister(b.ptr));
    munmap(b.ptr, b.bytes);
  }
  b.free_ms = seconds_since(t0) * 1e3;
  b.ptr     = nullptr;
}

void pin_to_cpu(int cpu)
{
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu % CPU_SETSIZE, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

// ---------------------------------------------------------------------------
// Workers
// ---------------------------------------------------------------------------

struct slot {
  std::uint8_t* pinned{nullptr};
  cudaEvent_t event{nullptr};
  bool in_flight{false};
};

// Shared pool of preallocated pinned slots (--sched pool). Producers take any
// free slot, fill it, issue the DMA on a shared stream and hand the slot to the
// reaper thread, which polls completion events and returns slots to the free
// list. Producers block only when every slot is queued on the GPU, so the DMA
// queue depth is bounded by the pool size, not by the producer count.
struct slot_pool {
  std::vector<slot> slots;
  std::mutex m;
  std::condition_variable cv;
  std::vector<slot*> free_list;
  std::vector<slot*> in_flight;
  bool stop{false};
  std::size_t peak_in_flight{0};

  void reset()
  {
    std::lock_guard lk(m);
    free_list.clear();
    for (auto& s : slots)
      free_list.push_back(&s);
    in_flight.clear();
    stop           = false;
    peak_in_flight = 0;
  }

  slot* acquire()
  {
    std::unique_lock lk(m);
    cv.wait(lk, [&] { return !free_list.empty(); });
    slot* s = free_list.back();
    free_list.pop_back();
    return s;
  }

  void submit(slot* s)
  {
    std::lock_guard lk(m);
    in_flight.push_back(s);
    peak_in_flight = std::max(peak_in_flight, in_flight.size());
    cv.notify_all();
  }

  void shutdown()
  {
    std::lock_guard lk(m);
    stop = true;
    cv.notify_all();
  }

  // Reaper: poll every queued slot's event, recycle the completed ones.
  void reap_loop()
  {
    std::vector<slot*> pending;
    for (;;) {
      {
        std::unique_lock lk(m);
        cv.wait(lk, [&] { return stop || !in_flight.empty(); });
        if (in_flight.empty()) return;  // stop requested and nothing left
        pending = in_flight;
      }
      bool any = false;
      for (slot* s : pending) {
        cudaError_t e = cudaEventQuery(s->event);
        if (e == cudaSuccess) {
          std::lock_guard lk(m);
          in_flight.erase(std::find(in_flight.begin(), in_flight.end(), s));
          free_list.push_back(s);
          any = true;
          cv.notify_all();
        } else if (e != cudaErrorNotReady) {
          CUDA_CHECK(e);
        }
      }
      if (!any) std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
  }
};

struct worker {
  int tid{0};
  cudaStream_t stream{nullptr};
  std::vector<slot> ring;
  // Accumulated over one pass; used to tell CPU-bound from DMA-bound.
  double memcpy_s{0};
  double wait_s{0};
  double issue_s{0};
  std::size_t chunks{0};
};

struct copy_job {
  const options* opt{nullptr};
  std::uint8_t* src{nullptr};  // pageable or pinned, total_bytes
  std::uint8_t* dst{nullptr};  // device, total_bytes
  std::size_t nchunks{0};
  std::atomic<std::size_t>* next{nullptr};
  slot_pool* pool{nullptr};                     // --sched pool only
  std::vector<cudaStream_t>* streams{nullptr};  // --sched pool only
};

void run_worker(worker& w, const copy_job& job)
{
  const options& opt = *job.opt;
  CUDA_CHECK(cudaSetDevice(opt.device));
  if (opt.pin_cpus) pin_to_cpu(w.tid);
  w.memcpy_s = w.wait_s = w.issue_s = 0;
  w.chunks                          = 0;
  const std::size_t S               = opt.chunk_bytes;
  std::size_t k                     = 0;
  for (;;) {
    std::size_t c = job.next->fetch_add(1, std::memory_order_relaxed);
    if (c >= job.nchunks) break;
    std::size_t off = c * S;
    switch (opt.mode) {
      case bench_mode::pinned:
      case bench_mode::pageable: {
        nvtx_scope r(opt.nvtx, "issue");
        auto t0 = bench_clock::now();
        CUDA_CHECK(
          cudaMemcpyAsync(job.dst + off, job.src + off, S, cudaMemcpyHostToDevice, w.stream));
        w.issue_s += seconds_since(t0);
        break;
      }
      case bench_mode::memcpy_only: {
        slot& s = w.ring[k];
        k       = (k + 1) % w.ring.size();
        nvtx_scope r(opt.nvtx, "memcpy");
        auto t0 = bench_clock::now();
        std::memcpy(s.pinned, job.src + off, S);
        w.memcpy_s += seconds_since(t0);
        break;
      }
      case bench_mode::staged: {
        if (job.pool != nullptr) {
          // Shared pool: any free slot, DMA on a shared stream, reaper recycles.
          slot* s = nullptr;
          {
            nvtx_scope r(opt.nvtx, "wait");
            auto t0 = bench_clock::now();
            s       = job.pool->acquire();
            w.wait_s += seconds_since(t0);
          }
          {
            nvtx_scope r(opt.nvtx, "memcpy");
            auto t1 = bench_clock::now();
            std::memcpy(s->pinned, job.src + off, S);
            w.memcpy_s += seconds_since(t1);
          }
          nvtx_scope r(opt.nvtx, "issue");
          auto t2         = bench_clock::now();
          cudaStream_t st = (*job.streams)[c % job.streams->size()];
          CUDA_CHECK(cudaMemcpyAsync(job.dst + off, s->pinned, S, cudaMemcpyHostToDevice, st));
          CUDA_CHECK(cudaEventRecord(s->event, st));
          job.pool->submit(s);
          w.issue_s += seconds_since(t2);
          break;
        }
        // Private ring: wait only when coming back around to a queued slot.
        slot& s = w.ring[k];
        k       = (k + 1) % w.ring.size();
        if (s.in_flight) {
          nvtx_scope r(opt.nvtx, "wait");
          auto t0 = bench_clock::now();
          CUDA_CHECK(cudaEventSynchronize(s.event));
          w.wait_s += seconds_since(t0);
        }
        {
          nvtx_scope r(opt.nvtx, "memcpy");
          auto t1 = bench_clock::now();
          std::memcpy(s.pinned, job.src + off, S);
          w.memcpy_s += seconds_since(t1);
        }
        nvtx_scope r(opt.nvtx, "issue");
        auto t2 = bench_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(job.dst + off, s.pinned, S, cudaMemcpyHostToDevice, w.stream));
        CUDA_CHECK(cudaEventRecord(s.event, w.stream));
        s.in_flight = true;
        w.issue_s += seconds_since(t2);
        break;
      }
      case bench_mode::alloc: break;
    }
    ++w.chunks;
  }
  // Pool scheduling drains the shared streams in run_pass instead.
  if (opt.mode != bench_mode::memcpy_only && job.pool == nullptr) {
    CUDA_CHECK(cudaStreamSynchronize(w.stream));
  }
  for (auto& s : w.ring)
    s.in_flight = false;
}

double run_pass(std::vector<worker>& workers, copy_job& job)
{
  job.next->store(0);
  nvtx_scope r(job.opt->nvtx, "pass");
  auto t0 = bench_clock::now();
  std::thread reaper;
  if (job.pool != nullptr) {
    job.pool->reset();
    reaper = std::thread([&] {
      CUDA_CHECK(cudaSetDevice(job.opt->device));
      job.pool->reap_loop();
    });
  }
  std::vector<std::thread> ts;
  ts.reserve(workers.size());
  for (auto& w : workers)
    ts.emplace_back(run_worker, std::ref(w), std::cref(job));
  for (auto& t : ts)
    t.join();
  if (job.pool != nullptr) {
    for (auto st : *job.streams)
      CUDA_CHECK(cudaStreamSynchronize(st));
    job.pool->shutdown();
    reaper.join();
  }
  return seconds_since(t0);
}

std::string human_bytes(std::size_t b)
{
  char buf[64];
  if (b >= (1ULL << 30) && b % (1ULL << 30) == 0) {
    std::snprintf(buf, sizeof buf, "%zu GiB", b >> 30);
  } else if (b >= (1ULL << 30)) {
    std::snprintf(buf, sizeof buf, "%.2f GiB", static_cast<double>(b) / (1ULL << 30));
  } else if (b >= (1ULL << 20)) {
    std::snprintf(buf, sizeof buf, "%.0f MiB", static_cast<double>(b) / (1ULL << 20));
  } else {
    std::snprintf(buf, sizeof buf, "%zu B", b);
  }
  return buf;
}

void print_csv_header()
{
  std::printf(
    "mode,pinned_alloc,total_bytes,chunk_bytes,threads,slots,pinned_bytes,"
    "pinned_alloc_ms,pinned_free_ms,median_gbps,best_gbps,"
    "memcpy_busy_frac,wait_frac,issue_frac,verify,sched,streams,huge,pin_cpus\n");
}

}  // namespace

int main(int argc, char** argv)
{
  options opt = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(opt.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));

  if (opt.csv_header) print_csv_header();

  // ---- alloc mode: startup-cost probe only ---------------------------------
  if (opt.mode == bench_mode::alloc) {
    std::fprintf(stderr,
                 "[alloc] %s %s on %s\n",
                 to_string(opt.pinned_alloc),
                 human_bytes(opt.total_bytes).c_str(),
                 prop.name);
    pinned_block blk = allocate_pinned(opt.total_bytes, opt.pinned_alloc, opt.huge);
    free_pinned(blk);
    std::fprintf(stderr,
                 "[alloc] allocate %.1f ms (%.2f GB/s), free %.1f ms\n",
                 blk.alloc_ms,
                 static_cast<double>(opt.total_bytes) / 1e9 / (blk.alloc_ms / 1e3),
                 blk.free_ms);
    if (opt.csv) {
      std::printf("alloc,%s,%zu,0,0,0,%zu,%.3f,%.3f,,,,,,,,,,\n",
                  to_string(opt.pinned_alloc),
                  opt.total_bytes,
                  opt.total_bytes,
                  blk.alloc_ms,
                  blk.free_ms);
    }
    return 0;
  }

  // ---- copy modes ----------------------------------------------------------
  const std::size_t S       = opt.chunk_bytes;
  const std::size_t nchunks = opt.total_bytes / S;
  const std::size_t total   = nchunks * S;
  if (nchunks == 0) {
    std::fprintf(stderr, "total-bytes smaller than one chunk\n");
    return 1;
  }
  const bool uses_ring = opt.mode == bench_mode::staged || opt.mode == bench_mode::memcpy_only;
  const bool use_pool  = opt.mode == bench_mode::staged && opt.sched == sched_kind::pool;
  const int n_streams  = use_pool ? (opt.streams > 0 ? opt.streams : opt.threads) : 0;
  const std::size_t slot_count =
    !uses_ring ? 0
    : use_pool ? static_cast<std::size_t>(opt.slots)
               : static_cast<std::size_t>(opt.threads) * static_cast<std::size_t>(opt.slots);
  const std::size_t ring_bytes       = slot_count * S;
  const std::size_t pinned_footprint = opt.mode == bench_mode::pinned ? total : ring_bytes;

  std::fprintf(stderr,
               "[cfg] mode=%s device=%s total=%s chunk=%s chunks=%zu threads=%d slots=%d "
               "sched=%s streams=%d pinned_footprint=%s pinned_alloc=%s huge=%d pin_cpus=%d\n",
               to_string(opt.mode),
               prop.name,
               human_bytes(total).c_str(),
               human_bytes(S).c_str(),
               nchunks,
               opt.threads,
               opt.slots,
               to_string(opt.sched),
               n_streams,
               human_bytes(pinned_footprint).c_str(),
               to_string(opt.pinned_alloc),
               opt.huge ? 1 : 0,
               opt.pin_cpus ? 1 : 0);

  // Device destination (full size so the whole dataset can be verified).
  std::uint8_t* dst = nullptr;
  if (opt.mode != bench_mode::memcpy_only) {
    std::size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (free_b < total + (256ULL << 20)) {
      std::fprintf(stderr,
                   "not enough device memory: need %s, free %s\n",
                   human_bytes(total).c_str(),
                   human_bytes(free_b).c_str());
      return 1;
    }
    CUDA_CHECK(cudaMalloc(&dst, total));
    CUDA_CHECK(cudaMemset(dst, 0, total));
  }

  // Source.
  pinned_block src_pinned;
  std::uint8_t* src = nullptr;
  {
    auto t0 = bench_clock::now();
    if (opt.mode == bench_mode::pinned) {
      src_pinned = allocate_pinned(total, opt.pinned_alloc, opt.huge);
      src        = static_cast<std::uint8_t*>(src_pinned.ptr);
      std::fprintf(stderr,
                   "[src] pinned %s via %s: %.1f ms\n",
                   human_bytes(total).c_str(),
                   to_string(opt.pinned_alloc),
                   src_pinned.alloc_ms);
    } else {
      src = static_cast<std::uint8_t*>(map_pageable(total, opt.huge));
    }
    fill_pattern(reinterpret_cast<std::uint64_t*>(src), total / 8);
    std::fprintf(stderr,
                 "[src] %s filled (pattern + prefault) in %.2f s\n",
                 opt.mode == bench_mode::pinned ? "pinned" : "pageable",
                 seconds_since(t0));
  }

  // Staging ring: one pinned pool carved into threads*slots slots.
  pinned_block ring_pool;
  std::vector<worker> workers(static_cast<std::size_t>(opt.threads));
  if (uses_ring) {
    ring_pool = allocate_pinned(ring_bytes, opt.pinned_alloc, false);
    std::fprintf(stderr,
                 "[ring] pinned %s via %s: %.2f ms\n",
                 human_bytes(ring_bytes).c_str(),
                 to_string(opt.pinned_alloc),
                 ring_pool.alloc_ms);
  }
  for (int t = 0; t < opt.threads; ++t) {
    worker& w = workers[static_cast<std::size_t>(t)];
    w.tid     = t;
    CUDA_CHECK(cudaStreamCreateWithFlags(&w.stream, cudaStreamNonBlocking));
    if (uses_ring && !use_pool) {
      w.ring.resize(static_cast<std::size_t>(opt.slots));
      for (int r = 0; r < opt.slots; ++r) {
        slot& s  = w.ring[static_cast<std::size_t>(r)];
        s.pinned = static_cast<std::uint8_t*>(ring_pool.ptr) +
                   (static_cast<std::size_t>(t) * opt.slots + r) * S;
        CUDA_CHECK(cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming));
      }
    }
  }

  // Shared pool + shared DMA streams (--sched pool).
  slot_pool pool;
  std::vector<cudaStream_t> shared_streams;
  if (use_pool) {
    pool.slots.resize(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
      pool.slots[i].pinned = static_cast<std::uint8_t*>(ring_pool.ptr) + i * S;
      CUDA_CHECK(cudaEventCreateWithFlags(&pool.slots[i].event, cudaEventDisableTiming));
    }
    shared_streams.resize(static_cast<std::size_t>(n_streams));
    for (auto& st : shared_streams)
      CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  }

  std::atomic<std::size_t> next{0};
  copy_job job;
  job.opt     = &opt;
  job.src     = src;
  job.dst     = dst;
  job.nchunks = nchunks;
  job.next    = &next;
  if (use_pool) {
    job.pool    = &pool;
    job.streams = &shared_streams;
  }

  // Warmup + measured passes.
  {
    double s = run_pass(workers, job);
    std::fprintf(stderr, "[pass] warmup   %.4f s  %.1f GB/s\n", s, total / 1e9 / s);
  }
  // Only the measured passes are profiled: with
  // `nsys profile --capture-range=cudaProfilerApi` the trace starts here and
  // stops after the last pass, excluding allocation, fill and warmup.
  CUDA_CHECK(cudaProfilerStart());
  std::vector<double> gbps;
  double memcpy_s = 0, wait_s = 0, issue_s = 0;
  for (int it = 0; it < opt.iters; ++it) {
    double s = run_pass(workers, job);
    gbps.push_back(total / 1e9 / s);
    std::fprintf(stderr, "[pass] %-8d %.4f s  %.1f GB/s\n", it, s, gbps.back());
    if (it == opt.iters - 1) {
      for (auto& w : workers) {
        memcpy_s += w.memcpy_s;
        wait_s += w.wait_s;
        issue_s += w.issue_s;
      }
      // Fractions of aggregate worker wall time (threads * pass seconds).
      double denom = s * opt.threads;
      memcpy_s /= denom;
      wait_s /= denom;
      issue_s /= denom;
    }
  }
  CUDA_CHECK(cudaProfilerStop());
  if (use_pool) {
    std::fprintf(stderr,
                 "[pool] last pass: peak %zu of %zu slots queued on the GPU at once\n",
                 pool.peak_in_flight,
                 slot_count);
  }
  std::vector<double> sorted = gbps;
  std::sort(sorted.begin(), sorted.end());
  double median = sorted[sorted.size() / 2];
  double best   = sorted.back();
  std::fprintf(stderr,
               "[result] median %.1f GB/s  best %.1f GB/s  (last pass: memcpy %.0f%% wait %.0f%% "
               "issue %.0f%% of worker time)\n",
               median,
               best,
               memcpy_s * 100,
               wait_s * 100,
               issue_s * 100);

  // Verify.
  std::string verify_str = "skipped";
  if (opt.verify && dst != nullptr) {
    unsigned long long* d_mism = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mism, sizeof(*d_mism)));
    CUDA_CHECK(cudaMemset(d_mism, 0, sizeof(*d_mism)));
    std::size_t nwords = total / 8;
    int blocks         = prop.multiProcessorCount * 8;
    verify_kernel<<<blocks, 256>>>(reinterpret_cast<const std::uint64_t*>(dst), nwords, d_mism);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long mism = 0;
    CUDA_CHECK(cudaMemcpy(&mism, d_mism, sizeof mism, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_mism));
    verify_str = mism == 0 ? "ok" : "FAIL";
    std::fprintf(
      stderr, "[verify] %s (%llu mismatching words of %zu)\n", verify_str.c_str(), mism, nwords);
  }

  // Teardown (timed for the big pinned buffers: exit cost matters too).
  for (auto& w : workers) {
    for (auto& s : w.ring)
      CUDA_CHECK(cudaEventDestroy(s.event));
    CUDA_CHECK(cudaStreamDestroy(w.stream));
  }
  for (auto& s : pool.slots)
    CUDA_CHECK(cudaEventDestroy(s.event));
  for (auto st : shared_streams)
    CUDA_CHECK(cudaStreamDestroy(st));
  double alloc_ms = 0, free_ms = 0;
  if (uses_ring) {
    free_pinned(ring_pool);
    alloc_ms = ring_pool.alloc_ms;
    free_ms  = ring_pool.free_ms;
  }
  if (opt.mode == bench_mode::pinned) {
    free_pinned(src_pinned);
    alloc_ms = src_pinned.alloc_ms;
    free_ms  = src_pinned.free_ms;
    std::fprintf(stderr, "[src] pinned free: %.1f ms\n", free_ms);
  } else {
    munmap(src, total);
  }
  if (dst != nullptr) CUDA_CHECK(cudaFree(dst));

  if (opt.csv) {
    std::printf("%s,%s,%zu,%zu,%d,%d,%zu,%.3f,%.3f,%.2f,%.2f,%.3f,%.3f,%.3f,%s,%s,%d,%d,%d\n",
                to_string(opt.mode),
                to_string(opt.pinned_alloc),
                total,
                S,
                opt.threads,
                uses_ring ? opt.slots : 0,
                pinned_footprint,
                alloc_ms,
                free_ms,
                median,
                best,
                memcpy_s,
                wait_s,
                issue_s,
                verify_str.c_str(),
                use_pool ? "pool" : (uses_ring ? "ring" : ""),
                n_streams,
                opt.huge ? 1 : 0,
                opt.pin_cpus ? 1 : 0);
  }
  return verify_str == "FAIL" ? 3 : 0;
}
