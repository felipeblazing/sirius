# Roadmap: Multi-GPU Execution for Sirius

## Overview

This roadmap delivers multi-GPU query execution for the Sirius GPU SQL engine. The work progresses from establishing correct multi-GPU infrastructure (device management, memory spaces, cucascade wiring, cross-GPU data transfer), through data-locality-aware task scheduling (the critical performance differentiator), to NUMA-aware memory management and P2P transfer optimization. Each phase delivers a verifiable capability: Phase 1 proves N GPUs work independently and can move data between them; Phase 2 proves tasks land on the right GPU; Phase 3 proves memory pressure is handled with NUMA awareness and transfers are optimized.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Multi-GPU Foundation** - Per-GPU executors, memory spaces, cucascade wiring, device guards, and host-staged cross-GPU transfer
- [ ] **Phase 2: Data-Locality Task Scheduling** - Route pipeline tasks to the GPU where input data resides, with cross-GPU scan distribution
- [ ] **Phase 3: NUMA-Aware Memory and Transfer Optimization** - NUMA-ordered downgrade, P2P direct transfer, and adaptive scan partitioning

## Phase Details

### Phase 1: Multi-GPU Foundation
**Goal**: Multiple GPUs operate as independent execution units with correct device isolation, and data can move between them via host staging
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, FOUND-05, FOUND-06, CUCS-01, CUCS-02, CUCS-03, CUCS-04, MEM-03
**Success Criteria** (what must be TRUE):
  1. On a multi-GPU system, each GPU has its own executor with independent CUDA context, streams, and thread pool -- verified by running a simple query that uses GPU 0 and GPU 1 concurrently
  2. Memory exhaustion on GPU 0 does not prevent GPU 1 from accepting and executing tasks -- verified by filling GPU 0 memory and confirming GPU 1 continues normally
  3. A data batch on GPU 0 can be transferred to GPU 1 via host staging (GPU0 -> host -> GPU1) and the resulting data is correct -- verified by a round-trip transfer test
  4. On a single-GPU system, behavior is identical to current Sirius with no performance regression -- verified by running TPC-H queries and comparing results and timing
  5. cudaSetDevice is correctly scoped on all execution threads -- verified by running with CUDA debug assertions or compute-sanitizer on a 2+ GPU system
**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md -- NUMA-aware downgrade, multi-device terminate sync, and P2P access enablement
- [x] 01-02-PLAN.md -- Device guard audit and multi-GPU foundation validation tests
- [x] 01-03-PLAN.md -- NUMA-aware downgrade tests and GPU-to-GPU transfer validation

### Phase 2: Data-Locality Task Scheduling
**Goal**: Pipeline tasks execute on the GPU where their input data already resides, minimizing cross-GPU data movement
**Depends on**: Phase 1
**Requirements**: SCHED-01, SCHED-02, SCHED-03, SCHED-04, SCHED-05
**Success Criteria** (what must be TRUE):
  1. A pipeline task whose input data is entirely on GPU 1 executes on GPU 1, not GPU 0 -- verified by logging which GPU each task runs on and confirming locality match
  2. When no GPU has the input data loaded, the task is routed to a GPU on the same NUMA node as the host memory where data resides -- verified on a multi-NUMA system
  3. When all GPUs are at capacity, the task waits on its preferred GPU rather than being dispatched to a random GPU -- verified by saturating GPU memory and observing wait behavior in logs
  4. A multi-pipeline query (e.g., TPC-H Q5 with multiple joins) can have different pipelines execute on different GPUs based on where each pipeline's data landed -- verified by log inspection showing cross-GPU pipeline distribution
**Plans:** 2 plans

Plans:
- [ ] 02-01-PLAN.md -- Data locality computation in task_creator and locality-aware routing in management_eventloop
- [ ] 02-02-PLAN.md -- Cross-GPU scan distribution and integration tests for locality scheduling

### Phase 3: NUMA-Aware Memory and Transfer Optimization
**Goal**: Memory downgrade respects NUMA topology for minimal latency, and GPU-to-GPU transfers use direct P2P when available
**Depends on**: Phase 2
**Requirements**: MEM-01, MEM-02, MEM-04, MEM-05
**Success Criteria** (what must be TRUE):
  1. When GPU 0 (on NUMA node 0) runs out of memory, data downgrades to pinned host memory on NUMA node 0 first, not NUMA node 1 -- verified by numastat or equivalent showing allocation locality
  2. If NUMA-local host memory is also exhausted, downgrade falls back to cross-NUMA host memory rather than failing -- verified by filling both GPU and local host memory
  3. On systems with NVLink or PCIe P2P access, GPU-to-GPU transfer uses cudaMemcpyPeerAsync directly (skipping host staging) and is measurably faster than host-staged path -- verified by transfer bandwidth comparison
  4. Scan batches are distributed across GPUs proportional to available memory, not round-robin -- verified by running a large scan with asymmetric GPU memory availability and checking batch distribution
**Plans:** 1 plan

Plans:
- [ ] 03-01-PLAN.md -- Verification tests for MEM-01/02/04/05 (NUMA downgrade, P2P transfer, proportional scan)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Multi-GPU Foundation | 3/3 | Complete | - |
| 2. Data-Locality Task Scheduling | 0/2 | Planning | - |
| 3. NUMA-Aware Memory and Transfer Optimization | 0/1 | Planning | - |
