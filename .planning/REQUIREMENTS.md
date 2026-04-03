# Requirements: Multi-GPU Execution for Sirius

**Defined:** 2026-04-02
**Core Value:** Any query can transparently execute across multiple GPUs, with tasks scheduled to GPUs where their data already resides, and memory pressure handled by downgrading to the correct NUMA domain.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Foundation

- [ ] **FOUND-01**: System discovers GPU count, NUMA domains, and GPU-to-NUMA mapping at runtime via cucascade topology discovery
- [x] **FOUND-02**: Each GPU has its own executor instance with independent CUDA context, stream pool, and thread pool
- [x] **FOUND-03**: Each GPU has independent memory spaces with separate reservation tracking (OOM on GPU 0 does not block GPU 1)
- [ ] **FOUND-04**: Single-GPU systems work identically to current behavior with zero regression
- [x] **FOUND-05**: Per-GPU downgrade executor monitors its own memory space and downgrades independently
- [ ] **FOUND-06**: Device guard conventions enforced — all GPU operations scoped to correct device via cudaSetDevice

### Task Scheduling

- [ ] **SCHED-01**: Task creator routes pipeline tasks to the GPU where the most input data (by bytes) already resides
- [ ] **SCHED-02**: When no GPU has input data loaded, task is routed to a GPU on the same NUMA node as the data's host memory
- [ ] **SCHED-03**: Reservation tries preferred GPU first, then other GPUs; only waits on preferred GPU if no GPU can reserve
- [ ] **SCHED-04**: Different pipelines of the same query can execute on different GPUs based on data locality
- [ ] **SCHED-05**: Cross-GPU scan data routing — scan batches are distributed across GPUs based on available memory

### Memory Management

- [ ] **MEM-01**: When GPU memory is exhausted, data downgrades to pinned host memory on the same NUMA domain as that GPU first
- [ ] **MEM-02**: If local NUMA host memory is exhausted, downgrade falls back to cross-NUMA host memory
- [x] **MEM-03**: GPU-to-GPU data transfer works via host staging (GPU0 -> host -> GPU1) using cucascade converters
- [ ] **MEM-04**: GPU-direct peer-to-peer transfer via cudaMemcpyPeerAsync when P2P access is available (NVLink/PCIe P2P)
- [ ] **MEM-05**: Scan batches distributed across GPUs by available memory (adaptive scan partitioning)

### cucascade Updates

- [ ] **CUCS-01**: GPU-to-GPU representation converter registered in cucascade converter registry
- [ ] **CUCS-02**: Per-NUMA host memory spaces configured with numa_region_pinned_host_allocator
- [x] **CUCS-03**: Downgrade strategy uses NUMA-aware ordering (local NUMA first, cross-NUMA fallback)
- [x] **CUCS-04**: Multi-GPU memory space configuration tested and validated with N>1 GPUs

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Optimization

- **OPT-01**: Coordinated multi-GPU OOM handling — migrate data to peer GPU with free space instead of downgrading to host
- **OPT-02**: Topology-aware logging with per-GPU utilization metrics, data movement volume, scheduling decisions
- **OPT-03**: Hash-partitioned scan routing by join key for join co-location
- **OPT-04**: Automatic data rebalancing across GPUs

## Out of Scope

| Feature | Reason |
|---------|--------|
| Distributed multi-node execution | Different problem domain — needs network serialization, fault tolerance |
| GPU-Direct RDMA (network) | Only relevant for multi-node; adds NIC/OFED dependencies |
| Custom NVLink/NVSwitch protocols | CUDA runtime handles interconnect routing transparently |
| Query optimizer GPU placement | Schedule at task time with actual data sizes, not plan time with estimates |
| Data repartitioning / shuffle exchange | Single-node batch-level scheduling avoids need for global shuffle |
| Heterogeneous GPU support | Require homogeneous GPUs (matches DGX/HGX configs) |
| Changes to legacy code path | Multi-GPU targets Super Sirius only (namespace sirius) |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Complete |
| FOUND-02 | Phase 1 | Complete |
| FOUND-03 | Phase 1 | Complete |
| FOUND-04 | Phase 1 | Complete |
| FOUND-05 | Phase 1 | Complete |
| FOUND-06 | Phase 1 | Complete |
| SCHED-01 | Phase 2 | Pending |
| SCHED-02 | Phase 2 | Pending |
| SCHED-03 | Phase 2 | Pending |
| SCHED-04 | Phase 2 | Pending |
| SCHED-05 | Phase 2 | Pending |
| MEM-01 | Phase 3 | Pending |
| MEM-02 | Phase 3 | Pending |
| MEM-03 | Phase 1 | Complete |
| MEM-04 | Phase 3 | Pending |
| MEM-05 | Phase 3 | Pending |
| CUCS-01 | Phase 1 | Complete |
| CUCS-02 | Phase 1 | Complete |
| CUCS-03 | Phase 1 | Complete |
| CUCS-04 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20
- Unmapped: 0

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-02 after roadmap creation*
