#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors.
# Licensed under the Apache License, Version 2.0. See LICENSE at the repo root.
"""Summarize an nsys sqlite export of pinned_staging_bench.

Answers "where does the time go" for one profiled run:

* per pass (NVTX "pass" range): wall time, bytes moved by host->device memcpys,
  aggregate GB/s, DMA busy fraction (union of memcpy intervals / pass wall),
  peak number of concurrent memcpys, and per-copy bandwidth statistics;
* the source memory kind CUPTI saw for the copies (pageable vs pinned);
* NVTX range totals per worker stage (memcpy / wait / issue) as a fraction of
  aggregate worker time;
* CUDA API time by call.

Usage: analyze_nsys.py report.sqlite
"""

import sqlite3
import statistics
import sys
from collections import defaultdict

SRC_KIND = {
    0: "unknown",
    1: "pageable",
    2: "pinned",
    3: "device",
    4: "array",
    5: "managed",
}
COPY_KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD"}


def table_exists(cur, name):
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def union_len(intervals):
    """Total length covered by a list of (start, end) intervals."""
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def peak_concurrency(intervals):
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


def nvtx_ranges(cur, name_filter=None):
    """Return list of (start, end, text, tid) NVTX push/pop ranges."""
    if not table_exists(cur, "NVTX_EVENTS"):
        return []
    cur.execute(
        "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL AND eventType IN (59, 60, 70, 71)"
    )
    rows = cur.fetchall()
    names = {}
    if table_exists(cur, "StringIds"):
        cur.execute("SELECT id, value FROM StringIds")
        names = dict(cur.fetchall())
    out = []
    for s, e, text, text_id, tid in rows:
        label = text if text is not None else names.get(text_id, "?")
        if name_filter is None or label == name_filter:
            out.append((s, e, label, tid))
    return out


def main(path):
    con = sqlite3.connect(path)
    cur = con.cursor()

    if not table_exists(cur, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        print("no CUPTI_ACTIVITY_KIND_MEMCPY table; nothing to analyze")
        return
    cur.execute(
        "SELECT start, end, bytes, copyKind, srcKind, dstKind, streamId "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 1"
    )
    copies = cur.fetchall()
    if not copies:
        print("no host->device memcpys recorded")
        return

    passes = sorted(nvtx_ranges(cur, "pass"))
    if not passes:
        # Fall back to one window covering every copy.
        passes = [(min(c[0] for c in copies), max(c[1] for c in copies), "all", 0)]

    # Prefer the enum table shipped inside the export over the hard-coded map.
    src_kind = dict(SRC_KIND)
    if table_exists(cur, "ENUM_CUDA_MEM_KIND"):
        cur.execute("SELECT id, name FROM ENUM_CUDA_MEM_KIND")
        src_kind.update(
            {
                i: n.lower().replace("cuda_memopr_memory_kind_", "")
                for i, n in cur.fetchall()
            }
        )

    print(f"report: {path}")
    print(f"host->device memcpys: {len(copies)}  passes: {len(passes)}")
    kinds = defaultdict(int)
    for c in copies:
        kinds[src_kind.get(c[4], str(c[4]))] += 1
    print(
        "source kind as seen by CUPTI: "
        + ", ".join(f"{k}={v}" for k, v in kinds.items())
    )
    print()
    print(
        f"{'pass':>4} {'wall_ms':>9} {'GiB':>7} {'agg_GB/s':>9} {'dma_busy':>8} "
        f"{'peak_conc':>9} {'n':>5} {'copy_GB/s med':>13} {'p10':>7} {'p90':>7}"
    )
    for i, (ps, pe, _, _) in enumerate(passes):
        inside = [c for c in copies if c[0] >= ps and c[1] <= pe]
        if not inside:
            continue
        wall = pe - ps
        nbytes = sum(c[2] for c in inside)
        ivals = [(c[0], c[1]) for c in inside]
        busy = union_len(ivals) / wall
        per_copy = sorted(
            c[2] / (c[1] - c[0]) for c in inside if c[1] > c[0]
        )  # bytes/ns = GB/s
        q = (
            statistics.quantiles(per_copy, n=10)
            if len(per_copy) >= 10
            else [per_copy[0]] * 9
        )
        print(
            f"{i:>4} {wall / 1e6:>9.2f} {nbytes / 2**30:>7.2f} {nbytes / wall:>9.1f} "
            f"{busy:>8.1%} {peak_concurrency(ivals):>9} {len(inside):>5} "
            f"{statistics.median(per_copy):>13.1f} {q[0]:>7.1f} {q[8]:>7.1f}"
        )

    # Worker-stage NVTX totals over the measured passes. Workers are fresh
    # std::threads each pass, so count threads per pass for the denominator.
    stage_ranges = [r for r in nvtx_ranges(cur) if r[2] in ("memcpy", "wait", "issue")]
    stages = defaultdict(int)
    for s, e, label, _ in stage_ranges:
        stages[label] += e - s
    if stages:
        denom = 0
        threads_per_pass = []
        for ps, pe, _, _ in passes:
            tids = {tid for s, e, _, tid in stage_ranges if ps <= s and e <= pe}
            threads_per_pass.append(len(tids))
            denom += (pe - ps) * max(1, len(tids))
        print()
        print(
            f"worker stages over {len(passes)} pass(es), "
            f"{max(threads_per_pass)} worker thread(s) per pass:"
        )
        for label in ("memcpy", "wait", "issue"):
            if label in stages:
                print(
                    f"  {label:>6}: {stages[label] / 1e6:>10.1f} ms total  "
                    f"{stages[label] / denom:>6.1%} of aggregate worker time"
                )

    # CUDA API time inside the passes.
    if table_exists(cur, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        cur.execute(
            "SELECT r.start, r.end, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id = r.nameId"
        )
        api = defaultdict(lambda: [0, 0])
        for s, e, name in cur.fetchall():
            if any(ps <= s and e <= pe for ps, pe, _, _ in passes):
                api[name][0] += 1
                api[name][1] += e - s
        if api:
            print()
            print("CUDA runtime API inside passes (count, total ms, mean us):")
            for name, (n, t) in sorted(api.items(), key=lambda kv: -kv[1][1])[:8]:
                print(f"  {name:<40} {n:>7} {t / 1e6:>10.1f} {t / n / 1e3:>10.1f}")

    # CPU samples (only present when the profile was taken with --sample). Leaf
    # frame per sample, so this names the hot host function (e.g. the memcpy
    # variant) without needing backtraces.
    if table_exists(cur, "COMPOSITE_EVENTS") and table_exists(
        cur, "SAMPLING_CALLCHAINS"
    ):
        try:
            cur.execute(
                "SELECT c.id, s.value, m.value FROM SAMPLING_CALLCHAINS c "
                "LEFT JOIN StringIds s ON s.id = c.symbol "
                "LEFT JOIN StringIds m ON m.id = c.module WHERE c.stackDepth = 0"
            )
            leaf = {
                cid: (sym or "?", (mod or "?").rsplit("/", 1)[-1])
                for cid, sym, mod in cur.fetchall()
            }
            cur.execute("SELECT start, callchainId FROM COMPOSITE_EVENTS")
            counts = defaultdict(int)
            total = 0
            for start, cid in cur.fetchall():
                if any(ps <= start <= pe for ps, pe, _, _ in passes):
                    counts[leaf.get(cid, ("?", "?"))] += 1
                    total += 1
            if total:
                print()
                print(
                    f"CPU samples inside passes: {total} (leaf function, module, share)"
                )
                for (sym, mod), n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
                    print(f"  {sym[:60]:<60} {mod[:24]:<24} {n / total:>6.1%}")
        except sqlite3.Error as exc:  # schema differences between nsys versions
            print(f"(cpu sample summary skipped: {exc})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
