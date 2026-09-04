#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors.
# Licensed under the Apache License, Version 2.0. See LICENSE at the repo root.
"""Plot host->device throughput vs buffer (chunk) size, one line per strategy.

Reads one or more pinned_staging_bench CSVs and draws small multiples, one panel
per thread count, with the three strategies as lines:

  pinned    source already pinned
  pageable  cudaMemcpyAsync straight from pageable memory
  buffered  pageable -> preallocated pinned ring (2 slots/thread) -> device

No third-party dependencies: the chart is written as an SVG by hand.

    plot_results.py results/*.csv --threads 1,8,32 -o out.svg
    plot_results.py results/*.csv --threads 1,8,32 --table   # markdown table

Rows flagged --huge / --pin-cpus are skipped (older CSVs without those columns
fall back to first-occurrence-wins, which is the plain run in run_sweep.sh).
"""

import argparse
import csv
import math
import sys
from collections import OrderedDict

MIB = 1 << 20

SERIES = OrderedDict(
    [
        ("pinned", "Pinned source"),
        ("pageable", "Pageable source, direct copy"),
        ("staged", "Buffered: pageable → pinned ring → device"),
    ]
)

THEMES = {
    # Validated three-slot categorical palette (blue, orange, aqua) and chrome
    # from the dataviz reference palette, in both modes.
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {"pinned": "#2a78d6", "pageable": "#eb6834", "staged": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {"pinned": "#3987e5", "pageable": "#d95926", "staged": "#199e70"},
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def load(paths):
    """Return {(mode, chunk_mib, threads): median_gbps}, first occurrence wins."""
    data = {}
    for path in paths:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                mode = row.get("mode", "")
                if mode not in SERIES or not row.get("median_gbps"):
                    continue
                if row.get("huge", "0") == "1" or row.get("pin_cpus", "0") == "1":
                    continue
                if mode == "staged":
                    if row.get("sched", "") not in ("", "ring"):
                        continue
                    if int(row.get("slots") or 0) != 2:
                        continue
                chunk = int(row["chunk_bytes"])
                if chunk % MIB:
                    continue
                key = (mode, chunk // MIB, int(row["threads"]))
                data.setdefault(key, float(row["median_gbps"]))
    return data


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_gbps(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}"


def render(data, threads, chunks, theme_name, title, subtitle):
    t = THEMES[theme_name]
    n = len(threads)
    width = 1180
    height = 430
    left, right, top, bottom = 76, 20, 96, 56
    gap = 36
    label_room = 44  # inside each panel, right of the last point, for end labels
    panel_w = (width - left - right - gap * (n - 1)) / n
    plot_w = panel_w - label_room
    plot_h = height - top - bottom

    values = [v for k, v in data.items() if k[1] in chunks and k[2] in threads]
    ymax = max(values) if values else 400.0
    ymax = math.ceil(ymax / 50.0) * 50
    ystep = 100 if ymax > 250 else 50

    lx = [math.log2(c) for c in chunks]
    x0, x1 = min(lx), max(lx)

    def xpos(px0, chunk):
        return px0 + (math.log2(chunk) - x0) / (x1 - x0) * plot_w

    def ypos(v):
        return top + plot_h - v / ymax * plot_h

    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family=\'{FONT}\' role="img" '
        f'aria-label="{esc(title)}">'
    )
    out.append(f'<rect width="{width}" height="{height}" fill="{t["surface"]}"/>')
    out.append(
        f'<text x="{left}" y="28" font-size="15" font-weight="600" fill="{t["ink"]}">'
        f"{esc(title)}</text>"
    )
    out.append(
        f'<text x="{left}" y="46" font-size="12" fill="{t["ink2"]}">{esc(subtitle)}</text>'
    )

    # Legend: line-key + label per series, one row.
    lx_cursor = left
    for mode, label in SERIES.items():
        c = t["series"][mode]
        y = 68
        out.append(
            f'<line x1="{lx_cursor}" y1="{y}" x2="{lx_cursor + 22}" y2="{y}" '
            f'stroke="{c}" stroke-width="2" stroke-linecap="round"/>'
        )
        out.append(
            f'<circle cx="{lx_cursor + 11}" cy="{y}" r="4" fill="{c}" '
            f'stroke="{t["surface"]}" stroke-width="2"/>'
        )
        out.append(
            f'<text x="{lx_cursor + 30}" y="{y + 4}" font-size="12" fill="{t["ink"]}">'
            f"{esc(label)}</text>"
        )
        lx_cursor += 30 + 6.3 * len(label) + 28

    for i, th in enumerate(threads):
        px0 = left + i * (panel_w + gap)
        # Panel title
        out.append(
            f'<text x="{px0}" y="{top - 10}" font-size="13" font-weight="600" '
            f'fill="{t["ink"]}">{th} thread{"s" if th != 1 else ""}</text>'
        )
        # Gridlines + y ticks (labels on the first panel only; shared axis).
        v = 0
        while v <= ymax + 1e-9:
            y = ypos(v)
            out.append(
                f'<line x1="{px0}" y1="{y:.1f}" x2="{px0 + plot_w:.1f}" y2="{y:.1f}" '
                f'stroke="{t["grid"]}" stroke-width="1"/>'
            )
            if i == 0:
                # Unit rides the top tick so it cannot collide with the panel title.
                label = f"{v:,} GB/s" if v + ystep > ymax else f"{v:,}"
                out.append(
                    f'<text x="{px0 - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
                    f'fill="{t["muted"]}" font-variant-numeric="tabular-nums">{label}</text>'
                )
            v += ystep
        # Baseline
        out.append(
            f'<line x1="{px0}" y1="{ypos(0):.1f}" x2="{px0 + plot_w:.1f}" y2="{ypos(0):.1f}" '
            f'stroke="{t["axis"]}" stroke-width="1"/>'
        )
        # x ticks
        for c in chunks:
            x = xpos(px0, c)
            out.append(
                f'<text x="{x:.1f}" y="{top + plot_h + 18}" font-size="11" text-anchor="middle" '
                f'fill="{t["muted"]}" font-variant-numeric="tabular-nums">{c}</text>'
            )
        out.append(
            f'<text x="{px0 + plot_w / 2:.1f}" y="{top + plot_h + 38}" font-size="11" '
            f'text-anchor="middle" fill="{t["ink2"]}">Buffer size (MiB)</text>'
        )
        # Lines, markers, end labels
        ends = []
        for mode in SERIES:
            c = t["series"][mode]
            pts = [
                (xpos(px0, ch), ypos(data[(mode, ch, th)]), ch, data[(mode, ch, th)])
                for ch in chunks
                if (mode, ch, th) in data
            ]
            if not pts:
                continue
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts)
            out.append(
                f'<polyline points="{path}" fill="none" stroke="{c}" stroke-width="2" '
                f'stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for x, y, ch, val in pts:
                out.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{c}" '
                    f'stroke="{t["surface"]}" stroke-width="2">'
                    f"<title>{esc(SERIES[mode])}, {ch} MiB, {th} thread(s): {val:.1f} GB/s</title>"
                    f"</circle>"
                )
            ends.append([pts[-1][0], pts[-1][1], pts[-1][3], mode])
        # Direct end labels: nudge apart if they collide, then draw a leader.
        ends.sort(key=lambda e: e[1])
        min_gap = 13.0
        for j in range(1, len(ends)):
            if ends[j][1] - ends[j - 1][1] < min_gap:
                ends[j][1] = ends[j - 1][1] + min_gap
        for x, ly, val, mode in ends:
            true_y = ypos(val)
            if abs(ly - true_y) > 1.0:
                out.append(
                    f'<line x1="{x + 6:.1f}" y1="{true_y:.1f}" x2="{x + 12:.1f}" y2="{ly:.1f}" '
                    f'stroke="{t["axis"]}" stroke-width="1"/>'
                )
            out.append(
                f'<text x="{x + 14:.1f}" y="{ly + 4:.1f}" font-size="11" fill="{t["ink2"]}" '
                f'font-variant-numeric="tabular-nums">{fmt_gbps(val)}</text>'
            )

    out.append("</svg>")
    return "\n".join(out) + "\n"


def table(data, threads, chunks):
    lines = []
    for th in threads:
        lines.append(f"**{th} thread{'s' if th != 1 else ''}** (GB/s)")
        lines.append("")
        lines.append(
            "| Buffer (MiB) | "
            + " | ".join(SERIES[m].split(":")[0].split(",")[0] for m in SERIES)
            + " |"
        )
        lines.append("| --- | " + " | ".join("---" for _ in SERIES) + " |")
        for ch in chunks:
            cells = []
            for m in SERIES:
                v = data.get((m, ch, th))
                cells.append(fmt_gbps(v) if v is not None else "n/a")
            lines.append(f"| {ch} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "csv", nargs="+", help="benchmark CSV files (first occurrence of a point wins)"
    )
    ap.add_argument(
        "--threads",
        default="1,8,32",
        help="comma-separated thread counts, one panel each",
    )
    ap.add_argument(
        "--chunks",
        default="1,2,4,8,16,32,64",
        help="comma-separated buffer sizes in MiB",
    )
    ap.add_argument("--theme", choices=THEMES, default="light")
    ap.add_argument("--title", default="Host-to-device throughput vs buffer size")
    ap.add_argument(
        "--subtitle",
        default="GB300, 32 GiB per pass, median of 3 passes, decimal GB/s; buffered uses 2 preallocated pinned slots per thread",
    )
    ap.add_argument("-o", "--output", help="SVG output path (default stdout)")
    ap.add_argument(
        "--table", action="store_true", help="print a markdown table instead of SVG"
    )
    args = ap.parse_args()

    threads = [int(x) for x in args.threads.split(",")]
    chunks = [int(x) for x in args.chunks.split(",")]
    data = load(args.csv)
    missing = [
        (m, c, t)
        for m in SERIES
        for c in chunks
        for t in threads
        if (m, c, t) not in data
    ]
    if missing:
        print(
            f"warning: {len(missing)} missing points, e.g. {missing[:4]}",
            file=sys.stderr,
        )

    if args.table:
        print(table(data, threads, chunks))
        return
    svg = render(data, threads, chunks, args.theme, args.title, args.subtitle)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(svg)
    else:
        sys.stdout.write(svg)


if __name__ == "__main__":
    main()
