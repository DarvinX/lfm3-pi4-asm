#!/usr/bin/env python3
"""Draw publication figures from the archived benchmark data. Requires Matplotlib."""

import csv
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

EVIDENCE = Path(__file__).resolve().parents[1] / "benchmarks"
INK = "#182739"
SECONDARY = "#42536A"
BLUE = "#1268A3"


def load_groups():
    with (EVIDENCE / "historical-timings.csv").open() as source:
        rows = list(csv.DictReader(source))

    cases = ("guide", "weather")
    engines = ("selected-h42", "pinned-cpp-3t")
    groups = {
        (case, engine): [row for row in rows if row["workload"] == case and row["engine"] == engine]
        for case in cases for engine in engines
    }
    assert all(len(group) == 3 for group in groups.values())
    return groups


def plot_results(groups):
    cases = ("guide", "weather")
    engines = ("selected-h42", "pinned-cpp-3t")
    labels = ("ARM64 assembly", "Pinned llama.cpp")
    colors = (BLUE, "#596B82")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    fig.subplots_adjust(left=.16, right=.965, top=.70, bottom=.23, wspace=.58)
    fig.text(.055, .91, "LFM3 1.7B on Raspberry Pi 4", fontsize=25, weight="bold", color="#182739")
    fig.text(.056, .835, "Q4_K_M checkpoint  /  128 generated tokens  /  3 paired runs per workload",
             fontsize=11.5, color="#42536A")

    panels = [
        ("Median decode throughput", "Tokens per second", 3.85, ".3f",
         lambda group: statistics.median(float(row["decode_tokens_per_second"]) for row in group)),
        ("Maximum peak RSS", "MiB", 1370, ",.1f",
         lambda group: max(int(row["peak_rss_kib"]) for row in group) / 1024),
    ]
    for ax, (title, unit, limit, precision, measure) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=14, color="#182739")
        for i, engine in enumerate(engines):
            y = [pos + (-.18 if i == 0 else .18) for pos in range(len(cases))]
            values = [measure(groups[case, engine]) for case in cases]
            bars = ax.barh(y, values, height=.30, color=colors[i], label=labels[i], zorder=3)
            ax.bar_label(bars, labels=[format(value, precision) for value in values],
                         padding=5, fontsize=10, color="#182739")
        ax.set_xlim(0, limit)
        ax.set_ylim(1.52, -.52)
        ax.set_yticks([0, 1], ["Tomato guide", "Rain explanation"], fontsize=10)
        ax.set_xlabel(unit, fontsize=10, color="#42536A")
        ax.xaxis.grid(True, color="#E3EAF1", linewidth=1)
        ax.tick_params(axis="both", length=0, colors="#42536A")
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.05, .08),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.056, .038, "Pinned llama.cpp 8642541. Token IDs and decoded bytes matched in every timed run.",
             fontsize=9, color="#42536A")
    fig.savefig(EVIDENCE / "pi4-results.png", dpi=180, facecolor="white")
    plt.close(fig)


def plot_greedy_head():
    fig = plt.figure(figsize=(12, 4.9), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.text(.055, .865, "How the engine picks a token", fontsize=26, weight="bold", color=INK)
    ax.text(.055, .775, "An exact greedy choice from 128,000 vocabulary rows",
            fontsize=13, color=SECONDARY)
    steps = (
        ("Coarse scan", "Score all rows with\ncompact weights.\nBound each score\nfrom above."),
        ("Set a threshold", "Read and score the\nbest coarse candidate\nwith its original\nQ6 weights."),
        ("Refine contenders", "Score original rows\nwhose upper bound\nis at or above\nthe threshold."),
        ("Pick the winner", "Use the highest\nexact score. Keep\nthe first row on\na tie."),
    )
    left_edges = (.055, .285, .515, .745)
    width = .2
    for index, (left, (title, detail)) in enumerate(zip(left_edges, steps)):
        ax.add_patch(FancyBboxPatch((left, .25), width, .43,
                     boxstyle="round,pad=0.005,rounding_size=0.012",
                     facecolor="#F1F7FC", edgecolor="#CBDCEB", linewidth=1.2))
        ax.text(left+.017, .59, f"0{index+1}", fontsize=13, weight="bold", color=BLUE)
        ax.text(left+.017, .51, title, fontsize=13, weight="bold", color=INK)
        ax.text(left+.017, .445, detail, fontsize=11.5, va="top", linespacing=1.38, color=SECONDARY)
        if index < len(steps)-1:
            ax.annotate("", xy=(left_edges[index+1]-.004, .475), xytext=(left+width+.004, .475),
                        arrowprops={"arrowstyle": "->", "color": BLUE, "linewidth": 1.6})

    ax.text(.055, .155, "Skip rows whose upper bound is below the threshold.", fontsize=12,
            weight="bold", color=INK)
    ax.text(.055, .095, "If the bounds cannot be used safely, score the full original head.",
            fontsize=11.5, color=SECONDARY)
    fig.savefig(EVIDENCE / "greedy-head-flow.png", dpi=180)
    plt.close(fig)


def plot_social_card(groups):
    case = "weather"
    assembly, baseline = (groups[case, engine] for engine in ("selected-h42", "pinned-cpp-3t"))
    decode = [statistics.median(float(row["decode_tokens_per_second"]) for row in group)
              for group in (assembly, baseline)]
    rss = [max(int(row["peak_rss_kib"]) for row in group)/1024 for group in (assembly, baseline)]
    background = "#141917"
    text = "#F2F0E6"
    muted = "#BBC5BA"
    accent = "#D3F677"
    baseline_color = "#7D968A"
    rule = "#3A463E"
    mono = "DejaVu Sans Mono"
    fig = plt.figure(figsize=(9, 9), facecolor=background)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    board = plt.imread(EVIDENCE / "pi4-board-illustration.png")
    ax.imshow(board, extent=(.628, .954, .628, .954), interpolation="lanczos")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(.066, .902, "ARM64 assembly\non a Raspberry Pi 4", fontsize=32, weight="bold",
            color=text, va="top", linespacing=1.04)
    ax.text(.07, .718, "LFM3-1.7B Q4_K_M  ·  4 GB RAM", fontsize=13.5, color=muted)
    ax.text(.07, .677, "Written with Codex  ·  Tuned with autoresearch", fontsize=13.5,
            color=muted)

    # Each pair shares a zero origin and a linear scale in its own unit.
    panels = (
        (.588, "Median decode speed", decode, 3.5, ".3f", "tok/s",
         f"+{100*(decode[0]/decode[1]-1):.1f}%"),
        (.355, "Peak process memory", rss, 1250, ",.1f", "MiB",
         f"−{rss[1]-rss[0]:.1f} MiB"),
    )
    for top, label, values, limit, precision, unit, delta in panels:
        ax.text(.07, top, label, fontsize=17, weight="bold", color=text, va="center")
        ax.text(.93, top, delta, fontsize=18, family=mono, weight="bold", color=accent,
                ha="right", va="center")
        ax.hlines(top-.028, .07, .93, color=rule, linewidth=1)
        for index, (engine, value, color) in enumerate(zip(
                ("Assembly", "llama.cpp"), values, (accent, baseline_color))):
            y = top-.071-index*.051
            ax.text(.07, y, engine, fontsize=13, family=mono,
                    color=text if index == 0 else muted, va="center")
            ax.barh(y, .435 * value/limit, left=.255, height=.026, color=color, zorder=3)
            ax.text(.93, y, f"{format(value, precision)} {unit}", fontsize=14, family=mono,
                    weight="bold" if index == 0 else "normal", color=text if index == 0 else muted,
                    ha="right", va="center")

    ax.hlines(.176, .07, .93, color=rule, linewidth=1)
    ax.text(.07, .137, "Rain explanation  ·  3 paired runs  ·  llama.cpp 8642541",
            fontsize=12, color=muted)
    ax.text(.07, .085, "768 tested greedy tokens matched across six prompts.",
            fontsize=13.5, weight="bold", color=text)
    fig.savefig(EVIDENCE / "pi4-social-card.png", dpi=120)
    plt.close(fig)


def main():
    groups = load_groups()
    plot_results(groups)
    plot_greedy_head()
    plot_social_card(groups)


if __name__ == "__main__":
    main()
