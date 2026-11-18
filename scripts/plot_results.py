#!/usr/bin/env python3
"""Draw docs/results-light.png and docs/results-dark.png from the per-epoch validation accuracy logged in training_jobs/*.json.

Usage:  python scripts/plot_results.py
"""
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
EPOCH_END = re.compile(r"E(\d+) END .*?ValAcc: ([\d.]+)%")
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", series="#2a78d6", context="#b9b8b1"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", series="#3987e5", context="#55544f"),
}


def load_curves():
    curves = {}
    for path in sorted((ROOT / "training_jobs").glob("*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        points = [(int(m.group(1)), float(m.group(2))) for line in job.get("log_messages", []) for m in [EPOCH_END.search(line)] if m]
        if len(points) >= 2:
            curves[job["job_id"]] = (job, sorted(points))
    return curves


def draw(theme, curves, total_runs):
    c = THEMES[theme]
    best_id = max((j for j, (job, _) in curves.items() if job["status"] == "completed"), key=lambda j: curves[j][0]["accuracy"])
    best_job, best_points = curves[best_id]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(c["surface"]); ax.set_facecolor(c["surface"])
    for job_id, (_, points) in curves.items():
        if job_id != best_id:
            ax.plot(*zip(*points), color=c["context"], linewidth=0.9, alpha=0.75, zorder=1)
    xs, ys = zip(*best_points)
    ax.plot(xs, ys, color=c["series"], linewidth=2, zorder=3)
    ax.scatter([xs[-1]], [ys[-1]], s=64, color=c["series"], edgecolor=c["surface"], linewidth=2, zorder=4)
    ax.annotate(f"Best run: {best_job['accuracy']:.2f}%", (xs[-1], ys[-1]), xytext=(8, -4), textcoords="offset points",
                color=c["ink"], fontsize=10, fontweight="bold", va="top")
    ax.set_ylim(97.0, 100.0)
    ax.set_xlim(1, max(p[-1][0] for _, p in curves.values()) + 9)
    ax.set_xlabel("Epoch", color=c["ink2"], fontsize=10)
    ax.set_ylabel("Validation accuracy (%)", color=c["ink2"], fontsize=10)
    ax.grid(axis="y", color=c["grid"], linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["grid"])
    ax.tick_params(colors=c["muted"], labelsize=9, length=0)
    handles = [plt.Line2D([], [], color=c["series"], linewidth=2, label="Best run"),
               plt.Line2D([], [], color=c["context"], linewidth=1.2, label=f"Other recorded runs ({len(curves) - 1})")]
    legend = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(c["ink2"])
    ax.text(0, 1.105, "Validation accuracy on MNIST across recorded training runs", transform=ax.transAxes, color=c["ink"],
            fontsize=12.5, fontweight="bold", va="bottom")
    ax.set_title(f"{len(curves)} of {total_runs} recorded runs logged accuracy per epoch. Runs below 97% are cut off by the axis.",
                 loc="left", color=c["ink2"], fontsize=9, pad=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = ROOT / "docs" / f"results-{theme}.png"
    fig.savefig(out, facecolor=c["surface"])
    plt.close(fig)
    return out


if __name__ == "__main__":
    curves = load_curves()
    total = len(list((ROOT / "training_jobs").glob("*.json")))
    for theme in THEMES:
        print(draw(theme, curves, total))
