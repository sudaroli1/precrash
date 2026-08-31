"""
Figure: the official score is a linear function of when the alarm fires.

WHAT THIS SHOWS
---------------
Fifteen submissions to the zero-shot-taa leaderboard, each a curve that reads no
pixels, scored by the organisers against ground truth no entrant has. Every one
is constant across clips, so all of them earn base-rate average precision and
chance area under the ROC curve; the only thing that varies is the frame at
which the risk score crosses 0.5.

The result is a straight line. Score falls at 1/60 per frame of delay, from
2.353 for an alarm at frame 0 down to a floor of 0.351 once the alarm arrives
after the accident. The two timing terms are worth 2.002 between them -- 5.7
times everything average precision and area under the ROC curve contribute
together.

A constant risk score of 0.51 therefore reaches 94% of the winning submission's
score while containing no information whatsoever.

FORM
----
The relationship is the finding, so the figure is a scatter of the measured
points with the fit drawn through them, not a bar chart of the scores. The
leaderboard entries are horizontal rules because they are levels to compare
against rather than a second series -- and drawing them as points on the same
axes would imply they have a crossing frame, which they do not.

Print constraints: single hue for measurements, ink for reference levels, and
the two kinds of reference line differ in dash pattern as well as weight so the
figure survives a monochrome printer.

  python scripts/make_figure_decomposition.py --scores submissions/scores.csv \
      --out results/fig_decomposition
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.console import safe_console  # noqa: E402

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e8e8e6"

# Measured on the leaderboard, for context rather than from our submissions.
LEADERBOARD = [
    ("winning submission", 2.50165, "solid"),
    ("our zero-shot ensemble", 2.00585, "dashed"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Draw the score-decomposition figure")
    p.add_argument("--scores", default=None,
                   help="CSV from `kaggle competitions submissions -v`. "
                        "Omit to use the values recorded in this file.")
    p.add_argument("--out", default="results/fig_decomposition")
    return p.parse_args()


# Scores as returned by the organisers, private leaderboard, 31 August 2026.
# Kept here so the figure regenerates without a network call; --scores
# overrides them from a fresh export.
MEASURED = {
    "never_crosses": 0.35105, "cross_then_drop": 1.22698,
    "cross_then_dip": 1.85347, "constant_0.51": 2.35291,
    "constant_0.99": 2.35291,
    "step_at_000": 2.35291, "step_at_010": 2.18624, "step_at_025": 1.93732,
    "step_at_050": 1.52088, "step_at_075": 1.10445, "step_at_100": 0.68822,
    "step_at_125": 0.37967, "step_at_140": 0.35105,
}


def load_scores(path: str | None) -> dict:
    if not path:
        return dict(MEASURED)
    out = dict(MEASURED)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        name = Path(r.get("fileName", "")).stem
        name = re.sub(r"^[pc]_", "", name)
        val = r.get("privateScore") or r.get("publicScore")
        if name in out and val:
            out[name] = float(val)
    return out


def main():
    safe_console()
    args = parse_args()
    S = load_scores(args.scores)

    ks = [0, 10, 25, 50, 75, 100, 125, 140]
    ys = [S[f"step_at_{k:03d}"] for k in ks]
    floor = S["never_crosses"]

    # Fit only where the alarm still precedes the accident in every clip. Past
    # that the curve bends as clip after clip stops earning any timing credit,
    # and fitting through the bend would misstate the slope.
    lin = [(k, y) for k, y in zip(ks, ys) if k <= 100]
    m, b = np.polyfit([k for k, _ in lin], [y for _, y in lin], 1)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=400)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9c5")

    # leaderboard levels
    for label, val, style in LEADERBOARD:
        ax.axhline(val, color=INK, linewidth=1.0, zorder=3,
                   linestyle="-" if style == "solid" else (0, (5, 2.5)))
        ax.annotate(f"{label}  {val:.3f}", xy=(148, val), xytext=(148, val + 0.045),
                    ha="right", va="bottom", fontsize=7.5, color=INK)

    # the floor
    ax.axhline(floor, color=MUTED, linewidth=0.9, linestyle=(0, (1.5, 2)), zorder=3)
    ax.annotate(f"floor: everything AP and AUC contribute  {floor:.3f}",
                xy=(4, floor), xytext=(4, floor - 0.10),
                ha="left", va="top", fontsize=7.5, color=MUTED)

    # fit, drawn only where it holds
    xs = np.linspace(0, 105, 50)
    ax.plot(xs, b + m * xs, color=BLUE, linewidth=1.2, alpha=0.55, zorder=4)

    ax.plot(ks, ys, "o", color=BLUE, markersize=5.5, zorder=5,
            markeredgecolor="white", markeredgewidth=0.7)

    ax.annotate("a constant 0.51 crosses at frame 0",
                xy=(0, ys[0]), xytext=(14, 2.245),
                fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                shrinkA=1, shrinkB=5))

    ax.annotate(f"slope {m:.4f} per frame\n(1 / {-1/m:.0f})",
                xy=(56, b + m * 56), xytext=(74, 1.72),
                fontsize=7.5, color=BLUE, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.6,
                                alpha=0.6, shrinkA=1, shrinkB=3))

    ax.annotate("alarms arriving after the accident\nearn no timing credit at all",
                xy=(140, ys[-1]), xytext=(148, 0.86),
                ha="right", fontsize=7.5, color=MUTED, linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                connectionstyle="arc3,rad=0.25",
                                shrinkA=3, shrinkB=4))

    ax.set_xlabel("Frame at which the risk score crosses 0.5", color=INK)
    ax.set_ylabel("Official leaderboard score", color=INK)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=3)
    ax.set_xlim(-4, 150)
    ax.set_ylim(0, 2.62)
    ax.set_xticks(range(0, 151, 25))

    fig.tight_layout(pad=0.4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")

    print(f"floor (AP+AUC)      {floor:.5f}")
    print(f"TTA term            {S['cross_then_drop'] - floor:.5f}")
    print(f"STTA term           {S['constant_0.51'] - S['cross_then_drop']:.5f}")
    print(f"timing total        {S['constant_0.51'] - floor:.5f}"
          f"   = {(S['constant_0.51'] - floor) / floor:.2f}x the floor")
    print(f"slope               {m:.6f} per frame  (1/{-1/m:.1f})")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
