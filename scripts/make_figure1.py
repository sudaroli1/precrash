"""
Figure 1 — the annotated anticipation ceiling.

Regenerates the paper's Section 3.3 figure from the corpus annotation file.
Needs no GPU, no features and no model: it reads two columns of train.csv.
That is the point of the figure — the check it depicts costs one pass over an
annotation file and can be applied to a published result without access to its
code.

  python scripts/make_figure1.py --csv data/train.csv --out results/fig1_ceiling

FORM
----
The question is "where does one claimed value sit in the distribution the corpus
admits", so the figure is a distribution with the claim marked on it, not a bar
chart of summary statistics.

Print constraints: single column, one hue, and the two reference lines separated
by dash pattern as well as position, so the figure survives a monochrome
printer. Labels are placed directly rather than in a legend box.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The figure marks the value reported as a *mean* anticipation in the earlier
# manuscript. It is a constant of the argument, not a parameter to tune.
CLAIM = 4.23

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e8e8e6"


# Windows consoles default to cp1252. Nothing printed here should be able to
# raise UnicodeEncodeError partway through a long run; see src/utils/console.py.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Draw the annotated-ceiling figure")
    p.add_argument("--csv", required=True, help="Nexar train.csv")
    p.add_argument("--out", default="results/fig1_ceiling",
                   help="Output path without extension; .pdf and .png are written")
    p.add_argument("--claim", type=float, default=CLAIM)
    return p.parse_args()


def load_intervals(path: Path) -> np.ndarray:
    """Alert-to-event interval, in seconds, for every annotated positive clip."""
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        if int(float(r["target"])) != 1:
            continue
        if not (r.get("time_of_event") and r.get("time_of_alert")):
            continue
        out.append(float(r["time_of_event"]) - float(r["time_of_alert"]))
    if not out:
        raise SystemExit(
            f"No annotated positive clips in {path}. This figure needs the "
            "time_of_event and time_of_alert columns."
        )
    return np.asarray(out)


def main():
    args = parse_args()
    lead = load_intervals(Path(args.csv))
    mean = float(lead.mean())
    n_admit = int((lead >= args.claim).sum())

    print(f"n={len(lead)}  mean={mean:.3f}s  median={np.median(lead):.3f}s  "
          f"max={lead.max():.3f}s")
    print(f"clips admitting >= {args.claim}s: {n_admit} "
          f"({100 * n_admit / len(lead):.2f}%)")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=400)

    hi = max(4.75, float(np.ceil(lead.max() * 4) / 4))
    ax.hist(lead, bins=np.arange(0, hi + 0.125, 0.125),
            color=BLUE, edgecolor="white", linewidth=0.3, zorder=2)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9c5")

    ax.axvline(mean, color=INK, linewidth=1.4, zorder=4)
    ax.axvline(args.claim, color=INK, linewidth=1.4,
               linestyle=(0, (4, 2)), zorder=4)

    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.20)

    ax.annotate(f"corpus mean\n{mean:.2f} s",
                xy=(mean, ymax * 0.90), xytext=(mean + 0.16, ymax * 1.02),
                color=INK, fontsize=8, linespacing=1.35, ha="left", va="top")

    ax.annotate(f"reported as a $\\it{{mean}}$\nanticipation: {args.claim:.2f} s",
                xy=(args.claim, ymax * 0.30),
                xytext=(args.claim - 0.18, ymax * 1.02),
                color=INK, fontsize=8, linespacing=1.35, ha="right", va="top")

    ax.annotate(f"{n_admit} of {len(lead)} clips "
                f"({100 * n_admit / len(lead):.1f}%)\nadmit it at all",
                xy=(args.claim + 0.10, 1.4),
                xytext=(args.claim - 0.81, ymax * 0.44),
                color=MUTED, fontsize=7.5, linespacing=1.35,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                shrinkA=3, shrinkB=2,
                                connectionstyle="arc3,rad=-0.30"))

    ax.set_xlabel("Alert-to-event interval (s)", color=INK)
    ax.set_ylabel("Positive clips", color=INK)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=3)
    ax.set_xlim(0, hi)
    ax.set_xticks(np.arange(0, hi + 0.5, 0.5))

    fig.tight_layout(pad=0.4)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
