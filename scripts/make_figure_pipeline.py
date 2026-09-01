"""
Figure 1: the pipeline as implemented.

Replaces three figures that between them said less and contradicted the text.
The old preprocessing diagram showed 1080p video, a horn/brake audio modality,
motion compensation, segmentation and a multiscale pyramid — none of which is
in this corpus or in the released code, and three of which Section 4 explicitly
withdraws. The old architecture and workflow diagrams overlapped almost
entirely; only the second named the operations, and it carried the tuned
ensemble weights and the temporal-compression stage without saying what became
of either.

This draws what the code does, and does one extra piece of work: the two stages
Section 6 identifies as mechanisms are drawn dashed, so a reader can see where
in the pipeline they sit instead of holding four pages of prose in mind.

  python paper/make_figure_pipeline.py --out results/fig_pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED = "#0b0b0b", "#55544f"
EDGE, FILL = "#8e8d88", "#eef3fa"
PLAIN = "#ffffff"


def box(ax, x, y, w, h, title, lines=(), dashed=False, fill=FILL):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.15 if dashed else 0.9,
        edgecolor=INK if dashed else EDGE,
        facecolor=PLAIN if dashed else fill,
        linestyle=(0, (3.2, 1.8)) if dashed else "-",
        zorder=3))
    cy = y + h / 2
    if lines:
        ax.text(x + w / 2, cy + 0.030, title, ha="center", va="center",
                fontsize=7.4, color=INK, fontweight="bold", zorder=4)
        ax.text(x + w / 2, cy - 0.028, "\n".join(lines), ha="center",
                va="center", fontsize=6.5, color=MUTED, linespacing=1.5,
                zorder=4)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                fontsize=6.9, color=INK, zorder=4)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=7.5,
        linewidth=0.8, color=EDGE, zorder=2, shrinkA=0, shrinkB=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fig_pipeline")
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(6.2, 4.6), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- input ----------------------------------------------------------
    box(ax, 0.255, 0.905, 0.49, 0.070, "Input",
        ["150 JPEG frames per clip, 30 fps, 5 s"])
    arrow(ax, (0.50, 0.905), (0.50, 0.862))

    # ---- preprocessing ---------------------------------------------------
    box(ax, 0.145, 0.770, 0.71, 0.092, "Preprocessing",
        ["crop top 20% and bottom 8%   ·   resize 224×224, bicubic",
         "per-frame normalisation for exposure"])

    # ---- three engines ---------------------------------------------------
    ey, eh, ew = 0.505, 0.185, 0.288
    xs = (0.030, 0.356, 0.682)
    engines = [
        ("CLIP ViT-L/14",
         ["cosine similarity of each frame", "to a danger and a safe prompt",
          "→  p_clip(t)"]),
        ("Frame difference",
         ["mean |f(t) − f(t−1)|,", "min–max normalised", "→  p_flow(t)"]),
        ("Motion-threshold text prior",
         ["the same frame-difference statistic,",
          "thresholded at 20 and 10 to select",
          "one of three fixed strings  →  p_prior(t)"]),
    ]
    for x, (t, ls) in zip(xs, engines):
        box(ax, x, ey, ew, eh, t, ls)
        arrow(ax, (0.50, 0.770), (x + ew / 2, ey + eh))
        arrow(ax, (x + ew / 2, ey), (0.50, 0.432))

    ax.text(0.50, 0.712, "no shared weights \u2014 but the right two are functions of one signal",
            ha="center", va="center", fontsize=6.4, color=MUTED,
            style="italic", zorder=5,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none"))

    # ---- ensemble --------------------------------------------------------
    box(ax, 0.145, 0.338, 0.71, 0.094, "Ensemble",
        ["weighted sum of the three curves, then 1-D Gaussian smoothing",
         "the released configuration weights the three equally"])
    arrow(ax, (0.50, 0.338), (0.50, 0.292))

    # ---- post-processing, as a row --------------------------------------
    ax.text(0.032, 0.245, "Post-processing", ha="left", va="center",
            fontsize=7.4, color=INK, fontweight="bold")
    py, ph = 0.135, 0.082
    cells = [
        (0.032, 0.200, "power transform\nγ = 1.8", False),
        (0.262, 0.215, "monotone clamp", True),
        (0.507, 0.240, "temporal compression", True),
        (0.787, 0.181, "clip to\n(0.001, 0.999)", False),
    ]
    for x, w, label, dash in cells:
        box(ax, x, py, w, ph, label, dashed=dash)
    for i in range(len(cells) - 1):
        x0 = cells[i][0] + cells[i][1]
        arrow(ax, (x0, py + ph / 2), (cells[i + 1][0], py + ph / 2))

    ax.text(0.627, 0.118, "disabled in the released code", ha="center",
            va="top", fontsize=6.2, color=MUTED, style="italic")

    arrow(ax, (0.878, py), (0.878, 0.062))
    ax.text(0.878, 0.045, "risk curve  p(t)", ha="center", va="center",
            fontsize=6.9, color=INK)

    ax.text(0.032, 0.045,
            "Dashed: the two stages analysed in Section 6. The clamp forces "
            "the condition the stable-timing\nmetric tests; the compression "
            "read a frame later than the one it reported.",
            ha="left", va="center", fontsize=6.3, color=MUTED,
            style="italic", linespacing=1.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
