"""
Single-clip diagnostic and figure generator.

Rewritten. The previous version called compute_tta() / compute_stta() on a bare
score curve and printed the result as a performance measure. Those functions had
no labels and have been removed; what they reported was the shape of the curve,
not whether it was right.

This version does the honest version of the same job: it draws the risk
trajectory for one clip, overlays the video-blind prior, and marks the annotated
collision onset. That comparison is the paper's argument in one picture, and it
is the figure to put in Section 4.

Any number it prints is labelled a diagnostic, not a metric. Corpus-level
performance comes from evaluate.py and protocol_comparison.py, which have
labels.

USAGE
-----
From a cached feature file — no GPU, no torch, runs in the local venv:

    python scripts/run_inference.py \
        --features features/dad_test/000821.npz \
        --fps 20 \
        --config configs/honest.yaml \
        --plot figures/clip_000821.png

From raw video — needs the extraction environment and a GPU:

    python scripts/run_inference.py \
        --video /path/to/clip.mp4 --onset 90 --fps 20 \
        --config configs/honest.yaml --plot figures/clip.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from postprocess import PostProcessor  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Plot the risk trajectory for one clip")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="Cached .npz from extract_features.py (no GPU needed)")
    src.add_argument("--video", help="Raw video file (requires the extraction environment)")
    p.add_argument("--onset", type=int, default=None,
                   help="Accident onset frame. Read from the .npz when using --features.")
    p.add_argument("--fps", type=float, required=True, help="DAD=20, CCD=10, MM-AU=30")
    p.add_argument("--config", default="configs/honest.yaml")
    p.add_argument("--plot", default=None, help="Where to save the figure (PNG)")
    p.add_argument("--prior_midpoint", type=float, default=0.6,
                   help="Midpoint of the video-blind sigmoid drawn for comparison")
    return p.parse_args()


def video_blind_prior(n_frames: int, midpoint: float = 0.6, steepness: float = 12.0):
    """A curve computed from the frame count alone. Reads no pixels."""
    t = np.linspace(0.0, 1.0, n_frames)
    return 1.0 / (1.0 + np.exp(-steepness * (t - midpoint)))


def load_curves(args, cfg):
    """Return (p_clip, p_flow, p_nlp, onset)."""
    if args.features:
        z = np.load(args.features)
        onset = args.onset if args.onset is not None else int(z["onset_frame"])
        return (z["p_clip"].astype(float), z["p_flow"].astype(float),
                z["p_nlp"].astype(float), onset if onset > 0 else None)

    # raw video path — heavy imports deferred so the cached path stays light
    from engines.clip_scorer import CLIPScorer
    from engines.flow_scorer import FlowScorer
    from engines.nlp_scorer import NLPScorer
    from utils.video_io import load_frames

    v = cfg["video"]
    frames = load_frames(
        args.video, num_frames=v["num_frames"], resize=tuple(v["resize"]),
        interpolation=v.get("interpolation", "bicubic"),
        crop_top_frac=v.get("crop_top_frac", 0.20),
        crop_bottom_frac=v.get("crop_bottom_frac", 0.08),
        normalise=v.get("normalise", True),
    )
    return (
        CLIPScorer(model_name=cfg["clip"]["model"],
                   danger_prompt=cfg["clip"]["danger_prompt"],
                   safe_prompt=cfg["clip"]["safe_prompt"]).score(frames),
        FlowScorer(use_raft=cfg["flow"].get("use_raft", False)).score(frames),
        NLPScorer(model_name=cfg["nlp"]["model"],
                  sudden_anchor=cfg["nlp"]["sudden_anchor"],
                  gradual_anchor=cfg["nlp"]["gradual_anchor"]).score(frames),
        args.onset,
    )


def first_crossing(p, threshold=0.5):
    hits = np.where(np.asarray(p) >= threshold)[0]
    return int(hits[0]) if len(hits) else None


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    p_clip, p_flow, p_nlp, onset = load_curves(args, cfg)
    T = len(p_clip)

    w = cfg["ensemble"]
    p_raw = w["clip_weight"] * p_clip + w["flow_weight"] * p_flow + w["nlp_weight"] * p_nlp
    p_final = PostProcessor(dict(cfg["postprocess"])).process(p_raw)
    p_prior = video_blind_prior(T, midpoint=args.prior_midpoint)

    thr = cfg["postprocess"].get("threshold", 0.5)

    print(f"clip length {T} frames @ {args.fps} fps"
          + (f", annotated onset at frame {onset}" if onset else ", onset unknown"))
    print("\n  diagnostic only — these are not performance metrics.")
    print("  Corpus performance needs labels: use evaluate.py.\n")
    print(f"  {'curve':<26}{'first crossing':>15}{'lead over onset':>18}")
    print("  " + "-" * 59)
    for name, curve in [("CLIP", p_clip), ("flow", p_flow), ("NLP prior", p_nlp),
                        ("ensemble", p_raw), ("ensemble + postproc", p_final),
                        ("VIDEO-BLIND prior", p_prior)]:
        tc = first_crossing(curve, thr)
        if tc is None:
            print(f"  {name:<26}{'never':>15}{'—':>18}")
        elif onset:
            lead = (onset - tc) / args.fps
            flag = "" if lead > 0 else "  (after onset)"
            print(f"  {name:<26}{tc:>15}{lead:>15.2f} s{flag}")
        else:
            print(f"  {name:<26}{tc:>15}{'—':>18}")

    prior_tc, ens_tc = first_crossing(p_prior, thr), first_crossing(p_final, thr)
    if prior_tc is not None and ens_tc is not None and prior_tc <= ens_tc:
        print("\n  Note: the video-blind prior crosses no later than the ensemble on this")
        print("  clip. On a crossing-frame metric it would score at least as well while")
        print("  having read nothing from the video.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.arange(T)
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        ax.plot(t, p_clip, lw=1.0, alpha=0.55, label="CLIP")
        ax.plot(t, p_flow, lw=1.0, alpha=0.55, label="Optical flow")
        ax.plot(t, p_nlp, lw=1.0, alpha=0.55, label="NLP prior")
        ax.plot(t, p_final, lw=2.2, color="#14202b", label="Ensemble + post-processing")
        ax.plot(t, p_prior, lw=2.0, ls="--", color="#b23a2e", label="Video-blind prior")

        ax.axhline(thr, color="grey", lw=0.8, ls=":")
        ax.text(1, thr + 0.015, f"threshold {thr}", fontsize=7, color="grey")
        if onset:
            ax.axvline(onset, color="#b23a2e", lw=1.0, alpha=0.7)
            ax.text(onset - 1, 1.02, "collision onset", fontsize=7,
                    color="#b23a2e", ha="right")

        ax.set_xlim(0, T - 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("frame")
        ax.set_ylabel("risk score")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        out = Path(args.plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"\nFigure: {out}")


if __name__ == "__main__":
    main()
