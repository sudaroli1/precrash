"""
Synthetic feature cache — for testing the downstream chain, never for results.

WHY THIS EXISTS
---------------
Everything after extraction (evaluate.py, protocol_comparison.py, the tables and
figures) reads a directory of .npz files and never touches a video. That means
the whole reporting chain can be exercised without a GPU — and it should be,
because the alternative is discovering a broken column layout after an hour of
T4 time.

The curves it writes are drawn from a generator, not measured. Every file is
stamped with `synthetic=1`; the loaders in this repository ignore that field, so
the guard against mistaking these for real results is the directory name this
script insists on: it refuses to write anywhere that does not contain "synth".

USAGE
-----
  python scripts/make_synthetic_features.py \
      --manifest data/nexar_fixed.dev.csv \
      --out_dir features/synth_fixed_dev --limit 120
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Write a synthetic feature cache (testing only)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n_frames", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--signal", type=float, default=0.35,
                   help="How much the CLIP curve actually knows about the onset. "
                        "0 makes every modality video-blind noise, which is the "
                        "useful setting for checking that the controls behave.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    if "synth" not in out_dir.as_posix():
        raise SystemExit(
            "Refusing to write synthetic features to a directory whose path does not\n"
            "contain 'synth'. These curves are generated, not measured, and a cache of\n"
            "them sitting under a plausible name is exactly how a fabricated number\n"
            "reaches a table."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    T = args.n_frames
    t = np.arange(T)

    for r in rows:
        label = int(r["label"])
        onset = int(float(r["onset_frame"])) if (r.get("onset_frame") or "").strip() else -1
        alert = int(float(r["alert_frame"])) if (r.get("alert_frame") or "").strip() else -1
        fps = float(r["fps"]) if (r.get("fps") or "").strip() else 30.0

        # A slow drift plus noise, shared by all three modalities: this is the
        # part that carries no information about the collision and is what the
        # video-blind control is meant to match.
        drift = 1.0 / (1.0 + np.exp(-8.0 * (t / T - 0.55)))
        base = 0.45 * drift + 0.10 * rng.standard_normal(T).cumsum() / np.sqrt(T)

        p_clip = base + 0.05 * rng.standard_normal(T)
        if label == 1 and onset >= 0 and args.signal > 0:
            # A bump that actually tracks the annotated onset, so a correct
            # evaluation can separate positives from negatives at all.
            p_clip = p_clip + args.signal * np.exp(-0.5 * ((t - onset) / 12.0) ** 2)

        p_flow = base + 0.08 * rng.standard_normal(T)
        p_nlp = base + 0.03 * rng.standard_normal(T)

        clip_id = r["clip_id"]
        np.savez_compressed(
            out_dir / f"{clip_id}.npz",
            p_clip=np.clip(p_clip, 0, 1).astype(np.float32),
            p_flow=np.clip(p_flow, 0, 1).astype(np.float32),
            p_nlp=np.clip(p_nlp, 0, 1).astype(np.float32),
            label=np.int32(label),
            onset_frame=np.int32(onset),
            alert_frame=np.int32(alert),
            fps=np.float32(fps),
            n_frames=np.int32(T),
            synthetic=np.int32(1),
        )

    n_pos = sum(int(r["label"]) for r in rows)
    print(f"wrote {len(rows)} SYNTHETIC clips ({n_pos} positive, {len(rows)-n_pos} negative) "
          f"to {out_dir}")
    print("These are generated curves. Nothing computed from them is a result.")


if __name__ == "__main__":
    main()
