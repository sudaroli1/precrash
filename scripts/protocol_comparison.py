"""
The central experiment of the critique paper.

CLAIM UNDER TEST
----------------
On the evaluation protocol commonly used in zero-shot accident anticipation —
crossover-frame / time-to-accident measured on accident-only clips — a baseline
that never looks at the video ranks competitively with, or ahead of, methods
that do. Under a protocol with negatives, annotated onsets and AP/AUC, the same
baseline drops to chance.

If both halves of that hold, the metric is measuring curve shape rather than
anticipation, and every result reported on it alone is uninterpretable.

WHAT THIS SCRIPT PRODUCES
-------------------------
One table with two panels over the same models and the same clips:

  LEGACY PANEL     mean crossover frame, TTA-to-clip-end, STTA compliance,
                   computed on positives only — deliberately reproducing the
                   flawed protocol, including this project's own earlier code.

  CORRECTED PANEL  AP, AUC, mTTA, TTA@R80 against labels and annotated onsets.

The video-blind priors are a FAMILY, not one hand-picked curve. A single
flattering sigmoid invites the reply "you tuned your strawman." Sweeping shapes
and midpoints shows the effect is a property of the metric, not of one curve.

USAGE
-----
  python scripts/protocol_comparison.py \
      --features features/nexar_fixed_dev \
      --config configs/honest.yaml \
      --out results/protocol_dad.json \
      --markdown results/table_dad.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from postprocess import PostProcessor
from utils.metrics import evaluate_anticipation, matched_windows, paired_bootstrap_test


# ----------------------------------------------------------------------
# The flawed protocol, reproduced faithfully so it can be compared against
# ----------------------------------------------------------------------

def legacy_metrics(scores, labels, fps, threshold=0.5):
    """
    Reproduces the protocol this paper originally used, and which appears in
    parts of the zero-shot anticipation literature:

      * positives only — negatives are not evaluated at all
      * crossover frame t_c = first frame with p(t) >= 0.5
      * TTA = (clip_length - t_c) / fps, i.e. measured to the END OF THE CLIP
        rather than to the annotated collision frame
      * "STTA compliance" = the score never drops back below threshold

    Reported here only as the object of study. Do not use these numbers as
    evidence of anything except that the protocol is uninformative.
    """
    fps_list = fps if isinstance(fps, (list, tuple)) else [fps] * len(scores)
    crossovers, ttas, compliant = [], [], []
    for i, (s, lab) in enumerate(zip(scores, labels)):
        if lab != 1:
            continue                      # the defect: negatives never examined
        s = np.asarray(s)
        hits = np.where(s >= threshold)[0]
        if len(hits) == 0:
            continue
        tc = int(hits[0])
        crossovers.append(tc)
        ttas.append((len(s) - tc) / fps_list[i])  # the defect: distance to clip end
        compliant.append(bool(np.all(s[tc:] >= threshold)))

    return {
        "mean_crossover_frame": float(np.mean(crossovers)) if crossovers else float("nan"),
        "mean_tta_to_clip_end": float(np.mean(ttas)) if ttas else float("nan"),
        "stta_compliance_rate": float(np.mean(compliant)) if compliant else float("nan"),
        "n_positives_scored": len(crossovers),
    }


# ----------------------------------------------------------------------
# The family of video-blind baselines
# ----------------------------------------------------------------------

def video_blind_priors(n_frames: int) -> dict[str, np.ndarray]:
    """
    Curves computed from the frame count alone. No pixels are read. Any of
    these scoring competitively under the legacy panel is the finding.
    """
    t = np.linspace(0.0, 1.0, n_frames)
    out = {}
    for mid in (0.40, 0.50, 0.60, 0.70):
        out[f"prior_sigmoid_m{mid:.2f}"] = 1.0 / (1.0 + np.exp(-12.0 * (t - mid)))
    out["prior_linear_ramp"] = t.copy()
    out["prior_step_at_half"] = (t >= 0.5).astype(float) * 0.99 + 0.005
    out["prior_constant_0.51"] = np.full(n_frames, 0.51)   # crosses at frame 0
    return out


def build_variants(data, cfg):
    w = cfg["ensemble"]
    ens = [w["clip_weight"] * c + w["flow_weight"] * f + w["nlp_weight"] * n
           for c, f, n in zip(data["p_clip"], data["p_flow"], data["p_nlp"])]

    pp = PostProcessor(dict(cfg["postprocess"]))

    variants = {
        "clip_only": list(data["p_clip"]),
        "flow_only": list(data["p_flow"]),
        "nlp_only": list(data["p_nlp"]),
        "ensemble_raw": ens,
        "ensemble_postproc": [pp.process(s) for s in ens],
    }

    prior_names = list(video_blind_priors(10).keys())
    for name in prior_names:
        variants[name] = [video_blind_priors(len(c))[name] for c in data["p_clip"]]

    return variants, prior_names


def load_features(feature_dir: Path) -> dict:
    files = sorted(feature_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files in {feature_dir}. Run extract_features.py first.")
    data = {"p_clip": [], "p_flow": [], "p_nlp": [], "label": [], "onset": [], "fps": []}
    for f in files:
        z = np.load(f)
        onset = int(z["onset_frame"])
        data["p_clip"].append(z["p_clip"].astype(np.float64))
        data["p_flow"].append(z["p_flow"].astype(np.float64))
        data["p_nlp"].append(z["p_nlp"].astype(np.float64))
        data["label"].append(int(z["label"]))
        data["onset"].append(onset if onset > 0 else None)
        rate = float(z["fps"]) if "fps" in z.files else -1.0
        data["fps"].append(rate if rate > 0 else None)
    return data


def main():
    ap = argparse.ArgumentParser(description="Legacy vs corrected protocol comparison")
    ap.add_argument("--features", required=True)
    ap.add_argument("--fps", type=float, default=None,
                    help="Fallback corpus rate; the per-clip rate cached at extraction is preferred.")
    ap.add_argument("--config", default="configs/honest.yaml")
    ap.add_argument("--out", default="results/protocol_comparison.json")
    ap.add_argument("--markdown", default=None, help="Also write the table as markdown")
    ap.add_argument("--dataset_name", default="dataset")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data = load_features(Path(args.features))
    n_pos = sum(l == 1 for l in data["label"])
    n_neg = len(data["label"]) - n_pos
    if n_neg == 0:
        raise SystemExit("No negative clips — the corrected panel cannot be computed.")

    if all(f is not None for f in data["fps"]):
        FPS = [float(f) for f in data["fps"]]
        import statistics
        rate_note = (f"per-clip rates {min(FPS):.2f}-{max(FPS):.2f}, "
                     f"median {statistics.median(FPS):.2f}")
    elif args.fps is not None:
        FPS = float(args.fps); rate_note = f"{FPS} fps (corpus-wide fallback)"
    else:
        raise SystemExit("No cached per-clip rate and no --fps given.")
    print(f"{args.dataset_name}: {n_pos} positive, {n_neg} negative clips, {rate_note}\n")

    # Decision windows. Positives are judged on frames before their onset; left
    # to themselves, negatives would be judged on the whole clip, which is 150
    # frames against 91-135 and hands any rising curve a higher maximum on every
    # negative for reasons unrelated to the video. `matched_windows` draws each
    # negative a pseudo-onset from the positives' distribution (seeded), which is
    # the equal-exposure design DAD and CCD get for free from fixed-length clips.
    # Both are computed: the difference between them is itself a result.
    n_frames = [len(c) for c in data["p_clip"]]
    win = matched_windows(data["label"], data["onset"], n_frames, seed=0)
    win_len = [w for w, lab in zip(win, data["label"]) if lab == 0]
    print(f"  decision windows: positives end at their onset; negatives matched to "
          f"{min(win_len)}-{max(win_len)} frames (unmatched would be {max(n_frames)})\n")

    variants, prior_names = build_variants(data, cfg)

    rows = []
    for name, scores in variants.items():
        legacy = legacy_metrics(scores, data["label"], FPS)
        corrected = evaluate_anticipation(scores, data["label"], data["onset"],
                                          fps=FPS, windows=win)
        unmatched = evaluate_anticipation(scores, data["label"], data["onset"], fps=FPS)
        rows.append({
            "method": name,
            "video_blind": name in prior_names,
            "legacy_crossover": legacy["mean_crossover_frame"],
            "legacy_tta": legacy["mean_tta_to_clip_end"],
            "legacy_stta": legacy["stta_compliance_rate"],
            "ap": corrected.ap,
            "auc": corrected.auc,
            "mtta": corrected.mtta,
            "tta_r80": corrected.tta_at_r80,
            # same metric, negatives given the whole clip instead of a matched
            # window — kept so the artefact can be shown rather than asserted
            "ap_unmatched_windows": unmatched.ap,
            "auc_unmatched_windows": unmatched.auc,
        })

    # legacy protocol ranks by earliest crossover — lower is "better"
    legacy_rank = sorted(rows, key=lambda r: (np.isnan(r["legacy_crossover"]), r["legacy_crossover"]))
    for i, r in enumerate(legacy_rank, 1):
        r["legacy_rank"] = i
    corrected_rank = sorted(rows, key=lambda r: -r["ap"])
    for i, r in enumerate(corrected_rank, 1):
        r["corrected_rank"] = i

    hdr = (f"{'method':<26}{'blind':>6}{'t_c':>8}{'TTA*':>8}{'STTA':>7}"
           f"{'| AP':>9}{'AUC':>8}{'TTA@R80':>9}{'rank L→C':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["legacy_rank"]):
        blind = "yes" if r["video_blind"] else ""
        print(f"{r['method']:<26}{blind:>6}"
              f"{r['legacy_crossover']:>8.1f}{r['legacy_tta']:>8.2f}{r['legacy_stta']:>7.2f}"
              f"{r['ap']:>9.4f}{r['auc']:>8.4f}{r['tta_r80']:>9.2f}"
              f"{r['legacy_rank']:>6} → {r['corrected_rank']:<3}")
    print("\n  t_c / TTA* / STTA = legacy protocol (positives only, TTA to clip end).")
    print("  AP / AUC / TTA@R80 = corrected protocol (negatives + annotated onsets,")
    print("                       negatives given a matched decision window).")

    # ---- the headline number ----
    best_blind = min((r for r in rows if r["video_blind"]), key=lambda r: r["legacy_rank"])
    real = [r for r in rows if not r["video_blind"]]
    beaten = [r["method"] for r in real if r["legacy_rank"] > best_blind["legacy_rank"]]

    print("\n" + "=" * 78)
    print(f"  Best video-blind baseline: {best_blind['method']}")
    print(f"    legacy rank {best_blind['legacy_rank']} of {len(rows)}  "
          f"(t_c = {best_blind['legacy_crossover']:.1f}, "
          f"TTA = {best_blind['legacy_tta']:.2f}s, "
          f"STTA = {best_blind['legacy_stta']:.0%})")
    print(f"    corrected: AP = {best_blind['ap']:.4f}, AUC = {best_blind['auc']:.4f}")
    if beaten:
        print(f"    outranks under the legacy protocol: {', '.join(beaten)}")
        print("\n  This is the paper. A curve that reads no pixels outranks methods that do,")
        print("  on the metric the subfield reports — and sits at chance once negatives")
        print("  and real onsets are introduced.")
    else:
        print("\n  The video-blind baselines do NOT outrank the real methods here.")
        print("  The critique does not hold on this dataset. Check a second benchmark")
        print("  before abandoning it, then report the negative result honestly.")
    print("=" * 78)

    # significance of the ensemble's advantage over the best blind prior
    test = paired_bootstrap_test(
        variants["ensemble_postproc"], variants[best_blind["method"]],
        data["label"], data["onset"], fps=FPS, metric="ap", windows=win,
    )
    print(f"\n  ensemble_postproc vs {best_blind['method']}: "
          f"ΔAP = {test['mean_diff']:+.4f}  95% CI [{test['ci_low']:+.4f}, {test['ci_high']:+.4f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dataset": args.dataset_name, "fps": FPS,
                   "n_positive": n_pos, "n_negative": n_neg,
                   "window_policy": "negatives matched to the positive onset "
                                    "distribution, seed 0",
                   "rows": rows, "ensemble_vs_best_prior": test}, f, indent=2)
    print(f"\nSaved: {out}")

    if args.markdown:
        md = Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        with open(md, "w") as f:
            f.write(f"**{args.dataset_name}** — {n_pos} positive / {n_neg} negative clips, "
                    f"{rate_note}\n\n")
            f.write("| Method | Video-blind | t_c ↓ | TTA* (s) | STTA | AP ↑ | AUC ↑ | TTA@R80 (s) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for r in sorted(rows, key=lambda r: r["legacy_rank"]):
                f.write(f"| {r['method']} | {'**yes**' if r['video_blind'] else ''} "
                        f"| {r['legacy_crossover']:.1f} | {r['legacy_tta']:.2f} "
                        f"| {r['legacy_stta']:.2f} | {r['ap']:.4f} | {r['auc']:.4f} "
                        f"| {r['tta_r80']:.2f} |\n")
            f.write("\n*TTA\\* is measured to the end of the clip, as in the legacy protocol, "
                    "not to the annotated collision frame.*\n")
            f.write("\n*AP and AUC give negative clips a decision window drawn from the "
                    "positive onset distribution (seed 0), so both classes are observed "
                    "for a comparable number of frames. Without that matching a rising "
                    "curve attains its maximum on every negative and the column measures "
                    "window length rather than risk; the unmatched values are in the "
                    "accompanying JSON.*\n")
        print(f"Saved: {md}")


if __name__ == "__main__":
    main()
