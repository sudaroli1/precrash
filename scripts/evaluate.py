"""
Pass 2 of 2 — CPU evaluation over cached features. Runs in seconds.

This replaces evaluate_mmau.py, which globbed a directory of videos, loaded no
labels, and computed every metric from the model's own output curve. Everything
here is measured against ground truth.

It also runs the ablation grid, including the control the whole paper turns
on: a FIXED PRIOR that ignores the video entirely and emits the same curve for
every clip. Tables 10 and 12 of the paper show the text-anchored prior — which
cannot see the video, and has three reachable states across the corpus —
posting an earlier mean crossover than the full ensemble. That is what the
paper's critique framing rests on. This script is how the same question would
be asked on a corpus that publishes labels, where AP and AUC are computable.

USAGE
-----
  python scripts/evaluate.py \
      --features features/nexar_fixed_dev \
      --config configs/honest.yaml \
      --out results/dad_test.json

Add --bootstrap for 95% confidence intervals (slower, ~1 min).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.console import safe_console  # noqa: E402
from postprocess import PostProcessor
from utils.metrics import (
    evaluate_anticipation,
    matched_windows,
    bootstrap_ci,
    paired_bootstrap_test,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate cached features against ground truth")
    p.add_argument("--features", required=True, help="Directory of .npz files from extract_features.py")
    p.add_argument("--fps", type=float, default=None,
                   help="Fallback corpus frame rate, used only when the cached features "
                        "carry no per-clip rate. Prefer the cached rate: a scalar is wrong "
                        "whenever clips differ, and mis-scales every reported TTA.")
    p.add_argument("--config", default="configs/honest.yaml")
    p.add_argument("--out", default="results/evaluation.json")
    p.add_argument("--bootstrap", action="store_true", help="Compute 95%% CIs (slower)")
    return p.parse_args()


def resolve_fps(cached, fallback):
    """Prefer the per-clip rate cached at extraction; fall back to a scalar.

    A single corpus rate is only correct when every clip shares it. Nexar spans
    23.6-31.0 fps across 36 distinct values, so a scalar mis-scales most clips.
    """
    if all(f is not None for f in cached):
        return [float(f) for f in cached]
    if fallback is None:
        raise SystemExit(
            "Cached features carry no per-clip frame rate and --fps was not given.\n"
            "Re-extract with a manifest built by prepare_nexar.py (which records fps),\n"
            "or pass --fps for a corpus where every clip shares one rate."
        )
    n_missing = sum(f is None for f in cached)
    print(f"  note: {n_missing} clips lack a cached rate; using --fps {fallback} for all")
    return float(fallback)


def load_features(feature_dir: Path) -> dict:
    """Load every cached clip. Returns dict of parallel lists."""
    files = sorted(feature_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files in {feature_dir}. Run extract_features.py first.")

    data = {"clip_id": [], "p_clip": [], "p_flow": [], "p_nlp": [],
            "label": [], "onset": [], "fps": []}
    for f in files:
        z = np.load(f)
        onset = int(z["onset_frame"])
        data["clip_id"].append(f.stem)
        data["p_clip"].append(z["p_clip"].astype(np.float64))
        data["p_flow"].append(z["p_flow"].astype(np.float64))
        data["p_nlp"].append(z["p_nlp"].astype(np.float64))
        data["label"].append(int(z["label"]))
        data["onset"].append(onset if onset > 0 else None)
        rate = float(z["fps"]) if "fps" in z.files else -1.0
        data["fps"].append(rate if rate > 0 else None)
    return data


def fixed_prior(n_frames: int, midpoint_frac: float = 0.6, steepness: float = 12.0) -> np.ndarray:
    """
    The control that matters: a sigmoid rising through the clip, identical for
    every input, computed without looking at a single pixel.

    If this scores comparably to the ensemble, the metric is measuring curve
    shape rather than anticipation — which is exactly what the paper's own
    ablation hints at, and exactly what a referee will test first.
    """
    t = np.linspace(0.0, 1.0, n_frames)
    return 1.0 / (1.0 + np.exp(-steepness * (t - midpoint_frac)))


def build_variants(data: dict, cfg: dict) -> dict[str, list[np.ndarray]]:
    """Construct every score trajectory variant to be compared."""
    w = cfg["ensemble"]
    ens = lambda c, f, n: w["clip_weight"] * c + w["flow_weight"] * f + w["nlp_weight"] * n

    variants = {
        "clip_only":    list(data["p_clip"]),
        "flow_only":    list(data["p_flow"]),
        "nlp_only":     list(data["p_nlp"]),
        "ensemble_raw": [ens(c, f, n) for c, f, n in
                         zip(data["p_clip"], data["p_flow"], data["p_nlp"])],
        "fixed_prior":  [fixed_prior(len(c)) for c in data["p_clip"]],
    }

    # post-processed ensemble, with and without the clamp
    pp_cfg = dict(cfg["postprocess"])
    pp = PostProcessor(pp_cfg)
    variants["ensemble_postproc"] = [pp.process(s) for s in variants["ensemble_raw"]]

    no_clamp = dict(pp_cfg)
    no_clamp["threshold"] = 2.0          # unreachable -> clamp never fires
    pp_nc = PostProcessor(no_clamp)
    variants["ensemble_postproc_noclamp"] = [pp_nc.process(s) for s in variants["ensemble_raw"]]

    return variants


def main():
    safe_console()
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    alpha = cfg.get("postprocess", {}).get("temporal_alpha", 1.0)
    if abs(alpha - 1.0) > 1e-9:
        print(f"\n  WARNING: temporal_alpha = {alpha}. This makes the pipeline non-causal:\n"
              f"  the value emitted at frame t is the score computed at frame {alpha}*t,\n"
              f"  a frame that has not happened yet. Set it to 1.0 before reporting\n"
              f"  anything you intend to publish.\n")

    data = load_features(Path(args.features))
    n_pos = sum(l == 1 for l in data["label"])
    n_neg = len(data["label"]) - n_pos

    FPS = resolve_fps(data["fps"], args.fps)
    if isinstance(FPS, list):
        import statistics
        print(f"Loaded {len(data['label'])} clips ({n_pos} positive, {n_neg} negative)")
        print(f"  per-clip frame rates: {min(FPS):.2f}-{max(FPS):.2f}, "
              f"median {statistics.median(FPS):.2f}, {len(set(FPS))} distinct")
    else:
        print(f"Loaded {len(data['label'])} clips ({n_pos} positive, {n_neg} negative) @ {FPS} fps")

    if n_neg == 0:
        raise SystemExit(
            "\nNo negative clips found. AP, AUC and false-positive rate are undefined on\n"
            "accident-only data: a model that always outputs a rising curve scores\n"
            "perfectly. Add normal driving clips to the manifest and re-extract.\n"
        )

    # Negatives get a decision window drawn from the positive onset
    # distribution, so both classes are observed for a comparable number of
    # frames. See matched_windows() for why the alternative is not neutral.
    win = matched_windows(data["label"], data["onset"],
                          [len(c) for c in data["p_clip"]], seed=0)

    variants = build_variants(data, cfg)

    print(f"\n{'variant':<28} {'AP':>8} {'AUC':>8} {'mTTA':>9} {'TTA@R80':>9}")
    print("-" * 66)

    results = {}
    for name, scores in variants.items():
        m = evaluate_anticipation(scores, data["label"], data["onset"], fps=FPS, windows=win)
        results[name] = m.to_dict()
        print(f"{name:<28} {m.ap:>8.4f} {m.auc:>8.4f} "
              f"{m.mtta:>8.3f}s {m.tta_at_r80:>8.3f}s")

        if args.bootstrap:
            lo, hi = bootstrap_ci(scores, data["label"], data["onset"], fps=FPS,
                                  metric="ap", windows=win)
            results[name]["ap_ci95"] = [lo, hi]
            print(f"{'':<28} {'AP 95% CI':>8} [{lo:.4f}, {hi:.4f}]")

    # ---- the decisive comparisons ----
    print("\nPaired comparisons (positive mean_diff favours the first method)")
    print("-" * 66)
    comparisons = [
        ("ensemble_postproc", "fixed_prior"),
        ("ensemble_postproc", "nlp_only"),
        ("ensemble_postproc", "clip_only"),
        ("ensemble_postproc", "ensemble_postproc_noclamp"),
    ]
    results["comparisons"] = {}
    for a, b in comparisons:
        t = paired_bootstrap_test(
            variants[a], variants[b], data["label"], data["onset"],
            fps=FPS, metric="ap", windows=win,
        )
        results["comparisons"][f"{a}_vs_{b}"] = t
        verdict = "significant" if (t["ci_low"] > 0 or t["ci_high"] < 0) else "NOT significant"
        print(f"{a} vs {b}")
        print(f"    dAP = {t['mean_diff']:+.4f}  95% CI [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]  -> {verdict}")

    ens_ap = results["ensemble_postproc"]["ap"]
    prior_ap = results["fixed_prior"]["ap"]
    print("\n" + "=" * 66)
    if ens_ap <= prior_ap:
        print("  The ensemble does not beat a curve that never looks at the video.")
        print("  Do not submit the current framing. Switch to the critique paper:")
        print("  report this as a finding about the evaluation protocol, not a defeat.")
    else:
        print(f"  Ensemble beats the video-blind prior by {ens_ap - prior_ap:+.4f} AP.")
        print("  Check the paired CI above before claiming it.")
    print("=" * 66)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "fps": FPS, "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
