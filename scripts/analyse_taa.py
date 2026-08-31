"""
What the zero-shot-taa benchmark permits an entrant to measure.

Every number this prints comes from two files the competition publishes —
`test.csv` and `sample_submission.csv` — totalling about 2 MB. No frames, no
GPU, no model. That is the point: the argument the paper makes about this
benchmark can be checked by anyone in under a minute, without downloading the
corpus and without running anything.

It establishes four things.

1. WHAT THE CORPUS DOES NOT PUBLISH. `test.csv` carries id, video_id,
   start_frame, end_frame and caption. There is no label, no event time and no
   alert time, and `train.csv` is a single row saying there is no train split.
   Average precision, area under the ROC curve, a false-positive rate and a
   time-to-accident measured to a collision onset are therefore uncomputable by
   any entrant. Only statistics of the predicted curve's shape remain.

2. WHAT THE REFERENCE SUBMISSION IS. The organisers ship
   `sample_submission.csv` as the example entry. This script measures whether
   that curve is identical across clips and how far it departs from a straight
   line.

3. WHAT A CURVE-SHAPE STATISTIC CANNOT SEPARATE. Given only the legacy
   definition -- crossing frame = first index at which risk reaches 0.5 -- the
   crossing frame of a video-blind family is fixed by arithmetic. Pass
   --reported_crossover to place a previously reported figure alongside them.

4. WHAT THE CAPTIONS ARE. They describe the outcome, and there are far fewer of
   them than there are clips. A method conditioned on the caption is being told
   what happens.

USAGE
-----
  python scripts/analyse_taa.py \
      --test_csv data/test.csv \
      --submission_csv data/sample_submission.csv \
      --reported_crossover 22.7 \
      --out results/taa_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.console import safe_console  # noqa: E402

# The columns a corrected protocol needs, and what each one buys.
REQUIRED_FOR = {
    "label":      "average precision, area under the ROC curve, false-positive rate",
    "event_time": "time-to-accident measured to the collision",
    "alert_time": "a ceiling to check a reported warning time against",
}
# Names those columns plausibly go by, so absence is a finding and not a typo.
ALIASES = {
    "label":      ("label", "target", "y", "class", "is_accident", "accident"),
    "event_time": ("event_time", "time_of_event", "onset", "onset_frame",
                   "accident_frame", "collision_time"),
    "alert_time": ("alert_time", "time_of_alert", "alert_frame"),
}

THRESHOLD = 0.5


def parse_args():
    p = argparse.ArgumentParser(
        description="What zero-shot-taa lets an entrant measure")
    p.add_argument("--test_csv", required=True)
    p.add_argument("--submission_csv", default=None,
                   help="sample_submission.csv; the organisers' reference entry")
    p.add_argument("--train_csv", default=None,
                   help="train.csv, if you want its note quoted in the output")
    p.add_argument("--reported_crossover", type=float, default=None,
                   help="A previously reported mean crossing frame, placed "
                        "alongside the video-blind family for comparison.")
    p.add_argument("--assume_fps", type=float, default=30.0,
                   help="Only used to express crossing frames in seconds. The "
                        "corpus states no frame rate; this is an assumption and "
                        "is labelled as one wherever it appears.")
    p.add_argument("--out", default="results/taa_analysis.json")
    return p.parse_args()


def blind_family(n_frames: int) -> dict[str, np.ndarray]:
    """Risk trajectories that are functions of the frame count alone.

    A family rather than one curve: a single flattering sigmoid invites the
    reply that the strawman was tuned, whereas a sweep of shapes and midpoints
    shows the effect belongs to the metric.
    """
    t = np.linspace(0.0, 1.0, n_frames)
    out = {f"sigmoid_m{m:.2f}": 1.0 / (1.0 + np.exp(-12.0 * (t - m)))
           for m in (0.40, 0.50, 0.60, 0.70)}
    out["linear_ramp"] = t.copy()
    out["step_at_half"] = (t >= 0.5).astype(float) * 0.99 + 0.005
    out["constant_0.51"] = np.full(n_frames, 0.51)
    return out


def probe_family(n_frames: int) -> dict[str, np.ndarray]:
    """Curves designed to DECOMPOSE the official score, not merely to score well.

    The official score is

        w_AP*AP + w_AUC*AUC + w_TTA*TTA@0.5 + w_STTA*STTA@0.5

    with the weights undisclosed. Every curve below is constant across clips, so
    each scores base-rate AP and chance AUC -- those two terms contribute the
    same amount to all of them and cancel in any difference. What differs is
    what the two timing terms see.

    TTA@0.5 = max{t_ai - t_a | p_t > 0.5}, where t_a is the FIRST frame above
    threshold. STTA@0.5 additionally requires p to stay above 0.5 continuously
    from t_a' through t_ai.

    THE THREE THAT MATTER
    ---------------------
      never_crosses     0.49 throughout. Never above threshold, so both timing
                        terms are zero. This measures the AP+AUC floor alone.

      cross_then_drop   0.51 at frame 0, then 0.49 forever. t_a = 0, so TTA is
                        the maximum the clip admits -- but the score never stays
                        above threshold, so STTA collapses. This adds exactly
                        the TTA term to the floor.

      constant_0.51     Above threshold everywhere. Both timing terms saturate.

    Subtracting in sequence isolates each weight-times-mean-onset product:
    (cross_then_drop - never_crosses) is the TTA contribution, and
    (constant_0.51 - cross_then_drop) is the STTA contribution. Three
    submissions, and the composite comes apart.

    THE SWEEP
    ---------
      step_at_K         0.49 before frame K, 0.51 from K onward. If the timing
                        terms are linear in the crossing frame, these fall on a
                        straight line, and its slope is the same quantity the
                        differences above give -- an independent check.
    """
    out = {
        "never_crosses": np.full(n_frames, 0.49),
        "constant_0.51": np.full(n_frames, 0.51),
        "constant_0.99": np.full(n_frames, 0.99),
    }

    drop = np.full(n_frames, 0.49)
    drop[0] = 0.51
    out["cross_then_drop"] = drop

    # crosses at 0, dips for ten frames in the middle, recovers: TTA unchanged,
    # STTA pushed back to the end of the dip
    dip = np.full(n_frames, 0.51)
    dip[n_frames // 3: n_frames // 3 + 10] = 0.49
    out["cross_then_dip"] = dip

    for k in (0, 10, 25, 50, 75, 100, 125, 140):
        c = np.full(n_frames, 0.49)
        c[k:] = 0.51
        out[f"step_at_{k:03d}"] = c

    return out


def parse_risk(text: str) -> np.ndarray:
    """Parse a submission's risk field into an array.

    Deliberately not ast.literal_eval: that demands one exact spelling, and a
    file written by pandas, numpy or a hand-rolled join differs in whitespace,
    in trailing commas and in whether values carry a type wrapper. Pull the
    numbers out and ignore the packaging.
    """
    # The lookbehind skips digits that belong to an identifier -- the 64 in
    # "np.float64(...)" is not a value -- while still admitting scientific
    # notation, where the exponent's digits follow an 'e' that IS part of the
    # number. Stripping identifiers wholesale would eat the 'e' in 1e-3.
    nums = re.findall(r"(?<![A-Za-z_0-9.])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not nums:
        raise ValueError(f"No numbers in risk field: {text[:60]!r}")
    return np.asarray([float(x) for x in nums], dtype=float)


def crossing_frame(curve: np.ndarray, threshold: float = THRESHOLD):
    hits = np.where(curve >= threshold)[0]
    return int(hits[0]) if len(hits) else None


def main():
    safe_console()
    args = parse_args()
    report: dict = {}

    rows = [r for r in csv.DictReader(open(args.test_csv, encoding="utf-8"))
            if r.get("id") and r["id"] != "NOTE"]
    cols = list(rows[0].keys()) if rows else []
    print(f"{args.test_csv}: {len(rows)} clips")
    print(f"columns: {cols}\n")

    # ---- 1. what is not published -------------------------------------
    print("=" * 70)
    print("WHAT THE CORPUS DOES NOT PUBLISH")
    print("=" * 70)
    lower = {c.lower() for c in cols}
    absent = {}
    for field, names in ALIASES.items():
        found = next((n for n in names if n in lower), None)
        absent[field] = found is None
        mark = f"present as '{found}'" if found else "ABSENT"
        print(f"  {field:<12} {mark:<22} -> {REQUIRED_FOR[field]}")
    report["columns"] = cols
    report["absent"] = absent

    if args.train_csv:
        tr = list(csv.DictReader(open(args.train_csv, encoding="utf-8")))
        print(f"\n  train.csv: {len(tr)} row(s)")
        for r in tr[:2]:
            note = r.get("caption") or ""
            if note:
                print(f"    note: {note}")
                report["train_note"] = note

    # ---- 2. windows ---------------------------------------------------
    spans = Counter(int(r["end_frame"]) - int(r["start_frame"]) for r in rows)
    n_frames = spans.most_common(1)[0][0]
    print(f"\n  window spans: {dict(spans.most_common(5))}")
    report["window_spans"] = {str(k): v for k, v in spans.items()}

    # ---- 3. the reference submission ----------------------------------
    if args.submission_csv:
        sub = list(csv.DictReader(open(args.submission_csv, encoding="utf-8")))
        curves = {s["risk"] for s in sub}
        first = parse_risk(sub[0]["risk"])
        line = np.linspace(first[0], first[-1], len(first))
        max_dev = float(np.max(np.abs(first - line)))

        print("\n" + "=" * 70)
        print("THE ORGANISERS' REFERENCE SUBMISSION")
        print("=" * 70)
        print(f"  entries              : {len(sub)}")
        print(f"  length               : {len(first)}")
        print(f"  first, last          : {first[0]:.6f}, {first[-1]:.6f}")
        print(f"  identical every clip : {len(curves) == 1}")
        print(f"  max deviation from a straight line: {max_dev:.2e}")
        if len(curves) == 1:
            print("\n  One curve, repeated for every clip. It reads no pixels.")
        report["sample_submission"] = {
            "n": len(sub), "length": len(first),
            "first": float(first[0]), "last": float(first[-1]),
            "identical_across_clips": len(curves) == 1,
            "max_deviation_from_linear": max_dev,
        }
        n_frames = len(first)

    # ---- 4. what a curve-shape statistic cannot separate ---------------
    print("\n" + "=" * 70)
    print("CROSSING FRAME OF A VIDEO-BLIND FAMILY")
    print(f"(first index at which risk reaches {THRESHOLD}; {n_frames}-frame window)")
    print("=" * 70)
    fam = blind_family(n_frames)
    print(f"  {'curve':<20}{'t_c':>7}{'TTA* (s)':>11}    reads pixels")
    print("  " + "-" * 52)
    blind = {}
    # `or 1e9` would be wrong here: a crossing frame of 0 is falsy, and the
    # constant curve -- the one that crosses at frame 0 and is the whole point --
    # would sort last. Test for None explicitly.
    def _sort_key(kv):
        tc = crossing_frame(kv[1])
        return float("inf") if tc is None else tc

    for name, curve in sorted(fam.items(), key=_sort_key):
        tc = crossing_frame(curve)
        tta = (n_frames - tc) / args.assume_fps if tc is not None else float("nan")
        blind[name] = {"crossing_frame": tc, "tta_to_clip_end_s": tta}
        print(f"  {name:<20}{tc:>7}{tta:>11.2f}    no")

    if args.reported_crossover is not None:
        rc = args.reported_crossover
        rtta = (n_frames - rc) / args.assume_fps
        print(f"  {'REPORTED':<20}{rc:>7.1f}{rtta:>11.2f}    yes")
        beaten = [n for n, v in blind.items()
                  if v["crossing_frame"] is not None and v["crossing_frame"] < rc]
        print(f"\n  video-blind curves with an EARLIER crossing frame than the\n"
              f"  reported figure: {len(beaten)} of {len(blind)}")
        if beaten:
            print(f"    {', '.join(sorted(beaten))}")
        report["reported_crossover"] = rc
        report["blind_curves_beating_reported"] = sorted(beaten)
    report["blind_family"] = blind
    print(f"\n  TTA* is measured to the end of the window, as the legacy protocol\n"
          f"  defines it, at an ASSUMED {args.assume_fps} fps. The corpus states no\n"
          f"  frame rate.")

    # ---- 5. captions ---------------------------------------------------
    if "caption" in cols:
        caps = Counter(r["caption"] for r in rows)
        print("\n" + "=" * 70)
        print("CAPTIONS")
        print("=" * 70)
        print(f"  {len(rows)} clips, {len(caps)} distinct captions")
        top = caps.most_common(8)
        cover = sum(n for _, n in top) / len(rows)
        for c, n in top:
            print(f"    {n:5d}  {c[:62]}")
        print(f"\n  top 8 cover {cover:.0%} of the corpus")
        print("  These describe the outcome. A method conditioned on them is\n"
              "  being told what happens.")
        report["captions"] = {"n_distinct": len(caps),
                              "top": [{"caption": c, "n": n} for c, n in top]}

    cats = Counter(r["id"].split("_")[0] for r in rows)
    print(f"\n  {len(cats)} categories; largest is '{cats.most_common(1)[0][0]}' "
          f"with {cats.most_common(1)[0][1]} clips "
          f"({cats.most_common(1)[0][1]/len(rows):.0%})")
    report["categories"] = {"n": len(cats), "top": dict(cats.most_common(8))}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
