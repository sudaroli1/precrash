"""
Build a leaderboard submission, so that metrics are computed by the organisers
rather than by us.

WHY THIS MATTERS MORE THAN IT LOOKS
-----------------------------------
zero-shot-taa publishes no labels, so an entrant cannot compute average
precision, area under the ROC curve, or a time-to-accident measured to the
annotated onset. Development therefore happens against some locally computable
proxy -- a crossing frame, a time-to-accident measured to the end of the window
-- and a proxy is not the metric. Ours shared the official metrics' NAMES while
differing in their definitions, which is the worst case: the numbers look
comparable and are not.

Late submission is open. That makes the divergence measurable instead of
arguable, and it means every row of a results table can carry a score computed
by someone else against ground truth we never see. A referee cannot ask whether
we implemented the metric correctly, because we did not implement it.

WHAT TO SUBMIT, AND IN WHAT ORDER
---------------------------------
Submission quotas are limited, so spend them in this order.

  1. --replicate_sample
     Regenerates the organisers' own reference submission: a linear ramp from
     0.001 to 0.999, identical for every clip. Its score is already known --
     0.58333 on the public leaderboard. If our file does not reproduce that
     exactly, something is wrong with the id matching or the formatting, and we
     find out before spending a quota slot on a result we care about. This is
     the control for the control.

  2. --curve constant_0.51
     The degenerate case. It maximises every locally computable proxy: crossing
     frame 0, the largest possible time-to-accident-to-window-end, and perfect
     "stability" since a constant never falls. What it scores officially is the
     number the paper is about.

  3. The rest of the video-blind family, establishing the floor empirically
     rather than by assertion.

  4. --features, for the real model variants and the ablation.

FORMAT
------
Mirrored from sample_submission.csv rather than assumed: same column order, same
separator, same bracket style. The competition names two failure modes -- an id
mismatch against test.csv, and a risk array whose length is not exactly 150 --
and both are checked here before anything is written.

USAGE
-----
  python scripts/make_submission.py --test_csv data/test.csv \
      --sample_csv data/sample_submission.csv \
      --replicate_sample --out submissions/00_replicate_sample.csv

  python scripts/make_submission.py --test_csv data/test.csv \
      --sample_csv data/sample_submission.csv \
      --curve constant_0.51 --out submissions/01_constant.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.console import safe_console  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_taa import blind_family, parse_risk  # noqa: E402

N_FRAMES = 150          # the competition rejects anything else
CLIP_LO, CLIP_HI = 0.0, 1.0


def parse_args():
    p = argparse.ArgumentParser(description="Build a leaderboard submission")
    p.add_argument("--test_csv", required=True)
    p.add_argument("--sample_csv", required=True,
                   help="sample_submission.csv, used to mirror the exact format")
    p.add_argument("--out", required=True)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--replicate_sample", action="store_true",
                     help="Regenerate the organisers' reference ramp. Its score "
                          "is known (0.58333), so this validates the pipeline.")
    src.add_argument("--curve", choices=sorted(blind_family(N_FRAMES)),
                     help="One member of the video-blind family")
    src.add_argument("--features", help="Directory of per-clip .npz feature files")

    p.add_argument("--variant", default="ensemble_postproc",
                   help="With --features: which score trajectory to submit")
    p.add_argument("--config", default="configs/honest.yaml")
    return p.parse_args()


def detect_format(sample_path: Path) -> dict:
    """Read the reference submission's formatting rather than guessing it."""
    rows = list(csv.DictReader(open(sample_path, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{sample_path} is empty")
    raw = rows[0]["risk"]
    bracketed = raw.strip().startswith("[")
    sep = ", " if ", " in raw else ","
    values = parse_risk(raw)
    style = calibrate_style(raw, values, sep, bracketed)
    return {
        "fieldnames": list(rows[0].keys()),
        "bracketed": bracketed,
        "sep": sep,
        "n_values": len(values),
        "ids": [r["id"] for r in rows],
        "style": style,
        "sample_values": values,
        "exact": style != "f6_strip" or format_risk(values, {
            "style": "f6_strip", "sep": sep, "bracketed": bracketed}) == raw.strip(),
    }


# Candidate ways a float might have been written into the reference file. The
# right one is not guessable -- it depends on whether the organisers used
# pandas, numpy, repr(), or an f-string, and with what precision -- so it is
# recovered by testing each against their own file rather than assumed.
FLOAT_STYLES = {
    "repr": repr,
    "str": str,
    "g": lambda v: f"{v:g}",
    "r17": lambda v: f"{v:.17g}",
    **{f"f{n}": (lambda n: lambda v: f"{v:.{n}f}")(n) for n in range(1, 11)},
    **{f"f{n}_strip": (lambda n: lambda v: f"{v:.{n}f}".rstrip("0").rstrip("."))(n)
       for n in range(1, 11)},
}


def calibrate_style(sample_raw: str, values: np.ndarray, sep: str,
                    bracketed: bool) -> str:
    """Find the float style that reproduces the reference string exactly.

    Falls back to a reasonable default when none matches -- the scorer parses
    floats, so trailing digits almost certainly do not affect the score. But an
    exact match turns --replicate_sample into a real check of the id order and
    the whole write path, at the cost of no submission slot.
    """
    for name, fn in FLOAT_STYLES.items():
        body = sep.join(fn(float(v)) for v in values)
        cand = f"[{body}]" if bracketed else body
        if cand == sample_raw.strip():
            return name
    return "f6_strip"


def format_risk(values: np.ndarray, fmt: dict) -> str:
    fn = FLOAT_STYLES[fmt["style"]]
    # float(v), not v. repr() of a numpy scalar is "np.float64(0.51)" under
    # numpy 2, which triples the file size and would be rejected by the scorer.
    # The style functions are calibrated on Python floats and must be applied to
    # Python floats.
    body = fmt["sep"].join(fn(float(v)) for v in values)
    return f"[{body}]" if fmt["bracketed"] else body


def load_feature_curves(feature_dir: Path, variant: str, config: str) -> dict:
    """Per-clip trajectories for a model variant, from the cached features."""
    import yaml
    from postprocess import PostProcessor

    cfg = yaml.safe_load(open(config))
    w = cfg["ensemble"]
    pp = PostProcessor(dict(cfg["postprocess"]))

    out = {}
    for f in sorted(Path(feature_dir).glob("*.npz")):
        z = np.load(f)
        c, fl, n = (z["p_clip"].astype(float), z["p_flow"].astype(float),
                    z["p_nlp"].astype(float))
        ens = w["clip_weight"] * c + w["flow_weight"] * fl + w["nlp_weight"] * n
        curves = {"clip_only": c, "flow_only": fl, "prior_only": n,
                  "ensemble_raw": ens, "ensemble_postproc": pp.process(ens)}
        if variant not in curves:
            raise SystemExit(f"--variant must be one of {sorted(curves)}")
        out[f.stem] = curves[variant]
    if not out:
        raise SystemExit(f"No .npz files in {feature_dir}")
    return out


def main():
    safe_console()
    args = parse_args()

    test_ids = [r["id"] for r in csv.DictReader(open(args.test_csv, encoding="utf-8"))
                if r.get("id") and r["id"] != "NOTE"]
    fmt = detect_format(Path(args.sample_csv))

    print(f"test.csv           : {len(test_ids)} ids")
    print(f"sample_submission  : {len(fmt['ids'])} ids, {fmt['n_values']} values, "
          f"bracketed={fmt['bracketed']}, sep={fmt['sep']!r}")
    print(f"float style        : {fmt['style']}"
          f"{'' if fmt['exact'] else '  (no exact match; using a default)'}")

    if set(test_ids) != set(fmt["ids"]):
        print("  note: sample and test id sets differ; test.csv is authoritative")

    # ---- build the curves ------------------------------------------------
    if args.replicate_sample:
        # Their own values, not a freshly computed linspace. The point of this
        # mode is to reproduce their file exactly, and a recomputed ramp differs
        # in the trailing digits -- the reference file holds values already
        # rounded to six places, so recomputing at full precision writes
        # 0.007697986577181208 where they wrote 0.007698.
        ramp = fmt["sample_values"]
        curves = {i: ramp for i in test_ids}
        label = "replicate_sample (known score: 0.58333)"
    elif args.curve:
        c = blind_family(fmt["n_values"])[args.curve]
        curves = {i: c for i in test_ids}
        label = f"video-blind: {args.curve}"
    else:
        got = load_feature_curves(Path(args.features), args.variant, args.config)
        missing = [i for i in test_ids if i not in got]
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(test_ids)} clips have no features, e.g. "
                f"{missing[:3]}. A submission must cover every id in test.csv; "
                "padding the gaps with a constant would silently mix a real "
                "method with a blind one."
            )
        curves = got
        label = f"{args.variant} from {args.features}"

    # ---- validate before writing ----------------------------------------
    problems = []
    for i in test_ids:
        v = np.asarray(curves[i], dtype=float)
        if len(v) != N_FRAMES:
            problems.append(f"{i}: {len(v)} values, need {N_FRAMES}")
        if not np.all(np.isfinite(v)):
            problems.append(f"{i}: contains NaN or inf")
        if v.min() < CLIP_LO - 1e-9 or v.max() > CLIP_HI + 1e-9:
            problems.append(f"{i}: values outside [0, 1] ({v.min():.3f}-{v.max():.3f})")
    if problems:
        for p in problems[:8]:
            print("  PROBLEM:", p)
        raise SystemExit(f"{len(problems)} clips would be rejected; nothing written")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fmt["fieldnames"])
        w.writeheader()
        for i in test_ids:
            w.writerow({"id": i,
                        "risk": format_risk(np.asarray(curves[i], float), fmt)})

    if args.replicate_sample:
        # Better than spending a submission slot to check the format: compare
        # the file we just wrote against the organisers' own, row by row. If
        # every risk string matches, the id order, the separator, the bracket
        # style and the value formatting are all confirmed, and the known
        # 0.58333 tells us what the scorer does with it.
        theirs = {r["id"]: r["risk"] for r in
                  csv.DictReader(open(args.sample_csv, encoding="utf-8"))}
        ours = {r["id"]: r["risk"] for r in
                csv.DictReader(open(out, encoding="utf-8"))}
        same_ids = set(theirs) == set(ours)
        exact = sum(1 for i in ours if theirs.get(i) == ours[i])
        print(f"\n  format check against sample_submission.csv")
        print(f"    id sets identical : {same_ids}")
        print(f"    risk strings exact: {exact} of {len(ours)}")
        if same_ids and exact == len(ours):
            print("    -> byte-identical. Format confirmed without using a "
                  "submission slot; this file would score 0.58333.")
        else:
            mism = next((i for i in ours if theirs.get(i) != ours[i]), None)
            if mism:
                print(f"    -> differs, e.g. {mism}")
                print(f"       theirs: {theirs[mism][:70]}")
                print(f"       ours  : {ours[mism][:70]}")

    first = np.asarray(curves[test_ids[0]], float)
    tc = np.where(first >= 0.5)[0]
    print(f"\nwrote {out}")
    print(f"  contents      : {label}")
    print(f"  rows          : {len(test_ids)}")
    print(f"  identical     : {len({id(curves[i]) for i in test_ids}) == 1}")
    print(f"  first curve   : {first[0]:.4f} .. {first[-1]:.4f}, "
          f"crossing frame {tc[0] if len(tc) else 'never'}")
    print(f"  size          : {out.stat().st_size / 1e6:.1f} MB")
    print("\nUpload it, then record the score in submissions/LOG.md.")


if __name__ == "__main__":
    main()
