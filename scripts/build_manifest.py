"""
Step 1 of the repair: build the manifest that gives the evaluation a ground truth.

This is the file whose absence caused the desk rejections. Everything else in
the revision depends on it existing and being correct.

WHAT A MANIFEST IS
------------------
A CSV with one row per clip:

    clip_id,video,label,onset_frame
    000821,positive/000821.mp4,1,90
    001402,negative/001402.mp4,0,

    label       1 = the clip contains an accident, 0 = normal driving
    onset_frame the frame index at which the collision begins.
                Required for positives. Blank for negatives.

Two things matter and are easy to get wrong:

  1. You need negatives. Without normal-driving clips there is no false-positive
     rate, no AP and no AUC — and a model that always outputs a rising curve
     scores perfectly. This is the single biggest defect in the current paper.

  2. onset_frame must be the CRASH frame, not the last frame of the clip. The
     old code computed time-to-accident as (clip_length - alarm_frame)/fps,
     which is only correct if every crash happens on the final frame. It does
     not.

DATASET CONVENTIONS
-------------------
DAD (Chan et al., ACCV 2016) — github.com/smallcorgi/Anticipating-Accidents
    Positive clips are constructed so the accident occupies the final stretch
    of the clip, which makes a constant --onset a reasonable approximation.
    VERIFY the clip length, fps and accident frame against the repo's own
    README before trusting a constant, and record whatever you use in the paper.

CCD (Bao et al., ACM MM 2020) — github.com/Cogito2012/CarCrashDataset
    Ships per-clip accident-time annotations. Use --onset_csv, not a constant.

MM-AU — per-clip accident windows are in the annotation files, not uniform.
    A constant onset is wrong for MM-AU. Use --onset_csv.

USAGE
-----
Constant onset (DAD-style):

    python scripts/build_manifest.py \
        --positive_dir /content/DAD/videos/testing/positive \
        --negative_dir /content/DAD/videos/testing/negative \
        --onset 90 \
        --out data/dad_test.csv

Per-clip onsets (CCD, MM-AU):

    python scripts/build_manifest.py \
        --positive_dir /content/CCD/crash \
        --negative_dir /content/CCD/normal \
        --onset_csv /content/CCD/crash_annotations.csv \
        --out data/ccd_all.csv

Then split off a development set you are allowed to tune on:

    python scripts/build_manifest.py --split data/ccd_all.csv --dev_frac 0.3

This produces data/ccd_all.dev.csv and data/ccd_all.test.csv. Tune on dev.
Touch test exactly once, at the very end. That discipline is what turns the
paper's contradicted zero-shot claim into a defensible methods paragraph.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    p = argparse.ArgumentParser(description="Build or split an evaluation manifest")

    p.add_argument("--positive_dir", help="Directory of accident clips")
    p.add_argument("--negative_dir", help="Directory of normal driving clips")
    p.add_argument("--onset", type=int,
                   help="Constant accident onset frame for all positives (DAD-style)")
    p.add_argument("--onset_csv",
                   help="CSV mapping clip_id to onset frame. Two columns, header required: "
                        "clip_id,onset_frame. Use this whenever onsets vary per clip.")
    p.add_argument("--root", default=None,
                   help="Path stored in the manifest is relative to this. "
                        "Defaults to the common parent of the two directories.")
    p.add_argument("--out", help="Output CSV path")

    p.add_argument("--split", help="Existing manifest to split into dev/test")
    p.add_argument("--dev_frac", type=float, default=0.3, help="Fraction to dev (default 0.3)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def list_videos(d: Path) -> list[Path]:
    if not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")
    vids = sorted(p for p in d.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not vids:
        raise SystemExit(f"No video files found under {d} "
                         f"(looked for {', '.join(sorted(VIDEO_EXTS))})")
    return vids


def load_onset_map(path: Path) -> dict[str, int]:
    mapping = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "clip_id" not in reader.fieldnames or "onset_frame" not in reader.fieldnames:
            raise SystemExit(f"{path} must have columns: clip_id,onset_frame "
                             f"(found: {reader.fieldnames})")
        for row in reader:
            mapping[row["clip_id"].strip()] = int(row["onset_frame"])
    return mapping


def build(args):
    if not (args.positive_dir and args.negative_dir and args.out):
        raise SystemExit("--positive_dir, --negative_dir and --out are all required")
    if bool(args.onset) == bool(args.onset_csv):
        raise SystemExit("Provide exactly one of --onset (constant) or --onset_csv (per clip)")

    pos_dir, neg_dir = Path(args.positive_dir), Path(args.negative_dir)
    root = Path(args.root) if args.root else Path(
        __import__("os").path.commonpath([pos_dir.resolve(), neg_dir.resolve()])
    )

    onset_map = load_onset_map(Path(args.onset_csv)) if args.onset_csv else None

    rows, missing_onsets = [], []

    for v in list_videos(pos_dir):
        cid = v.stem
        if onset_map is not None:
            if cid not in onset_map:
                missing_onsets.append(cid)
                continue
            onset = onset_map[cid]
        else:
            onset = args.onset
        rows.append({"clip_id": cid, "video": str(v.resolve().relative_to(root)),
                     "label": 1, "onset_frame": onset})

    for v in list_videos(neg_dir):
        rows.append({"clip_id": v.stem, "video": str(v.resolve().relative_to(root)),
                     "label": 0, "onset_frame": ""})

    ids = [r["clip_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate clip_id values — positive and negative filenames collide. "
                         "Prefix them (pos_/neg_) so cached features cannot overwrite each other.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "video", "label", "onset_frame"])
        w.writeheader()
        w.writerows(rows)

    n_pos = sum(r["label"] == 1 for r in rows)
    n_neg = len(rows) - n_pos
    print(f"Wrote {out}")
    print(f"  {n_pos} positive, {n_neg} negative  (video paths relative to {root})")
    if missing_onsets:
        print(f"  SKIPPED {len(missing_onsets)} positives with no onset in the CSV: "
              f"{', '.join(missing_onsets[:5])}{' ...' if len(missing_onsets) > 5 else ''}")
    if n_neg == 0:
        print("\n  STOP. No negatives. AP, AUC and false-positive rate are undefined,\n"
              "  and this is the exact defect that got the paper rejected.\n")
    elif n_neg < n_pos * 0.5:
        print(f"\n  Note: only {n_neg} negatives for {n_pos} positives. Workable, but state\n"
              f"  the ratio explicitly in the paper — precision depends on it.\n")


def split(args):
    src = Path(args.split)
    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        # Preserve whatever columns the input carries. prepare_nexar.py adds
        # alert_frame, start_frame, end_frame and fps; a hardcoded field list
        # silently drops them, and without start_frame/end_frame the extraction
        # falls back to sampling across the whole clip -- the exact defect the
        # windowing exists to remove.
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    required = {"clip_id", "video", "label", "onset_frame"}
    missing = required - set(fields)
    if missing:
        raise SystemExit(f"{src} is missing required columns: {sorted(missing)}")

    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]

    rng = random.Random(args.seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_dev_pos = int(len(pos) * args.dev_frac)
    n_dev_neg = int(len(neg) * args.dev_frac)
    dev = pos[:n_dev_pos] + neg[:n_dev_neg]     # stratified: keeps the class ratio
    test = pos[n_dev_pos:] + neg[n_dev_neg:]
    rng.shuffle(dev)
    rng.shuffle(test)

    for name, subset in [("dev", dev), ("test", test)]:
        out = src.with_suffix(f".{name}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)
        p = sum(int(r["label"]) == 1 for r in subset)
        print(f"Wrote {out}  ({p} positive, {len(subset) - p} negative)")

    print(f"\nSeed {args.seed} recorded. Commit both files — a split you cannot")
    print("reproduce is a split a referee cannot trust.")
    print("Tune on dev. Run test once, at the end, and report whatever it gives you.")


def main():
    args = parse_args()
    if args.split:
        split(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
