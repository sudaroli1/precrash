"""
Build the evaluation manifest for the zero-shot-taa corpus — and report what
the corpus actually contains rather than assuming it.

WHAT THIS CORPUS PROVIDES, AND WHAT IT DOES NOT
-----------------------------------------------
`test.csv` carries five columns:

    id, video_id, start_frame, end_frame, caption

There is **no label, no event time and no alert time**. `train.csv` is a single
row stating that there is no train split. Ground truth is held by the
organisers and exposed only through the leaderboard.

The manifest this script writes therefore has **no `label` and no `onset_frame`
column**, and that is deliberate. Downstream, `evaluate.py` and
`metrics.evaluate_anticipation` refuse to run without them. That refusal is
correct and it is the point: average precision, area under the ROC curve, a
false-positive rate and a time-to-accident measured to an onset cannot be
computed on this corpus by anyone outside the competition. A manifest that
invented a label column would hide that.

Only the legacy curve-shape statistics are computable here. Use
`protocol_comparison.py --legacy_only`.

WHAT THE SCRIPT RESOLVES
------------------------
`start_frame` and `end_frame` in `test.csv` index the ORIGINAL source video.
The distributed clip directory may hold either

  (a) exactly the frames of that window, in which case the window is already
      applied and the reader should take the directory whole; or
  (b) a longer sequence, in which case the window must be located inside it.

Which it is, is a property of the distribution, not something to guess. This
script measures it per clip and writes `dir_start`/`dir_end` — positions in the
directory — alongside the original `start_frame`/`end_frame`, so the reader
never has to re-derive the mapping.

FRAME RATE
----------
A folder of JPGs carries no frame rate, and nothing in the distribution states
one. `fps` is written empty. Any conversion to seconds is an assumption by
whoever reports the number and must be declared with `--assume_fps`, which
records the value in the manifest so it appears in the provenance rather than
hiding in a default.

USAGE
-----
  python scripts/prepare_taa.py \
      --csv data/test.csv \
      --root /path/to/Test \
      --out data/taa_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.console import safe_console  # noqa: E402
from utils.frame_io import describe_clip  # noqa: E402

FIELDS = [
    "clip_id", "category", "video_id", "frame_dir",
    "n_frames_on_disk", "src_start_frame", "src_end_frame",
    "dir_start", "dir_end", "window_span",
    "fps", "caption",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build the zero-shot-taa manifest and describe the corpus")
    p.add_argument("--csv", required=True, help="test.csv from the competition")
    p.add_argument("--root", required=True,
                   help="The Test/ directory holding <type>/<id>/images/")
    p.add_argument("--out", default="data/taa_manifest.csv")
    p.add_argument("--assume_fps", type=float, default=None,
                   help="Record an assumed frame rate. The corpus states none; "
                        "supplying one is a reporting decision and is written "
                        "into the manifest so it cannot be lost.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def find_frame_dir(root: Path, category: str, video_num: str) -> Path | None:
    """Locate a clip's frame directory, tolerating the layout variations these
    archives tend to have."""
    candidates = [
        root / category / video_num / "images",
        root / category / video_num,
        root / f"{category}_{video_num}" / "images",
    ]
    for c in candidates:
        if c.is_dir() and any(
            p.suffix.lower() in (".jpg", ".jpeg", ".png") for p in c.iterdir()
        ):
            return c
    return None


def main():
    safe_console()
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")

    rows = list(csv.DictReader(open(args.csv)))
    rows = [r for r in rows if r.get("id") and r["id"] != "NOTE"]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{args.csv}: {len(rows)} clips\n")

    out_rows, missing = [], []
    span_counts, disk_counts, shape_counts = Counter(), Counter(), Counter()
    already_windowed = longer_sequence = 0

    for r in rows:
        clip_id = r["id"]
        # id is <type>_<videoid>_<start>_<end>; video_id is <type>/<videoid>
        video_id = r.get("video_id", "")
        if "/" in video_id:
            category, video_num = video_id.split("/", 1)
        else:
            parts = clip_id.split("_")
            category, video_num = parts[0], parts[1]

        src_start, src_end = int(r["start_frame"]), int(r["end_frame"])
        span = src_end - src_start
        span_counts[span] += 1

        fdir = find_frame_dir(root, category, video_num)
        if fdir is None:
            missing.append(clip_id)
            continue

        info = describe_clip(fdir)
        n_disk = info["n_frames"]
        disk_counts[n_disk] += 1
        shape_counts[(info["height"], info["width"])] += 1

        # Is the directory the window, or the whole source sequence?
        if n_disk == span:
            dir_start, dir_end = 0, n_disk
            already_windowed += 1
        elif info["index_min"] is not None and n_disk > span:
            # frame filenames carry source indices; locate the window in them
            offset = info["index_min"]
            dir_start = max(0, src_start - offset)
            dir_end = min(n_disk, src_end - offset)
            if dir_end - dir_start < 1:          # numbering did not line up
                dir_start, dir_end = 0, n_disk
            longer_sequence += 1
        else:
            dir_start, dir_end = 0, n_disk

        out_rows.append({
            "clip_id": clip_id,
            "category": category,
            "video_id": video_id,
            "frame_dir": str(fdir),
            "n_frames_on_disk": n_disk,
            "src_start_frame": src_start,
            "src_end_frame": src_end,
            "dir_start": dir_start,
            "dir_end": dir_end,
            "window_span": span,
            "fps": "" if args.assume_fps is None else f"{args.assume_fps}",
            "caption": r.get("caption", ""),
        })

    # ---------------- report ----------------
    print("=" * 66)
    print("CORPUS STRUCTURE, as measured")
    print("=" * 66)
    print(f"  clips located          : {len(out_rows)} of {len(rows)}")
    if missing:
        print(f"  MISSING frame dirs     : {len(missing)}  e.g. {missing[:5]}")
    print(f"  window span in test.csv: {dict(span_counts.most_common(5))}")
    print(f"  frames on disk         : {dict(disk_counts.most_common(5))}")
    print(f"  frame size (h, w)      : {dict(shape_counts.most_common(3))}")
    print(f"  directory == window    : {already_windowed}")
    print(f"  directory  > window    : {longer_sequence}")

    print("\n" + "=" * 66)
    print("WHAT THIS CORPUS DOES NOT PROVIDE")
    print("=" * 66)
    print("  no label column      -> AP, AUC and false-positive rate are undefined")
    print("  no event time        -> time-to-accident has no reference point")
    print("  no alert time        -> no annotated ceiling to check a claim against")
    print("  no frame rate        -> frames cannot be converted to seconds without")
    print("                          an assumption, which must be stated")
    if args.assume_fps is None:
        print("\n  fps written empty. Pass --assume_fps to record one explicitly.")
    else:
        print(f"\n  fps recorded as {args.assume_fps} BY ASSUMPTION, not from the data.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out}  ({len(out_rows)} clips, no label column by design)")
    print("Next: scripts/extract_features.py --manifest "
          f"{out} --config configs/honest.yaml")


if __name__ == "__main__":
    main()
