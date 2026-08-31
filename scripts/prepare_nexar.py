"""
Turn the Kaggle Nexar collision-prediction training set into an evaluation
manifest with real labels, real onsets, and a controlled onset position.

WHY THIS SCRIPT EXISTS
----------------------
The original evaluation ran the pipeline over every clip in this dataset and
reported an anticipation time for each. Two things were wrong with that:

  1. Half the corpus is normal driving. The dataset is 750 positive
     (400 collisions + 350 near-collisions) and 750 negative. `target` was in
     train.csv the whole time; the evaluation never read it, so it computed a
     "crossover frame" for 750 clips in which nothing happens.

  2. Clips average 40 seconds and vary widely; frame rates are not 30 either
     (the first clip is 289/10 = 28.9 fps). load_frames() sampled 150 frames
     across the WHOLE clip, so one sampled frame is ~0.27 s of real time, and
     TTA was computed as (150 - t_c)/30 -- a count of sampled frames divided by
     a rate those frames do not have, measured to the end of the clip rather
     than to the collision. The result is dimensionally incoherent: it is
     neither the time to the clip end nor the time to the event. That is how a
     mean anticipation of 4.23 s was reported on a dataset whose annotators put
     the mean alert-to-event interval at 1.60 s and the maximum at 4.47 s.

This script fixes both. It cuts a fixed-length window at the video's NATIVE
frame rate, so one sampled frame is one real frame and seconds mean seconds.
Nothing is re-encoded: the window is recorded as start/end frame indices in the
manifest, and load_frames reads that range directly from the original file.

THE ONSET-POSITION EXPERIMENT
-----------------------------
--onset_mode fixed    every positive clip places the collision at the same
                      frame index. This reproduces how DAD and CCD are built.
--onset_mode random   the collision lands at a uniformly random position
                      within a band.

That flag is the controlled variable of the critique paper. A video-blind
template can only win when the onset sits at a predictable index; run both
conditions and the difference isolates benchmark construction from model
quality.

USAGE
-----
    python scripts/prepare_nexar.py \
        --csv /kaggle/input/nexar-collision-prediction/train.csv \
        --video_dir /kaggle/input/nexar-collision-prediction/train \
        --out data/nexar_fixed.csv --onset_mode fixed

    python scripts/prepare_nexar.py ... --out data/nexar_random.csv --onset_mode random

Then split, extract, evaluate:

    python scripts/build_manifest.py --split data/nexar_fixed.csv --dev_frac 0.3
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def parse_args():
    p = argparse.ArgumentParser(description="Build a Nexar manifest with windows and onsets")
    p.add_argument("--csv", required=True, help="train.csv with id,time_of_event,time_of_alert,target")
    p.add_argument("--video_dir", required=True, help="Directory holding the training videos")
    p.add_argument("--out", required=True, help="Output manifest CSV")

    p.add_argument("--window_frames", type=int, default=150,
                   help="Window length in NATIVE frames (default 150 ~ 5 s at 30 fps)")
    p.add_argument("--onset_mode", choices=["fixed", "random"], default="fixed",
                   help="fixed: onset at a constant index (DAD/CCD-style, template-gameable). "
                        "random: onset uniformly placed in a band (the control condition).")
    p.add_argument("--onset_pos", type=int, default=135,
                   help="Onset index within the window when --onset_mode fixed")
    p.add_argument("--onset_band", type=int, nargs=2, default=[75, 145],
                   help="Low and high onset index when --onset_mode random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Only process the first N rows")
    p.add_argument("--workers", type=int, default=12,
                   help="Parallel ffprobe workers (I/O bound; raise on fast disks)")
    return p.parse_args()


def find_video(video_dir: Path, clip_id: str) -> Path | None:
    """Kaggle ids may or may not be zero-padded in the filename. Try the
    obvious spellings before giving up."""
    candidates = [clip_id, clip_id.lstrip("0") or "0"]
    for width in (5, 6, 4):
        candidates.append(clip_id.zfill(width))
    for name in dict.fromkeys(candidates):
        for ext in VIDEO_EXTS:
            p = video_dir / f"{name}{ext}"
            if p.exists():
                return p
    hits = list(video_dir.glob(f"*{clip_id}*"))
    return hits[0] if hits else None


_HAVE_FFPROBE = subprocess.run(["which", "ffprobe"],
                               capture_output=True).returncode == 0


def _probe_cv2(path: Path) -> tuple[float, int]:
    """Fallback when ffmpeg is not installed (common on Windows).

    Slower than ffprobe because OpenCV may decode to count frames, but it
    avoids a dependency the user may not be able to install.
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        raise IOError(f"could not determine frame count for {path}")
    return float(fps), total


def probe(path: Path) -> tuple[float, int]:
    """Return (fps, total_frames) read from the container header.

    Uses ffprobe rather than cv2.VideoCapture. OpenCV's CAP_PROP_FRAME_COUNT
    decodes the stream for many MP4s, which means reading the whole file; across
    1,500 clips totalling 30 GB on a network volume that is hours. ffprobe reads
    the header only, in well under a second per file.

    Frame rate is NOT assumed to be 30. Measured rates in this dataset vary
    (289/10 = 28.9 fps in the first clip), and an assumed rate propagates
    directly into every reported time-to-accident.
    """
    if not _HAVE_FFPROBE:
        return _probe_cv2(path)

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,nb_frames,duration",
         "-of", "default=nw=1:nk=0", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        raise IOError(f"ffprobe failed on {path}")

    fields = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()

    rate = fields.get("r_frame_rate", "30/1")
    if "/" in rate:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    else:
        fps = float(rate or 30.0)

    nb = fields.get("nb_frames", "")
    if nb.isdigit() and int(nb) > 0:
        total = int(nb)
    else:
        # some containers omit nb_frames; fall back to duration x rate
        dur = fields.get("duration", "")
        total = int(float(dur) * fps) if dur not in ("", "N/A") else 0

    if total <= 0:
        raise IOError(f"could not determine frame count for {path}")
    return fps, total


def main():
    args = parse_args()
    if not _HAVE_FFPROBE:
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise SystemExit(
                "Needs either ffmpeg (for ffprobe) or opencv-python.\n"
                "  pip install opencv-python"
            )
        print("ffprobe not found; falling back to OpenCV (slower).", flush=True)

    rng = random.Random(args.seed)
    video_dir = Path(args.video_dir)
    L = args.window_frames
    lo_band, hi_band = args.onset_band

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    required = {"id", "target", "time_of_event", "time_of_alert"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"train.csv is missing columns: {sorted(missing)}")

    out_rows = []
    skipped = {"no_video": 0, "too_short": 0, "bad_onset": 0, "unreadable": 0}

    # ---- resolve paths, then probe in parallel -----------------------
    # Probing is I/O bound on a mounted volume, so threads help even though
    # ffprobe is a subprocess. Serial probing of 1,500 clips takes ~16 minutes;
    # with a pool it is a couple.
    resolved = []
    for r in rows:
        clip_id = str(r["id"]).strip()
        path = find_video(video_dir, clip_id)
        if path is None:
            skipped["no_video"] += 1
            continue
        resolved.append((r, clip_id, path))

    print(f"Resolved {len(resolved)} of {len(rows)} clips; probing metadata...",
          flush=True)

    meta: dict[str, tuple[float, int]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe, path): cid for _, cid, path in resolved}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                meta[cid] = fut.result()
            except Exception:
                skipped["unreadable"] += 1
            done += 1
            if done % 250 == 0:
                print(f"  probed {done}/{len(resolved)}", flush=True)

    print(f"Probed {len(meta)} clips.", flush=True)

    for r, clip_id, path in resolved:
        label = int(float(r["target"]))
        if clip_id not in meta:
            continue
        fps, total = meta[clip_id]
        if total < L:
            skipped["too_short"] += 1
            continue

        if label == 1:
            toe = (r["time_of_event"] or "").strip()
            if not toe:
                skipped["bad_onset"] += 1
                continue
            event_frame = int(round(float(toe) * fps))

            # place the collision at the chosen index inside the window
            onset_in_window = (args.onset_pos if args.onset_mode == "fixed"
                               else rng.randint(lo_band, hi_band))
            start = event_frame - onset_in_window
            # keep the window inside the video, adjusting the onset to match
            start = max(0, min(start, total - L))
            onset_in_window = event_frame - start
            if not (0 < onset_in_window < L):
                skipped["bad_onset"] += 1
                continue

            toa = (r["time_of_alert"] or "").strip()
            alert_in_window = ""
            if toa:
                a = int(round(float(toa) * fps)) - start
                if 0 <= a < L:
                    alert_in_window = a
        else:
            # negatives: a window drawn at random, independent of anything
            start = rng.randint(0, total - L)
            onset_in_window = ""
            alert_in_window = ""

        out_rows.append({
            "clip_id": clip_id,
            "video": str(path.relative_to(video_dir)),
            "label": label,
            "onset_frame": onset_in_window,
            "alert_frame": alert_in_window,
            "start_frame": start,
            "end_frame": start + L,
            "fps": round(fps, 4),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip_id", "video", "label", "onset_frame", "alert_frame",
              "start_frame", "end_frame", "fps"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    n_pos = sum(r["label"] == 1 for r in out_rows)
    n_neg = len(out_rows) - n_pos
    onsets = [r["onset_frame"] for r in out_rows if r["label"] == 1]
    alerts = [(r["onset_frame"] - r["alert_frame"]) / r["fps"]
              for r in out_rows if r["label"] == 1 and r["alert_frame"] != ""]

    print(f"\nWrote {out}")
    print(f"  {n_pos} positive, {n_neg} negative   (window {L} frames at native fps)")
    print(f"  onset mode: {args.onset_mode}"
          + (f" at index {args.onset_pos}" if args.onset_mode == "fixed"
             else f", band {lo_band}-{hi_band}"))
    if onsets:
        print(f"  onset index: min {min(onsets)}, max {max(onsets)}, "
              f"distinct values {len(set(onsets))}")
    if alerts:
        mean_a = sum(alerts) / len(alerts)
        print(f"\n  ANTICIPATION CEILING from the dataset's own annotators:")
        print(f"    mean alert-to-event {mean_a:.2f} s, max {max(alerts):.2f} s")
        print(f"    Any method reporting a mean warning time above {max(alerts):.2f} s")
        print(f"    on this data has a measurement error, not a result.")
    if any(skipped.values()):
        print(f"\n  skipped: {skipped}")

    if args.onset_mode == "fixed" and onsets and len(set(onsets)) < 5:
        print("\n  Note: onsets are near-constant across clips, which is exactly the\n"
              "  construction that lets a video-blind template score well. That is the\n"
              "  intended condition here — now build the random-onset counterpart and\n"
              "  compare.")


if __name__ == "__main__":
    main()
