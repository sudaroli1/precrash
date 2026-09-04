"""
Pass 1 of 2 — GPU feature extraction, run once per dataset.

WHY THIS EXISTS
---------------
The original evaluate_mmau.py ran the whole pipeline end-to-end per clip, which
means every ablation, every weight sweep and every post-processing variant
re-ran CLIP over every frame. On a free Colab T4 with session limits, that makes
the nine-step revision protocol impossible.

This script separates the expensive part (three neural engines over every frame)
from the cheap part (weights, smoothing, thresholds). It writes one small .npz
per clip holding the three raw modality curves. Everything downstream then runs
on CPU in seconds, as many times as you like.

It is resumable: clips already extracted are skipped, so a Colab disconnect
costs you nothing but the clip in flight.

USAGE
-----
  # unlabelled corpus, clips as folders of JPGs (zero-shot-taa)
  python scripts/extract_features.py \
      --manifest data/taa_manifest.csv \
      --out_dir features/taa \
      --config configs/honest.yaml

  # labelled corpus, clips as video files
  python scripts/extract_features.py \
      --manifest data/dad_test.csv \
      --video_root /path/to/videos \
      --out_dir features/dad_test \
      --config configs/honest.yaml

MANIFEST FORMATS (CSV, with header)
-----------------------------------
Two shapes, because the two corpora differ in what they can supply.

  UNLABELLED — written by prepare_taa.py
    clip_id,frame_dir,dir_start,dir_end,fps,caption
    1_009334_14_164,Test/1/009334/images,0,150,,a pedestrian crosses the road

    No label and no onset, because zero-shot-taa publishes neither. Extraction
    runs normally; AP, AUC, false-positive rate and time-to-accident-to-onset
    then cannot be computed from the cache by anyone without the ground truth.
    That is a fact about the benchmark, and this script says so rather than
    defaulting a label and hiding it.

  LABELLED
    clip_id,video,label,onset_frame
    000821,positive/000821.mp4,1,90
    001402,negative/001402.mp4,0,

    label       1 = accident clip, 0 = normal clip
    onset_frame frame index at which the collision begins; blank for negatives.

Do not fall back to globbing a directory for videos. That is how the original
evaluation ended up with no ground truth, no frame rate and no window.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.console import safe_console  # noqa: E402
from engines.clip_scorer import CLIPScorer
from engines.flow_scorer import FlowScorer
from engines.nlp_scorer import NLPScorer
from utils.frame_io import load_frames_from_dir
from utils.video_io import load_frames


def parse_args():
    p = argparse.ArgumentParser(description="Extract per-frame modality curves (GPU pass)")
    p.add_argument("--manifest", required=True, help="CSV with clip_id,video,label,onset_frame")
    # Accept both spellings. prepare_nexar.py takes --video_dir for the same
    # directory; requiring a different flag here is a needless trap.
    p.add_argument("--video_root", "--video_dir", dest="video_root", default=None,
                   help="Directory the manifest's video paths are relative to "
                        "(--video_dir accepted as an alias)")
    p.add_argument("--out_dir", required=True, help="Where to write per-clip .npz feature files")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--use_raft", action="store_true",
                   help="Use RAFT dense optical flow instead of the frame-difference proxy. "
                        "The '90%% of RAFT at 1%% compute' claim was withdrawn from the paper "
                        "because it was never measured. Running once with and once without is "
                        "how it would be measured, if anyone wants the number.")
    p.add_argument("--limit", type=int, default=None, help="Extract only the first N clips (smoke test)")
    p.add_argument("--clip_batch", type=int, default=32,
                   help="Frames per CLIP forward pass. Affects throughput, and moves "
                        "results by a few units in the last float32 place; record it "
                        "with any cache you intend to reproduce exactly.")
    return p.parse_args()


def read_manifest(path: Path) -> list[dict]:
    """Read either manifest shape.

    Two corpora are in play and they differ in what they can supply:

      LABELLED   clip_id, video, label, onset_frame [, start_frame, end_frame,
                 alert_frame, fps] — a corpus that publishes ground truth, so
                 the corrected protocol can be computed on it.

      UNLABELLED clip_id, frame_dir, dir_start, dir_end [, fps, caption] — the
                 zero-shot-taa shape. No label and no onset, because the corpus
                 publishes neither.

    A missing label is recorded as None and carried through to the cache as -1.
    It is NOT defaulted to 0 or 1: inventing a class for an unlabelled clip is
    how an evaluation ends up measuring something other than what it claims.
    Downstream, `metrics.evaluate_anticipation` refuses to run without labels,
    which is the correct behaviour and the point of the paper.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        labelled = "label" in cols
        frames_on_disk = "frame_dir" in cols

        for row in reader:
            def _opt(key):
                v = (row.get(key) or "").strip()
                return int(float(v)) if v else None

            label = int(row["label"]) if labelled and (row.get("label") or "").strip() else None
            onset = (row.get("onset_frame") or "").strip()

            if label == 1 and not onset:
                raise ValueError(
                    f"Clip {row['clip_id']} is positive but has no onset_frame. "
                    "Time-to-accident cannot be computed without it."
                )

            rows.append({
                "clip_id": row["clip_id"],
                # exactly one of these is set
                "video": row.get("video") or None,
                "frame_dir": row.get("frame_dir") or None,
                "label": label,
                "onset_frame": int(onset) if onset else None,
                "start_frame": _opt("dir_start") if frames_on_disk else _opt("start_frame"),
                "end_frame": _opt("dir_end") if frames_on_disk else _opt("end_frame"),
                "alert_frame": _opt("alert_frame"),
                "fps": float(row["fps"]) if (row.get("fps") or "").strip() else None,
            })

    if not labelled:
        print("\n  This manifest carries NO LABEL COLUMN.\n"
              "  Features will extract normally, but AP, AUC, false-positive rate\n"
              "  and time-to-accident-to-onset cannot be computed from them by\n"
              "  anyone without the ground truth. Only the legacy curve-shape\n"
              "  statistics are available: protocol_comparison.py --legacy_only\n")
    return rows


def check_device(device: str) -> str:
    """Fail early and readably if a GPU was asked for and is not usable.

    Without this, an unusable CUDA build surfaces as an AssertionError raised
    deep inside CLIP's model loader, several frames from anything the caller
    wrote. The most common cause on Colab is a dependency resolution that
    replaces the preinstalled CUDA torch with a CPU-only wheel.
    """
    if not device.startswith("cuda"):
        return device
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is not installed.")

    if torch.cuda.is_available():
        return device

    raise SystemExit(
        "\n  CUDA was requested but is not available.\n\n"
        f"  torch {torch.__version__}, built with CUDA: {torch.version.cuda}\n\n"
        "  If 'built with CUDA' is None you have a CPU-only build. On Colab this\n"
        "  usually means a pip install pulled a CPU torch in as a dependency.\n"
        "  Reinstall the extras without letting them touch torch:\n\n"
        "      pip install -q ftfy regex\n"
        "      pip install -q --no-deps sentence-transformers\n"
        "      pip install -q transformers tokenizers huggingface-hub Pillow\n"
        "      pip install -q --no-deps git+https://github.com/openai/CLIP.git\n\n"
        "  If instead no GPU is attached, enable the accelerator in the runtime\n"
        "  settings. To run on CPU anyway (slow, for a smoke test only):\n"
        "      --device cpu\n"
    )


def main():
    safe_console()
    args = parse_args()
    args.device = check_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root) if args.video_root else None

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rows = read_manifest(Path(args.manifest))
    if args.limit:
        rows = rows[:args.limit]

    n_lab = sum(r["label"] is not None for r in rows)
    if n_lab == 0:
        print(f"Manifest: {len(rows)} clips, unlabelled")
    else:
        n_pos = sum(r["label"] == 1 for r in rows)
        n_neg = n_lab - n_pos
        print(f"Manifest: {len(rows)} clips  ({n_pos} positive, {n_neg} negative)")
    print(f"Device: {args.device}, CLIP batch {args.clip_batch}")
    if n_lab > 0 and n_neg == 0:
        print("\n  WARNING: no negative clips in this manifest.\n"
              "  AP, AUC and false-positive rate cannot be computed without them,\n"
              "  and a model that always outputs a rising curve will score perfectly.\n")

    suffix = "_raft" if args.use_raft else ""

    clip_scorer = CLIPScorer(
        model_name=cfg["clip"]["model"],
        danger_prompt=cfg["clip"]["danger_prompt"],
        safe_prompt=cfg["clip"]["safe_prompt"],
        device=args.device,
        batch_size=args.clip_batch,
    )
    flow_scorer = FlowScorer(use_raft=args.use_raft, device=args.device)
    nlp_scorer = NLPScorer(
        model_name=cfg["nlp"]["model"],
        sudden_anchor=cfg["nlp"]["sudden_anchor"],
        gradual_anchor=cfg["nlp"]["gradual_anchor"],
        rho1=cfg["nlp"].get("rho1", 1.0),
        rho2=cfg["nlp"].get("rho2", 0.5),
    )

    vcfg = cfg["video"]
    done = skipped = failed = 0
    t_start = time.time()

    for i, row in enumerate(rows, 1):
        out_path = out_dir / f"{row['clip_id']}{suffix}.npz"
        if out_path.exists():
            skipped += 1
            continue

        try:
            common = dict(
                num_frames=vcfg["num_frames"],
                resize=tuple(vcfg["resize"]),
                interpolation=vcfg.get("interpolation", "bicubic"),
                crop_top_frac=vcfg.get("crop_top_frac", 0.20),
                crop_bottom_frac=vcfg.get("crop_bottom_frac", 0.08),
                normalise=vcfg.get("normalise", True),
                start_frame=row.get("start_frame"),
                end_frame=row.get("end_frame"),
            )
            if row["frame_dir"]:
                frames = load_frames_from_dir(row["frame_dir"], **common)
            elif row["video"]:
                if video_root is None:
                    raise ValueError(
                        "This manifest names video files; pass --video_root.")
                frames = load_frames(video_root / row["video"], **common)
            else:
                raise ValueError(
                    f"Clip {row['clip_id']} has neither frame_dir nor video.")

            np.savez_compressed(
                out_path,
                p_clip=clip_scorer.score(frames).astype(np.float32),
                p_flow=flow_scorer.score(frames).astype(np.float32),
                p_nlp=nlp_scorer.score(frames).astype(np.float32),
                label=np.int32(row["label"] if row["label"] is not None else -1),
                onset_frame=np.int32(row["onset_frame"] if row["onset_frame"] is not None else -1),
                alert_frame=np.int32(row["alert_frame"] if row.get("alert_frame") is not None else -1),
                fps=np.float32(row["fps"] if row.get("fps") is not None else -1.0),
                n_frames=np.int32(len(frames)),
            )
            done += 1
        except Exception as e:  # noqa: BLE001 — one bad clip must not kill a long run
            failed += 1
            print(f"  [FAIL] {row['clip_id']}: {e}")

        if i % 25 == 0 or i == len(rows):
            rate = done / max(time.time() - t_start, 1e-6)
            remaining = len(rows) - i
            eta = remaining / rate / 60 if rate > 0 else float("nan")
            print(f"  {i}/{len(rows)}  extracted={done} skipped={skipped} failed={failed}  "
                  f"({rate:.2f} clips/s, ETA {eta:.1f} min)")

    print(f"\nDone. {done} extracted, {skipped} already present, {failed} failed.")
    print(f"Features in: {out_dir}")
    print(f"Next: python scripts/evaluate.py --features {out_dir} "
          f"--config configs/honest.yaml")


if __name__ == "__main__":
    main()
