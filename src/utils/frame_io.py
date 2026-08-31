"""
Reading clips that ship as folders of JPG frames.

WHY THIS EXISTS
---------------
The zero-shot-taa corpus does not distribute video. Each clip is a directory:

    Test/<type>/<id>/images/*.jpg

The original pipeline globbed a directory for `*.mp4` and opened each with
`cv2.VideoCapture`, which is not how this dataset ships. Reading the frames the
way the dataset actually provides them removes a conversion step, removes the
question of what an intermediate encode did to the pixels, and removes the frame
rate assumption that a container would otherwise smuggle in.

THE FRAME RATE PROBLEM, STATED ONCE
-----------------------------------
A folder of JPGs has no frame rate. Nothing in the distribution states one. Any
conversion from frames to seconds is therefore an assumption made by the person
reporting the number, not a property of the data, and it must be declared rather
than defaulted. This module returns frames and frame indices only; it converts
nothing to seconds, and the manifest carries `fps` as an explicit, nullable
field so that a missing rate stays visible instead of silently becoming 30.

ORDERING
--------
Frame files are sorted by the integer in their name, not lexicographically.
`sorted()` on unpadded names gives 1, 10, 100, 11, 2 — which reorders time and
produces a plausible-looking risk curve computed over shuffled frames. That is
the kind of error that does not raise and does not look wrong in a plot.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
_NUM = re.compile(r"(\d+)")


def frame_files(frame_dir: str | Path) -> list[Path]:
    """Image files in a clip directory, in temporal order.

    Sorted by the last run of digits in the filename stem, falling back to the
    stem itself when a name carries no digits. See the module docstring for why
    plain `sorted()` is not safe here.
    """
    d = Path(frame_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a frame directory: {d}")

    files = [p for p in d.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No image files in {d}")

    def key(p: Path):
        nums = _NUM.findall(p.stem)
        return (0, int(nums[-1]), p.stem) if nums else (1, 0, p.stem)

    return sorted(files, key=key)


def load_frames_from_dir(
    frame_dir: str | Path,
    num_frames: int = 150,
    start_frame: int | None = None,
    end_frame: int | None = None,
    resize: tuple[int, int] = (224, 224),
    interpolation: str = "bicubic",
    crop_top_frac: float = 0.20,
    crop_bottom_frac: float = 0.08,
    normalise: bool = True,
) -> list[np.ndarray]:
    """
    Load and preprocess frames from a directory of images.

    Parameters
    ----------
    frame_dir : path
        Directory of image files, one per frame.
    num_frames : int
        How many frames to return.
    start_frame, end_frame : int, optional
        Half-open window, as POSITIONS IN THIS DIRECTORY, not as indices in any
        original source video. `prepare_taa.py` resolves the distinction and
        writes positions here; see its docstring.

        Leave both unset to sample across everything present. When the directory
        already holds exactly the window, that is the same thing.
    resize, interpolation, crop_top_frac, crop_bottom_frac, normalise
        As in `video_io.load_frames`, so that a clip read from frames and a clip
        read from video receive identical preprocessing.

    Returns
    -------
    list of np.ndarray, RGB uint8, length `num_frames`.
    """
    files = frame_files(frame_dir)
    total = len(files)

    lo = 0 if start_frame is None else max(0, int(start_frame))
    hi = total if end_frame is None else min(total, int(end_frame))
    if hi - lo < 1:
        raise ValueError(
            f"Empty window [{lo}, {hi}) for {frame_dir}, which holds {total} frames"
        )

    indices = np.linspace(lo, hi - 1, num_frames, dtype=int)
    interp = cv2.INTER_CUBIC if interpolation == "bicubic" else cv2.INTER_LINEAR

    out: list[np.ndarray] = []
    cache: dict[int, np.ndarray] = {}

    for idx in indices:
        i = int(idx)
        if i in cache:
            # linspace repeats indices when num_frames exceeds the window; copy
            # so callers can mutate a frame without touching its twin
            out.append(cache[i].copy())
            continue

        img = cv2.imread(str(files[i]), cv2.IMREAD_COLOR)
        if img is None:
            if out:
                out.append(out[-1].copy())
                continue
            raise ValueError(f"Could not decode {files[i]}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h = img.shape[0]
        img = img[int(h * crop_top_frac):h - int(h * crop_bottom_frac), :]
        img = cv2.resize(img, (resize[1], resize[0]), interpolation=interp)
        if normalise:
            img = _normalise_frame(img)

        cache[i] = img
        out.append(img)

    return out[:num_frames]


def _normalise_frame(frame: np.ndarray) -> np.ndarray:
    """Rescale the 2nd-98th percentile to [0, 255]; overexposure handling.

    Kept identical to `video_io._normalise_frame` so the two readers cannot
    drift apart.
    """
    lo = np.percentile(frame, 2)
    hi = np.percentile(frame, 98)
    if hi - lo < 1:
        return frame
    normed = (frame.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(normed, 0, 255).astype(np.uint8)


def describe_clip(frame_dir: str | Path) -> dict:
    """What is actually in a clip directory. Used by prepare_taa.py to report
    the corpus's structure rather than assume it."""
    files = frame_files(frame_dir)
    nums = []
    for p in files:
        m = _NUM.findall(p.stem)
        if m:
            nums.append(int(m[-1]))

    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    return {
        "n_frames": len(files),
        "first_name": files[0].name,
        "last_name": files[-1].name,
        "index_min": min(nums) if nums else None,
        "index_max": max(nums) if nums else None,
        "contiguous": (len(nums) > 1
                       and max(nums) - min(nums) + 1 == len(nums)),
        "height": None if first is None else int(first.shape[0]),
        "width": None if first is None else int(first.shape[1]),
    }
