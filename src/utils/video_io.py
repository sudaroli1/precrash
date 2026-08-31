"""
Video loading and preprocessing utilities (Stage 1 of the pipeline).

Preprocessing steps (Section 3.2 of the paper):
  - Resize to 224×224 with bicubic interpolation (edge preservation)
  - Distribution shift normalisation (handles overexposed frames)
  - Crop top 20% (sky / HUD) and bottom 8% (vehicle hood)
  - Homography-based camera-shake compensation (for optical flow)
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Union


def load_frames(
    video_path: Union[str, Path],
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
    Load, preprocess, and sample frames from a dashcam video clip.

    Parameters
    ----------
    video_path : str or Path
        Path to video file.
    num_frames : int
        Number of frames to sample.
    start_frame, end_frame : int, optional
        Half-open native-frame window to sample within. Set both, with
        end_frame - start_frame == num_frames, for 1:1 sampling at the video's
        own frame rate. Leave unset to sample across the whole clip.
    resize : tuple
        Target (H, W) after crop.
    interpolation : str
        Resize interpolation — "bicubic" (default) or "linear".
    crop_top_frac : float
        Fraction of frame height to strip from top (sky/HUD).
    crop_bottom_frac : float
        Fraction of frame height to strip from bottom (hood).
    normalise : bool
        Apply per-frame distribution shift normalisation.

    Returns
    -------
    frames : list of np.ndarray
        Preprocessed RGB frames, each shape (H, W, 3) uint8.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Video has no readable frames: {video_path}")

    # Sample within [start_frame, end_frame) when a window is given, otherwise
    # across the whole clip.
    #
    # The whole-clip default is only safe when the clip length already matches
    # num_frames. On long footage it silently rescales time: sampling 150
    # frames from a 40-second clip makes each sampled frame ~0.27 s of real
    # time, so any metric that later assumes 1/fps seconds per frame is wrong
    # by the ratio between them. Pass a window whose length equals num_frames
    # and sampling is 1:1, which keeps seconds meaning seconds.
    lo = 0 if start_frame is None else max(0, int(start_frame))
    hi = total if end_frame is None else min(total, int(end_frame))
    if hi - lo < 1:
        raise ValueError(
            f"Empty frame window [{lo}, {hi}) for {video_path} (total {total})"
        )
    indices = np.linspace(lo, hi - 1, num_frames, dtype=int)

    interp_flag = cv2.INTER_CUBIC if interpolation == "bicubic" else cv2.INTER_LINEAR

    def _prep(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        top = int(h * crop_top_frac)
        bot = h - int(h * crop_bottom_frac)
        frame = frame[top:bot, :]
        frame = cv2.resize(frame, (resize[1], resize[0]), interpolation=interp_flag)
        return _normalise_frame(frame) if normalise else frame

    # HOW THE FRAMES ARE FETCHED, AND WHY IT IS NOT THE OBVIOUS WAY
    # ------------------------------------------------------------
    # The obvious loop is `cap.set(POS_FRAMES, i); cap.read()` for each wanted
    # index. In an inter-frame-coded format that is not a seek — it is a decode
    # from the nearest preceding keyframe, repeated for every frame. On this
    # corpus the windows start several hundred frames in, so 150 wanted frames
    # cost 150 keyframe-to-target decodes: measured at 31.3 s for one clip,
    # against 2.5 s for reading the same span sequentially. Identical pixels,
    # 12.7x the time. That single line accounted for most of an 81 s/clip
    # extraction rate, which would have been 20 hours for the dev split alone.
    #
    # So: seek once to the start of the span, then read forward and keep the
    # wanted indices. Sequential decoding is cheap per frame, and stays cheaper
    # than seeking even when the span is several times the number of frames
    # wanted. The fallback below exists only for a pathologically sparse
    # request — a handful of frames from a very long recording — where decoding
    # everything in between really would cost more than seeking.
    span = hi - lo
    frames = []

    if span > 20 * num_frames:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1].copy())
                continue
            frames.append(_prep(frame))
    else:
        if lo > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(lo))
        wanted = [int(i) for i in indices]
        cur, k = lo, 0
        while k < len(wanted):
            ret, frame = cap.read()
            if not ret:
                break
            if cur == wanted[k]:
                prepped = _prep(frame)
                # linspace can repeat an index when num_frames exceeds the span
                while k < len(wanted) and wanted[k] == cur:
                    frames.append(prepped if k == 0 else prepped.copy())
                    k += 1
            cur += 1

    cap.release()

    # Ensure exactly num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else np.zeros((*resize, 3), dtype=np.uint8))

    return frames[:num_frames]


def _normalise_frame(frame: np.ndarray) -> np.ndarray:
    """
    Clip-level distribution shift normalisation.
    Rescales frame so that the 2nd–98th percentile spans [0, 255],
    handling overexposed dashcam frames.
    """
    lo = np.percentile(frame, 2)
    hi = np.percentile(frame, 98)
    if hi - lo < 1:
        return frame
    normed = (frame.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(normed, 0, 255).astype(np.uint8)
