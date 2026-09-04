"""
Stage 3 — Kinematic scorer (frame-difference proxy).

Implements the frame-difference proxy of Section 4.2, Equation (2), of the
paper. It is NOT optical flow. RAFT was the intended estimator and did not fit
the compute budget.

An earlier version of this docstring, and of the paper, claimed the proxy
retains "~90% of RAFT-equivalent sensitivity at ~1% of the compute". That was
an estimate with no ablation behind it and IT IS WITHDRAWN. FlowScorer supports
use_raft=True, so the comparison can still be run; until it is, no claim about
the trade-off should be made anywhere.

  flow(t) = mean(|frame(t) − frame(t−1)|)  [L1 norm of frame difference]

The raw flow values are min-max normalised across the clip to produce
a per-frame kinematic risk score p_flow(t) ∈ [0, 1].

Optionally, RAFT dense optical flow can be enabled via `use_raft=True`
for higher fidelity at the cost of compute.
"""

from __future__ import annotations

import numpy as np
import cv2


class FlowScorer:
    """
    Kinematic motion scorer using frame-differencing (or RAFT).

    Parameters
    ----------
    use_raft : bool
        If True, use RAFT dense optical flow (requires torchvision ≥ 0.15).
        If False (default), use lightweight grayscale frame-differencing proxy.
    device : str
        PyTorch device (used only when use_raft=True).
    """

    def __init__(self, use_raft: bool = False, device: str = "cuda"):
        self.use_raft = use_raft
        self.device = device

        if use_raft:
            try:
                import torchvision.models.optical_flow as of
                self._raft = of.raft_large(pretrained=True).to(device).eval()
            except Exception as e:
                raise ImportError(
                    "RAFT requires torchvision >= 0.15 with pretrained weights. "
                    f"Original error: {e}"
                )

    def score(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Compute per-frame kinematic motion scores.

        Parameters
        ----------
        frames : list of np.ndarray
            RGB frames, each shape (H, W, 3) uint8.

        Returns
        -------
        scores : np.ndarray, shape (T,)
            Normalised kinematic risk in [0, 1].
        """
        if self.use_raft:
            return self._score_raft(frames)
        return self._score_frame_diff(frames)

    # ------------------------------------------------------------------
    # Frame-difference proxy (default — fast)
    # ------------------------------------------------------------------

    def _score_frame_diff(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Lightweight grayscale frame-differencing proxy.
        Equation 1 from the paper:
          flow(t) = mean(|frame(t) − frame(t−1)|)
        """
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]

        raw = np.zeros(len(frames), dtype=np.float32)
        for t in range(1, len(frames)):
            raw[t] = np.mean(np.abs(grays[t] - grays[t - 1]))
        raw[0] = raw[1]  # pad first frame

        # Min-max normalise across the clip (Equation 1)
        lo, hi = raw.min(), raw.max()
        if hi - lo < 1e-8:
            return np.zeros_like(raw)
        return (raw - lo) / (hi - lo)

    # ------------------------------------------------------------------
    # RAFT dense optical flow (optional — high fidelity, slow)
    # ------------------------------------------------------------------

    def _score_raft(self, frames: list[np.ndarray]) -> np.ndarray:
        import torch
        import torchvision.transforms.functional as TF

        def to_tensor(frame):
            return TF.to_tensor(frame).unsqueeze(0).to(self.device)

        raw = np.zeros(len(frames), dtype=np.float32)
        with torch.no_grad():
            for t in range(1, len(frames)):
                img1 = to_tensor(frames[t - 1])
                img2 = to_tensor(frames[t])
                flow = self._raft(img1, img2)[-1]  # (1, 2, H, W)
                magnitude = flow.norm(dim=1).mean().item()
                raw[t] = magnitude
        raw[0] = raw[1]

        lo, hi = raw.min(), raw.max()
        if hi - lo < 1e-8:
            return np.zeros_like(raw)
        return (raw - lo) / (hi - lo)
