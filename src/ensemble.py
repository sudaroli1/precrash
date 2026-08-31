"""
Stage 5 — Weighted Ensemble Fusion (Equation 4 in the paper).

Combines the three modality scores as a convex combination:

  p_raw(t) = 0.55 · p_clip(t)  +  0.25 · p_flow(t)  +  0.20 · p_caption(t)

The weights reflect the relative zero-shot discriminative power of each
modality on the MM-AU evaluation corpus.
"""

from __future__ import annotations

import numpy as np


def ensemble_scores(
    p_clip: np.ndarray,
    p_flow: np.ndarray,
    p_nlp: np.ndarray,
    w_clip: float = 0.55,
    w_flow: float = 0.25,
    w_nlp: float = 0.20,
) -> np.ndarray:
    """
    Compute weighted convex combination of the three modality risk scores.

    Parameters
    ----------
    p_clip : np.ndarray, shape (T,)
        Per-frame CLIP semantic danger scores.
    p_flow : np.ndarray, shape (T,)
        Per-frame optical flow kinematic motion scores.
    p_nlp : np.ndarray, shape (T,)
        Per-frame NLP caption temporal prior scores.
    w_clip, w_flow, w_nlp : float
        Fusion weights (must sum to 1.0).

    Returns
    -------
    p_raw : np.ndarray, shape (T,)
        Fused raw ensemble score in [0, 1].
    """
    assert abs(w_clip + w_flow + w_nlp - 1.0) < 1e-6, \
        f"Weights must sum to 1.0, got {w_clip + w_flow + w_nlp:.4f}"
    assert p_clip.shape == p_flow.shape == p_nlp.shape, \
        "All score arrays must have the same shape."

    p_raw = w_clip * p_clip + w_flow * p_flow + w_nlp * p_nlp
    return np.clip(p_raw, 0.0, 1.0).astype(np.float32)
