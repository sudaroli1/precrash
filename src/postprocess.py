"""
Stage 6 — Five-Stage Post-Processing Pipeline.

Transforms the raw ensemble score into a monotonically non-decreasing,
STTA-compliant risk trajectory through five sequential operations:

  Stage 1 — Gaussian smoothing         (Equation 5)
  Stage 2 — Power-curve amplification  (Equation 6)
  Stage 3 — Temporal axis compression  (Equation 8)
  Stage 4 — Monotone clamping          (Equation 7)
  Stage 5 — Numerical stability clip

NOTE: Stage 3 (temporal compression) MUST precede Stage 4 (clamping).
Applying compression after clamping would violate the monotone guarantee.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


class PostProcessor:
    """
    Five-stage post-processing pipeline for STTA compliance.

    Parameters
    ----------
    config : dict
        Post-processing hyperparameters from config YAML.
        Expected keys: gaussian_sigma, power_gamma, temporal_alpha,
                       threshold, epsilon, score_min, score_max.
    """

    def __init__(self, config: dict):
        self.sigma = config.get("gaussian_sigma", 3)
        self.gamma = config.get("power_gamma", 1.8)
        self.alpha = config.get("temporal_alpha", 1.3)
        self.theta = config.get("threshold", 0.5)
        self.epsilon = config.get("epsilon", 0.01)
        self.score_min = config.get("score_min", 0.001)
        self.score_max = config.get("score_max", 0.999)

    def process(self, p_raw: np.ndarray) -> np.ndarray:
        """
        Apply the full five-stage post-processing pipeline.

        Parameters
        ----------
        p_raw : np.ndarray, shape (T,)
            Raw ensemble score from Stage 5 fusion.

        Returns
        -------
        p_final : np.ndarray, shape (T,)
            Post-processed, STTA-compliant risk trajectory in (score_min, score_max).
        """
        p = p_raw.copy().astype(np.float64)

        # Stage 1 — Gaussian smoothing (Equation 5)
        p = self._gaussian_smooth(p)

        # Stage 2 — Power-curve amplification (Equation 6)
        p = self._power_curve(p)

        # Stage 3 — Temporal axis compression (Equation 8)
        # NOTE: must precede monotone clamping (Stage 4)
        p = self._temporal_compress(p)

        # Stage 4 — Monotone clamping (Equation 7)
        p = self._monotone_clamp(p)

        # Stage 5 — Numerical stability clip
        p = np.clip(p, self.score_min, self.score_max)

        return p.astype(np.float32)

    # ------------------------------------------------------------------
    # Individual stages
    # ------------------------------------------------------------------

    def _gaussian_smooth(self, p: np.ndarray) -> np.ndarray:
        """
        Stage 1: Convolve with 1D Gaussian kernel (σ=3) to reduce
        inter-frame appearance variance while preserving risk trajectory shape.
        Equation 5: p_smooth(t) = G_σ * p_raw(t)
        """
        return gaussian_filter1d(p, sigma=self.sigma)

    def _power_curve(self, p: np.ndarray) -> np.ndarray:
        """
        Stage 2: Monotone contrast stretch via power transformation.
        Equation 6: p_norm(t) = p_smooth(t)^γ
        With γ=1.8 > 1: sub-threshold scores suppressed toward 0,
        supra-threshold scores amplified toward 1.
        """
        p_clipped = np.clip(p, 0.0, 1.0)
        return np.power(p_clipped, self.gamma)

    def _temporal_compress(self, p: np.ndarray) -> np.ndarray:
        """
        Stage 3: Compress time axis by factor α=1.3 via linear interpolation.
        Equation (10) of the paper: p_final(t) = interp( p_norm, α·t ).

        Note what that means, and note that it runs BEFORE the clamp: the
        value emitted at frame t is the value the curve took at frame 1.3t —
        a frame that has not been observed at time t. The stage is non-causal
        and cannot run in a deployed system.

        An earlier version of this docstring wrote the operand as t/α, which
        matches neither this code nor the direction of the shift, and claimed
        a ~6-frame mean TTA gain. That figure IS WITHDRAWN: no table in the
        paper measures this stage on its own. The released configuration sets
        alpha=1.0, disabling it.
        """
        T = len(p)
        t_orig = np.arange(T, dtype=np.float64)
        t_compressed = t_orig / self.alpha
        t_compressed = np.clip(t_compressed, 0, T - 1)
        return np.interp(t_orig, t_compressed, p)

    def _monotone_clamp(self, p: np.ndarray) -> np.ndarray:
        """
        Stage 4: Hard causal constraint — once score crosses θ, it cannot
        fall below θ + ε in subsequent frames.
        Equation 7:
          p_clamp(s) = p_norm(s)                    if s ≤ t*
                       max(p_norm(s), θ + ε)        if s > t*
        where t* = min{t : p_norm(t) ≥ θ}

        This eliminates false recoveries and formally guarantees STTA compliance.
        """
        threshold = self.theta
        floor = threshold + self.epsilon

        # Find first crossing frame t*
        crossing = np.where(p >= threshold)[0]
        if len(crossing) == 0:
            return p  # no crossing — no clamping needed

        t_star = crossing[0]
        p_clamped = p.copy()
        p_clamped[t_star:] = np.maximum(p_clamped[t_star:], floor)
        return p_clamped
