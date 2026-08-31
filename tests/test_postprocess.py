"""Tests for the five-stage post-processing pipeline."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from postprocess import PostProcessor

DEFAULT_CFG = {
    "gaussian_sigma": 3,
    "power_gamma": 1.8,
    "temporal_alpha": 1.3,
    "threshold": 0.5,
    "epsilon": 0.01,
    "score_min": 0.001,
    "score_max": 0.999,
}


@pytest.fixture
def pp():
    return PostProcessor(DEFAULT_CFG)


def _ramp(T=150):
    """Linearly increasing risk from 0 to 1."""
    return np.linspace(0, 1, T, dtype=np.float32)


class TestBounds:
    def test_output_within_bounds(self, pp):
        scores = _ramp()
        out = pp.process(scores)
        assert out.min() >= DEFAULT_CFG["score_min"]
        assert out.max() <= DEFAULT_CFG["score_max"]

    def test_output_length_preserved(self, pp):
        scores = _ramp(150)
        out = pp.process(scores)
        assert len(out) == 150


class TestSTTACompliance:
    def test_monotone_after_crossing(self, pp):
        """Once p(t) ≥ 0.5, it must stay ≥ threshold + ε."""
        scores = _ramp()
        out = pp.process(scores)
        crossings = np.where(out >= DEFAULT_CFG["threshold"])[0]
        if len(crossings) > 0:
            t_star = crossings[0]
            assert np.all(out[t_star:] >= DEFAULT_CFG["threshold"] + DEFAULT_CFG["epsilon"]), \
                "STTA violation: score fell below threshold after crossing"

    def test_no_crossing_unchanged_monotonicity(self, pp):
        """If score never crosses 0.5, no clamping should distort."""
        scores = np.full(150, 0.1, dtype=np.float32)
        out = pp.process(scores)
        assert np.all(out < DEFAULT_CFG["threshold"])


class TestPowerCurve:
    def test_gamma_suppresses_low_scores(self, pp):
        low = np.full(150, 0.2, dtype=np.float32)
        out = pp._power_curve(low)
        assert np.all(out < low), "Power curve with γ>1 should suppress sub-0.5 scores"

    def test_gamma_amplifies_high_scores(self, pp):
        # After Gaussian (identity here), power on 0.8 → 0.8^1.8 ≈ 0.66 — wait,
        # γ > 1 SUPPRESSES all scores. This is correct per paper: contrast stretch.
        # Scores above 0.5 are relatively less suppressed than scores below 0.5.
        low = np.full(150, 0.3, dtype=np.float32)
        high = np.full(150, 0.8, dtype=np.float32)
        out_low = pp._power_curve(low)
        out_high = pp._power_curve(high)
        # The ratio high/low should be amplified
        assert (out_high / out_low).mean() > (high / low).mean()


class TestTemporalCompression:
    def test_compression_shifts_crossover_earlier(self, pp):
        """Temporal compression (α=1.3) should shift risk curve earlier."""
        scores = _ramp()
        t_before = np.where(scores >= 0.5)[0][0]
        compressed = pp._temporal_compress(scores)
        t_after = np.where(compressed >= 0.5)[0][0]
        assert t_after <= t_before, "Temporal compression should advance crossover"


class TestEdgeCases:
    def test_all_zeros(self, pp):
        out = pp.process(np.zeros(150, dtype=np.float32))
        assert out.min() >= DEFAULT_CFG["score_min"]

    def test_all_ones(self, pp):
        out = pp.process(np.ones(150, dtype=np.float32))
        assert out.max() <= DEFAULT_CFG["score_max"]

    def test_single_frame(self, pp):
        out = pp.process(np.array([0.8], dtype=np.float32))
        assert len(out) == 1
