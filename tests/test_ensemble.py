"""Tests for Stage 5 ensemble fusion."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ensemble import ensemble_scores


def test_weights_sum_to_one():
    p = np.ones(10, dtype=np.float32) * 0.5
    out = ensemble_scores(p, p, p, 0.55, 0.25, 0.20)
    assert out is not None


def test_invalid_weights_raise():
    p = np.ones(10, dtype=np.float32) * 0.5
    with pytest.raises(AssertionError):
        ensemble_scores(p, p, p, 0.5, 0.5, 0.5)


def test_output_in_range():
    rng = np.random.default_rng(42)
    p_clip = rng.random(150).astype(np.float32)
    p_flow = rng.random(150).astype(np.float32)
    p_nlp = rng.random(150).astype(np.float32)
    out = ensemble_scores(p_clip, p_flow, p_nlp)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_dominant_clip_weight():
    """With w_clip=1, result should equal p_clip."""
    p_clip = np.linspace(0, 1, 50, dtype=np.float32)
    p_zero = np.zeros(50, dtype=np.float32)
    out = ensemble_scores(p_clip, p_zero, p_zero, 1.0, 0.0, 0.0)
    np.testing.assert_array_almost_equal(out, p_clip, decimal=5)


def test_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        ensemble_scores(np.ones(10), np.ones(20), np.ones(10))
