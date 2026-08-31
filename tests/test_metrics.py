"""
Tests for the corrected evaluation metrics.

The old tests exercised compute_tta() and compute_stta(), which took a score
curve and no labels. They passed, and the thing they were testing was the bug:
metrics computed with no notion of whether the prediction was right.

These tests assert the properties that matter instead:

  * accident-only data is refused, not silently scored
  * time-to-accident is measured to the annotated onset, not the clip end
  * a video-blind curve lands at chance on AP/AUC
  * an informed model beats a video-blind curve, and the paired test says so

Run:  python -m pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.metrics import (  # noqa: E402
    evaluate_anticipation,
    paired_bootstrap_test,
)

T = 100
ONSET = 90
FPS = 20.0


def _corpus(seed=0, n=120):
    """Half accident clips with a rising informative curve, half normal clips
    that stay low. Onset at frame 90 of 100."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    scores, labels, onsets = [], [], []
    for i in range(n):
        positive = i < n // 2
        if positive:
            s = 1 / (1 + np.exp(-12 * (t - 0.45))) + rng.normal(0, 0.02, T)
            onsets.append(ONSET)
        else:
            s = rng.normal(0.15, 0.05, T)
            onsets.append(None)
        scores.append(np.clip(s, 0, 1))
        labels.append(1 if positive else 0)
    return scores, labels, onsets


def _blind_prior(n_frames, midpoint=0.6):
    """A curve computed from the frame count alone — reads no pixels."""
    t = np.linspace(0, 1, n_frames)
    return 1 / (1 + np.exp(-12 * (t - midpoint)))


# ----------------------------------------------------------------------

def test_accident_only_data_is_refused():
    """The defect that caused the desk rejections must now be an error.

    With no negative clips there is no false-positive rate, and any rising
    curve achieves perfect recall. Scoring such a corpus at all is the bug.
    """
    scores, labels, onsets = _corpus()
    pos = [i for i, l in enumerate(labels) if l == 1]
    with pytest.raises(ValueError, match="positive and negative"):
        evaluate_anticipation(
            [scores[i] for i in pos], [labels[i] for i in pos],
            [onsets[i] for i in pos], fps=FPS,
        )


def test_positive_clip_requires_an_onset():
    scores, labels, onsets = _corpus()
    onsets[0] = None  # a positive clip with no onset
    with pytest.raises(ValueError, match="onset"):
        evaluate_anticipation(scores, labels, onsets, fps=FPS)


def test_tta_is_measured_to_onset_not_clip_end():
    """Regression test for the original bug.

    A clip of 100 frames with onset at 90, whose score rises at frame 70:

        correct  (onset - t_c) / fps = (90 - 70) / 20 = 1.0 s
        old bug  (T     - t_c) / fps = (100 - 70) / 20 = 1.5 s

    If this ever reads 1.5 again, the reference point has regressed to the
    end of the clip and every TTA in the paper is inflated.
    """
    lead = np.zeros(T)
    lead[ONSET - 20:] = 1.0
    neg = np.zeros(T)

    m = evaluate_anticipation([lead, neg], [1, 0], [ONSET, None], fps=FPS)
    assert m.tta_at_r80 == pytest.approx(1.0, abs=1e-6)
    assert m.tta_at_r80 != pytest.approx(1.5, abs=1e-6)


def test_crossing_at_onset_is_not_anticipation():
    """Detecting a crash as it begins has zero warning value. Only frames
    strictly before the onset count towards the anticipation window."""
    at_onset = np.zeros(T)
    at_onset[ONSET:] = 1.0
    neg = np.zeros(T)

    m = evaluate_anticipation([at_onset, neg], [1, 0], [ONSET, None], fps=FPS)
    # never crosses inside the pre-onset window at any meaningful threshold
    assert m.tta_at_r80 == pytest.approx(0.0, abs=1e-9) or np.isnan(m.tta_at_r80)


def test_video_blind_prior_is_at_chance():
    """Identical curve on every clip cannot discriminate, whatever its TTA."""
    scores, labels, onsets = _corpus()
    blind = [_blind_prior(len(s)) for s in scores]
    m = evaluate_anticipation(blind, labels, onsets, fps=FPS)
    assert m.ap == pytest.approx(0.5, abs=0.05)   # == base rate
    assert m.auc == pytest.approx(0.5, abs=0.05) or m.auc == pytest.approx(0.0, abs=0.05)


def test_informed_model_beats_blind_prior():
    scores, labels, onsets = _corpus()
    blind = [_blind_prior(len(s)) for s in scores]

    informed = evaluate_anticipation(scores, labels, onsets, fps=FPS)
    prior = evaluate_anticipation(blind, labels, onsets, fps=FPS)

    assert informed.ap > prior.ap
    assert informed.auc > prior.auc


def test_blind_prior_can_win_on_tta_alone():
    """The finding the critique paper rests on.

    A video-blind curve can post a *longer* warning time than a model with
    real discriminative power, because a TTA-style metric rewards crossing
    early and is indifferent to being wrong. This is why TTA must never be
    reported without AP and AUC beside it.
    """
    scores, labels, onsets = _corpus()
    blind = [_blind_prior(len(s), midpoint=0.05) for s in scores]  # crosses very early

    informed = evaluate_anticipation(scores, labels, onsets, fps=FPS)
    prior = evaluate_anticipation(blind, labels, onsets, fps=FPS)

    assert prior.tta_at_r80 > informed.tta_at_r80   # blind curve "wins" on TTA
    assert prior.ap < informed.ap                   # while being uninformative


def test_paired_test_detects_a_real_difference():
    scores, labels, onsets = _corpus()
    blind = [_blind_prior(len(s)) for s in scores]
    result = paired_bootstrap_test(
        scores, blind, labels, onsets, fps=FPS, metric="ap", n_boot=100,
    )
    assert result["mean_diff"] > 0
    assert result["ci_low"] > 0            # significant
    assert result["p_a_better"] > 0.95


# ----------------------------------------------------------------------
# Per-clip frame rates must survive resampling
# ----------------------------------------------------------------------

def _equal_tta_corpus():
    """Two positive clips whose true anticipation time is 1.000 s each, reached
    by different routes: 30 frames at 30 fps, and 60 frames at 60 fps.

    Any evaluation that pairs each clip with its own frame rate must report
    exactly 1.0 s, for every bootstrap resample, with zero width. One that
    carries the rates in the original order while the clips are resampled
    reports 0.5 s or 2.0 s for the mismatched pairs — a wrong number with no
    exception raised, which is why this is a test and not a comment.
    """
    def step(T, cross):
        s = np.zeros(T)
        s[cross:] = 1.0
        return s

    scores = [step(100, 30), step(100, 30), np.zeros(100), np.zeros(100)]
    labels = [1, 1, 0, 0]
    onsets = [60, 90, None, None]
    fps    = [30.0, 60.0, 30.0, 60.0]
    return scores, labels, onsets, fps


def test_per_clip_fps_is_used_for_each_clip():
    scores, labels, onsets, fps = _equal_tta_corpus()
    m = evaluate_anticipation(scores, labels, onsets, fps=fps)
    assert m.mtta == pytest.approx(1.0, abs=1e-9)

    # a single corpus rate cannot be right for both clips
    m30 = evaluate_anticipation(scores, labels, onsets, fps=30.0)
    assert m30.mtta == pytest.approx(1.5, abs=1e-9)


def test_bootstrap_carries_the_per_clip_frame_rate():
    from utils.metrics import bootstrap_ci

    scores, labels, onsets, fps = _equal_tta_corpus()
    lo, hi = bootstrap_ci(scores, labels, onsets, fps=fps,
                          metric="mtta", n_boot=200, seed=0)
    assert lo == pytest.approx(1.0, abs=1e-9)
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_paired_test_carries_the_per_clip_frame_rate():
    scores, labels, onsets, fps = _equal_tta_corpus()
    t = paired_bootstrap_test(scores, scores, labels, onsets, fps=fps,
                              metric="mtta", n_boot=200, seed=0)
    # identical inputs: the difference is zero in every resample, whatever the
    # rates are, provided the same rates reach both sides.
    assert t["mean_diff"] == pytest.approx(0.0, abs=1e-12)
    assert t["ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert t["ci_high"] == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------
# Decision windows must be comparable across classes
# ----------------------------------------------------------------------

def test_unmatched_windows_invert_a_video_blind_prior():
    """The artefact `matched_windows` exists to remove.

    Positives are judged on frames before onset; negatives, by default, on the
    whole clip. When onsets fall short of the clip end, a rising curve reaches a
    higher maximum on every negative than on any positive — so a control that
    never looks at the video lands at AUC 0, not chance. Nothing about the model
    produced that number; the window lengths did.
    """
    from utils.metrics import matched_windows

    T, onset, n = 150, 100, 40
    blind = [_blind_prior(T) for _ in range(n)]
    labels = [1] * (n // 2) + [0] * (n // 2)
    onsets = [onset] * (n // 2) + [None] * (n // 2)

    unmatched = evaluate_anticipation(blind, labels, onsets, fps=FPS)
    assert unmatched.auc == pytest.approx(0.0, abs=1e-9)

    win = matched_windows(labels, onsets, [T] * n, seed=0)
    matched = evaluate_anticipation(blind, labels, onsets, fps=FPS, windows=win)
    assert matched.auc == pytest.approx(0.5, abs=1e-9)


def test_matched_windows_are_deterministic():
    from utils.metrics import matched_windows

    labels = [1, 1, 0, 0, 0]
    onsets = [90, 120, None, None, None]
    a = matched_windows(labels, onsets, [150] * 5, seed=0)
    b = matched_windows(labels, onsets, [150] * 5, seed=0)
    assert a == b
    assert a[0] == 90 and a[1] == 120          # positives keep their onset
    assert all(w in (90, 120) for w in a[2:])  # negatives drawn from that pool


def test_matched_windows_never_exceed_the_clip():
    from utils.metrics import matched_windows

    labels = [1, 0]
    onsets = [140, None]
    assert matched_windows(labels, onsets, [150, 60], seed=0) == [140, 60]


def test_constant_curve_scores_the_base_rate_not_zero():
    """A predictor that outputs the same value everywhere carries no
    information, so its AP must equal the positive base rate. Sweeping only
    thresholds strictly above the minimum scored it 0.000 instead — below
    chance, which reads as a broken metric rather than an uninformative model.

    The all-detecting endpoint stays in the PR curve and out of the timing
    statistics, which is what test_blind_prior_can_win_on_tta_alone pins.
    """
    n = 40
    const = [np.full(100, 0.51) for _ in range(n)]
    labels = [1] * 24 + [0] * 16
    onsets = [90] * 24 + [None] * 16

    m = evaluate_anticipation(const, labels, onsets, fps=FPS)
    assert m.ap == pytest.approx(24 / 40, abs=1e-9)
    assert m.auc == pytest.approx(0.5, abs=1e-9)


def test_free_tta_endpoint_stays_out_of_the_timing_metrics():
    """The threshold at the minimum score fires on frame 0 of every clip and
    would report the largest warning time the clip admits. It must not enter
    mTTA."""
    n = 20
    T, onset = 100, 80
    rng = np.random.default_rng(0)
    scores = [np.clip(np.linspace(0.1, 0.9, T) + 0.02 * rng.standard_normal(T), 0, 1)
              for _ in range(n)]
    labels = [1] * 10 + [0] * 10
    onsets = [onset] * 10 + [None] * 10

    m = evaluate_anticipation(scores, labels, onsets, fps=FPS)
    # onset/FPS is what the free endpoint would report for every clip
    assert m.mtta < onset / FPS
