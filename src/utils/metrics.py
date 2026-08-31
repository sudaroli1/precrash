"""
Evaluation metrics for accident anticipation — corrected implementation.

WHAT CHANGED AND WHY
--------------------
The previous version computed every metric from the risk curve alone, with no
ground truth. Consequences:

  * compute_tta() returned (T - t_c)/fps, i.e. the distance from the alarm to
    the END OF THE CLIP. That is only the time-to-accident if the collision
    happens on the last frame of every clip, which is not true of any accident
    anticipation benchmark.
  * compute_stta() checked that the score stays above threshold after crossing
    — which the post-processing clamp forces to be true by construction. It
    could only ever return True.
  * compute_ap() and compute_auc() existed but were never called, because the
    evaluation script had no labels to pass them.

This module replaces those with the evaluation protocol used by the accident
anticipation literature (Chan et al. ACCV 2016; Bao et al. ACM MM 2020), so
that numbers produced here are directly comparable to published DSA / UString /
DSTA results on DAD and CCD.

THE PROTOCOL
------------
Evaluation is at the level of VIDEOS, not frames, and requires both positive
(accident) and negative (no accident) clips.

For a decision threshold q:
  * A positive clip counts as DETECTED if any frame BEFORE the accident onset
    scores >= q. Its time-to-accident is (onset_frame - first_crossing)/fps.
    Frames at or after onset do not count — predicting a crash that has already
    started is not anticipation.
  * A negative clip counts as a FALSE POSITIVE if any frame scores >= q.
  * Recall    = detected positives / all positives
  * Precision = detected positives / (detected positives + false positive clips)
  * mTTA at q = mean TTA over detected positives

Sweeping q gives a precision-recall curve. AP is its area. mTTA is the mean TTA
across thresholds. TTA@R80 is the mTTA at the threshold where recall first
reaches 0.80 — the number most papers headline, because a warning time is
meaningless without stating the recall it was achieved at.

Report AP, AUC, mTTA and TTA@R80 together. Any one of them alone can be gamed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class AnticipationMetrics:
    """Corpus-level anticipation metrics. All fields are directly comparable
    to published numbers when computed on DAD or CCD with the same splits."""
    ap: float                 # average precision over the video-level PR curve
    auc: float                # ROC-AUC on clip-level scores (max score per clip)
    mtta: float               # mean TTA in seconds, averaged across thresholds
    tta_at_r80: float         # mean TTA in seconds at the threshold giving recall >= 0.80
    precision_at_r80: float
    n_positive: int
    n_negative: int
    fps: float

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"AP      = {self.ap:.4f}\n"
            f"AUC     = {self.auc:.4f}\n"
            f"mTTA    = {self.mtta:.3f} s\n"
            f"TTA@R80 = {self.tta_at_r80:.3f} s  (precision {self.precision_at_r80:.3f})\n"
            f"N       = {self.n_positive} positive / {self.n_negative} negative clips"
        )


# ----------------------------------------------------------------------
# Core evaluation
# ----------------------------------------------------------------------

def matched_windows(
    labels: Sequence[int],
    onsets: Sequence[int | None],
    n_frames: Sequence[int],
    seed: int = 0,
) -> list[int]:
    """Give every clip a decision window of comparable length.

    WHY THIS IS NEEDED
    ------------------
    A positive clip's decision window ends at its annotated onset: firing after
    the collision has begun is not anticipation. A negative clip has no onset,
    so the obvious thing is to let it run to the end of the clip — and that is
    what makes the comparison unfair in a way that is easy to miss.

    Every Nexar window here is 150 frames, and positive onsets fall between 91
    and 135. So a negative clip is observed for 150 frames and a positive for
    91-135. Any score curve that tends to rise therefore attains a HIGHER
    maximum on negatives than on positives, for reasons that have nothing to do
    with the video. On synthetic data a video-blind rising curve scores AUC
    0.000 — not chance, but perfectly inverted — and that number is a property
    of the window lengths alone.

    DAD and CCD never hit this because their clips are fixed-length with the
    accident at a fixed frame, so every clip is observed for the same 90 frames.
    Reproducing that here means giving each negative a pseudo-onset drawn from
    the empirical distribution of positive onsets. The draw is seeded and
    depends only on the manifest, so it is identical on every re-run.

    This is the same class of defect the paper is about: a detail of the
    evaluation, not of any model, deciding the ranking. It is reported rather
    than quietly fixed — protocol_comparison.py shows the metric computed both
    ways.
    """
    pos_onsets = [int(o) for lab, o in zip(labels, onsets) if lab == 1 and o is not None]
    if not pos_onsets:
        raise ValueError("Cannot match windows without at least one annotated onset.")

    rng = np.random.default_rng(seed)
    pool = np.asarray(pos_onsets)

    out = []
    for lab, o, T in zip(labels, onsets, n_frames):
        if lab == 1:
            out.append(int(o))
        else:
            # draw a plausible onset, then clip to the frames this clip has
            out.append(int(min(int(rng.choice(pool)), int(T))))
    return out


def evaluate_anticipation(
    scores: list[np.ndarray],
    labels: list[int],
    onsets: list[int | None],
    fps: float | Sequence[float] = 30.0,
    n_thresholds: int = 200,
    windows: Sequence[int] | None = None,
) -> AnticipationMetrics:
    """
    Evaluate accident anticipation with ground truth.

    Parameters
    ----------
    scores : list of np.ndarray
        One risk trajectory per clip, shape (T_i,). Lengths may differ.
    labels : list of int
        1 for an accident clip, 0 for a normal clip. Must be the same length
        as `scores`, and must contain at least one of each.
    onsets : list of int or None
        Accident onset frame index for positive clips. Ignored (pass None) for
        negatives. A positive clip with onset=None is an error: without it
        there is no way to distinguish anticipation from post-hoc detection.
    fps : float or sequence of float
        Frame rate used to convert frame counts to seconds. Pass a sequence of
        the same length as `scores` to use each clip's own rate.

        A single corpus-wide rate is only correct when every clip shares it.
        In the Nexar corpus rates span 23.6-31.0 fps across 36 distinct values,
        so a scalar 30.0 misconverts most clips by up to 27%. Prefer the
        per-clip form; extract_features.py caches the rate for exactly this.
    n_thresholds : int
        Number of thresholds swept over the observed score range.
    windows : sequence of int, optional
        Per-clip decision window end, in frames. A clip is judged only on
        frames before this index. Defaults to the onset for positives and the
        whole clip for negatives — which is asymmetric whenever onsets fall
        short of the clip end. Pass `matched_windows(...)` to give negatives a
        comparable window; see that function for why it matters.

    Returns
    -------
    AnticipationMetrics
    """
    if not (len(scores) == len(labels) == len(onsets)):
        raise ValueError("scores, labels and onsets must be the same length")

    if np.isscalar(fps):
        fps_arr = np.full(len(scores), float(fps), dtype=float)
    else:
        fps_arr = np.asarray(fps, dtype=float)
        if len(fps_arr) != len(scores):
            raise ValueError("per-clip fps must be the same length as scores")
        if not np.all(fps_arr > 0):
            raise ValueError("every clip needs a positive frame rate")

    labels_arr = np.asarray(labels, dtype=int)
    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Need both positive and negative clips to evaluate "
            f"(got {n_pos} positive, {n_neg} negative). "
            "AP, AUC and false-positive rate are undefined on accident-only data — "
            "a model that always outputs a rising curve scores perfectly."
        )

    for s, lab, onset in zip(scores, labels, onsets):
        if lab == 1:
            if onset is None:
                raise ValueError("Positive clips require an accident onset frame index.")
            if not (0 < onset <= len(s)):
                raise ValueError(f"Onset {onset} outside clip of length {len(s)}.")

    # ---- per-clip summaries, computed once ----
    #
    # The sweep below needs two things per clip and nothing else:
    #
    #   positives  the RUNNING MAXIMUM of the pre-onset window. It is
    #              non-decreasing by construction, so the first frame to cross a
    #              threshold q is searchsorted(M, q) — one binary search for
    #              every threshold at once, instead of rescanning the clip.
    #   negatives  the maximum over the whole clip, since "any frame >= q" is
    #              "max >= q".
    #
    # The earlier implementation looped over thresholds and clips in Python and
    # rescanned every frame each time. That is ~200x the work, and it made the
    # bootstrap (1000 resamples x 2 methods x 4 comparisons) cost an hour on the
    # dev split alone. This form is arithmetically identical — the tests pin
    # that — and finishes in seconds, which is the difference between reporting
    # confidence intervals and quietly dropping them.
    if windows is None:
        win = [int(o) if lab == 1 else len(s)
               for s, lab, o in zip(scores, labels, onsets)]
    else:
        win = [int(w) for w in windows]
        if len(win) != len(scores):
            raise ValueError("windows must be the same length as scores")

    clip_scores = []
    pos_runmax, pos_onset, pos_fps, neg_max = [], [], [], []

    for i, (s, lab, onset) in enumerate(zip(scores, labels, onsets)):
        s = np.asarray(s, dtype=float)
        obs = s[:win[i]]
        if lab == 1:
            clip_scores.append(float(obs.max()) if len(obs) else 0.0)
            pos_runmax.append(np.maximum.accumulate(obs))
            pos_onset.append(int(onset))
            pos_fps.append(float(fps_arr[i]))
        else:
            m = float(obs.max()) if len(obs) else 0.0
            clip_scores.append(m)
            neg_max.append(m)

    auc = float(roc_auc_score(labels_arr, np.asarray(clip_scores)))

    # ---- threshold sweep ----
    all_vals = np.concatenate([np.asarray(s).ravel() for s in scores])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    if hi <= lo:
        hi = lo + 1e-6

    # The threshold at the minimum observed score is satisfied by every frame of
    # every clip. It therefore "detects" all positives at frame 0 and reports the
    # maximum possible warning time — a free TTA that is exactly the failure mode
    # this module exists to catch — while on the PR curve it is the ordinary
    # recall=1 / precision=base-rate corner that AP is defined to include.
    #
    # So it is swept, and then excluded from the TIMING statistics only. An
    # earlier version dropped it from the sweep entirely, which had the effect of
    # scoring a constant curve at AP 0.000 instead of the base rate: for a
    # constant predictor that corner is the whole PR curve. A control that scores
    # below chance is a bug in the metric, and it would have been read as one.
    thresholds = np.linspace(lo, hi, n_thresholds + 1)

    # false-positive clips per threshold: how many negative maxima reach q
    neg_sorted = np.sort(np.asarray(neg_max, dtype=float))
    fp_counts = len(neg_sorted) - np.searchsorted(neg_sorted, thresholds, side="left")

    detected = np.zeros(len(thresholds), dtype=np.int64)
    tta_sum = np.zeros(len(thresholds), dtype=float)

    for M, onset, rate in zip(pos_runmax, pos_onset, pos_fps):
        first = np.searchsorted(M, thresholds, side="left")
        hit = first < len(M)
        detected += hit
        # convert with THIS clip's rate, then average in seconds
        tta_sum[hit] += (onset - first[hit]) / rate

    recalls = detected / n_pos
    denom = detected + fp_counts
    precisions = np.divide(detected, denom, out=np.zeros(len(thresholds)),
                           where=denom > 0)
    ttas = np.divide(tta_sum, detected, out=np.zeros(len(thresholds)),
                     where=detected > 0)

    ap = _average_precision(precisions, recalls)

    # Timing metrics ignore the all-detecting endpoint (index 0). See above.
    precisions, recalls, ttas = precisions[1:], recalls[1:], ttas[1:]
    thresholds = thresholds[1:]

    # mTTA: mean over thresholds that detect anything at all
    valid = recalls > 0
    mtta = float(ttas[valid].mean()) if valid.any() else 0.0

    # TTA at 80% recall — the threshold closest to recall 0.80 from above
    at80 = np.where(recalls >= 0.80)[0]
    if len(at80):
        # Among thresholds reaching 80% recall, take the STRICTEST — the
        # highest threshold. `thresholds` is ascending, so that is the last
        # qualifying index.
        #
        # Selecting by "recall closest to 0.80" looks equivalent but is not:
        # when a score distribution is well separated, every qualifying
        # threshold ties at recall 1.0, and argmin then silently returns the
        # LOOSEST threshold. That reports the maximum possible warning time
        # for every method and makes TTA@R80 useless for comparing them.
        idx = int(at80[-1])
        tta_r80, prec_r80 = float(ttas[idx]), float(precisions[idx])
    else:
        tta_r80, prec_r80 = float("nan"), float("nan")

    return AnticipationMetrics(
        ap=ap,
        auc=auc,
        mtta=mtta,
        tta_at_r80=tta_r80,
        precision_at_r80=prec_r80,
        n_positive=n_pos,
        n_negative=n_neg,
        fps=float(np.median(fps_arr)),
    )


def _average_precision(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """Area under the precision-recall curve, computed by sorting on recall and
    integrating with the standard step rule."""
    order = np.argsort(recalls)
    r = recalls[order]
    p = precisions[order]

    # collapse duplicate recall values to their maximum precision
    uniq_r, uniq_p = [], []
    for ri, pi in zip(r, p):
        if uniq_r and np.isclose(ri, uniq_r[-1]):
            uniq_p[-1] = max(uniq_p[-1], pi)
        else:
            uniq_r.append(float(ri))
            uniq_p.append(float(pi))

    r = np.asarray([0.0] + uniq_r)
    p = np.asarray([uniq_p[0] if uniq_p else 0.0] + uniq_p)
    return float(np.sum(np.diff(r) * p[1:]))


# ----------------------------------------------------------------------
# Bootstrap confidence intervals
# ----------------------------------------------------------------------

def _resample_fps(fps, idx):
    """Carry the per-clip frame rate through a resample.

    This is not a detail. When `fps` is a per-clip sequence, resampling the
    clips without resampling the rates hands clip idx[k]'s scores to clip k's
    frame rate. The lengths still match, so nothing raises — every TTA in the
    interval is just converted with the wrong divisor. On Nexar, where rates
    span 23.6-31.0 fps, that is a silent error of up to 27% inside the very
    numbers meant to quantify uncertainty.
    """
    if np.isscalar(fps):
        return float(fps)
    return np.asarray(fps, dtype=float)[idx]


def bootstrap_ci(
    scores: list[np.ndarray],
    labels: list[int],
    onsets: list[int | None],
    fps: float | Sequence[float] = 30.0,
    metric: str = "ap",
    n_boot: int = 1000,
    seed: int = 0,
    n_thresholds: int = 200,
    windows: Sequence[int] | None = None,
) -> tuple[float, float]:
    """
    Percentile bootstrap 95% CI for one metric, resampling clips with replacement.

    Report these. A single point estimate on a few hundred clips will be
    challenged in review, and a CI that overlaps your baseline's is something
    you want to discover before a referee does.

    `n_thresholds` defaults to the same grid the point estimate uses, so the
    interval brackets the estimator that produced it rather than a coarser
    cousin of it.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_labels = [labels[i] for i in idx]
        if len(set(sub_labels)) < 2:
            continue
        try:
            m = evaluate_anticipation(
                [scores[i] for i in idx], sub_labels, [onsets[i] for i in idx],
                fps=_resample_fps(fps, idx), n_thresholds=n_thresholds,
                windows=None if windows is None else [windows[i] for i in idx],
            )
        except ValueError:
            continue
        vals.append(getattr(m, metric))

    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ----------------------------------------------------------------------
# Paired significance test
# ----------------------------------------------------------------------

def paired_bootstrap_test(
    scores_a: list[np.ndarray],
    scores_b: list[np.ndarray],
    labels: list[int],
    onsets: list[int | None],
    fps: float | Sequence[float] = 30.0,
    metric: str = "ap",
    n_boot: int = 1000,
    seed: int = 0,
    n_thresholds: int = 200,
    windows: Sequence[int] | None = None,
) -> dict:
    """
    Is method A actually better than method B, or is the gap noise?

    Resamples the same clip indices for both methods and reports the
    distribution of the difference. Use this for ensemble vs. best single
    modality. If the 95% interval of the difference contains zero, say so in
    the paper — an honest null on one component is publishable; a claimed gain
    that the data does not support is not.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    diffs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_labels = [labels[i] for i in idx]
        if len(set(sub_labels)) < 2:
            continue
        sub_onsets = [onsets[i] for i in idx]
        sub_fps = _resample_fps(fps, idx)
        sub_win = None if windows is None else [windows[i] for i in idx]
        try:
            ma = evaluate_anticipation([scores_a[i] for i in idx], sub_labels, sub_onsets,
                                       fps=sub_fps, n_thresholds=n_thresholds, windows=sub_win)
            mb = evaluate_anticipation([scores_b[i] for i in idx], sub_labels, sub_onsets,
                                       fps=sub_fps, n_thresholds=n_thresholds, windows=sub_win)
        except ValueError:
            continue
        diffs.append(getattr(ma, metric) - getattr(mb, metric))

    if not diffs:
        return {"mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_a_better": float("nan")}

    diffs = np.asarray(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_a_better": float((diffs > 0).mean()),
    }
