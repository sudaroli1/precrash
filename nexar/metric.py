"""
The published composite metric, implemented against real labels.

    score = w_AP*AP + w_AUC*AUC + w_TTA*TTA@0.5 + w_STTA*STTA@0.5

    TTA@0.5  = max{ t_ai - t_a  | p_{t_a} > 0.5, 0 <= t_a <= t_ai }
    STTA@0.5 = max{ t_ai - t'_a | p_t > 0.5 for all t in [t'_a, t_ai] }

Three things have to be decided here that the published definition leaves open,
and each is stated rather than buried:

1. **The clip-level score for AP and AUC.** The definition gives a per-frame
   curve and a clip-level discrimination metric without saying how one becomes
   the other. We use max over the clip, and `reduction_sensitivity` reports
   what happens under mean and last-frame instead. On the competition corpus
   this choice is invisible because no labels are published; here it is
   measurable, and it is one candidate explanation for why the published
   formula does not reproduce the organisers' scorer.

2. **Units for the timing terms.** The definition counts frames. This corpus
   has 36 distinct frame rates between 23.6 and 31.0, so a frame is not a
   fixed quantity of time and frame-counted earliness is not commensurable
   across clips. Both are returned: `_f` in frames, as published, and `_s` in
   seconds, which is the only one that means anything.

3. **Negatives.** Both timing terms are defined only where an accident occurs.
   Negatives are excluded from them -- not scored as zero, which would be a
   different and wrong metric. That exclusion is the whole of the paper's
   safety argument, so it is enforced here by raising on a negative rather
   than by convention.

  python3 metric.py        # runs the self-tests
"""
from __future__ import annotations

import numpy as np

THRESH = 0.5


def tta_frames(p: np.ndarray, f_event: int) -> float:
    """Earliness of the first above-threshold frame at or before the event."""
    if not np.isfinite(f_event):
        raise ValueError("TTA is undefined on a clip with no event")
    e = int(f_event)
    if e < 0 or e >= len(p):
        raise ValueError(f"event frame {e} outside clip of {len(p)}")
    hit = np.flatnonzero(p[:e + 1] > THRESH)
    return float(e - hit[0]) if hit.size else 0.0


def stta_frames(p: np.ndarray, f_event: int) -> float:
    """Length of the unbroken above-threshold run that ends at the event.

    The definition requires p > 0.5 at every frame of [t'_a, t_ai], so if the
    curve is not above threshold AT the event there is no admissible t'_a and
    the term is zero -- however early the curve first crossed.
    """
    if not np.isfinite(f_event):
        raise ValueError("STTA is undefined on a clip with no event")
    e = int(f_event)
    if e < 0 or e >= len(p):
        raise ValueError(f"event frame {e} outside clip of {len(p)}")
    if not p[e] > THRESH:
        return 0.0
    below = np.flatnonzero(p[:e + 1] <= THRESH)
    start = int(below[-1]) + 1 if below.size else 0
    return float(e - start)


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """AP summed over DISTINCT score thresholds.

    Ties have to be grouped. Ranking tied scores one-by-one makes AP depend on
    the order the clips happen to sit in the manifest: a constant submission,
    whose scores are all identical, scored 0.5121 that way instead of the base
    rate 0.5 -- a tenth of a point of discrimination conjured out of row order.
    Grouping by distinct threshold is what scikit-learn does and is the only
    tie-handling that leaves a constant at chance, where it belongs.
    """
    npos = int(y.sum())
    if not npos:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    ys, ss = y[o], s[o]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    # last index of each run of equal scores
    last = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    prec = tp[last] / (tp[last] + fp[last])
    rec = tp[last] / npos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def auc_rank(y: np.ndarray, s: np.ndarray) -> float:
    """AUC in its rank formulation, ties averaged (Mann-Whitney U / n_p n_n)."""
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if not npos or not nneg:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    sr = s[o]
    i = 0
    while i < len(sr):                      # average ranks within ties
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def evaluate(curves: dict, man, reduction: str = "max") -> dict:
    """Score a set of per-clip risk curves against the manifest.

    curves: id -> float array over that clip's full frame grid.
    man:    DataFrame with id, fps, n_frames, target, f_event.
    """
    red = {"max": np.max, "mean": np.mean,
           "last": lambda a: a[-1]}[reduction]

    ids = man.id.tolist()
    y = man.target.to_numpy().astype(int)
    s = np.array([red(curves[i]) for i in ids], float)

    tta_f, tta_s, stta_f, stta_s = [], [], [], []
    for row in man[man.target == 1].itertuples(index=False):
        p = curves[row.id]
        tf = tta_frames(p, row.f_event)
        sf = stta_frames(p, row.f_event)
        tta_f.append(tf); stta_f.append(sf)
        tta_s.append(tf / row.fps); stta_s.append(sf / row.fps)

    return {
        "AP": average_precision(y, s),
        "AUC": auc_rank(y, s),
        "TTA_f": float(np.mean(tta_f)), "STTA_f": float(np.mean(stta_f)),
        "TTA_s": float(np.mean(tta_s)), "STTA_s": float(np.mean(stta_s)),
        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
    }


# --------------------------------------------------------------------------
def _tests():
    # TTA: crosses at 10, event at 100 -> 90 frames of earliness.
    p = np.zeros(200); p[10:] = 0.9
    assert tta_frames(p, 100) == 90, tta_frames(p, 100)
    assert stta_frames(p, 100) == 90

    # A dip breaks STTA but not TTA. Crosses at 10, dips 50-59, event 100.
    p2 = p.copy(); p2[50:60] = 0.2
    assert tta_frames(p2, 100) == 90
    assert stta_frames(p2, 100) == 100 - 60, stta_frames(p2, 100)

    # Below threshold AT the event -> STTA is zero however early it crossed.
    p3 = p.copy(); p3[95:] = 0.1
    assert tta_frames(p3, 100) == 90
    assert stta_frames(p3, 100) == 0.0

    # Never crosses -> both zero.
    p4 = np.full(200, 0.49)
    assert tta_frames(p4, 100) == 0.0 and stta_frames(p4, 100) == 0.0

    # Exactly 0.5 is not above threshold (strict inequality, as published).
    p5 = np.full(200, 0.5)
    assert tta_frames(p5, 100) == 0.0 and stta_frames(p5, 100) == 0.0

    # A constant above threshold attains the maximum each clip admits.
    p6 = np.full(200, 0.51)
    assert tta_frames(p6, 137) == 137 and stta_frames(p6, 137) == 137

    # Crossing after the event earns nothing: the window is [0, t_ai].
    p7 = np.zeros(200); p7[150:] = 0.9
    assert tta_frames(p7, 100) == 0.0

    # AP / AUC against hand-worked cases.
    y = np.array([1, 1, 0, 0]); s = np.array([0.9, 0.8, 0.7, 0.6])
    assert abs(average_precision(y, s) - 1.0) < 1e-12
    assert abs(auc_rank(y, s) - 1.0) < 1e-12
    # Both negatives outrank the weaker positive: 2 of 4 pairs correct.
    s2 = np.array([0.9, 0.5, 0.7, 0.6])
    assert abs(auc_rank(y, s2) - 0.5) < 1e-12, auc_rank(y, s2)
    assert abs(average_precision(y, s2) - (1.0 + 0.5) / 2) < 1e-12

    # One pair wrong of four: AUC 0.75, AP (1 + 2/3)/2.
    s3 = np.array([0.9, 0.65, 0.7, 0.6])
    assert abs(auc_rank(y, s3) - 0.75) < 1e-12, auc_rank(y, s3)
    assert abs(average_precision(y, s3) - (1.0 + 2 / 3) / 2) < 1e-12

    # All-tied scores: AUC must be exactly chance, not 1.0.
    assert abs(auc_rank(y, np.full(4, 0.51)) - 0.5) < 1e-12

    # All-tied scores: AP must be the base rate, not an artefact of row order.
    # Shuffling the rows must not move it.
    for n_p, n_n in ((2, 2), (3, 1), (1, 9)):
        yy = np.r_[np.ones(n_p, int), np.zeros(n_n, int)]
        base = n_p / (n_p + n_n)
        assert abs(average_precision(yy, np.full(len(yy), 0.51)) - base) < 1e-12
        rng = np.random.default_rng(0)
        for _ in range(5):
            o = rng.permutation(len(yy))
            got = average_precision(yy[o], np.full(len(yy), 0.51))
            assert abs(got - base) < 1e-12, (n_p, n_n, got)

    # Ties at the decision boundary must not be split in the ranker's favour.
    yt = np.array([1, 0, 1, 0]); st = np.array([0.9, 0.9, 0.1, 0.1])
    assert abs(average_precision(yt, st) - 0.5) < 1e-12, average_precision(yt, st)
    assert abs(auc_rank(yt, st) - 0.5) < 1e-12

    # Timing terms must refuse a clip with no event rather than score it 0.
    for fn in (tta_frames, stta_frames):
        try:
            fn(p, float("nan")); raise SystemExit(f"{fn.__name__} accepted NaN")
        except ValueError:
            pass
    print("metric.py: all self-tests pass")


if __name__ == "__main__":
    _tests()
