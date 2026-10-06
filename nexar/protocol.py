"""
The corrected protocol of Section 5.3, implemented and made falsifiable.

Item 2 is the one under test. It has two halves and the paper is explicit that
only the second removes the exploit:

  BOUNDING   normalise earliness per clip, achieved / achievable, into [0, 1].
             Stops the term dominating the sum and stops it depending on how
             the corpus was windowed. Does NOT remove the constant as
             maximiser -- it still scores 1 and still gets it free.

  CONDITIONING  report earliness at a fixed operating point of the
             discrimination metric. This is what removes the exploit, because
             a submission that cannot discriminate cannot reach the operating
             point at all.

So the protocol is validated by showing exactly that pattern, not by showing
the constant simply losing: under bounding alone the constant should still
score 1.0, and only under conditioning should it collapse. If the constant fell
over at the bounding step, either the implementation or the paper's analysis
would be wrong.

Two conditioned forms are computed:

  nTTA@R80     threshold set to the lowest achieving 80% recall -- the form
               Section 5.3 names. Reported WITH the false-positive rate it
               costs, because for a constant that rate is 1.0 and the number
               is meaningless without it.
  nTTA@FPR10   threshold set to the highest achieving <= 10% false positives.
               The decisive one: a submission that cannot separate positives
               from negatives detects nothing here, so its earliness is not
               small, it is undefined.

Normalisation uses two different notions of "achievable", and they disagree in
a way only a labelled corpus can show:

  /event   achieved / t_event        the whole clip before the collision
  /alert   achieved / (t_event - t_alert)   the annotators' actionable window

The second is the honest ceiling. Nexar marks t_alert as the earliest moment a
driver could still act; anticipation claimed before it is not usable warning.

  python3 protocol.py      # self-tests
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def first_cross_time(p, thresh, fps):
    """Seconds from clip start to the first sample above `thresh`, or None.

    `p` must be float64. Under NumPy 2's weak scalar promotion a float32 array
    compared against a Python float demotes THE THRESHOLD to float32, so an
    operating point of 0.899999976158142 silently became 0.9 and every clip
    scoring exactly 0.9 fell on the wrong side. Thresholds here come from
    `operating_point`, which sits one float64 ULP away from a score by
    construction, so that demotion erases precisely the distinction being
    drawn. Callers cast; `evaluate` does it once at the top.
    """
    assert p.dtype == np.float64, "cast curves to float64 before thresholding"
    hit = np.flatnonzero(p > thresh)
    return float(hit[0] / fps) if hit.size else None


def operating_point(scores, y, *, recall=None, fpr=None):
    """Pick a threshold by recall or by false-positive rate.

    Returns (threshold, achieved_recall, achieved_fpr). Ties are handled by
    scanning distinct score values, so an all-tied submission gets the honest
    answer -- there is no threshold that buys recall without buying every
    false positive too.
    """
    s = np.asarray(scores, float)
    y = np.asarray(y, int)
    cand = np.unique(s)
    # thresholds strictly below each distinct value, plus one above all
    ths = np.r_[np.nextafter(cand, -np.inf), cand.max() + 1.0]
    best = None
    for t in ths:
        pred = s > t
        r = pred[y == 1].mean() if (y == 1).any() else 0.0
        f = pred[y == 0].mean() if (y == 0).any() else 0.0
        if recall is not None:
            if r >= recall - 1e-12 and (best is None or f < best[2]):
                best = (float(t), float(r), float(f))
        else:
            if f <= fpr + 1e-12 and (best is None or r > best[1]):
                best = (float(t), float(r), float(f))
    if best is None:                       # unreachable operating point
        return float(cand.max() + 1.0), 0.0, 0.0
    return best


def evaluate(curves, man, *, bound="event"):
    """Bounded and conditioned earliness for one set of risk curves."""
    ids = man.id.tolist()
    y = man.target.to_numpy().astype(int)
    # float64 throughout: see first_cross_time on why this is load-bearing.
    cur = {i: np.asarray(curves[i], np.float64) for i in ids}
    scores = np.array([cur[i].max() for i in ids], np.float64)
    pos = man[man.target == 1]

    def ceiling(r):
        return (float(r.t_event_s) if bound == "event"
                else max(float(r.t_event_s) - float(r.t_alert_s), EPS))

    def mean_norm_tta(th):
        """Returns (mean over detected, mean over ALL positives, n detected).

        Averaging over detected positives only is what the protocol said, and
        validating it here showed that to be a hole: a submission that detects
        5 positives of 85 at the allowed false-positive rate, all with a flat
        curve, scores a perfect 1.0 -- better than a model that detects 27 and
        warns genuinely early. Earliness you never deliver because you never
        raised the alarm is not earliness. The honest statistic charges an
        undetected positive zero, and it is the one the paper should report.
        """
        vals, n_det = [], 0
        for r in pos.itertuples(index=False):
            p = cur[r.id]
            if p.max() <= th:                    # never detected
                vals.append(0.0)
                continue
            n_det += 1
            tc = first_cross_time(p, th, r.fps)
            if tc is None or tc > r.t_event_s:   # crossed only after the event
                vals.append(0.0)
                continue
            vals.append(min((r.t_event_s - tc) / ceiling(r), 1.0))
        det = [v for v, r in zip(vals, pos.itertuples(index=False))
               if cur[r.id].max() > th]
        return (float(np.mean(det)) if det else 0.0,
                float(np.mean(vals)) if vals else 0.0, n_det)

    # Unconditioned: the threshold the published metric fixes at 0.5.
    bounded, bounded_all, n_b = mean_norm_tta(0.5)

    t80, r80, f80 = operating_point(scores, y, recall=0.80)
    c80, c80a, n80 = mean_norm_tta(t80)
    t10, r10, f10 = operating_point(scores, y, fpr=0.10)
    c10, c10a, n10 = mean_norm_tta(t10)

    return {
        "bounded@0.5": bounded, "bounded_all@0.5": bounded_all,
        "n_detected@0.5": n_b,
        "nTTA@R80": c80, "nTTA_all@R80": c80a,
        "fpr@R80": f80, "n_detected@R80": n80,
        "nTTA@FPR10": c10, "nTTA_all@FPR10": c10a,
        "recall@FPR10": r10, "n_detected@FPR10": n10,
    }


def _tests():
    import pandas as pd

    fps, n = 10.0, 200           # 20 s clips, event at 10 s
    man = pd.DataFrame({
        "id": [f"p{i}" for i in range(10)] + [f"n{i}" for i in range(10)],
        "fps": fps, "n_frames": n,
        "target": [1] * 10 + [0] * 10,
        "t_event_s": [10.0] * 10 + [np.nan] * 10,
        "t_alert_s": [8.0] * 10 + [np.nan] * 10,
    })

    # A constant above threshold: bounding leaves it at 1.0, exactly as the
    # paper says it should, and conditioning is what takes it away.
    const = {i: np.full(n, 0.51, np.float32) for i in man.id}
    r = evaluate(const, man)
    assert abs(r["bounded@0.5"] - 1.0) < 1e-9, r
    assert r["fpr@R80"] == 1.0, r              # 80% recall costs every negative
    assert r["n_detected@FPR10"] == 0, r       # cannot reach 10% FPR at all
    assert r["nTTA@FPR10"] == 0.0, r

    # A perfect discriminator that fires at 5 s on positives, never on
    # negatives: keeps its earliness under every conditioning.
    good = {}
    for i in man.id:
        p = np.full(n, 0.1, np.float32)
        if i.startswith("p"):
            p[int(5.0 * fps):] = 0.9
        good[i] = p
    r = evaluate(good, man)
    assert abs(r["bounded@0.5"] - 0.5) < 1e-9, r      # 5 s of 10 s
    assert abs(r["nTTA@FPR10"] - 0.5) < 1e-9, r
    assert r["fpr@R80"] == 0.0 and r["n_detected@FPR10"] == 10, r

    # Against the annotators' window the same curve is at its ceiling: it
    # fires 5 s before the event and only 2 s were ever actionable.
    r2 = evaluate(good, man, bound="alert")
    assert abs(r2["nTTA@FPR10"] - 1.0) < 1e-9, r2

    # Crossing only after the event earns nothing, even though it is detected.
    late = {}
    for i in man.id:
        p = np.full(n, 0.1, np.float32)
        if i.startswith("p"):
            p[int(15.0 * fps):] = 0.9
        late[i] = p
    r = evaluate(late, man)
    assert r["nTTA@FPR10"] == 0.0 and r["n_detected@FPR10"] == 10, r

    # Undetected positives are charged zero, so a submission cannot buy a
    # perfect earliness score by detecting almost nothing.
    stingy = {}
    for k, i in enumerate(man.id):
        p = np.full(n, 0.1, np.float32)
        if i.startswith("p") and k < 2:          # fires on 2 positives of 10
            p[:] = 0.9
        stingy[i] = p
    r = evaluate(stingy, man)
    assert r["n_detected@FPR10"] == 2, r
    assert abs(r["nTTA@FPR10"] - 1.0) < 1e-9, r          # over detected
    assert abs(r["nTTA_all@FPR10"] - 0.2) < 1e-9, r      # over all positives

    # The float32 trap, pinned: an operating point one float64 ULP below a
    # score must still detect that clip. In float32 it would not.
    fine = {}
    for i in man.id:
        p = np.full(n, 0.1, np.float32)
        if i.startswith("p"):
            p[int(5.0 * fps):] = np.float32(0.9)
        fine[i] = p
    th = np.nextafter(np.float64(np.float32(0.9)), -np.inf)
    assert first_cross_time(np.asarray(fine["p0"], np.float64), th, fps) == 5.0
    r = evaluate(fine, man)
    assert r["n_detected@FPR10"] == 10, r

    print("protocol.py: all self-tests pass")


if __name__ == "__main__":
    _tests()
