"""
The repair: measure the alarm at FRAME level, not clip level.

Validating Section 5.3 on labelled data showed it is not sufficient. Its
conditioning step uses a CLIP-level discrimination metric, and a curve that is
flat within each clip can pass it: the metadata-only control, which reads the
container header and never decodes a frame, scored 0.093 against the sighted
model's 0.014, because every clip it detects at all hands it maximal earliness.
Bounding and conditioning between them never ask when the alarm is on.

Two frame-level quantities fix it, and both need the negatives:

  COVERAGE   fraction of the annotated actionable window [t_alert, t_event]
             during which the alarm is raised. What the warning delivers.
  DUTY       fraction of all non-actionable time during which the alarm is
             raised -- every frame of every negative clip, plus every frame of
             a positive before t_alert. What the warning costs.

A permanent alarm has coverage 1.0 and duty 1.0. A system that never fires has
0 and 0. Neither is useful, and only the pair says so; either number alone can
be bought for nothing.

The headline statistic is coverage at a capped duty cycle -- the earliness a
system delivers when it is allowed to be wrong only a stated fraction of the
time. `duty_hours` also reports it the way a driver would experience it:
minutes of alarm per hour of ordinary driving.

  python3 duty.py      # self-tests
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _partition(curves, man):
    """Split every frame of the corpus into actionable and non-actionable.

    ACTIONABLE      frames of a positive clip inside [t_alert, t_event]: the
                    interval the annotators call still-avoidable.
    NON-ACTIONABLE  every frame of every negative clip, plus frames of a
                    positive BEFORE t_alert.

    Frames after the collision belong to neither. Nothing is being anticipated
    there, and charging them would reward a curve for switching off after the
    event -- which no deployed system can do, since it does not know.
    """
    act, non = [], []
    for r in man.itertuples(index=False):
        p = np.asarray(curves[r.id], np.float64)
        if r.target == 1:
            t = np.arange(len(p)) / r.fps
            act.append(p[(t >= r.t_alert_s) & (t <= r.t_event_s)])
            non.append(p[t < r.t_alert_s])
        else:
            non.append(p)
    return np.concatenate(act), np.concatenate(non)


def _rate_above(sorted_scores, th):
    """Fraction of `sorted_scores` strictly greater than th."""
    if not len(sorted_scores):
        return 0.0
    return float(len(sorted_scores) -
                 np.searchsorted(sorted_scores, th, side="right")) / len(sorted_scores)


def evaluate(curves, man, duty_cap=0.10):
    """Coverage and duty across all thresholds, in one sort.

    Scanning distinct score values and re-walking every clip for each was
    O(thresholds x frames) and did not finish: the sighted model has ~90,000
    distinct values over ~360,000 frames. Sorting the two frame populations
    once and using searchsorted gives every threshold at once.
    """
    act, non = _partition(curves, man)
    act_s, non_s = np.sort(act), np.sort(non)
    cand = np.unique(np.concatenate([act_s, non_s]))
    ths = np.r_[np.nextafter(cand, -np.inf), cand.max() + 1.0]

    n_act, n_non = len(act_s), len(non_s)
    cov = (n_act - np.searchsorted(act_s, ths, side="right")) / max(n_act, 1)
    dut = (n_non - np.searchsorted(non_s, ths, side="right")) / max(n_non, 1)

    ok = dut <= duty_cap + 1e-12
    best_c = float(cov[ok].max()) if ok.any() else 0.0
    best_d = float(dut[ok][np.argmax(cov[ok])]) if ok.any() else 0.0

    c50, d50 = _rate_above(act_s, 0.5), _rate_above(non_s, 0.5)
    return {
        "coverage@0.5": c50, "duty@0.5": d50,
        f"coverage@duty{int(duty_cap*100)}": best_c,
        "duty_achieved": best_d,
        "alarm_min_per_hour@0.5": 60.0 * d50,
        "n_actionable_frames": n_act, "n_non_actionable_frames": n_non,
    }


def _tests():
    import pandas as pd
    fps, n = 10.0, 200
    man = pd.DataFrame({
        "id": [f"p{i}" for i in range(5)] + [f"n{i}" for i in range(5)],
        "fps": fps, "n_frames": n, "target": [1] * 5 + [0] * 5,
        "t_event_s": [10.0] * 5 + [np.nan] * 5,
        "t_alert_s": [8.0] * 5 + [np.nan] * 5,
    })

    # A permanent alarm: perfect coverage, and it pays for it in full.
    const = {i: np.full(n, 0.9) for i in man.id}
    r = evaluate(const, man)
    assert abs(r["coverage@0.5"] - 1.0) < 1e-9, r
    assert abs(r["duty@0.5"] - 1.0) < 1e-9, r
    assert abs(r["alarm_min_per_hour@0.5"] - 60.0) < 1e-9, r
    assert r["coverage@duty10"] == 0.0, r          # cannot reach the cap

    # Silence: no cost, no benefit.
    quiet = {i: np.full(n, 0.1) for i in man.id}
    r = evaluate(quiet, man)
    assert r["coverage@0.5"] == 0.0 and r["duty@0.5"] == 0.0

    # An oracle alarming exactly on the actionable window of positives.
    orc = {}
    for i in man.id:
        p = np.full(n, 0.1)
        if i.startswith("p"):
            t = np.arange(n) / fps
            p[(t >= 8.0) & (t <= 10.0)] = 0.9
        orc[i] = p
    r = evaluate(orc, man)
    assert abs(r["coverage@0.5"] - 1.0) < 1e-9, r
    assert abs(r["duty@0.5"] - 0.0) < 1e-9, r
    assert abs(r["coverage@duty10"] - 1.0) < 1e-9, r

    # Half the window covered, and a little leakage before t_alert.
    half = {}
    for i in man.id:
        p = np.full(n, 0.1)
        if i.startswith("p"):
            t = np.arange(n) / fps
            p[(t >= 9.0) & (t <= 10.0)] = 0.9
        half[i] = p
    r = evaluate(half, man)
    assert 0.45 < r["coverage@0.5"] < 0.55, r
    assert r["duty@0.5"] == 0.0, r
    print("duty.py: all self-tests pass")


if __name__ == "__main__":
    _tests()
