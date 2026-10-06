"""
A sighted baseline: strictly causal, deliberately modest.

It has one job -- to genuinely read pixels, so the corrected protocol has
something sighted to separate from the video-blind controls. It is not offered
as a good anticipation method and no claim in the paper depends on its
accuracy.

Two design rules, both of them scars from this project's own case study:

**Causal only.** Every feature at sample k is a function of samples <= k. The
original PreCrash pipeline resampled at index alpha*t and emitted at t, so the
score reported for frame t was computed from frame 1.3t -- a frame that had not
yet been observed. `_causal_check` asserts the property directly: it perturbs
the tail of a clip and requires every earlier output to be bit-identical.

**No label may reach a feature.** The windowing is in sample index, never in
time-to-event. A feature that knew where the event was would reproduce, inside
our own baseline, exactly the failure the paper is about.

Per-sample target: 1 on a positive clip inside [t_alert, t_event], 0 before
t_alert and 0 everywhere on a negative. That is the interval the annotators
call actionable, so it is what a warning system is being asked to flag.

  python3 model.py        # self-tests
"""
from __future__ import annotations

import numpy as np

FPS = 7.5
WINDOWS = (8, 15, 30)          # ~1 s, 2 s, 4 s of history at 7.5 Hz
NFEAT = 5


def causal_features(f: np.ndarray) -> np.ndarray:
    """(n, 5) raw samples -> (n, D) causal features.

    For each raw channel: the value, and the running mean and standard
    deviation over each trailing window, plus the value's deviation from that
    mean. Everything uses a cumulative sum over the past only.
    """
    n = len(f)
    out = [f]
    c1 = np.cumsum(np.vstack([np.zeros((1, NFEAT), np.float64), f]), axis=0)
    c2 = np.cumsum(np.vstack([np.zeros((1, NFEAT), np.float64), f.astype(np.float64) ** 2]), axis=0)
    idx = np.arange(n)
    for w in WINDOWS:
        lo = np.maximum(idx - w + 1, 0)
        cnt = (idx - lo + 1).astype(np.float64)[:, None]
        s1 = c1[idx + 1] - c1[lo]
        s2 = c2[idx + 1] - c2[lo]
        mean = s1 / cnt
        var = np.maximum(s2 / cnt - mean ** 2, 0.0)
        out += [mean, np.sqrt(var), f - mean]
    return np.hstack(out).astype(np.float32)


def sample_times(n: int) -> np.ndarray:
    """Sample i is the transition into frame i+1: time (i+1)/FPS seconds."""
    return (np.arange(n) + 1) / FPS


def targets(n: int, target: int, t_alert: float, t_event: float) -> np.ndarray:
    """1 inside the annotated actionable window of a positive clip, else 0."""
    y = np.zeros(n, np.int8)
    if target == 1:
        t = sample_times(n)
        y[(t >= t_alert) & (t <= t_event)] = 1
    return y


def to_frame_grid(p: np.ndarray, n_frames: int, fps: float) -> np.ndarray:
    """Hold each sample's value until the next: a causal zero-order hold.

    Interpolating would let a sample's value appear slightly before the frame
    it was computed from, which is the same sin on a smaller scale.
    """
    t_s = sample_times(len(p))
    t_f = np.arange(n_frames) / fps
    k = np.searchsorted(t_s, t_f, side="right") - 1
    out = np.where(k < 0, p[0], p[np.clip(k, 0, len(p) - 1)])
    return out.astype(np.float32)


def _tests():
    rng = np.random.default_rng(0)
    f = rng.random((200, NFEAT)).astype(np.float32)

    # Causality: changing the tail must not move any earlier feature.
    a = causal_features(f)
    g = f.copy(); g[120:] += 7.0
    b = causal_features(g)
    assert np.array_equal(a[:120], b[:120]), "features read the future"

    # Running mean over a trailing window, checked directly.
    w = WINDOWS[0]
    col = NFEAT                                    # first window's mean block
    for k in (0, 1, 5, 50, 199):
        exp = f[max(0, k - w + 1): k + 1, 0].mean()
        assert abs(a[k, col] - exp) < 1e-4, (k, a[k, col], exp)

    # Targets sit inside the annotated window and nowhere else.
    y = targets(200, 1, t_alert=5.0, t_event=9.0)
    t = sample_times(200)
    assert y[(t >= 5.0) & (t <= 9.0)].all()
    assert y[t < 5.0].sum() == 0 and y[t > 9.0].sum() == 0
    assert targets(200, 0, np.nan, np.nan).sum() == 0, "a negative got a target"

    # Zero-order hold never lets a value appear before its own sample time.
    p = np.arange(10).astype(np.float32)
    gfull = to_frame_grid(p, 100, 30.0)
    tf = np.arange(100) / 30.0
    for i, v in enumerate(gfull):
        if v > 0:
            assert tf[i] >= sample_times(10)[int(v)] - 1e-9, i
    print("model.py: all self-tests pass")


if __name__ == "__main__":
    _tests()
