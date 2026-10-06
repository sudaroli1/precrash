"""
The video-blind control family, on Nexar.

Same shapes as the competition submissions, regenerated for a corpus whose
clips differ in length and frame rate. Each curve is a closed-form expression
in the frame index.

The one property that matters is enforced structurally rather than promised:
a generator is handed a `Clip` carrying ONLY n_frames and fps. It cannot see
the target or the event time, because they are not in the object. A curve that
needed them would raise AttributeError, not quietly cheat. That is the claim
the whole paper rests on, so it is worth making unfakeable.

  python3 controls.py      # self-tests
"""
from __future__ import annotations

from typing import NamedTuple
import numpy as np


class Clip(NamedTuple):
    """Everything a video-blind submission is allowed to know."""
    n_frames: int
    fps: float


def constant(v):
    return lambda c: np.full(c.n_frames, v, dtype=np.float32)


def ramp(lo=0.001, hi=0.999):
    return lambda c: np.linspace(lo, hi, c.n_frames, dtype=np.float32)


def step_at_second(s, lo=0.49, hi=0.51):
    def f(c):
        p = np.full(c.n_frames, lo, dtype=np.float32)
        k = int(round(s * c.fps))
        if k < c.n_frames:
            p[max(k, 0):] = hi
        return p
    return f


def step_at_fraction(fr, lo=0.49, hi=0.51):
    def f(c):
        p = np.full(c.n_frames, lo, dtype=np.float32)
        p[int(round(fr * c.n_frames)):] = hi
        return p
    return f


def cross_then_drop(s=0.0, drop_s=5.0):
    """Crosses early, then falls back below and stays there."""
    def f(c):
        p = np.full(c.n_frames, 0.51, dtype=np.float32)
        p[: int(round(s * c.fps))] = 0.49
        p[int(round(drop_s * c.fps)):] = 0.49
        return p
    return f


def cross_then_dip(s=0.0, dip_from=5.0, dip_to=6.0):
    """Crosses early, dips below for a window, recovers. Breaks STTA only."""
    def f(c):
        p = np.full(c.n_frames, 0.51, dtype=np.float32)
        p[: int(round(s * c.fps))] = 0.49
        p[int(round(dip_from * c.fps)): int(round(dip_to * c.fps))] = 0.49
        return p
    return f


FAMILY = {
    "constant_0.51": constant(0.51),
    "constant_0.99": constant(0.99),
    "never_crosses_0.49": constant(0.49),
    "ramp_0_to_1": ramp(),
    "cross_then_drop": cross_then_drop(),
    "cross_then_dip": cross_then_dip(),
    **{f"step_at_{s}s": step_at_second(s)
       for s in (0, 1, 2, 5, 10, 15, 20, 25, 30)},
    **{f"step_at_{int(fr*100)}pct": step_at_fraction(fr)
       for fr in (0.0, 0.25, 0.5, 0.75)},
}


def build(name, man):
    """id -> curve, for every clip in the manifest."""
    g = FAMILY[name]
    return {r.id: g(Clip(int(r.n_frames), float(r.fps)))
            for r in man.itertuples(index=False)}


def _tests():
    c = Clip(n_frames=300, fps=30.0)

    p = FAMILY["constant_0.51"](c)
    assert p.shape == (300,) and np.all(p == np.float32(0.51))

    p = FAMILY["step_at_5s"](c)
    assert np.all(p[:150] == np.float32(0.49)) and np.all(p[150:] == np.float32(0.51))

    # A step beyond the clip never crosses -- it must not wrap or crash.
    short = Clip(n_frames=60, fps=30.0)
    assert np.all(FAMILY["step_at_30s"](short) == np.float32(0.49))

    # step_at_0s and constant_0.51 must agree: both are above threshold always.
    assert np.array_equal(FAMILY["step_at_0s"](c) > 0.5,
                          FAMILY["constant_0.51"](c) > 0.5)

    p = FAMILY["cross_then_dip"](c)
    assert p[0] > 0.5 and p[160] < 0.5 and p[-1] > 0.5

    p = FAMILY["cross_then_drop"](c)
    assert p[0] > 0.5 and p[-1] < 0.5

    # The load-bearing one: no generator can reach a label.
    class Trap(Clip):
        def __getattr__(self, k):
            raise AssertionError(f"a control tried to read {k!r}")
    for name, g in FAMILY.items():
        out = g(Trap(300, 30.0))
        assert len(out) == 300, name
    print(f"controls.py: {len(FAMILY)} curves, all self-tests pass")


if __name__ == "__main__":
    _tests()
