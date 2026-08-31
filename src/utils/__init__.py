"""
Utility exports, imported lazily.

Eager imports here used to pull video_io -> cv2 into every consumer, which
meant the pure-numpy evaluation code could not run without the full video
stack installed. That defeats the split between the CPU analysis environment
(requirements-analysis.txt) and the GPU extraction environment
(requirements-extract.txt).

PEP 562 module __getattr__ keeps the same public names available while
deferring each import until the name is actually used.
"""

from __future__ import annotations

__all__ = [
    "load_frames",
    "AnticipationMetrics",
    "evaluate_anticipation",
    "bootstrap_ci",
    "paired_bootstrap_test",
]

_METRICS = {
    "AnticipationMetrics",
    "evaluate_anticipation",
    "bootstrap_ci",
    "paired_bootstrap_test",
}


def __getattr__(name: str):
    if name == "load_frames":
        from .video_io import load_frames   # needs opencv
        return load_frames
    if name in _METRICS:
        from . import metrics               # numpy + scikit-learn only
        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
