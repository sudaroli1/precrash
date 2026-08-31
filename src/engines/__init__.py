"""Scoring engines, imported lazily so that needing one does not require the
dependencies of the other two."""

from __future__ import annotations

__all__ = ["CLIPScorer", "FlowScorer", "NLPScorer"]

_MODULES = {
    "CLIPScorer": ".clip_scorer",
    "FlowScorer": ".flow_scorer",
    "NLPScorer": ".nlp_scorer",
}


def __getattr__(name: str):
    if name in _MODULES:
        from importlib import import_module
        return getattr(import_module(_MODULES[name], __package__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
