"""PreCrash package. The pipeline import is deferred: it pulls torch, CLIP and
sentence-transformers, none of which the evaluation scripts need."""

from __future__ import annotations

__all__ = ["PreCrashPipeline"]
__version__ = "1.1.0"


def __getattr__(name: str):
    if name == "PreCrashPipeline":
        from .pipeline import PreCrashPipeline
        return PreCrashPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
