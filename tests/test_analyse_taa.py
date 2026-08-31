"""
Tests for the benchmark analysis, which is the part of the paper a reader can
check in a minute without downloading anything.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import analyse_taa as A  # noqa: E402


def test_constant_curve_sorts_first():
    """A crossing frame of 0 is falsy. Sorting with `tc or BIG` puts the
    constant curve last -- the one curve whose early crossing is the entire
    point. This pins the ordering."""
    fam = A.blind_family(150)
    order = sorted(fam.items(),
                   key=lambda kv: (float("inf") if A.crossing_frame(kv[1]) is None
                                   else A.crossing_frame(kv[1])))
    assert order[0][0] == "constant_0.51"
    assert A.crossing_frame(fam["constant_0.51"]) == 0


def test_blind_family_crossings_are_arithmetic():
    """These values follow from the definitions alone, so they can be stated in
    the paper without running anything on any corpus."""
    fam = A.blind_family(150)
    assert A.crossing_frame(fam["constant_0.51"]) == 0
    assert A.crossing_frame(fam["linear_ramp"]) == 75
    assert A.crossing_frame(fam["step_at_half"]) == 75
    assert A.crossing_frame(fam["sigmoid_m0.40"]) == 60
    assert A.crossing_frame(fam["sigmoid_m0.70"]) == 105


def test_curve_that_never_crosses_returns_none():
    assert A.crossing_frame(np.full(150, 0.4)) is None


@pytest.mark.parametrize("text", [
    "[0.001,0.0077,0.0144]",
    "[0.001, 0.0077, 0.0144]",
    "0.001,0.0077,0.0144",
    "[np.float64(0.001), np.float64(0.0077), np.float64(0.0144)]",
    "[1e-3, 7.7e-3, 1.44e-2]",
])
def test_risk_parsing_tolerates_formatting(text):
    """The risk column is a serialised array, and how it was serialised depends
    on whoever wrote it. Pull the numbers out; ignore the packaging."""
    v = A.parse_risk(text)
    assert len(v) == 3
    assert v[0] == pytest.approx(0.001, rel=1e-6)


def test_risk_parsing_rejects_empty():
    with pytest.raises(ValueError):
        A.parse_risk("[]")


def test_absence_detection_uses_aliases():
    """A missing label column is a finding about the benchmark. It must not be
    reported because someone spelled it `target`."""
    for field, names in A.ALIASES.items():
        assert len(names) > 1, f"{field} needs alternative spellings checked"
    assert "target" in A.ALIASES["label"]
    assert "time_of_event" in A.ALIASES["event_time"]
    assert "time_of_alert" in A.ALIASES["alert_time"]
