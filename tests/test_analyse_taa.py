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


# ----------------------------------------------------------------------
# Submission building
# ----------------------------------------------------------------------

def test_submission_matches_the_reference_format(tmp_path):
    """The competition names two failure modes: an id mismatch against test.csv,
    and a risk array whose length is not 150. Both are checked before writing,
    because a rejected submission costs a quota slot."""
    import csv as _csv
    import subprocess

    ids = [f"11_00{i:04d}_1_151" for i in range(20)]
    ramp = np.linspace(0.001, 0.999, 150)
    risk = "[" + ",".join(f"{v:.6f}".rstrip("0").rstrip(".") for v in ramp) + "]"

    test_csv = tmp_path / "test.csv"
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "video_id", "start_frame",
                                           "end_frame", "caption"])
        w.writeheader()
        for i in ids:
            w.writerow({"id": i, "video_id": "11/0001", "start_frame": 1,
                        "end_frame": 151, "caption": "x"})

    sample = tmp_path / "sample.csv"
    with open(sample, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "risk"])
        w.writeheader()
        for i in ids:
            w.writerow({"id": i, "risk": risk})

    out = tmp_path / "sub.csv"
    r = subprocess.run([sys.executable, "scripts/make_submission.py",
                        "--test_csv", str(test_csv), "--sample_csv", str(sample),
                        "--replicate_sample", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1000:]
    assert "byte-identical" in r.stdout, r.stdout

    rows = list(_csv.DictReader(open(out, encoding="utf-8")))
    assert [x["id"] for x in rows] == ids
    assert all(len(A.parse_risk(x["risk"])) == 150 for x in rows)


def test_submission_refuses_incomplete_feature_coverage(tmp_path):
    """A submission must cover every id. Padding the gaps with a constant would
    quietly mix a real method with a blind one -- which, in this paper above
    all, must not be possible by accident."""
    import csv as _csv
    import subprocess

    ids = [f"11_00{i:04d}_1_151" for i in range(5)]
    ramp = np.linspace(0.001, 0.999, 150)
    risk = "[" + ",".join(str(round(v, 6)) for v in ramp) + "]"
    with open(tmp_path / "test.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "video_id", "start_frame",
                                           "end_frame", "caption"])
        w.writeheader()
        for i in ids:
            w.writerow({"id": i, "video_id": "11/0001", "start_frame": 1,
                        "end_frame": 151, "caption": "x"})
    with open(tmp_path / "sample.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["id", "risk"])
        w.writeheader()
        for i in ids:
            w.writerow({"id": i, "risk": risk})

    feats = tmp_path / "feats"; feats.mkdir()
    np.savez(feats / f"{ids[0]}.npz", p_clip=np.zeros(150), p_flow=np.zeros(150),
             p_nlp=np.zeros(150))          # only 1 of 5 clips

    r = subprocess.run([sys.executable, "scripts/make_submission.py",
                        "--test_csv", str(tmp_path / "test.csv"),
                        "--sample_csv", str(tmp_path / "sample.csv"),
                        "--features", str(feats), "--out", str(tmp_path / "s.csv")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "have no features" in (r.stdout + r.stderr)
    assert not (tmp_path / "s.csv").exists()
