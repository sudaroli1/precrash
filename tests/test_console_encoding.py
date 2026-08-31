"""
Console output must survive a Windows terminal.

The first local run of this repository died with

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'

partway through printing the central comparison table, because the header
contained a real arrow and the console codec was cp1252. The rows already
printed looked fine, so the failure arrived without warning halfway down a
table.

Two properties are pinned here:

  1. Everything a script prints is ASCII. Alignment matters in these tables, and
     a replacement character shifts a column just as badly as a crash ends one.
  2. Encoding is nonetheless never allowed to raise, so that a non-ASCII
     filename or a caption from the corpus cannot end a long run.
"""

import io
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_help_text_is_ascii(script):
    """--help renders on any console. argparse prints the help strings, so a
    non-ASCII character in one is a crash on Windows before the tool has done
    anything at all."""
    r = subprocess.run([sys.executable, str(script), "--help"],
                       capture_output=True, text=True, cwd=ROOT)
    out = r.stdout + r.stderr
    offenders = sorted({c for c in out if ord(c) > 127})
    assert not offenders, (
        f"{script.name} --help emits non-ASCII: "
        + ", ".join(f"{c!r} (U+{ord(c):04X})" for c in offenders)
    )


def test_safe_console_survives_a_cp1252_stream():
    sys.path.insert(0, str(ROOT / "src"))
    from utils.console import safe_console

    buf = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    real_out, real_err = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = buf
        safe_console()
        print("arrow → delta Δ dash —")   # would raise before
        buf.flush()
    finally:
        sys.stdout, sys.stderr = real_out, real_err


def test_safe_console_is_idempotent_and_tolerates_odd_streams():
    sys.path.insert(0, str(ROOT / "src"))
    from utils.console import safe_console

    real_out, real_err = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = io.StringIO()   # no reconfigure attribute
        safe_console()
        safe_console()
    finally:
        sys.stdout, sys.stderr = real_out, real_err


def test_runtime_output_is_ascii(tmp_path):
    """Not just --help: the tables themselves. This is the path that actually
    failed, and it fails only once real output starts, which is why --help
    passing is not enough."""
    import os

    feats = tmp_path / "synth"
    subprocess.run([sys.executable, "scripts/make_synthetic_features.py",
                    "--manifest", "data/nexar_fixed.dev.csv",
                    "--out_dir", str(feats), "--limit", "40"],
                   cwd=ROOT, capture_output=True, check=True)

    env = dict(os.environ, PYTHONIOENCODING="ascii")
    r = subprocess.run([sys.executable, "scripts/protocol_comparison.py",
                        "--features", str(feats), "--config", "configs/honest.yaml",
                        "--out", str(tmp_path / "p.json"),
                        "--dataset_name", "SYNTHETIC"],
                       cwd=ROOT, capture_output=True, env=env)

    assert r.returncode == 0, r.stderr.decode("ascii", errors="replace")[-1500:]
    out = r.stdout.decode("ascii", errors="replace")
    offenders = sorted({c for c in out if ord(c) > 127})
    assert not offenders, (
        "comparison table emits non-ASCII: "
        + ", ".join(f"{c!r} (U+{ord(c):04X})" for c in offenders)
    )
