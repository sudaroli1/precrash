"""
Diagnostics for one anomaly: three curves that cross 0.5 at frame 75 score
1.10435-1.10445, and a linear ramp that also crosses at 75 scores 1.06725.

Under the model in the paper -- every video-blind curve earns identical AP and
AUC, and both timing terms depend only on the crossing frame -- those four
numbers must be equal. They are not, and the gap is 0.0372, which is 2.24 frames
of slope: not an integer, so it is not a crossing-frame effect either.

Something other than the crossing frame is being read. These submissions find
out what. Each isolates one candidate while holding the crossing frame at 75.

    step_at_075   0.49 before, 0.51 after         (measured 1.10445)
    pre_graded    0.00 -> 0.49 before, 0.51 after  does the run-up matter?
    post_graded   0.49 before, 0.51 -> 0.99 after  does the run-out matter?
    both_graded   equals 02_linear_ramp in shape   should reproduce 1.06725
    step_at_076   0.49 before 76, 0.51 after       one frame, for calibration
    sample_exact  the organisers' own file         is 0.58333 really its score?

Read the four middle rows against step_at_075. If pre_graded differs, the metric
reads the curve before the alarm. If post_graded differs, it reads the curve
after. If neither differs but both_graded does, the effect is an interaction.
If none differs, the anomaly belongs to that one early submission and the
crossing-frame model stands.

  python scripts/make_diagnostics.py
"""
import csv, sys
from pathlib import Path

csv.field_size_limit(10**9)
SUB = Path(__file__).resolve().parent.parent / "submissions"
REF = SUB / "00_replicate_sample.csv"
N = 150
K = 75


def curves():
    lo, hi = 0.49, 0.51
    out = {}

    out["d_step_at_075"] = [lo] * K + [hi] * (N - K)
    out["d_step_at_076"] = [lo] * (K + 1) + [hi] * (N - K - 1)
    out["d_pre_graded"] = [0.49 * t / (K - 1) for t in range(K)] + [hi] * (N - K)
    out["d_post_graded"] = [lo] * K + [
        0.51 + 0.48 * t / (N - K - 1) for t in range(N - K)
    ]
    out["d_both_graded"] = [t / (N - 1) for t in range(N)]
    return out


def write(name, values, ids):
    assert len(values) == N, name
    assert all(0.0 <= v <= 1.0 for v in values), name
    cross = next((i for i, v in enumerate(values) if v > 0.5), None)
    payload = "[" + ",".join(repr(float(v)) for v in values) + "]"
    path = SUB / f"{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "risk"])
        for cid in ids:
            w.writerow([cid, payload])
    print(f"  {name:18s} crosses at {cross}  ->  {path.name}")
    return cross


def main():
    if not REF.exists():
        raise SystemExit(f"missing reference file: {REF}")
    with open(REF) as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    print(f"id order taken from {REF.name}: {len(ids)} clips\n")

    crossings = {}
    for name, vals in curves().items():
        crossings[name] = write(name, vals, ids)

    bad = {k: v for k, v in crossings.items() if v != K and k != "d_step_at_076"}
    if bad:
        raise SystemExit(f"\nFAILED: these should all cross at {K}: {bad}")
    if crossings["d_step_at_076"] != K + 1:
        raise SystemExit("FAILED: d_step_at_076 does not cross at 76")

    print(f"\nAll five hold the crossing frame at {K} except d_step_at_076 at {K+1}.")
    print("Also submit submissions/00_replicate_sample.csv unchanged: it is the")
    print("organisers' own file, and its score settles whether the 0.58333")
    print("benchmark row on the leaderboard is that file or something else.")


if __name__ == "__main__":
    main()
