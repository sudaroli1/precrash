"""
Score everything on the same held-out clips, under both metrics.

Rows compared, all on the 300-clip test split:

  video-blind family   the controls of Section 3.2, reading nothing
  metadata-only        reads the FILE's frame rate and duration, no pixels.
                       This corpus leaks the label through both (AUC 0.533 and
                       0.537 on their own), so a submission that opens the
                       container and never decodes it beats chance. It is the
                       control the paper did not know it needed.
  sighted              causal logistic regression on motion features

Columns: the published composite's four terms, and the corrected protocol's
bounded and conditioned earliness.

The prediction that is actually under test is not "the sighted model wins".
It is the pattern Section 5.3 claims: under BOUNDING the constant still scores
1.0, and only under CONDITIONING does it collapse while the sighted model
survives. A result where the constant failed at the bounding step would mean
the implementation, or the paper's analysis, is wrong.

  python3 train_eval.py
"""
from __future__ import annotations

import numpy as np, pandas as pd, pathlib
import controls, duty, metric, model, protocol
from logreg import LogReg

ROOT = pathlib.Path(__file__).parent
OUT, FEAT = ROOT / "out", ROOT / "out" / "f75"


def load(cid):
    z = np.load(FEAT / f"{cid}.npz")
    return z["f"]


def build_sighted(man):
    tr = man[man.split == "train"]
    te = man[man.split == "test"]

    Xs, ys = [], []
    for r in tr.itertuples(index=False):
        f = load(r.id)
        Xs.append(model.causal_features(f))
        ys.append(model.targets(len(f), r.target, r.t_alert_s, r.t_event_s))
    X, y = np.vstack(Xs), np.concatenate(ys)
    print(f"  train: {len(tr)} clips, {len(X):,} samples, "
          f"{100*y.mean():.1f}% positive")
    clf = LogReg(l2=10.0).fit(X, y)

    cur = {}
    for r in te.itertuples(index=False):
        f = load(r.id)
        p = clf.predict_proba(model.causal_features(f))
        cur[r.id] = model.to_frame_grid(p, int(r.n_frames), float(r.fps))
    return clf, cur


def build_metadata_only(man):
    """Reads the container header. Never opens a frame."""
    tr, te = man[man.split == "train"], man[man.split == "test"]
    feat = lambda d: np.c_[d.fps.to_numpy(float),
                           (d.n_frames / d.fps).to_numpy(float)]
    clf = LogReg(l2=1.0).fit(feat(tr), tr.target.to_numpy(float))
    p = clf.predict_proba(feat(te))
    # One value per clip, held flat: it has no temporal information at all,
    # which means it collects the full earliness term like any constant.
    return {r.id: np.full(int(r.n_frames), p[i], np.float32)
            for i, r in enumerate(te.itertuples(index=False))}


def main():
    man = pd.read_csv(OUT / "manifest.csv", dtype={"id": str})
    have = {p.stem for p in FEAT.glob("*.npz")}
    man = man[(man.split != "unused") & (man.id.isin(have))].reset_index(drop=True)
    te = man[man.split == "test"].reset_index(drop=True)
    print(f"features present: {len(man)} of 900 "
          f"({(man.split=='train').sum()} train, {len(te)} test)")

    subs = {}
    for name in ("constant_0.51", "ramp_0_to_1", "step_at_50pct",
                 "never_crosses_0.49"):
        subs[name] = controls.build(name, te)
    subs["metadata_only"] = build_metadata_only(man)
    _, subs["sighted_causal"] = build_sighted(man)

    rows = []
    for name, cur in subs.items():
        m = metric.evaluate(cur, te)
        p_ev = protocol.evaluate(cur, te, bound="event")
        p_al = protocol.evaluate(cur, te, bound="alert")
        dy = duty.evaluate(cur, te)
        rows.append({
            "submission": name, "AP": m["AP"], "AUC": m["AUC"],
            "TTA_s": m["TTA_s"], "STTA_s": m["STTA_s"],
            "bounded": p_ev["bounded_all@0.5"],
            "nTTA@R80": p_ev["nTTA_all@R80"], "fpr@R80": p_ev["fpr@R80"],
            "nTTA@FPR10": p_ev["nTTA_all@FPR10"],
            "recall@FPR10": p_ev["recall@FPR10"],
            "n_det@FPR10": p_ev["n_detected@FPR10"],
            "nTTA@FPR10_alert": p_al["nTTA_all@FPR10"],
            "coverage@0.5": dy["coverage@0.5"], "duty@0.5": dy["duty@0.5"],
            "alarm_min_per_h": dy["alarm_min_per_hour@0.5"],
            "coverage@duty10": dy["coverage@duty10"],
        })
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "test_scores.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nPUBLISHED METRIC")
    print(d[["submission", "AP", "AUC", "TTA_s", "STTA_s"]].round(4).to_string(index=False))
    print("\nCORRECTED PROTOCOL")
    print(d[["submission", "bounded", "nTTA@R80", "fpr@R80",
             "nTTA@FPR10", "recall@FPR10", "n_det@FPR10",
             "nTTA@FPR10_alert"]]
          .round(4).to_string(index=False))
    print("\nFRAME-LEVEL ALARM (the repair)")
    print(d[["submission", "coverage@0.5", "duty@0.5", "alarm_min_per_h",
             "coverage@duty10"]].round(4).to_string(index=False))
    print(f"\nwrote {OUT/'test_scores.csv'}")


if __name__ == "__main__":
    main()
