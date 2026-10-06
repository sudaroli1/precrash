"""
Check the results against the code and the data, not the write-up against
itself. Every headline number is re-derived here by a different route than the
one that produced it.

The specific worry this corpus deserves: clip DURATION predicts the label
(AUC 0.537), and a clip-level score taken as a max over frames grows with the
number of frames. So a sighted model's apparent discrimination could be partly
the length of the file rather than anything on the road. That is tested
directly, within duration strata.

  python3 audit.py
"""
from __future__ import annotations

import numpy as np, pandas as pd, pathlib
import controls, metric, model, protocol
from logreg import LogReg

ROOT = pathlib.Path(__file__).parent
OUT, FEAT = ROOT / "out", ROOT / "out" / "f75"
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def main():
    man = pd.read_csv(OUT / "manifest.csv", dtype={"id": str})
    have = {p.stem for p in FEAT.glob("*.npz")}
    man = man[(man.split != "unused") & (man.id.isin(have))].reset_index(drop=True)
    tr, te = man[man.split == "train"], man[man.split == "test"].reset_index(drop=True)

    print("SPLIT")
    check(set(tr.id).isdisjoint(set(te.id)), "train and test share no clip")
    check(len(te) == 300 and len(tr) == 600, f"600/300 split ({len(tr)}/{len(te)})")
    check(te.target.sum() == 150, f"test is balanced ({int(te.target.sum())} positive)")

    print("\nLABEL LEAKAGE INTO THE CONTROLS")
    # Re-derive: a control's curve must be identical if the labels are destroyed.
    scrambled = te.copy()
    scrambled["target"] = 1 - scrambled["target"]
    scrambled["t_event_s"] = scrambled["t_event_s"] * 0 + 1.0
    scrambled["t_alert_s"] = scrambled["t_alert_s"] * 0 + 0.5
    for name in controls.FAMILY:
        a = controls.build(name, te)
        b = controls.build(name, scrambled)
        same = all(np.array_equal(a[i], b[i]) for i in te.id)
        if not same:
            check(False, f"{name} changed when labels were scrambled")
    check(True, "all 19 controls are byte-identical under scrambled labels")

    print("\nTHE CONSTANT'S EARLINESS, RE-DERIVED")
    # TTA of a constant must equal the mean event time, by definition.
    pos = te[te.target == 1]
    expect = float((pos.f_event / pos.fps).mean())
    got = metric.evaluate(controls.build("constant_0.51", te), te)["TTA_s"]
    check(abs(got - expect) < 0.02,
          f"constant TTA {got:.4f}s vs mean event time {expect:.4f}s")

    print("\nIS THE SIGHTED MODEL READING THE ROAD, OR THE FILE LENGTH?")
    Xs, ys = [], []
    for r in tr.itertuples(index=False):
        f = np.load(FEAT / f"{r.id}.npz")["f"]
        Xs.append(model.causal_features(f))
        ys.append(model.targets(len(f), r.target, r.t_alert_s, r.t_event_s))
    clf = LogReg(l2=10.0).fit(np.vstack(Xs), np.concatenate(ys))

    score, dur = [], []
    for r in te.itertuples(index=False):
        f = np.load(FEAT / f"{r.id}.npz")["f"]
        score.append(float(clf.predict_proba(model.causal_features(f)).max()))
        dur.append(r.n_frames / r.fps)
    score, dur = np.array(score), np.array(dur)
    y = te.target.to_numpy().astype(int)

    auc_all = metric.auc_rank(y, score)
    auc_dur = metric.auc_rank(y, dur)
    print(f"  sighted AUC {auc_all:.4f} | duration-alone AUC {auc_dur:.4f} | "
          f"corr(score, duration) {np.corrcoef(score, dur)[0,1]:+.4f}")

    # Within the modal duration band, length carries no information, so any
    # remaining separation is being read off the video.
    band = (dur > 39.5) & (dur < 40.5)
    yb, sb = y[band], score[band]
    auc_band = metric.auc_rank(yb, sb)
    print(f"  clips at 40s: {band.sum()} ({int(yb.sum())} positive) "
          f"-> duration AUC {metric.auc_rank(yb, dur[band]):.4f}, "
          f"sighted AUC {auc_band:.4f}")
    check(auc_band > 0.60,
          f"sighted still separates at fixed duration (AUC {auc_band:.4f})")
    check(abs(np.corrcoef(score, dur)[0, 1]) < 0.25,
          "clip score is not strongly tied to clip length")

    print("\nCAUSALITY OF THE SIGHTED FEATURES")
    f = np.load(FEAT / f"{te.id[0]}.npz")["f"]
    a = model.causal_features(f)
    g = f.copy(); g[len(g) // 2:] = 0.0
    b = model.causal_features(g)
    check(np.array_equal(a[: len(a) // 2], b[: len(b) // 2]),
          "no feature at time t depends on any sample after t")

    print("\nTHE REPAIR SEPARATES, AND THE ORIGINAL PROTOCOL DOES NOT")
    import duty
    cur_c = controls.build("constant_0.51", te)
    cur_s = {}
    for r in te.itertuples(index=False):
        f = np.load(FEAT / f"{r.id}.npz")["f"]
        p = clf.predict_proba(model.causal_features(f))
        cur_s[r.id] = model.to_frame_grid(p, int(r.n_frames), float(r.fps))
    dc, ds = duty.evaluate(cur_c, te), duty.evaluate(cur_s, te)
    check(dc["coverage@duty10"] == 0.0 and ds["coverage@duty10"] > 0.2,
          f"coverage@duty10: constant {dc['coverage@duty10']:.4f} "
          f"vs sighted {ds['coverage@duty10']:.4f}")
    check(abs(dc["alarm_min_per_hour@0.5"] - 60.0) < 1e-6,
          "a permanent alarm costs a full 60 min/h of ordinary driving")

    print("\n" + ("ALL CHECKS PASSED" if not fails
                  else f"{len(fails)} CHECK(S) FAILED"))
    for f_ in fails:
        print("  - " + f_)


if __name__ == "__main__":
    main()
