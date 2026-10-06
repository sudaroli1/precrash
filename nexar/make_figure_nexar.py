"""
The warning operating characteristic.

Coverage of the annotated actionable window against duty cycle over all
non-actionable time, swept over every threshold. It is the figure the published
metric cannot draw, because its timing terms are defined only where an accident
occurs and so it never asks what the alarm costs when nothing is happening.

A permanent alarm sits at the top right corner: it covers everything and it
alarms always. The published composite scores that submission at the maximum;
this figure shows why that is the wrong answer, in one picture.

  python3 make_figure_nexar.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, pathlib
import controls, duty, model
from logreg import LogReg

ROOT = pathlib.Path(__file__).parent
OUT, FEAT = ROOT / "out", ROOT / "out" / "f75"


def sweep(curves, man):
    act, non = duty._partition(curves, man)
    a, n = np.sort(act), np.sort(non)
    cand = np.unique(np.concatenate([a, n]))
    ths = np.r_[-np.inf, np.nextafter(cand, -np.inf), cand.max() + 1.0]
    cov = (len(a) - np.searchsorted(a, ths, "right")) / len(a)
    dut = (len(n) - np.searchsorted(n, ths, "right")) / len(n)
    o = np.argsort(dut)
    return dut[o], cov[o]


def main():
    man = pd.read_csv(OUT / "manifest.csv", dtype={"id": str})
    have = {p.stem for p in FEAT.glob("*.npz")}
    man = man[(man.split != "unused") & (man.id.isin(have))]
    tr = man[man.split == "train"]
    te = man[man.split == "test"].reset_index(drop=True)

    Xs, ys = [], []
    for r in tr.itertuples(index=False):
        f = np.load(FEAT / f"{r.id}.npz")["f"]
        Xs.append(model.causal_features(f))
        ys.append(model.targets(len(f), r.target, r.t_alert_s, r.t_event_s))
    clf = LogReg(l2=10.0).fit(np.vstack(Xs), np.concatenate(ys))
    sighted = {}
    for r in te.itertuples(index=False):
        f = np.load(FEAT / f"{r.id}.npz")["f"]
        p = clf.predict_proba(model.causal_features(f))
        sighted[r.id] = model.to_frame_grid(p, int(r.n_frames), float(r.fps))

    mfeat = lambda d: np.c_[d.fps.to_numpy(float), (d.n_frames/d.fps).to_numpy(float)]
    mclf = LogReg(l2=1.0).fit(mfeat(tr), tr.target.to_numpy(float))
    mp = mclf.predict_proba(mfeat(te))
    meta = {r.id: np.full(int(r.n_frames), mp[i], np.float32)
            for i, r in enumerate(te.itertuples(index=False))}

    series = [
        ("risk model (reads the road)", sighted, "#1f77b4", "-", 2.0),
        ("metadata only (reads the file header)", meta, "#d62728", "--", 1.6),
        ("linear ramp (reads nothing)", controls.build("ramp_0_to_1", te),
         "#7f7f7f", "-.", 1.4),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.9), dpi=300)
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.8, zorder=1)
    for lab, cur, c, ls, lw in series:
        d, v = sweep(cur, te)
        ax.plot(d, v, color=c, ls=ls, lw=lw, label=lab, zorder=3)

    dc, vc = sweep(controls.build("constant_0.51", te), te)
    ax.plot([1], [1], marker="o", ms=7, mfc="#ff7f0e", mec="black", mew=0.8,
            ls="none", zorder=5, label="constant 0.51 (reads nothing)")
    ax.annotate("a warning that never\nturns off: the published\nmetric's maximum",
                xy=(1.0, 1.0), xytext=(0.47, 0.72), fontsize=6.0,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))

    ax.axvline(0.10, color="#999999", lw=0.7, ls=":", zorder=2)
    ax.text(0.115, 0.03, "duty cap 10%", fontsize=5.8, color="#555555")

    ax.set_xlabel("duty cycle: share of non-actionable time alarming", fontsize=7)
    ax.set_ylabel("coverage: share of the actionable\nwindow alarming", fontsize=7)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=5.8, loc="lower right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_woc.{ext}", bbox_inches="tight")
    print(f"wrote {OUT/'fig_woc.png'} and .pdf")


if __name__ == "__main__":
    main()
