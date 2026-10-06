"""
Score the video-blind family on all 1,500 labelled Nexar clips.

This is the result the AAP referee asked for: the metric pathology reproduced
on a second corpus, independent of the competition, with ground truth that
makes every term computable rather than inferred from a leaderboard.

Two things are possible here that were not possible there:

  * AP and AUC are real numbers, not a floor recovered by differencing.
  * The timing terms are measured to the ANNOTATED EVENT, not to the end of
    the window -- which is the measurement error the paper is about.

  python3 run_controls.py
"""
import numpy as np, pandas as pd, pathlib
import controls, metric

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"


def main():
    man = pd.read_csv(OUT / "manifest.csv", dtype={"id": str})
    rows = []
    for name in controls.FAMILY:
        cur = controls.build(name, man)
        r = metric.evaluate(cur, man)
        r["submission"] = name
        rows.append(r)
        print(f"  {name:22s} AP={r['AP']:.4f} AUC={r['AUC']:.4f} "
              f"TTA={r['TTA_s']:6.2f}s STTA={r['STTA_s']:6.2f}s", flush=True)

    d = pd.DataFrame(rows)[["submission", "AP", "AUC",
                            "TTA_s", "STTA_s", "TTA_f", "STTA_f"]]
    d.to_csv(OUT / "controls_scores.csv", index=False)

    pos = man[man.target == 1]
    ceiling_s = (pos.f_event / pos.fps).mean()
    print(f"\nMean event time over positives: {ceiling_s:.2f} s")
    print("That is the maximum the earliness terms admit, and a constant "
          "above\nthreshold attains it on every clip.")

    c = d[d.submission == "constant_0.51"].iloc[0]
    print(f"\nconstant_0.51: TTA={c.TTA_s:.2f}s STTA={c.STTA_s:.2f}s "
          f"AP={c.AP:.4f} AUC={c.AUC:.4f}")
    print(f"  TTA reaches {100*c.TTA_s/ceiling_s:.1f}% of the ceiling, "
          f"STTA {100*c.STTA_s/ceiling_s:.1f}%")

    # The scale argument, now measurable WITHIN one corpus rather than across
    # two: the unbounded term tracks how long the clip happened to be.
    pos = pos.assign(ev_s=pos.f_event / pos.fps)
    q = pos.ev_s.quantile([0, .25, .5, .75, 1.0]).round(2)
    print(f"\nEvent time across positives (s): {q.tolist()}")
    print(f"  shortest {pos.ev_s.min():.2f}s vs longest {pos.ev_s.max():.2f}s "
          f"= a {pos.ev_s.max()/pos.ev_s.min():.1f}x spread in the maximum "
          f"earliness\n  a single constant can collect, within one corpus.")
    print(f"\nwrote {OUT/'controls_scores.csv'}")


if __name__ == "__main__":
    main()
