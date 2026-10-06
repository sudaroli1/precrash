"""
Build the clip manifest: fps, frame count, and the event frame index.

Nexar annotates time_of_event and time_of_alert in SECONDS, and the clips do
not share a frame rate (28.9 and 30.0 both occur) or a length. Every later
stage needs the event as a frame index in that clip's own timebase, so it is
resolved once, here, rather than by each consumer assuming 30 fps -- which is
the exact class of error this paper is about.

Negatives carry no event. Their timing terms are undefined, not zero; they are
left as NaN so that any code which forgets to exclude them fails loudly
instead of silently scoring them.

  python3 prep.py
"""
import cv2, numpy as np, pandas as pd, pathlib, sys

ROOT = pathlib.Path(__file__).parent
DATA = ROOT.parent.parent / "nexar-collision-prediction"
OUT = ROOT / "out"

def main():
    lab = pd.read_csv(DATA / "train.csv", dtype={"id": str})
    rows = []
    for i, r in enumerate(lab.itertuples(index=False)):
        f = DATA / "train" / f"{r.id}.mp4"
        if not f.exists():
            rows.append((r.id, np.nan, np.nan, r.target, np.nan, np.nan, np.nan))
            continue
        cap = cv2.VideoCapture(str(f))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        ev = float(r.time_of_event) if pd.notna(r.time_of_event) else np.nan
        al = float(r.time_of_alert) if pd.notna(r.time_of_alert) else np.nan
        rows.append((r.id, fps, n, int(r.target), ev, al,
                     ev * fps if ev == ev else np.nan))
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(lab)}", flush=True)

    m = pd.DataFrame(rows, columns=["id", "fps", "n_frames", "target",
                                    "t_event_s", "t_alert_s", "f_event"])
    # Sanity: the event must fall inside the clip, or the timebase is wrong.
    pos = m[m.target == 1]
    bad = pos[(pos.f_event < 0) | (pos.f_event >= pos.n_frames)]
    print(f"\nclips={len(m)} pos={int((m.target==1).sum())} "
          f"neg={int((m.target==0).sum())}")
    print(f"missing files: {int(m.fps.isna().sum())}")
    print(f"events outside their clip: {len(bad)}")
    print(f"fps values: {sorted(m.fps.dropna().round(2).unique().tolist())}")
    print(f"duration s: min={ (m.n_frames/m.fps).min():.1f} "
          f"median={(m.n_frames/m.fps).median():.1f} "
          f"max={(m.n_frames/m.fps).max():.1f}")
    print(f"event frame: min={pos.f_event.min():.0f} "
          f"median={pos.f_event.median():.0f} max={pos.f_event.max():.0f}")
    # alert must precede event
    print(f"alert >= event: {int((pos.t_alert_s >= pos.t_event_s).sum())}")

    OUT.mkdir(exist_ok=True)
    m.to_csv(OUT / "manifest.csv", index=False)
    print(f"\nwrote {OUT/'manifest.csv'}")

if __name__ == "__main__":
    main()
