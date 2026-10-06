"""
Per-clip motion features on a common 7.5 Hz timebase.

Sampling is by TIME, not by frame stride. This corpus has 36 distinct frame
rates between 23.6 and 31.0, so "every 4th frame" is a different interval in
almost every clip. ffmpeg's fps filter resamples by timestamp, so sample k is
at k/7.5 seconds in every clip, which is what makes the series comparable.

Features per sample:

  d_mean    mean |frame - prev|                    bulk motion
  d_p90     90th percentile of |diff|              localised motion
  d_centre  mean |diff| over the lower-centre box   the ego lane
  lum       mean luminance                         exposure, tunnels, night
  d_lum     |lum - prev lum|                       cuts and lighting steps

Two sandbox facts shape the rest. Nothing survives the end of a shell call, so
the job is deadline-bounded and resumable: it stops before the call times out
and the next run continues. And multiprocessing.Pool hangs here (it wants
POSIX semaphores on /dev/shm), so parallelism is threads -- enough, because the
decode runs inside ffmpeg, a separate process.

  python3 extract.py <budget_seconds> [workers]
"""
import numpy as np, pandas as pd, pathlib, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).parent
DATA = ROOT.parent.parent / "nexar-collision-prediction"
FEAT = ROOT / "out" / "f75"
W, H, FPS = 160, 90, 7.5


def one(cid):
    out = FEAT / f"{cid}.npz"
    if out.exists():
        return cid, -1
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-threads", "1",
         "-i", str(DATA / "train" / f"{cid}.mp4"),
         "-vf", f"fps={FPS},scale={W}:{H}", "-pix_fmt", "gray",
         "-f", "rawvideo", "-"], capture_output=True)
    n = len(r.stdout) // (W * H)
    if n < 2:
        return cid, 0
    a = np.frombuffer(r.stdout[: n * W * H], np.uint8).reshape(n, H, W)
    d = np.abs(a[1:].astype(np.int16) - a[:-1].astype(np.int16)).astype(np.float32)
    lum = a.astype(np.float32).mean(axis=(1, 2))
    f = np.stack([
        d.mean(axis=(1, 2)),
        np.percentile(d, 90, axis=(1, 2)),
        d[:, H // 2:, W // 4: 3 * W // 4].mean(axis=(1, 2)),
        lum[1:],
        np.abs(np.diff(lum)),
    ], axis=1).astype(np.float32)
    # Sample i is the transition into frame i+1, i.e. time (i+1)/FPS seconds.
    np.savez_compressed(out, f=f, fps=np.float32(FPS))
    return cid, len(f)


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 140
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    FEAT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(ROOT / "out" / "manifest.csv", dtype={"id": str})
    ids = man.loc[man.split != "unused", "id"].tolist()   # the fixed 900
    todo = [c for c in ids if not (FEAT / f"{c}.npz").exists()]
    t0, done, bad = time.time(), 0, []
    it = iter(todo)
    with ThreadPoolExecutor(nw) as ex:
        futs = []
        for _ in range(nw):
            try:
                futs.append(ex.submit(one, next(it)))
            except StopIteration:
                break
        while futs:
            cid, n = futs.pop(0).result()
            done += 1
            if n == 0:
                bad.append(cid)
            if time.time() - t0 > budget:
                break
            try:
                futs.append(ex.submit(one, next(it)))
            except StopIteration:
                pass
    el, have = time.time() - t0, len(list(FEAT.glob("*.npz")))
    rate = el / max(done, 1)
    print(f"+{done} in {el:.0f}s ({rate:.2f}s/clip) | {have}/{len(ids)} done "
          f"| {len(ids)-have} left, eta {(len(ids)-have)*rate/60:.0f} min")
    if bad:
        print(f"  [!!] no frames decoded: {bad}")


if __name__ == "__main__":
    main()
