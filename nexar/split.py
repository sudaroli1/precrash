"""
Fix the train/test split before any modelling, and before extraction chooses
which clips to spend an hour decoding.

900 clips, stratified 450/450 on the label: 600 train, 300 test. The full 1,500
would be better and costs about 110 minutes of decode on two cores; 900 costs
65 and is ample for the one thing the sighted baseline has to establish, which
is that the corrected protocol separates it from the blind controls. The
unused 600 stay available if the split turns out to be too small.

The split is drawn from the id and the label and nothing else, with a fixed
seed, and written to disk once. Anything that reads it later cannot move it.

  python3 split.py
"""
import numpy as np, pandas as pd, pathlib

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "out"
SEED, N_PER_CLASS, N_TEST_PER_CLASS = 20260906, 450, 150


def main():
    man = pd.read_csv(OUT / "manifest.csv", dtype={"id": str})
    if "split" in man.columns:
        print("split already fixed; refusing to redraw")
        print(man.split.value_counts().to_dict())
        return

    rng = np.random.default_rng(SEED)
    man["split"] = "unused"
    for t in (0, 1):
        ids = man.loc[man.target == t, "id"].to_numpy()
        pick = rng.permutation(ids)[:N_PER_CLASS]
        man.loc[man.id.isin(pick[:N_TEST_PER_CLASS]), "split"] = "test"
        man.loc[man.id.isin(pick[N_TEST_PER_CLASS:]), "split"] = "train"

    print(pd.crosstab(man.split, man.target).to_string())
    # Duration and fps leak the label (AUC 0.537 and 0.533), so check the
    # split did not hand one side a different distribution on top of that.
    man["dur"] = man.n_frames / man.fps
    print()
    print(man[man.split != "unused"].groupby(["split", "target"])
          [["dur", "fps"]].mean().round(2).to_string())
    man.drop(columns=["dur"]).to_csv(OUT / "manifest.csv", index=False)
    print(f"\nwrote split into {OUT/'manifest.csv'}")


if __name__ == "__main__":
    main()
