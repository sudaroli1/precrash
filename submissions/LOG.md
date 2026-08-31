# Leaderboard submissions

Every score in this file is computed by the competition organisers against
ground truth we do not have. That is the point: no metric in the paper's results
table is one we implemented ourselves.

Record each submission here as soon as its score appears. An unrecorded score is
a result we will not be able to reconstruct later.

## Known reference points

| Entry | Official score | Source |
|---|---|---|
| Top of leaderboard (CVLAB) | 2.50165 | public leaderboard |
| **SuryaInBytes (ours, 2 entries)** | **2.00585** | public leaderboard, rank 9 of 13 |
| `sample_submission.csv` (organisers' linear ramp) | 0.58333 | public leaderboard |

The official score is a weighted average of AP, AUC, TTA@0.5 and STTA@0.5, with
weights fixed by the organisers. TTA and STTA are measured to `t_ai`, the
annotated accident frame within the 150-frame clip -- not to the end of the clip.

## Submissions

| # | File | What it is | Reads pixels | Official score | Date |
|---|---|---|---|---|---|
| 00 | `00_replicate_sample.csv` | the organisers' ramp, regenerated | no | *(byte-identical to theirs; 0.58333 without submitting)* | |
| 01 | `01_constant.csv` | constant 0.51 | no | | |
| 02 | `02_linear_ramp.csv` | linear ramp 0 to 1 | no | | |
| 03 | `03_sigmoid_m0.50.csv` | sigmoid, midpoint 0.50 | no | | |
| 04 | `04_step_at_half.csv` | step at frame 75 | no | | |
| 05 | `05_sigmoid_m0.40.csv` | sigmoid, midpoint 0.40 | no | | |
| 06 | `06_ensemble_postproc.csv` | the full pipeline | yes | | |
| 07 | `07_clip_only.csv` | CLIP alone | yes | | |
| 08 | `08_flow_only.csv` | motion alone | yes | | |
| 09 | `09_prior_only.csv` | the three-state prior alone | yes | | |
| 10 | `10_ensemble_raw.csv` | ensemble, no post-processing | yes | | |

Rows 01-05 need no frames and no GPU. Rows 06-10 need the feature cache.

## What each row is for

**01, constant 0.51.** The degenerate case, and the one the paper turns on. It
maximises every locally computable proxy -- crossing frame 0, the largest
possible time-to-window-end, and perfect "stability", since a constant never
falls. Its official score is the measurement that shows the proxy and the metric
disagree.

**02-05.** The video-blind floor, established empirically rather than asserted.
A referee can object that a constant is a strawman; a family of shapes and
midpoints, all scoring near the floor, is harder to dismiss.

**06-10.** The ablation. Every number computed by the organisers, so no reviewer
needs to trust our implementation of AP, AUC or TTA -- we did not implement them.

## Local proxy, for comparison

These are what the earlier manuscript reported, computed by us, on definitions
that share the official metrics' names but not their definitions:

| Curve | Crossing frame | TTA to window end (assumed 30 fps) | "STTA compliance" |
|---|---|---|---|
| constant 0.51 | 0 | 5.00 s | 100% |
| linear ramp | 75 | 2.50 s | 100% |
| sigmoid m0.40 | 60 | 3.00 s | 100% |
| *reported for the ensemble* | *22.7* | *4.24 s* | *100%* |

The official TTA measures to the annotated accident frame; ours measured to the
end of the window. The names collide; the quantities do not.
