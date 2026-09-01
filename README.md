# precrash — a video-blind control for a composite accident-anticipation metric

Code, submissions and score log for the paper

> **Unbounded Timing Terms Make a Composite Accident-Anticipation Score
> Video-Blind: Evidence from a Live Benchmark**

## The result, in one line

On the [Zero-shot Accident Anticipation](https://kaggle.com/competitions/zero-shot-taa)
competition (AUTOPILOT, CVPR 2026 workshop; 1,417 clips curated from MM-AU;
13 teams), a **constant risk score of 0.51** — which reads no pixels — scores
**2.35291** on the private leaderboard. That equals the eighth-placed team's
score at the precision the leaderboard displays, exceeds five of thirteen teams
including our own entry, and falls 0.14874 short of the winner.

```python
risk = [0.51] * 150
```

## What the probes recovered

Twenty-four video-blind submissions, designed so that unknown terms cancel
under differencing, decompose a metric whose weights are undisclosed and whose
labels are withheld:

| | |
|---|---|
| AP + AUC, to a blind submission | 0.35105 |
| TTA@0.5, saturated | 0.87593 |
| STTA@0.5, saturated | 1.12593 |
| **Timing terms together** | **2.00186 — 5.70× the discrimination floor** |
| Slope in the crossing frame | −1/60 per frame, confirmed by a one-frame test |
| Accident onset, recovered without labels | frames 100–140, mean 120 |

And a second result: **three curves that cross the threshold at the same frame,
with identical values at the crossing, score 0.091 apart** — a spread the
published formula cannot produce. The benchmark's stated definition does not
reproduce its own scorer.

## Layout

```
paper/        the manuscript, its figures, and the reference audit
latex/        LaTeX build — `pdflatex main.tex`
scripts/      submission builders, probes, analysis, figure scripts
src/          the ensemble that was entered, and the corrected metrics
submissions/  every submitted file, and LOG.md with every score
results/      both figures, as PDF and PNG
tests/        73 tests
```

## Reproducing the finding

```bash
python scripts/make_submission.py --probe constant_0.51
python -m kaggle competitions submit -c zero-shot-taa \
    -f submissions/p_constant_0.51.csv -m "video-blind control"
```

`submissions/LOG.md` records every score the organisers returned, including the
diagnostics that established the shape-dependence in §5.4.

`scripts/make_submission.py --replicate_sample` regenerates the organisers'
reference file and diffs it byte-for-byte against the distributed one
(1,417/1,417), which fixes the identifier order and float format without
spending a submission slot.

## Figures

```bash
python scripts/make_figure_pipeline.py      --out results/fig_pipeline
python scripts/make_figure_decomposition.py --out results/fig_decomposition
```

Both regenerate offline from the scores recorded in the scripts.

## Compliance

Submissions were made through the competition's own late-submission facility,
within the stated limit of 100 per day. Every probe is a curve generated from a
closed-form expression in the frame index; none was derived from any label. No
part of the competition's data is redistributed here. The corpus and the
competition are cited as the rules require. See the paper's compliance section.

## Corrections to earlier versions of this work

Recorded here because the paper is partly a case study of its own authors:

- A temporal-compression stage emitted at frame *t* the score computed from
  frame 1.3*t* — a frame not yet observed. It produced the reported six-frame
  gain in time-to-accident. `configs/honest.yaml` disables it.
- A monotone clamp forces the exact condition the stable-timing metric tests,
  so any compliance figure measures the clamp.
- "90% of RAFT sensitivity at 1% of the compute" was never measured. Withdrawn.
- The ensemble weights were searched over the evaluation corpus. The released
  configuration weights the three modalities equally, chosen a priori.
- Timing figures in seconds were measured to the end of the clip, not to a
  collision, because this corpus annotates none.

`paper/REFERENCE_AUDIT.md` records what was wrong with the original reference
list, including two entries whose author lists did not match the cited work.
