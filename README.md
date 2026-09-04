# precrash — a video-blind control for a composite accident-anticipation metric

Code, submissions and score log for the paper

> **Unbounded Timing Terms Make a Composite Accident-Anticipation Score
> Video-Blind: Evidence from a Live Benchmark**

## The result, in one line

On the [Zero-shot Accident Anticipation](https://kaggle.com/competitions/zero-shot-taa)
competition (AUTOPILOT-COG; 1,417 clips curated from MM-AU; 13 teams), a **constant risk score of 0.51** — which reads no pixels — scores
**2.35291** on the private leaderboard. That equals the eighth-placed team's
score at the precision the leaderboard displays, exceeds five of thirteen teams
including our own entry, and falls 0.14874 short of the winner.

```python
risk = [0.51] * 150
```

## What the probes recovered

Twenty-five video-blind submissions — twenty-one distinct curves, four of them
sent twice — designed so that unknown terms cancel under differencing,
decompose a metric whose weights are undisclosed and whose
labels are withheld:

| | |
|---|---|
| AP + AUC, to a blind submission | 0.35105 |
| TTA@0.5, saturated | 0.87593 |
| STTA@0.5, saturated | 1.12593 |
| **Timing terms together** | **2.00186 — 5.70× the discrimination floor** |
| Slope in the crossing frame | −1/60 per frame, confirmed by a one-frame test |
| Accident onset, recovered without labels | mean 120.1 frames of 150 |

The onset figure is a mean and not a support: a step at frame 140 scores the
same as a curve that never crosses, which bounds the mean excess beyond frame
140 but permits a few late clips invisible at five decimals. §5.5 of the paper
is explicit about the difference.

And a second result: **four curves that cross the threshold at the same frame,
and never fall below it afterwards, score 0.091 apart** — a spread the
published formula cannot produce. The benchmark's stated definition does not
reproduce its own scorer.

## Layout

```
latex/        the paper. `main.pdf` is the submitted version; build with
              `pdflatex main && bibtex main && pdflatex main && pdflatex main`
paper/        the working manuscript and the edit scripts that produced it
scripts/      submission builders, probes, analysis, figure scripts
src/          the ensemble that was entered, and the corrected metrics
submissions/  every submitted file, and LOG.md with every score
data/         manifests for the Nexar corpus. Not used by any result in the
              paper; kept for the labelled-corpus study named as future work
tests/        74 tests, no GPU needed: `pytest`
```

## Reproducing the finding

The probe builders need the competition's own `test.csv` and
`sample_submission.csv`, which the rules forbid us to redistribute. Download
them from the competition page first, then:

```bash
pip install kaggle          # not in the requirements files
python scripts/make_submission.py --probe constant_0.51 \
    --test_csv   path/to/test.csv \
    --sample_csv path/to/sample_submission.csv \
    --out submissions/p_constant_0.51.csv
python -m kaggle competitions submit -c zero-shot-taa \
    -f submissions/p_constant_0.51.csv -m "video-blind control"
```

Every file this produces is already committed under `submissions/`, so the
decomposition can be checked without submitting anything.

`submissions/LOG.md` records every score the organisers returned, including the
diagnostics that established the shape-dependence in §5.4.

`scripts/make_submission.py --replicate_sample` regenerates the organisers'
reference file and diffs it byte-for-byte against the distributed one
(1,417/1,417), which fixes the identifier order and the float format. We then
submitted it anyway, and it scored **1.04203**, not the 0.58333 shown on the
leaderboard's benchmark row — so those are two different files. Some comments
in that script still predate the submission and say the score is known to be
0.58333; they are wrong, and `submissions/scores.csv` is the record.

### What is and is not reproducible here

Tables 3 to 8 of the paper — the leaderboard, the controls, the probes, the
decomposition, the linear fit and the diagnostics — come entirely from
`submissions/scores.csv` and the two figure scripts, and reproduce offline.

Tables 9 to 12 — the curve-shape statistics, the modality ablation, the
twenty-five selected clips and the seven configurations — require running the
ensemble over all 1,417 clips, which needs the corpus. The rules forbid us to
redistribute it, and the paper does not rest any of its findings on those
tables.

## Figures

```bash
python scripts/make_figure_pipeline.py      --out results/fig_pipeline
python scripts/make_figure_decomposition.py --out results/fig_decomposition
```

Both regenerate offline from the scores recorded in the scripts. `results/` is
git-ignored, so the figures are not committed here; the versions used in the
paper are in `latex/media/`.

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
