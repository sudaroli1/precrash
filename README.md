# Anticipation without ground truth

Evaluation code for a study of how zero-shot traffic accident anticipation is
measured. This repository is the artefact for a paper in preparation; it is not
a method release.

## The short version

We built a zero-shot vision-language ensemble for the
[zero-shot-taa](https://www.kaggle.com/competitions/zero-shot-taa) benchmark and
reported a mean crossing frame of 22.7. Two facts about that benchmark, both
checkable from its public files:

**It publishes no ground truth.** `test.csv` carries `id, video_id,
start_frame, end_frame, caption` — no label, no event time, no alert time.
`train.csv` is a single row stating there is no train split. So average
precision, area under the ROC curve, a false-positive rate and a
time-to-accident measured to a collision onset **cannot be computed by any
entrant**. Only curve-shape statistics are available, and those are what our
earlier work reported.

**Its own reference submission is video-blind.** `sample_submission.csv` is a
linear ramp from 0.001 to 0.999 across 150 frames, identical for all 1,417
clips. It reads no pixels.

A crossing-frame statistic cannot separate a method from a ramp. Under the
definition our earlier work used, a constant risk score of 0.51 attains a
crossing frame of **0** against our reported **22.7**.

## What this repository contains

An evaluation implementation that measures anticipation against ground truth
where ground truth exists, and refuses to pretend where it does not.

```
src/utils/metrics.py     the corrected protocol: AP, AUC, mTTA, TTA@R80,
                         timing measured to the annotated onset using each
                         clip's own frame rate, bootstrap CIs, paired tests
src/utils/frame_io.py    clips that ship as folders of JPGs
src/utils/video_io.py    clips that ship as video files
src/engines/             CLIP ViT-L/14, frame-difference motion, motion prior
src/postprocess.py       the pipeline under study, including the two stages
                         that should not be reported (see below)

scripts/prepare_taa.py   manifest for zero-shot-taa; measures the corpus and
                         states plainly what it does not provide
scripts/prepare_nexar.py manifest for Nexar, which does publish ground truth
scripts/extract_features.py   GPU pass, once per corpus, resumable
scripts/evaluate.py           ablations with confidence intervals
scripts/protocol_comparison.py  legacy and corrected panels, side by side
scripts/make_figure1.py       the annotated-ceiling figure
scripts/make_synthetic_features.py  exercise the chain with no GPU
```

## Two corpora, for a reason

| | zero-shot-taa | Nexar |
|---|---|---|
| Labels | **none published** | yes |
| Event time | **none published** | yes |
| Alert time | **none published** | yes |
| Clips | 1,417 | 1,500 (750/750) |
| Ships as | JPG frames + gaze maps | video |
| What can be computed | curve-shape statistics only | the full corrected protocol |

The first is where the reported figures came from and where the argument lives.
The second is where the corrected protocol is demonstrated, because
demonstrating it requires labels the first does not release.

## Quick start, no GPU and no data

Everything downstream of feature extraction reads cached `.npz` files:

```bash
pip install -r requirements-analysis.txt
python -m pytest tests -q                       # 40 tests

python scripts/make_synthetic_features.py \
    --manifest data/nexar_fixed.dev.csv --out_dir features/synth --limit 120
python scripts/protocol_comparison.py \
    --features features/synth --config configs/honest.yaml \
    --out results/synth.json --dataset_name "SYNTHETIC"
```

Those curves are generated, not measured; the script refuses to write anywhere
the path does not contain `synth`.

## Two warnings that belong on the front page

**`src/postprocess.py` contains two stages that must not be reported.**
A time-axis compression at α = 1.3 is non-causal — the value emitted at frame
*t* is the score computed at frame 1.3·*t*, which has not happened. And a clamp
that, once the score crosses θ, floors every later value at θ + ε — which
enforces by construction the very monotonicity criterion the earlier work
reported as a 100% result. `configs/honest.yaml` disables the first; the second
is reported with and without.

**The third engine never captioned anything.** It was described as an NLP
caption prior. It computes the mean grayscale inter-frame difference over the
final third of a clip — the same statistic as the motion engine — thresholds it
at two fixed values to select one of three fixed strings, and embeds that. No
frame is captioned. It has three reachable states across an entire corpus and is
not independent of the motion engine. The class is `MotionThresholdPrior`;
`NLPScorer` remains as an alias and the behaviour is unchanged.

Note also that zero-shot-taa **supplies a caption per clip** — 79 distinct ones
across 1,417 clips — which this engine does not read.

## Tests

```bash
python -m pytest tests -q
```

Forty tests, pinning the properties that matter rather than the ones that are
easy: that accident-only data is refused; that timing is measured to the onset
and not the clip end; that a video-blind curve lands at chance on AP and AUC;
that a video-blind curve can nonetheless *win* on a timing metric alone; that
the bootstrap carries each clip's own frame rate through resampling; that frame
files are ordered numerically rather than lexicographically; and that the two
extraction speed-ups are value-preserving against reference implementations of
what they replaced.

## Data

Neither corpus is redistributed here.

- **zero-shot-taa** — CSVs from the competition's Data tab; frames and gaze maps
  from the Google Drive or Baidu Netdisk link there. Note that
  `kaggle competitions files` lists only the CSVs; the media is not a
  competition file.
- **Nexar** — the [Kaggle collision-prediction competition](https://www.kaggle.com/competitions/nexar-collision-prediction).

The manifests in `data/` record clip identifiers, windows, labels where they
exist, and frame rates, and are enough to reproduce the evaluation given the
media.

## License

MIT.
