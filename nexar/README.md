# Validation on a labelled corpus

The competition this paper analyses publishes no labels, so the metric can be
*exploited* there but the corrected protocol cannot be *tested* there. This
directory does the testing, on the Kaggle Nexar collision-prediction train
split: 1,500 clips, 750 positive and 750 negative, every positive carrying
`time_of_event` and `time_of_alert`.

Three things become possible that the competition corpus does not allow:
average precision and AUC are real numbers rather than a floor recovered by
differencing; earliness is measured to the **annotated collision** rather than
to the end of the window; and because there are 750 negatives, the cost a
permanent alarm imposes on ordinary driving is a measured quantity rather than
an argument.

## Running it

The video is not in this repository. Put the Kaggle
`nexar-collision-prediction` directory beside this checkout, then:

```
python3 prep.py            # clip manifest: fps, frame count, event frame
python3 split.py           # fixed 600/300 split, drawn once, seed in the file
python3 extract.py 150 4   # motion features; repeat until it reports 0 left
python3 run_controls.py    # the pathology, on all 1,500 clips
python3 train_eval.py      # held-out comparison under both metrics
python3 audit.py           # every headline number re-derived by another route
```

`extract.py` takes a time budget in seconds because the environment this was
built in kills anything still running at the end of a shell call. It is
resumable: a clip whose `.npz` exists is skipped, so run it repeatedly. About
75 minutes for 900 clips on two cores; the cost is H.264 decode and nothing
else (11.5 s of CPU against 0.4 s of I/O per clip).

Every module runs its own self-tests when executed directly.

## What each file is

| file | |
|---|---|
| `prep.py` | clip manifest. Resolves the event time, annotated in seconds, into a frame index in **that clip's own timebase** — the corpus has 36 distinct frame rates |
| `split.py` | the train/test split, fixed before any modelling and before extraction chose which clips to spend an hour decoding |
| `extract.py` | motion features at a common 7.5 Hz. Sampling is by **time**, not frame stride: "every 4th frame" is a different interval in almost every clip here |
| `metric.py` | the published composite — AP, AUC, TTA@0.5, STTA@0.5 — against real labels |
| `controls.py` | the video-blind family |
| `logreg.py` | L2 logistic regression by IRLS. Neither sklearn nor scipy is installed on the target machine, and a paper about evaluation defects should not have a black box in its own baseline |
| `model.py` | strictly causal features for the sighted baseline |
| `protocol.py` | the corrected protocol of the paper's Section 5.3 |
| `duty.py` | the repair that validation turned out to require |
| `train_eval.py` | scores everything on the same held-out clips |
| `audit.py` | the checks |

## Two things the design enforces rather than promises

**No control can see a label.** A curve generator is handed a `Clip` carrying
only `n_frames` and `fps`. A curve that wanted the target or the event time
would raise `AttributeError`, not quietly cheat. `audit.py` also rebuilds all
19 curves against scrambled labels and requires every one to be byte-identical.

**No feature can see the future.** The original PreCrash pipeline resampled at
index `alpha*t` and emitted at `t`, so the score reported for frame `t` was
computed from frame `1.3t` — a frame that had not yet been observed. Here the
property is asserted by perturbation: zeroing the tail of a clip must leave
every earlier feature bit-identical.

## Two defects found in this code before any number was used

**Average precision was order-dependent under ties.** A submission whose scores
are all identical scored 0.5121 instead of the base rate 0.5000 — a tenth of a
point of discrimination conjured out of the order rows happened to sit in.
Ties are grouped by distinct threshold now, and the test shuffles the rows and
demands the number not move.

**NumPy 2's weak scalar promotion silently zeroed a result.** Comparing a
`float32` curve against a Python-float threshold demotes *the threshold* to
float32, so an operating point of 0.899999976158142 became 0.9 and every clip
scoring exactly 0.9 fell on the wrong side. The protocol reported "detected
nothing". All thresholding is float64 now, with an assertion at the boundary
and a test pinning the case.

## A note on a shortcut not taken

`ffmpeg -skip_frame bidir` decodes 2.3x faster at 0.982 correlation with the
features used here, which would have cut extraction from 75 minutes to 33. It
is not used. It drops B-frames, so the result depends on each file's GOP
structure — and this corpus already leaks the label through encoder metadata
(the video's frame rate alone separates positives from negatives at AUC 0.533).
Putting an encoder-dependent artefact into the features of a baseline whose
entire job is to demonstrate that it reads the road would have been
indefensible.
