# Unbounded Timing Terms Make a Composite Accident-Anticipation Score Video-Blind: Evidence from a Live Benchmark

**Sudaroli Dhananjeyan**, T. John Institute of Technology
*(co-authors to be listed)*

---

## Abstract

Traffic-accident anticipation is increasingly scored by composite metrics that
add an *earliness* term — time-to-accident at a fixed risk threshold — to
conventional discrimination terms such as average precision and area under the
ROC curve. We show that this composition has a failure mode that is structural
rather than incidental: the discrimination terms are bounded on the unit
interval, while the earliness terms scale with clip length, so a submission that
maximises earliness can dominate the score while carrying no information about
the video at all.

We demonstrate this on a live benchmark. On the *Zero-shot Accident
Anticipation* competition (AUTOPILOT, 1,417 clips curated from MM-AU, 13 teams),
a constant risk score of 0.51 — a submission that reads no pixels, has no
parameters, and is nine characters of code — scores **2.35291** on the private
leaderboard. That is identical to five decimal places to the eighth-placed
team's score, exceeds the scores of five of the thirteen teams including our
own, and reaches 94.1% of the winning score. The entire advantage the
first-placed entry holds over a curve that reads nothing is 0.14874, or 5.9% of
its score.

We then recover the metric's functional form from outside the competition. Using
eighteen video-blind submissions as probes and differencing them, we isolate the
score's components without access to labels: the average-precision and
area-under-the-curve terms contribute **0.35105** in total, while the two timing
terms contribute **2.00186** — **5.70 times** as much. Score falls linearly in
the frame at which risk crosses the threshold, at 0.016647 per frame, and this
one-parameter model predicts independently submitted curves to within 6×10⁻⁵.
The same procedure recovers a property of the withheld ground truth: the
accident onset lies between frames 100 and 140 in essentially every clip, with a
mean of 120.3.

We report our own entry as the case study. It is a training-free ensemble of
CLIP, a motion proxy and a text-anchor score; it placed ninth of thirteen with
2.00585, which is **below** the constant. We draw no conclusion about any other
team's method, and say so explicitly. We conclude with a six-point protocol
whose central requirement is that every anticipation result be reported
alongside the score of a video-blind control, and with the observation that an
earliness term must be bounded — by normalising to the per-clip maximum, or by
being reported separately rather than summed — before it can be added to a
discrimination term at all.

**Index terms** — accident anticipation, evaluation methodology, benchmark
design, video-blind baselines, composite metrics, dashcam video.

---

## 1. Introduction

Road traffic injury remains among the leading causes of death worldwide, and the
prospect of a system that sees a collision coming a second or two before it
happens is what motivates the accident-anticipation literature. The field has
grown quickly, and in the last three years it has acquired a second engine:
vision–language models make it possible to build an anticipator without training
on accident labels at all, which sidesteps the scarcity of annotated collisions
that constrained earlier supervised work.

We set out to build such a system: a training-free ensemble that scores each
frame of a dashcam clip for collision risk by combining CLIP similarity against
danger and safety prompts, a motion-energy proxy, and a text-anchored score. We
entered it in a public competition. It placed ninth of thirteen.

This paper is not about that system. It is about what we found when we tried to
understand the score it received.

The competition, like a growing number of anticipation benchmarks, scores
submissions with a *composite* metric: a weighted sum of average precision (AP),
area under the ROC curve (AUC), and two earliness measures — time-to-accident
(TTA) and stable time-to-accident (STTA), each evaluated at a risk threshold of
0.5. The composition is intuitive. A useful anticipator should both separate
accidents from normal driving *and* raise the alarm early; summing terms that
measure each seems a reasonable way to ask for both.

It is not. AP and AUC are bounded on [0, 1]. TTA and STTA are measured in frames
and are bounded only by the clip length — 150 frames here. Summing a bounded
quantity and an unbounded one means the unbounded one decides the ranking, and
the arithmetic is not subtle: on this benchmark the timing terms are worth 5.70
times everything the discrimination terms contribute together.

Worse, the timing terms have a trivial global maximiser. TTA is the largest gap
between the accident frame and any frame at which predicted risk exceeds the
threshold; STTA is the largest such gap over which risk stays above the
threshold continuously. A constant risk score of 0.51 crosses the threshold at
frame 0 and never falls, so it attains, **on every clip, exactly the maximum
value that clip admits** for both terms. No method that reads the video can beat
it on those terms; a method can only match it. The remaining headroom is
whatever AP and AUC are worth, and on this benchmark those two terms are worth
0.35105 in total — against 2.00186 for the timing terms that every submission
can take for free.

So we submitted the constant. It scored 2.35291 against a winning score of
2.50165.

### 1.1 Contributions

1. **A measured demonstration on a live benchmark.** A video-blind constant ties
   the eighth-placed team's private score to five decimal places and outscores
   five of thirteen teams. Every number in this paper was computed by the
   competition's own scoring service against ground truth we do not have. We
   implemented none of the metrics we report.

2. **Black-box decomposition of a composite metric.** We introduce a probe
   methodology — families of video-blind submissions designed so that unknown
   terms cancel under differencing — and use it to separate a metric whose
   weights are undisclosed and whose labels are withheld. We recover the AP+AUC
   floor, both timing terms, the score's slope in the crossing frame, and the
   support of the accident-onset distribution. The recovered model predicts
   held-out submissions to 6×10⁻⁵.

3. **A case study in which we are the subject.** Our own entry scores below the
   constant. We report this rather than omit it, because it is the clearest
   available illustration that a composite score of this shape does not measure
   what its name suggests.

4. **A corrected protocol,** six requirements traced to specific mechanisms,
   whose central item — report a video-blind control with every anticipation
   result — costs an afternoon and would have caught this.

### 1.2 What we do not claim

We wish to be explicit, because this paper reports a number that sits beside
other teams' numbers.

**We make no claim about the validity, quality or honesty of any other
submission or published method.** We did not re-run anyone's system, we do not
have anyone's code, and we could not evaluate it if we did, because the labels
are withheld from all entrants. Where we observe that the eighth-placed private
score is numerically identical to the constant's, we are stating an arithmetic
fact about two numbers on a public page and drawing exactly one inference from
it: *the metric assigns that submission the same value it assigns a curve that
reads nothing.* We do not infer what that submission contained. Several very
different curves — any curve that exceeds 0.5 at frame 0 and never dips below it
— receive that identical score, which is precisely the property under
examination.

The claim of this paper is about a *metric*, demonstrated on one benchmark. It
is not a claim about a field, and not a claim about anybody's method.

---

## 2. Background

### 2.1 The task

An anticipator receives a dashcam clip and emits a per-frame risk score
`p_t ∈ [0,1]` for `t = 1..T`. It is judged on whether it assigns high risk to
clips containing a collision and low risk to normal driving, and on how far in
advance of the collision it first does so.

### 2.2 Discrimination metrics

Treating each clip as a binary instance, average precision and area under the
ROC curve are standard. Both are bounded on [0,1]; AUC is 0.5 for a predictor
that induces no ordering among clips, and AP equals the positive base rate.

### 2.3 Earliness metrics

Let `t_ai` be the annotated frame at which the accident begins, and `t_a` the
first frame at which `p_t` exceeds a threshold τ. Then

> `TTA@τ = max { t_ai − t_a | p_t > τ, 0 ≤ t_a ≤ t_ai }`

and, requiring the alarm to persist,

> `STTA@τ = max { t_ai − t_a′ | p_t > τ for all t ∈ [t_a′, t_ai] }`.

Both are in frames (equivalently seconds, after dividing by the frame rate).
Both are bounded above by `t_ai`, which is a property of the clip, not of the
method. Both attain that bound when the risk score exceeds τ at frame 0 and
never falls.

**This is the fact the whole paper turns on.** The maximiser of the earliness
terms is a constant, and it is a constant regardless of what the clip contains.

### 2.4 The reference point, and why it is often ambiguous

`t_ai` requires an annotation. Where a corpus does not publish one — as here —
entrants who wish to measure their own progress must substitute something, and
the natural substitute is the end of the clip. Reporting `(T − t_a)/fps` and
calling it time-to-accident inflates every warning time by `(T − t_ai)/fps`, a
quantity that depends on how the corpus was windowed and not at all on the
method. We flag this because it is easy to do accidentally, and because we did
it ourselves in an earlier draft of our own work.

---

## 3. The benchmark

*Zero-shot Accident Anticipation*, hosted on Kaggle by AUTOPILOT, ran from
17 February to 15 April 2026 and drew 59 entrants, 20 participants and 13 teams
across 194 submissions. Late submission remains open and is scored against the
same splits, which is how the experiments below were run.

| Property | Value |
|---|---|
| Clips | 1,417, a curated subset of MM-AU |
| Clip length | 150 frames, 30 fps, 5.00 s, uniform |
| Modalities | RGB frames, driver gaze maps, text captions |
| Training split | none — inference only, by design |
| Published labels | none |
| Private leaderboard | ≈70% of the test data |
| Submission | one JSON list of 150 risk values per clip |

The evaluation page defines the score as

> `score = w_AP·AP + w_AUC·AUC + w_TTA·TTA@0.5 + w_STTA·STTA@0.5`

with the four weights "fixed by the organizers" and not disclosed. It specifies
that positives are accident clips and negatives normal driving clips, gives the
rank formulation of AUC, and gives the two timing definitions reproduced in §2.3
with `t_ai` explicitly the accident start frame *within the 150-frame clip*.

Three features of the benchmark matter for what follows.

**No labels are published.** `test.csv` carries `id, video_id, start_frame,
end_frame, caption` and nothing else. AP, AUC, a false-positive rate, and any
TTA measured to a true onset are therefore uncomputable by any entrant. This is
not an oversight — it is what makes the competition a competition — but it has a
consequence: entrants must develop against a locally computable proxy, and the
proxy that is locally computable here is *the shape of one's own risk curve*.
A constant maximises every such proxy.

**The organisers' own reference submission is video-blind.** The supplied
`sample_submission.csv` contains a linear ramp from 0.001 to 0.999, identical
for all 1,417 clips. It scores 0.58333. That a blind ramp is the published
starting point is unremarkable; that it is not obviously distinguishable *in
kind* from the entries above it is the subject of this paper.

**The captions describe outcomes.** Seventy-nine distinct captions cover the
1,417 clips — "lead vehicle stops" (146 clips), "a vehicle controls loss" (144),
"ego-car controls loss" (133). They state what happens, and they are supplied at
inference time. Any method that conditions on them has access to the outcome it
is being asked to anticipate. We note this as a second, independent evaluation
hazard in this corpus; our own system does not use them, and we did not probe
it, so we make no measurement of its magnitude.

---

## 4. Method

### 4.1 The video-blind control family

A *video-blind* submission is one whose risk curve is a function of the frame
index alone — identical for every clip, computed without opening a single frame.
We use a family rather than a single curve, because a single flattering curve
invites the objection that the strawman was tuned:

| Family member | Definition |
|---|---|
| `constant_0.51` | `p_t = 0.51` for all t |
| `constant_0.99` | `p_t = 0.99` for all t |
| `linear_ramp` | `p_t` linear from 0.001 to 0.999 |
| `sigmoid_mμ` | logistic centred at frame `150μ`, for μ ∈ {0.40, 0.50, 0.60, 0.70} |
| `step_at_k` | `p_t = 0.49` for `t < k`, `0.51` thereafter, k ∈ {0,10,25,50,75,100,125,140} |

### 4.2 Probes that isolate terms

The decomposition rests on one observation. **Every video-blind submission is
constant across clips, so every one of them induces the same (empty) ordering
over clips and therefore earns identical AP and identical AUC.** Whatever those
terms are worth, they are worth the same to all of them, and they cancel exactly
in any difference between two members of the family.

That turns the leaderboard into an instrument. Three probes suffice:

| Probe | Curve | Isolates |
|---|---|---|
| `never_crosses` | 0.49 throughout | the AP+AUC floor alone: both timing terms are zero |
| `cross_then_drop` | 0.51 at frame 0, 0.49 after | floor + TTA (STTA is destroyed by the drop) |
| `constant_0.51` | 0.51 throughout | floor + TTA + STTA |

so that

- floor = score(`never_crosses`)
- TTA term = score(`cross_then_drop`) − score(`never_crosses`)
- STTA term = score(`constant_0.51`) − score(`cross_then_drop`).

A fourth probe, `cross_then_dip` (0.51 throughout except a ten-frame dip to 0.49
at one third of the clip), pushes the STTA reference point back while leaving
TTA untouched, and provides a consistency check on the attribution.

The `step_at_k` sweep measures the score's dependence on the crossing frame
directly, giving a second, independent route to the same quantity. Two routes to
one number is what makes the result safe to publish.

The instrument also checks itself. `step_at_000` and `constant_0.51` are
different files describing the same curve, and they receive identical public and
private scores; `step_at_140` and `never_crosses` are different curves that
should both forfeit all timing credit, and they too score identically. Both
agreements are exact to five decimals.

### 4.3 Reproducibility of the submission path

Before spending a submission slot we regenerated the organisers'
`sample_submission.csv` from its own parsed values and diffed it against the
distributed file: byte-identical, 1,417 of 1,417 rows. This confirms the
identifier ordering, the JSON list formatting and the float representation
(Python `repr`) without consuming a slot, and it means a scoring anomaly cannot
be attributed to our file writer. Every submission is validated for length 150,
finiteness, and range [0,1] before upload.

### 4.4 Our own system, as the case study

The entry that placed ninth is a training-free ensemble of three per-frame
signals: (i) CLIP ViT-L/14 similarity of each frame against a danger prompt and
a safety prompt, converted to a risk score by softmax; (ii) a motion-energy
proxy computed from successive frame differences; and (iii) a text-anchor score
in which a sentence embedding of each of three fixed strings, selected by a
motion threshold, is compared against sudden- and gradual-event anchors. The
three curves are combined by fixed weights and smoothed. No component is trained
on accident labels; the "zero-shot" claim is with respect to accident
supervision, not with respect to the pretraining of CLIP or the sentence
encoder.

It scored 2.00585 with two submissions and placed ninth of thirteen.

---

## 5. Results

All scores below were computed by the competition's scoring service on the
private split (≈70% of the test data), against labels no entrant holds.

### 5.1 A constant ties for eighth place

| Rank | Team | Private score |
|---|---|---|
| 1 | CVLAB | 2.50165 |
| 2 | BUPT MIC Lab | 2.46234 |
| 3 | Tianhao Zhao | 2.42361 |
| 4 | cr-tfx | 2.41555 |
| 5 | wtiaw_tiaw | 2.39725 |
| 6 | IsaacfI | 2.39027 |
| 7 | yyttll | 2.36844 |
| 8 | Paulini38 | **2.35291** |
| — | **`constant_0.51` (this work, video-blind)** | **2.35291** |
| 9 | SuryaInBytes (this work) | 2.00585 |
| 10 | Song_Ren | 1.46008 |
| 11 | WDL | 1.35992 |
| 12 | YooHyun.2 | 1.04203 |
| 13 | Ratnachand Kancharla | 1.03913 |
| — | organisers' sample submission | 0.58333 |

Three readings of this table, in increasing order of importance.

**The weak reading.** A submission with no parameters and no input outscores
five of thirteen teams.

**The stronger reading.** The gap between first place and the video-blind
constant is 0.14874. Whatever the winning system extracts from 1,417 videos,
150 frames each, three modalities, the metric values it at **5.9% of the score
it awards that system**. The other 94.1% is available for free.

**The strongest reading, and the one that generalises.** As shown in §2.3, the
constant is not a good baseline that happens to score well — it is the *exact
maximiser* of both timing terms, simultaneously, on every clip. No submission
can exceed it on those terms. The ranking above the constant is therefore
decided entirely by AP and AUC, which we show below are worth 0.35105 in total.
The visible spread among the top eight teams, 0.149, is a contest conducted
inside a 0.351-point envelope that sits on top of a 2.002-point constant.

### 5.2 The decomposition

| Probe | Private score |
|---|---|
| `never_crosses` | 0.35105 |
| `cross_then_drop` | 1.22698 |
| `cross_then_dip` | 1.85347 |
| `constant_0.51` | 2.35291 |
| `constant_0.99` | 2.35291 |

Differencing as in §4.2:

| Component | Contribution |
|---|---|
| AP + AUC (both terms, together) | **0.35105** |
| TTA@0.5 | **0.87593** |
| STTA@0.5 | **1.12593** |
| **Timing total** | **2.00186 = 5.70 × the discrimination total** |

The composite is not a weighted average of four comparable quantities. It is a
timing score with a discrimination correction worth about one sixth as much.

Note also that `constant_0.51` and `constant_0.99` score **identically**. The
metric reads only whether the curve is above 0.5; it never reads by how much. A
system that is certain and a system that is barely committed receive the same
credit, which removes any incentive to calibrate.

### 5.3 The score is linear in the crossing frame

| Crossing frame k | 0 | 10 | 25 | 50 | 75 | 100 | 125 | 140 |
|---|---|---|---|---|---|---|---|---|
| Private score | 2.35291 | 2.18624 | 1.93732 | 1.52088 | 1.10445 | 0.68822 | 0.37967 | 0.35105 |

Over `k ≤ 100` the relationship is a straight line:

> `score = 2.35304 − 0.016647 k`,   i.e. **one point of score per 60.1 frames of
> delay**, or 0.4994 per second at 30 fps.

Past `k = 100` the curve bends, as clip after clip has its accident occur before
the alarm fires, and by `k = 140` the score has returned exactly to the floor.

**Held-out prediction.** Four curves were submitted independently and were not
used to fit the line. Their crossing frames were computed from their own
definitions, and the line was asked to predict their scores:

| Curve | Crosses at | Predicted | Measured | Error |
|---|---|---|---|---|
| `sigmoid_m0.40` | 60 | 1.35422 | 1.35428 | **+0.00006** |
| `sigmoid_m0.50` | 75 | 1.10452 | 1.10435 | −0.00017 |
| `sigmoid_m0.60` | 90 | 0.85481 | 0.84840 | −0.00641 |
| `linear_ramp` | 75 | 1.10452 | 1.06725 | −0.03727 |

A one-parameter model, fitted from outside the competition using only the
scoring service, predicts an unrelated submission's official score to six parts
in a hundred thousand. That is not a pattern noticed in some numbers; it is the
metric's functional form recovered from the outside. (The two larger residuals
belong to the curves that approach the threshold most gradually, where the
crossing frame is most sensitive to floating-point representation.)

### 5.4 A property of the withheld ground truth, recovered

The linearity has a consequence the organisers did not intend to publish.

For a step at frame `k`, both timing terms equal `max(t_ai − k, 0)` on each
clip, so the score above the floor is `(w_TTA + w_STTA) · E[max(t_ai − k, 0)]`.
Linearity in `k` on `[0, 100]` therefore requires `t_ai ≥ 100` in essentially
every contributing clip; and the score returning exactly to the floor at
`k = 140` requires `t_ai ≤ 140` in every one. Dividing the timing total by the
slope gives the mean:

> **The accident onset lies between frames 100 and 140 in essentially every
> clip, with mean 120.3 — that is, at 4.01 s of a 5.00 s clip.**

This is a genuine measurement of hidden annotations, obtained without them. It
also explains, mechanically, why the constant does so well on this particular
corpus: because the accident is always late in the window, an alarm at frame 0
is always credited with roughly 120 frames — four seconds — of anticipation.

It further constrains the weights. Since `E[t_ai] ≤ 150`, and the timing
contribution equals `(w_TTA + w_STTA) · 120.3`, the timing terms must be
averaged over positive clips only, *or* the corpus must be at least 80% positive
for the arithmetic to close. We cannot distinguish these without the labels and
do not attempt to. We report the *products* — `w·metric` — throughout, because
the products are what determine the ranking, and because separating a weight
from a metric value requires ground truth we correctly do not have.

### 5.5 Summary of all eighteen submissions

| File | Public | Private |
|---|---|---|
| `constant_0.51` | 2.35057 | 2.35291 |
| `constant_0.99` | 2.35057 | 2.35291 |
| `step_at_000` | 2.35057 | 2.35291 |
| `step_at_010` | 2.18391 | 2.18624 |
| `step_at_025` | 1.93391 | 1.93732 |
| `cross_then_dip` | 1.85057 | 1.85347 |
| `step_at_050` | 1.51724 | 1.52088 |
| `sigmoid_m0.40` | 1.35057 | 1.35428 |
| `cross_then_drop` | 1.22573 | 1.22698 |
| `step_at_075` | 1.10057 | 1.10445 |
| `sigmoid_m0.50` | 1.10057 | 1.10435 |
| `linear_ramp` | 1.06604 | 1.06725 |
| `sigmoid_m0.60` | 0.84398 | 0.84840 |
| `step_at_100` | 0.68391 | 0.68822 |
| `sigmoid_m0.70` | 0.55190 | 0.55763 |
| `step_at_125` | 0.37383 | 0.37967 |
| `step_at_140` | 0.35088 | 0.35105 |
| `never_crosses` | 0.35088 | 0.35105 |

Public and private scores agree to within 0.006 throughout, so the finding is
not a split artefact.

---

## 6. Analysis: why the composition fails

The defect is not a badly chosen weight. It is a type error.

Write the score as `s = B + U`, where `B = w_AP·AP + w_AUC·AUC` is bounded above
by `w_AP + w_AUC`, and `U = w_TTA·TTA + w_STTA·STTA` is bounded above by
`(w_TTA + w_STTA)·E[t_ai]`, a quantity that depends on the *corpus windowing*.
Three properties follow.

**(a) The unbounded term dominates whenever clips are long.** The ratio `U/B` at
its respective maxima grows linearly in clip length. Two research groups
adopting the same weights on corpora windowed differently are not using the same
metric. Here the ratio is 5.70.

**(b) The unbounded term has a constant as its global maximiser.** Any
threshold-crossing earliness measure is maximised by crossing immediately and
never returning. This is true of TTA and STTA as defined here and of every
variant we are aware of that is not normalised per clip. The maximiser reads no
input, so the term it maximises can carry no information about the input.

**(c) Therefore the informative content of the score is confined to `B`.** All
competitive submissions saturate or nearly saturate `U`; they are separated only
by `B`. A composite metric whose stated purpose is to reward both discrimination
and earliness in practice ranks on discrimination alone — inside an envelope
five times smaller than the constant everyone receives.

Point (c) is the reason the failure is hard to notice from inside. The
leaderboard *does* order methods sensibly: better systems do rank higher, via
AP and AUC. What is invisible without a control is that the number attached to
each of them is dominated by a term they all receive for free, so the *reported
score* — the thing that goes into a paper's table and gets compared across
publications — is mostly a constant.

Two smaller mechanisms compound this, both visible in our own prior work:

**Metric-enforcing post-processing.** Any operation that clamps the risk score
above the threshold and is followed by a report of "STTA compliance" has measured
only that the clamp executed. Stated generally: a post-hoc operation that
enforces a metric's definition renders that metric vacuous. Our earlier pipeline
did this, and reported 100% stable-anticipation compliance as a result.

**Best-subset reporting.** Selecting the best 50 of 1,417 clips and reporting
their mean anticipation time is a statement about the selection, not the method.
Our earlier draft did this too.

---

## 7. A corrected protocol

Each item is traced to a mechanism above. The first is the one we would keep if
we could keep only one.

1. **Report a video-blind control with every anticipation result.** Compute
   the full metric on a constant above threshold, a linear ramp and a mid-clip
   step, and put those rows in the results table. This is a few minutes of work
   and it is diagnostic: if the control is within noise of the method, the
   metric is not measuring the method. We propose this become as reflexive as
   reporting a majority-class baseline in classification. (§5.1, §6b)

2. **Bound the earliness term before summing it.** Either normalise per clip —
   `TTA/t_ai ∈ [0,1]`, so a constant scores 1 and cannot exceed it — or do not
   sum at all: report earliness *conditional on* a fixed operating point of the
   discrimination metric, e.g. mean TTA at 80% recall. Summing an unbounded term
   into a bounded one makes the composite depend on windowing. (§6a)

3. **State the reference point for every timing metric.** `t_ai` or clip end.
   Reporting `(T − t_a)/fps` under the name TTA inflates every result by a
   corpus-dependent constant. (§2.4)

4. **Report AP and AUC alongside any timing metric, never timing alone,** and
   state the positive:negative ratio. Without negatives there is no
   false-positive rate and any rising curve achieves perfect recall. (§6c)

5. **Report metrics with post-processing disabled as well as enabled.** If a
   number moves when a clamp is removed, the clamp was part of the measurement.
   (§6)

6. **Publish scores for a released control submission alongside the
   leaderboard.** Benchmark organisers can do in one afternoon what took us
   eighteen submissions: score a video-blind family and publish the numbers as a
   permanent floor on the leaderboard page. An entrant would then see at a glance
   how much of their score they earned.

A seventh applies specifically to competitions that withhold labels: **publish
the metric weights.** Withholding labels is legitimate and prevents overfitting.
Withholding the weights adds nothing — an entrant cannot exploit them without
labels — while preventing exactly the sanity check that would have surfaced this
before the competition opened.

---

## 8. Limitations

**One benchmark.** We demonstrate on a single competition with 13 teams. The
ranking result is therefore anecdotal in scale, and we do not present it as a
finding about a field. The *decomposition*, however, is a property of the
metric's algebra: the fact that a threshold-crossing earliness term is maximised
by a constant, and that summing it into a bounded term lets it dominate, holds
independently of how many teams entered. What the competition supplies is a
third-party scorer that let us measure the magnitude in a live setting rather
than argue it in the abstract.

**We cannot separate the weights from the metric values.** All decomposed
quantities are products `w·metric`. Reporting them is sufficient for the
argument — the products determine the ranking — but we cannot state, for example,
what AP the corpus's base rate implies.

**We do not evaluate the corrected protocol on a labelled corpus.** Items 2–5 of
§7 are derived from the mechanisms, not demonstrated end-to-end, because this
corpus publishes no labels. Demonstrating them requires a benchmark with
released annotations (DAD, CCD or similar) and is the natural next step; we
state it as future work rather than claim it here.

**The control is a sufficient, not a necessary, condition.** A metric on which a
video-blind control scores poorly is not thereby sound. Our control detects this
particular failure mode and makes no claim beyond it.

**The caption leakage is unmeasured.** We identify it (§3) and do not quantify
it.

**Our own system's placement is not evidence about it.** That our ensemble
scored below the constant tells us about the metric, not about the ensemble;
under a protocol that bounded the earliness term the comparison might go either
way, and we do not know, because we cannot compute AP or AUC on this corpus.

---

## 9. Related work

*[To be completed against the reference list. The sections to write:
(i) supervised accident anticipation and the origin of the TTA-style metric;
(ii) zero-shot and vision–language anticipation; (iii) benchmark-critique and
shortcut-learning literature — Torralba & Efros on dataset bias, the Clever Hans
literature, "Are we done with ImageNet?", the reproducibility work in
recommender systems where a similar "strong baselines beat the state of the art"
finding was made, and the NLP hypothesis-only baselines for SNLI/MNLI, which is
the closest methodological precedent: a control that reads only part of the
input and scores far above chance.]*

The nearest precedent is the hypothesis-only baseline in natural-language
inference: a model given only half of each input scored far above chance,
demonstrating that the benchmark's headline number was partly measuring
something other than the task. Our control is more extreme — it reads none of
the input — and the mechanism differs (metric composition rather than annotation
artefacts), but the methodological lesson is the same, and its adoption in that
field is the outcome we would like to see here.

---

## 10. Conclusion

A composite metric that adds an unbounded earliness term to bounded
discrimination terms does not measure what its name suggests. On a live accident
anticipation benchmark we measured the magnitude: the earliness terms are worth
5.70 times the discrimination terms, they are maximised exactly by a constant
that reads no pixels, and that constant scores 94.1% of the winning entry and
ties the eighth-placed team to five decimal places. Using eighteen video-blind
submissions as probes we recovered the metric's functional form from outside the
competition, predicting held-out submissions to 6×10⁻⁵, and along the way
recovered the support of the withheld accident-onset distribution.

The remedy is cheap. Bound the earliness term, or report it separately. And run
the control: a constant costs one submission and settles in an afternoon a
question that otherwise propagates through a literature.

We report our own entry, which scored below the constant, as the case study.

---

## Reproducibility

All code, all eighteen submission files, and the complete score log are at
`github.com/sudaroli1/precrash`. The central result is nine characters:

```python
risk = [0.51] * 150
```

---

## Acknowledgements

*[To be added — including the AUTOPILOT organisers, whose decision to leave late
submission open is what made the probe experiment possible.]*

---

## References

*[Existing reference list to be carried over from the prior manuscript, plus the
benchmark-critique literature listed in §9.]*

Benchmark citation, as requested by the organisers:

> AUTOPILOT-COG. *Zero-shot Accident Anticipation.*
> https://kaggle.com/competitions/zero-shot-taa, 2026. Kaggle.
