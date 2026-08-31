# Unbounded Timing Terms Make a Composite Accident-Anticipation Score Video-Blind: Evidence from a Live Benchmark

*Author block withheld for double-blind review. Camera-ready front matter — full
author list, affiliations, corresponding author and the repository URL — is
pending and must be completed before submission.*

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
parameters, and is one line of code — scores **2.35291** on the private
leaderboard. That is identical to five decimal places to the eighth-placed
team's score, exceeds the scores of five of the thirteen teams including our
own, and reaches 94.1% of the winning score. The entire advantage the
first-placed entry holds over a curve that reads nothing is 0.14874, or 5.9% of
its score.

We then recover the metric's functional form from outside the competition. Using
eighteen video-blind submissions as probes and differencing them, we isolate the
score's components without access to labels: the average-precision and
area-under-the-curve terms contribute **0.35105** to a video-blind submission,
while the two timing terms contribute **2.00186** — **5.70 times** as much.
Score falls linearly in the frame at which risk crosses the threshold, at
0.016647 per frame, and this model predicts three held-out step-like curves to
within 7×10⁻³. It does not predict a linear ramp, which crosses at the same
frame as a step yet scores 0.037 lower; we report that discrepancy rather than
smooth it, since it bounds how far the crossing-frame model can be pushed. The
same procedure recovers a property of the withheld ground truth: the accident
onset lies between frames 100 and 140 in essentially every accident clip, with a
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
   five of thirteen teams. Every *leaderboard score* in this paper was computed
   by the competition's own scoring service against ground truth we do not have;
   we implemented none of the metrics. All derived quantities — the
   decomposition, the fit, the ratios — are our arithmetic on those scores, and
   we mark them as such.

2. **Black-box decomposition of a composite metric.** We introduce a probe
   methodology — families of video-blind submissions designed so that unknown
   terms cancel under differencing — and use it to separate a metric whose
   weights are undisclosed and whose labels are withheld. We recover the AP+AUC
   floor, both timing terms, the score's slope in the crossing frame, and the
   support of the accident-onset distribution — and we report the one place the
   recovered model fails, which locates the residual structure it does not yet
   capture.

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
`p_t ∈ [0,1]` for `t = 0..T−1`. Frame indices are 0-based throughout this paper,
because one frame is worth 0.0166 of score here and an off-by-one would exceed
every residual we report. It is judged on whether it assigns high risk to
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
same splits, which is how the experiments below were run. All competition
figures in this section were read from the competition's own Overview,
Evaluation, Data and Leaderboard pages on 31 August 2026; archived copies
accompany the submission.

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
for all 1,417 clips; it reads no pixels. That a blind ramp is the published
starting point is unremarkable in itself; that it is not distinguishable *in
kind* from the entries above it is the subject of this paper.

We note one thing we cannot yet resolve. The leaderboard carries a benchmark row
scoring 0.58333, and it is tempting to attribute that score to
`sample_submission.csv`. Nothing published says so, and our own ramp — which
crosses the threshold at the same frame — scored 1.06725, so the attribution
would contradict our own measurements. We therefore make no claim about what the
benchmark row is, and have submitted the organisers' file verbatim to settle it.

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
identical across clips, so — provided the scorer reduces each 150-value curve to
one clip-level score by any deterministic function of that curve — every
submission in the family produces an all-way tie among clips, and therefore
earns identical AP and identical AUC.** Whatever those terms are worth, they are
worth the same to all of them, and they cancel exactly in any difference between
two members of the family.

Two assumptions are doing work here and we state them rather than assume them.
The first is that the reduction is clip-level at all: if AP and AUC were instead
pooled over all 1,417 × 150 frames against per-frame labels, curve *shape* would
enter them and the cancellation would fail. §5.3 reports a result that tests
this, and §5.3 also reports one observation the assumption does not explain. The
second is that the all-way tie is resolved conventionally (rank-averaging for
AUC, giving 0.5; base rate for AP).

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

A fourth probe, `cross_then_dip` (0.51 throughout except a dip to 0.49 across
frames 50–59), pushes the STTA reference point back to frame 60 while leaving
TTA untouched, and provides a consistency check on the attribution. It is
reported, with its result, in §5.2.

The `step_at_k` sweep measures the score's dependence on the crossing frame
directly, giving a second, independent route to the same quantity. Two routes to
one number is what makes the result safe to publish.

The instrument also checks itself: `step_at_000` and `constant_0.51` are
separately generated files describing the same curve, and they receive identical
public and private scores. (`step_at_140` also scores identically to
`never_crosses`, but we use that equality as a *measurement* in §5.4 and so
cannot also count it as a validity check.) Throughout, "identical" means
agreeing at the five decimal places the leaderboard displays, which is all the
precision it exposes.

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

### 5.1 A constant matches the eighth-placed score

The final private leaderboard, as published:

| Rank | Team | Private score |
|---|---|---|
| 1 | CVLAB | 2.50165 |
| 2 | BUPT MIC Lab | 2.46234 |
| 3 | Tianhao Zhao | 2.42361 |
| 4 | cr-tfx | 2.41555 |
| 5 | wtiaw_tiaw | 2.39725 |
| 6 | IsaacfI | 2.39027 |
| 7 | yyttll | 2.36844 |
| 8 | Paulini38 | 2.35291 |
| 9 | SuryaInBytes (this work) | 2.00585 |
| 10 | Song_Ren | 1.46008 |
| 11 | WDL | 1.35992 |
| 12 | YooHyun.2 | 1.04203 |
| 13 | Ratnachand Kancharla | 1.03913 |
| — | benchmark row (`submission.csv`) | 0.58333 |

Our video-blind controls, submitted after the deadline and scored by the same
service against the same private split, are *not* leaderboard entries and are
reported separately:

| Video-blind control | Private score |
|---|---|
| **`constant_0.51`** | **2.35291** |
| `linear_ramp` | 1.06725 |
| `never_crosses` | 0.35105 |

The constant's score and the eighth-placed team's score agree at the precision
the leaderboard displays. Per §1.2, the single inference we draw is that the
metric assigns those two submissions the same value; we do not infer anything
about what the team submitted, and we note that a whole family of curves — every
curve exceeding 0.5 at frame 0 that never dips — receives that same value.

Three readings of these tables, in increasing order of importance.

**The weak reading.** A submission with no parameters and no input outscores
five of thirteen teams.

**The stronger reading.** The gap between first place and the video-blind
constant is **0.14874**. Whatever the winning system extracts from 1,417 clips,
150 frames each, three modalities, the metric values it at 5.9% of the score it
awards that system. We quote the difference first and the percentage second on
purpose: the metric is a weighted sum with no meaningful origin, so a ratio of
two scores can be made to say almost anything by shifting the scale. The
difference and the decomposition below are scale-free; the percentage is a
convenience, not evidence.

**The strongest reading, and the one that generalises.** As shown in §2.3, the
constant is not a good baseline that happens to score well — it is the *exact
maximiser* of both timing terms, simultaneously, on every clip. No submission
can exceed it on those terms; a submission can only match it, and then compete
on AP and AUC. Every point of separation visible on this leaderboard above
2.35291 must therefore have come from the discrimination terms, and §5.2 shows
those terms are worth 0.35105 to a blind submission. The 0.14874 that separates
first place from a constant is what the discrimination terms bought the winner
above chance.

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
| AP + AUC, to a blind submission | **0.35105** |
| TTA@0.5, saturated | **0.87593** |
| STTA@0.5, saturated | **1.12593** |
| **Timing total, saturated** | **2.00186 = 5.70 × the discrimination floor** |

Read that ratio precisely. It compares the timing terms *at their maximum*
against the discrimination terms *at chance* — not maximum against maximum. The
maxima ratio is not identifiable from outside, because `w_AP + w_AUC` is
undisclosed; but it is bounded, because the winner's discrimination terms are
worth at least `0.14874 + 0.35105 = 0.49979`, so `(w_TTA+w_STTA)·E[t_ai] /
(w_AP+w_AUC) ≤ 2.00186/0.49979 = 4.01`. Both framings say the same thing and we
use the measured one: **a submission that reads nothing collects 2.00186, and
the best discrimination anyone demonstrated on this benchmark added 0.14874 to
it.**

The composite is not a weighted average of four comparable quantities. It is a
timing score with a discrimination correction.

Note also that `constant_0.51` and `constant_0.99` score **identically**. The
metric reads only whether the curve is above 0.5; it never reads by how much. A
system that is certain and a system that is barely committed receive the same
credit, which removes any incentive to calibrate.

**The consistency check, and its residual.** `cross_then_dip` dips below the
threshold across frames 50–59, which should move the STTA reference point to
frame 60 while leaving TTA at frame 0. With the per-frame STTA coefficient
implied above (1.12593/120.1 = 0.009375), that predicts
2.35291 − 60 × 0.009375 = 1.79041. The measured score is **1.85347** — the
observed loss corresponds to a reference point at frame 53.3 rather than 60, an
11% discrepancy. So the additive TTA/STTA attribution is close but not exact.
The floor and the timing total are measured directly from flat curves and are
unaffected; it is the *split* of 2.00186 into 0.87593 and 1.12593 that this
residual puts a tolerance on, and we report the split with that caveat attached
rather than to five decimals of implied precision.

### 5.3 The score is linear in the crossing frame

| Crossing frame k | 0 | 10 | 25 | 50 | 75 | 100 | 125 | 140 |
|---|---|---|---|---|---|---|---|---|
| Private score | 2.35291 | 2.18624 | 1.93732 | 1.52088 | 1.10445 | 0.68822 | 0.37967 | 0.35105 |

Over `k ≤ 100` the relationship is a straight line:

> `score = 2.35304 − 0.016647 k`,   i.e. **one point of score per 60.1 frames of
> delay**, or 0.4994 per second at 30 fps.

Past `k = 100` the curve bends, as clip after clip has its accident occur before
the alarm fires, and by `k = 140` the score has returned to the floor.

Two conventions have to be fixed before any curve's crossing frame can be
stated, and both were determined empirically rather than assumed: indices are
0-based, and the comparison is strict (`p_t > 0.5`). We report the crossing
frame of every submitted curve as the index of its first value exceeding 0.5,
computed from the submitted file itself. One frame is worth 0.0166 of score, so
an off-by-one here would exceed every residual below.

**Held-out prediction.** Five curves were submitted independently and were not
used to fit the line:

| Curve | Crosses at | Predicted | Measured | Error |
|---|---|---|---|---|
| `sigmoid_m0.40` | 60 | 1.35422 | 1.35428 | +0.00006 |
| `sigmoid_m0.50` | 75 | 1.10452 | 1.10435 | −0.00017 |
| `sigmoid_m0.60` | 90 | 0.85481 | 0.84840 | −0.00641 |
| `linear_ramp` | 75 | 1.10452 | 1.06725 | **−0.03727** |
| `sigmoid_m0.70` | 105 | (0.60511) | 0.55763 | (−0.04748) |

The first three are predicted to within 6.4×10⁻³, and the best of them to
6×10⁻⁵. `sigmoid_m0.70` crosses at frame 105, beyond the `k ≤ 100` region where
the line was fitted and where the curve is already bending, so the model is not
expected to hold there and its residual is bracketed. We list it because
omitting a held-out point that missed would be the wrong kind of reporting.

**`linear_ramp` is a genuine discrepancy and we do not have an explanation.**
It crosses at frame 75, exactly like `sigmoid_m0.50` (1.10435) and
`step_at_075` (1.10445) and a third curve, `step_at_half` (1.10442) — three
independent curves agreeing to 1×10⁻⁴ — yet it scores 0.0372 lower. That gap is
2.24 frames of slope, which is not an integer, so it cannot be a crossing-frame
effect. Two candidate explanations survive, and each costs something:

- The scorer's AP or AUC is *not* invariant to curve shape among video-blind
  submissions — for instance because it pools frames rather than reducing each
  clip to one value. This would undermine the cancellation argument of §4.2. It
  is, however, hard to reconcile with `step_at_140` and `never_crosses` scoring
  identically, since those two curves have very different frame orderings.
- The timing terms read something beyond the first crossing that the four
  agreeing curves happen to share and the ramp does not.

We have submitted a diagnostic family that holds the crossing frame at 75 and
varies only the shape before it, after it, and both, to separate these. The
result will be reported in the final version; until it is, the honest statement
of scope is that **the decomposition is established for step-like curves, the
floor 0.35105 is measured on flat curves, and the crossing-frame model predicts
step-like curves well and one graded curve poorly.** The headline result — a
constant scoring 2.35291 — does not depend on any of this, because both
`constant_0.51` and `never_crosses` are flat curves and their difference is
measured directly.

### 5.4 A property of the withheld ground truth, recovered

The linearity has a consequence the organisers did not intend to publish.

For a step at frame `k`, both timing terms equal `max(t_ai − k, 0)` on each
contributing clip, so the score above the floor is
`c · E[max(t_ai − k, 0)]` with `c = w_TTA + w_STTA`. Differentiating,
`d/dk` of that expectation is `−P(t_ai > k)`: **the slope at `k` measures the
survival function of the accident-onset distribution, scaled by `c`.** The
leaderboard is therefore reporting, one submission at a time, the shape of an
annotation no entrant has seen.

Segment slopes, from consecutive pairs in the table above:

| Interval | 0–10 | 10–25 | 25–50 | 50–75 | 75–100 | 100–125 | 125–140 |
|---|---|---|---|---|---|---|---|
| −slope | 0.016667 | 0.016595 | 0.016658 | 0.016657 | 0.016649 | 0.012342 | 0.001908 |

Three things follow, stated with the bounds they actually support.

**`c ≈ 1/60`.** The initial segment gives 0.016667, which is 1/60.00 to five
figures; since `P(t_ai > 0) = 1`, that segment estimates `c` directly.

**Almost no onset before frame 100.** The ratio of the 75–100 slope to the 0–10
slope is 0.016649/0.016667 = 0.99894, so `P(t_ai ≤ 100) ≤ 0.11%` of contributing
clips — about one or two clips in the private split, not zero, and we do not
claim zero.

**Almost none after frame 140.** `step_at_140` and `never_crosses` agree at the
displayed precision, so the excess `E[max(t_ai − 140, 0)]` is below
`5×10⁻⁶/0.016667 ≈ 3×10⁻⁴` frames. That bounds a *mean*, not a support: it
permits a handful of late-onset clips while remaining invisible at five decimals.

Dividing the timing total by `c`:

> **The accident onset lies between frames 100 and 140 in all but a fraction of
> a percent of accident clips, with mean 120.1 (120.3 using the fitted rather
> than the initial slope) — that is, at about 4.0 s of a 5.00 s clip.**

This is a measurement of hidden annotations obtained without them, and it
explains mechanically why the constant does so well on this corpus: because the
accident is always late in the window, an alarm at frame 0 is credited with
roughly 120 frames — four seconds — of anticipation on every clip.

**An unexplained residual.** Under this model `|slope|` must be non-increasing
in `k`, because a survival function cannot rise. It is not: the 25–50 segment is
steeper than the 10–25 segment by 6.3×10⁻⁵, roughly ninety times the ±6.7×10⁻⁷
that five-decimal rounding permits over a fifteen-frame span. The straight line
of §5.3 is therefore very good but not exact, and the deviation is real. We
suspect it is the same second-order effect that produces the `linear_ramp`
discrepancy, and we report it rather than fit around it.

**Two conventions this rests on, neither of them verified.** When `k > t_ai` the
set `{t_a : p_t > 0.5, t_a ≤ t_ai}` is empty and the definition as published
gives an undefined maximum; we assume the scorer takes it as zero. We also
assume clips with no accident contribute nothing to the timing terms. Both
assumptions are load-bearing for the `k > 100` region and we flag them as such.

**What we deliberately do not do is separate the weights from the metric
values.** All quantities above are products `w · metric`. Note in particular
that the ratio of the timing total to the slope equals the mean onset
*conditional on the clips that contribute*, and that the positive rate cancels
out of that ratio identically — so this measurement says nothing whatever about
what fraction of the corpus contains an accident, and we make no such claim. The
products are what determine the ranking, which is why reporting them is
sufficient; separating a weight from a metric value would require the ground
truth we correctly do not have.

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

Public and private scores agree to within 0.00584 across all eighteen, so the
*blind* scores are stable across the split. We claim nothing from this about
whether the teams' relative positions are split-stable, which we cannot observe.

---

## 6. Analysis: why the composition fails

*Scope, fixed before the argument rather than after it: what follows is an
analysis of a metric's algebra, illustrated by measurements on one benchmark
with thirteen teams. Properties (a) and (b) are consequences of the definitions
in §2.3 and hold wherever those definitions are used. Property (c) is a
statement about what the algebra implies, not a survey finding. We have not
audited the literature, and §8 states what that costs us.*

The defect is not a badly chosen weight. It is a type error.

Write the score as `s = B + U`, where `B = w_AP·AP + w_AUC·AUC` is bounded above
by `w_AP + w_AUC`, and `U = w_TTA·TTA + w_STTA·STTA` is bounded above by
`(w_TTA + w_STTA)·E[t_ai]`, a quantity that depends on the *corpus windowing*.
Three properties follow.

**(a) The unbounded term grows with clip length while the bounded one does
not.** `max U / max B` scales linearly in `E[t_ai]`, so two research groups
adopting the same weights on corpora windowed differently are not using the same
metric, even though they report the same metric's name. On this benchmark
`max U` is 2.00186, measured; `max B` is not identifiable from outside but is at
least 0.49979, giving `max U / max B ≤ 4.01`. Against `B` at chance the measured
ratio is 5.70.

**(b) The unbounded term has a constant as its global maximiser.** Any
*unconditioned* threshold-crossing earliness measure is maximised by crossing
immediately and never returning; this holds for TTA and STTA as defined here.
The maximiser reads no input, so the term it maximises can carry no information
about the input. The qualifier matters and is the seed of the fix: an earliness
measure that is *conditioned* — restricted to correctly classified clips, or
evaluated at a fixed operating point of the discrimination metric, or penalised
by the false-alarm rate — is not maximised by a constant. §7.2 recommends
exactly such a variant.

**(c) Consequently the separating power of the score lies in `B`.** To the
extent that a submission crosses the threshold early and holds it, its `U` is at
or near the maximum and it is separated from other such submissions only by `B`.
We cannot verify this for any particular entry — we have no team's curves but
our own — so we state it as what the metric's algebra implies for early-crossing
submissions, not as an observation about anyone's system.

Point (c) is why the failure is hard to notice from inside. A leaderboard of
this shape can still order methods usefully, because `B` still varies. What is
invisible without a control is that the *number* attached to each of them is
dominated by a term available for free, so the reported score — the thing that
goes into a paper's table and gets compared across publications — is mostly a
constant.

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

2. **Bound the earliness term, and condition it.** These are two separate
   repairs and both are needed. *Bounding* — normalising per clip so
   `TTA/t_ai ∈ [0,1]` — stops the term dominating the sum and stops the metric
   depending on how the corpus was windowed; it does **not** remove the constant
   as maximiser, which still scores 1 and still gets it for free. *Conditioning*
   is what removes the exploit: report earliness at a fixed operating point of
   the discrimination metric — mean TTA at 80% recall is the established form —
   so that a submission must first classify before its earliness is counted. If
   only one is adopted, adopt conditioning. (§6a, §6b)

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
Withholding the weights does not prevent the metric being exploited — this paper
exploited it comprehensively without them, and eighteen submissions were enough
to recover most of its structure — while it does prevent the one-line sanity
check that would have surfaced the problem before the competition opened.

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

**One discrepancy in the crossing-frame model is open.** `linear_ramp` scores
0.037 below three curves that cross the threshold at the same frame (§5.3), and
the segment slopes are not monotone as the model requires (§5.4). The
diagnostics that separate the candidate explanations have been submitted and
will be reported. Neither the headline result nor the timing total depends on
the resolution — both are measured between flat curves — but the *split* of the
timing total between TTA and STTA, and the interpretation of the fit as exact,
do.

**The character of the venue.** This is a community competition with an
undisclosed custom metric, not a long-running benchmark with a published scorer.
That is what made the probe experiment possible; it also means we cannot inspect
the scoring code, and every statement about the metric's implementation in this
paper is an inference from its outputs.

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
inference (Gururangan et al., 2018; Poliak et al., 2018): a model given only
half of each input scored far above chance,
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
5.70 times what the discrimination terms are worth to a blind submission, they
are maximised exactly by a constant that reads no pixels, and that constant
scores within 0.14874 of the winning entry and matches the eighth-placed team's
score at the precision the leaderboard displays. Using eighteen video-blind
submissions as probes we recovered most of the metric's functional form from
outside the competition — predicting held-out step-like curves to within
6.4×10⁻³, failing on one graded curve in a way we report rather than smooth —
and along the way recovered the support and mean of the withheld accident-onset
distribution.

The remedy is cheap. Bound the earliness term, or report it separately. And run
the control: a constant costs one submission and settles in an afternoon a
question that otherwise propagates through a literature.

We report our own entry, which scored below the constant, as the case study.

---

## Reproducibility

All code, all submission files, and the complete score log will be released on
publication. *(Repository URL withheld here for double-blind review; it is
supplied to the editor.)* The central result is one line:

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
