# Leaderboard submissions and what they establish

Every score below was computed by the competition organisers against ground
truth no entrant has. No metric in this table was implemented by us.

Competition: https://www.kaggle.com/competitions/zero-shot-taa
Submitted 31 August 2026. Late submission open; cap is 100/day.

## The result

**A constant risk score of 0.51, reading no pixels, scores 2.35291 — eighth of
thirteen, above our own ensemble at 2.00585 and above four other teams. It
reaches 94.1% of the winning score.**

| | Official score |
|---|---|
| CVLAB (1st) | 2.50165 |
| Paulini38 (8th) | 2.35291 |
| **a constant 0.51 — reads no pixels** | **2.35291** |
| SuryaInBytes (9th, ours) | 2.00585 |
| 13th | 1.03913 |
| organisers' sample submission | 0.58333 |

The constant's score equals the eighth-place score to five decimals.

## The decomposition

Official score = `w_AP*AP + w_AUC*AUC + w_TTA*TTA@0.5 + w_STTA*STTA@0.5`.

Every probe is constant across clips, so each earns base-rate AP and chance AUC;
those terms are identical for all of them and cancel in any difference.

| Probe | Private | Isolates |
|---|---|---|
| `never_crosses` (0.49 throughout) | 0.35105 | the AP + AUC floor alone |
| `cross_then_drop` (0.51 at frame 0, 0.49 after) | 1.22698 | floor + TTA |
| `constant_0.51` | 2.35291 | floor + TTA + STTA |

| Term | Value |
|---|---|
| AP + AUC floor | **0.35105** |
| TTA term | **0.87593** |
| STTA term | **1.12593** |
| timing total | **2.00186** = **5.70x** the floor |

## Score is linear in the crossing frame

| Crossing frame | Private | slope per frame |
|---|---|---|
| 0 | 2.35291 | |
| 10 | 2.18624 | -0.016667 |
| 25 | 1.93732 | -0.016595 |
| 50 | 1.52088 | -0.016658 |
| 75 | 1.10445 | -0.016657 |
| 100 | 0.68822 | -0.016649 |
| 125 | 0.37967 | -0.012342 |
| 140 | 0.35105 | -0.001908 |

Linear fit over k <= 100: `score = 2.35304 - 0.016647*k`, i.e. **1/60 per frame**.
The bend past 125 is clips whose accident has already happened; at 140 the
score equals the floor to the five decimals the leaderboard displays. That
bounds the mean excess beyond frame 140 below 3e-4 frames — a bound on a MEAN,
not on the support. A few clips with a later onset would be invisible at this
precision, and we do not claim there are none. The slope
implies a mean accident frame of **120.3 of 150**.

### The fit predicts curves it was not fitted to

| Curve | Crosses | Predicted | Measured | Error |
|---|---|---|---|---|
| `sigmoid_m0.40` | 60 | 1.35422 | 1.35428 | +0.00006 |
| `sigmoid_m0.50` | 75 | 1.10452 | 1.10435 | -0.00017 |
| `sigmoid_m0.60` | 90 | 0.85481 | 0.84840 | -0.00641 |
| `linear_ramp` | 75 | 1.10452 | 1.06725 | -0.03727 |

## Magnitude is irrelevant

`constant_0.51` and `constant_0.99` both score **2.35291**, identically. Only
whether the score crosses 0.5 matters, never by how much.

## Full results

| File | Public | Private |
|---|---|---|
| `p_constant_0.51` | 2.35057 | 2.35291 |
| `p_constant_0.99` | 2.35057 | 2.35291 |
| `p_step_at_000` | 2.35057 | 2.35291 |
| `p_step_at_010` | 2.18391 | 2.18624 |
| `p_step_at_025` | 1.93391 | 1.93732 |
| `p_cross_then_dip` | 1.85057 | 1.85347 |
| `p_step_at_050` | 1.51724 | 1.52088 |
| `c_sigmoid_m0.40` | 1.35057 | 1.35428 |
| `p_cross_then_drop` | 1.22573 | 1.22698 |
| `p_step_at_075` | 1.10057 | 1.10445 |
| `c_sigmoid_m0.50` | 1.10057 | 1.10435 |
| `c_linear_ramp` | 1.06604 | 1.06725 |
| `c_sigmoid_m0.60` | 0.84398 | 0.84840 |
| `p_step_at_100` | 0.68391 | 0.68822 |
| `c_sigmoid_m0.70` | 0.55190 | 0.55763 |
| `p_step_at_125` | 0.37383 | 0.37967 |
| `p_step_at_140` | 0.35088 | 0.35105 |
| `p_never_crosses` | 0.35088 | 0.35105 |

`00_replicate_sample.csv` reproduces the organisers' reference file
byte-for-byte (1417/1417 rows), which fixed the id order and every formatting
decision. We expected it to score the 0.58333 shown on the leaderboard's
benchmark row, so we planned not to spend a slot on it. We submitted it anyway,
in the diagnostics batch below, and **it scored 1.04203** — so the distributed
sample and the published benchmark row are two different files. That is the
26th submission, and it is why the total is 26 rather than 25.

---

# Diagnostics, 31 August 2026 — the crossing-frame model is incomplete

Six submissions. Five hold the threshold crossing at frame 75 and vary only the
shape of the curve away from it; the sixth is the organisers' own file verbatim.

| File | Public | Private | vs step_at_075 |
|---|---|---|---|
| `d_pre_graded` (0 -> 0.49 run-up, then 0.51) | 1.12598 | **1.12964** | +0.02519 |
| `d_step_at_075` (0.49 / 0.51 step) | 1.10057 | **1.10445** | reference |
| `d_step_at_076` (same, one frame later) | 1.08391 | **1.08779** | -0.01666 |
| `d_both_graded` (0 -> 1 throughout) | 1.06604 | **1.06725** | -0.03720 |
| `d_post_graded` (0.49, then 0.51 -> 0.99) | 1.03507 | **1.03871** | -0.06574 |
| `00_replicate_sample` (organisers' file) | 1.04063 | **1.04203** | — |

## 1. The crossing-frame law is exact for steps

`step_at_075 - step_at_076 = 0.01666`, against 1/60 = 0.016667. One frame of
delay costs exactly one sixtieth of a point. The slope is confirmed to the
displayed precision by a dedicated one-frame experiment.

## 2. The score reads the curve away from the crossing, which the published
## metric definition cannot do

`d_step_at_075`, `d_pre_graded` and `d_post_graded` have **identical values at
frames 74 and 75**, cross 0.5 at frame 75, and never dip afterwards. Under the
published definitions -- TTA and STTA are functions of threshold crossings
alone, and every video-blind curve is identical across clips so AP and AUC
cannot separate them -- these three must score identically.

They span 0.09093, which is 5.5 frames of slope.

Direction of the effect, which is the useful part:

- grading the run-up **down** (0 -> 0.49 instead of a flat 0.49) **gains** 0.025
- grading the run-out **up** (0.51 -> 0.99 instead of a flat 0.51) **loses** 0.066

Lower before the alarm helps; higher after the alarm hurts. Neither is
expressible in the published formula.

The two effects are nearly but not exactly additive: -0.04055 predicted against
-0.03720 measured for the combination.

## 3. The linear_ramp anomaly is real and reproducible

`d_both_graded` was generated independently and scores **1.06725** — identical
to `02_linear_ramp`. So the outlier is a property of graded curves, not a
one-off fault in an early submission.

## 4. The organisers' sample submission scores 1.04203, not 0.58333

Submitted verbatim, byte-identical to the distributed file. So the leaderboard's
0.58333 benchmark row is **not** `sample_submission.csv`, and the draft was right
to refuse the attribution.

Separately: 1.04203 is also the private score of the twelfth-placed team. As
with the eighth-place tie, state the arithmetic and infer nothing further.

## What survives untouched

The headline and the decomposition are measured between **flat** curves —
`constant_0.51` (2.35291) and `never_crosses` (0.35105) — which have no shape to
vary. Timing total 2.00186 and the 5.70x ratio stand.

What now carries a caveat: the *split* of the timing total into TTA and STTA,
and any claim that the linear fit is exact for arbitrary curves. It is exact for
steps and approximate otherwise.

## What this adds to the paper

A second finding, and a sharper one for a methodology venue: **the published
metric definition does not reproduce the scorer's behaviour.** An entrant cannot
compute this benchmark's score from its stated formula, even given the labels.
That strengthens protocol item 6 from "publish a control's score" to "publish
the scorer."
