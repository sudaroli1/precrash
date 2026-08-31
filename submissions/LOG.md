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
The bend past 125 is clips whose accident has already happened; at 140 the score
equals the floor exactly, so every accident occurs by frame 140. The slope
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

`00_replicate_sample.csv` was not submitted: it reproduces the organisers'
reference file byte-for-byte (1417/1417 rows), which confirms the id order and
every formatting decision without spending a slot, and its score is already
published as 0.58333.

## Still to submit

The model variants, once features are extracted:
`ensemble_postproc`, `clip_only`, `flow_only`, `prior_only`, `ensemble_raw`.
Use `make_submission.py --features DIR --variant NAME`. Every ablation row then
carries a third-party score.
