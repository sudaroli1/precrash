# Mendeley Data deposit — what to put in each field

Upload `precrash_data.zip` (1.2 MB, 50 files). Then:

## Title

```
Video-blind probe submissions and returned leaderboard scores, Zero-shot Accident Anticipation benchmark
```

## Description

```
Twenty-six submissions to the Zero-shot Accident Anticipation competition
(AUTOPILOT, 2026) and the private-leaderboard scores the organisers returned
for each, together with the code that generated them and the analysis that
decomposes the competition's composite metric from those scores alone.

Twenty-five of the submissions are video-blind: each is a risk curve computed
from a closed-form expression in the frame index, reading no video and using no
label. They include constants, step functions crossing the alarm threshold at
named frames, logistic curves, and diagnostics designed so that the metric's
undisclosed weights cancel when two submissions are differenced. The
twenty-sixth is the organisers' own sample submission, resubmitted verbatim as
a control on the submission path. Every file is a full-length prediction over
all 1,417 clips of the private split.

These data support the finding that a constant risk score of 0.51 scores
2.35291 on the private leaderboard — matching the eighth-placed team of
thirteen and exceeding five entries including the authors' own — and the
decomposition that attributes 0.35105 of that to discrimination and 2.00186 to
the two timing terms.

The competition's corpus is not included and is not redistributed: its rules
restrict it to academic and non-commercial use and prohibit redistribution, and
it is obtainable from the organisers on those terms. No file here contains any
part of that corpus, and none contains a label — the competition publishes
none, which is the premise of the study.
```

## Categories

Road safety; Traffic safety; Machine learning; Research methodology
(pick the closest Mendeley offers — road/traffic safety first, then
machine learning)

## Licence

**CC BY 4.0** for the data. The code in `scripts/` is MIT; the archive
carries its own `LICENSE` file saying so.

## Related links

- Article: add the DOI once the paper is accepted
- Code: https://github.com/sudaroli1/precrash
- Competition: https://www.kaggle.com/competitions/zero-shot-taa

## Before you publish the deposit

Mendeley Data lets you keep a dataset private and share a reviewer link. Do
that until the paper is accepted — then publish, so the DOI resolves for
readers.
