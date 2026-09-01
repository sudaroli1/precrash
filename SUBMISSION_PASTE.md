# Everything you'll paste or send tonight

Four things, in the order you'll need them. Nothing here needs editing except
where marked.

---

## 1. The abstract, as plain text for the OpenReview form

OpenReview's abstract box takes plain text, not LaTeX. This version has the
em-dashes as real characters and no markup left in it. It is also saved as
`latex/abstract_plaintext.txt` so you can copy it without scrolling.

> Traffic-accident anticipation is increasingly scored by composite metrics
> that add an earliness term — time-to-accident at a fixed risk threshold — to
> discrimination terms such as average precision and area under the ROC curve.
> We show that this composition fails structurally rather than incidentally.
> The discrimination terms are bounded on the unit interval while the earliness
> terms scale with clip length, and the earliness terms have a trivial global
> maximiser: a constant risk score above the threshold attains, on every clip,
> the largest value that clip admits. We demonstrate the consequence on a live
> benchmark. On the Zero-shot Accident Anticipation competition (AUTOPILOT-COG;
> 1,417 clips curated from MM-AU; 13 teams), a constant risk score of 0.51,
> which reads no pixels, scores 2.35291 on the private leaderboard — matching
> the eighth-placed team at the precision the leaderboard displays, exceeding
> five of thirteen teams including our own, and falling 0.14874 short of the
> winning score. Using twenty-five video-blind submissions as probes, designed
> so that unknown terms cancel under differencing, we then decompose the metric
> from outside the competition without access to any label: average precision
> and area under the curve together contribute 0.35105 to a blind submission,
> the two timing terms contribute 2.00186 — 5.70 times as much — and the score
> falls at almost exactly one sixtieth of a point per frame of delay, confirmed
> by a one-frame experiment. The same probes place the withheld accident-onset
> distribution, recovering a mean of 120.1 frames in a 150-frame clip and
> bounding what can lie beyond frame 140, and establish a second result: four
> curves that cross the threshold at the same frame, and never fall below it
> afterwards, score 0.091 apart, a spread the published metric definition
> cannot produce. The benchmark's stated formula therefore does not reproduce
> its own scorer. We report our ninth-placed entry, a training-free ensemble
> that scores below the constant, as the case study, and close with a protocol
> whose central requirement is that every anticipation result be reported
> alongside a video-blind control.

**Title**, for the title box:

> Unbounded Timing Terms Make a Composite Accident-Anticipation Score
> Video-Blind: Evidence from a Live Benchmark

---

## 2. To your co-authors, about their OpenReview profiles

Send this now so it is not what holds tonight up. OpenReview will not let the
submission through with an incomplete profile, and the fix takes each of them
about ten minutes.

> Subject: OpenReview profile — needed before we submit tonight
>
> Hi both,
>
> The accident-anticipation paper is ready and I want to submit it to TMLR
> tonight. Before I can, OpenReview needs each of us to have a complete
> profile, so could you do this today:
>
> 1. Go to openreview.net and sign in, or sign up if you don't have an account.
>    Confirm the email address on the account.
> 2. Fill in **Education & Career History** completely. It has to cover every
>    year with no gaps — OpenReview flags gaps and won't let the submission
>    through.
> 3. Fill in **Advisors & Other Relations** and **Expertise**.
> 4. Tell me the exact name on your profile and the email you used, so the
>    names I type on the submission match your profiles.
>
> One thing I need from you before I submit, because it cannot be changed
> afterwards — TMLR's policy is that no author can be added or removed after
> submission, no exceptions. Please confirm:
>
> - the affiliation you want printed, and
> - that you're happy to be listed as an author, which for TMLR means you made
>   a substantial intellectual contribution, you contributed to drafting or
>   revising, and you take full responsibility for the content.
>
> The paper is attached. Please read at least the abstract and Section 5 before
> you confirm — it reports that our own entry scored below a constant that
> reads no pixels, and that's the point of the paper rather than a problem with
> it, but you should know that's what your name is going on.
>
> Thanks,
> Sudaroli

---

## 3. To the AUTOPILOT organisers

Courtesy copy, before you submit. Keep it factual and offer them the data —
they may well want to fix the scorer, and a friendly organiser is worth more
than a surprised one.

> Subject: Findings from the Zero-shot Accident Anticipation benchmark
>
> Dear organisers,
>
> Thank you for running the Zero-shot Accident Anticipation competition. We
> entered it (ninth of thirteen) and afterwards used the late-submission
> facility to study the scoring function itself. We're about to submit the
> write-up to a journal and wanted you to see it first.
>
> Two findings you should know about:
>
> 1. A constant risk score of 0.51, identical for every clip and computed
>    without reading a single frame, scores 2.35291 on the private leaderboard
>    — equal to the eighth-placed team at the precision the leaderboard
>    displays, above five of thirteen teams including ours, and 0.14874 short
>    of the winning score. The two timing terms are worth 5.70 times what
>    average precision and AUC contribute to such a submission, and a constant
>    above the threshold maximises both of them exactly.
>
> 2. The published formula does not appear to reproduce the scorer. Four curves
>    that first exceed 0.5 at frame 75 and never fall below it afterwards score
>    0.091 apart. Under the definition on the Evaluation page they should score
>    identically. We report this as an open discrepancy with two candidate
>    explanations rather than claiming to have resolved it.
>
> All twenty-five video-blind submissions were made through the late-submission
> facility you left open, within the stated daily limit. Every one is a curve
> generated from a closed-form expression in the frame index; none was derived
> from any label, and we redistribute no part of the dataset. We cite MM-AU and
> the competition as your rules require.
>
> The paper is attached, and we're happy to share the submission files and the
> score log. If any of the above rests on a misreading of the evaluation setup
> we would genuinely rather know before publication than after, so please do
> tell us.
>
> Best regards,
> Sudaroli Dhananjeyan

**Attach:** `latex/main_preprint.pdf` — the named version, not the anonymous
one.

---

## 4. Action editors, for the email that arrives after you submit

You'll be asked by email to recommend action editors. Reply with three or four
names from https://www.jmlr.org/tmlr/editorial-board.html.

**Choose on this basis:** people who work on evaluation methodology,
benchmarking, measurement, or empirical rigour — *not* on accident
anticipation, autonomous driving or video understanding. An editor who reads
this as an anticipation paper will look for a method contribution it does not
make and does not claim. An editor who reads it as a paper about how a
composite score behaves will see what the evidence actually is.

> Thank you. Suggested action editors, in order of preference:
>
> 1. [name]
> 2. [name]
> 3. [name]
>
> The submission is not an accident-anticipation method paper — it is a
> measurement of how a composite anticipation metric behaves, demonstrated on a
> live benchmark, so an editor whose expertise is in evaluation methodology or
> benchmark design would be a better fit than one in video understanding or
> autonomous driving.
>
> We have no conflicts with any of the above. [Or: we have a conflict with
> [name] — ...]

---

## Field answers for the form itself

| Field | Answer |
|---|---|
| Human subjects / IRB | Not applicable. No human subjects: the work analyses a scoring function using curves generated from the frame index, on dashcam clips the authors did not collect. |
| Funding | [state it, or "none"] |
| Competing interests | The authors were entrants in the competition analysed in this paper, placing ninth of thirteen. This is disclosed in the paper (Section 5.1 and the Compliance section). |
| Broader impact statement | Not required. The work carries no significant risk of harm. |
| Previous submission URL | (blank — first submission) |

**Upload:** `latex/main.pdf` and `tmlr_supplementary.zip`. Nothing else — and
not `latex/`, which contains your repository URL in `convert.py`.
