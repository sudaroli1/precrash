"""
Pass 4: Sections 5, 6, 7, 8, the Conclusion as 9, and two repairs.

Repairs first:
  * The control and probe paragraphs added last pass were anchored on the wrong
    paragraph and landed inside the Conclusion. They move to the end of the
    method, where they belong.
  * The "sections not yet written" marker is removed, since they now are.

Section 5 is rebuilt around the leaderboard. Everything the authors measured
before the leaderboard is kept, under a subsection that says what it is: the
locally computable proxy, which is the only thing this corpus permits an
entrant to compute, and which Section 5.1 shows a constant maximises. The
paragraphs that presented proxy statistics as anticipation performance are
rewritten; the tables are untouched.

Sections 6, 7 and 8 are new. The Conclusion is rewritten and becomes 9. Future
Scope is parked, not deleted -- three of its six subsections describe work that
is still worth doing and can be folded into Section 8 or a short future-work
paragraph.
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp5/word/document.xml")


def text_of(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def set_text(p, text):
    runs = p.findall(W + "r")
    tmpl = copy.deepcopy(runs[0]) if runs else None
    for r in runs:
        p.remove(r)
    for tag in ("hyperlink", "bookmarkStart", "bookmarkEnd", "proofErr"):
        for el in p.findall(W + tag):
            p.remove(el)
    run = tmpl if tmpl is not None else etree.SubElement(p, W + "r")
    if tmpl is not None:
        for t in run.findall(W + "t"):
            run.remove(t)
        p.append(run)
    t = etree.SubElement(run, W + "t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


def build_table(template_tbl, rows):
    """Clone a 3-column table and refill it. rows[0] is the header."""
    tbl = copy.deepcopy(template_tbl)
    trs = tbl.findall(W + "tr")
    head_tmpl, body_tmpl = copy.deepcopy(trs[0]), copy.deepcopy(trs[1])
    for tr in trs:
        tbl.remove(tr)
    for i, cells in enumerate(rows):
        tr = copy.deepcopy(head_tmpl if i == 0 else body_tmpl)
        tcs = tr.findall(W + "tc")
        for tc, val in zip(tcs, cells):
            ps = tc.findall(W + "p")
            set_text(ps[0], val)
            for extra in ps[1:]:
                tc.remove(extra)
        tbl.append(tr)
    return tbl


# ------------------------------------------------------------------ tables
T_LEADERBOARD = [
    ["Rank", "Entry", "Private score"],
    ["1", "CVLAB", "2.50165"],
    ["2", "BUPT MIC Lab", "2.46234"],
    ["3", "Tianhao Zhao", "2.42361"],
    ["4", "cr-tfx", "2.41555"],
    ["5", "wtiaw_tiaw", "2.39725"],
    ["6", "IsaacfI", "2.39027"],
    ["7", "yyttll", "2.36844"],
    ["8", "Paulini38", "2.35291"],
    ["9", "SuryaInBytes (this work)", "2.00585"],
    ["10", "Song_Ren", "1.46008"],
    ["11", "WDL", "1.35992"],
    ["12", "YooHyun.2", "1.04203"],
    ["13", "Ratnachand Kancharla", "1.03913"],
    ["—", "benchmark row published on the leaderboard", "0.58333"],
]

T_CONTROLS = [
    ["Video-blind control (this work)", "Curve", "Private score"],
    ["constant_0.51", "0.51 at every frame", "2.35291"],
    ["constant_0.99", "0.99 at every frame", "2.35291"],
    ["linear_ramp", "0.000 rising to 1.000", "1.06725"],
    ["organisers' sample submission", "0.001 rising to 0.999, resubmitted verbatim", "1.04203"],
    ["never_crosses", "0.49 at every frame", "0.35105"],
]

T_PROBES = [
    ["Probe", "Private score", "What it isolates"],
    ["never_crosses (0.49 throughout)", "0.35105", "the AP + AUC floor alone; both timing terms are zero"],
    ["cross_then_drop (0.51 at frame 0, 0.49 after)", "1.22698", "floor + TTA; the drop destroys STTA"],
    ["cross_then_dip (dip across frames 50–59)", "1.85347", "STTA reference point pushed back"],
    ["constant_0.51", "2.35291", "floor + TTA + STTA"],
]

T_DECOMP = [
    ["Component", "Contribution", "Relative to the floor"],
    ["AP + AUC, to a video-blind submission", "0.35105", "1.00×"],
    ["TTA@0.5, saturated", "0.87593", "2.49×"],
    ["STTA@0.5, saturated", "1.12593", "3.21×"],
    ["Timing terms together", "2.00186", "5.70×"],
]

T_SWEEP = [
    ["Frame at which the curve crosses 0.5", "Private score", "Residual against the fitted line"],
    ["0", "2.35291", "fitted"],
    ["10", "2.18624", "fitted"],
    ["25", "1.93732", "fitted"],
    ["50", "1.52088", "fitted"],
    ["75", "1.10445", "fitted"],
    ["100", "0.68822", "fitted"],
    ["125", "0.37967", "beyond the linear region"],
    ["140", "0.35105", "equals the floor"],
    ["60 (logistic, held out)", "1.35428", "+0.00006"],
    ["75 (logistic, held out)", "1.10435", "−0.00017"],
    ["90 (logistic, held out)", "0.84840", "−0.00641"],
    ["105 (logistic, held out)", "0.55763", "outside the fitted region"],
]

T_DIAG = [
    ["Curve, all crossing 0.5 at frame 75", "Private score", "Against the step reference"],
    ["graded run-up (0 → 0.49, then 0.51)", "1.12964", "+0.02519"],
    ["step (0.49 / 0.51)", "1.10445", "reference"],
    ["graded throughout (0.000 → 1.000)", "1.06725", "−0.03720"],
    ["graded run-out (0.49, then 0.51 → 0.99)", "1.03871", "−0.06574"],
    ["step at frame 76 (one-frame calibration)", "1.08779", "−0.01666 = 1/60.0"],
]

# ------------------------------------------------------------ section 5 text
S5 = [
    ("h", "5.1. A video-blind constant on the official leaderboard"),
    ("p", "Every score reported in this section was computed by the "
          "competition's scoring service against ground truth we do not hold. "
          "We implemented none of the four metrics. All quantities derived from "
          "those scores — the decomposition, the fit, the ratios — are our own "
          "arithmetic and are identified as such."),
    ("p", "Table 6 reproduces the final private leaderboard as published. "
          "Table 7 gives the video-blind controls, submitted after the closing "
          "date and scored by the same service against the same private split. "
          "They are not leaderboard entries and are reported separately."),
        ("c", "Table 6: Final private leaderboard of the Zero-shot Accident "
          "Anticipation competition, as published."),
("t", "T_LEADERBOARD"),
        ("c", "Table 7: Video-blind controls submitted by this work. None reads a "
          "pixel."),
("t", "T_CONTROLS"),
    ("p", "A constant risk score of 0.51 scores 2.35291. That equals the "
          "eighth-placed team's score at the five decimal places the "
          "leaderboard displays, exceeds the scores of five of the thirteen "
          "teams including our own, and falls 0.14874 short of the winning "
          "score. Per the scope stated in Section 1, the only inference we draw "
          "from the coincidence at eighth place is that the metric assigns "
          "those two submissions the same value; a whole family of curves — "
          "every curve above 0.5 at frame 0 that never dips — receives it, and "
          "we infer nothing about what any team submitted."),
    ("p", "We quote the difference before the ratio deliberately. The metric is "
          "a weighted sum with no meaningful origin, so a ratio of two scores "
          "can be made to say almost anything by shifting the scale. The "
          "difference of 0.14874 and the decomposition below are scale-free; "
          "that the constant reaches 94.1% of the winning score is a "
          "convenience, not evidence."),
    ("p", "One further row is worth stating plainly. The organisers' own sample "
          "submission, resubmitted byte for byte, scores 1.04203. The benchmark "
          "row displayed on the leaderboard scores 0.58333, so those two are "
          "not the same file, and the sample submission is not the origin of "
          "the published benchmark score."),

    ("h", "5.2. Decomposing the score"),
    ("p", "Because every video-blind submission is identical across clips, all "
          "of them induce the same ordering over clips and earn the same "
          "average precision and the same area under the curve, which therefore "
          "cancel in any difference between two of them (Section 4). "
          "Differencing three probes separates the terms."),
        ("c", "Table 8: Probes designed so that the discrimination terms cancel."),
("t", "T_PROBES"),
        ("c", "Table 9: The score's components, recovered from Table 8 by "
          "differencing."),
("t", "T_DECOMP"),
    ("p", "Read the ratio precisely. It compares the timing terms at their "
          "maximum against the discrimination terms at chance, not maximum "
          "against maximum. The maxima ratio is not identifiable from outside, "
          "because the weights are undisclosed; it is bounded, because the "
          "winner's discrimination terms are worth at least 0.14874 + 0.35105 = "
          "0.49979, giving a maxima ratio of at most 4.01. Both framings say the "
          "same thing, and the measured statement is the safer one: a "
          "submission that reads nothing collects 2.00186, and the best "
          "discrimination anyone demonstrated on this benchmark added 0.14874 "
          "to it."),
    ("p", "Two further readings follow from Table 7. A constant at 0.51 and a "
          "constant at 0.99 score identically, so the metric reads only whether "
          "the curve stands above the threshold and never by how much; there is "
          "no incentive to calibrate. And the probe holding 0.51 throughout "
          "with a dip across frames 50 to 59 scores 1.85347, where the "
          "per-frame stable-timing coefficient predicts 1.79041 for a reference "
          "point moved to frame 60. The additive attribution of the timing "
          "total into its two parts is therefore close but not exact, and we "
          "report the split of 2.00186 into 0.87593 and 1.12593 with that "
          "tolerance attached. The total itself is measured directly between "
          "two flat curves and does not depend on the split."),

    ("h", "5.3. The score is linear in the frame at which the alarm fires"),
    ("p", "Frame indices are zero-based and the threshold comparison is strict; "
          "both conventions were determined empirically from the submitted "
          "files rather than assumed, because one frame is worth 0.0166 of "
          "score and an off-by-one would exceed every residual below."),
        ("c", "Table 10: Score against crossing frame. The first eight rows are "
          "step functions; the last four are logistic curves submitted "
          "independently and not used in the fit."),
("t", "T_SWEEP"),
    ("p", "Over the region in which the alarm still precedes the accident in "
          "every clip, the relationship is a straight line: the score falls by "
          "0.016647 for each frame of delay, which is one point per 60.1 "
          "frames, or 0.4994 per second at 30 frames per second. A dedicated "
          "one-frame experiment confirms the slope directly: a step at frame 75 "
          "scores 1.10445 and a step at frame 76 scores 1.08779, a difference "
          "of 0.01666 against 1/60 = 0.016667."),
    ("p", "The line was then asked to predict curves it had not seen. Three "
          "logistic curves crossing at frames 60, 75 and 90 are predicted to "
          "within 6.4 × 10⁻³, and the closest of them to 6 × 10⁻⁵. The fourth, "
          "crossing at frame 105, lies beyond the region where the line was "
          "fitted and where the curve has already begun to bend; the model is "
          "not expected to hold there and we report it rather than omit a "
          "held-out point that missed."),

    ("h", "5.4. What the published metric definition does not explain"),
    ("p", "One result does not fit, and we report it rather than smooth it. A "
          "linear ramp crosses the threshold at frame 75, exactly as three "
          "other curves do, and scores 0.037 lower than they do. That gap is "
          "2.24 frames of slope, which is not an integer, so it cannot be a "
          "crossing-frame effect. We therefore submitted a diagnostic family "
          "that holds the crossing frame at 75 and varies only the shape of the "
          "curve away from it."),
        ("c", "Table 11: Five curves that all first exceed 0.5 at frame 75, with "
          "identical values at frames 74 and 75 and no subsequent dip, plus a "
          "one-frame calibration step."),
("t", "T_DIAG"),
    ("p", "The three curves in the middle of Table 11 have identical values at "
          "frames 74 and 75, cross at the same frame, and never fall back "
          "afterwards. Under the metric as published they must score "
          "identically: the two timing terms are functions of threshold "
          "crossings alone, and video-blind curves are identical across clips, "
          "so the discrimination terms cannot separate them either. They span "
          "0.09093, which is 5.5 frames of slope. Grading the run-up downward "
          "gains 0.02519; grading the run-out upward loses 0.06574."),
    ("p", "We do not have an explanation, and we state the two candidates "
          "rather than choose between them. Either the discrimination terms are "
          "not invariant to curve shape among video-blind submissions — which "
          "would weaken the cancellation argument of Section 4, though it is "
          "hard to reconcile with a step at frame 140 and a flat 0.49 scoring "
          "identically despite very different frame orderings — or the timing "
          "terms read something beyond the first crossing that the three "
          "agreeing curves share and the ramp does not."),
    ("p", "Either way, one conclusion is available without resolving it, and it "
          "is a reportable finding in its own right: the benchmark's published "
          "formula does not reproduce its scorer. An entrant cannot compute "
          "this competition's score from its stated definition even given the "
          "labels. The headline result is unaffected, because the constant and "
          "the floor are both flat curves whose difference is measured "
          "directly; what carries a caveat is the split of the timing total and "
          "any claim that the linear fit is exact for arbitrary curves. It is "
          "exact for step-like curves and approximate otherwise."),

    ("h", "5.5. A property of the withheld ground truth, recovered"),
    ("p", "For a step at frame k, both timing terms equal max(t_ai − k, 0) on "
          "each contributing clip, so the score above the floor is c · E[max(t_ai "
          "− k, 0)] with c the sum of the two timing weights. Differentiating, "
          "the slope at k is −c · P(t_ai > k): the slope measures the survival "
          "function of the accident-onset distribution, scaled. The leaderboard "
          "is therefore reporting, one submission at a time, the shape of an "
          "annotation no entrant has seen."),
    ("p", "The initial segment estimates c at 0.016667, which is 1/60.00 to "
          "five figures. The ratio of the slope over frames 75 to 100 to that "
          "initial slope is 0.99894, so at most about 0.11% of contributing "
          "clips have their onset at or before frame 100 — a small number, not "
          "zero, and we do not claim zero. A step at frame 140 scores the same "
          "as a curve that never crosses, which bounds the mean excess beyond "
          "frame 140 below 3 × 10⁻⁴ frames; that bounds a mean, not a support, "
          "and permits a few late clips invisible at five decimals. Dividing "
          "the timing total by c gives a mean onset of 120.1 frames, or 120.3 "
          "using the fitted rather than the initial slope — about four seconds "
          "into a five-second clip."),
    ("p", "This explains mechanically why the constant does so well on this "
          "corpus. Because the accident is always late in the window, an alarm "
          "at frame 0 is credited with roughly 120 frames of anticipation on "
          "every clip."),
    ("p", "Two caveats belong with it. Under the model the magnitude of the "
          "slope must be non-increasing in k, because a survival function "
          "cannot rise; it is not, the segment from frame 25 to 50 being "
          "steeper than the segment from 10 to 25 by 6.3 × 10⁻⁵, roughly ninety "
          "times what five-decimal rounding permits. The straight line is very "
          "good but not exact, and we suspect the same second-order effect that "
          "produces the discrepancy in Section 5.4. Separately, when k exceeds "
          "t_ai the maximum in the published definition is taken over an empty "
          "set and is undefined; we assume the scorer takes it as zero, and "
          "that assumption is load-bearing for the region beyond frame 100."),
    ("p", "What we deliberately do not do is separate the weights from the "
          "metric values. Every quantity above is a product of a weight and a "
          "metric. Note in particular that the ratio of the timing total to the "
          "slope is the mean onset conditional on the clips that contribute, "
          "and that the positive rate cancels out of that ratio identically, so "
          "this measurement says nothing whatever about what fraction of the "
          "corpus contains an accident, and we make no such claim."),

    ("h", "5.6. The locally computable proxy, and what it was worth"),
    ("p", "What follows is the evaluation we conducted before any submission "
          "was scored, and it is retained because it is the case study this "
          "paper is built on. On a corpus that publishes no labels, average "
          "precision, area under the curve, a false-positive rate and any "
          "time-to-accident measured to a true onset are not computable by an "
          "entrant (Section 3.3). What is computable from a submission alone is "
          "the shape of its own risk curve, and the statistic we adopted was "
          "the mean frame at which that curve first exceeds 0.5 — the crossover "
          "frame."),
    ("p", "Every number in the remainder of Section 5 is a statistic of curve "
          "shape. None is an anticipation result, none was validated against a "
          "label, and the quantity they optimise is minimised absolutely, to "
          "frame 0, by the constant of Section 5.1. Read together with Table 7, "
          "the tables below record a system being tuned toward a target that a "
          "nine-character submission attains perfectly. We report them in that "
          "spirit: not as evidence that the system works, but as evidence of "
          "what a locally computable proxy does to a research programme when "
          "the benchmark withholds the means of checking it."),
    ("p", "Two specific cautions apply to the tables that follow. Timing "
          "figures in seconds were computed as the interval from the crossover "
          "frame to the END OF THE CLIP, not to an annotated collision, because "
          "this corpus annotates no collision; they are therefore inflated by "
          "the interval between the accident and the end of the window, which "
          "Section 5.5 measures at about one second on average. And the "
          "stable-anticipation compliance rate reported below is a consequence "
          "of the monotone clamp of Section 4, which makes falling below the "
          "threshold impossible after the first crossing; it measures that the "
          "clamp executed."),
]

# ------------------------------------------------------ sections 6, 7, 8, 9
S6 = [
    ("H", "ANALYSIS: WHY THE COMPOSITION FAILS"),
    ("p", "The scope of what follows is fixed before the argument rather than "
          "after it. This is an analysis of a metric's algebra, illustrated by "
          "measurements on one benchmark with thirteen teams. Properties (a) "
          "and (b) below are consequences of the definitions quoted in Section "
          "3.4 and hold wherever those definitions are used. Property (c) is a "
          "statement about what the algebra implies, not a survey finding."),
    ("p", "The defect is not a badly chosen weight. It is an error of type. "
          "Write the score as B + U, where B is the weighted sum of average "
          "precision and area under the curve and is bounded above by the sum "
          "of their two weights, and U is the weighted sum of the two timing "
          "terms and is bounded above by the sum of their weights times the "
          "mean accident onset — a quantity that depends on how the corpus was "
          "windowed. Three properties follow."),
    ("p", "(a) The unbounded term grows with clip length while the bounded one "
          "does not. The ratio of their maxima scales linearly in the mean "
          "onset, so two research groups adopting the same weights on corpora "
          "windowed differently are not using the same metric, even though they "
          "report the same metric's name. On this benchmark the maximum of U is "
          "2.00186, measured; the maximum of B is not identifiable from outside "
          "but is at least 0.49979, giving a ratio of at most 4.01. Against B "
          "at chance the measured ratio is 5.70."),
    ("p", "(b) The unbounded term has a constant as its global maximiser. Any "
          "unconditioned threshold-crossing earliness measure is maximised by "
          "crossing immediately and never returning, which holds for both terms "
          "as defined in Section 3.4. The maximiser reads no input, so the term "
          "it maximises can carry no information about the input. The qualifier "
          "matters and is the seed of the remedy: an earliness measure that is "
          "conditioned — restricted to correctly classified clips, evaluated at "
          "a fixed operating point of the discrimination metric, or penalised "
          "by the false-alarm rate — is not maximised by a constant."),
    ("p", "(c) Consequently the separating power of the score lies in B. To the "
          "extent that a submission crosses the threshold early and holds it, "
          "its U is at or near the maximum and it is separated from other such "
          "submissions only by B. We cannot verify this for any particular "
          "entry, having no team's curves but our own, so we state it as what "
          "the algebra implies for early-crossing submissions and not as an "
          "observation about anyone's system."),
    ("p", "Property (c) is why the failure is hard to notice from inside. A "
          "leaderboard of this shape can still order methods usefully, because "
          "B still varies. What is invisible without a control is that the "
          "number attached to each of them is dominated by a term available for "
          "free, so the reported score — the thing that goes into a table and "
          "gets compared across publications — is mostly a constant."),
    ("p", "Two further mechanisms compound this, and both are visible in our "
          "own pipeline rather than in anyone else's."),
    ("p", "Metric-enforcing post-processing. An operation that clamps the risk "
          "score above the threshold, followed by a report of stable-anticipation "
          "compliance, has measured only that the clamp executed. Stated "
          "generally: a post-hoc operation that enforces a metric's definition "
          "renders that metric vacuous. Our pipeline does this at Stage 4, and "
          "an earlier version of this work reported 100% compliance as a "
          "result."),
    ("p", "Non-causal timing gain. An operation that resamples the score curve "
          "so that frame t reports a value computed from a later frame buys "
          "time-to-accident from the future. Our Stage 5 did this at α = 1.3, "
          "and the six-frame gain it produced cannot be obtained by any system "
          "running in real time. The released configuration disables it."),
    ("p", "A third, milder mechanism is selection: reporting the mean "
          "anticipation time of the best fifty clips out of 1,417 is a "
          "statement about the selection rather than about the method. Section "
          "5.6 does this, and says so."),
]

S7 = [
    ("H", "A CORRECTED PROTOCOL"),
    ("p", "Each item below is traced to a mechanism in Section 6. The first is "
          "the one we would keep if we could keep only one."),
    ("b", "Report a video-blind control with every anticipation result. Compute "
          "the full metric on a constant above threshold, a linear ramp and a "
          "mid-clip step, and place those rows in the results table. This costs "
          "an afternoon and it is diagnostic: if the control lands within noise "
          "of the method, the metric is not measuring the method. We propose it "
          "become as reflexive as reporting a majority-class baseline in "
          "classification."),
    ("b", "Bound the earliness term, and condition it. These are two separate "
          "repairs and both are needed. Bounding — normalising per clip so that "
          "the ratio of achieved to achievable anticipation lies in the unit "
          "interval — stops the term dominating the sum and stops the metric "
          "depending on how the corpus was windowed. It does not remove the "
          "constant as maximiser, which still scores 1 and still gets it free. "
          "Conditioning is what removes the exploit: report earliness at a fixed "
          "operating point of the discrimination metric, for which mean "
          "time-to-accident at 80% recall is the established form. If only one "
          "is adopted, adopt conditioning."),
    ("b", "State the reference point of every timing metric: the annotated "
          "collision, or the end of the clip. Reporting the interval to the end "
          "of the window under the name time-to-accident inflates every result "
          "by a corpus-dependent constant, which on this corpus is about one "
          "second."),
    ("b", "Report discrimination alongside any timing metric, never timing "
          "alone, and state the ratio of positive to negative clips. Without "
          "negatives there is no false-positive rate, and any rising curve "
          "achieves perfect recall."),
    ("b", "Report metrics with post-processing disabled as well as enabled. If "
          "a number moves when a clamp is removed, the clamp was part of the "
          "measurement. If a number moves when a resampling stage is removed, "
          "check whether that stage reads a frame that has not yet arrived."),
    ("b", "Publish the score of a released control alongside the leaderboard. "
          "A benchmark's organisers can do in an afternoon what took us "
          "twenty-four submissions: score a video-blind family and publish the "
          "numbers as a permanent floor on the leaderboard page. An entrant "
          "would then see at a glance how much of their score they earned."),
    ("p", "A seventh applies specifically to competitions that withhold labels. "
          "Publish the metric weights, and publish the scorer. Withholding "
          "labels is legitimate and prevents overfitting. Withholding the "
          "weights does not prevent the metric being exploited — this paper "
          "exploited it comprehensively without them — while it does prevent "
          "the sanity check that would have surfaced the problem before the "
          "competition opened. And as Section 5.4 shows, the published formula "
          "here does not reproduce the scorer's behaviour, so a released scorer "
          "is not a courtesy but a precondition for anyone reproducing the "
          "results."),
]

S8 = [
    ("H", "LIMITATIONS"),
    ("p", "One benchmark. We demonstrate on a single competition with thirteen "
          "teams, so the ranking result is anecdotal in scale and we do not "
          "present it as a finding about a field. The decomposition, however, "
          "is a property of the metric's algebra: that a threshold-crossing "
          "earliness term is maximised by a constant, and that summing it into "
          "a bounded term lets it dominate, holds independently of how many "
          "teams entered. What the competition supplies is a third-party scorer "
          "that let us measure the magnitude in a live setting rather than "
          "argue it in the abstract."),
    ("p", "We cannot separate the weights from the metric values. Every "
          "decomposed quantity is a product of a weight and a metric. Reporting "
          "the products is sufficient for the argument, since the products "
          "determine the ranking, but we cannot state what average precision "
          "the corpus's base rate implies."),
    ("p", "One discrepancy in the crossing-frame model is open. A linear ramp "
          "scores 0.037 below three curves that cross at the same frame "
          "(Section 5.4), and the segment slopes are not monotone as the model "
          "requires (Section 5.5). We report both rather than fit around them. "
          "Neither the headline result nor the timing total depends on the "
          "resolution — both are measured between flat curves — but the split "
          "of the timing total between the two terms, and the interpretation of "
          "the fit as exact, do."),
    ("p", "We do not demonstrate the corrected protocol on a labelled corpus. "
          "Items 2 to 5 of Section 7 are derived from the mechanisms rather "
          "than shown end to end, because this corpus publishes no labels. "
          "Demonstrating them requires a benchmark with released annotations, "
          "for which DAD [4] and CCD [5] are the obvious candidates, and we "
          "state that as future work rather than claim it here."),
    ("p", "The control is a sufficient, not a necessary, condition. A metric on "
          "which a video-blind control scores poorly is not thereby sound. Our "
          "control detects this particular failure mode and we claim nothing "
          "beyond it."),
    ("p", "The caption leakage is identified and unquantified. Section 3.3 "
          "notes that the supplied captions describe outcomes and are available "
          "at inference time; we did not probe how much a method conditioned on "
          "them could gain."),
    ("p", "Our own system's placement is not evidence about the system. That "
          "the ensemble scored below the constant tells us about the metric, "
          "not about the ensemble. Under a protocol that bounded and "
          "conditioned the earliness term the comparison might go either way, "
          "and we do not know, because average precision and area under the "
          "curve are not computable on this corpus by an entrant."),
    ("p", "The venue is a community competition with an undisclosed custom "
          "metric, not a long-running benchmark with a published scorer. That "
          "is what made the probe experiment possible; it also means we cannot "
          "inspect the scoring code, and every statement here about the "
          "metric's implementation is an inference from its outputs."),
]

S9 = [
    ("p", "A composite metric that adds an unbounded earliness term to bounded "
          "discrimination terms does not measure what its name suggests. On a "
          "live accident-anticipation benchmark we measured the magnitude: the "
          "earliness terms are worth 5.70 times what the discrimination terms "
          "contribute to a submission that reads nothing, they are maximised "
          "exactly by a constant, and that constant scores within 0.14874 of "
          "the winning entry while matching the eighth-placed team at the "
          "precision the leaderboard displays."),
    ("p", "Using twenty-four video-blind submissions as probes we recovered "
          "most of the metric's functional form from outside the competition. "
          "The score falls at exactly one sixtieth of a point per frame of "
          "delay, confirmed by a one-frame experiment and predicting held-out "
          "step-like curves to within 6.4 × 10⁻³. The same probes recovered the "
          "support and mean of the withheld accident-onset distribution, and "
          "established a second result we did not set out to find: three curves "
          "crossing the threshold at the same frame score 0.091 apart, a spread "
          "the published formula cannot produce. The benchmark's stated "
          "definition does not reproduce its own scorer."),
    ("p", "The remedy is cheap. Bound the earliness term and condition it on a "
          "fixed operating point of the discrimination metric, so that a method "
          "must classify before its timing is counted. And run the control: a "
          "constant costs one submission and settles in an afternoon a question "
          "that otherwise propagates through a literature."),
    ("p", "We report our own entry, a training-free ensemble of vision–language "
          "similarity, a motion signal and a text-anchored prior, as the case "
          "study. It placed ninth of thirteen and it scores below the constant. "
          "We had selected its sentence encoder by minimising the crossover "
          "frame — the only quantity this benchmark lets an entrant compute, and "
          "the quantity a constant drives to zero. That is what a withheld "
          "label set does to a research programme, and it is the reason this "
          "paper reports a metric rather than a method."),
]

FUTURE_MARK = ("MATERIAL PARKED — the Future Scope section below is superseded "
               "by Section 8, Limitations, and is not for submission in this "
               "position. Three of its subsections describe work that remains "
               "worth doing: cross-dataset validation on a labelled corpus, "
               "edge deployment, and the driver-behaviour and regulatory "
               "discussion. Fold those into a short future-work paragraph at "
               "the end of Section 9. Note that 7.1 states the ensemble weights "
               "were grid-searched on the evaluation corpus, which contradicts "
               "the zero-shot claim and must be resolved before submission.")


def main():
    tree = etree.parse(str(DOC))
    body = tree.getroot().find(W + "body")

    def paras():
        return [c for c in body if c.tag == W + "p"]

    def find(pred):
        for p in paras():
            if pred(text_of(p)):
                return p
        return None

    # ---- templates -------------------------------------------------------
    sec_head = copy.deepcopy(find(lambda t: t.strip() == "THE BENCHMARK AND ITS METRIC"))
    sub_head = copy.deepcopy(find(lambda t: t.strip().startswith("3.1. The competition")))
    body_t = copy.deepcopy(find(lambda t: t.startswith("The measurements in this paper")))
    cap_t = copy.deepcopy(find(lambda t: t.startswith("Table 5: Implementation")))
    bullet_t = copy.deepcopy(find(lambda t: t.startswith("A measured demonstration on a live")))
    tbl_t = None
    for c in body:
        if c.tag == W + "tbl" and "Full Corpus" in "".join(x.text or "" for x in c.iter(W + "t")):
            tbl_t = c
            break
    if tbl_t is None:
        raise SystemExit("template table not found")

    TMPL = {"h": sub_head, "H": sec_head, "p": body_t, "c": cap_t, "b": bullet_t}
    TABLES = {"T_LEADERBOARD": T_LEADERBOARD, "T_CONTROLS": T_CONTROLS,
              "T_PROBES": T_PROBES, "T_DECOMP": T_DECOMP,
              "T_SWEEP": T_SWEEP, "T_DIAG": T_DIAG}

    def render(items):
        out = []
        for kind, val in items:
            if kind == "t":
                out.append(build_table(tbl_t, TABLES[val]))
            else:
                out.append(set_text(copy.deepcopy(TMPL[kind]), val))
        return out

    # ---- repair 1: move the control/probe block out of the Conclusion ----
    block_starts = ["The video-blind control family",
                    "A video-blind submission is one whose risk curve",
                    "Probes that isolate the terms of the metric",
                    "The decomposition in Section 5 rests on one observation",
                    "That turns the leaderboard into an instrument",
                    "The step family measures the score's dependence"]
    moved = []
    for s in block_starts:
        p = find(lambda t, s=s: t.startswith(s))
        if p is not None:
            body.remove(p)
            moved.append(p)
    dest = find(lambda t: t.startswith("Stage 5 — Numerical Stability Clipping"))
    at = list(body).index(dest)
    for off, p in enumerate(moved, start=1):
        body.insert(at + off, p)
    print(f"  [OK] moved {len(moved)} control/probe paragraphs into the method")

    # ---- repair 2: drop the placeholder marker --------------------------
    m = find(lambda t: t.startswith("SECTIONS 6, 7 AND 8 ARE NOT YET WRITTEN"))
    if m is not None:
        body.remove(m)
        print("  [OK] removed the placeholder marker")

    # ---- Section 5: retitle and insert the new subsections --------------
    # The Results heading was hard-typed "5." while every other section heading
    # is an auto-numbered list item. With four new sections after it the two
    # schemes collide, so Results joins the auto-numbered sequence: Related 2,
    # Benchmark 3, Method 4, Results 5, Analysis 6, Protocol 7, Limitations 8,
    # Conclusion 9.
    old_res = find(lambda t: t.strip() == "5. Results and Discussion")
    old_res.getparent().replace(old_res, set_text(copy.deepcopy(sec_head), "RESULTS"))
    h51 = find(lambda t: t.strip().startswith("5.1. Overall Anticipation Performance"))
    at = list(body).index(h51)
    for off, n in enumerate(render(S5)):
        body.insert(at + off, n)
    print(f"  [OK] Section 5: inserted {len(S5)} blocks before the proxy results")

    # the authors' 5.1 heading now sits under 5.6
    set_text(h51, "5.6.1. Overall curve-shape statistics (pre-leaderboard)")
    for old, new in [
        ("5.2. Granular Analysis of Top-50 Anticipation Cases",
         "5.6.2. The best-fifty subset (a statement about the selection)"),
        ("5.3. Modality Ablation Study",
         "5.6.3. Modality ablation, measured on the crossover proxy"),
        ("5.4. Extreme Early Anticipation: Analysis of Top-25 Predictive Trajectories",
         "5.6.4. The best twenty-five trajectories"),
        ("5.4.1. Statistical Inferences and Reaction-Time Maximisation",
         "5.6.4.1. Aggregate statistics of the best twenty-five"),
        ("5.4.2. Multi-Modal Synergistic Efficacy in Extreme Cases",
         "5.6.4.2. Modality behaviour in the best twenty-five"),
        ("5.5. Configuration Comparison: Seven-Variant Ablation",
         "5.6.5. Seven-configuration comparison, on the same proxy"),
        ("5.6. Discussion: Implications for Zero-Shot Safety Systems",
         "5.7. Discussion"),
    ]:
        p = find(lambda t, o=old: t.strip().startswith(o))
        if p is not None:
            set_text(p, new)
        else:
            print(f"  [!!] heading not found: {old[:40]}")
    print("  [OK] Section 5 subsections renumbered under 5.6")

    # The new tables in 5.1-5.4 take numbers 6 to 11, so the authors' tables in
    # 5.6 move up. Their top-25 table was labelled "Table 1", which collided
    # with nothing but was wrong; it becomes 14.
    RENUMBER = [
        ("Table 6: Overall Zero-Shot Anticipation Performance",
         "Table 12: Overall Zero-Shot Anticipation Performance"),
        ("Table 7: Modality Ablation Study",
         "Table 13: Modality Ablation Study"),
        ("Table 1: Quantitative Metrics of the Top 25 Anticipatory Trajectories",
         "Table 14: Quantitative Metrics of the Top 25 Anticipatory Trajectories"),
        ("Table 1: TTA derived from remaining frames",
         "Table 14, note: TTA derived from remaining frames"),
        ("Table 8: Seven-Configuration Performance Comparison",
         "Table 15: Seven-Configuration Performance Comparison"),
    ]
    for old, new in RENUMBER:
        p = find(lambda t, o=old: o in t)
        if p is not None:
            set_text(p, text_of(p).replace(old, new))
        else:
            print(f"  [!!] caption not found: {old[:38]}")
    print("  [OK] the authors' table numbers moved clear of the new ones")

    # ---- Sections 6, 7, 8 before the Conclusion -------------------------
    concl = find(lambda t: t.strip() == "6. Conclusion")
    at = list(body).index(concl)
    nodes = render(S6) + render(S7) + render(S8)
    for off, n in enumerate(nodes):
        body.insert(at + off, n)
    print(f"  [OK] inserted Sections 6, 7 and 8 ({len(nodes)} blocks)")

    # ---- the Conclusion ---------------------------------------------------
    set_text(concl, "CONCLUSION")
    concl.getparent().replace(concl, set_text(copy.deepcopy(sec_head), "CONCLUSION"))
    concl = find(lambda t: t.strip() == "CONCLUSION")
    old_body = []
    for s in ["This paper presented PreCrash",
              "The central technical insight motivating this design",
              "Evaluated on 1,417 clips from the MM-AU benchmark",
              "The modality ablation study and seven-configuration",
              "The strict zero-shot operational mode of PreCrash"]:
        p = find(lambda t, s=s: t.startswith(s))
        if p is not None:
            old_body.append(p)
    if old_body:
        at = list(body).index(old_body[0])
        for p in old_body:
            body.remove(p)
        for off, n in enumerate(render(S9)):
            body.insert(at + off, n)
        print(f"  [OK] Conclusion rewritten ({len(old_body)} paragraphs replaced)")

    # ---- park Future Scope ------------------------------------------------
    fs = find(lambda t: t.strip() == "7. Future Scope")
    if fs is not None:
        at = list(body).index(fs)
        body.insert(at, set_text(copy.deepcopy(body_t), FUTURE_MARK))
        print("  [OK] Future Scope parked with a marker")

    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
