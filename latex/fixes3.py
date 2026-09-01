"""
Pass 3: what the final verification found.

Mostly places the previous two passes reached the statement but not its
neighbours -- the abstract still carried the Table 8 claim that had been
corrected in three other places, Table 2 still named fusion weights alpha and
gamma after equation (6) had been rewritten to w_clip and w_prior, Section 4.3
still called the three engines independent after Section 4.2 had withdrawn it,
and the subsection heading over the third engine still read "Caption NLP
Prior".

Four are more than tidying.

**The metric definition was not typeset.** The three formulae the whole paper
is about -- the score, TTA@0.5 and STTA@0.5 -- sat in Section 3.4 as running
text with escaped underscores, so they printed as `w_AP · AP + ...` in body
roman while every other formula in the paper was set as mathematics. They are
display equations now.

**Two live cross-references pointed at limitations Section 8 does not state.**
Sections 4.3 and 5.7 both say the searched fusion weights are recorded as a
limitation in Section 8. They were not. Rather than delete the pointers, the
limitation is added, along with the single-corpus deployability point that 5.7
also promises.

**Section 4.4 counted its own family wrong.** The list enumerates twenty-one
curves and then says twenty-five. Twenty-five files were submitted and
twenty-one distinct shapes among them; four were sent twice, and the repeats
agree to 3 x 10^-5, which is worth stating because it is a check on the
scoring service rather than an accident of bookkeeping.

**Table 1's column is described as the ensemble's crossover.** Its selected
value, 21.7, is exactly the prior-alone figure in Tables 10 and 12, where the
ensemble reads 22.7. We cannot now recover which configuration produced the
column, and say so.

  python fixes3.py
"""
from __future__ import annotations

import pathlib
import sys

ABSTRACT = [
    # Table 8's caption now says only the three step-like curves hold
    # identical values at 74 and 75; the ramp does not.
    ("four curves that cross the threshold at the same frame, with identical "
     "values at the crossing, score 0.091 apart, a spread the published "
     "metric definition cannot produce",
     "four curves that cross the threshold at the same frame, and never fall "
     "below it afterwards, score 0.091 apart, a spread the published metric "
     "definition cannot produce"),

    # Section 5.5 says in terms that the frame-140 result bounds a mean and
    # not a support.
    ("The same probes recover the support and mean of the withheld "
     "accident-onset distribution,",
     "The same probes place the withheld accident-onset distribution, "
     "recovering a mean of 120.1 frames in a 150-frame clip and bounding what "
     "can lie beyond frame 140,"),
]

BODY = [
    # ---- Section 3.4: the metric, as mathematics ----------------------
    ("score = w\\_AP · AP + w\\_AUC · AUC + w\\_TTA · TTA@0.5 + w\\_STTA · "
     "STTA@0.5",
     "\\[\\text{score} = w_{\\text{AP}}\\,\\text{AP} + w_{\\text{AUC}}\\,"
     "\\text{AUC} + w_{\\text{TTA}}\\,\\text{TTA@0.5} + w_{\\text{STTA}}\\,"
     "\\text{STTA@0.5}\\]"),

    ("TTA@0.5 = max \\{ t\\_ai − t\\_a \\textbar{} p\\_t \\textgreater{} 0.5, "
     "0 ≤ t\\_a ≤ t\\_ai \\}",
     "\\[\\text{TTA@0.5} = \\max\\left\\{ t_{ai} - t_{a} \\ \\middle|\\ "
     "p_{t} > 0.5,\\ 0 \\leq t_{a} \\leq t_{ai} \\right\\}\\]"),

    ("STTA@0.5 = max \\{ t\\_ai − t\\_a′ \\textbar{} p\\_t \\textgreater{} "
     "0.5 for all t in [t\\_a′, t\\_ai] \\}",
     "\\[\\text{STTA@0.5} = \\max\\left\\{ t_{ai} - t'_{a} \\ \\middle|\\ "
     "p_{t} > 0.5 \\ \\text{ for all } t \\in [t'_{a},\\,t_{ai}] "
     "\\right\\}\\]"),

    ("where t\\_ai is the accident start frame within the 150-frame clip and "
     "t\\_a is the first frame at which the risk score exceeds the threshold. "
     "STTA additionally requires the score to remain above the threshold "
     "continuously from t\\_a′ to t\\_ai.",
     "where \\(t_{ai}\\) is the accident start frame within the 150-frame "
     "clip and \\(t_{a}\\) the first frame at which the risk score exceeds "
     "the threshold. STTA additionally requires the score to remain above the "
     "threshold continuously from \\(t'_{a}\\) to \\(t_{ai}\\)."),

    # ---- the heading over the third engine ----------------------------
    ("  \\emph{Caption NLP Prior:}", "  \\emph{Text-Anchored Temporal Prior:}"),

    # ---- Table 1: whose crossover frame is the column? ----------------
    ("The column on which those four encoders were compared is the average "
     "crossover frame: the mean index at which the ensemble's risk score "
     "first exceeds 0.5.",
     "The column on which those four encoders were compared is the average "
     "crossover frame: the mean index at which a risk score first exceeds "
     "0.5. Which configuration produced it we can no longer establish from "
     "our records --- the selected value, 21.7, is exactly the prior-alone "
     "figure of Tables 10 and 12, where the full ensemble reads 22.7, so the "
     "column is most likely the prior on its own rather than the ensemble."),

    ("all-mpnet-base-v2 & 23.0 frames & Worse --- larger model, lower "
     "performance \\\\",
     "all-mpnet-base-v2 & 23.0 frames & Later crossover than the two above "
     "it \\\\"),

    # ---- Section 4.3: the independence claim, again -------------------
    ("The ensemble architecture integrates three independently operating, "
     "pre-trained modality engines with no shared weights or activation "
     "pathways between them, thereby preserving the modular interpretability "
     "of each cognitive signal.",
     "The three pretrained engines run as separate processes with no shared "
     "weights or activation pathways, though two of the three read the same "
     "frame-difference statistic (Section 4.2)."),

    # ---- Table 2: alpha and gamma were double-booked -------------------
    ("Fusion Weights & α\\,=\\,0.55 (CLIP), β\\,=\\,0.25 (Flow), "
     "γ\\,=\\,0.20 (NLP) \\\\",
     "Fusion Weights & \\(w_{\\text{clip}}\\,{=}\\,0.55\\), "
     "\\(w_{\\text{flow}}\\,{=}\\,0.25\\), \\(w_{\\text{prior}}\\,{=}\\,"
     "0.20\\) \\\\"),

    # ---- Section 4.4: twenty-one shapes in twenty-five files ----------
    ("and the two graded curves of Section 5.4. Twenty-five in all. Each is a "
     "single list of 150 numbers, repeated across all 1,417 clips.",
     "and the two graded curves of Section 5.4. That is twenty-one distinct "
     "shapes, submitted as twenty-five files: four were sent twice, once in "
     "the control family and again in the probe or diagnostic family, and the "
     "repeats agree to within 3 × 10\\textsuperscript{-5}, which is a check on "
     "the scoring service and not only on our bookkeeping. Each is a single "
     "list of 150 numbers, repeated across all 1,417 clips."),

    # ---- equation (2) was the one left in italic letter-products ------
    ("\\[flow(t) = mean\\left( \\left| frame(t) - frame(t - 1) \\right| "
     "\\right)\\ldots\\ldots\\ldots..(2)\\]",
     "\\[d(t) = \\operatorname{mean}\\left( \\left| f_{t} - f_{t-1} \\right| "
     "\\right)\\ldots\\ldots\\ldots..(2)\\]"),

    # ---- equation (10): print what was implemented --------------------
    (" & p_{\\text{final}}(t) = \\text{interp}\\left( p_{\\text{clamp}},\\ "
     "\\frac{t}{\\alpha} \\right),\\ \\alpha = 1.3 & "
     "\\ldots\\ldots\\ldots\\ldots(10) & ",
     " & p_{\\text{final}}(t) = \\text{interp}\\left( p_{\\text{clamp}},\\ "
     "\\alpha t \\right),\\ \\alpha = 1.3 & "
     "\\ldots\\ldots\\ldots\\ldots(10) & "),

    ("Equation (10) as printed reads t divided by alpha; the implementation, "
     "and the direction of the shift it was reported to produce, both "
     "correspond to alpha times t, and the equation should be corrected "
     "accordingly.",
     "Equation (10) above states the operation as implemented. An earlier "
     "version of this text printed it as t divided by alpha, which matches "
     "neither the code nor the direction of the shift the stage was reported "
     "to produce."),

    # ---- Section 5.7: 4.24 is the full corpus, not a subset -----------
    ("It is the interval from the crossover frame to the end of the clip, on "
     "a corpus that annotates no collision, averaged over a selected subset; "
     "Section 5.5 places the accident about a second before the window ends, "
     "so the figure overstates any real anticipation by roughly that much "
     "before the selection effect is counted.",
     "It is the interval from the crossover frame to the end of the clip, on "
     "a corpus that annotates no collision; Section 5.5 places the accident "
     "about a second before the window ends, so the figure overstates any "
     "real anticipation by roughly that much. The 4.30-second figure beside "
     "it in Table 9 is worse still, being the same quantity over the fifty "
     "clips selected for having crossed earliest."),

    # ---- Table 11's caption ------------------------------------------
    ("\\floatcap{Table 11}{Quantitative Metrics of the Top 25 Anticipatory "
     "Trajectories}",
     "\\floatcap{Table 11}{The twenty-five clips whose curves cross 0.5 "
     "earliest. A selection by the quantity reported, not a ranking of "
     "anticipation: read with the note below.}"),

    # ---- Section 8: the limitations 4.3 and 5.7 promise ---------------
    ("We do not demonstrate the corrected protocol on a labelled corpus.",
     "Two admissions about our own system belong here, because Sections 4.3 "
     "and 5.7 point at them. The fusion weights of 0.55, 0.25 and 0.20 were "
     "chosen by searching over the evaluation corpus. That is tuning on the "
     "test set, and it sits badly with a zero-shot claim; the released "
     "configuration therefore weights the three engines equally, chosen a "
     "priori, and the leaderboard score we report is the entry that used the "
     "searched ones. And the claim that a training-free system can be pointed "
     "at a new road environment without retraining is untested here, because "
     "we evaluated on one corpus and never on a second.\n\nWe do not "
     "demonstrate the corrected protocol on a labelled corpus."),
]

# Section 7's protocol is referred to by item number twice, so it is numbered.
ITEMIZE = (
    "\\begin{itemize}\n\\item\n  Report a video-blind control with every "
    "anticipation result.",
    "\\begin{enumerate}\n\\item\n  Report a video-blind control with every "
    "anticipation result.")


def apply(path: str, pairs) -> int:
    p = pathlib.Path(path)
    s = p.read_text()
    bad = 0
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            print(f"  [!! {n}] {path}: {old[:70]!r}")
            bad += 1
            continue
        s = s.replace(old, new)
    p.write_text(s)
    return bad


def main() -> None:
    bad = apply("body_clean.tex", BODY) + apply("abstract.tex", ABSTRACT)

    # number the protocol list, and close the environment it opened
    p = pathlib.Path("body_clean.tex")
    s = p.read_text()
    i = s.find(ITEMIZE[0])
    if i < 0:
        print("  [!!] protocol list not found")
        bad += 1
    else:
        j = s.find("\\end{itemize}", i)
        s = (s[:i] + ITEMIZE[1] + s[i + len(ITEMIZE[0]):j]
             + "\\end{enumerate}" + s[j + len("\\end{itemize}"):])
        p.write_text(s)
        print("  [OK] protocol list numbered, so \"item 5\" resolves")

    if bad:
        print(f"\n{bad} replacement(s) did not apply cleanly.")
        sys.exit(1)
    print(f"\n{len(BODY) + len(ABSTRACT) + 1} corrections applied")


if __name__ == "__main__":
    main()
