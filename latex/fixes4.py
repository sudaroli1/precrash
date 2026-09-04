"""
Pass 4: the last read before upload.

Two are substantive.

**Equation (10) named the wrong operand, and the paper had the stage order
backwards.** `src/postprocess.py` runs the temporal compression as Stage 3 and
the monotone clamp as Stage 4 -- compression first, clamp applied to the
compressed curve. The paper printed the compression as Stage 5 taking
p_clamp as its argument, which is the opposite. The two orderings differ in
general, so this is not a presentational choice; the implemented one is stated
and the equation now reads p_norm.

**TTA@0.5 as printed constrained nothing.** The set was written
`max{t_ai - t_a | p_t > 0.5, 0 <= t_a <= t_ai}` -- the condition binds a free
`t` that appears nowhere else, so the maximum is attained at t_a = 0 for every
submission and the definition is vacuous. The condition belongs on p_{t_a},
which is also the only reading under which the paper's own next paragraph
("a risk score that exceeds 0.5 at frame 0 ... has t_a = 0") makes sense. The
STTA line below it binds its t correctly, so this was a transcription slip in
the one formula the whole argument rests on.

The rest: reference 23's given names, resolved from the arXiv HTML (Tarandeep
Singh, Soumyanetra Pal, Soham Biswas, Nishanth Chandran) -- it was the only
surname-only entry among thirty-two; a table cell equating a negative quantity
to a positive one; a column header that its own last row contradicts; the
notation in equation (1), which introduced a symbol used nowhere else; an
explainability claim that does not survive the correction made eight pages
earlier; and the one leaderboard coincidence the paper's own scope rule says
to state.

  python fixes4.py
"""
from __future__ import annotations

import pathlib
import sys

BODY = [
    # ---- Section 1: plural ------------------------------------------------
    ("this paper prints a number beside other teams' number.",
     "this paper prints a number beside other teams' numbers."),

    # ---- Section 3.4: the condition bound a free index --------------------
    ("\\[\\text{TTA@0.5} = \\max\\left\\{ t_{ai} - t_{a} \\ \\middle|\\ "
     "p_{t} > 0.5,\\ 0 \\leq t_{a} \\leq t_{ai} \\right\\}\\]",
     "\\[\\text{TTA@0.5} = \\max\\left\\{ t_{ai} - t_{a} \\ \\middle|\\ "
     "p_{t_{a}} > 0.5,\\ 0 \\leq t_{a} \\leq t_{ai} \\right\\}\\]"),

    # ---- Section 4.1: a dangling noun phrase and a missing stop -----------
    ("The anticipation problem is then a predictive mapping that assigns a "
     "risk score to every frame of the clip. The learning predictive mapping "
     "is formally denoted as mentioned in equation (1)",
     "The anticipation problem is then a predictive mapping that assigns a "
     "risk score to every frame of the clip, as in equation (1)."),

    # equation (1) introduced a symbol used nowhere else in the paper
    ("\\[g\\ :\\ V_{s} \\longrightarrow \\left( \\rho_{1},\\ldots,\\rho_{m} "
     "\\right) \\in \\lbrack 0,1\\rbrack^{m}\\ldots\\ldots\\ldots.(1)\\]",
     "\\[g\\ :\\ V_{s} \\longrightarrow \\left( p_{1},\\ldots,p_{m} "
     "\\right) \\in \\lbrack 0,1\\rbrack^{m}\\ldots\\ldots\\ldots.(1)\\]"),

    ("Here \\(\\rho_{t}\\) is the predicted probability of an accident at "
     "frame \\(t\\), computed without observing one.",
     "Here \\(p_{t}\\) is the predicted probability of an accident at frame "
     "\\(t\\), computed without observing one; it is the same quantity the "
     "metric of Section 3.4 reads, and the quantity every later equation "
     "post-processes."),

    ("This problem statement is specifically difficult to solve for the "
     "following reasons:",
     "This problem is specifically difficult for the following reasons:"),

    # ---- Section 4.2: the engines are sub-items of Parallel Processing -----
    # pandoc gave each list item its own enumerate, so the three engines
    # restarted at 1 at the same margin as the stages around them and
    # "4. Ensembled Model" read as a fourth engine.
    ("\\def\\labelenumi{\\arabic{enumi}.}\n\\item\n  \\emph{CLIP ViT-L/14:}",
     "\\def\\labelenumi{3.\\arabic{enumi}}\n\\item\n  \\emph{CLIP ViT-L/14:}"),
    ("\\def\\labelenumi{\\arabic{enumi}.}\n\\setcounter{enumi}{1}\n\\item\n"
     "  \\emph{Optical Flow:}",
     "\\def\\labelenumi{3.\\arabic{enumi}}\n\\setcounter{enumi}{1}\n\\item\n"
     "  \\emph{Optical Flow:}"),
    ("\\def\\labelenumi{\\arabic{enumi}.}\n\\setcounter{enumi}{2}\n\\item\n"
     "  \\emph{Text-Anchored Temporal Prior:}",
     "\\def\\labelenumi{3.\\arabic{enumi}}\n\\setcounter{enumi}{2}\n\\item\n"
     "  \\emph{Text-Anchored Temporal Prior:}"),

    # ---- Section 4.2, stage 5: the code compresses before it clamps -------
    ("\\[\\begin{matrix}\n & p_{\\text{final}}(t) = \\text{interp}\\left( "
     "p_{\\text{clamp}},\\ \\alpha t \\right),\\ \\alpha = 1.3 & "
     "\\ldots\\ldots\\ldots\\ldots(10) & \n\\end{matrix}\n\\]",
     "\\[\\begin{matrix}\n & p_{\\text{final}}(t) = \\text{interp}\\left( "
     "p_{\\text{norm}},\\ \\alpha t \\right),\\ \\alpha = 1.3 & "
     "\\ldots\\ldots\\ldots\\ldots(10) & \n\\end{matrix}\n\\]"),

    ("Equation (10) above states the operation as implemented. An earlier "
     "version of this text printed it as t divided by alpha, which matches "
     "neither the code nor the direction of the shift the stage was reported "
     "to produce.",
     "Equation (10) above states the operation as implemented. An earlier "
     "version of this text printed it as t divided by alpha, which matches "
     "neither the code nor the direction of the shift the stage was reported "
     "to produce. Two further corrections belong with it. The released code "
     "applies this stage before the monotone clamp of Equation (9) rather "
     "than after it, so the clamp acts on the compressed curve and not the "
     "reverse; the two orderings do not agree in general, and the "
     "implemented one is the one stated here. Equation (10) is written on "
     "p\\textsubscript{norm} accordingly, and the stages are numbered in the "
     "order this section presents them rather than the order the code runs "
     "them."),

    # ---- Section 5.1: the coincidence the paper's own scope rule covers ---
    ("The benchmark row displayed on the leaderboard scores 0.58333, so those "
     "two are not the same file, and the sample submission is not the origin "
     "of the published benchmark score.",
     "The benchmark row displayed on the leaderboard scores 0.58333, so those "
     "two are not the same file, and the sample submission is not the origin "
     "of the published benchmark score. It also coincides, to the five "
     "decimals the leaderboard displays, with the twelfth-placed entry in "
     "Table 3. We state the arithmetic and draw the same single inference as "
     "in Section 1 --- the metric assigns the two submissions the same value "
     "--- and nothing about any team's method."),

    # ---- Table 8: the header is false for its own last row ---------------
    ("\\textbf{Curve, all crossing 0.5 at frame 75}", "\\textbf{Curve}"),

    ("step at frame 76 (one-frame calibration) & 1.08779 & −0.01666 = 1/60.0 "
     "\\\\",
     "step at frame 76 (one-frame calibration) & 1.08779 & −0.01666, i.e.\\ "
     "one sixtieth below \\\\"),

    # ---- Table 11's note was set in body italic, unlike every caption -----
    ("\\emph{Table 11, note: the seconds column is the interval from the "
     "crossover frame to the end of the clip at 30 FPS. It is not a "
     "time-to-accident: this corpus annotates no collision frame "
     "(Section 3.3), and the figure is inflated by the interval Section 5.5 "
     "estimates at about a second. Initial risk vectors are read after the "
     "power transform and the Gaussian filter.}",
     "\\floatcap{Table 11, note}{the seconds column is the interval from the "
     "crossover frame to the end of the clip at 30 FPS. It is not a "
     "time-to-accident: this corpus annotates no collision frame "
     "(Section 3.3), and the figure is inflated by the interval Section 5.5 "
     "estimates at about a second. Initial risk vectors are read after the "
     "power transform and the Gaussian filter.}"),

    # ---- Section 5.7: the claim does not survive the Section 4.2 fix ------
    ("Because the three engines share no weights and are combined only at the "
     "fusion layer, any anticipation can be attributed to its constituent "
     "signals, and a failure can be localised to one of them. That is a "
     "property of the architecture, not a measured result, and we state it as "
     "such.",
     "Because the three engines share no weights and are combined only at the "
     "fusion layer, any score can be attributed to its constituent signals. "
     "Localising a failure is weaker than we first claimed: two of the three "
     "curves are functions of the same frame-difference statistic "
     "(Section 4.2), so a fault in that statistic appears in two of them at "
     "once and the decomposition separates two signals rather than three. "
     "This is in any case a property of the architecture, not a measured "
     "result, and we state it as such."),
]

BIB = [
    ("  author  = {Singh and Pal and Biswas and Chandran},",
     "  author  = {Tarandeep Singh and Soumyanetra Pal and Soham Biswas and\n"
     "             Nishanth Chandran},"),
    ("% NOTE ON THE ENTRY BELOW: the surnames are confirmed, the given names "
     "are\n"
     "% not. Verify them against the arXiv abstract page before the "
     "camera-ready.\n"
     "% Do not record that as a `note' field inside the entry -- it prints in "
     "the\n"
     "% typeset reference list.\n",
     "% Given names confirmed against the arXiv HTML rendering of "
     "2606.12047v1.\n"),
]


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
    bad = apply("body_clean.tex", BODY) + apply("references.bib", BIB)
    if bad:
        print(f"\n{bad} replacement(s) did not apply cleanly.")
        sys.exit(1)
    print(f"\n{len(BODY) + len(BIB)} corrections applied")


if __name__ == "__main__":
    main()
