"""
Corrections found by the pre-submission audit of the compiled PDF.

Three kinds of defect, all of them real:

Conversion damage. Pandoc split several of Word's inline equations away from
the sentences that governed them, so four sentences in Section 4 began with a
string of orphaned symbols and then continued mid-clause -- "gamma > 1 gamma
gamma in [1.0,2.5] Since , this operation is..." was on the page. Equation (4)
had "min" indexed by \\propto beside a "max" indexed by \\alpha; equations (3)
and (5) each opened two parentheses and closed one; equation (1) mapped a
symbol the paper never defines.

Arithmetic. Two rows of the ablation table stated a difference against the
ensemble that does not follow from the two numbers beside it; the mean peak
risk quoted in the prose is not the mean of the column it summarises; two
figures were rounded the wrong way.

Statements that survived an earlier correction. Table 11's note still says the
seconds were measured to a collision, which Sections 5.6 and 5.7 spend two
paragraphs withdrawing. Table 8's caption counts five curves crossing at frame
75 when four do, and the prose then reads a span off four of them while calling
them three -- the number itself, 0.09093, was right, and so is the abstract's
0.091, but the description around it was not.

  python fixes.py
"""
from __future__ import annotations

import pathlib
import sys

BODY = [
    # ---- conversion damage: orphaned equation fragments ----------------
    (r"\(V_{s =}\{ f_{1},f_{2}\ldots\ldots\ldots f_{m}\)\}",
     r"\(V_{s} = \{ f_{1},f_{2},\ldots,f_{m}\}\)"),

    (r"f_{t} \in \mathbb{R}^{H \times w \times 3}",
     r"f_{t} \in \mathbb{R}^{H \times W \times 3}"),

    (r"\[g\ :\ \mathcal{W}_{t} \longrightarrow \rho_{t} \in "
     r"\lbrack 0,1\rbrack\ldots\ldots\ldots.(1)\]",
     r"\[g\ :\ V_{s} \longrightarrow \rho_{t} \in "
     r"\lbrack 0,1\rbrack\ldots\ldots\ldots.(1)\]"),

    (r"\(\rho_{t}\)Where ρ(t) is the predicted probability of an accident, "
     r"computed without observing one.",
     r"Here \(\rho_{t}\) is the predicted probability of an accident at frame "
     r"\(t\), computed without observing one."),

    (r"\(\gamma > 1\gamma\gamma \in \lbrack 1.0,2.5\rbrack\)Since , this "
     r"operation is a monotone contrast stretch:",
     r"Since \(\gamma > 1\), with \(\gamma\) searched over "
     r"\(\lbrack 1.0,2.5\rbrack\), this operation is a monotone contrast "
     r"stretch:"),

    (r"\(\epsilon = 0.01\)with . This enforces a hard causal constraint:",
     r"with \(\epsilon = 0.01\). This enforces a hard causal constraint:"),

    (r"\(interpt \approx 28t \approx 22\alpha = 1.3\left( 0.001,\text{\,}"
     r"0.999 \right)\)where denotes linear interpolation onto the original "
     r"frame grid.",
     r"where \(\text{interp}\) denotes linear interpolation onto the original "
     r"frame grid."),

    ("Finally, the output is clipped to the open interval to ensure numerical "
     "stability in downstream log-loss evaluation.",
     "Finally, the output is clipped to the open interval "
     "\\((0.001,\\ 0.999)\\) to ensure numerical stability in downstream "
     "log-loss evaluation."),

    # ---- conversion damage: malformed equations ------------------------
    (r"\[{p\_}_{clip(t)} = \sigma((\cos\left( \varphi\left( e_{t} \right),"
     r"f_{danger} \right) - cos(\varphi\left( e_{t} \right),f_{safe})"
     r"\ldots\ldots\ldots\ldots\ldots\ldots.(3)\]",
     r"\[{p\_}_{clip(t)} = \sigma\left( \cos\left( \varphi\left( e_{t} "
     r"\right),f_{\text{danger}} \right) - \cos\left( \varphi\left( e_{t} "
     r"\right),f_{\text{safe}} \right) \right)"
     r"\ldots\ldots\ldots\ldots\ldots\ldots.(3)\]"),

    (r"\min_{\propto}{|\left| e_{\alpha} - e_{\alpha - 1} \right||}_{1}}"
     r"{\max_{\alpha}{||e_{\alpha} - e_{\alpha - 1}}|| - "
     r"\min_{\propto}{|\left| e_{\alpha} - e_{\alpha - 1} \right||}_{1}}",
     r"\min_{\alpha}{|\left| e_{\alpha} - e_{\alpha - 1} \right||}_{1}}"
     r"{\max_{\alpha}{||e_{\alpha} - e_{\alpha - 1}}|| - "
     r"\min_{\alpha}{|\left| e_{\alpha} - e_{\alpha - 1} \right||}_{1}}"),

    (r"\[{p\_}_{caption(t)} = \sigma(\left( \rho_{1}D_{sudden}(t) + "
     r"\rho_{2}D_{gradual}(t) - t_{0} \right)\ldots\ldots\ldots\ldots..(5)\]",
     r"\[{p\_}_{caption(t)} = \sigma\left( \rho_{1}D_{\text{sudden}}(t) + "
     r"\rho_{2}D_{\text{gradual}}(t) - t_{0} "
     r"\right)\ldots\ldots\ldots\ldots..(5)\]"),

    # ---- the engine list: one item had become a lone subsubsection -----
    (r"\subsubsection{\emph{Caption NLP Prior:}}",
     "\\begin{enumerate}\n"
     "\\def\\labelenumi{\\arabic{enumi}.}\n"
     "\\setcounter{enumi}{2}\n"
     "\\item\n"
     "  \\emph{Caption NLP Prior:}\n"
     "\\end{enumerate}"),

    ("\\item\n  CLIP ViT-L/14:\n", "\\item\n  \\emph{CLIP ViT-L/14:}\n"),

    # ...which pushes the two stages after it down by one
    ("\\setcounter{enumi}{2}\n\\item\n  \\emph{Ensembled Model:}",
     "\\setcounter{enumi}{3}\n\\item\n  \\emph{Ensembled Model:}"),
    ("\\setcounter{enumi}{3}\n\\item\n  \\emph{Pipeline Overview and "
     "Ensemble calculation:}",
     "\\setcounter{enumi}{4}\n\\item\n  \\emph{Pipeline Overview and "
     "Ensemble calculation:}"),

    # ---- an item that stated the opposite of the difficulty it lists ---
    ("1.~~~~ ~As there are no training data, it becomes impossible to "
     "fine-tune the model.",
     "1.~~~~ As there are no training data, it becomes impossible to "
     "fine-tune the model."),

    ("3.~~~~ The considered dataset is diverse in terms of accident "
     "categories, and without any training instances, it might generalize "
     "well.",
     "3.~~~~ Pretrained knowledge is the only knowledge available, and with "
     "no training instances there is no way to establish in advance that it "
     "transfers to any of these accident categories."),

    # ---- the control family: the ramp's endpoints ----------------------
    ("a linear ramp from 0.001 to 0.999; logistic curves centred",
     "a linear ramp from 0.000 to 1.000; logistic curves centred"),

    # ---- the dip probe is in Table 5, not Table 4 ----------------------
    ("Two further readings follow from Table 4. A constant at 0.51",
     "Two further readings follow from Tables 4 and 5. A constant at 0.51"),

    # ---- Table 6: 0.87593 / 0.35105 = 2.4952 --------------------------
    ("TTA@0.5, saturated & 0.87593 & 2.49× \\\\",
     "TTA@0.5, saturated & 0.87593 & 2.50× \\\\"),

    # ---- Table 8: four curves cross at 75, not five -------------------
    ("\\floatcap{Table 8}{Five curves that all first exceed 0.5 at frame 75,",
     "\\floatcap{Table 8}{Four curves that all first exceed 0.5 at frame 75,"),

    ("The three curves in the middle of Table 8 have identical values at "
     "frames 74 and 75,",
     "The four curves at the head of Table 8 have identical values at "
     "frames 74 and 75,"),

    # ---- Table 9: (150 - 22.7) / 30 = 4.24 ----------------------------
    ("Mean TTA (seconds) & 4.23 s & 4.30 s \\\\",
     "Mean TTA (seconds) & 4.24 s & 4.30 s \\\\"),

    # ---- p_max, split across three scripts by the conversion ----------
    ("p\\textsubscript{m}\\textsubscript{a}\\textsuperscript{x}",
     "p\\textsubscript{max}"),

    # ---- Table 10: two rows whose deltas do not follow ----------------
    ("CLIP Only & 26--28 & +4.3 to +5.3 &",
     "CLIP Only & 26--28 & +3.3 to +5.3 &"),
    ("NLP Prior Only & \\textasciitilde21.7 & +0.0 to −0.7 &",
     "NLP Prior Only & \\textasciitilde21.7 & −1.0 &"),

    # ---- the mean of Table 11's peak-risk column is 0.98584 -----------
    ("mean peak risk reached 0.9871", "mean peak risk reached 0.9858"),

    # ---- Table 11's note contradicted the correction above it ---------
    ("\\emph{Table 11, note: TTA derived from remaining frames prior to "
     "collision at 30 FPS. Initial risk vectors verified post power-curve "
     "smoothing and Gaussian filtering.}",
     "\\emph{Table 11, note: the seconds column is the interval from the "
     "crossover frame to the end of the clip at 30 FPS. It is not a "
     "time-to-accident: this corpus annotates no collision frame "
     "(Section 3.3), and the figure is inflated by the interval Section 5.5 "
     "estimates at about a second. Initial risk vectors are read after the "
     "power transform and the Gaussian filter.}"),
]

ABSTRACT = [
    ("the two-timing terms contribute 2.00186",
     "the two timing terms contribute 2.00186"),
    ("three curves that cross the threshold at the same frame, with identical "
     "values at the crossing, score 0.091 apart",
     "four curves that cross the threshold at the same frame, with identical "
     "values at the crossing, score 0.091 apart"),
]

BIB = [
    ("  note    = {Given names to be confirmed before camera-ready}\n", ""),
    ("  year    = {2026},\n  \n", "  year    = {2026}\n"),
]


def apply(path: str, pairs) -> int:
    p = pathlib.Path(path)
    s = p.read_text()
    bad = 0
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            print(f"  [!! {n}] {path}: {old[:64]!r}")
            bad += 1
            continue
        s = s.replace(old, new)
    p.write_text(s)
    return bad


def main() -> None:
    bad = 0
    bad += apply("body_clean.tex", BODY)
    bad += apply("abstract.tex", ABSTRACT)

    # the .bib note prints in the reference list, so it becomes a comment
    p = pathlib.Path("references.bib")
    s = p.read_text()
    old = "  note    = {Given names to be confirmed before camera-ready}\n"
    if old in s:
        s = s.replace(
            "  year    = {2026},\n" + old,
            "  year    = {2026}\n"
            "  % NOTE: given names unconfirmed. Verify against the arXiv\n"
            "  % abstract page before the camera-ready. A `note' field here\n"
            "  % prints in the typeset reference list.\n")
        p.write_text(s)
        print("  [OK] references.bib: build note moved out of the printed entry")
    else:
        print("  [!!] references.bib: note field not found")
        bad += 1

    if bad:
        print(f"\n{bad} replacement(s) did not apply cleanly -- check above.")
        sys.exit(1)
    print("\nall corrections applied")


if __name__ == "__main__":
    main()
