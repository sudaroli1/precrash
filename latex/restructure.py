"""
Restructure the paper for a journal submission: put the finding first and the
case study in an appendix.

TMLR desk-rejected the paper without review, citing both "unlikely to meet one
or both of TMLR criterion" and reviewer bandwidth. The likely failing criterion
is audience interest, and two structural facts worked against it under triage:
twenty-eight pages for a measurement result, and roughly a third of the length
spent on the authors' own ensemble -- a system the paper itself says is not the
contribution.

So the ensemble moves out of the argument and into appendices, where a reader
can check it without having to read past it.

  Body      1 Introduction
            2 Related work and evaluation practice
            3 The benchmark and its metric
            4 Method: the video-blind controls and the probes   (was 4.4, 4.5)
            5 Results                                           (was 5.1-5.5)
            6 Analysis: why the composition fails
            7 A corrected protocol
            8 Limitations
            9 Conclusion
              Compliance and data availability
  Appendix A The system under study                             (was 4.1-4.3)
  Appendix B What we measured before the leaderboard            (was 5.6, 5.7)

Nothing is deleted. Every correction, every withdrawal and every table survives
the move; what changes is where a reader meets them.

Renumbering is the risky part, because every float and cross-reference in this
paper is hand-typed rather than a LaTeX label. Tables 3-8 become 1-6, Tables
1-2 become A1-A2, Tables 9-12 become B1-B4, Figure 2 becomes Figure 1 and
Figure 1 becomes Figure A1 -- so a naive sequential rename would clobber
itself. Every rename therefore goes through a placeholder pass first.

  python restructure.py
"""
from __future__ import annotations

import pathlib
import re
import sys

SRC = pathlib.Path("body_clean.tex")

# 1-indexed, inclusive line ranges, from the section map
SEG = {
    "intro":      (1, 31),
    "related":    (32, 82),
    "bench":      (83, 123),
    "s4head":     (124, 127),
    "a41":        (128, 149),
    "a42":        (150, 335),
    "a43":        (336, 365),
    "controls":   (366, 369),
    "probes":     (370, 377),
    "s5head":     (378, 379),
    "s51":        (380, 447),
    "s52":        (448, 500),
    "s53":        (501, 541),
    "s54":        (542, 574),
    "s55":        (575, 585),
    "b56":        (586, 740),
    "b57":        (741, 749),
    "s6":         (750, 771),
    "s7":         (772, 792),
    "s8":         (793, 812),
    "s9":         (813, 820),
    "compliance": (821, None),
}

NEW_S4 = r"""\section{Method: the Video-Blind Controls and the Probes}

Every measurement in this paper is made with a submission that reads no video.
This section defines the two families we used: a control family, which
establishes what a submission earns without looking at anything, and a probe
family, designed so that the terms we cannot compute cancel when two probes are
differenced. The system we entered in the competition is the case study rather
than the contribution, and is described in Appendix A; nothing in Sections 5
to 7 depends on it."""

APP_A = r"""\appendix

\section{The System under Study}

This appendix describes the training-free ensemble we entered in the
competition. It is the instrument that produced the finding and the case study
that illustrates it, and it is reported here in full because Section 6
identifies two of its stages as mechanisms by which a system can inflate a
timing metric without anticipating anything. It is not the contribution of this
paper, and no result in Sections 5 to 7 depends on any of it."""

APP_B = r"""\section{What We Measured Before the Leaderboard}

This appendix keeps the evaluation we ran before any submission was scored,
together with the reckoning that followed. It is retained because it is the
case study Section 6 analyses: an honest record of what a team measures when
the corpus publishes no labels, and of what that measurement turned out to be
worth."""

# ---------------------------------------------------------------- renaming
# Tables 3-8 -> 1-6 (body), 1-2 -> A1-A2, 9-12 -> B1-B4 (appendices).
TABLE = {1: "A1", 2: "A2", 3: "1", 4: "2", 5: "3", 6: "4",
         7: "5", 8: "6", 9: "B1", 10: "B2", 11: "B3", 12: "B4"}
FIGURE = {1: "A1", 2: "1"}

# Bare "Section 4" means the ensemble in some places and the probe argument in
# others, so those are replaced on their surrounding phrase rather than by
# pattern.
PHRASES = [
    ("Section 4 states which of the two the system examined here uses",
     "Appendix A states which of the two the system examined here uses"),
    ("which is what motivated the system described in Section 4",
     "which is what motivated the system described in Appendix A"),
    ("The ensemble described in Section 4 is the instrument",
     "The ensemble described in Appendix A is the instrument"),
    ("Section 4 records that we did exactly this",
     "Appendix A records that we did exactly this"),
    ("the monotone clamp of Section 4", "the monotone clamp of Appendix A"),
    ("which Section 4.3 records as a limitation",
     "which Appendix A records as a limitation"),
    # the two that genuinely still point at Section 4
    ("cancel in any difference between two of them (Section 4)",
     "cancel in any difference between two of them (Section 4)"),
    ("weaken the cancellation argument of Section 4",
     "weaken the cancellation argument of Section 4"),
]

# Numbered subsection references, placeholder-mapped so 4.5->4.2 does not
# collide with 4.2->Appendix A.
SECREF = {
    "Section 4.5": "Section 4.2",
    "Section 4.4": "Section 4.1",
    "Section 4.3": "Appendix A",
    "Section 4.2": "Appendix A",
    "Section 4.1": "Appendix A",
    "Section 5.6.1": "Appendix B",
    "Section 5.7": "Appendix B",
    "Section 5.6": "Appendix B",
}


def seg(lines, name):
    a, b = SEG[name]
    return "\n".join(lines[a - 1:(b if b else len(lines))])


def rename_floats(s: str) -> str:
    """Two-pass so the new numbers cannot be re-matched by a later rule."""
    s = re.sub(r"\b(Tables?)\s+(\d{1,2})\s+and\s+(\d{1,2})\b",
               lambda m: f"{m.group(1)} @@T{m.group(2)}@@ and @@T{m.group(3)}@@",
               s)
    s = re.sub(r"\b(Tables?)\s+(\d{1,2})\b",
               lambda m: f"{m.group(1)} @@T{m.group(2)}@@", s)
    s = re.sub(r"\b(Figures?)\s+(\d{1,2})\b",
               lambda m: f"{m.group(1)} @@F{m.group(2)}@@", s)
    s = re.sub(r"@@T(\d{1,2})@@", lambda m: TABLE[int(m.group(1))], s)
    s = re.sub(r"@@F(\d{1,2})@@", lambda m: FIGURE[int(m.group(1))], s)
    return s


def rename_sections(s: str) -> str:
    for old, new in PHRASES:
        s = s.replace(old, new)
    for i, key in enumerate(SECREF):
        s = s.replace(key, f"@@S{i}@@")
    for i, key in enumerate(SECREF):
        s = s.replace(f"@@S{i}@@", SECREF[key])
    return s


def renumber_equations(s: str) -> str:
    """Equations 1-10 all live in Appendix A now, so they are tagged A.n."""
    out = []
    for line in s.split("\n"):
        if "\\ldots" in line:
            line = re.sub(r"\((\d{1,2})\)",
                          lambda m: "(A." + m.group(1) + ")", line)
        out.append(line)
    s = "\n".join(out)
    s = re.sub(r"\b([Ee]quation) \((\d{1,2})\)",
               lambda m: f"{m.group(1)} (A.{m.group(2)})", s)
    return s


def main() -> None:
    lines = SRC.read_text().split("\n")

    body = "\n\n".join([
        seg(lines, "intro"), seg(lines, "related"), seg(lines, "bench"),
        NEW_S4, seg(lines, "controls"), seg(lines, "probes"),
        seg(lines, "s5head"), seg(lines, "s51"), seg(lines, "s52"),
        seg(lines, "s53"), seg(lines, "s54"), seg(lines, "s55"),
        seg(lines, "s6"), seg(lines, "s7"), seg(lines, "s8"),
        seg(lines, "s9"), seg(lines, "compliance"),
    ])

    appa = "\n\n".join([
        APP_A, seg(lines, "a41"), seg(lines, "a42"), seg(lines, "a43"),
    ])
    appb = "\n\n".join([
        APP_B, seg(lines, "b56"), seg(lines, "b57"),
    ])

    appa = renumber_equations(appa)
    whole = body + "\n\n" + appa + "\n\n" + appb
    whole = rename_floats(whole)
    whole = rename_sections(whole)
    whole = re.sub(r"\n{4,}", "\n\n\n", whole)

    SRC.write_text(whole)

    # ---- report ---------------------------------------------------------
    print(f"  [OK] body + Appendix A + Appendix B written to {SRC}")
    leftovers = sorted(set(re.findall(r"Section [45]\.\d(?:\.\d)?", whole)))
    print("  section refs into 4.x / 5.x now:",
          ", ".join(leftovers) if leftovers else "none")
    heads = re.findall(r"^\\(?:sub)*section\*?\{(.+?)\}$", whole, re.M)
    print(f"  {len(heads)} headings, {whole.count(chr(92)+'floatcap')} captions")
    for pat in ("@@",):
        if pat in whole:
            print(f"  [!!] placeholder {pat} survived")
            sys.exit(1)


if __name__ == "__main__":
    main()
