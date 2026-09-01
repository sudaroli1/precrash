"""
Give the converted body real sectioning.

Word's headings were auto-numbered list paragraphs, so pandoc emitted them as
one-item enumerates with a hand-set counter, and the subheadings as bold-italic
paragraphs carrying their number as literal text. The result compiled, but the
paper had exactly one \\section in it: nothing was numbered by LaTeX, nothing
was cross-referenceable, and every heading was set in body type.

This pass rewrites them as \\section, \\subsection and \\subsubsection. The
numbers LaTeX generates reproduce the numbers that were typed by hand -- the
subheadings run 2.1-2.5, 3.1-3.5, 5.1-5.7 and 5.6.1-5.6.3 with no gaps, and
Section 4's five subheadings are unnumbered in the source but sit in order --
so every "Section 5.3" in the running text still points where it did.

Two other things go with it. The hand-written reference list is removed, since
BibTeX now sets the references and the paper was carrying both. And the bold
paragraphs that serve as table and figure captions become a caption macro, so
they are set in caption type rather than in body bold.

  python structure.py
"""
from __future__ import annotations

import re
import pathlib

# in document order; the flag marks the one section TMLR should not number
SECTIONS = [
    ("RELATED WORK AND EVALUATION PRACTICE",
     "Related Work and Evaluation Practice", False),
    ("THE BENCHMARK AND ITS METRIC",
     "The Benchmark and its Metric", False),
    ("Method: the system under study, the video-blind controls, and the probes",
     "Method: the System under Study, the Video-Blind Controls, and the Probes",
     False),
    ("Problem Statement:", "Problem Statement", "sub"),
    ("Overall System Architecture:", "Overall System Architecture", "sub"),
    ("Implementation and Hardware:", "Implementation and Hardware", "sub"),
    ("Video-blind control family:", "Video-Blind Control Family", "sub"),
    ("Probes that isolate the terms of the metric:",
     "Probes that Isolate the Terms of the Metric", "sub"),
    ("RESULTS", "Results", False),
    ("ANALYSIS: WHY THE COMPOSITION FAILS",
     "Analysis: Why the Composition Fails", False),
    ("A CORRECTED PROTOCOL", "A Corrected Protocol", False),
    ("LIMITATIONS", "Limitations", False),
    ("CONCLUSION", "Conclusion", False),
    ("COMPLIANCE AND DATA AVAILABILITY",
     "Compliance and Data Availability", "star"),
]

# \begin{enumerate} \def\labelenumi{...} [\setcounter{enumi}{n}] \item
#   \textbf{TITLE} \end{enumerate}
BLOCK = re.compile(
    r"\\begin\{enumerate\}\s*\n"
    r"\\def\\labelenumi\{\\arabic\{enumi\}\.\}\s*\n"
    r"(?:\\setcounter\{enumi\}\{\d+\}\s*\n)?"
    r"\\item\s*\n"
    r"\s*\\textbf\{(?P<title>[^{}]*)\}\s*\n"
    r"\\end\{enumerate\}")

# \emph{\textbf{5.6.1. Title}}
SUB = re.compile(r"^\\emph\{\\textbf\{(\d+(?:\.\d+)*)\.\s*(.+?)\}\}\s*$",
                 re.MULTILINE)

# \textbf{Table 3: caption}  /  \textbf{Figure 1: caption}
CAP = re.compile(r"^\\textbf\{(Table|Figure)\s+(\d+):\s*(.*)\}\s*$",
                 re.MULTILINE)


def main() -> None:
    p = pathlib.Path("body_clean.tex")
    s = p.read_text()

    # ---- drop the hand-written reference list --------------------------
    i = s.find("\\textbf{References:}")
    if i >= 0:
        s = s[:i].rstrip() + "\n"
        print("  [OK] removed the hand-written reference list")
    else:
        print("  [!!] reference list marker not found")

    # ---- sections ------------------------------------------------------
    lookup = {raw: (nice, kind) for raw, nice, kind in SECTIONS}
    seen = set()

    def section(m):
        raw = m.group("title").strip()
        if raw not in lookup:
            print(f"  [!!] unmapped heading: {raw[:60]}")
            return m.group(0)
        nice, kind = lookup[raw]
        seen.add(raw)
        if kind == "sub":
            return "\\subsection{" + nice + "}"
        if kind == "star":
            return "\\section*{" + nice + "}"
        return "\\section{" + nice + "}"

    s, n = BLOCK.subn(section, s)
    missing = [r for r, _, _ in SECTIONS if r not in seen]
    print(f"  [OK] {n} headings promoted to sections"
          + (f"   MISSING {missing}" if missing else ""))

    # ---- subsections ---------------------------------------------------
    def sub(m):
        depth = m.group(1).count(".")
        cmd = "subsubsection" if depth >= 2 else "subsection"
        return "\\" + cmd + "{" + m.group(2).strip() + "}"

    s, n = SUB.subn(sub, s)
    print(f"  [OK] {n} numbered paragraphs promoted to sub(sub)sections")

    # ---- captions ------------------------------------------------------
    def cap(m):
        return ("\\floatcap{" + m.group(1) + " " + m.group(2) + "}{"
                + m.group(3).strip() + "}")

    s, n = CAP.subn(cap, s)
    print(f"  [OK] {n} table and figure captions set in caption type")

    p.write_text(s)
    print("wrote body_clean.tex")


if __name__ == "__main__":
    main()
