"""
Convert the numeric [n] citations to natbib author-year, which is what the TMLR
style file sets up: \\setcitestyle{authoryear,round,citesep={;},...}.

The paper is written author-prominent throughout, so almost every citation is
"Chan et al. [4]" and becomes \\citet{chan2016dad}, which natbib sets as
"Chan et al. (2016)" — the name is no longer typed by hand. The few citations
that are not attributed in the sentence become \\citep{...}.

Also removes the hand-written reference list, since BibTeX generates it now.
"""
from __future__ import annotations

import re
import pathlib

KEYS = {
    1: "who2023", 2: "zhang2025review", 3: "moura2025nexar", 4: "chan2016dad",
    5: "bao2020ccd", 6: "karim2022dsta", 7: "fang2024mmau", 8: "autopilot2026",
    9: "radford2021clip", 10: "teed2020raft", 11: "liu2023llava",
    12: "chen2024internvl", 13: "liao2025cot", 14: "zhang2025camera",
    15: "zhang2025seeunsafe", 16: "kandacharam2025fusion", 17: "liu2026sctnet",
    18: "yao2019unsupervised", 19: "zhong2025early",
    20: "pjetri2025nondecreasing", 21: "zou2026riskprop", 22: "saha2026zeroshot",
    23: "singh2026metadata", 24: "thakur2026modular", 25: "torralba2011bias",
    26: "gururangan2018artifacts", 27: "poliak2018hypothesis",
    28: "ferraridacrema2019progress", 29: "korkut2026blind", 30: "zhao2025top",
    31: "goldshmidt2025badas", 32: "caselli2026flara",
}

# An author name sitting immediately before the bracket. natbib will set the
# name itself, so the hand-typed one is removed along with the bracket.
NAME = re.compile(
    r"(?:[A-Z][A-Za-zÀ-ſ'\-]+"
    r"(?:\s+[A-Z]\.)?"
    r"(?:\s+(?:et\s+al\.|and\s+[A-Z][A-Za-z'\-]+))?"
    r"|World Health Organization|AUTOPILOT)\s*$"
)


def main() -> None:
    p = pathlib.Path("body_clean.tex")
    s = p.read_text()

    for marker in ("\\section{References", "\\section*{References"):
        i = s.find(marker)
        if i >= 0:
            s = s[:i]
            print("  [OK] removed the hand-written reference list")
            break

    # multi-citations first: "[20], [21]" -> \citep{a,b}
    def multi(m):
        nums = [int(x) for x in re.findall(r"\d+", m.group(0))]
        keys = ",".join(KEYS[n] for n in nums if n in KEYS)
        return "\\citep{" + keys + "}"

    s = re.sub(r"\[\d{1,2}\](?:\s*,\s*\[\d{1,2}\])+", multi, s)

    counts = {"t": 0, "p": 0}
    out, last = [], 0
    for m in re.finditer(r"\[(\d{1,2})\]", s):
        key = KEYS.get(int(m.group(1)))
        if key is None:
            continue
        head = s[last:m.start()]
        nm = NAME.search(head.rstrip())
        if nm:
            # drop the hand-typed name; \citet prints it
            head = head[:len(head.rstrip()) - len(nm.group(0).rstrip())]
            out.append(head)
            out.append("\\citet{" + key + "}")
            counts["t"] += 1
        else:
            out.append(head)
            out.append("\\citep{" + key + "}")
            counts["p"] += 1
        last = m.end()
    out.append(s[last:])
    s = "".join(out)

    p.write_text(s)
    print(f"  [OK] {counts['t']} author-prominent -> \\citet, "
          f"{counts['p']} -> \\citep")


if __name__ == "__main__":
    main()
