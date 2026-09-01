"""
Pass 16: the corrections the pre-submission audit found, applied to the
manuscript so that the Word document and the TMLR build say the same thing.

The audit read the compiled PDF end to end against the tables. Eleven of its
findings are defects in the text rather than in the conversion, so they belong
here too:

  - a difficulty listed in Section 4.1 stated the opposite of a difficulty:
    "without any training instances, it might generalize well"
  - the control family in Section 4.4 described the linear ramp as running
    0.001 to 0.999, which is the organisers' sample submission; the ramp runs
    0.000 to 1.000, and the two score differently (1.06725 against 1.04203)
  - Section 5.2 attributed the dip probe to Table 4; it is in Table 5
  - Table 6 rounded 0.87593/0.35105 = 2.4952 down to 2.49
  - Table 8's caption counted five curves crossing at frame 75; four do, the
    fifth row being the one-frame calibration step the caption then names
    separately. The prose called them three and read the span off four. The
    span itself, 0.09093, was right, and so was the abstract's 0.091
  - Table 9 rounded (150 - 22.7)/30 = 4.243 down to 4.23, having rounded the
    Top-50 cell up by the same convention
  - two of Table 10's deltas do not follow from the numbers beside them:
    26 - 22.7 is +3.3, not +4.3, and 21.7 - 22.7 is -1.0, which the stated
    range "+0.0 to -0.7" does not contain
  - the prose reports a mean peak risk of 0.9871 for the best twenty-five;
    the column it summarises means to 0.98584
  - Table 11's note still said the seconds were measured to a collision, which
    Sections 5.6 and 5.7 spend two paragraphs withdrawing
  - the abstract read "the two-timing terms", and called the diagnostic family
    three curves where it is four

  python paper/edit_audit.py
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp8/word/document.xml")

# (paragraph text must contain, old substring, new substring)
FIXES = [
    ("Traffic-accident anticipation is increasingly scored",
     "the two-timing terms contribute 2.00186",
     "the two timing terms contribute 2.00186"),

    ("Traffic-accident anticipation is increasingly scored",
     "three curves that cross the threshold at the same frame",
     "four curves that cross the threshold at the same frame"),

    ("The considered dataset is diverse in terms of accident categories",
     "The considered dataset is diverse in terms of accident categories, and "
     "without any training instances, it might generalize well.",
     "Pretrained knowledge is the only knowledge available, and with no "
     "training instances there is no way to establish in advance that it "
     "transfers to any of these accident categories."),

    ("A video-blind submission is one whose risk curve",
     "a linear ramp from 0.001 to 0.999",
     "a linear ramp from 0.000 to 1.000"),

    ("Two further readings follow from Table 4.",
     "Two further readings follow from Table 4.",
     "Two further readings follow from Tables 4 and 5."),

    ("Table 8: Five curves that all first exceed 0.5 at frame 75",
     "Table 8: Five curves that all first exceed 0.5 at frame 75",
     "Table 8: Four curves that all first exceed 0.5 at frame 75"),

    ("The three curves in the middle of Table 8",
     "The three curves in the middle of Table 8",
     "The four curves at the head of Table 8"),

    ("mean peak risk reached 0.9871",
     "mean peak risk reached 0.9871",
     "mean peak risk reached 0.9858"),

    ("Table 11, note: TTA derived from remaining frames",
     "Table 11, note: TTA derived from remaining frames prior to collision at "
     "30 FPS. Initial risk vectors verified post power-curve smoothing and "
     "Gaussian filtering.",
     "Table 11, note: the seconds column is the interval from the crossover "
     "frame to the end of the clip at 30 FPS. It is not a time-to-accident: "
     "this corpus annotates no collision frame (Section 3.3), and the figure "
     "is inflated by the interval Section 5.5 estimates at about a second. "
     "Initial risk vectors are read after the power transform and the "
     "Gaussian filter."),
]

# table cells, matched on their whole text so a bare number cannot collide
CELLS = [
    ("2.49×", "2.50×"),
    ("4.23 s", "4.24 s"),
    ("+4.3 to +5.3", "+3.3 to +5.3"),
    ("+0.0 to −0.7", "−1.0"),
]


def text_of(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def set_text(p, text):
    runs = p.findall(W + "r")
    tmpl = copy.deepcopy(runs[0]) if runs else None
    for r in runs:
        p.remove(r)
    run = tmpl if tmpl is not None else etree.SubElement(p, W + "r")
    if tmpl is not None:
        for t in run.findall(W + "t"):
            run.remove(t)
        p.append(run)
    t = etree.SubElement(run, W + "t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


def main():
    tree = etree.parse(str(DOC))
    body = tree.getroot().find(W + "body")
    paras = list(body.iter(W + "p"))

    hits = {i: 0 for i in range(len(FIXES))}
    for p in paras:
        txt = text_of(p)
        for i, (anchor, old, new) in enumerate(FIXES):
            if anchor in txt and old in txt:
                txt = txt.replace(old, new)
                set_text(p, txt)
                hits[i] += 1

    bad = [FIXES[i][1][:50] for i, n in hits.items() if n != 1]
    print(f"  [OK] {len(FIXES) - len(bad)} of {len(FIXES)} prose fixes"
          + (f"   FAILED {bad}" if bad else ""))

    cells = {old: 0 for old, _ in CELLS}
    for p in paras:
        txt = text_of(p).strip()
        for old, new in CELLS:
            if txt == old:
                set_text(p, new)
                cells[old] += 1
    bad = [o for o, n in cells.items() if n != 1]
    print(f"  [OK] {len(CELLS) - len(bad)} of {len(CELLS)} table cells"
          + (f"   FAILED {bad}" if bad else ""))

    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8",
               standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
