"""
Pass 7: close the parking bay.

The "MATERIAL MOVED FROM THE INTRODUCTION" block was a holding pen from pass 1,
kept so that displaced text was not destroyed. It then acquired an automatic
number and began presenting itself as Section 10.

Two of its four paragraphs are superseded: Section 2.1 now covers the
dependence of supervised methods on their training corpus, and 2.2 covers the
case for vision-language models, both with citations that survive checking.
Those two go.

The other two are kept and moved into Section 2.1, where they open the section:
the distinction between anticipation and detection, which the paper defines
nowhere else, and the precursor-state example, which is the most concrete piece
of writing in the related work and explains why the task is hard at all. Both
are lightly edited for grammar and recited against the current reference list --
their old brackets pointed at SCTNet and at a pedestrian-psychology paper that
the audit removed.
"""
import copy
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = "unp6/word/document.xml"

KEEP = [
    "Anticipation is a different and harder problem than detection. In "
    "detection the task is to identify what is present in the scene — a "
    "vehicle, a pedestrian, a traffic light — or what is being done in it, and "
    "to classify it; the evidence is in the frame, as in the unsupervised "
    "detection work of Liu et al. [17]. In anticipation the object of the "
    "prediction is an event that may or may not occur, so the model is asked "
    "to reason forward from evidence that is ambiguous and intermittent.",

    "What makes that reasoning possible is the precursor state: a visual "
    "signal that danger is developing before anything has happened. A driver "
    "on a highway notices that the car fifty metres ahead has drifted twenty "
    "centimetres to the right and lost a little speed. No brake has been "
    "applied and no collision has occurred, but the drift and the deceleration "
    "together indicate drowsiness or distraction. An anticipation model must "
    "read signals of the same kind, and the precursors available to it are of "
    "the same kind: deviation from an expected vehicle track, congestion "
    "patterns, and the projected intersection of two objects extrapolated from "
    "their current motion — the basis on which Yao et al. [18] treat an "
    "accident as a departure from predicted trajectories.",
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


tree = etree.parse(DOC)
body = tree.getroot().find(W + "body")

# everything from the parking heading to the end of the body
kids = list(body)
start = None
for i, c in enumerate(kids):
    if c.tag == W + "p" and text_of(c).startswith("MATERIAL MOVED FROM THE INTRODUCTION"):
        start = i
        break
if start is None:
    raise SystemExit("parking heading not found")

doomed = [c for c in kids[start:] if c.tag == W + "p"]
for c in doomed:
    body.remove(c)
print(f"  [OK] removed the parking bay ({len(doomed)} paragraphs, heading included)")

# reinsert the two survivors at the head of 2.1
anchor = None
for c in body.iter(W + "p"):
    if text_of(c).startswith("Vision-based accident anticipation was posed"):
        anchor = c
        break
if anchor is None:
    raise SystemExit("2.1 opening paragraph not found")

tmpl = copy.deepcopy(anchor)
at = list(body).index(anchor)
for off, txt in enumerate(KEEP):
    body.insert(at + off, set_text(copy.deepcopy(tmpl), txt))
print(f"  [OK] moved {len(KEEP)} paragraphs to the head of Section 2.1")

tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
