"""
Pass 10: condense Section 5.6 from 2,845 words to about 800.

Seven subsections become three. All four tables stay exactly where they are;
what goes is the prose around them, which had grown to argue with itself at
length. The argument 5.6 needs to make is short — here is what we measured
locally, here is why it was not a measurement, here is the table — and it was
taking a page and a half to make it three times.

Nothing honest is lost. Every correction still stands: the crossover frame is a
curve property and not performance, no false-alarm rate is computable on this
corpus, the best-fifty and best-twenty-five analyses are statements about the
selection, the anticipation margin was measured to the end of the clip rather
than to a collision, the compliance column records that the clamp executed, and
the C6-to-C7 improvement is read from a frame that has not yet arrived.
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp6/word/document.xml")

FRAMING = (
    "What follows is the evaluation we conducted before any submission was "
    "scored. It is retained because it is the case study this paper is built "
    "on. On a corpus that publishes no labels, average precision, area under "
    "the curve, a false-positive rate and any time-to-accident measured to a "
    "true onset are not computable by an entrant (Section 3.3); what is "
    "computable from a submission alone is the shape of its own risk curve, "
    "and the statistic we adopted was the mean frame at which that curve first "
    "exceeds 0.5 — the crossover frame. Every number below is therefore a "
    "statistic of curve shape. None is an anticipation result, none was checked "
    "against a label, and the quantity they optimise is minimised absolutely, "
    "to frame 0, by the constant of Section 5.1. Two cautions apply throughout. "
    "Timing figures in seconds were computed from the crossover frame to the "
    "end of the clip, not to a collision, because this corpus annotates none, "
    "so they are inflated by the interval Section 5.5 estimates at about a "
    "second. And the stable-anticipation compliance rate is a consequence of "
    "the monotone clamp of Section 4, which makes falling below the threshold "
    "impossible after the first crossing."
)

HEAD_1 = "5.6.1. What the proxy reported"

PARA_A = (
    "Across the full corpus of 1,417 clips the ensemble produced a mean "
    "crossover frame of 22.7, and 21.1 over the best-fifty subset discussed "
    "below. We reported these as predictive capability; they are not. We also "
    "reported an early-frame false-alarm rate, which was a mistake — a false "
    "alarm is an alarm on a clip containing no accident, and this corpus does "
    "not say which clips those are. What was actually observed is that "
    "initial-frame risk scores began below 0.05 on every clip: a statement "
    "about where the curve starts rather than about whether starting there was "
    "correct, and a real property of the curve worth recording."
)

PARA_B = (
    "The modality ablation of Table 13 compares single-modality configurations "
    "against the full ensemble on the same proxy. Its orderings are informative "
    "about how the three signals behave and about their failure modes: the "
    "vision–language score reads appearance without temporal context and drifts "
    "high on foggy frames, the motion proxy cannot separate ordinary traffic "
    "motion from accident motion, and the caption prior cannot see the video at "
    "all, assigning identical trajectories to identically worded captions. The "
    "ensemble does resolve those three blind spots against one another. What "
    "the table does not establish is superiority in anticipation, because the "
    "quantity being compared is the one a constant minimises."
)

HEAD_2 = "5.6.2. The selected subsets"

PARA_C = (
    "Two further analyses select the fifty and then the twenty-five clips with "
    "the earliest crossings. Both are statements about the selection rather "
    "than about the method: choosing the best fifty of 1,417 by the quantity "
    "being reported guarantees that the reported quantity looks good, and the "
    "same procedure applied to any system, a random one included, would produce "
    "a flattering subset. Within the best twenty-five, every sequence began at "
    "p(1) = 0.001 despite crossing as early as frame 7, and mean peak risk "
    "reached 0.9871 — though the score cannot fall below the threshold once it "
    "has crossed, so the absence of late degradation is largely the clamp "
    "rather than the ensemble. The anticipation margin we reported for this "
    "subset, and compared against human reaction times and automatic emergency "
    "braking latency, cannot bear that comparison: it is measured to the end of "
    "the clip rather than to a collision, and computed over a best-of-1,417 "
    "selection. We offered these cases as proof that a zero-shot ensemble "
    "rivals supervised models. They are not. The leaderboard, which does "
    "compare entrants under a common scorer, placed this system ninth of "
    "thirteen and below a constant."
)

HEAD_3 = "5.6.3. The seven configurations"

PARA_D = (
    "Table 15 compares seven pipeline configurations. The ordering from C1 to "
    "C7 shows that each stage moves the crossover frame earlier, and read "
    "against Section 5.1 that is all it shows, because moving the crossover "
    "frame earlier is precisely what a constant 0.51 does perfectly. Two stages "
    "deserve particular note in that light. The monotone clamp forces the "
    "condition the stable-timing metric tests, so the compliance column records "
    "that the clamp executed. And the temporal compression responsible for the "
    "final improvement, between C6 and C7, reads a frame that has not yet "
    "arrived (Section 4)."
)

# (prefix to find, replacement text) — applied to the first match only
REPLACE = [
    ("What follows is the evaluation we conducted before any submission", FRAMING),
    ("5.6.1. Overall curve-shape statistics", HEAD_1),
    ("Across the full corpus of 1,417 clips the ensemble produced", PARA_A),
    ("The modality ablation below compares single-modality", PARA_B),
    ("5.6.4. The best twenty-five trajectories", HEAD_2),
    ("The following selects the twenty-five clips with the earliest", PARA_C),
    ("5.6.5. Seven-configuration comparison", HEAD_3),
    ("Beyond the modality ablation study, the framework was validated", PARA_D),
]

DELETE = [
    "Every number in the remainder of Section 5 is a statistic of curve shape",
    "Two specific cautions apply to the tables that follow",
    "Our framework achieved a mean crossover frame of 22.7 across the entire",
    "We also reported an early-frame false alarm rate, which was a mistake",
    "5.6.2. The best-fifty subset",
    "The following analysis selects the fifty clips with the earliest",
    "Within this high-performance cohort, the framework achieved",
    "Initial risk suppression: every one of the fifty began",
    "Peak Confidence Saturation: Maximum risk scores at the collision terminus",
    "5.6.3. Modality ablation, measured on the crossover proxy",
    "The ablation results expose a clear blind-spot complementarity",
    "The ensemble resolves all three blind spots simultaneously",
    "5.6.4.1. Aggregate statistics of the best twenty-five",
    "Aggregate analysis of the Top-25 subset yields",
    "We originally drew a road-safety conclusion from this figure",
    "5.6.4.2. Modality behaviour in the best twenty-five",
    "Two properties hold across all twenty-five sequences",
    "Low initial risk despite early crossing: every sequence began",
    "Peak saturation, largely by construction",
    "These twenty-five cases were originally offered as proof",
    "The ordering from C1 to C7 shows that each stage moves",
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

    done = set()
    for p in body.iter(W + "p"):
        t = text_of(p).strip()
        for pre, new in REPLACE:
            if pre not in done and t.startswith(pre):
                set_text(p, new)
                done.add(pre)
                break
    missing = [p for p, _ in REPLACE if p not in done]
    print(f"  [OK] {len(done)} of {len(REPLACE)} replaced"
          + (f"   MISSING: {missing}" if missing else ""))

    gone = 0
    seen = set()
    for p in list(body.iter(W + "p")):
        t = text_of(p).strip()
        for pre in DELETE:
            if pre not in seen and t.startswith(pre):
                body.remove(p)
                seen.add(pre)
                gone += 1
                break
    absent = [p for p in DELETE if p not in seen]
    print(f"  [OK] {gone} of {len(DELETE)} removed"
          + (f"   NOT FOUND: {absent}" if absent else ""))

    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
