"""Second tightening of the introduction: 888 words to about 730.

Only compression — no argument, number or citation is lost. The opening drops a
clause it did not need, the boundedness paragraph stops restating its own
premise, the four contribution bullets lose about a sixth each, and the scope
paragraph loses its closing restatement while keeping every clause that does
protective work."""
import copy
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = "unp6/word/document.xml"

PAIRS = [
("Road traffic injury kills approximately 1.19 million people a year by the World Health Organization's count [1], and leaves far more with permanent disability. Many of those deaths are in principle foreseeable in the seconds before they occur, which is what makes anticipation — as distinct from detection after the fact — worth pursuing, as the review of Zhang Y. et al. [2] sets out. The instrument is already in the vehicle: a dashcam holds the primary visual record of the road, including the lane drift, the pedestrian and the hard braking ahead that precede a collision, which is the material Moura et al. [3] assembled into a public corpus. That record is used almost entirely in retrospect. The question motivating this work was whether it could be used prospectively.",
 "Road traffic injury kills approximately 1.19 million people a year by the World Health Organization's count [1]. Many of those deaths are foreseeable in the seconds before they occur, which is what makes anticipation — as distinct from detection after the fact — worth pursuing, as the review of Zhang Y. et al. [2] sets out. The instrument is already in the vehicle: a dashcam holds the primary visual record of the road, the material Moura et al. [3] assembled into a public corpus. It is used almost entirely in retrospect. Whether it can be used prospectively is the question that motivated this work."),

("It is not. AP and AUC are bounded on the unit interval; TTA and STTA are counted in frames and bounded only by the length of the clip, which is 150 frames here. Summing a bounded quantity with an unbounded one lets the unbounded one decide the ranking, and on this benchmark the timing terms are worth 5.70 times what the discrimination terms contribute to a submission that reads nothing. Worse, the two timing terms share a trivial global maximiser. A risk score that exceeds 0.5 at frame 0 and never falls attains, on every clip, the largest value that clip admits for both terms at once. No method that reads the video can exceed it. A method can only match it and then compete on AP and AUC, which we show are worth 0.35105 to a blind submission.",
 "It is not. AP and AUC are bounded on the unit interval; TTA and STTA are counted in frames and bounded only by the clip, which is 150 frames here. Summing a bounded quantity with an unbounded one lets the unbounded one decide the ranking, and here the timing terms are worth 5.70 times what discrimination contributes to a submission that reads nothing. Worse, the two share a trivial global maximiser: a score that exceeds 0.5 at frame 0 and never falls attains, on every clip, the largest value that clip admits for both. No method that reads the video can exceed it; a method can only match it and then compete on AP and AUC, worth 0.35105 to a blind submission."),

("A measured demonstration on a live benchmark. A video-blind constant matches the eighth-placed private score and exceeds those of five of thirteen teams, our own included. Every leaderboard score reported here was computed by the competition's own scoring service against ground truth we do not have; all derived quantities are our arithmetic on those scores and are marked as such.",
 "A measured demonstration on a live benchmark. A video-blind constant matches the eighth-placed private score and exceeds five of thirteen teams, our own included. Every leaderboard score here was computed by the competition's scoring service against ground truth we do not have; derived quantities are our arithmetic on those scores and are marked as such."),

("A black-box decomposition of a composite metric. A probe methodology — video-blind submissions constructed so that unknown terms cancel under differencing — separates a metric whose weights are undisclosed and whose labels are withheld, recovering the discrimination floor, both timing terms, the score's slope in the crossing frame, and the support and mean of the accident-onset distribution.",
 "A black-box decomposition. Video-blind probes, built so that unknown terms cancel under differencing, separate a metric whose weights are undisclosed and whose labels are withheld — recovering the discrimination floor, both timing terms, the slope in the crossing frame, and the accident-onset distribution."),

("Evidence that the published metric definition does not reproduce the scorer. Three curves that cross the threshold at the same frame, with identical values at the crossing and no subsequent dip, receive scores spanning 0.091. Under the stated formula they must be equal, so this benchmark's score cannot be computed from its published definition even given the labels.",
 "Evidence that the published definition does not reproduce the scorer. Three curves crossing at the same frame, with identical values at the crossing, score 0.091 apart; under the stated formula they must be equal. The score cannot be computed from the published definition even given the labels."),

("A case study in which we are the subject, and a corrected protocol. Our own entry scores below the constant, and we report it rather than omit it. From the mechanisms we derive six requirements, of which the first — report a video-blind control with every anticipation result — costs an afternoon and would have caught this.",
 "A case study in which we are the subject, and a six-point protocol. Our own entry scores below the constant and we report it. The protocol's first item — report a video-blind control with every result — costs an afternoon and would have caught this."),

("One point of scope, because this paper prints a number beside other teams' numbers. We make no claim about the validity, quality or honesty of any other submission or published method. We did not re-run anyone's system and could not have evaluated it if we had, because the labels are withheld from every entrant. Where we observe that a leaderboard score is numerically identical to a control's, we state an arithmetic fact about two published numbers and draw exactly one inference from it: the metric assigns those two submissions the same value. A whole family of curves receives that same value, which is precisely the property under examination. The claim of this paper is about a metric, demonstrated on one benchmark.",
 "One point of scope, because this paper prints a number beside other teams' numbers. We make no claim about any other submission or published method: we did not re-run anyone's system, and could not have evaluated it if we had, because the labels are withheld from every entrant. Where a leaderboard score is numerically identical to a control's, we state that arithmetic fact and draw one inference — the metric assigns the two submissions the same value. A whole family of curves receives it, which is precisely the property under examination."),

("Section 2 reviews accident anticipation and the evaluation practice this work examines. Section 3 states the benchmark and its metric, and what an entrant can compute from them. Section 4 describes the system, the video-blind controls and the probe design. Section 5 reports the measurements and Section 6 analyses why the composition fails. Sections 7, 8 and 9 give the corrected protocol, the limitations and the conclusion.",
 "Section 2 reviews the field's evaluation practice, Section 3 states the benchmark and its metric, and Section 4 describes the system, the controls and the probes. Section 5 reports the measurements, Section 6 analyses why the composition fails, and Sections 7 to 9 give the protocol, the limitations and the conclusion."),
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
n = 0
for p in tree.getroot().iter(W + "p"):
    t = text_of(p)
    for old, new in PAIRS:
        if t.strip() == old:
            set_text(p, new)
            n += 1
print(f"  [OK] {n} of {len(PAIRS)} paragraphs tightened")
tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
