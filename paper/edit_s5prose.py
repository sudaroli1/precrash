"""
Pass 4b: the prose inside Section 5.6.

Section 5.6 now says clearly that everything under it is a statistic of curve
shape rather than an anticipation result. The paragraphs themselves still read
as though they were results — "state-of-the-art predictive capability",
"definitive empirical validation", "biomechanical significance", a
false-positive rate on a corpus with no labels, and a 100% stable-anticipation
compliance figure that measures a clamp. A framing sentence at the top of a
subsection does not repair claims made four pages later, and a referee reads
the claims.

Each paragraph below is rewritten to say what was actually measured. Nothing is
dropped: where there is a real observation underneath the language — the
ablation orderings, the initial-risk suppression, the peak saturation — it is
kept and re-stated as what it is, a property of the curve. Where the claim
cannot survive at all, the paragraph says so rather than disappearing, because
the point of Section 5.6 is to be an honest record of how the work was done
before the leaderboard settled it.
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp5/word/document.xml")

REWRITES = [
    ("IV. EXPERIMENTAL SETUP AND RESULTS",
     "4.6 continues below."),

    ("The proposed zero-shot ensemble framework demonstrated robust and "
     "state-of-the-art predictive capability",
     "Across the full corpus of 1,417 clips the ensemble produced a mean "
     "crossover frame of 22.7. We reported this at the time as predictive "
     "capability; it is not. The crossover frame is the earliest frame at which "
     "the predicted risk exceeds 0.5, and it is a property of the curve the "
     "system emits, measurable without any reference to whether an accident "
     "occurs or when. Section 5.1 gives the same statistic for a submission "
     "that reads no pixels: crossover frame 0. The primary performance "
     "indicator in accident anticipation systems is the crossover frame (tᴄ)"),

    ("Moreover, the early-frame false alarm rate",
     "We also reported an early-frame false alarm rate, which was a mistake: a "
     "false alarm is an alarm on a clip containing no accident, and this corpus "
     "does not say which clips those are, so no false-alarm rate is computable "
     "from it by an entrant. What was actually observed is that initial-frame "
     "risk scores initialised below 0.05 on every clip — a statement about "
     "where the curve starts, not about whether starting there was correct. "
     "The original claim read: the early-frame false alarm rate"),

    ("To establish the theoretical performance ceiling of the zero-shot "
     "framework, we isolated",
     "The following analysis selects the fifty clips with the earliest "
     "crossover frames and reports statistics over them. This is a statement "
     "about the selection and not about the method: choosing the best fifty of "
     "1,417 by the quantity being reported guarantees the reported quantity "
     "looks good, and the same procedure applied to any system, including a "
     "random one, would produce a flattering subset. We retain it because it is "
     "part of the record. It was originally presented as establishing a "
     "performance ceiling, and it does not. We isolated"),

    ("Absolute Initial Risk Suppression: 100% of the top-50 predictions began "
     "with an initial risk score strictly within the range [0.001, 0.05]. This "
     "definitively demonstrates that ultra-early crossover frames are driven by "
     "the genuine, multi-modal recognition of pre-crash kinematic and semantic "
     "cues — not by a poorly calibrated, globally elevated baseline.",
     "Initial risk suppression: every one of the fifty began with an initial "
     "risk score in [0.001, 0.05]. This shows that the early crossings are not "
     "produced by a globally elevated baseline, which is a real and useful "
     "property of the curve. It does not show that the crossings track pre-crash "
     "cues, because nothing here is compared against a label; the earlier claim "
     "that it demonstrates genuine multi-modal recognition does not follow from "
     "this measurement."),

    ("The quantitative superiority of the full ensemble architecture is most "
     "compellingly demonstrated through a systematic modality ablation "
     "analysis.",
     "The modality ablation below compares single-modality configurations "
     "against the full ensemble on the same crossover-frame proxy. The "
     "orderings it produces are informative about how the three signals behave "
     "and about their failure modes, and we retain the analysis for that "
     "reason. It does not establish superiority in anticipation, because the "
     "quantity being compared is minimised by a constant."),

    ("To rigorously bound the theoretical maximum efficacy of the zero-shot "
     "ensemble, we isolated",
     "The following selects the twenty-five clips with the earliest crossings, "
     "subject to an initial-risk constraint. The caveat of Section 5.6.2 "
     "applies with more force at this sample size: a best-of-1,417 subset "
     "bounds nothing. It was originally presented as bounding the maximum "
     "efficacy of the framework, and it does not. We isolated"),

    ("The biomechanical significance of this performance envelope is "
     "substantial.",
     "We originally drew a road-safety conclusion from this figure, comparing "
     "it against human reaction times and automatic emergency braking latency. "
     "That comparison cannot be made from these numbers, for two reasons. The "
     "interval reported is measured from the crossover frame to the end of the "
     "clip, not to a collision, because this corpus annotates no collision; "
     "Section 5.5 estimates that the accident falls about a second before the "
     "window ends, so the figure is inflated by roughly that much. And it is "
     "computed over a best-of-1,417 subset. The paragraph is retained as "
     "written because it is the clearest single example of what the proxy "
     "encouraged, and it should be read as such rather than as a claim. It read:"),

    ("The Top-25 subset provides definitive empirical validation of the "
     "ensemble's synergistic architecture. Two critical properties characterise "
     "all 25 sequences:",
     "Two properties hold across all twenty-five sequences. They are "
     "descriptions of the curves in a selected subset, not validation of the "
     "architecture:"),

    ("Zero-Shot False-Positive Eradication: Despite ultra-early crossover "
     "frames as low as frame 7, every sequence maintained an absolute minimal "
     "initial risk boundary of p(1) = 0.001. This is a non-trivial result:",
     "Low initial risk despite early crossing: every sequence began at "
     "p(1) = 0.001 despite crossing as early as frame 7. The heading under "
     "which this appeared, false-positive eradication, was wrong — a "
     "false-positive rate is not computable here — but the underlying "
     "observation stands and is worth keeping:"),

    ("Monotonic Saturation via NLP Prior Stabilisation: As sequences progress "
     "toward the collision terminus, the mean peak risk across the Top-25 "
     "saturates at 0.9871 (range: 0.9629–0.9986). The strict monotone clamping",
     "Peak saturation, largely by construction: mean peak risk across the "
     "twenty-five reaches 0.9871 (range 0.9629–0.9986). Note that the score "
     "cannot fall below the threshold after the first crossing, because "
     "Stage 4 forbids it, so the absence of degradation late in the clip is "
     "mostly a property of the clamp rather than of the NLP prior. The strict "
     "monotone clamping"),

    ("These 25 extreme cases constitute robust empirical proof that the "
     "integration of multi-modal semantic embeddings with kinematic flow "
     "metrics — under strictly zero-shot conditions — can achieve anticipation "
     "performance that rivals or exceeds fully supervised, domain-specific deep "
     "learning models.",
     "These twenty-five cases were originally offered as proof that a "
     "zero-shot ensemble can rival supervised models. They are not: they are a "
     "selected subset scored on a proxy, with no supervised model evaluated "
     "alongside them and no label against which either could be checked. The "
     "leaderboard, which does compare against other entrants under a common "
     "scorer, placed this system ninth of thirteen and below a constant."),

    ("The progressive performance improvement from C1 through C7 demonstrates "
     "that each architectural component",
     "The ordering from C1 to C7 shows that each stage moves the crossover "
     "frame earlier. Read against Section 5.1 that is what it says and no more, "
     "because moving the crossover frame earlier is exactly what a constant 0.51 "
     "does perfectly. Two of the stages deserve particular note in this light: "
     "the monotone clamp forces the condition the stable-timing metric tests, "
     "and the temporal compression that produces the final improvement reads a "
     "frame that has not yet arrived (Section 4). The original text read: each "
     "architectural component"),

    ("Second, the 100% STTA compliance rate across the full evaluation corpus "
     "is a non-trivial and practically significant result.",
     "Second, the 100% stable-anticipation compliance rate we reported is not a "
     "result at all. The metric requires the risk score to stay above the "
     "threshold from the alarm until the accident; Stage 4 of the "
     "post-processing makes falling below the threshold impossible once the "
     "score has crossed. The figure therefore records that the clamp executed. "
     "This is the clearest instance in our own work of the general mechanism "
     "Section 6 identifies: a post-hoc operation that enforces a metric's "
     "definition renders that metric vacuous."),
]


def text_of(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def set_text(p, text):
    runs = p.findall(W + "r")
    tmpl = copy.deepcopy(runs[0]) if runs else None
    for r in runs:
        p.remove(r)
    for tag in ("hyperlink", "bookmarkStart", "bookmarkEnd", "proofErr"):
        for el in p.findall(W + tag):
            p.remove(el)
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
    root = tree.getroot()
    hits = {old: 0 for old, _ in REWRITES}
    for p in root.iter(W + "p"):
        txt = text_of(p)
        for old, new in REWRITES:
            if old in txt:
                txt = txt.replace(old, new)
                set_text(p, txt)
                hits[old] += 1
    ok = 0
    for old, n in hits.items():
        if n == 1:
            ok += 1
        else:
            print(f"  [!! {n}] {old[:64]}...")
    print(f"  [OK] {ok} of {len(REWRITES)} rewrites applied cleanly")
    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
