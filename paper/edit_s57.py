"""
Pass 4c: Section 5.7, the Discussion.

Four paragraphs survived the previous pass with their original claims intact,
because that pass replaced leading fragments and left the tails. The tails are
where the claims lived: a 4.23-second warning horizon offered as one of the
strongest reported figures in the literature, a mathematically guaranteed
mechanism for stable-anticipation compliance, and immediate deployability on
novel road environments.

All four are replaced whole. The explainability and deployability observations
are kept, because both are true of the architecture independently of any
metric, but they are stated as architectural properties rather than as
validated performance.
"""
import copy
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = "unp5/word/document.xml"

NEW = [
    ("The results in this section admit one reading and it is not the one we "
     "expected when we ran them. The 4.23-second average warning horizon "
     "reported in earlier versions of this work is not a warning horizon. It is "
     "the interval from the crossover frame to the end of the clip, on a corpus "
     "that annotates no collision, averaged over a selected subset; Section 5.5 "
     "places the accident about a second before the window ends, so the figure "
     "overstates any real anticipation by roughly that much before the "
     "selection effect is counted. Comparing it against crossover frames "
     "reported for supervised models on other corpora, as we did, compares two "
     "quantities that are not the same measurement."),

    ("Second, the 100% stable-anticipation compliance rate we reported is not a "
     "result at all. The metric requires the risk score to stay above the "
     "threshold from the alarm until the accident; Stage 4 of the "
     "post-processing makes falling below the threshold impossible once the "
     "score has crossed. The figure therefore records that the clamp executed. "
     "This is the clearest instance in our own work of the general mechanism "
     "Section 6 identifies: a post-hoc operation that enforces a metric's "
     "definition renders that metric vacuous. It is worth being precise about "
     "what remains true. Raw vision–language risk scores do oscillate, the "
     "clamp does remove the oscillation, and a system that alarms and then "
     "retracts is genuinely worse for a driver than one that does not. The "
     "clamp is a reasonable engineering choice. What it cannot do is serve as "
     "evidence under a metric that tests for the property it imposes."),

    ("Third, the explainability afforded by the decoupled architecture is real "
     "and is independent of everything above. Because the three engines share "
     "no weights and are combined only at the fusion layer, any anticipation "
     "can be attributed to its constituent signals, and a failure can be "
     "localised to one of them. That is a property of the architecture, not a "
     "measured result, and we state it as such."),

    ("Finally, the zero-shot operational mode — no training data, no domain "
     "fine-tuning, no architecture modification — does mean the system can be "
     "pointed at a new road environment without retraining. Two qualifications "
     "belong with that claim. The ensemble weights of 0.55, 0.25 and 0.20 were "
     "chosen by searching over the evaluation corpus, which is a form of "
     "tuning on the test set and sits awkwardly with the zero-shot claim; the "
     "released configuration therefore uses equal weights, chosen a priori. And "
     "deployability on new environments is untested here, because we evaluated "
     "on one corpus. Section 8 records both as limitations."),
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
starts = ["The results collectively establish several important implications",
          "Second, the 100% stable-anticipation compliance rate",
          "Third, the explainability advantage afforded",
          "Finally, the strict zero-shot operational mode of the framework"]
n = 0
for p in tree.getroot().iter(W + "p"):
    t = text_of(p)
    for i, s in enumerate(starts):
        if t.startswith(s):
            set_text(p, NEW[i])
            n += 1
print(f"  [OK] {n} of 4 Discussion paragraphs replaced")
tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
