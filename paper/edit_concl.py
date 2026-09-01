"""
Pass 6: a shorter conclusion, and the Future Scope section removed.

The conclusion ran to four paragraphs that restated Section 5 at length. It is
now three: what was found, what the method makes possible, and what the work
opens up for anyone who follows.

The Future Scope section goes entirely -- an introduction, six subsections and
a summary, all written for a systems paper that no longer exists. Nothing
substantive is lost with it. Its one disclosure that mattered, that the ensemble
weights were grid-searched on the evaluation corpus, is already recorded in
Section 5.7 and again in Section 8; and the two directions worth keeping,
validation on a labelled corpus and the deployment question, are folded into
the final paragraph of the conclusion.
"""
import copy
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = "unp6/word/document.xml"

NEW_CONCLUSION = [
    "A composite score that adds an unbounded earliness term to bounded "
    "discrimination terms does not measure what its name suggests. On a live "
    "accident-anticipation benchmark the earliness terms are worth 5.70 times "
    "what discrimination contributes to a submission that reads nothing; they "
    "are maximised exactly by a constant; and that constant scores within "
    "0.14874 of the winning entry, matching the eighth-placed team at the "
    "precision the leaderboard displays. Our own entry, a training-free "
    "ensemble, placed ninth of thirteen and scores below it.",

    "The method that established this is as portable as the finding. "
    "Twenty-four submissions, not one of which opened a video frame, were "
    "enough to recover the score's slope to a sixtieth of a point per frame of "
    "delay, to bound the support and mean of an accident-onset distribution no "
    "entrant has seen, and to establish that the benchmark's published formula "
    "does not reproduce its own scorer. A leaderboard that accepts arbitrary "
    "submissions is an instrument, and any benchmark that accepts one can be "
    "measured from outside in this way.",

    "Three things are consequently cheap that were not. A video-blind control "
    "costs a single submission and settles in an afternoon whether a reported "
    "score is measuring a method or a metric; we would like it to become as "
    "routine as a majority-class baseline in classification. Probe submissions "
    "turn a leaderboard into a means of auditing a scorer whose code is "
    "closed, without labels and without the weights. And an earliness term "
    "that is bounded per clip and conditioned on a fixed operating point of "
    "the discrimination metric, rather than summed raw, gives the field a "
    "composite it can compare across corpora. Demonstrating that corrected "
    "protocol end to end on a benchmark that publishes its annotations — where "
    "the discrimination half can actually be computed, and where the question "
    "of what a bounded earliness term costs a real method can finally be "
    "answered — is the natural next study, and it is the one we intend.",
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
kids = list(body)

# Everything from the first conclusion paragraph to the last line of Future
# Scope, inclusive.
start = end = None
for i, c in enumerate(kids):
    if c.tag != W + "p":
        continue
    t = text_of(c)
    if start is None and t.startswith("A composite metric that adds an unbounded"):
        start = i
    if t.startswith("In summary, PreCrash establishes that zero-shot"):
        end = i
if start is None or end is None:
    raise SystemExit(f"bounds not found: start={start} end={end}")

tmpl = copy.deepcopy(kids[start])
removed = kids[start:end + 1]
for c in removed:
    body.remove(c)

at = list(body).index(kids[start - 1]) + 1
for off, txt in enumerate(NEW_CONCLUSION):
    body.insert(at + off, set_text(copy.deepcopy(tmpl), txt))

print(f"  [OK] removed {len(removed)} paragraphs (conclusion + Future Scope)")
print(f"  [OK] inserted {len(NEW_CONCLUSION)} paragraphs")
tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
