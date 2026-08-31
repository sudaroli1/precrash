"""
Pass 1 on Precrash.docx: title, abstract, keywords, code link, introduction.

Edits the XML in place so the authors' styles, numbering and section setup
survive. Nothing is deleted: the four background paragraphs that the new
introduction displaces are moved to the end of the document under a marked
heading, to be folded into Section 2 in the next pass.
"""
from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp/word/document.xml")

TITLE = ("Unbounded Timing Terms Make a Composite Accident-Anticipation Score "
         "Video-Blind: Evidence from a Live Benchmark")

ABSTRACT = (
    "Traffic-accident anticipation is increasingly scored by composite metrics "
    "that add an earliness term — time-to-accident at a fixed risk threshold "
    "— to discrimination terms such as average precision and area under the "
    "ROC curve. We show that this composition fails structurally rather than "
    "incidentally. The discrimination terms are bounded on the unit interval "
    "while the earliness terms scale with clip length, and the earliness terms "
    "have a trivial global maximiser: a constant risk score above the threshold "
    "attains, on every clip, the largest value that clip admits. We demonstrate "
    "the consequence on a live benchmark. On the Zero-shot Accident "
    "Anticipation competition (AUTOPILOT; 1,417 clips curated from MM-AU; 13 "
    "teams), a constant risk score of 0.51, which reads no pixels, scores "
    "2.35291 on the private leaderboard — matching the eighth-placed team at "
    "the precision the leaderboard displays, exceeding five of thirteen teams "
    "including our own, and falling 0.14874 short of the winning score. Using "
    "twenty-four video-blind submissions as probes, designed so that unknown "
    "terms cancel under differencing, we then decompose the metric from outside "
    "the competition without access to any label: average precision and area "
    "under the curve together contribute 0.35105 to a blind submission, the two "
    "timing terms contribute 2.00186 — 5.70 times as much — and the score "
    "falls at exactly one sixtieth of a point per frame of delay, confirmed by a "
    "one-frame experiment. The same probes recover the support and mean of the "
    "withheld accident-onset distribution, and establish a second result: three "
    "curves that cross the threshold at the same frame, with identical values at "
    "the crossing, score 0.091 apart, a spread the published metric definition "
    "cannot produce. The benchmark's stated formula therefore does not reproduce "
    "its own scorer. We report our ninth-placed entry, a training-free ensemble "
    "that scores below the constant, as the case study, and close with a "
    "protocol whose central requirement is that every anticipation result be "
    "reported alongside a video-blind control."
)

CODE = "Code: https://github.com/sudaroli1/precrash"

KEYWORDS = ("Accident anticipation, evaluation methodology, benchmark design, "
            "video-blind baselines, composite metrics, time-to-accident, "
            "dashcam video analysis, intelligent transportation systems.")

INTRO = [
    ("p", "Road traffic injury kills approximately 1.19 million people every "
          "year [1], and leaves far more with permanent disability. A large "
          "share of those deaths are in principle foreseeable in the seconds "
          "before they occur, which is what makes anticipation — as opposed "
          "to detection after the fact — worth pursuing as an obligation and "
          "not only as a research problem [2]. Modern vehicles already carry the "
          "necessary instrument. A dashcam holds the primary visual record of "
          "the driving environment, including the sudden lane change, the "
          "pedestrian stepping off the kerb, and the hard braking of the vehicle "
          "ahead [3]. That record is used almost entirely in retrospect, to "
          "establish what happened after a collision. The question motivating "
          "this work was whether it could be used prospectively instead."),

    ("p", "We set out to answer it with a training-free system: a zero-shot "
          "ensemble that scores each frame of a dashcam clip for collision risk "
          "by combining vision–language similarity against danger and safety "
          "prompts, a motion-energy signal, and a text-anchored score, with no "
          "supervision from accident labels at any stage. We entered it in a "
          "public competition, the Zero-shot Accident Anticipation challenge "
          "hosted by AUTOPILOT, where it was scored by the organisers against "
          "ground truth we never held. It placed ninth of thirteen."),

    ("p", "This paper is not about that system. It is about what we found when "
          "we tried to understand the score it received."),

    ("p", "The competition scores submissions with a composite metric: a "
          "weighted sum of average precision (AP), area under the ROC curve "
          "(AUC), and two earliness measures — time-to-accident (TTA) and "
          "stable time-to-accident (STTA) — each evaluated at a risk threshold "
          "of 0.5. The composition is intuitive, and it is becoming common. A "
          "useful anticipator should both separate accident clips from normal "
          "driving and raise the alarm early, and summing terms that measure "
          "each seems a reasonable way to ask for both."),

    ("p", "It is not. AP and AUC are bounded on the unit interval. TTA and STTA "
          "are measured in frames and are bounded only by the length of the "
          "clip, which is 150 frames here. Summing a bounded quantity with an "
          "unbounded one means the unbounded one decides the ranking, and on "
          "this benchmark the arithmetic is not subtle: the timing terms are "
          "worth 5.70 times everything the discrimination terms contribute to a "
          "submission that reads nothing."),

    ("p", "Worse, the timing terms have a trivial global maximiser. TTA is the "
          "largest interval between the accident frame and any frame at which "
          "predicted risk exceeds the threshold; STTA is the largest such "
          "interval over which risk stays above the threshold without "
          "interruption. A constant risk score of 0.51 crosses the threshold at "
          "frame 0 and never falls, so it attains, on every clip, exactly the "
          "maximum value that clip admits, for both terms simultaneously. No "
          "method that reads the video can exceed it there. A method can only "
          "match it, and then compete on AP and AUC, which we show below are "
          "worth 0.35105 to a blind submission."),

    ("p", "So we submitted the constant. It scored 2.35291 against a winning "
          "score of 2.50165, matching the eighth-placed team at the precision "
          "the leaderboard displays."),

    ("p", "The major contributions of this work are as follows."),

    ("b", "A measured demonstration on a live benchmark. A video-blind constant "
          "matches the eighth-placed private score and exceeds those of five of "
          "thirteen teams, including our own. Every leaderboard score reported "
          "here was computed by the competition's own scoring service against "
          "ground truth we do not have; we implemented none of the metrics. All "
          "derived quantities are our arithmetic on those scores, and are marked "
          "as such."),

    ("b", "A black-box decomposition of a composite metric. We introduce a probe "
          "methodology — families of video-blind submissions constructed so "
          "that unknown terms cancel under differencing — and use it to "
          "separate a metric whose weights are undisclosed and whose labels are "
          "withheld. It recovers the discrimination floor, both timing terms, "
          "the score's slope in the crossing frame, and the support and mean of "
          "the accident-onset distribution."),

    ("b", "Evidence that the published metric definition does not reproduce the "
          "scorer. Three curves that cross the threshold at the same frame, with "
          "identical values at the crossing and no subsequent dip, receive "
          "scores spanning 0.091. Under the stated formula they must be equal. "
          "An entrant therefore cannot compute this benchmark's score from its "
          "published definition even given the labels, which is a reportable "
          "fact about the benchmark independent of everything above."),

    ("b", "A case study in which we are the subject, and a corrected protocol. "
          "Our own entry scores below the constant. We report this rather than "
          "omit it, because it is the clearest available illustration that a "
          "composite score of this shape does not measure what its name "
          "suggests. From the mechanisms we derive six requirements, of which "
          "the first — report a video-blind control with every anticipation "
          "result — costs an afternoon and would have caught this."),

    ("p", "We wish to be explicit about scope, because this paper reports a "
          "number that sits beside other teams' numbers. We make no claim about "
          "the validity, quality or honesty of any other submission or published "
          "method. We did not re-run anyone's system, we do not have anyone's "
          "code, and we could not evaluate it if we did, because the labels are "
          "withheld from every entrant. Where we observe that a leaderboard "
          "score is numerically identical to a control's, we are stating an "
          "arithmetic fact about two published numbers and drawing exactly one "
          "inference from it: that the metric assigns those two submissions the "
          "same value. We do not infer what any submission contained, and we "
          "note that a whole family of curves receives that identical score, "
          "which is precisely the property under examination. The claim of this "
          "paper is about a metric, demonstrated on one benchmark. It is not a "
          "claim about a field, and not a claim about anybody's method."),

    ("p", "The rest of the paper is organised as follows. Section 2 reviews "
          "accident anticipation and the evaluation practice this work "
          "examines. Section 3 describes the benchmark and states its metric "
          "precisely. Section 4 defines the video-blind control family and the "
          "probe methodology. Section 5 reports the measurements. Section 6 "
          "analyses why the composition fails. Section 7 sets out the corrected "
          "protocol, Section 8 the limitations, and Section 9 concludes."),
]

MOVED_HEADING = ("MATERIAL MOVED FROM THE INTRODUCTION — to be reintegrated "
                 "into Section 2, not for submission in this position")


def set_text(p, text):
    """Replace a paragraph's runs with a single run carrying `text`.

    Keeps the paragraph properties and the formatting of its first run, so the
    document's styles are untouched.
    """
    runs = p.findall(W + "r")
    template = copy.deepcopy(runs[0]) if runs else None
    for r in runs:
        p.remove(r)
    # drop any stray hyperlinks or bookmarks that carried the old text
    for tag in ("hyperlink", "bookmarkStart", "bookmarkEnd", "proofErr"):
        for el in p.findall(W + tag):
            p.remove(el)

    if template is None:
        run = etree.SubElement(p, W + "r")
    else:
        run = template
        for t in run.findall(W + "t"):
            run.remove(t)
        p.append(run)
    t = etree.SubElement(run, W + "t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


def main():
    if not DOC.exists():
        raise SystemExit(f"missing {DOC}; unzip the docx into unp/ first")

    tree = etree.parse(str(DOC))
    body = tree.getroot().find(W + "body")
    paras = [c for c in body if c.tag == W + "p"]

    def para(i):
        return paras[i]

    # ---- templates, captured before anything is removed ----
    body_tmpl = copy.deepcopy(para(24))     # a plain body paragraph
    bullet_tmpl = copy.deepcopy(para(34))   # a ListParagraph bullet
    head_tmpl = copy.deepcopy(para(21))     # Heading1
    blank_tmpl = copy.deepcopy(para(23))    # the empty spacer between paragraphs

    # ---- single-paragraph replacements ----
    set_text(para(2), TITLE)
    set_text(para(15), ABSTRACT)
    set_text(para(17), CODE)
    set_text(para(20), KEYWORDS)
    print("replaced: title, abstract, code link, keywords")

    # ---- the introduction: paragraphs 22..40 ----
    old_intro = [para(i) for i in range(22, 41)]
    keep_for_section2 = [copy.deepcopy(para(i)) for i in (24, 26, 28, 30)]
    anchor = old_intro[0]
    idx = list(body).index(anchor)

    for p in old_intro:
        body.remove(p)

    # The document separates body paragraphs with an empty paragraph rather
    # than with paragraph spacing. Match that, or the new introduction reads
    # denser than every other section. Bullets in a list are not separated.
    new_nodes = []
    for n, (kind, text) in enumerate(INTRO):
        tmpl = bullet_tmpl if kind == "b" else body_tmpl
        new_nodes.append(set_text(copy.deepcopy(tmpl), text))
        nxt = INTRO[n + 1][0] if n + 1 < len(INTRO) else None
        if kind != "b" and nxt != "b" and nxt is not None:
            new_nodes.append(copy.deepcopy(blank_tmpl))
    for offset, node in enumerate(new_nodes):
        body.insert(idx + offset, node)
    print(f"replaced introduction: {len(old_intro)} paragraphs -> {len(new_nodes)}")

    # ---- park the displaced background material at the end ----
    sect = body.find(W + "sectPr")
    tail = [set_text(copy.deepcopy(head_tmpl), MOVED_HEADING)] + keep_for_section2
    for node in tail:
        if sect is not None:
            sect.addprevious(node)
        else:
            body.append(node)
    print(f"parked {len(keep_for_section2)} background paragraphs at the end")

    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
