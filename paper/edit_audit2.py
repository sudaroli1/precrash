"""
Pass 17: the second audit pass, mirrored into the manuscript.

Applies to the Word document the corrections that pass 2 of the audit made to
the TMLR build, so the two forms continue to say the same thing. Three are
substantive and the rest are self-consistency; the reasoning is recorded in
`latex/fixes2.py` and in the project note, so this file only lists what moves.

Not mirrored: the equation and typesetting repairs. Those were damage from the
Word-to-LaTeX conversion, and Word renders the originals correctly.

  python paper/edit_audit2.py
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp9/word/document.xml")

# (anchor in the paragraph, old substring, new substring)
FIXES = [
    # ---------------- abstract ----------------
    ("Traffic-accident anticipation is increasingly scored",
     "Using twenty-four video-blind submissions as probes",
     "Using twenty-five video-blind submissions as probes"),
    ("Traffic-accident anticipation is increasingly scored",
     "matching the eighth-place team at the precision",
     "matching the eighth-placed team at the precision"),
    ("Traffic-accident anticipation is increasingly scored",
     "(AUTOPILOT; 1,417 clips curated from MM-AU; 13 teams)",
     "(AUTOPILOT-COG; 1,417 clips curated from MM-AU; 13 teams)"),
    ("Traffic-accident anticipation is increasingly scored",
     "the score falls at exactly one sixtieth of a point per frame of delay",
     "the score falls at almost exactly one sixtieth of a point per frame of "
     "delay"),

    # ---------------- Section 1 ----------------
    ("Evidence that the published definition does not reproduce the scorer.",
     "Three curves crossing at the same frame, with identical values at the "
     "crossing, score 0.091 apart;",
     "Four curves crossing at the same frame, none of which dips below it "
     "afterwards, score 0.091 apart;"),
    ("A case study in which we are the subject",
     "and a six-point protocol.", "and a seven-point protocol."),

    # ---------------- Section 3 ----------------
    ("The measurements in this paper were made on Zero-shot Accident",
     "drew 59 entrants, 20 participants and 13 teams across 194 submissions.",
     "drew 59 entrants, of whom 20 went on to submit, forming 13 teams "
     "between them, across 194 submissions."),
    ("The evaluation page defines the score as",
     "a weighted average of four quantities:",
     "a weighted sum of four quantities:"),

    # ---------------- Section 4.1 ----------------
    ("The considered problem is anticipating accidents",
     "The considered problem is anticipating accidents without any training "
     "data and it is expected to generalize well only with pretrained "
     "knowledge.",
     "The considered problem is anticipating accidents without any training "
     "data, using pretrained knowledge alone."),

    # ---------------- Section 4.2 ----------------
    ("PreCrash is modular and decoupled.",
     "PreCrash is modular and decoupled. Three pretrained engines — a "
     "vision–language semantic engine, a motion engine and a text-anchored "
     "prior — run independently, sharing no weights or activations, and are "
     "combined only at a late fusion layer. Each is matched to its own "
     "representational domain, so no single model carries the whole burden of "
     "reasoning about appearance, motion and semantics at once, and any "
     "engine can be replaced without disturbing the others. That separation "
     "is also what makes the per-modality failure analysis of Section 5.6.1 "
     "possible.",
     "The system is modular in implementation. Three scoring engines — a "
     "vision–language semantic engine, a motion engine and a text-anchored "
     "temporal prior — produce per-frame curves independently, sharing no "
     "weights or activations, and are combined only at a late fusion layer, "
     "so any one can be replaced without disturbing the others. They are not, "
     "however, three independent views of the scene, and an earlier version "
     "of this work said they were. The text-anchored prior is a three-level "
     "quantisation of the same grayscale frame-difference statistic the "
     "motion engine uses, as the subsection on it below sets out, so two of "
     "the three curves are functions of one signal. We state it here because "
     "it is the correction that explains the ablation of Section 5.6.1."),

    ("The system takes a 150-frame clip",
     "motion scoring, caption prior, ensemble, and post-processing",
     "motion scoring, temporal prior, ensemble, and post-processing"),

    ("Visual scoring asks whether a frame resembles danger",
     "CLIP does this at zero shot because", "CLIP does this zero-shot because"),

    ("Vision and motion between them still do not answer",
     "Vision and motion between them still do not answer what is happening. A "
     "text prior adds that: captions translate the scene into language, which "
     "can be compared against high-risk action patterns, and sentence "
     "encoders trained on large image–text corpora carry enough of that "
     "structure to reason about danger without task-specific training. Table "
     "1 records how the encoder was chosen.",
     "The third engine was intended to answer what is happening rather than "
     "what is present or what is moving, by translating the scene into "
     "language and comparing it against high-risk action patterns. It does "
     "not do that, and the description is corrected here rather than the "
     "behaviour, so that the reported numbers remain reproducible. As "
     "implemented, it computes the mean grayscale inter-frame difference over "
     "the last third of the clip — the same statistic the motion engine uses "
     "— thresholds that scalar at 20 and at 10, and selects one of three "
     "fixed strings written into the source. The selected string is embedded "
     "with a sentence encoder and compared against two fixed text anchors, "
     "and a sigmoid over the frame index is parameterised by the two "
     "similarities. No frame is captioned, no image–text model is involved, "
     "and the encoder never sees anything but one of those three strings. The "
     "modality therefore has three reachable states across the whole corpus. "
     "Table 1 records how the encoder was chosen; Section 5.6.1 records what "
     "that choice was worth."),

    ("Table 1 needs a caveat that only became clear later",
     "The stated explanation for the ordering — that smaller sentence "
     "encoders suit these short, domain-specific captions better than large "
     "general-purpose ones — remains plausible, but this experiment does not "
     "establish it.",
     "The stated explanation for the ordering — that smaller sentence "
     "encoders suit these short, domain-specific captions better than large "
     "general-purpose ones — does not survive the correction above either: "
     "the four encoders were not compared on captions, but on which of three "
     "fixed strings the clip had been assigned, so what the column separates "
     "is four embeddings of the same three sentences."),

    ("Each engine alone is blind to what the others see.",
     "The caption prior cannot see the video at all, so identically worded "
     "captions receive identical trajectories. Ensembling is intended to let "
     "each engine cover the others' blind spot: the semantic score supplies "
     "what is in the scene, the motion proxy whether it is moving abnormally, "
     "and the caption prior when in the clip risk should rise.",
     "The temporal prior does not see the video at all, and has three "
     "reachable states, so every clip in the same motion bucket receives the "
     "same curve. Ensembling was intended to let each engine cover the "
     "others' blind spot: the semantic score supplying what is in the scene, "
     "the motion proxy whether it is moving abnormally, and the prior when in "
     "the clip risk should rise. It does not achieve the third of those, "
     "because the prior is a quantisation of the motion proxy rather than an "
     "independent reading of the clip."),

    ("NLP captioning model generates the zero-shot text caption",
     "NLP captioning model generates the zero-shot text caption for each "
     "frame. This model performs the scoring against sudden onset of danger "
     "and gradual onset of danger. Thus, the final score is estimated using "
     "the temporal offset and relative score trajectory as mentioned in "
     "equation (5)",
     "The temporal prior scores its selected string against a sudden-onset "
     "anchor and a gradual-onset anchor, and the two similarities parameterise "
     "a sigmoid over the frame index, as in equation (5)"),

    ("The varied weights are exhibited due to the variation",
     "The varied weights are exhibited due to the variation with the power of "
     "the models.",
     "The entered configuration used 0.55 on the semantic score, 0.25 on the "
     "motion proxy and 0.20 on the temporal prior. Those values were not "
     "chosen a priori; they were searched over the evaluation corpus, which "
     "Section 4.3 records as a limitation. The released configuration weights "
     "the three equally."),

    ("with . This enforces a hard causal constraint",
     "We retain the stage, and Section 5 reports every metric with it "
     "disabled as well as enabled.",
     "We retain the stage. Section 5 does not report the leaderboard score of "
     "a run with it disabled — we did not make that submission, and say so "
     "rather than imply otherwise — which is exactly the omission item 5 of "
     "the protocol in Section 7 asks others not to repeat."),

    ("where denotes linear interpolation onto the original frame grid",
     "the implementation, and the reported shift of the crossover from about "
     "frame 28 to about frame 22, both correspond to alpha times t, and the "
     "equation should be corrected accordingly. This is the origin of the "
     "reported mean gain of six frames in time-to-accident: the six frames "
     "are read from the future.",
     "the implementation, and the direction of the shift it was reported to "
     "produce, both correspond to alpha times t, and the equation should be "
     "corrected accordingly. An earlier version of this work attributed a "
     "mean gain of six frames in time-to-accident to this stage. That figure "
     "is withdrawn: no table in this paper measures the stage on its own. "
     "Table 12 separates C6 from C7 by 1.2 frames and bundles the compression "
     "with the clamp, so what the resampling bought is not identified "
     "anywhere in our records. The objection does not depend on the magnitude "
     "— whatever it bought was read from a frame that had not arrived."),

    # ---------------- Section 4.4 ----------------
    ("A video-blind submission is one whose risk curve",
     "The family comprises a constant 0.51 and a constant 0.99; a linear ramp "
     "from 0.000 to 1.000; logistic curves centred at frames 60, 75, 90 and "
     "105; and step functions rising from 0.49 to 0.51 at frames 0, 10, 25, "
     "50, 75, 100, 125 and 140. Each is a single list of 150 numbers, "
     "repeated across all 1,417 clips.",
     "The family comprises a constant 0.51 and a constant 0.99; a curve held "
     "at 0.49 throughout, which never crosses; a linear ramp from 0.000 to "
     "1.000; logistic curves centred at frames 60, 75, 90 and 105; step "
     "functions rising from 0.49 to 0.51 at frames 0, 10, 25, 50, 75, 100, "
     "125 and 140, and a calibration step at frame 76; a curve that crosses "
     "at frame 0 and drops back, and one that crosses and dips across frames "
     "50 to 59; and the two graded curves of Section 5.4. Twenty-five in all. "
     "Each is a single list of 150 numbers, repeated across all 1,417 clips."),

    # ---------------- Section 5.4 ----------------
    ("One result does not fit, and we report it rather than smooth it.",
     "A linear ramp crosses the threshold at frame 75, exactly as three other "
     "curves do, and scores 0.037 lower than they do.",
     "A linear ramp crosses the threshold at frame 75, exactly as a step "
     "function does, and scores 0.037 lower than the step."),

    ("Table 8: Four curves that all first exceed 0.5 at frame 75",
     "Table 8: Four curves that all first exceed 0.5 at frame 75, with "
     "identical values at frames 74 and 75 and no subsequent dip, plus a "
     "one-frame calibration step.",
     "Table 8: Four curves that all first exceed 0.5 at frame 75 and never "
     "fall below it afterwards, plus a one-frame calibration step. The three "
     "step-like curves hold identical values at frames 74 and 75."),

    ("The four curves at the head of Table 8",
     "The four curves at the head of Table 8 have identical values at frames "
     "74 and 75, cross at the same frame, and never fall back afterwards.",
     "The four curves at the head of Table 8 cross the threshold at the same "
     "frame and never fall back afterwards, and the three step-like ones hold "
     "identical values at frames 74 and 75."),

    # ---------------- Section 5.6 ----------------
    ("The modality ablation of Table 10 compares single-modality",
     "and the caption prior cannot see the video at all, assigning identical "
     "trajectories to identically worded captions.",
     "and the temporal prior cannot see the video at all, assigning identical "
     "trajectories to every clip that falls in the same motion bucket."),

    ("The modality ablation of Table 10 compares single-modality",
     "What the table does not establish is superiority in anticipation, "
     "because the quantity being compared is the one a constant minimises.",
     "What the table does not establish is superiority in anticipation, "
     "because the quantity being compared is the one a constant minimises — "
     "and the table makes that concrete, since the temporal prior alone posts "
     "an earlier mean crossover, 21.7 against 22.7, than the full ensemble "
     "does. That is not evidence that the prior anticipates better. It has "
     "three reachable states across the corpus, so it emits nearly the same "
     "curve everywhere, and a nearly constant curve is well matched to a "
     "benchmark whose events sit at a nearly constant position."),

    ("Table 12 compares seven pipeline configurations.",
     "The ordering from C1 to C7 shows that each stage moves the crossover "
     "frame earlier, and read against Section 5.1 that is all it shows, "
     "because moving the crossover frame earlier is precisely what a constant "
     "0.51 does perfectly. Two stages deserve particular note in that light. "
     "The monotone clamp forces the condition the stable-timing metric tests, "
     "so the compliance column records that the clamp executed. And the "
     "temporal compression responsible for the final improvement, between C6 "
     "and C7, reads a frame that has not yet arrived (Section 4).",
     "Each post-processing stage added to the raw ensemble moves the "
     "crossover frame earlier, from C4 at 25.4 to C7 at 22.7; the "
     "single-modality rows C1 to C3 are not steps in that sequence and do not "
     "order monotonically with it. Read against Section 5.1 that is all the "
     "table shows, because moving the crossover frame earlier is precisely "
     "what a constant 0.51 does perfectly — and C3, the three-state temporal "
     "prior alone, already beats the full pipeline on it at 21.7, for the "
     "reason Section 5.6.1 gives. Two stages deserve particular note. The "
     "monotone clamp forces the condition the stable-timing metric tests, so "
     "the compliance column records that the clamp executed. And the "
     "C6-to-C7 step bundles the clamp with the temporal compression, which "
     "reads a frame that has not yet arrived (Section 4.2), so neither the "
     "1.2 frames between them nor the compliance flag beside them is "
     "attributable to a single stage."),

    # ---------------- Section 5.7 ----------------
    ("The results in this section admit one reading",
     "The 4.23-second average warning horizon",
     "The 4.24-second average warning horizon"),

    # ---------------- Section 7 ----------------
    ("Publish the score of a released control alongside the leaderboard.",
     "what took us twenty-four submissions:",
     "what took us twenty-five submissions:"),

    ("Non-causal timing gain.",
     "Our Stage 5 did this at α = 1.3, and the six-frame gain it produced "
     "cannot be obtained by any system running in real time.",
     "Our Stage 5 did this at a compression factor of 1.3, and whatever "
     "earliness it bought cannot be obtained by any system running in real "
     "time."),

    # ---------------- Section 8 ----------------
    ("One discrepancy in the crossing-frame model is open.",
     "One discrepancy in the crossing-frame model is open. A linear ramp "
     "scores 0.037 below three curves that cross at the same frame (Section "
     "5.4),",
     "One discrepancy in the crossover-frame model is open. A linear ramp "
     "scores 0.037 below a step function that crosses at the same frame, and "
     "0.029 above another curve that does (Section 5.4),"),

    ("We do not demonstrate the corrected protocol on a labelled corpus.",
     "Items 2 to 5 of Section 7 are derived from the mechanisms rather than "
     "shown end to end,",
     "The protocol's recommendations on bounding and conditioning the "
     "earliness term, on stating a timing metric's reference point, on "
     "reporting discrimination beside it and on reporting metrics with "
     "post-processing disabled are derived from the mechanisms rather than "
     "shown end to end,"),

    # ---------------- Section 9 and compliance ----------------
    ("The method that established this is as portable as the finding.",
     "Twenty-four submissions, not one of which opened a video frame, were "
     "enough",
     "Twenty-five submissions, not one of which opened a video frame, were "
     "enough"),

    ("All submissions reported here were made through the competition's own",
     "Twenty-four submissions were made in total, across two days, against a "
     "stated limit of one hundred per day.",
     "Twenty-six submissions were made in total, all on 31 August 2026, "
     "against a stated limit of one hundred per day: twenty-five video-blind "
     "curves we generated, and the organisers' own sample submission "
     "resubmitted verbatim as a control on the submission path."),

    ("The submission files, the code that generated them,",
     "the scripts that produce both figures are released. Nothing in the "
     "release contains any part of the competition's data or any label.",
     "the scripts that produce both figures accompany this submission as "
     "anonymised supplementary material, and will be released publicly on "
     "acceptance. Nothing in them contains any part of the competition's data "
     "or any label."),

    # ---------------- table captions ----------------
    ("Table 9: Overall Zero-Shot Anticipation Performance",
     "Table 9: Overall Zero-Shot Anticipation Performance on MM-AU "
     "(N = 1,417 clips)",
     "Table 9: Curve-shape statistics of the ensemble over the 1,417-clip "
     "corpus. Not anticipation results: read with the framing above."),

    ("Table 12: Seven-Configuration Performance Comparison",
     "Table 12: Seven-Configuration Performance Comparison",
     "Table 12: Seven pipeline configurations, compared on mean crossover "
     "frame — a statistic of curve shape, not performance."),
]

# whole-paragraph table cells
CELLS = [
    ("Best performance; all modality blind spots mutually compensated",
     "The three blind spots compensate one another; the crossover frame is "
     "nonetheless later than the prior alone"),
    ("Best performance; full STTA compliance",
     "Earliest crossover of the ensembled configurations; STTA compliance by "
     "construction"),
    ("Temporal ceiling; identical t₀ values for clips with similar captions "
     "— no per-clip visual personalisation",
     "Three reachable states for the whole corpus, so clips in the same "
     "motion bucket receive identical curves; no per-clip visual detail"),
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
    bad = [FIXES[i][1][:52] for i, n in hits.items() if n != 1]
    print(f"  [OK] {len(FIXES) - len(bad)} of {len(FIXES)} prose fixes"
          + ("\n       FAILED: " + "\n               ".join(bad) if bad else ""))

    cells = {old: 0 for old, _ in CELLS}
    for p in paras:
        txt = text_of(p).strip()
        for old, new in CELLS:
            if txt == old:
                set_text(p, new)
                cells[old] += 1
    bad2 = [o[:52] for o, n in cells.items() if n != 1]
    print(f"  [OK] {len(CELLS) - len(bad2)} of {len(CELLS)} table cells"
          + ("   FAILED " + str(bad2) if bad2 else ""))

    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8",
               standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
