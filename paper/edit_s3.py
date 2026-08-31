"""
Pass 3: a new Section 3 (the benchmark and its metric), corrections inside the
authors' methods section, and the roadmap in Section 1.

Nothing the authors wrote is deleted. Their "Materials and Methods" becomes
Section 4 and keeps all of its text; where a sentence is wrong it is rewritten
in place, and where a claim was never measured it is withdrawn in the text
rather than quietly dropped, so the record of what changed is in the paper.

The substantive corrections, and why:

  * The temporal-compression stage. As implemented, the resampling evaluates
    the curve at index alpha*t and emits it at index t, so frame t reports the
    score computed from frame 1.3t. That is where the "mean gain of 6 frames in
    TTA" came from: the six frames are read from the future. Equation (8) as
    printed says t/alpha, which is neither what the code does nor what the
    reported 28 -> 22 crossover shift requires.

  * The monotone clamp forces the exact condition STTA tests, so any STTA
    compliance figure measures the clamp.

  * The 90%-of-RAFT-at-1%-of-the-cost figure was never measured. It appeared
    twice and is withdrawn in both places.

  * gamma could not have been selected by maximising average precision, which
    no entrant can compute on this corpus.

  * Table 3 selected a sentence encoder by earliest average crossover frame --
    the quantity a constant 0.51 minimises to zero. The table stays; the
    caveat is now under it.

  * Preprocessing: the corpus ships JPEG frames, not 1080p video, and carries
    no audio. Homography stabilisation is described in the text but is not in
    the released code.
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp4/word/document.xml")

# ---------------------------------------------------------------- inline fixes
# (unique fragment to find, replacement for that fragment)
FIXES = [
    # --- Section 1 roadmap ---
    ("Section 3 describes the benchmark and states its metric precisely. "
     "Section 4 defines the video-blind control family and the probe "
     "methodology.",
     "Section 3 states the benchmark and its metric exactly, and establishes "
     "what an entrant can and cannot compute from them. Section 4 describes the "
     "system that was entered, the video-blind control family, and the probe "
     "design."),

    # --- section title ---
    ("Materials and Methods:",
     "Method: the system under study, the video-blind controls, and the probes"),

    # --- problem statement ---
    ("The earlier the risk crosses 0.5, the better the score.",
     "The earlier the risk crosses 0.5, the better the score. That last "
     "sentence is not a design preference; it is a literal description of the "
     "official metric, and it is the subject of this paper. Section 5 shows "
     "that a score which crosses at frame 0 and never falls collects almost "
     "all of what the metric awards."),

    ("Stable Time to Accident (STTA) @0.5 must be optimized simultaneously.",
     "Stable Time to Accident (STTA) @0.5 must be optimized simultaneously. "
     "None of the four can be computed by an entrant, because the corpus "
     "publishes no labels (Section 3.3); all four are computed by the "
     "organisers alone."),

    # --- input and preprocessing ---
    ("that are encoded at 1080 p with frame rate as 30 FPS and the video "
     "duration is of 5 seconds with the total number of frames as 150. The "
     "video sequences are multimodal possessing audio and metadata information "
     "in the form of weather, road type etc.",
     "at 30 frames per second, five seconds in duration, 150 frames in total. "
     "It is distributed as a directory of JPEG frames rather than as an encoded "
     "video file, and it carries no audio. Alongside the frames the "
     "distribution supplies driver gaze maps and one short text caption per "
     "clip; the scene categories and the weather and lighting conditions are "
     "described in the competition documentation rather than supplied as "
     "per-clip fields."),

    ("Clip based segmentation is performed to remove the precursor straddles "
     "and with regard to the optical flow model, to address the camera shake "
     "associated with road bumps, the consecutive frames are feature mapped "
     "with derotation and detranslation and thus homography based warping is "
     "performed.",
     "Homography-based stabilisation against camera shake from road bumps, by "
     "derotating and detranslating consecutive frames, was specified in the "
     "design but is not present in the released code. The motion signal is "
     "therefore computed on unstabilised frames and camera shake remains in "
     "it. We state this because the frame-difference proxy described below is "
     "sensitive to exactly that."),

    # --- the 90% / 1% claim, first occurrence ---
    ("With frame difference, 90% of RAFT accident detection can be computed at "
     "1% of the compute cost. Due to this lower computation cost, frame "
     "difference is computed as mentioned in equation (1) using optical flow.",
     "The trade-off between the two was never measured. An earlier version of "
     "this work asserted that frame differencing retains about 90% of RAFT's "
     "sensitivity at about 1% of its cost; that figure was an estimate, no "
     "ablation was run, and it is withdrawn here. What can be said is that "
     "frame differencing was chosen because RAFT [10] did not fit the compute "
     "budget available, and that the substitution is an untested "
     "approximation. Frame difference is computed as follows."),

    # --- the 90% / 1% claim, second occurrence ---
    ("This proxy achieves approximately 90% of RAFT-equivalent accident "
     "detection sensitivity at roughly 1% of the GPU computational overhead, "
     "making it suitable for high-throughput zero-shot inference over the "
     "1,417-clip evaluation corpus.",
     "The substitution was made for compute reasons and its cost in "
     "sensitivity was never measured; the figure of 90% of RAFT at 1% of the "
     "cost, asserted in an earlier version of this work, is withdrawn."),

    # --- Gaussian smoothing ---
    ("This gaussian kernel averages the precision and area under the ROC "
     "contributing towards the preservation of the shape of the safe "
     "transitions.",
     "The Gaussian kernel suppresses frame-to-frame noise arising from "
     "inter-frame appearance variation, so that a single bright, blurred or "
     "occluded frame cannot by itself trigger a threshold crossing. It does "
     "not act on average precision or on the area under the ROC curve, which "
     "are computed from clip-level scores by the organisers; an earlier "
     "version of this sentence said otherwise and was wrong."),

    # --- gamma selection ---
    ("The optimal was selected by maximising AP over a grid with step 0.1",
     "The exponent was selected over a grid in steps of 0.1. It could not have "
     "been selected by maximising average precision, as an earlier version of "
     "this text stated, because average precision is not computable on this "
     "corpus by an entrant (Section 3.3). The grid was scored against the "
     "crossover-frame proxy, with the consequences discussed under Table 3."),

    # --- monotone clamping ---
    ("and directly maximises the Score-Triggered Time-to-Action (STTA) metric "
     "by eliminating false recoveries that would delay alarm generation.",
     "and it also, by construction, forces the exact condition that the "
     "official Stable Time-to-Accident metric tests. STTA@0.5 requires the risk "
     "score to remain above 0.5 continuously from the alarm to the accident; "
     "the clamp makes falling below 0.5 impossible once the score has crossed. "
     "Any subsequent report of STTA compliance therefore measures that the "
     "clamp executed, and nothing about the video. We retain the stage, and "
     "Section 5 reports every metric with it disabled as well as enabled. Note "
     "also that the metric's name is Stable Time-to-Accident; an earlier "
     "version of this text expanded STTA as Score-Triggered Time-to-Action, "
     "which is not what the benchmark defines."),

    # --- temporal compression ---
    ("Compression shifts the decision-boundary crossover from to frames, "
     "yielding a mean gain of 6 frames in Time-to-Alarm (TTA). The value was "
     "selected from the Pareto front of TTA gain versus false-positive rate "
     "increase. Note that Stage 5 must precede Stage 4 in implementation: "
     "applying temporal compression after clamping would violate the monotone "
     "guarantee established in Eq. (5).",
     "This stage requires a correction and a warning. As implemented, the "
     "resampling evaluates the curve at index alpha times t and emits it at "
     "index t, so the score reported for frame t is the score computed from "
     "frame 1.3t — a frame that has not yet been observed. Equation (8) as "
     "printed reads t divided by alpha; the implementation, and the reported "
     "shift of the crossover from about frame 28 to about frame 22, both "
     "correspond to alpha times t, and the equation should be corrected "
     "accordingly. This is the origin of the reported mean gain of six frames "
     "in time-to-accident: the six frames are read from the future. The "
     "operation cannot run in a deployed system, where frame 1.3t does not "
     "exist at time t, and every time-to-accident figure produced with alpha "
     "not equal to 1 is therefore unreportable. The released configuration "
     "sets alpha to 1.0, disabling the stage. It is documented here because it "
     "is one of the mechanisms Section 6 identifies, and because we introduced "
     "it ourselves."),

    # --- Table 2 cell ---
    ("VLMs process 1 frame in ~2-5 seconds — 1417 clips × 150 frames would "
     "take days",
     "Autoregressive decoding per frame; over 1417 clips × 150 frames the cost "
     "exceeded the compute available. No per-frame latency was measured here."),

    # --- Table 3 cells and caption ---
    ("Table 3: Justification showing the importance of NLP Caption model for "
     "Semantic Scoring",
     "Table 3: Sentence encoders compared by average crossover frame — the "
     "locally computable proxy, not accuracy. Read with the caveat below."),

    ("BEST — earliest and most accurate predictions",
     "Earliest crossover of the four"),

    ("WORST — biggest model gave worst results",
     "Latest crossover of the four"),
]

OLD_TABLE3_PARA = (
    "From Table 3, we can justify that the larger NLP models have performed "
    "worse for the accident anticipation."
)

NEW_TABLE3_PARA = (
    "Table 3 needs a caveat that only became clear later, and it is the reason "
    "this paper exists. The column on which those four encoders were compared "
    "is the average crossover frame: the mean index at which the ensemble's "
    "risk score first exceeds 0.5. That quantity was chosen because it is the "
    "only thing computable locally on a corpus that publishes no labels "
    "(Section 3.3). It is not accuracy, and the selection it drove cannot be "
    "described as choosing the most accurate encoder. Worse, it improves "
    "monotonically as the crossing moves earlier, and it is minimised "
    "absolutely — to frame 0 — by a constant risk score of 0.51 that reads no "
    "pixels at all. The selection recorded in Table 3 was therefore made "
    "against a criterion that a degenerate submission optimises perfectly. We "
    "keep the table because it is an honest record of how the choice was made, "
    "and because Section 5 measures what that criterion is worth. The stated "
    "explanation for the ordering — that smaller sentence encoders suit these "
    "short, domain-specific captions better than large general-purpose ones — "
    "remains plausible, but this experiment does not establish it."
)

# ------------------------------------------------------------- new Section 3
SECTION3_HEADING = "THE BENCHMARK AND ITS METRIC"

SECTION3 = [
    ("h", "3.1. The competition"),
    ("p", "The measurements in this paper were made on Zero-shot Accident "
          "Anticipation [8], a public competition hosted on Kaggle by "
          "AUTOPILOT. It ran from 17 February to 15 April 2026 and drew 59 "
          "entrants, 20 participants and 13 teams across 194 submissions. Late "
          "submission remains open and is scored against the same data splits, "
          "which is how the experiments of Section 5 were run after the closing "
          "date. Every figure in this section was read from the competition's "
          "Overview, Evaluation, Data and Leaderboard pages on 31 August 2026."),
    ("p", "Two properties make it a suitable instrument for this study. The "
          "organisers hold the labels and compute every score, so nothing "
          "reported here depends on our own implementation of a metric. And a "
          "submission is a plain list of numbers per clip, so it need not "
          "contain a model at all — which is what makes a video-blind control "
          "submittable in the first place."),

    ("h", "3.2. What the corpus provides"),
    ("p", "The corpus is a curated 1,417-clip subset of MM-AU [7]. Every clip "
          "is exactly 150 frames at 30 frames per second, five seconds long, "
          "and is distributed as a directory of JPEG frames rather than as an "
          "encoded video file. Alongside the frames the distribution supplies "
          "driver gaze maps and one short text caption per clip. The "
          "competition documentation describes five scene categories — "
          "highway, urban, rural, mountainous and tunnel — together with rare "
          "weather (rain, snow, fog) and lighting (night) conditions. There is "
          "no training split; the competition is inference-only by design, "
          "which is what the word zero-shot denotes in its title."),

    ("h", "3.3. What the corpus withholds, and what follows from it"),
    ("p", "The published test file carries five fields: an identifier, a video "
          "identifier, a start frame, an end frame, and a caption. There is no "
          "label, no accident time and no alert time. Ground truth is held by "
          "the organisers and exposed only through the leaderboard."),
    ("p", "One consequence deserves emphasis, because the rest of this paper "
          "turns on it. Average precision, area under the ROC curve, a "
          "false-positive rate, and any time-to-accident measured to a true "
          "onset are not computable by any entrant. An entrant who wants to "
          "measure progress locally must substitute something, and the only "
          "quantities computable from a submission alone are properties of the "
          "risk curve's own shape: the frame at which it crosses the threshold, "
          "whether it falls back afterwards, how steeply it rises. Section 4 "
          "records that we did exactly this — the encoder in Table 3 was chosen "
          "on crossover frame — and Section 5 measures what that proxy is "
          "worth."),
    ("p", "A second consequence is worth recording without overstating it. The "
          "captions describe outcomes. Seventy-nine distinct captions cover the "
          "1,417 clips, the most frequent being “lead vehicle stops” "
          "(146 clips), “a vehicle controls loss” (144) and "
          "“ego-car controls loss” (133). They state what happens, "
          "and they are supplied at inference time, so a method conditioned on "
          "them has access to the outcome it is being asked to anticipate. The "
          "system described in this section does not use them; it manufactures "
          "its own text anchors from a motion statistic. We identify the hazard "
          "and do not quantify it."),

    ("h", "3.4. The metric, as published"),
    ("p", "The evaluation page defines the score as a weighted average of four "
          "quantities:"),
    ("e", "score = w_AP · AP + w_AUC · AUC + w_TTA · TTA@0.5 + w_STTA · STTA@0.5"),
    ("p", "with the four weights stated to be fixed by the organisers and not "
          "disclosed. Positives are accident clips and negatives are normal "
          "driving clips; the area under the curve is given in its rank "
          "formulation. The two timing terms are defined at a risk threshold of "
          "0.5:"),
    ("e", "TTA@0.5 = max { t_ai − t_a  |  p_t > 0.5,  0 ≤ t_a ≤ t_ai }"),
    ("e", "STTA@0.5 = max { t_ai − t_a′  |  p_t > 0.5 for all t in [t_a′, t_ai] }"),
    ("p", "where t_ai is the accident start frame within the 150-frame clip and "
          "t_a is the first frame at which the risk score exceeds the "
          "threshold. STTA additionally requires the score to remain above the "
          "threshold continuously from t_a′ to t_ai."),
    ("p", "Three observations follow from these definitions alone, before any "
          "measurement is made."),
    ("p", "First, AP and AUC are bounded on the unit interval, while TTA and "
          "STTA are counted in frames and bounded only by t_ai, and hence by "
          "how the corpus was windowed. The four terms are not commensurable, "
          "and their sum inherits the scale of the larger pair."),
    ("p", "Second, both timing terms are maximised by the same trivial "
          "submission. A risk score that exceeds 0.5 at frame 0 and never falls "
          "has t_a = 0 and satisfies the continuity condition on every clip, so "
          "it attains TTA = STTA = t_ai, the largest value each clip admits. No "
          "submission of any kind can exceed it on either term; a submission "
          "can only match it, and then compete on AP and AUC."),
    ("p", "Third, neither timing term reads the magnitude of the score, only "
          "whether it stands above 0.5. A submission that is barely committed "
          "and one that is certain receive identical timing credit, so the "
          "metric offers no incentive to calibrate."),

    ("h", "3.5. Reproducibility of the submission path"),
    ("p", "Before spending a submission slot, we regenerated the organisers' "
          "sample submission file from its own parsed values and compared it "
          "byte for byte against the distributed file: identical, 1,417 rows of "
          "1,417. This fixes the identifier ordering, the list formatting and "
          "the float representation, and it means a scoring anomaly cannot be "
          "attributed to our file writer. Every submission reported in Section "
          "5 was additionally checked for length 150, for finiteness, and for "
          "range within [0, 1] before upload."),
]

# ------------------------------------------- control and probe subsections
CONTROLS = [
    ("h", "The video-blind control family"),
    ("p", "A video-blind submission is one whose risk curve is a function of "
          "the frame index alone: identical for every clip, computed without "
          "opening a single frame. We use a family rather than a single curve, "
          "because one flattering curve invites the objection that the "
          "comparison was chosen to flatter it. The family comprises a constant "
          "0.51 and a constant 0.99; a linear ramp from 0.001 to 0.999; "
          "logistic curves centred at frames 60, 75, 90 and 105; and step "
          "functions rising from 0.49 to 0.51 at frames 0, 10, 25, 50, 75, 100, "
          "125 and 140. Each is a single list of 150 numbers, repeated across "
          "all 1,417 clips."),

    ("h", "Probes that isolate the terms of the metric"),
    ("p", "The decomposition in Section 5 rests on one observation. Every "
          "video-blind submission is identical across clips, so — provided the "
          "scorer reduces each 150-value curve to one clip-level score by any "
          "deterministic function of that curve — every member of the family "
          "produces an all-way tie among clips and therefore earns identical "
          "average precision and identical area under the curve. Whatever those "
          "terms are worth, they are worth the same to all of them, and they "
          "cancel exactly in any difference between two members."),
    ("p", "That turns the leaderboard into an instrument. Three probes suffice. "
          "A curve held at 0.49 throughout never crosses the threshold, so both "
          "timing terms are zero and its score is the discrimination floor "
          "alone. A curve at 0.51 in frame 0 and 0.49 thereafter crosses once "
          "and then falls, so it earns the time-to-accident term but forfeits "
          "the stable one. A constant 0.51 earns both. Differencing the three "
          "gives each term separately. A fourth probe, holding 0.51 throughout "
          "except for a dip across frames 50 to 59, moves the stable term's "
          "reference point while leaving the first crossing untouched, and "
          "serves as a consistency check on the attribution."),
    ("p", "The step family measures the score's dependence on the crossing "
          "frame directly, which gives a second and independent route to the "
          "same quantity. Two routes to one number is what makes the result "
          "safe to report."),
]


def text_of(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def set_text(p, text):
    runs = p.findall(W + "r")
    template = copy.deepcopy(runs[0]) if runs else None
    for r in runs:
        p.remove(r)
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
    tree = etree.parse(str(DOC))
    root = tree.getroot()
    body = root.find(W + "body")

    # --- inline fixes, across every paragraph including those inside tables ---
    hits = {old: 0 for old, _ in FIXES}
    for p in root.iter(W + "p"):
        txt = text_of(p)
        for old, new in FIXES:
            if old in txt:
                txt = txt.replace(old, new)
                set_text(p, txt)
                hits[old] += 1
    for old, n in hits.items():
        flag = "OK" if n == 1 else f"!! {n} matches"
        print(f"  [{flag}] {old[:60].strip()}...")

    # --- the Table 3 discussion paragraph, replaced wholesale ---
    for p in root.iter(W + "p"):
        if OLD_TABLE3_PARA in text_of(p):
            set_text(p, NEW_TABLE3_PARA)
            print("  [OK] Table 3 discussion replaced")
            break
    else:
        print("  [!!] Table 3 discussion paragraph not found")

    # --- templates ---
    paras = [c for c in body if c.tag == W + "p"]
    sec_head_tmpl = None
    body_tmpl = None
    sub_head_tmpl = None
    for p in paras:
        t = text_of(p)
        if sec_head_tmpl is None and t.strip() == "RELATED WORK AND EVALUATION PRACTICE":
            sec_head_tmpl = copy.deepcopy(p)
        if sub_head_tmpl is None and t.strip().startswith("2.1."):
            sub_head_tmpl = copy.deepcopy(p)
        if body_tmpl is None and t.startswith("Vision-based accident anticipation"):
            body_tmpl = copy.deepcopy(p)
    if not all((sec_head_tmpl, body_tmpl, sub_head_tmpl)):
        raise SystemExit("could not locate the Section 2 templates")

    tmpl = {"h": sub_head_tmpl, "p": body_tmpl, "e": body_tmpl}

    # --- insert Section 3 before the authors' methods heading ---
    anchor = None
    for p in paras:
        if text_of(p).strip().startswith("Method: the system under study"):
            anchor = p
            break
    if anchor is None:
        raise SystemExit("could not find the methods heading")

    at = list(body).index(anchor)
    nodes = [set_text(copy.deepcopy(sec_head_tmpl), SECTION3_HEADING)]
    for kind, txt in SECTION3:
        nodes.append(set_text(copy.deepcopy(tmpl[kind]), txt))
    for off, n in enumerate(nodes):
        body.insert(at + off, n)
    print(f"  [OK] inserted Section 3 ({len(nodes)} paragraphs)")

    # --- append the control and probe subsections at the end of the method ---
    tail_anchor = None
    for p in root.iter(W + "p"):
        if "numerical stability" in text_of(p):
            tail_anchor = p
    if tail_anchor is None:
        raise SystemExit("could not find the end of the post-processing stage")
    at = list(body).index(tail_anchor)
    nodes = [set_text(copy.deepcopy(tmpl[k]), t) for k, t in CONTROLS]
    for off, n in enumerate(nodes, start=1):
        body.insert(at + off, n)
    print(f"  [OK] appended {len(nodes)} paragraphs on controls and probes")

    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
