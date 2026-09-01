"""
Pass 11: reduce Section 4 from 4,008 words to about 2,970.

The cut falls almost entirely on the case made for the architecture. Roughly a
thousand words argued for design choices — three bulleted lists of what each
modality cannot do, four bullets on why a text prior helps, two paragraphs
restating the architecture overview a third time — for a system that is no
longer this paper's contribution. Table 2 and Table 4 already make most of that
case, and the ablation in 5.6.1 makes the rest with numbers.

Protected and untouched: the Table 3 caveat, the Gaussian and gamma
corrections, the clamp paragraph, the temporal-compression paragraph, the
fusion-weight disclosure, every equation and its gloss, and the control and
probe subsections. Those are the load-bearing exhibits.
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp6/word/document.xml")

REPLACE = [
    ("The development of pre-crash systems is predicated on the transition",
     "Pre-crash systems differ from reactive ones in that the prediction must "
     "precede the event. This section states the problem, describes the system "
     "that was entered, and defines the video-blind controls and the probes "
     "with which the metric was later measured."),

    ("Where  denotes the predicted probability score for accident occurrence",
     "Where ρ(t) is the predicted probability of an accident, computed without "
     "observing one. The risk score should begin near 0 and rise toward 1 "
     "before the collision, and the earlier it crosses 0.5 the better the "
     "official score. That last clause is not a design preference but a literal "
     "description of the metric, and it is this paper's subject: Section 5 "
     "shows that a score crossing at frame 0 and never falling collects almost "
     "all of what the metric awards."),

    ("2.     The variety exhibited in the video sequences was diverse",
     "2. The clips are diverse in accident type and in condition — lane "
     "changes, rear-end collisions, red-light violations, pedestrians; rain, "
     "fog, night, tunnels, mountain roads — and no training data is available "
     "to adapt to any of them."),

    ("4.      Evaluation metrics such as Area under Precision-recall",
     "4. The four official metrics — AP, AUC, TTA@0.5 and STTA@0.5 — must be "
     "optimised together, and none of them can be computed by an entrant, "
     "because the corpus publishes no labels (Section 3.3)."),

    ("Precrash is designed as a modular and decoupled architecture",
     "PreCrash is modular and decoupled. Three pretrained engines — a "
     "vision–language semantic engine, a motion engine and a text-anchored "
     "prior — run independently, sharing no weights or activations, and are "
     "combined only at a late fusion layer. Each is matched to its own "
     "representational domain, so no single model carries the whole burden of "
     "reasoning about appearance, motion and semantics at once, and any engine "
     "can be replaced without disturbing the others. That separation is also "
     "what makes the per-modality failure analysis of Section 5.6.1 possible."),

    ("The architecture, as shown in figure 1, adopts a six-stage pipeline",
     "The system takes a 150-frame clip, dispatches it to the three engines in "
     "parallel with no inter-stream communication, and combines their per-frame "
     "curves in a six-stage pipeline: frame loading, semantic scoring, motion "
     "scoring, caption prior, ensemble, and post-processing (Figure 1)."),

    ("The input to zero shot anticipation is typically a raw dash cam",
     "Each clip arrives as 150 JPEG frames at 30 frames per second, five "
     "seconds long, with no audio; the distribution also supplies driver gaze "
     "maps and one caption per clip. Frames are resized to 224×224 with bicubic "
     "interpolation, the upper 20% and lower 8% are stripped to remove sky and "
     "bonnet, and per-frame normalisation handles exposure variation — which "
     "matters because a zero-shot pipeline cannot adapt to a distribution "
     "shift. Homography-based stabilisation against camera shake was specified "
     "in the design but is absent from the released code, so the motion signal "
     "is computed on unstabilised frames and the shake remains in it. That "
     "matters for the frame-difference proxy described below."),

    ("The framework suggests an ensembled approach for Zero shot",
     "The three engines were chosen by trial rather than by training, since no "
     "training data exists. The reason for each is given below."),

    ("One of the primary requirements in the accident anticipation is to "
     "perform the visual scoring",
     "Visual scoring asks whether a frame resembles danger — which requires "
     "reading the scene semantically rather than as pixels. CLIP does this at "
     "zero shot because it was pretrained on 400 million image–text pairs and "
     "can therefore compare a frame against a danger prompt with no "
     "task-specific training. ViT-L/14 was chosen over the smaller variants for "
     "its finer spatial detail. Table 2 sets out why the alternatives were "
     "rejected."),

    ("Another important question that gains importance in accident anticipation",
     "Motion asks a different question: are objects moving as normal physics "
     "would predict, or not? Accidents are generally preceded by kinematic "
     "abnormality — a pedestrian stepping out, a sharp deceleration, an "
     "unexpected object. The RAFT estimator of Teed and Deng [10] is the more "
     "accurate way to capture this, but it did not fit the compute budget "
     "available, so a frame-difference proxy was used instead. The trade-off "
     "was never measured: an earlier version of this work asserted that frame "
     "differencing retains about 90% of RAFT's sensitivity at about 1% of its "
     "cost, and that figure was an estimate with no ablation behind it. It is "
     "withdrawn. Frame difference is computed as follows."),

    ("Vision and the motion models though contribute for understanding the scene",
     "Vision and motion between them still do not answer what is happening. A "
     "text prior adds that: captions translate the scene into language, which "
     "can be compared against high-risk action patterns, and sentence encoders "
     "trained on large image–text corpora carry enough of that structure to "
     "reason about danger without task-specific training. Table 3 records how "
     "the encoder was chosen."),

    ("Based on this analysis, it is identified that the central problem",
     "Each engine alone is blind to what the others see. The semantic score "
     "reads appearance without temporal context, so a foggy frame can look "
     "dangerous in ordinary driving and early frames receive moderate rather "
     "than near-zero scores. The motion proxy cannot separate ordinary traffic "
     "movement from accident movement, and is high at the start of clips where "
     "the camera itself is moving. The caption prior cannot see the video at "
     "all, so identically worded captions receive identical trajectories. Table "
     "4 sets the three against one another; ensembling is intended to let each "
     "cover the others' blind spot."),

    ("The total 1417 images are processed individually through the six-stage",
     "All 1,417 clips pass through the six-stage pipeline of Figure 3. Each "
     "clip, standardised to 150 frames bounded by its start and end frames, is "
     "fed to the three engines, which compute their curves in parallel."),
]

DELETE = [
    "After defining the problem statement and understanding the complexities",
    "The pre-crash accident anticipation framework operates with three specialized",
    "The system receives a video sequence from the dashcam",
    "The actions presented in the scene are semantically translated with NLP captions",
    "NLP captioning can aid in better prediction because it can encodes",
    "The NLP caption models are trained on vast image-text pairs with famous corpora",
    "The contextual signals present at the scenes like",
    "Thus, to derive clear semantics interpreted at the visual scene",
    "Adopting a single model for this accident anticipation, the individual models",
    "CLIP alone — problems:",
    "CLIP scores frames based on visual appearance only",
    "A foggy frame may look 'dangerous' even in normal driving",
    "CLIP gives moderate scores (~0.38-0.42) for early frames",
    "Without NLP prior, CLIP has no information about WHEN",
    "Result: high crossover frame (~26-28), poor TTA score",
    "NLP Caption alone — problems:",
    "Text model cannot see the actual video",
    "All captions with similar wording get identical t0 values",
    "Cannot detect visual cues: sudden swerve, pedestrian running",
    "Ceiling effect: avg crossover stuck at ~21.7",
    "Optical Flow alone — problems:",
    "Motion is high at the START of clips too",
    "Cannot distinguish between normal driving motion and accident motion",
    "No semantic understanding — a busy intersection looks",
    "Very noisy signal — needs to be smoothed and combined",
    "Thus, ensembling provides us to find give clues regarding",
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
    miss = [p[:44] for p, _ in REPLACE if p not in done]
    print(f"  [OK] {len(done)}/{len(REPLACE)} replaced" + (f"  MISSING {miss}" if miss else ""))

    seen = set()
    for p in list(body.iter(W + "p")):
        t = text_of(p).strip()
        for pre in DELETE:
            if pre not in seen and t.startswith(pre):
                body.remove(p)
                seen.add(pre)
                break
    absent = [p[:40] for p in DELETE if p not in seen]
    print(f"  [OK] {len(seen)}/{len(DELETE)} removed" + (f"  NOT FOUND {absent}" if absent else ""))

    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
