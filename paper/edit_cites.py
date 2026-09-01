"""
Pass 5: make every citation author-prominent, and disambiguate the surnames.

Every reference in the running text now carries the author name in front of the
bracket. Where a sentence supported by several references was not about any of
them, the sentence is rephrased so the naming reads naturally rather than
stacking four groups into a clause that did not want them.

Two ambiguities are resolved at the same time, and they were the real defect:

  Zhang  [2] Zhang, Y. et al. (survey)
         [14] Zhang, J. et al. (CAMERA)      <- [14] and [15] appeared in
         [15] Zhang, R. et al. (SeeUnsafe)      consecutive sentences, both as
                                                "Zhang et al."
  Liu    [11] Liu, H. et al. (LLaVA)         <- same surname AND same initial
         [17] Liu, H. et al. (SCTNet)

Zhangs take initials, per IEEE practice for ambiguous surnames. The two Lius
share an initial, so initials do not separate them; each is named with its
system instead, which is unambiguous and reads better.

Not touched: the parked introduction material after the references. Its bracket
numbers refer to the old reference list and its heading already says so.
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp6/word/document.xml")

FIXES = [
    # ---------------- Section 1 ----------------
    ("Road traffic injury kills approximately 1.19 million people every year "
     "[1], and leaves far more with permanent disability.",
     "The World Health Organization [1] records approximately 1.19 million "
     "road traffic deaths every year, and far more people left with permanent "
     "disability."),

    ("worth pursuing as an obligation and not only as a research problem [2].",
     "worth pursuing as an obligation and not only as a research problem — the "
     "case the review of Zhang Y. et al. [2] sets out at length."),

    ("and the hard braking of the vehicle ahead [3].",
     "and the hard braking of the vehicle ahead, which is the material Moura "
     "et al. [3] assembled into a public collision-prediction corpus."),

    # ---------------- Section 2.1 ----------------
    ("temporal models trained on labelled dashcam footage [4], [5], [6], [16].",
     "temporal models trained on labelled dashcam footage, in the line running "
     "from Chan et al. [4] and Bao et al. [5] through Karim et al. [6] to "
     "Kandacharam et al. [16]."),

    ("and collision labels are expensive and rare [2].",
     "and collision labels are expensive and rare, which Zhang Y. et al. [2] "
     "identify as a persistent constraint on the field."),

    ("Liu et al. [17] apply a causality-guided spatiotemporal diffusion "
     "network to unsupervised detection",
     "Liu et al. [17], in SCTNet, apply a causality-guided spatiotemporal "
     "diffusion network to unsupervised detection"),

    # ---------------- Section 2.2 ----------------
    ("CLIP [9] demonstrated that contrastive pretraining on 400 million "
     "image–text pairs",
     "Radford et al. [9] demonstrated with CLIP that contrastive pretraining "
     "on 400 million image–text pairs"),

    ("Instruction-tuned multimodal models such as LLaVA [11] and the InternVL "
     "series [12] extend this",
     "Instruction-tuned multimodal models — LLaVA, from Liu et al. [11], and "
     "the InternVL series of Chen et al. [12] — extend this"),

    ("dense optical flow, for which RAFT [10] is the standard modern "
     "estimator,",
     "dense optical flow, for which the RAFT estimator of Teed and Deng [10] "
     "is the modern standard,"),

    ("Zhang et al. [15] deploy multimodal large language models",
     "Zhang R. et al. [15] deploy multimodal large language models"),

    ("Zhang et al. [14] integrate scene-level context",
     "Zhang J. et al. [14] integrate scene-level context"),

    # ---------------- Section 2.3 ----------------
    ("DAD [4] and CCD [5] remain the standard labelled benchmarks",
     "The Dashcam Accident Dataset of Chan et al. [4] and the Car Crash "
     "Dataset of Bao et al. [5] remain the standard labelled benchmarks"),

    ("MM-AU [7] is a large ego-view corpus of 11,727 accident videos",
     "MM-AU, released by Fang et al. [7], is a large ego-view corpus of 11,727 "
     "accident videos"),

    ("The Nexar dashcam collision prediction dataset [3] is a separate "
     "resource,",
     "The Nexar dashcam collision prediction dataset of Moura et al. [3] is a "
     "separate resource,"),

    ("The benchmark examined in this paper [8] distributes a curated "
     "1,417-clip subset of MM-AU.",
     "The benchmark examined in this paper, released by AUTOPILOT [8], "
     "distributes a curated 1,417-clip subset of MM-AU."),

    # ---------------- Section 2.4 ----------------
    ("The convention of [6] pairs timing with discrimination",
     "The convention of Karim et al. [6] pairs timing with discrimination"),

    ("adding two threshold-crossing timing terms to average precision and area "
     "under the ROC curve [8].",
     "adding two threshold-crossing timing terms to average precision and area "
     "under the ROC curve, as the evaluation page of AUTOPILOT [8] sets out."),

    ("Two of the benchmark papers examined here [3], [7] publish no "
     "anticipation scores of their own",
     "Two of the benchmark papers examined here, those of Moura et al. [3] and "
     "Fang et al. [7], publish no anticipation scores of their own"),

    # ---------------- Section 2.5 ----------------
    ("Monotonicity constraints on risk scores are prior work [20], [21], and "
     "this paper claims none.",
     "Monotonicity constraints on risk scores are prior work, due to Pjetri et "
     "al. [20] and Zou et al. [21], and this paper claims none."),

    ("Training-free pipelines combining CLIP with a motion signal already "
     "exist [24].",
     "Training-free pipelines combining CLIP with a motion signal already "
     "exist, as Thakur and Talele [24] show."),

    ("the specific suspicion that time-to-accident is inflatable has been "
     "voiced before [30], [31], [32], as has the use of a blind control on an "
     "adjacent traffic-video task [29].",
     "the specific suspicion that time-to-accident is inflatable has been "
     "voiced before by Zhao et al. [30], Goldshmidt et al. [31] and Caselli et "
     "al. [32], as has the use of a blind control on an adjacent traffic-video "
     "task by Korkut et al. [29]."),

    # ---------------- Section 3 ----------------
    ("were made on Zero-shot Accident Anticipation [8], a public competition "
     "hosted on Kaggle by AUTOPILOT.",
     "were made on Zero-shot Accident Anticipation, the public competition "
     "hosted on Kaggle by AUTOPILOT [8]."),

    ("The corpus is a curated 1,417-clip subset of MM-AU [7].",
     "The corpus is a curated 1,417-clip subset of the MM-AU corpus of Fang et "
     "al. [7]."),

    # ---------------- Section 4 ----------------
    ("frame differencing was chosen because RAFT [10] did not fit the compute "
     "budget available",
     "frame differencing was chosen because the RAFT estimator of Teed and "
     "Deng [10] did not fit the compute budget available"),

    ("CLIP ViT-L/14 [9] takes the frame as input",
     "The CLIP ViT-L/14 encoder of Radford et al. [9] takes the frame as "
     "input"),

    # ---------------- Section 8 ----------------
    ("for which DAD [4] and CCD [5] are the obvious candidates",
     "for which the corpora of Chan et al. [4] and Bao et al. [5] are the "
     "obvious candidates"),
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

    # The parked material after the references keeps the old numbering, so it
    # is excluded from this pass.
    parked = False
    live = []
    for p in root.iter(W + "p"):
        if text_of(p).startswith("MATERIAL MOVED FROM THE INTRODUCTION"):
            parked = True
        if not parked:
            live.append(p)

    hits = {old: 0 for old, _ in FIXES}
    for p in live:
        txt = text_of(p)
        for old, new in FIXES:
            if old in txt:
                txt = txt.replace(old, new)
                set_text(p, txt)
                hits[old] += 1

    bad = 0
    for old, n in hits.items():
        if n != 1:
            bad += 1
            print(f"  [!! {n}] {old[:66]}...")
    print(f"  [OK] {len(FIXES) - bad} of {len(FIXES)} applied cleanly")

    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
