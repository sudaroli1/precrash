"""
Pass 2b: corrections to Section 2, found by re-checking it.

Four things were wrong.

1. Renumbering the reference list broke a citation in Section 3. "CLIP
   ViT-L/14 [11]" pointed at Dong et al. under the old numbering; under the new
   list [11] is LLaVA. It becomes [9], Radford et al.

2. Section 2.4 claimed that no surveyed work reports a video-blind control.
   Checking it properly: the claim survives, but two of the works it swept in
   (MM-AU, Nexar) publish no anticipation scores at all and so cannot bear on
   it, and four of the ten could not be opened at table level. The sentence is
   narrowed to what was actually verified.

3. Section 2 did not cite the existing critiques of the TTA family. Three exist
   and all are on-topic: Zhao et al. state that the standard TTA calculation
   yields "artificially inflated" values; Goldshmidt et al. report baselines
   making physically implausible predictions 9-10 s before a collision; Caselli
   et al. publish, on Nexar, DSTA at AUC 45.5 with mTTA 10.216 and UString at
   AUC 41.0 with mTTA 10.263 -- below-chance discrimination paired with
   ten-second anticipation, which is this paper's argument already visible in
   someone else's results table. Presenting the observation as unprecedented
   with these three uncited would have been the paper's weakest point.

4. Korkut et al. audit traffic VideoQA benchmarks, MM-AU among them, against
   text-only "blind" models and find that removing the video can improve
   accuracy. Different task and different metric, so not a counterexample --
   but close enough that a referee would hand it over as prior art. It is now
   cited and distinguished.

Also: the parked introduction material carries bracket numbers from the old
reference list, so its heading now says so.
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp3/word/document.xml")

# --- exact-text replacements, (old fragment, new fragment) ---
INLINE = [
    ("CLIP ViT-L/14 [11] takes the frame as input",
     "CLIP ViT-L/14 [9] takes the frame as input"),

    ("Karim et al. [6] established the reporting convention that most "
     "subsequent work follows",
     "Karim et al. [6] gave the reporting convention that most subsequent work "
     "adopts"),

    ("MATERIAL MOVED FROM THE INTRODUCTION — to be reintegrated into Section 2, "
     "not for submission in this position",
     "MATERIAL MOVED FROM THE INTRODUCTION — to be reintegrated into Section 2, "
     "not for submission in this position. NOTE: the bracket numbers in these "
     "paragraphs refer to the OLD reference list and must be renumbered against "
     "the current one before any of this text is reused."),
]

OLD_NO_CONTROL = (
    "Yet across the work surveyed in this section we found no instance in which "
    "the score of a video-blind control is reported alongside a result. That "
    "absence, rather than any defect in any particular method, is what this "
    "paper addresses."
)

NEW_NO_CONTROL = (
    "Yet of the works surveyed in this section that report anticipation results "
    "on average precision, area under the curve, time-to-accident or a "
    "composite of these, we found none that reports them alongside an "
    "input-independent control — a baseline whose per-frame risk output is a "
    "function of the frame index alone. Two of the benchmark papers examined "
    "here [3], [7] publish no anticipation scores of their own and so do not "
    "bear on the point. That absence, rather than any defect in any particular "
    "method, is what this paper addresses."
)

OLD_SCOPE = (
    "We state the scope of that observation precisely. It is a statement about "
    "the works cited in this section, which we read for it. It is not a "
    "systematic audit of the field, and we do not present it as one."
)

NEW_SCOPE = (
    "We state the scope of that observation precisely. It covers the works "
    "cited in this section whose results tables we were able to inspect "
    "directly; four of those we consulted are behind paywalls we could not "
    "open, and we exclude them from the count rather than assume. It is not a "
    "systematic audit of the field, and we do not present it as one."
)

# Inserted immediately after the NEW_NO_CONTROL paragraph.
PRIOR_CRITIQUE = [
    "This is not to say that the timing metrics have gone unquestioned. Zhao et "
    "al. [30] observe that the standard time-to-accident calculation credits a "
    "false alarm in a safe scene with the interval to some later, causally "
    "unrelated collision, producing what they call artificially inflated "
    "estimates. Goldshmidt et al. [31] report that baseline methods on Nexar "
    "make physically implausible predictions nine to ten seconds before a "
    "collision. Most directly, Caselli et al. [32] publish on Nexar a table in "
    "which two established anticipation methods record areas under the curve of "
    "45.5 and 41.0 — below chance — beside mean times-to-accident of 10.216 and "
    "10.263 seconds. A method that cannot separate accident clips from normal "
    "ones is there credited with ten seconds of anticipation, which is this "
    "paper's argument already legible in someone else's results table. What "
    "these accounts share is that the pathology is noticed as an anomaly in the "
    "numbers of trained models. None of them isolates it, and none reports what "
    "a submission that reads nothing would score.",

    "The nearest methodological precedent lies in an adjacent task. Korkut et "
    "al. [29] audit four traffic video question-answering benchmarks, MM-AU "
    "among them, by running vision–language models blind — given the question "
    "and answer options but no video — and find that on MM-AU removing the "
    "video consistently improves accuracy. That is a control of exactly the "
    "kind advocated here, applied to multiple-choice accuracy rather than to "
    "per-frame risk. We are not aware of an equivalent control reported for the "
    "timing and discrimination metrics used in anticipation, which is the gap "
    "this paper fills.",
]

OLD_NOT_NEW = (
    "Nor is the general idea that a benchmark can be satisfied by a degenerate "
    "submission new."
)

NEW_NOT_NEW = (
    "Nor is the general idea that a benchmark can be satisfied by a degenerate "
    "submission new, and the specific suspicion that time-to-accident is "
    "inflatable has been voiced before [30], [31], [32], as has the use of a "
    "blind control on an adjacent traffic-video task [29]."
)

NEW_REFS = [
    "Korkut, S., Bravo Sarmiento, M. A., Kim, S., & Akata, Z. (2026). From "
    "accuracy to visual dependence: Auditing and filtering modality collapse in "
    "traffic VideoQA. In 1st Workshop on Combining Theory and Benchmarks "
    "(CTB@ICML 2026), Seoul. arXiv:2606.30220.",

    "Zhao, T., Zou, Y., Mao, Z., Xiao, P., Huang, Y., Yang, H., Li, Y., Li, Q., "
    "Wu, G., & Lin, Y. (2025). Accident anticipation via temporal occurrence "
    "prediction. arXiv:2510.22260.",

    "Goldshmidt, R., Scott, H., Niccolini, L., Zhu, S., Moura, D., & Zvitia, O. "
    "(2025). BADAS: Context-aware collision prediction using real-world dashcam "
    "data. arXiv:2510.14876.",

    "Caselli, L., Trinci, T., Bianconcini, T., Magistri, S., Taccari, L., "
    "Sambo, F., & Bagdanov, A. D. (2026). FLaRA: Predicting future latent "
    "representations for accident anticipation. arXiv:2606.14380.",
]


def para_text(p):
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
    body = tree.getroot().find(W + "body")
    paras = [c for c in body if c.tag == W + "p"]

    done = {k: 0 for k, _ in INLINE}
    for p in paras:
        txt = para_text(p)
        for old, new in INLINE:
            if old in txt:
                set_text(p, txt.replace(old, new))
                done[old] += 1
                txt = para_text(p)
    for old, n in done.items():
        status = "OK" if n == 1 else f"!! matched {n} times"
        print(f"  [{status}] {old[:58]}...")

    # --- the two rewritten paragraphs, and the inserted pair ---
    anchor = None
    for p in paras:
        t = para_text(p)
        if OLD_NO_CONTROL in t:
            set_text(p, t.replace(OLD_NO_CONTROL, NEW_NO_CONTROL))
            anchor = p
            print("  [OK] narrowed the no-control claim")
        elif OLD_SCOPE in t:
            set_text(p, t.replace(OLD_SCOPE, NEW_SCOPE))
            print("  [OK] scope disclaimer now states what was not inspected")
        elif OLD_NOT_NEW in t:
            set_text(p, t.replace(OLD_NOT_NEW, NEW_NOT_NEW))
            print("  [OK] 2.5 now names the prior critiques")

    if anchor is None:
        raise SystemExit("could not find the no-control paragraph")

    tmpl = copy.deepcopy(anchor)
    at = list(body).index(anchor)
    for off, text in enumerate(PRIOR_CRITIQUE, start=1):
        body.insert(at + off, set_text(copy.deepcopy(tmpl), text))
    print(f"  [OK] inserted {len(PRIOR_CRITIQUE)} paragraphs on prior critiques")

    # --- append the four new references ---
    last_ref = None
    for p in paras:
        if para_text(p).startswith("Ferrari Dacrema, M."):
            last_ref = p
    if last_ref is None:
        raise SystemExit("could not find the last reference entry")
    at = list(body).index(last_ref)
    for off, text in enumerate(NEW_REFS, start=1):
        body.insert(at + off, set_text(copy.deepcopy(last_ref), text))
    print(f"  [OK] appended {len(NEW_REFS)} references (list is now 32)")

    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
