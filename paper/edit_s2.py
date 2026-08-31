"""
Pass 2 on Precrash.docx: Section 2, and the reference list.

Section 2 is rewritten to carry the paper's motivation — that the field
reports a timing metric with a trivial maximiser, is moving toward composite
single-number scores, and does not report controls — and to place that beside
the episodes in adjacent fields where a control settled the same question.

The reference list is rebuilt. The previous list had two entries with author
lists that do not match the cited work, one dataset attributed to the wrong
paper and the wrong group, CLIP attributed to a fine-tuning study rather than
to Radford et al., and a quantitative latency claim supported by a range of
eight references none of which reports it. Details are in REFERENCE_AUDIT.md.

Also removed here: "Table 9: Comparative Positioning", whose final row read
"Key Limitation: None identified"; and the Section 2.5 claim that monotone
clamping for STTA compliance is introduced "for the first time", which is
contradicted by reference [20] of this same paper.
"""
from __future__ import annotations

import copy
from pathlib import Path

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = Path("unp2/word/document.xml")

HEADING = "RELATED WORK AND EVALUATION PRACTICE"

# (kind, text). "h" = sub-heading, "p" = body paragraph.
SECTION2 = [
    ("h", "2.1. Accident anticipation and the origin of its metrics"),

    ("p", "Vision-based accident anticipation was posed in its modern form by "
          "Chan et al. [4], who introduced the Dashcam Accident Dataset (DAD) "
          "together with the measure the field still reports: the interval "
          "between the first frame at which a model's risk score crosses a "
          "threshold and the annotated onset of the collision. Bao et al. [5] "
          "contributed the Car Crash Dataset (CCD) and an uncertainty-based "
          "formulation, and Karim et al. [6] established the reporting "
          "convention that most subsequent work follows — average precision, "
          "mean time-to-accident over thresholds, and time-to-accident at 80% "
          "recall."),

    ("p", "It is worth stating plainly what that convention gets right, because "
          "this paper is about what happens when it is set aside. It reports a "
          "discrimination metric alongside every timing metric, and it reads "
          "timing at a fixed operating point of the discrimination metric. A "
          "method therefore cannot buy time-to-accident by alarming "
          "indiscriminately: the recall constraint holds it to a fixed "
          "false-alarm behaviour before its timing is counted at all."),

    ("p", "Supervised architectures built on this foundation combine "
          "convolutional feature extractors with recurrent or attention-based "
          "temporal models trained on labelled dashcam footage [4], [5], [6], "
          "[16]. Their shared limitation is dependence on the labelled corpus. "
          "Performance degrades on road geometry, weather and traffic culture "
          "not represented in training, and collision labels are expensive and "
          "rare [2]."),

    ("p", "Self-supervised and unsupervised alternatives reduce that "
          "dependence. Yao et al. [18] treat accidents as anomalies in "
          "predicted object trajectories, removing the need for accident labels "
          "entirely. Liu et al. [17] apply a causality-guided spatiotemporal "
          "diffusion network to unsupervised detection, and Zhong et al. [19] "
          "use contrastive feature-consistency representation with soft-label "
          "regression for early anticipation on DAD and CCD."),

    ("p", "Two lines of work deserve particular attention here, because they "
          "anticipate part of this paper's subject. Pjetri et al. [20] train "
          "with a self-supervised objective requiring the predicted danger "
          "score to be non-decreasing as the collision approaches, and Zou et "
          "al. [21] introduce an adaptive monotonic constraint loss with the "
          "same intent, evaluated on MM-AU and Nexar. Both are well-motivated "
          "responses to score oscillation and we make no criticism of either. "
          "But both also illustrate the hazard this paper examines. A risk "
          "score constrained not to fall is, by construction, closer to the "
          "maximiser of a threshold-crossing timing metric than a score that is "
          "free to fall. The metric therefore cannot separate the benefit of "
          "the constraint from the artefact of it, and neither paper reports a "
          "control that would."),

    ("h", "2.2. Vision–language models and training-free anticipation"),

    ("p", "CLIP [9] demonstrated that contrastive pretraining on 400 million "
          "image–text pairs yields a model whose zero-shot classification and "
          "retrieval transfer without task-specific supervision. "
          "Instruction-tuned multimodal models such as LLaVA [11] and the "
          "InternVL series [12] extend this to open-ended scene description and "
          "reasoning, at substantially higher inference cost per frame. Motion "
          "is the other signal these systems draw on: dense optical flow, for "
          "which RAFT [10] is the standard modern estimator, yields pixel-level "
          "motion fields, while frame differencing is the cheap approximation. "
          "Section 4 states which of the two the system examined here uses, and "
          "declines to assert an accuracy-for-compute trade-off that was not "
          "measured."),

    ("p", "Applied to traffic safety, these models have mostly been used for "
          "accident understanding rather than anticipation. Zhang et al. [15] "
          "deploy multimodal large language models for video-based accident "
          "analysis, showing that language-level reasoning surfaces causal "
          "structure that purely visual models miss. Zhang et al. [14] "
          "integrate scene-level context with visual perception for risk "
          "anticipation. Liao et al. [13] add chain-of-thought reasoning to a "
          "multimodal large language model and report improved anticipation. "
          "These systems require fine-tuning, a large-model inference pipeline, "
          "or both."),

    ("p", "Training-free work has appeared very recently. Thakur and Talele "
          "[24] combine frame-difference peak detection, Farnebäck optical flow "
          "and CLIP prompt matching with no fine-tuning; Saha et al. [22] "
          "propose a coarse-to-fine vision–language-model tracking pipeline; "
          "and Singh et al. [23] use metadata-aware multi-prompt reasoning. All "
          "three target the detection, localisation or classification of an "
          "accident in surveillance video, and none reports a time-to-accident "
          "metric. Zero-shot anticipation evaluated on a timing metric "
          "therefore remains sparse, which is what motivated the system "
          "described in Section 4 and why it was submitted to a competition "
          "rather than scored locally."),

    ("h", "2.3. Corpora"),

    ("p", "DAD [4] and CCD [5] remain the standard labelled benchmarks and "
          "publish their annotations, which is what would make the corrected "
          "protocol of Section 7 computable on them. MM-AU [7] is a large "
          "ego-view corpus of 11,727 accident videos with object boxes and "
          "video–text pairs across 58 accident-reason categories. The Nexar "
          "dashcam collision prediction dataset [3] is a separate resource, by "
          "a separate group, of roughly 1,500 clips released for a public "
          "challenge. The two are sometimes conflated in the literature and "
          "should not be."),

    ("p", "The benchmark examined in this paper [8] distributes a curated "
          "1,417-clip subset of MM-AU. Its labels are withheld from entrants "
          "and exposed only through a leaderboard. That design is legitimate, "
          "and it is what makes the measurements of Section 5 third-party "
          "rather than self-reported. It also removes every locally computable "
          "check an entrant might otherwise apply, a consequence Section 3 "
          "develops."),

    ("h", "2.4. What the field reports, and what it does not"),

    ("p", "The convention of [6] pairs timing with discrimination and fixes an "
          "operating point. Competitions need something that convention does "
          "not supply: a total order over submissions. The natural way to "
          "obtain one is to sum the metrics into a single score, and that is "
          "what the benchmark studied here does, adding two threshold-crossing "
          "timing terms to average precision and area under the ROC curve [8]."),

    ("p", "Summing is where the difficulty enters, and it is a difficulty of "
          "type rather than of tuning. Average precision and area under the "
          "curve are bounded on the unit interval. A time-to-accident measured "
          "in frames is bounded only by the length of the clip. Adding them "
          "makes the sum depend on how the corpus was windowed, and lets the "
          "unbounded term decide the ranking. Section 6 develops the argument; "
          "Section 5 measures its magnitude."),

    ("p", "The check that exposes such a problem is a control that does not "
          "read the input, and its value has been established repeatedly in "
          "adjacent fields. In each case the control was cheap, and in each "
          "case nobody had run it. Torralba and Efros [25] showed that image "
          "datasets carry signatures strong enough for a classifier to name the "
          "dataset a photograph came from, which changed how benchmark results "
          "were read. In natural language inference, Gururangan et al. [26] and "
          "Poliak et al. [27] showed independently that a model given only the "
          "hypothesis — half of the input — scores far above chance on SNLI and "
          "MNLI, demonstrating that the headline numbers were partly measuring "
          "annotation artefacts rather than inference; partial-input baselines "
          "became standard practice afterwards. In recommender systems, Ferrari "
          "Dacrema et al. [28] found that properly tuned simple baselines "
          "matched or beat most recently published neural methods, and "
          "evaluation practice in that field changed in consequence."),

    ("p", "Accident anticipation has every precondition those episodes had. It "
          "reports a timing metric whose maximiser is trivial. It is moving "
          "toward composite single-number scores. Its newest benchmarks "
          "withhold labels, so an entrant cannot compute the discrimination "
          "half at all and must develop against a locally computable proxy — "
          "and the only proxy that is locally computable is the shape of one's "
          "own risk curve, which a constant maximises. Yet across the work "
          "surveyed in this section we found no instance in which the score of "
          "a video-blind control is reported alongside a result. That absence, "
          "rather than any defect in any particular method, is what this paper "
          "addresses."),

    ("p", "We state the scope of that observation precisely. It is a statement "
          "about the works cited in this section, which we read for it. It is "
          "not a systematic audit of the field, and we do not present it as "
          "one."),

    ("h", "2.5. Positioning of this work"),

    ("p", "The contribution of this paper is not an anticipation method. The "
          "ensemble described in Section 4 is the instrument that led to the "
          "finding and the case study that illustrates it. As a system it is "
          "unremarkable, and it placed ninth of thirteen."),

    ("p", "What is new is threefold: a measurement, on a live benchmark, of how "
          "much of a composite anticipation score a video-blind submission "
          "collects; a probe methodology that decomposes such a score from "
          "outside a competition, without labels and without knowledge of the "
          "weights; and a protocol whose central requirement is that a control "
          "be reported."),

    ("p", "We are equally explicit about what is not new. Monotonicity "
          "constraints on risk scores are prior work [20], [21], and this paper "
          "claims none. Training-free pipelines combining CLIP with a motion "
          "signal already exist [24]. Nor is the general idea that a benchmark "
          "can be satisfied by a degenerate submission new. What is new is the "
          "measurement of how far it goes on this class of metric, and the "
          "demonstration that the measurement can be made from outside the "
          "competition that holds the labels."),
]

REFERENCES = [
    "World Health Organization, \"Road traffic injuries,\" WHO Fact Sheet, "
    "Geneva, Switzerland, 13 Dec. 2023. [Online]. Available: "
    "https://www.who.int/news-room/fact-sheets/detail/road-traffic-injuries",

    "Zhang, Y., Zhou, W., Lin, R., Yang, X., & Zheng, H. (2025). Deep learning "
    "advances in vision-based traffic accident anticipation: A comprehensive "
    "review of methods, datasets, and future directions. arXiv:2505.07611.",

    "Moura, D., Zhu, S., & Zvitia, O. (2025). Nexar dashcam collision "
    "prediction dataset and challenge. In Proc. IEEE/CVF Conf. on Computer "
    "Vision and Pattern Recognition Workshops (CVPRW), pp. 2608–2616. "
    "arXiv:2503.03848.",

    "Chan, F.-H., Chen, Y.-T., Xiang, Y., & Sun, M. (2016). Anticipating "
    "accidents in dashcam videos. In Computer Vision – ACCV 2016, Lecture Notes "
    "in Computer Science, vol. 10114, pp. 136–153. Springer.",

    "Bao, W., Yu, Q., & Kong, Y. (2020). Uncertainty-based traffic accident "
    "anticipation with spatio-temporal relational learning. In Proc. 28th ACM "
    "Int. Conf. on Multimedia, pp. 2682–2690. doi:10.1145/3394171.3413827.",

    "Karim, M. M., Li, Y., Qin, R., & Yin, Z. (2022). A dynamic spatial-temporal "
    "attention network for early anticipation of traffic accidents. IEEE Trans. "
    "Intelligent Transportation Systems, 23(7), 9590–9600. "
    "doi:10.1109/TITS.2022.3155613.",

    "Fang, J., Li, L.-L., Zhou, J., Xiao, J., Yu, H., Lv, C., Xue, J., & Chua, "
    "T.-S. (2024). Abductive ego-view accident video understanding for safe "
    "driving perception. In Proc. IEEE/CVF Conf. on Computer Vision and Pattern "
    "Recognition (CVPR), pp. 22030–22040. arXiv:2403.00436.",

    "AUTOPILOT-COG (2026). Zero-shot Accident Anticipation. Kaggle. [Online]. "
    "Available: https://kaggle.com/competitions/zero-shot-taa",

    "Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., "
    "Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, "
    "I. (2021). Learning transferable visual models from natural language "
    "supervision. In Proc. 38th Int. Conf. on Machine Learning, PMLR 139, pp. "
    "8748–8763.",

    "Teed, Z., & Deng, J. (2020). RAFT: Recurrent all-pairs field transforms for "
    "optical flow. In Computer Vision – ECCV 2020, Lecture Notes in Computer "
    "Science, vol. 12347, pp. 402–419. Springer. "
    "doi:10.1007/978-3-030-58536-5_24.",

    "Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual instruction tuning. "
    "In Advances in Neural Information Processing Systems, vol. 36. "
    "arXiv:2304.08485.",

    "Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, "
    "Q., Zhu, X., Lu, L., Li, B., Luo, P., Lu, T., Qiao, Y., & Dai, J. (2024). "
    "InternVL: Scaling up vision foundation models and aligning for generic "
    "visual-linguistic tasks. In Proc. IEEE/CVF Conf. on Computer Vision and "
    "Pattern Recognition (CVPR), pp. 24185–24198. arXiv:2312.14238.",

    "Liao, H., Rao, B., Sun, H., Wang, C., Chang, Q., Li, S. E., Xu, C., & Li, "
    "Z. (2025). Chain-of-thought guided multimodal large language models for "
    "scene-aware accident anticipation in autonomous driving. IEEE Trans. "
    "Intelligent Transportation Systems, 26(11), 19371–19380. "
    "doi:10.1109/TITS.2025.3597411.",

    "Zhang, J., Wang, C., Liao, H., Guan, Y., Li, Z., Xie, Y., & Rao, B. (2025). "
    "Eyes on the road, mind beyond vision: Context-aware multi-modal enhanced "
    "risk anticipation. In Proc. 33rd ACM Int. Conf. on Multimedia. "
    "doi:10.1145/3746027.3755378. arXiv:2507.06444.",

    "Zhang, R., Wang, B., Zhang, J., Bian, Z., Feng, C., & Ozbay, K. (2025). "
    "When language and vision meet road safety: Leveraging multimodal large "
    "language models for video-based traffic accident analysis. Accident "
    "Analysis & Prevention, 219, 108077. doi:10.1016/j.aap.2025.108077.",

    "Kandacharam, S., Rajathilagam, B., & Vasudevan, S. K. (2025). Fusion "
    "framework for accident anticipation and incident detection in dashcam "
    "videos. SN Computer Science, 6(5), Art. 548. "
    "doi:10.1007/s42979-025-04085-z.",

    "Liu, H., Hu, X., Jiang, Y., Wan, T., & Ma, W. (2026). SCTNet: Structured "
    "and causality-guided spatiotemporal diffusion network for unsupervised "
    "traffic accident detection. Information Processing & Management, 63(4), "
    "Art. 104598. doi:10.1016/j.ipm.2025.104598.",

    "Yao, Y., Xu, M., Wang, Y., Crandall, D. J., & Atkins, E. M. (2019). "
    "Unsupervised traffic accident detection in first-person videos. In Proc. "
    "IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS), pp. 273–280.",

    "Zhong, Y., Yan, G., Zhu, R., Gan, P., & Shen, X. (2025). Early traffic "
    "accident anticipation via feature consistency representation and soft "
    "label regression. ACM Trans. Multimedia Computing, Communications, and "
    "Applications, 21. doi:10.1145/3767737.",

    "Pjetri, A., Abbondandolo, D., de Andrade, D. C., Caprasecca, S., Sambo, F., "
    "& Bagdanov, A. D. (2025). Self-supervised road accident anticipation with "
    "non-decreasing danger. In Computer Vision – ECCV 2024 Workshops, Lecture "
    "Notes in Computer Science, vol. 15629, pp. 65–79. Springer. "
    "doi:10.1007/978-3-031-91767-7_5.",

    "Zou, Y., Zhao, T., Xiao, P., Jin, H., Qi, L., Li, Y., Liang, L., Qian, Y., "
    "Lai, C., Lin, Y., Li, Z., & Wu, Y. (2026). RiskProp: Collision-anchored "
    "self-supervised risk propagation for early accident anticipation. In Proc. "
    "IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR).",

    "Saha, D., Mannan, S. M. A., Rashid, M. R., Naswan, R., & Tahmid, A. (2026). "
    "Zero-shot traffic accident detection via a coarse-to-fine VLM-tracking "
    "pipeline. arXiv:2608.08867.",

    "Singh, Pal, Biswas, & Chandran (2026). Metadata-aware multi-prompt "
    "reasoning for zero-shot accident understanding. arXiv:2606.12047. "
    "[VERIFY FULL GIVEN NAMES BEFORE SUBMISSION]",

    "Thakur, A., & Talele, S. (2026). A modular zero-shot pipeline for accident "
    "detection, localization, and classification in traffic surveillance video. "
    "arXiv:2604.09685.",

    "Torralba, A., & Efros, A. A. (2011). Unbiased look at dataset bias. In "
    "Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), pp. "
    "1521–1528. doi:10.1109/CVPR.2011.5995347.",

    "Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S. R., & "
    "Smith, N. A. (2018). Annotation artifacts in natural language inference "
    "data. In Proc. NAACL-HLT 2018, vol. 2 (Short Papers), pp. 107–112. "
    "doi:10.18653/v1/N18-2017.",

    "Poliak, A., Naradowsky, J., Haldar, A., Rudinger, R., & Van Durme, B. "
    "(2018). Hypothesis only baselines in natural language inference. In Proc. "
    "7th Joint Conf. on Lexical and Computational Semantics (*SEM), pp. "
    "180–191. doi:10.18653/v1/S18-2023.",

    "Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). Are we really "
    "making much progress? A worrying analysis of recent neural recommendation "
    "approaches. In Proc. 13th ACM Conf. on Recommender Systems (RecSys), pp. "
    "101–109. doi:10.1145/3298689.3347058.",
]


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


def replace_block(body, paras, first, last, items, templates, blank_tmpl):
    """Swap paragraphs [first, last] for `items`, keeping document styles."""
    old = [paras[i] for i in range(first, last + 1)]
    idx = list(body).index(old[0])
    for p in old:
        body.remove(p)
    nodes = []
    for n, (kind, text) in enumerate(items):
        nodes.append(set_text(copy.deepcopy(templates[kind]), text))
        nxt = items[n + 1][0] if n + 1 < len(items) else None
        if blank_tmpl is not None and nxt is not None:
            nodes.append(copy.deepcopy(blank_tmpl))
    for off, node in enumerate(nodes):
        body.insert(idx + off, node)
    return len(old), len(nodes)


def main():
    tree = etree.parse(str(DOC))
    body = tree.getroot().find(W + "body")
    paras = [c for c in body if c.tag == W + "p"]

    head_tmpl = copy.deepcopy(paras[46])   # "2.1. Supervised and Deep..."
    body_tmpl = copy.deepcopy(paras[47])   # a section-2 body paragraph
    ref_tmpl = copy.deepcopy(paras[302])   # a numbered reference entry

    # --- the references first, so the earlier indices stay valid ---
    n_old, n_new = replace_block(
        body, paras, 302, 317,
        [("p", r) for r in REFERENCES],
        {"p": ref_tmpl}, None)
    print(f"references: {n_old} -> {n_new}")

    # --- Table 9 and its caption ---
    caption = paras[67]
    tbl = caption.getnext()
    body.remove(caption)
    if tbl is not None and tbl.tag == W + "tbl":
        body.remove(tbl)
        print("removed Table 9 and its caption")
    else:
        print("WARNING: table after the caption not found; caption removed only")

    # --- section 2 body ---
    set_text(paras[45], HEADING)
    n_old, n_new = replace_block(
        body, paras, 46, 66,
        SECTION2, {"h": head_tmpl, "p": body_tmpl}, None)
    print(f"section 2: {n_old} -> {n_new} paragraphs")

    tree.write(str(DOC), xml_declaration=True, encoding="UTF-8", standalone=True)
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
