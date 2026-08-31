# Reference audit — 31 August 2026

Every entry in the original sixteen-item reference list was checked against the
published record, and every citation was checked against the claim it was
attached to. Sources: publisher pages, CVF open access, arXiv, ACL Anthology,
Springer, Semantic Scholar.

## The two that matter most

**Two references carried author lists that do not match the cited work.** The
papers are real and the identifiers are right; the names are not.

| Cited as | Actually by |
|---|---|
| [6] Lin, R., Tang, T., Liu, Y., Zhou, W., Yang, X., Zheng, H., ... & Zhang, Y. (arXiv:2505.07611) | **Zhang, Y., Zhou, W., Lin, R., Yang, X., & Zheng, H.** — five authors. "Tang, T." and "Liu, Y." are not on the paper; the first author is wrong. |
| [13] Liu, W., Li, Y., Zhang, T., Gao, Y., Wei, L., & Chen, J. (doi:10.1145/3767737) | **Zhong, Y., Yan, G., Zhu, R., Gan, P., & Shen, X.** — none of the six cited names appears on it. |

This is the kind of thing a referee checks and an editor does not forgive, so
both are corrected in the new list. Worth knowing where they came from before
the next paper, because the pattern — right identifier, right title, invented
names — is characteristic of citations produced by a language model rather than
from a database record.

## Misattributed foundational work

**CLIP.** The manuscript wrote "Contrastive Language-Image Pre-Training (CLIP)
[11], trained on 400 million image-text pairs, demonstrated that without
training and task-specific supervision..." — but [11] was Dong et al., *CLIP
Itself Is a Strong Fine-Tuner*, which is a **supervised fine-tuning** study and
the opposite of a zero-shot result. CLIP is Radford et al., ICML 2021.

**MM-AU.** The manuscript wrote "The MM-AU dataset used for evaluation in this
work was introduced and characterised by Moura et al. [3] through the Nexar
dashcam collision prediction challenge." MM-AU and Nexar are **different
datasets by different groups**. MM-AU is Fang et al., CVPR 2024 (11,727 ego-view
accident videos). Nexar is Moura et al., CVPR 2025 *Workshops* (~1,500 clips);
its paper never mentions MM-AU. Both are now cited correctly and separately.

**RAFT.** Cited to [5], which is *Spotting Danger: How child and adult
pedestrians assess distracted drivers* — a human-subjects psychology experiment
with no optical flow, no RAFT, no computer vision at all. Two consecutive
sentences about pixel-level motion fields and kinematic precursors were both
attached to it. RAFT is Teed & Deng, ECCV 2020.

## Unsupported quantitative claims, now removed

- *"InternVL2 and LLaVA-based architectures ... typically 2 to 5 seconds per
  frame ... [9-16]."* None of the eight references reports a latency figure, and
  six of the eight are not vision–language models. InternVL2 and LLaVA were
  named but never cited at all; both now are.
- *"frame-differencing achieved approximately 90% of RAFT-equivalent sensitivity
  at 1% of the computational overhead ... empirically validated in the ablation
  studies presented in this work."* **This was never measured.** Removed.
- *"Tsai et al. [10] demonstrated that without domain specific image training..."*
  The paper's own abstract says it uses CLIP **fine-tuning**. The citation
  asserted the opposite of its source. Removed.

## The novelty claim that does not survive

Section 2.5 claimed the framework "introduces, for the first time, a
mathematically formalised monotone clamping constraint that provides hard STTA
compliance guarantees."

Reference [12] of the same manuscript is Pjetri et al., *Self-supervised road
accident anticipation with **non-decreasing danger***, which imposes exactly this
monotonicity. Zou et al., *RiskProp* (CVPR 2026), independently introduce an
"Adaptive Monotonic Constraint Loss" and evaluate on MM-AU and Nexar — the same
corpora. Claiming priority here while citing the first of them was untenable.

The new Section 2.5 says the opposite: monotonicity constraints are prior work,
this paper claims none of them, and — because a score constrained not to fall
sits closer to the timing metric's maximiser — they are an *illustration* of the
problem rather than a solution to it.

## Removed from the list

Six references were dropped as either miscited or off-topic. Restore any of them
if a use is found, but they were not supporting the claims attached to them:

- Tsai et al., construction-site CLIP captioning — fine-tuned, not zero-shot,
  not traffic.
- Dong et al., *CLIP itself is a strong fine-tuner* — was standing in for CLIP.
- Liu et al., *Spotting Danger* — pedestrian psychology, was standing in for RAFT.
- Qutaishat & Li, Toronto highway prediction — tabular GIS regression, no video.
- Zhang et al., KDD multimodal embeddings — city-scale crash-risk maps.
- Wang et al. 2021, urban big data fusion — city-scale risk maps.

## Other metadata corrections

| Ref | Was | Now |
|---|---|---|
| WHO | "Nov. 2023" | 13 Dec. 2023 (the revision stating 1.19 million) |
| Nexar | "CVPR, pp. 2583–2591" | CVPR **Workshops**, pp. 2608–2616 |
| ACM MM risk anticipation | authors 2–7 scrambled | Zhang, Wang, Liao, Guan, Li, Xie, Rao |
| Liao et al. T-ITS | no volume or pages | 26(11), 19371–19380, doi:10.1109/TITS.2025.3597411 |
| ISPRS IJGI | title truncated | (dropped, see above) |

## Added

DAD (Chan et al. 2016), CCD (Bao et al. 2020), the AP/mTTA/TTA@R80 reporting
convention (Karim et al. 2022), MM-AU (Fang et al. 2024), CLIP (Radford et al.
2021), RAFT (Teed & Deng 2020), LLaVA, InternVL, Yao et al. 2019, RiskProp,
three contemporaneous training-free accident papers, and the four
benchmark-critique references the argument rests on: Torralba & Efros 2011,
Gururangan et al. 2018, Poliak et al. 2018, Ferrari Dacrema et al. 2019.

## One item still to check

Reference [23], *Metadata-aware multi-prompt reasoning for zero-shot accident
understanding* (arXiv:2606.12047): the surnames Singh, Pal, Biswas and Chandran
are confirmed but the given names are not. The entry is marked in the document.
Confirm from the arXiv listing before submission.
