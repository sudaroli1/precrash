"""
Pass 8: dissolve the dangling "4.6 Experimental Setup".

The heading was a leftover. It had been a top-level section in the original
paper; renumbering it as 4.6 fixed a collision but left it containing
subsections numbered 4.1 to 4.4, which is incoherent on its own. The deeper
problem is that almost everything inside it duplicates material now covered
properly elsewhere, and the duplicate copies still carry the errors that were
corrected in the originals:

  4.1 Dataset and Evaluation Protocol   -> Section 3.2 and 3.3 cover this, and
      this copy revives "one of the most comprehensive benchmarks available"
      for what is a 1,417-clip subset.
  4.2 Evaluation Metrics                -> Section 3.4 quotes the metric from
      the competition page. This copy names only two of the four terms, calls
      STTA "Smooth Time-to-Accident", uses a non-strict threshold, and states
      that a lower crossover frame is a superior score -- the belief the paper
      exists to correct.
  4.3 Architecture and Implementation   -> the three modality paragraphs repeat
      4.3.1 to 4.3.3. The implementation paragraph and the hardware table do
      not repeat anything and are kept.
  4.4 Post-Processing Pipeline          -> repeats the five stages that 4.3.5
      gives with equations, and repeats them in their uncorrected form: the
      six-frame gain with no mention that it is read from a future frame, the
      clamp "eliminating false recoveries" with no mention that it forces the
      condition STTA tests, and a gamma chosen by maximising average precision,
      which no entrant can compute here.

What survives becomes 4.4, Implementation and Hardware. The control family and
the probe design, which were left as unnumbered headings, become 4.5 and 4.6 in
the section's own numbering scheme.
"""
import copy
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = "unp6/word/document.xml"

WEIGHTS = (
    "The three modality curves are combined by the fixed weights of Equation "
    "(4), with the visual signal given the dominant share. Those weights were "
    "chosen by searching over the evaluation corpus, which is a form of tuning "
    "on the test set and is recorded as a limitation in Section 8; the released "
    "configuration uses equal weights, chosen a priori, and Section 5 reports "
    "the leaderboard score of the entry that used the searched ones."
)


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


def starts(p, s):
    return text_of(p).strip().startswith(s)


tree = etree.parse(DOC)
body = tree.getroot().find(W + "body")

# the auto-numbered subsection heading style used inside Section 4
sub_tmpl = None
for p in body.iter(W + "p"):
    if text_of(p).strip() == "Parallel Processing:":
        sub_tmpl = copy.deepcopy(p)
        break
if sub_tmpl is None:
    raise SystemExit("subsection heading template not found")

DELETE = [
    "4.6. Experimental Setup",
    "4.1. Dataset and Evaluation Protocol",
    "The proposed zero-shot accident anticipation framework was evaluated on a curated test subset",
    "Each clip is standardised to exactly 150 frames at 30 frames per second (FPS), representing",
    "4.2. Evaluation Metrics",
    "To comprehensively and rigorously assess the predictive capability",
    "Time-to-Accident (TTA): TTA quantifies anticipation earliness.",
    "Smooth Time-to-Accident (STTA): STTA imposes a stricter temporal coherence",
    "4.3. Architecture and Implementation Details",
    "Vision-Language Semantics (CLIP ViT-L/14): For each frame",
    "Semantic NLP Prior (all-MiniLM-L6-v2): To provide a temporally structured",
    "Kinematic Motion Estimation (Frame-Differencing Optical Flow Proxy)",
    "4.4. Post-Processing Pipeline for STTA Compliance",
    "To satisfy the stringent monotonicity requirement imposed by the STTA metric",
    "Stage 1 — Gaussian Smoothing:",
    "Stage 2 — Power-Curve Amplification:",
    "Stage 3 — Temporal Axis Compression:",
    "Stage 4 — Monotone Clamping:",
    "Stage 5 — Numerical Stability Clipping:",
]

impl = None
gone = 0
for p in list(body.iter(W + "p")):
    t = text_of(p).strip()
    if starts(p, "The zero-shot inference pipeline was implemented in Python 3.10"):
        impl = p
        continue
    if starts(p, "The modality fusion weights were empirically determined"):
        set_text(p, WEIGHTS)
        continue
    for d in DELETE:
        if t.startswith(d):
            body.remove(p)
            gone += 1
            break
print(f"  [OK] removed {gone} paragraphs of duplicated setup")

if impl is None:
    raise SystemExit("implementation paragraph not found")

# a proper subsection heading in front of what survives
at = list(body).index(impl)
body.insert(at, set_text(copy.deepcopy(sub_tmpl), "Implementation and Hardware:"))
print("  [OK] survivors now sit under 'Implementation and Hardware'")

# the control family and probe design become numbered subsections too
for old in ["The video-blind control family",
            "Probes that isolate the terms of the metric"]:
    for p in body.iter(W + "p"):
        if text_of(p).strip() == old:
            new = set_text(copy.deepcopy(sub_tmpl), old.replace("The v", "V") + ":")
            p.getparent().replace(p, new)
            break
print("  [OK] control family and probe design are numbered subsections")

tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
