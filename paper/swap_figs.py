"""
Pass 14: one system figure instead of three.

Figure 1 (architecture) and Figure 3 (workflow) covered the same ground; only
the second named the operations, and it carried the tuned ensemble weights and
the temporal-compression stage without saying what became of either. Figure 2
(preprocessing) contradicted the corrected text in four places: 1080p video, a
horn/brake audio modality, motion compensation, and a segmentation and
multiscale-pyramid stage, none of which is in this corpus or in the released
code.

All three are replaced by one figure of the pipeline as implemented, with the
two stages Section 6 analyses drawn dashed. The decomposition figure becomes
Figure 2.
"""
import copy, shutil
from pathlib import Path
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
UNP = Path("unp6")

CAP1 = ("Figure 1: The pipeline as implemented. Dashed stages are the two "
        "analysed in Section 6: the monotone clamp forces the condition the "
        "stable-timing metric tests, and the temporal compression reported at "
        "each frame a score computed from a later one.")
CAP2 = ("Figure 2: The official score is a straight line in the frame at which "
        "the alarm fires. Points are video-blind submissions scored by the "
        "organisers; the fit is over crossings up to frame 100. A constant "
        "crosses at frame 0.")


def text_of(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def set_text(p, s):
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
    t.text = s
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


# 1. new artwork over image1, and correct its aspect ratio
shutil.copy("/home/claude/newrepo/results/fig_pipeline.png",
            UNP / "word" / "media" / "image1.png")

tree = etree.parse(str(UNP / "word" / "document.xml"))
body = tree.getroot().find(W + "body")

CX = 5731510
CY = int(CX / 1.3340)          # the new figure's own ratio
for p in body.iter(W + "p"):
    for blip in p.iter(A + "blip"):
        rid = blip.get("{http://schemas.openxmlformats.org/officeDocument/"
                       "2006/relationships}embed")
        if rid == "rId6":
            for e in p.iter(WP + "extent"):
                e.set("cx", str(CX)); e.set("cy", str(CY))
            for e in p.iter(A + "ext"):
                e.set("cx", str(CX)); e.set("cy", str(CY))
            print(f"  [OK] image1 replaced, extent set to {CX} x {CY}")

# 2. drop the two other figures, caption and image together
for cap in ("Figure 2: Preprocessing pipeline", "Figure 3: Workflow with"):
    target = None
    for p in body.iter(W + "p"):
        if text_of(p).strip().startswith(cap):
            target = p
            break
    if target is None:
        print(f"  [!!] not found: {cap}")
        continue
    # the image sits in a nearby paragraph; find it within four either side
    kids = list(body)
    i = kids.index(target)
    killed = False
    for j in list(range(i - 4, i)) + list(range(i + 1, i + 5)):
        if 0 <= j < len(kids) and kids[j].tag == W + "p":
            if kids[j].find(f".//{A}blip") is not None:
                body.remove(kids[j]); killed = True; break
    body.remove(target)
    print(f"  [OK] removed '{cap}...'" + ("" if killed else "  (image not found)"))

# 3. captions and cross-references
for p in body.iter(W + "p"):
    t = text_of(p).strip()
    if t.startswith("Figure 1: Overall System Architecture"):
        set_text(p, CAP1)
    elif t.startswith("Figure 4:"):
        set_text(p, CAP2)
    elif "six-stage pipeline of Figure 3" in t:
        set_text(p, t.replace("six-stage pipeline of Figure 3",
                              "pipeline of Figure 1"))
print("  [OK] captions rewritten and cross-references remapped")

tree.write(str(UNP / "word" / "document.xml"), xml_declaration=True,
           encoding="UTF-8", standalone=True)
