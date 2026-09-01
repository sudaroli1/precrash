"""
Pass 13: put the decomposition figure into Section 5.3.

The paper has three figures, all in Section 4, all of the system. Sections 5 to
9 — where the finding is — carry only tables. The single most useful thing a
reader could be shown is the one relationship the whole argument rests on: the
official score falling linearly in the frame at which the alarm fires, with the
discrimination floor and the leaderboard levels drawn across it. That figure
was built when the measurements were made and never placed.

It goes at the end of 5.3, immediately after the paragraph reporting the
held-out predictions, and becomes Figure 4.
"""
import copy
import shutil
from pathlib import Path
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"

UNP = Path("unp6")
SRC = Path("/home/claude/newrepo/results/fig_decomposition.png")

CAPTION = ("Figure 4: The official score is a straight line in the frame at "
           "which the alarm fires. Points are video-blind submissions scored by "
           "the organisers; the fit is over crossings up to frame 100. A "
           "constant crosses at frame 0.")


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


# --- 1. the media file and its relationship -------------------------------
media = UNP / "word" / "media"
media.mkdir(exist_ok=True)
shutil.copy(SRC, media / "image4.png")

rels_path = UNP / "word" / "_rels" / "document.xml.rels"
rels = etree.parse(str(rels_path))
root = rels.getroot()
ns = root.nsmap[None]
used = {c.get("Id") for c in root}
rid = next(f"rId{n}" for n in range(900, 999) if f"rId{n}" not in used)
etree.SubElement(root, f"{{{ns}}}Relationship", {
    "Id": rid,
    "Type": "http://schemas.openxmlformats.org/officeDocument/2006/"
            "relationships/image",
    "Target": "media/image4.png",
})
rels.write(str(rels_path), xml_declaration=True, encoding="UTF-8",
           standalone=True)
print(f"  [OK] media/image4.png added as {rid}")

# --- 2. the drawing -------------------------------------------------------
CX, CY = 5486400, 3331464          # 6.00in x 3.64in, the figure's own ratio
DRAWING = f'''<w:p xmlns:w="{W[1:-1]}">
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r><w:drawing>
    <wp:inline xmlns:wp="{WP}" distT="0" distB="0" distL="0" distR="0">
      <wp:extent cx="{CX}" cy="{CY}"/>
      <wp:docPr id="4001" name="Figure 4" descr="Official score against the
        frame at which the risk curve crosses 0.5, for video-blind
        submissions"/>
      <a:graphic xmlns:a="{A}">
        <a:graphicData uri="{PIC}">
          <pic:pic xmlns:pic="{PIC}">
            <pic:nvPicPr>
              <pic:cNvPr id="0" name="fig_decomposition.png"/>
              <pic:cNvPicPr/>
            </pic:nvPicPr>
            <pic:blipFill>
              <a:blip xmlns:r="{R[1:-1]}" r:embed="{rid}"/>
              <a:stretch><a:fillRect/></a:stretch>
            </pic:blipFill>
            <pic:spPr>
              <a:xfrm><a:off x="0" y="0"/><a:ext cx="{CX}" cy="{CY}"/></a:xfrm>
              <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
            </pic:spPr>
          </pic:pic>
        </a:graphicData>
      </a:graphic>
    </wp:inline>
  </w:drawing></w:r>
</w:p>'''

doc_path = UNP / "word" / "document.xml"
tree = etree.parse(str(doc_path))
body = tree.getroot().find(W + "body")

anchor = None
for p in body.iter(W + "p"):
    if text_of(p).startswith("The line was then asked to predict curves"):
        anchor = p
if anchor is None:
    raise SystemExit("5.3 held-out paragraph not found")

cap_tmpl = None
for p in body.iter(W + "p"):
    if text_of(p).strip().startswith("Figure 1:"):
        cap_tmpl = copy.deepcopy(p)
        break
if cap_tmpl is None:
    raise SystemExit("figure caption template not found")

at = list(body).index(anchor)
body.insert(at + 1, etree.fromstring(DRAWING))
body.insert(at + 2, set_text(copy.deepcopy(cap_tmpl), CAPTION))
print("  [OK] figure and caption inserted at the end of 5.3")

tree.write(str(doc_path), xml_declaration=True, encoding="UTF-8",
           standalone=True)
