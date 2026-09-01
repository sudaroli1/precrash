"""The three new headings were cloned from "Parallel Processing:", which sits
at 4.2.3 — one level too deep. Implementation, the control family and the probe
design are siblings of "Problem Statement" (4.1), not children of "Overall
System Architecture". Recloned at the right level."""
import copy
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = "unp6/word/document.xml"


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


tree = etree.parse(DOC)
body = tree.getroot().find(W + "body")

lvl1 = None
for p in body.iter(W + "p"):
    if text_of(p).strip() == "Problem Statement:":
        lvl1 = copy.deepcopy(p)
        break
if lvl1 is None:
    raise SystemExit("4.1 heading not found")

for target in ["Implementation and Hardware:",
               "Video-blind control family:",
               "Probes that isolate the terms of the metric:"]:
    for p in list(body.iter(W + "p")):
        if text_of(p).strip() == target:
            p.getparent().replace(p, set_text(copy.deepcopy(lvl1), target))
            print(f"  [OK] {target}")
            break

tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
