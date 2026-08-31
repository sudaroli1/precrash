"""Renumber the tail after inserting Section 3.

The auto-numbered section headings now run Introduction 1, Related Work 2,
Benchmark 3, Method 4. The headings after those are hard-typed numbers and no
longer agree: "4. Experimental Setup" collides with the Method section.

Experimental setup is part of the method, not a section of its own, so it
becomes a subsection of 4 — which also makes "5. Results and Discussion"
correct as printed, and matches the roadmap in Section 1.

Sections 6 (Analysis), 7 (Corrected Protocol) and 8 (Limitations) are not
written yet, so a marker goes in front of the Conclusion rather than a
speculative renumbering.
"""
import copy
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = "unp4/word/document.xml"

MARKER = ("SECTIONS 6, 7 AND 8 ARE NOT YET WRITTEN — Analysis (why the "
          "composition fails), A Corrected Protocol, and Limitations. The "
          "Conclusion below becomes Section 9, and the Future Scope section "
          "that follows it is to be rewritten as Section 8, Limitations. "
          "Section 1 states the intended order.")

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

concl = None
for p in body.iter(W + "p"):
    s = text_of(p).strip()
    if s == "4. Experimental Setup":
        set_text(p, "4.6. Experimental Setup")
        print("  [OK] Experimental Setup is now a subsection of the method")
    elif s == "6. Conclusion":
        concl = p

if concl is None:
    raise SystemExit("Conclusion heading not found")

at = list(body).index(concl)
body.insert(at, set_text(copy.deepcopy(concl), MARKER))
print("  [OK] marker inserted before the Conclusion")

tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
