"""
Pass 12: remove Table 2 and Table 4, and renumber the rest.

Neither table carries a measurement. Table 2 ("Justification showing the
importance of CLIP ViT-L/14") is a list of reasons alternatives were rejected,
with no numbers attached; Table 4 ("Advantages of Using Ensembling") is a grid
of ticks and crosses restating the paragraph above it. Both read as advocacy
rather than evidence, which is the wrong register for a paper whose argument is
that a leaderboard number was not evidence either.

What each table said is preserved in the prose that introduced it — the
rejection reasons become a sentence, the tick grid becomes a clause.

Every remaining table moves up: 3 becomes 2, 5 becomes 3, and so on to 15
becoming 13. In-text references are remapped in the same pass, in one
substitution, so that no number is mapped twice.
"""
import re
import copy
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DOC = "unp6/word/document.xml"

# old -> new, applied simultaneously
MAP = {3: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 13: 11,
       14: 12, 15: 13}

PROSE = [
    ("ViT-L/14 was chosen over the smaller variants for its finer spatial "
     "detail. Table 2 sets out why the alternatives were rejected.",
     "ViT-L/14 was chosen over the smaller variants for its finer spatial "
     "detail. Convolutional backbones were rejected because they cannot score a "
     "frame against a text prompt without accident-specific training, object "
     "detectors because they locate objects rather than assess risk, and the "
     "larger instruction-tuned models because their per-frame decoding cost "
     "exceeded the compute available."),

    ("Table 4 sets the three against one another; ensembling is intended to let "
     "each cover the others' blind spot.",
     "Ensembling is intended to let each engine cover the others' blind spot: "
     "the semantic score supplies what is in the scene, the motion proxy "
     "whether it is moving abnormally, and the caption prior when in the clip "
     "risk should rise."),
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


tree = etree.parse(DOC)
body = tree.getroot().find(W + "body")

# 1. the prose that pointed at the two tables
for old, new in PROSE:
    for p in body.iter(W + "p"):
        if old in text_of(p):
            set_text(p, text_of(p).replace(old, new))
            break
    else:
        print(f"  [!!] prose not found: {old[:50]}")
print("  [OK] prose rewritten to carry what the two tables said")

# 2. the captions and the tables themselves
for cap in ("Table 2: Justification showing", "Table 4: Advantages of Using"):
    for p in list(body.iter(W + "p")):
        if text_of(p).strip().startswith(cap):
            nxt = p.getnext()
            body.remove(p)
            if nxt is not None and nxt.tag == W + "tbl":
                body.remove(nxt)
                print(f"  [OK] removed '{cap}...' and its table")
            else:
                print(f"  [!!] '{cap}...' caption removed but table not adjacent")
            break

# 3. renumber every surviving reference in one pass
pat = re.compile(r"\bTable (\d+)\b")


def remap(m):
    n = int(m.group(1))
    return f"Table {MAP.get(n, n)}"


changed = 0
for p in body.iter(W + "p"):
    t = text_of(p)
    if pat.search(t):
        new = pat.sub(remap, t)
        if new != t:
            set_text(p, new)
            changed += 1
print(f"  [OK] renumbered table references in {changed} paragraphs")

tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)
