"""
Pass 16: equation numbering.

Ten displayed equations, numbered 1, 1, 2, 3, 5, 5, 6, 7, 8, 9 — two numbers
used twice, (4) never used, and the last equation printed as (9). Every in-text
reference inherited the error, and one of them pointed at an equation two places
away from the one it meant.

The numbers live in their own text nodes inside the equation objects, between a
node ending in "(" and a node starting with ")", except in the first three where
the whole trailing run is a single node. Both shapes are handled.

Renumbered 1 to 10 in document order, and every reference remapped to the
equation it actually points at:

    (2) -> (3)   the CLIP danger-affinity score
    (3) -> (4)   the frame-difference motion score
    (4) -> (5)   the caption prior
    (5) -> (6)   the convex combination
    (6) -> (7)   Gaussian smoothing
    (7) -> (8)   the power transform
    (8) -> (10)  temporal compression, which was pointing at the clamp
    (4) -> (6)   the fusion-weight paragraph in 4.3
"""
import re
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
DOC = "unp7/word/document.xml"

# body-child index -> (expected fragment, correct equation number)
EQUATIONS = {
    95:  ("Wt", 1),
    136: ("flow", 2),
    157: ("clip", 3),
    164: ("flow", 4),
    167: ("caption", 5),
    173: ("raw", 6),
    176: ("smooth", 7),
    180: ("norm", 8),
    185: ("clamp", 9),
    190: ("final", 10),
}

# body-child index -> (old number, new number)
REFERENCES = {
    155: (2, 3),
    163: (3, 4),
    166: (4, 5),
    172: (5, 6),
    175: (6, 7),
    179: (7, 8),
    191: (8, 10),
    196: (4, 6),
}


def nodes(p):
    return [el for el in p.iter() if el.tag in (W + "t", M + "t")
            and el.text is not None]


def set_equation_number(p, n):
    ns = nodes(p)
    # single-node form: "………(N)"
    for el in reversed(ns):
        if re.search(r"\(\s*\d{1,2}\s*\)", el.text):
            el.text = re.sub(r"\((\s*)\d{1,2}(\s*)\)", rf"(\g<1>{n}\g<2>)",
                             el.text)
            return True
    # split form: a node that is only digits, wrapped in "(" and ")"
    for i, el in enumerate(ns):
        if el.text.strip().isdigit():
            before = ns[i - 1].text if i else ""
            after = ns[i + 1].text if i + 1 < len(ns) else ""
            if before.rstrip().endswith("(") and after.lstrip().startswith(")"):
                el.text = str(n)
                return True
    return False


def main():
    tree = etree.parse(DOC)
    body = tree.getroot().find(W + "body")

    ok = 0
    for idx, (frag, n) in sorted(EQUATIONS.items()):
        p = body[idx]
        text = "".join(el.text for el in nodes(p))
        if frag not in text:
            print(f"  [!!] para {idx}: expected '{frag}', found "
                  f"{text[:50]!r} — skipped")
            continue
        if set_equation_number(p, n):
            print(f"  [OK] equation at para {idx:4d}  ->  ({n})")
            ok += 1
        else:
            print(f"  [!!] para {idx}: no number node found")
    print(f"  {ok} of {len(EQUATIONS)} equations renumbered")

    done = 0
    for idx, (old, new) in sorted(REFERENCES.items()):
        p = body[idx]
        hit = False
        for el in p.iter(W + "t"):
            if el.text and re.search(rf"\(\s*{old}\s*\)", el.text):
                el.text = re.sub(rf"\((\s*){old}(\s*)\)",
                                 rf"(\g<1>{new}\g<2>)", el.text, count=1)
                hit = True
                break
        print(f"  [{'OK' if hit else '!!'}] reference at para {idx:4d}  "
              f"({old}) -> ({new})")
        done += hit
    print(f"  {done} of {len(REFERENCES)} references remapped")

    tree.write(DOC, xml_declaration=True, encoding="UTF-8", standalone=True)


if __name__ == "__main__":
    main()
