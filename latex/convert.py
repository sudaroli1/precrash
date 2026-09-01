"""
docx -> LaTeX, in one pass.

Pandoc's output needs cleaning in ways that are easier to do all at once than
incrementally: the docx carries ad-hoc paragraph styling that becomes stray
headings, its equations arrive with Unicode that a TeX install without a
Unicode font cannot set, and pandoc's table column widths use a \\real macro it
assumes the class provides.

  python convert.py          # writes body_clean.tex and abstract.tex
"""
import re
import subprocess
import pathlib

RAW = pathlib.Path("body.tex")
subprocess.run(["pandoc", "../paper/Precrash.docx", "-o", str(RAW),
                "--extract-media=.", "-t", "latex", "--wrap=preserve"],
               check=True)

s = RAW.read_text()

# --- structure -----------------------------------------------------------
s = re.sub(r'\\hypertarget\{[^}]*\}\{%\n', '', s)
s = re.sub(r'\\texorpdfstring\{(.*?)\}\{.*?\}', r'\1', s, flags=re.S)
s = re.sub(r'\\label\{[^}]*\}', '', s)
s = re.sub(r'\\addcontentsline\{toc\}[^\n]*\n', '', s)
s = s.replace('\\textquotesingle ', "'").replace('\\textquotesingle', "'")
s = s.replace('{[}', '[').replace('{]}', ']')

# headings: strip the bold wrapper, and demote anything too long to be one
s = re.sub(r'\\((?:sub)*section)(\*?)\{\\textbf\{(.*?)\}\}', r'\\\1\2{\3}', s)


def demote(m):
    kind, star, text = m.group(1), m.group(2), m.group(3)
    if len(text) > 90:                      # a paragraph, not a heading
        return text
    if not text.strip():                    # an empty one
        return ''
    return f'\\{kind}{star}{{{text}}}'


for _ in range(3):
    s = re.sub(r'\\((?:sub)*section)(\*?)\{((?:[^{}]|\{[^{}]*\})*)\}',
               demote, s)

# --- characters ----------------------------------------------------------
SUP = {'\u207b': '-', '\u2070': '0', '\u00b9': '1', '\u00b2': '2',
       '\u00b3': '3', '\u2074': '4', '\u2075': '5', '\u2076': '6',
       '\u2077': '7', '\u2078': '8', '\u2079': '9'}
s = re.sub('[' + ''.join(SUP) + ']+',
           lambda m: '\\textsuperscript{' + ''.join(SUP[c] for c in m.group(0)) + '}',
           s)
SUB = {'\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2090': 'a',
       '\u1d62': 'i', '\u209c': 't', '\u2098': 'm', '\u1d9c': 'c',
       '\u1d04': 'c'}
for a, b in SUB.items():
    s = s.replace(a, '\\textsubscript{' + b + '}')
for a, b in [('\u2705', '\\checkmark '), ('\u274c', '--'),
             ('\u2009', '\\,'), ('\u202f', '\\,'), ('\u00a0', '~'),
             ('\u0394', '$\\Delta$'), ('\u02e3', '\\textsuperscript{x}')]:
    s = s.replace(a, b)

# The hypertarget wrapper contributed a closing brace that the heading
# rewrite above leaves stranded. Balance every heading line, and drop lines
# that are now nothing but an orphan brace.
def balance(line):
    if line.startswith('\\') and 'section' in line[:14]:
        d = line.count('{') - line.count('}')
        while d < 0:
            line = line[::-1].replace('}', '', 1)[::-1]
            d += 1
    return line


s = '\n'.join(balance(l) for l in s.split('\n'))
s = re.sub(r'\n\}\n', '\n', s)
s = re.sub(r'(?m)^\}$\n', '', s)

# --- tidy ----------------------------------------------------------------
s = s.replace('./media/', 'media/')
s = re.sub(r'\n\n\\\\\n', '\n\n', s)
s = re.sub(r'\n{3,}', '\n\n', s)

# --- split front matter from body ----------------------------------------
i = s.find('\\section{Introduction}')
front, body = s[:i], s[i:]
pathlib.Path("body_clean.tex").write_text(body)

m = re.search(r'\\textbf\{Abstract\}\s*(.*?)\n\n', front, re.S)
abstract = m.group(1).strip() if m else ''
k = re.search(r'\\textbf\{(Accident anticipation,.*?)\}', front, re.S)
keywords = k.group(1).strip() if k else ''
pathlib.Path("abstract.tex").write_text(
    "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n"
    "\\noindent\\textbf{Keywords:} " + keywords + "\n\n"
    "\\noindent\\textbf{Code:} \\url{https://github.com/sudaroli1/precrash}\n")

print(f"body {len(body.split())} words, abstract {len(abstract.split())} words")
