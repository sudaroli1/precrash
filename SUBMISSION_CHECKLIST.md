# TMLR submission checklist

Everything in Part A is done. Parts B and C are yours — they need your
OpenReview account and an authorship attestation that only an author can make.

Files to upload:

- `latex/main.pdf` — the manuscript, 27 pages, TMLR style, anonymised
- `tmlr_supplementary.zip` — 1.2 MB, anonymised

---

## A. Done, and what was checked

**Style.** `tmlr.sty`, `tmlr.bst` and `fancyhdr.sty` are the current files from
the official `JmlrOrg/tmlr-style-file` repository. The build uses
`\usepackage{tmlr}` with no options, which is the submission form: the header
reads "Under review as submission to TMLR" and the by-line prints "Anonymous
authors / Paper under double-blind review" automatically. **Do not add the
`accepted` option until the paper is accepted** — it would print the author
block that sits commented in `main.tex`.

**Build.** 0 LaTeX errors, 0 undefined references, 0 undefined citations, 0
overfull boxes. Rebuild with:

```
cd latex
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**Citations.** TMLR's style file sets `authoryear` natbib, so the paper's
numeric brackets were converted: 57 `\citet`, 3 `\citep`, none left numeric.
The hand-written reference list has been removed and BibTeX now sets the
references from `references.bib` (32 entries).

**Sectioning.** Word's auto-numbered headings had converted to one-item lists,
so nothing in the paper was numbered by LaTeX. All 9 sections, 20 subsections
and the unnumbered compliance section are now real headings, and the numbers
LaTeX generates match every "Section 5.3" typed in the running text.

**Anonymity.** The compiled PDF was searched for every identifying string —
your name, the team name, the institution, the city, `github.com`, an email,
an ORCID — and is clean. Three things were changed to get there:

- the leaderboard row `9 | SuryaInBytes (this work)` now reads `9 | our entry`
- the `Code:` line's GitHub URL is now an anonymised availability note
- the author block is commented out of the typeset document

The supplementary ZIP is anonymised the same way: `LOG.md` says "our entry".
The PDF has no document metadata. The other twelve teams are named only as the
competition's own public leaderboard names them.

**Supplementary.** 1.2 MB, well inside the 100 MB cap, and a ZIP as the guide
requires. It holds the 27 submission files, the score log, the scripts that
generated and analysed them, and both figures, with a README explaining what
each family of curves isolates. No part of the dataset and no label is in it —
which is also what the competition's rules require.

**Dual submission.** TMLR prohibits overlap with anything published, accepted,
or under review at an archival peer-reviewed venue, but explicitly permits
overlap with workshops and with arXiv. The competition was a CVPR workshop
challenge and non-archival, and no paper on this work has been submitted
anywhere else. Nothing here breaches the policy — but see C.1.

---

## B. Before you submit

**B.1 — Fix the affiliations.** Author 1 currently reads "Independent
Researcher, Bengaluru" while authors 2 and 3 read "T. John Institute of
Technology". Decide which is right for each of you and put the final list in
`main.tex` (it is commented out and does not print during review, but it must
be correct before camera-ready). **The author roster cannot be changed after
submission, under any circumstances** — this is the one thing on the whole list
that is genuinely irreversible, so settle it now.

**B.2 — Confirm reference 23.** `singh2026metadata` (arXiv:2606.12047) has its
given names unconfirmed; the `.bib` entry carries a `note` saying so. Open the
arXiv abstract page, copy the full author list, and correct the entry. Then
rebuild. This is the only unverified item left in the bibliography — the other
31 were audited against their DOIs.

**B.3 — Post to arXiv first, and do it today.** This is the item I would not
delay. arXiv endorsement for cs.CV can take several days if you do not already
have it, and the priority risk here is live: the second-placed team published
on arXiv in June 2026, and a public write-up already notes that the weights are
undisclosed. TMLR permits an arXiv preprint and instructs reviewers not to go
looking for it, so posting does not compromise the double-blind review. Build
the preprint version by swapping the style option in `main.tex`:

```latex
\usepackage[preprint]{tmlr}    % names and affiliations printed
```

and un-commenting the author block. Keep the anonymised `main.pdf` for
OpenReview and the named one for arXiv — they are two different builds of the
same source.

**B.4 — Email the AUTOPILOT organisers.** A courtesy copy before you submit.
The paper reports that their published formula does not reproduce their own
scorer, and they should hear that from you rather than from a reviewer. It also
gives them a chance to correct the record, which can only help the paper.

**B.5 — Push the repository.** 20 commits are sitting local-only; the shell I
work through has no GitHub credentials, so this one has to be you:

```
cd <your folder>\precrash-eval
git push origin main
```

---

## C. On the OpenReview form

TMLR's author guide requires a complete, active OpenReview profile for every
author — affiliations, conflicts of interest and publication history all filled
in — before the form will accept the submission. Check each co-author's profile
now rather than at the deadline; an incomplete profile is a common cause of a
submission being held.

The form itself asks for:

**C.1 Action editor recommendation.** You may name action editors you think
suitable, and flag conflicts. Suggest editors who work on evaluation
methodology or benchmark design rather than on accident anticipation — the
paper's claim is about how a composite score behaves, and an AE who reads it as
an anticipation paper will look for a method contribution the paper does not
make and does not claim.

**C.2 Human subjects / IRB.** Not applicable. The work involves no human
subjects: it analyses a scoring function using curves generated from the frame
index, on a corpus of dashcam clips the authors did not collect.

**C.3 Funding.** State it if there is any; "none" is a valid answer.

**C.4 Competing interests.** State any. Note that you were an entrant in the
competition the paper analyses — that is disclosed in the paper itself
(Section 5.1 and the compliance section), and disclosing it again here costs
nothing and pre-empts the question.

**C.5 Broader impact statement.** Required only if the work carries significant
risk of harm. It does not, and none is included.

**C.6 The attestations.** By submitting you confirm that every author is aware
of the submission, that all of you take responsibility for the correctness and
integrity of the work, that it is original, and that everyone who qualifies as
an author is listed. TMLR's authorship criteria are all three of: substantial
intellectual contribution, contribution to drafting or revising, and full
responsibility for the published content. Make sure the list you submit
satisfies all three for every name on it.

---

## What a desk rejection would be for

Action editors may reject without review for: being out of scope, being
unreviewable for want of expertise, obvious poor quality, being unlikely to
meet the acceptance criterion, or violating the format. The format and
anonymity grounds are closed off by Part A. The one to think about is scope:
TMLR's criterion is *are the claims supported by convincing evidence*, not
*is this novel* — which suits this paper, since its evidence is 27 scored
submissions against a live leaderboard and its claim is exactly what those
submissions show. The single-benchmark limitation, which would be a real
problem at a conference, is not a defect against that criterion, and Section 8
already states it plainly.
