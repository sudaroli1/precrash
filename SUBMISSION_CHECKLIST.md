# TMLR submission — the steps

Written against TMLR's own pages, read on 1 September 2026:
[submissions](https://www.jmlr.org/tmlr/submissions.html),
[editorial policies](https://www.jmlr.org/tmlr/editorial-policies.html),
[author guide](https://www.jmlr.org/tmlr/author-guide.html),
[FAQ](https://www.jmlr.org/tmlr/faq.html).

**The two files you upload:**

- `latex/main.pdf` — 28 pages, TMLR style, anonymised
- `tmlr_supplementary.zip` — 1.2 MB, anonymised

**Nothing else.** In particular do not upload `latex/` — `convert.py` in it
contains your repository URL.

---

## Step 0 — push the repository

There is nothing left to commit; the working tree is clean and 22 commits are
sitting on `main` waiting to go out. From a terminal on your machine:

```
cd C:\Users\sudar\Desktop\Preparation\tier-a-prep\precrash-eval
git push origin main
```

If it asks for a password, GitHub no longer accepts your account password over
HTTPS — generate a personal access token at
`github.com/settings/tokens` (classic token, `repo` scope) and paste that as
the password. Verify with `git status`, which should then say
`Your branch is up to date with 'origin/main'`.

I cannot run this for you: the shell I work through has no GitHub credentials,
and it should not have them.

---

## Step 1 — settle the author list. Do this first, and do not rush it.

> "No authors may be added or removed after submission. There are no
> exceptions to this policy." — editorial policies

Everything else on this list is reversible. This is not.

Author 1 currently reads "Independent Researcher, Bengaluru"; authors 2 and 3
read "T. John Institute of Technology". Decide the final list and the final
affiliation for each person, and put it in `latex/main.tex` (it is commented
out and does not print during review, but it must be right for the
camera-ready and for arXiv).

TMLR's authorship test is all three of:

1. a substantial intellectual contribution to the work;
2. a contribution to drafting or revising the submission;
3. full responsibility for all content in the published work.

Every name on the list must satisfy all three.

## Step 2 — every author needs a complete OpenReview profile

The author guide requires "complete and active OpenReview profiles" with
affiliations, conflicts of interest and publication history. An incomplete
co-author profile is a common reason a submission stalls, so check each one
now rather than at upload time.

Each author, at `openreview.net`:

- signs up or signs in, and confirms the email address on the profile
- fills in **Education & Career History** — every affiliation, with dates, and
  no gaps (OpenReview flags gaps)
- fills in **Advisors & Other Relations**, and **Expertise**
- confirms the name matches the one you will type on the submission form

Conflicts of interest, per the policies, are: any domain you were affiliated
with in the past three years, anyone you collaborated with in the past three
years, family relationships, and advisor/advisee relationships.

## Step 3 — confirm reference 23

`singh2026metadata` (arXiv:2606.12047): the surnames are confirmed, the given
names are not, and the entry currently prints surnames only. Open the arXiv
abstract page, copy the full author list into `latex/references.bib`, and
rebuild. There is a comment above the entry marking this. It is the only
unverified item in the bibliography; the other 31 were audited against their
DOIs.

## Step 4 — post to arXiv, before you submit

Do this today if you are going to do it at all. arXiv endorsement for cs.CV
can take several days if you do not already have it, and the priority risk is
real: the second-placed team published on arXiv in June 2026.

TMLR explicitly permits it:

> "It is acceptable for a submission to overlap with the author's previous work
> if it was shared at venues or tracks that are publicly declared … to be
> non-archival, such as workshops, or on preprint servers such as arXiv."

and the FAQ confirms reviewers are told not to go looking. Build the named
version by swapping one line in `latex/main.tex`:

```latex
% \usepackage{tmlr}
\usepackage[preprint]{tmlr}     % names and affiliations printed
```

and un-commenting the `\author{...}` block. Rebuild, and keep the two PDFs
separate — the anonymous one goes to OpenReview, the named one to arXiv.

## Step 5 — email the AUTOPILOT organisers

A courtesy copy before you submit. The paper reports that their published
formula does not reproduce their own scorer; they should hear it from you
rather than from a reviewer, and it gives them a chance to correct the record.

## Step 6 — submit

Go to **https://openreview.net/group?id=TMLR** and use the submission button
there. TMLR is rolling — there is no deadline and no cycle to wait for.

What the form asks for, and what to put:

| Field | What to enter |
|---|---|
| Title | Unbounded Timing Terms Make a Composite Accident-Anticipation Score Video-Blind: Evidence from a Live Benchmark |
| Authors | the list settled in Step 1, each matched to their OpenReview profile |
| Abstract | paste from the PDF (plain text, one paragraph) |
| PDF | `latex/main.pdf` |
| Supplementary material | `tmlr_supplementary.zip` |
| Human subjects / IRB | **Not applicable.** No human subjects: the work analyses a scoring function using curves generated from the frame index, on dashcam clips the authors did not collect. |
| Funding | state it, or "none" |
| Competing interests | disclose that you were an entrant in the competition the paper analyses. It is already stated in the paper (§5.1 and the compliance section); saying it again here costs nothing and pre-empts the question. |
| Broader impact statement | not required — it is required only where the work carries significant risk of harm, and this does not. |
| Previous submission URL | leave blank; this is a first submission. |

Before you press submit you are confirming that every author is aware of the
submission, that all of you take responsibility for the correctness and
integrity of the work, that it is original, and that everyone who qualifies as
an author is listed.

## Step 7 — recommend action editors, by email, after submitting

> "You will receive an email once your submission has been made, asking you to
> recommend potential Action Editors that would be appropriate for your
> submission." — submissions page

So this is not on the form; it arrives afterwards, and it matters. Pick from
the [editorial board](https://www.jmlr.org/tmlr/editorial-board.html) and
choose people who work on **evaluation methodology, benchmarking or
measurement**, not on accident anticipation. An AE who reads this as an
anticipation paper will look for a method contribution it does not make and
does not claim; an AE who reads it as a paper about how a composite score
behaves will see the evidence for what it is. Flag any conflicts in the same
reply.

---

## What happens next

- An action editor is assigned **within a week**, and checks eligibility.
- **At least three qualified reviewers** are assigned.
- An open-ended rebuttal, discussion and revision phase follows. You may
  revise as many times as you like; leave a comment on the OpenReview page
  each time so reviewers see the revision.
- You respond **no later than two weeks after the third review** arrives.
- Decision: **accept as is, accept with minor revisions, or reject.**
- Target: **a final decision about nine weeks after submission.** Longer papers
  can take longer, and at 28 pages this one is on the long side.
- On acceptance: a camera-ready with `\usepackage[accepted]{tmlr}`, the author
  block restored, and `\month` / `\year` / `\openreview` filled in. Papers may
  additionally be awarded Featured, Survey or Reproducibility certification.
- Everything is published CC BY 4.0 with copyright retained by you.
- You can withdraw any time before a decision.

---

## What is already done, and what was checked

**Style.** `tmlr.sty`, `tmlr.bst` and `fancyhdr.sty` verified **byte-identical**
to the official `JmlrOrg/tmlr-style-file` files. This matters: the submissions
page warns that "any changes to the stylefile or template that alters the
formatting, font, or layout of the manuscript may result in rejection without
review." `main.tex` matches the official template's preamble —
`\documentclass[10pt]{article}`, `\usepackage{tmlr}` with no options. Nothing
that alters font, margins or layout has been added.

**Build.** 0 LaTeX errors, 0 undefined references, 0 undefined citations, 0
overfull boxes, 28 pages. Rebuild with:

```
cd latex
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**Tool-use disclosure.** The FAQ requires that authors who used an LLM "mention
explicitly in their submission that they have used this tool, as a footnote on
the first page." That footnote is now on page 1, and states what the assistant
did and that the research question, design, claims and conclusions are yours.

**Anonymity.** The compiled PDF was searched for every identifying string — your
name, the team name, the institution, the city, `github.com`, emails, ORCIDs,
the system name "PreCrash" — in its text, its metadata and its raw object
streams, and is clean. It carries no `/Author` or `/Title` and no XMP stream.
Three things were changed to get there: the leaderboard row naming the team,
the code-availability line's repository URL, and the author block.

Be aware of one thing, and decide it knowingly rather than discover it: the
paper reproduces the public leaderboard with the other twelve teams' real
names, cites the competition URL, and states your rank and score. Anyone can
resolve your team name in one click. That is **not** a TMLR violation — the
double-blind rules bind reviewers, and the PDF carries no author identity — but
it is a stronger de-anonymiser than an arXiv preprint. It is also unavoidable
given what the paper is about.

**Supplementary.** 1.2 MB against a 100 MB cap, and a ZIP as the guide requires.
The 26 submission files, the score log, the scripts that generated and analysed
them, both figures, and a README explaining what each family of curves
isolates. Anonymised — `LOG.md` says "our entry". No dataset content and no
labels, which is what the competition's rules require.

**Dual submission.** TMLR prohibits overlap with anything published, accepted or
under review at an archival peer-reviewed venue, and explicitly permits overlap
with workshops and arXiv. The competition was a CVPR workshop challenge and
non-archival; nothing here has been submitted elsewhere.

**Audited three times** — against the paper's own tables, then against the code
and the score log, then against the corrections themselves. Roughly ninety
defects fixed, including three that mattered: the code contradicted the paper's
claim that the three engines are independent; the submission count was wrong in
four places; and a six-frame timing gain was claimed that no table supports.
`claude/precrash-audit-findings.md` in the project has the full record.

**Desk-rejection risk.** Assessed against all five grounds the policies give
(out of scope, no reviewer expertise, obviously poor quality, unlikely to meet
the criterion, format violation). None applies. TMLR's criterion is *are the
claims supported by accurate, convincing and clear evidence* and *would some of
TMLR's audience be interested* — not conference novelty. The single-benchmark
limitation, which would be a real problem at a conference, is not a defect
against that criterion, and Section 8 states it plainly.
