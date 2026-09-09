"""
Repair what the restructure broke.

Moving a third of the paper into appendices left six things behind, and all of
them are the same kind of fault: a sentence that was correct while its
neighbour was three pages away and is not correct now that its neighbour is in
a different part of the document.

  1. Section 8 pointed at "Sections 4.3 and 5.7", which no longer exist. The
     rename map matched the singular forms only, so the plural phrase slipped
     through.
  2. Section 6 analyses "Stage 4" and "Stage 5" of the pipeline. Those stages
     are now described only in Appendix A, and neither sentence said so.
  3. Three references collapsed to the appendix they already sit in, which
     makes them vacuous: "which Appendix A records" inside Appendix A.
  4. Section 8 uses "the three engines" and the fusion weights without
     introducing either, now that Appendix A holds the description.
  5. "No result in Sections 5 to 7 depends on any of it" is contradicted by
     Section 6, which analyses two of the system's stages. Stated three times.
  6. The new Appendix B introduction repeats, almost word for word, the two
     sentences that already opened B.1; the Appendix A introduction repeats a
     sentence from Section 2.5.

  python fixes5.py
"""
from __future__ import annotations

import pathlib
import sys

BODY = [
    # ---- 5: the claim Section 6 contradicts, in all three places ---------
    ("Section 4 defines the video-blind controls and the probes, Section 5 "
     "reports the measurements, and Section 6 analyses why the composition "
     "fails. Sections 7 to 9 give the corrected protocol, the limitations and "
     "the conclusion. Appendix A describes the system we entered and Appendix "
     "B the evaluation we ran on it before any submission was scored; no "
     "result in Sections 5 to 7 depends on either.",
     "Section 4 defines the video-blind controls and the probes, Section 5 "
     "reports the measurements, and Section 6 analyses why the composition "
     "fails. Sections 7 to 9 give the corrected protocol, the limitations and "
     "the conclusion. Appendix A describes the system we entered and Appendix "
     "B the evaluation we ran on it before any submission was scored. No "
     "measurement in Section 5 and no recommendation in Section 7 depends on "
     "either; Section 6 draws on two stages of that system as examples, and "
     "says where they are described."),

    ("The system we entered in the competition is the case study rather\n"
     "than the contribution, and is described in Appendix A; nothing in "
     "Sections 5\nto 7 depends on it.",
     "The system we entered in the competition is the case study rather than "
     "the contribution, and is described in Appendix A. The measurements in "
     "Section 5 do not depend on it."),

    ("It is not the contribution of this\npaper, and no result in Sections 5 "
     "to 7 depends on any of it.",
     "It is not the contribution of this paper, and no measurement in\n"
     "Section 5 depends on any of it."),

    # ---- 6: Appendix A's intro repeated Section 2.5 ---------------------
    ("This appendix describes the training-free ensemble we entered in the\n"
     "competition. It is the instrument that produced the finding and the "
     "case study\nthat illustrates it, and it is reported here in full "
     "because Section 6\nidentifies two of its stages as mechanisms by which "
     "a system can inflate a\ntiming metric without anticipating anything.",
     "This appendix describes the training-free ensemble we entered in the\n"
     "competition. It is reported in full for one reason: Section 6 identifies "
     "two\nof its post-processing stages as mechanisms by which a system can "
     "inflate a\ntiming metric without anticipating anything, and a reader "
     "should be able to\ncheck that reading against the implementation."),

    # ---- 6: Appendix B's intro repeated B.1's opening --------------------
    ("This appendix keeps the evaluation we ran before any submission was "
     "scored,\ntogether with the reckoning that followed. It is retained "
     "because it is the\ncase study Section 6 analyses: an honest record of "
     "what a team measures when\nthe corpus publishes no labels, and of what "
     "that measurement turned out to be\nworth.",
     "This appendix keeps the evaluation we ran on that system before any\n"
     "submission was scored, and the reckoning that followed once the "
     "leaderboard\nhad spoken."),

    # ---- 2: Section 6 must say where its stages are described ------------
    ("Our pipeline does this at Stage 4, and an earlier version of this work "
     "reported 100\\% compliance as a result.",
     "The system of Appendix A does this at the fourth of its five "
     "post-processing stages, a monotone clamp (Appendix A.2), and an earlier "
     "version of this work reported 100\\% compliance as a result."),

    ("Our Stage 5 did this at a compression factor of 1.3, and whatever "
     "earliness it bought cannot be obtained by any system running in real "
     "time.",
     "The fifth stage of the same pipeline, a temporal resampling described in "
     "Appendix A.2, did this at a compression factor of 1.3, and whatever "
     "earliness it bought cannot be obtained by any system running in real "
     "time."),

    # ---- 1 and 4: Section 8's broken pointer and its undefined terms -----
    ("Two admissions about our own system belong here, because Sections 4.3 "
     "and 5.7 point at them. The fusion weights of 0.55, 0.25 and 0.20 were "
     "chosen by searching over the evaluation corpus. That is tuning on the "
     "test set, and it sits badly with a zero-shot claim; the released "
     "configuration therefore weights the three engines equally, chosen a "
     "priori, and the leaderboard score we report is the entry that used the "
     "searched ones.",
     "Two admissions about the system we entered belong here; Appendix A "
     "describes it, and Appendix B reports what we measured on it. It "
     "combines three scoring engines, and the weights with which their three "
     "curves are summed --- 0.55, 0.25 and 0.20 --- were chosen by searching "
     "over the evaluation corpus. That is tuning on the test set, and it sits "
     "badly with a zero-shot claim; the released configuration therefore "
     "weights the three equally, chosen a priori, and the leaderboard score we "
     "report is the entry that used the searched ones."),

    # ---- 3: three pointers that collapsed onto themselves ----------------
    ("Those values were not chosen a priori; they were searched over the "
     "evaluation corpus, which Appendix A records as a limitation.",
     "Those values were not chosen a priori; they were searched over the "
     "evaluation corpus, which Section 8 records as a limitation."),

    ("though two of the three read the same frame-difference statistic "
     "(Appendix A).",
     "though two of the three read the same frame-difference statistic "
     "(Appendix A.2)."),

    ("already beats the full pipeline on it at 21.7, for the reason Appendix "
     "B gives.",
     "already beats the full pipeline on it at 21.7, for the reason "
     "Appendix B.1.1 gives."),
]

# The abstract was described as trimmed and was barely shortened. This is the
# actual trim: three passages that restate what the sentence beside them
# already says.
ABSTRACT = [
    ("We show that this composition fails structurally rather than "
     "incidentally. The discrimination terms are bounded on the unit interval "
     "while the earliness terms scale with clip length, and the earliness "
     "terms have a trivial global maximiser: a constant risk score above the "
     "threshold attains, on every clip, the largest value that clip admits.",
     "The composition fails structurally rather than incidentally: the "
     "discrimination terms are bounded on the unit interval, the earliness "
     "terms scale with clip length, and the earliness terms have a trivial "
     "global maximiser --- a constant risk score above the threshold attains, "
     "on every clip, the largest value that clip admits."),

    ("The same probes place the withheld accident-onset distribution, "
     "recovering a mean of 120.1 frames in a 150-frame clip and bounding what "
     "can lie beyond frame 140, and establish a second result: four curves "
     "that cross the threshold at the same frame, and never fall below it "
     "afterwards, score 0.091 apart, a spread the published metric definition "
     "cannot produce. The benchmark's stated formula therefore does not "
     "reproduce its own scorer.",
     "The same probes place the withheld accident-onset distribution, "
     "recovering a mean of 120.1 frames in a 150-frame clip, and establish a "
     "second result: four curves that cross the threshold at the same frame "
     "and never fall below it afterwards score 0.091 apart, a spread the "
     "published definition cannot produce. The benchmark's stated formula does "
     "not reproduce its own scorer."),
]


def apply(path, pairs, label):
    p = pathlib.Path(path)
    s = p.read_text()
    bad = 0
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            print(f"  [!! {n}] {label}: {old[:66]!r}")
            bad += 1
            continue
        s = s.replace(old, new)
    p.write_text(s)
    return bad


def main() -> None:
    bad = apply("body_clean.tex", BODY, "body")
    bad += apply("abstract.tex", ABSTRACT, "abstract")
    if bad:
        print(f"\n{bad} replacement(s) did not apply.")
        sys.exit(1)
    print(f"\n{len(BODY) + len(ABSTRACT)} corrections applied")


if __name__ == "__main__":
    main()
