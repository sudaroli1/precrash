"""
Reframe the opening so the general result comes first.

The introduction previously spent two paragraphs describing the system we
entered before saying, in a one-line paragraph, "this paper is not about that
system". A reader skimming under time pressure -- which is what a desk editor
is -- met a system paper and stopped there.

The claim about composite metrics now arrives in the second paragraph and the
demonstration in the fourth. How we came to the question is compressed to two
sentences and moved after both, which is the order in which the paper actually
wants to be read.

The abstract is trimmed rather than rewritten: it already led with the general
claim, but at 332 words it was long for a journal, and several clauses said
twice what the tables say once.

  python reframe.py
"""
from __future__ import annotations

import pathlib
import sys

OLD_OPENING = r"""We built a training-free system to find out: a zero-shot ensemble scoring each frame of a dashcam clip for collision risk from vision--language similarity, a motion signal and a text-anchored prior, with no accident supervision at any stage. We entered it in a public competition, where the organisers scored it against ground truth we never held. It placed ninth of thirteen.

This paper is not about that system. It is about what we found when we tried to understand the score it received.

The competition scores submissions with a composite metric: a weighted sum of average precision (AP), area under the ROC curve (AUC), and two earliness measures --- time-to-accident (TTA) and stable time-to-accident (STTA) --- each read at a risk threshold of 0.5. The composition is intuitive, and it is becoming common. An anticipator should both separate accident clips from normal driving and raise the alarm early, and summing terms that measure each seems a reasonable way to ask for both.

It is not. AP and AUC are bounded on the unit interval; TTA and STTA are counted in frames and bounded only by the clip, which is 150 frames here. Summing a bounded quantity with an unbounded one lets the unbounded one decide the ranking, and here the timing terms are worth 5.70 times what discrimination contributes to a submission that reads nothing. Worse, the two share a trivial global maximiser: a score that exceeds 0.5 at frame 0 and never falls attains, on every clip, the largest value that clip admits for both. No method that reads the video can exceed it; a method can only match it and then compete on AP and AUC, worth 0.35105 to a blind submission.

So we submitted the constant. It scored 2.35291 against a winning score of 2.50165, matching the eighth-placed team at the precision the leaderboard displays."""

NEW_OPENING = r"""Anticipation is increasingly scored by a composite metric: a weighted sum of a discrimination term --- average precision (AP), area under the ROC curve (AUC) --- and an earliness term, the time between the alarm and the collision, read at a fixed risk threshold. The composition is intuitive. An anticipator should separate accident clips from ordinary driving and should raise the alarm early, and summing terms that measure each seems a reasonable way to ask for both.

It is not, and the reason is a difficulty of type rather than of tuning. AP and AUC are bounded on the unit interval. An earliness term counted in frames is bounded only by the length of the clip. Summing a bounded quantity with an unbounded one lets the unbounded one decide the ranking, and it makes the ranking depend on how the corpus was windowed. Worse, an earliness term read at a threshold has a trivial global maximiser: a risk score that exceeds the threshold at frame 0 and never falls attains, on every clip, the largest value that clip admits. No method that reads the video can exceed it. A method can only match it, and then compete on the bounded half.

That argument is available from the definitions alone. What has been missing is a measurement of what it costs in practice, and the measurement is not easy to obtain: the benchmarks that use these metrics withhold their labels, and the competitions that use them do not publish the weights. This paper supplies it. On a live accident-anticipation benchmark --- 1,417 dashcam clips, thirteen teams, a scorer we never saw --- a constant risk score of 0.51 that reads no pixels scores \textbf{2.35291} against a winning score of 2.50165, matching the eighth-placed team at the precision the leaderboard displays and exceeding five of thirteen entries. Using a family of submissions that likewise read nothing, designed so that the undisclosed terms cancel when two are differenced, we then recover the metric's parts from outside the competition: the timing terms are worth \textbf{5.70 times} what discrimination contributes to such a submission, and the score falls at almost exactly one sixtieth of a point for every frame of delay.

We came to the question the awkward way. We built a training-free ensemble, entered it in that competition, and placed ninth of thirteen. This paper is not about that system --- it scores below the constant, and we report it as the case study --- but it is what made the metric worth examining."""

OLD_STRUCTURE = r"""Section 2 reviews the field's evaluation practice, Section 3 states the benchmark and its metric, and Section 4 describes the system, the controls and the probes. Section 5 reports the measurements, Section 6 analyses why the composition fails, and Sections 7 to 9 give the protocol, the limitations and the conclusion."""

NEW_STRUCTURE = r"""Section 2 reviews the field's evaluation practice and Section 3 states the benchmark and its metric. Section 4 defines the video-blind controls and the probes, Section 5 reports the measurements, and Section 6 analyses why the composition fails. Sections 7 to 9 give the corrected protocol, the limitations and the conclusion. Appendix A describes the system we entered and Appendix B the evaluation we ran on it before any submission was scored; no result in Sections 5 to 7 depends on either."""

OLD_CONTRIB = r"""  A case study in which we are the subject, and a seven-point protocol. Our own entry scores below the constant and we report it. The protocol's first item --- report a video-blind control with every result --- costs an afternoon and would have caught this."""

NEW_CONTRIB = r"""  A seven-point protocol, and a case study in which we are the subject. The protocol's first item --- report a video-blind control with every anticipation result --- costs a single submission and would have caught this. The case study is our own entry, which scores below the constant; Appendix B reports what we measured on it before the leaderboard, and what that turned out to be worth."""

ABSTRACT = [
    # the competition is the demonstration, not the subject
    ("We demonstrate the consequence on a live benchmark. On the Zero-shot "
     "Accident Anticipation competition (AUTOPILOT-COG; 1,417 clips curated "
     "from MM-AU; 13 teams), a constant risk score of 0.51, which reads no "
     "pixels, scores 2.35291 on the private leaderboard --- matching the "
     "eighth-placed team at the precision the leaderboard displays, exceeding "
     "five of thirteen teams including our own, and falling 0.14874 short of "
     "the winning score.",
     "We measure the consequence on a live benchmark of 1,417 dashcam clips "
     "and thirteen teams, whose labels are withheld and whose metric weights "
     "are undisclosed. A constant risk score of 0.51, which reads no pixels, "
     "scores 2.35291 on the private leaderboard: it matches the eighth-placed "
     "team at the precision the leaderboard displays, exceeds five of thirteen "
     "entries including our own, and falls 0.14874 short of the winner."),

    ("Using twenty-five video-blind submissions as probes, designed so that "
     "unknown terms cancel under differencing, we then decompose the metric "
     "from outside the competition without access to any label: average "
     "precision and area under the curve together contribute 0.35105 to a "
     "blind submission, the two timing terms contribute 2.00186 --- 5.70 times "
     "as much --- and the score falls at almost exactly one sixtieth of a "
     "point per frame of delay, confirmed by a one-frame experiment.",
     "Twenty-five such submissions, designed so that the undisclosed terms "
     "cancel when two are differenced, then decompose the metric from outside "
     "the competition and without any label: discrimination contributes "
     "0.35105, the two timing terms 2.00186 --- 5.70 times as much --- and the "
     "score falls at almost exactly one sixtieth of a point per frame of "
     "delay, confirmed by a one-frame experiment."),

    ("We report our ninth-placed entry, a training-free ensemble that scores "
     "below the constant, as the case study, and close with a protocol whose "
     "central requirement is that every anticipation result be reported "
     "alongside a video-blind control.",
     "We close with a protocol whose central requirement is that every "
     "anticipation result be reported alongside a video-blind control, which "
     "costs one submission. Our own ninth-placed entry, which scores below the "
     "constant, is the case study."),
]


def apply(path, pairs, label):
    p = pathlib.Path(path)
    s = p.read_text()
    bad = 0
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            print(f"  [!! {n}] {label}: {old[:64]!r}")
            bad += 1
            continue
        s = s.replace(old, new)
    p.write_text(s)
    return bad


def main() -> None:
    bad = apply("body_clean.tex", [
        (OLD_OPENING, NEW_OPENING),
        (OLD_STRUCTURE, NEW_STRUCTURE),
        (OLD_CONTRIB, NEW_CONTRIB),
    ], "body")
    bad += apply("abstract.tex", ABSTRACT, "abstract")
    if bad:
        print(f"\n{bad} replacement(s) did not apply.")
        sys.exit(1)
    print("  [OK] opening, contributions, structure paragraph and abstract "
          "reframed")


if __name__ == "__main__":
    main()
