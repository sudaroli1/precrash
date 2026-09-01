"""
Pass 2 of the pre-submission audit: the corrections that needed facts checked
against the code and the score log rather than against the paper's own tables.

Three of these are substantive.

**The third engine is not a captioning model.** `src/engines/nlp_scorer.py`
says so in its own docstring: it computes the mean grayscale inter-frame
difference over the last third of the clip -- the same statistic the motion
engine uses -- thresholds it at 20 and 10, and selects one of three fixed
strings written into the source. The sentence encoder never sees anything else.
So the modality has three reachable states across the whole corpus, and, more
importantly, two of the three "independent" engines are functions of one
signal. Section 4.2 claimed three independent modalities and described a model
that captions each frame. Corrected, and the correction explains the ablation:
a modality with three states posts an earlier mean crossover than the full
ensemble because a curve that is nearly the same everywhere suits a benchmark
whose events sit at a nearly constant position.

**Twenty-six submissions, not twenty-four, and all on one day.** `scores.csv`
has 26 rows: 25 video-blind curves we generated plus the organisers' own sample
resubmitted verbatim, all dated 31 August 2026. The count appeared four times
and the two-day span once.

**The six-frame gain is not in any table.** Section 4.2 and Section 7 both
attribute six frames of time-to-accident to the temporal compression. Table 12
puts C6 at 23.9 and C7 at 22.7 -- 1.2 frames -- and C7 bundles the compression
with the clamp, so the stage is not separately measured anywhere in the paper.
The magnitude is withdrawn. The structural point does not depend on it: an
operation that reports at frame t a value computed from frame 1.3t cannot run
in real time whatever it buys.

The rest are self-consistency: a contribution bullet that still said three
curves where the abstract says four; a warning horizon quoted as 4.23 against a
table that now reads 4.24; a claim that the ramp scores below three curves when
Table 8 shows it above one of them; a promise that Section 5 reports metrics
with the clamp disabled, which it does not; two tables calling a configuration
"best performance" when their own numbers rank another above it; a six-point
protocol with seven points; and the notation collisions -- alpha, gamma and
sigma each carrying two meanings, and p_clip printing three different ways.

  python fixes2.py
"""
from __future__ import annotations

import pathlib
import sys

# ---------------------------------------------------------------- abstract
ABSTRACT = [
    ("Using twenty-four video-blind submissions as probes",
     "Using twenty-five video-blind submissions as probes"),

    ("matching the eighth-place team at the precision",
     "matching the eighth-placed team at the precision"),

    ("(AUTOPILOT; 1,417 clips curated from MM-AU; 13 teams)",
     "(AUTOPILOT-COG; 1,417 clips curated from MM-AU; 13 teams)"),

    # Section 5.5 states the segment slopes are not monotone as an exact
    # linear model requires; "exactly" claims more than that allows.
    ("the score falls at exactly one sixtieth of a point per frame of delay",
     "the score falls at almost exactly one sixtieth of a point per frame of "
     "delay"),
]

# ------------------------------------------------------------------- body
BODY = [
    # ---- Section 1: the contribution bullets --------------------------
    ("Evidence that the published definition does not reproduce the scorer. "
     "Three curves crossing at the same frame, with identical values at the "
     "crossing, score 0.091 apart;",
     "Evidence that the published definition does not reproduce the scorer. "
     "Four curves crossing at the same frame, none of which dips below it "
     "afterwards, score 0.091 apart;"),

    ("A case study in which we are the subject, and a six-point protocol.",
     "A case study in which we are the subject, and a seven-point protocol."),

    # ---- Section 3.1: entrants/participants/teams are distinct counters
    ("drew 59 entrants, 20 participants and 13 teams across 194 submissions.",
     "drew 59 entrants, of whom 20 went on to submit, forming 13 teams "
     "between them, across 194 submissions."),

    # ---- Section 3.4 ---------------------------------------------------
    ("The evaluation page defines the score as a weighted average of four "
     "quantities:",
     "The evaluation page defines the score as a weighted sum of four "
     "quantities:"),

    # ---- Section 4.1: the lead-in asserted what item 3 now denies ------
    ("The considered problem is anticipating accidents without any training "
     "data and it is expected to generalize well only with pretrained "
     "knowledge. The formal definition of the problem is as follows:",
     "The considered problem is anticipating accidents without any training "
     "data, using pretrained knowledge alone. The formal definition of the "
     "problem is as follows:"),

    ("and the dimension of each frame denotes the RGB observation at time t "
     "as \\(f_{t} \\in \\mathbb{R}^{H \\times W \\times 3}\\). The accident "
     "anticipation problem thus denotes a learning predictive mapping that "
     "predicts the risk score for each frame.",
     "with \\(m = 150\\) for every clip in this corpus, and each frame the "
     "RGB observation at time \\(t\\), \\(f_{t} \\in \\mathbb{R}^{H \\times "
     "W \\times 3}\\), where \\(H\\) and \\(W\\) are the frame height and "
     "width. The anticipation problem is then a predictive mapping that "
     "assigns a risk score to every frame of the clip."),

    # equation (1) mapped a clip to a single scalar
    ("\\[g\\ :\\ V_{s} \\longrightarrow \\rho_{t} \\in "
     "\\lbrack 0,1\\rbrack\\ldots\\ldots\\ldots.(1)\\]",
     "\\[g\\ :\\ V_{s} \\longrightarrow \\left( \\rho_{1},\\ldots,\\rho_{m} "
     "\\right) \\in \\lbrack 0,1\\rbrack^{m}\\ldots\\ldots\\ldots.(1)\\]"),

    # ---- Section 4.2: what the three engines actually are --------------
    ("PreCrash is modular and decoupled. Three pretrained engines --- a "
     "vision--language semantic engine, a motion engine and a text-anchored "
     "prior --- run independently, sharing no weights or activations, and are "
     "combined only at a late fusion layer. Each is matched to its own "
     "representational domain, so no single model carries the whole burden of "
     "reasoning about appearance, motion and semantics at once, and any "
     "engine can be replaced without disturbing the others. That separation "
     "is also what makes the per-modality failure analysis of Section 5.6.1 "
     "possible.",
     "The system is modular in implementation. Three scoring engines --- a "
     "vision--language semantic engine, a motion engine and a text-anchored "
     "temporal prior --- produce per-frame curves independently, sharing no "
     "weights or activations, and are combined only at a late fusion layer, "
     "so any one can be replaced without disturbing the others. They are not, "
     "however, three independent views of the scene, and an earlier version "
     "of this work said they were. The text-anchored prior is a three-level "
     "quantisation of the same grayscale frame-difference statistic the "
     "motion engine uses, as the subsection on it below sets out, so two of "
     "the three curves are functions of one signal. We state it here because "
     "it is the correction that explains the ablation of Section 5.6.1."),

    ("CLIP does this at zero shot because it was pretrained",
     "CLIP does this zero-shot because it was pretrained"),

    # the third engine, described honestly
    ("Vision and motion between them still do not answer what is happening. A "
     "text prior adds that: captions translate the scene into language, which "
     "can be compared against high-risk action patterns, and sentence "
     "encoders trained on large image--text corpora carry enough of that "
     "structure to reason about danger without task-specific training. Table "
     "1 records how the encoder was chosen.",
     "The third engine was intended to answer what is happening rather than "
     "what is present or what is moving, by translating the scene into "
     "language and comparing it against high-risk action patterns. It does "
     "not do that, and the description is corrected here rather than the "
     "behaviour, so that the reported numbers remain reproducible. As "
     "implemented, it computes the mean grayscale inter-frame difference over "
     "the last third of the clip --- the same statistic the motion engine "
     "uses --- thresholds that scalar at 20 and at 10, and selects one of "
     "three fixed strings written into the source. The selected string is "
     "embedded with a sentence encoder and compared against two fixed text "
     "anchors, and a sigmoid over the frame index is parameterised by the two "
     "similarities. No frame is captioned, no image--text model is involved, "
     "and the encoder never sees anything but one of those three strings. The "
     "modality therefore has three reachable states across the whole corpus. "
     "Table 1 records how the encoder was chosen; Section 5.6.1 records what "
     "that choice was worth."),

    ("The stated explanation for the ordering --- that smaller sentence "
     "encoders suit these short, domain-specific captions better than large "
     "general-purpose ones --- remains plausible, but this experiment does "
     "not establish it.",
     "The stated explanation for the ordering --- that smaller sentence "
     "encoders suit these short, domain-specific captions better than large "
     "general-purpose ones --- does not survive the correction above either: "
     "the four encoders were not compared on captions, but on which of three "
     "fixed strings the clip had been assigned, so what the column separates "
     "is four embeddings of the same three sentences."),

    # ---- Section 4.2: equations and their glosses ---------------------
    ("The CLIP ViT-L/14 encoder of \\citet{radford2021clip} takes the frame "
     "as input and encode that frame so as to identify the dangerous and safe "
     "scene. This is carried out with the computation of the cosine "
     "similarity between the frame embedding and hand-crafted text anchors. "
     "This similarity value thus aids in the identification of the danger "
     "affinity score for each frame as mentioned in equation (3)",
     "The CLIP ViT-L/14 encoder of \\citet{radford2021clip} embeds each "
     "frame, and the frame's danger affinity is the difference between its "
     "cosine similarity to a hand-written danger prompt and to a safe prompt, "
     "passed through a logistic, as in equation (3)."),

    ("\\[{p\\_}_{clip(t)} = \\sigma\\left( \\cos\\left( \\varphi\\left( e_{t} "
     "\\right),f_{\\text{danger}} \\right) - \\cos\\left( \\varphi\\left( "
     "e_{t} \\right),f_{\\text{safe}} \\right) \\right)"
     "\\ldots\\ldots\\ldots\\ldots\\ldots\\ldots.(3)\\]",
     "\\[p_{\\text{clip}}(t) = \\sigma\\left( \\cos\\left( \\varphi(f_{t}),"
     "\\tau_{\\text{danger}} \\right) - \\cos\\left( \\varphi(f_{t}),"
     "\\tau_{\\text{safe}} \\right) \\right)"
     "\\ldots\\ldots\\ldots\\ldots\\ldots\\ldots.(3)\\]"),

    ("Where \\(\\varphi\\left( e_{t} \\right)\\) denotes the visual encoder "
     "and \\(\\sigma\\) is the logistic function.",
     "where \\(\\varphi\\) is the CLIP image encoder and \\(\\varphi(f_{t})\\) "
     "the embedding of frame \\(t\\); \\(\\tau_{\\text{danger}}\\) and "
     "\\(\\tau_{\\text{safe}}\\) are the embeddings of the two text prompts; "
     "and \\(\\sigma\\) is the logistic function."),

    ("The sudden changes in the scene and appearance of any form of "
     "kinematics is extracted from the frame using the optical flow "
     "estimation that basically finds the inter-frame pixel difference "
     "between the consecutive frames as shown in equation (4)",
     "Kinematic change is extracted as the mean absolute pixel difference "
     "between consecutive frames --- not by optical flow, which was rejected "
     "for the compute budget --- min--max normalised across the clip, as in "
     "equation (4)."),

    ("\\[{p\\_}_{flow(t)} = \\frac{{|\\left| e_{t} - e_{t - 1} "
     "\\right||}_{1} - \\min_{\\alpha}{|\\left| e_{\\alpha} - e_{\\alpha - 1} "
     "\\right||}_{1}}{\\max_{\\alpha}{||e_{\\alpha} - e_{\\alpha - 1}}|| - "
     "\\min_{\\alpha}{|\\left| e_{\\alpha} - e_{\\alpha - 1} \\right||}_{1}}"
     "\\ldots\\ldots\\ldots\\ldots..(4)\\]",
     "\\[p_{\\text{flow}}(t) = \\frac{\\left\\| f_{t} - f_{t-1} "
     "\\right\\|_{1} - \\min_{u}\\left\\| f_{u} - f_{u-1} \\right\\|_{1}}"
     "{\\max_{u}\\left\\| f_{u} - f_{u-1} \\right\\|_{1} - "
     "\\min_{u}\\left\\| f_{u} - f_{u-1} \\right\\|_{1}}"
     "\\ldots\\ldots\\ldots\\ldots..(4)\\]"),

    ("NLP captioning model generates the zero-shot text caption for each "
     "frame. This model performs the scoring against sudden onset of danger "
     "and gradual onset of danger. Thus, the final score is estimated using "
     "the temporal offset and relative score trajectory as mentioned in "
     "equation (5)",
     "where \\(u\\) ranges over the frames of the clip, so the minimum and "
     "maximum are taken over that clip alone. The temporal prior scores its "
     "selected string against a sudden-onset anchor and a gradual-onset "
     "anchor, and the two similarities parameterise a sigmoid over the frame "
     "index, as in equation (5)."),

    ("\\[{p\\_}_{caption(t)} = \\sigma\\left( \\rho_{1}D_{\\text{sudden}}(t) + "
     "\\rho_{2}D_{\\text{gradual}}(t) - t_{0} "
     "\\right)\\ldots\\ldots\\ldots\\ldots..(5)\\]",
     "\\[p_{\\text{prior}}(t) = \\sigma\\left( c_{1}D_{\\text{sudden}} + "
     "c_{2}D_{\\text{gradual}} + t - t_{0} "
     "\\right)\\ldots\\ldots\\ldots\\ldots..(5)\\]"),

    ("Where \\(\\rho_{1}\\) and \\(\\rho_{2}\\) are scaling constants and "
     "\\(t_{0}\\) denotes the temporal offset.",
     "where \\(D_{\\text{sudden}}\\) and \\(D_{\\text{gradual}}\\) are the "
     "cosine similarities of the selected string to the two fixed anchors --- "
     "constants for the clip, since the string is --- \\(c_{1}\\) and "
     "\\(c_{2}\\) are scaling constants, and \\(t_{0}\\) is a temporal offset "
     "in frames."),

    ("Then all the feature extracted from three models are combined to a "
     "unified trajectory p\\_final(t) using the 5 stages as mentioned below:",
     "The three curves are then combined into a single trajectory "
     "\\(p_{\\text{final}}(t)\\) in the five stages below."),

    ("\\[{p\\_}_{raw(t)} = 0.55\\text{\\,}p_{\\text{\\_}\\text{clip}}(t) + "
     "0.25\\text{\\,}p_{\\text{\\_}\\text{flow}}(t) + "
     "0.20\\text{\\,}p_{\\text{\\_}\\text{caption}}(t)"
     "\\ldots\\ldots\\ldots(6)\\]",
     "\\[p_{\\text{raw}}(t) = w_{\\text{clip}}\\,p_{\\text{clip}}(t) + "
     "w_{\\text{flow}}\\,p_{\\text{flow}}(t) + "
     "w_{\\text{prior}}\\,p_{\\text{prior}}(t)"
     "\\ldots\\ldots\\ldots(6)\\]"),

    ("The varied weights are exhibited due to the variation with the power of "
     "the models.",
     "The entered configuration used \\(w_{\\text{clip}} = 0.55\\), "
     "\\(w_{\\text{flow}} = 0.25\\) and \\(w_{\\text{prior}} = 0.20\\). Those "
     "values were not chosen a priori; they were searched over the evaluation "
     "corpus, which Section 4.3 records as a limitation. The released "
     "configuration weights the three equally."),

    ("The raw frame is then convolved with a 1D Gaussian kernel for reducing "
     "the frame-level noise that are caused due to the inter-frame appearance "
     "variation. Thus, the smoothened frames are outputted after the equation "
     "(7)",
     "The combined curve is convolved with a one-dimensional Gaussian kernel "
     "to suppress frame-level noise caused by inter-frame appearance "
     "variation, as in equation (7)."),

    ("\\[p_{\\__{smooth}}(t) = \\left( G_{\\sigma}*p_{raw} \\right)(t)"
     "\\ldots\\ldots\\ldots\\ldots\\ldots\\ldots\\ldots(7)\\]",
     "\\[p_{\\text{smooth}}(t) = \\left( G_{h} * p_{\\text{raw}} "
     "\\right)(t)\\ldots\\ldots\\ldots\\ldots\\ldots\\ldots\\ldots(7)\\]"),

    ("The Gaussian kernel suppresses frame-to-frame noise arising from "
     "inter-frame appearance variation, so that a single bright,",
     "Here \\(G_{h}\\) is a Gaussian kernel of bandwidth \\(h\\); the symbol "
     "\\(\\sigma\\) is reserved throughout for the logistic function. The "
     "kernel suppresses frame-to-frame noise, so that a single bright,"),

    ("\\[{p\\_}_{norm(t)} = {p\\_ smooth(t)}^{\\gamma}"
     "\\ldots\\ldots\\ldots\\ldots(8)\\]",
     "\\[p_{\\text{norm}}(t) = p_{\\text{smooth}}(t)^{\\,\\gamma}"
     "\\ldots\\ldots\\ldots\\ldots(8)\\]"),

    ("Let \\(t^{*} = \\min\\{ t:p_{\\text{norm}}(t) \\geq \\theta\\}\\)denote "
     "the first frame",
     "Let \\(t^{*} = \\min\\{ t:p_{\\text{norm}}(t) \\geq \\theta\\}\\) "
     "denote the first frame"),

    # ---- Section 4.2: the six-frame gain is not in any table ----------
    ("This is the origin of the reported mean gain of six frames in "
     "time-to-accident: the six frames are read from the future.",
     "An earlier version of this work attributed a mean gain of six frames in "
     "time-to-accident to this stage. That figure is withdrawn: no table in "
     "this paper measures the stage on its own. Table 12 separates C6 from "
     "C7 by 1.2 frames and bundles the compression with the clamp, so what "
     "the resampling bought is not identified anywhere in our records. The "
     "objection does not depend on the magnitude --- whatever it bought was "
     "read from a frame that had not arrived."),

    # ---- Section 4.2: a promise Section 5 does not keep ---------------
    ("We retain the stage, and Section 5 reports every metric with it "
     "disabled as well as enabled.",
     "We retain the stage. Section 5 does not report the leaderboard score of "
     "a run with it disabled --- we did not make that submission, and say so "
     "rather than imply otherwise --- which is exactly the omission item 5 of "
     "the protocol in Section 7 asks others not to repeat."),

    # ---- Section 4.4: the family, stated in full ----------------------
    ("The family comprises a constant 0.51 and a constant 0.99; a linear ramp "
     "from 0.000 to 1.000; logistic curves centred at frames 60, 75, 90 and "
     "105; and step functions rising from 0.49 to 0.51 at frames 0, 10, 25, "
     "50, 75, 100, 125 and 140. Each is a single list of 150 numbers, "
     "repeated across all 1,417 clips.",
     "The family comprises a constant 0.51 and a constant 0.99; a curve held "
     "at 0.49 throughout, which never crosses; a linear ramp from 0.000 to "
     "1.000; logistic curves centred at frames 60, 75, 90 and 105; step "
     "functions rising from 0.49 to 0.51 at frames 0, 10, 25, 50, 75, 100, "
     "125 and 140, and a calibration step at frame 76; a curve that crosses "
     "at frame 0 and drops back, and one that crosses and dips across frames "
     "50 to 59; and the two graded curves of Section 5.4. Twenty-five in "
     "all. Each is a single list of 150 numbers, repeated across all 1,417 "
     "clips."),

    # ---- Section 5.4: Table 8 says the ramp is above one of them ------
    ("A linear ramp crosses the threshold at frame 75, exactly as three other "
     "curves do, and scores 0.037 lower than they do.",
     "A linear ramp crosses the threshold at frame 75, exactly as a step "
     "function does, and scores 0.037 lower than the step."),

    ("\\floatcap{Table 8}{Four curves that all first exceed 0.5 at frame 75, "
     "with identical values at frames 74 and 75 and no subsequent dip, plus a "
     "one-frame calibration step.}",
     "\\floatcap{Table 8}{Four curves that all first exceed 0.5 at frame 75 "
     "and never fall below it afterwards, plus a one-frame calibration step. "
     "The three step-like curves hold identical values at frames 74 and 75.}"),

    ("The four curves at the head of Table 8 have identical values at frames "
     "74 and 75, cross at the same frame, and never fall back afterwards.",
     "The four curves at the head of Table 8 cross the threshold at the same "
     "frame and never fall back afterwards, and the three step-like ones hold "
     "identical values at frames 74 and 75."),

    # ---- Section 5.6.3: the ordering is not monotone ------------------
    ("Table 12 compares seven pipeline configurations. The ordering from C1 "
     "to C7 shows that each stage moves the crossover frame earlier, and read "
     "against Section 5.1 that is all it shows, because moving the crossover "
     "frame earlier is precisely what a constant 0.51 does perfectly. Two "
     "stages deserve particular note in that light. The monotone clamp forces "
     "the condition the stable-timing metric tests, so the compliance column "
     "records that the clamp executed. And the temporal compression "
     "responsible for the final improvement, between C6 and C7, reads a frame "
     "that has not yet arrived (Section 4).",
     "Table 12 compares seven pipeline configurations. Each post-processing "
     "stage added to the raw ensemble moves the crossover frame earlier, from "
     "C4 at 25.4 to C7 at 22.7; the single-modality rows C1 to C3 are not "
     "steps in that sequence and do not order monotonically with it. Read "
     "against Section 5.1 that is all the table shows, because moving the "
     "crossover frame earlier is precisely what a constant 0.51 does "
     "perfectly --- and C3, the three-state temporal prior alone, already "
     "beats the full pipeline on it at 21.7, for the reason Section 5.6.1 "
     "gives. Two stages deserve particular note. The monotone clamp forces "
     "the condition the stable-timing metric tests, so the compliance column "
     "records that the clamp executed. And the C6-to-C7 step bundles the "
     "clamp with the temporal compression, which reads a frame that has not "
     "yet arrived (Section 4.2), so neither the 1.2 frames between them nor "
     "the compliance flag beside them is attributable to a single stage."),

    ("C7 (Proposed) & Full Pipeline: Ensemble + Smoothing + Power Curve + "
     "Temporal Compression + Clamping & 22.7 & Yes & Best performance; full "
     "STTA compliance \\\\",
     "C7 (Proposed) & Full Pipeline: Ensemble + Smoothing + Power Curve + "
     "Temporal Compression + Clamping & 22.7 & Yes & Earliest crossover of "
     "the ensembled configurations; STTA compliance by construction \\\\"),

    ("\\textbf{Full Ensemble (Proposed)} & \\textbf{22.7} & \\textbf{---} & "
     "\\textbf{Best performance; all modality blind spots mutually "
     "compensated} \\\\",
     "\\textbf{Full Ensemble (Proposed)} & \\textbf{22.7} & \\textbf{---} & "
     "\\textbf{The three blind spots compensate one another; the crossover "
     "frame is nonetheless later than the prior alone} \\\\"),

    ("NLP Prior Only & $\\approx$21.7 & −1.0 & Temporal ceiling; identical "
     "t\\textsubscript{0} values for clips with similar captions --- no "
     "per-clip visual personalisation \\\\",
     "NLP Prior Only & $\\approx$21.7 & −1.0 & Three reachable states for the "
     "whole corpus, so clips in the same motion bucket receive identical "
     "curves; no per-clip visual detail at all \\\\"),

    # ---- Section 5.6.1: what the ablation ordering actually means -----
    ("The ensemble does resolve those three blind spots against one another. "
     "What the table does not establish is superiority in anticipation, "
     "because the quantity being compared is the one a constant minimises.",
     "The ensemble does resolve those three blind spots against one another. "
     "What the table does not establish is superiority in anticipation, "
     "because the quantity being compared is the one a constant minimises --- "
     "and the table makes that concrete, since the temporal prior alone posts "
     "an earlier mean crossover, 21.7 against 22.7, than the full ensemble "
     "does. That is not evidence that the prior anticipates better. It has "
     "three reachable states across the corpus, so it emits nearly the same "
     "curve everywhere, and a nearly constant curve is well matched to a "
     "benchmark whose events sit at a nearly constant position."),

    # ---- Section 5.7 ---------------------------------------------------
    ("The 4.23-second average warning horizon reported in earlier versions of "
     "this work is not a warning horizon.",
     "The 4.24-second average warning horizon reported in earlier versions of "
     "this work is not a warning horizon."),

    # ---- Section 7 -----------------------------------------------------
    ("A benchmark's organisers can do in an afternoon what took us "
     "twenty-four submissions:",
     "A benchmark's organisers can do in an afternoon what took us "
     "twenty-five submissions:"),

    ("Our Stage 5 did this at α = 1.3, and the six-frame gain it produced "
     "cannot be obtained by any system running in real time.",
     "Our Stage 5 did this at a compression factor of 1.3, and whatever "
     "earliness it bought cannot be obtained by any system running in real "
     "time."),

    # ---- Section 8 -----------------------------------------------------
    ("One discrepancy in the crossing-frame model is open. A linear ramp "
     "scores 0.037 below three curves that cross at the same frame (Section "
     "5.4),",
     "One discrepancy in the crossover-frame model is open. A linear ramp "
     "scores 0.037 below a step function that crosses at the same frame, and "
     "0.029 above another curve that does (Section 5.4),"),

    ("Items 2 to 5 of Section 7 are derived from the mechanisms rather than "
     "shown end to end,",
     "The protocol's recommendations on bounding and conditioning the "
     "earliness term, on stating a timing metric's reference point, on "
     "reporting discrimination beside it and on reporting metrics with "
     "post-processing disabled are derived from the mechanisms rather than "
     "shown end to end,"),

    # ---- Section 9 -----------------------------------------------------
    ("Twenty-four submissions, not one of which opened a video frame, were "
     "enough",
     "Twenty-five submissions, not one of which opened a video frame, were "
     "enough"),

    # ---- Compliance ----------------------------------------------------
    ("Twenty-four submissions were made in total, across two days, against a "
     "stated limit of one hundred per day.",
     "Twenty-six submissions were made in total, all on 31 August 2026, "
     "against a stated limit of one hundred per day: twenty-five video-blind "
     "curves we generated, and the organisers' own sample submission "
     "resubmitted verbatim as a control on the submission path."),

    ("The submission files, the code that generated them, the complete score "
     "log and the scripts that produce both figures are released. Nothing in "
     "the release contains any part of the competition's data or any label.",
     "The submission files, the code that generated them, the complete score "
     "log and the scripts that produce both figures accompany this submission "
     "as anonymised supplementary material, and will be released publicly on "
     "acceptance. Nothing in them contains any part of the competition's data "
     "or any label."),

    # ---- table captions that still carried the withdrawn framing ------
    ("\\floatcap{Table 9}{Overall Zero-Shot Anticipation Performance on MM-AU "
     "(N\\,=\\,1,417 clips)}",
     "\\floatcap{Table 9}{Curve-shape statistics of the ensemble over the "
     "1,417-clip corpus. Not anticipation results: read with the framing "
     "above.}"),

    ("\\floatcap{Table 12}{Seven-Configuration Performance Comparison}",
     "\\floatcap{Table 12}{Seven pipeline configurations, compared on mean "
     "crossover frame --- a statistic of curve shape, not performance.}"),
]

REPLACE_ALL = [
    ("\\textasciitilde", "$\\approx$"),
]


def apply(path: str, pairs) -> int:
    p = pathlib.Path(path)
    s = p.read_text()
    bad = 0
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            print(f"  [!! {n}] {path}: {old[:70]!r}")
            bad += 1
            continue
        s = s.replace(old, new)
    p.write_text(s)
    return bad


def main() -> None:
    # the tildes go first: two later replacements match text containing them
    p = pathlib.Path("body_clean.tex")
    s = p.read_text()
    for old, new in REPLACE_ALL:
        n = s.count(old)
        s = s.replace(old, new)
        print(f"  [OK] {n} × {old} -> {new}")
    p.write_text(s)

    bad = apply("body_clean.tex", BODY) + apply("abstract.tex", ABSTRACT)
    if bad:
        print(f"\n{bad} replacement(s) did not apply cleanly.")
        sys.exit(1)
    print(f"\n{len(BODY) + len(ABSTRACT)} corrections applied")


if __name__ == "__main__":
    main()
