"""
Stage 4 — Temporal prior from a motion threshold.

WHAT THIS ENGINE ACTUALLY IS
----------------------------
Earlier versions of this file, and the corresponding sections of the
manuscript, described this component as an NLP captioning engine that reasons
over visual observations. It is not one, and the description has been corrected
rather than the behaviour, so that previously reported numbers remain
reproducible.

What the engine does:

  1. Computes the mean grayscale inter-frame difference over the last third of
     the clip -- the same statistic the optical-flow engine uses.
  2. Thresholds that scalar at 20 and 10 and selects one of THREE fixed
     strings.
  3. Embeds the selected string with a SentenceTransformer and takes cosine
     similarity against two fixed anchors.
  4. Emits a sigmoid over frame index, parameterised by those similarities.

No frame is captioned. No image-text model is involved. No language is read
from the video. The sentence encoder only ever sees one of three strings this
module wrote itself, so the modality has **three reachable states** for the
whole corpus.

TWO CONSEQUENCES WORTH STATING PLAINLY
--------------------------------------
**It is not independent of the flow engine.** Both are functions of the same
grayscale frame-difference statistic. This engine is a three-level quantisation
of it, mapped through a sentence encoder to a sigmoid. Any claim that the
ensemble fuses three independent views of a scene does not hold for this
implementation.

**It explains the ablation.** With three reachable states, clips that fall in
the same motion bucket receive identical curves. That is the "all captions with
similar wording get identical t0 values" behaviour reported in the ablation
study, and it is why this modality alone can post an earlier mean crossover
than the full ensemble on a crossing-frame metric: a curve that is nearly the
same everywhere is well matched to a benchmark whose events sit at a nearly
constant position.

The class is named for what it does. `NLPScorer` remains as an alias so that
existing scripts and cached configurations continue to work.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit  # numerically stable sigmoid


class MotionThresholdPrior:
    """
    Temporal risk prior derived from a three-level motion threshold.

    Parameters
    ----------
    model_name : str
        SentenceTransformer identifier used to embed the selected string.
        Note that it never sees video, and only ever sees one of the three
        strings in `MOTION_STRINGS`.
    sudden_anchor, gradual_anchor : str
        Fixed text anchors the selected string is compared against.
    rho1, rho2 : float
        Scaling constants on the two anchor similarities.
    """

    # The complete output vocabulary of the "captioner".
    MOTION_STRINGS = (
        "sudden vehicle collision impact crash danger",   # mean motion > 20
        "vehicle losing control near miss risk",          # mean motion > 10
        "normal traffic driving road scene",              # otherwise
    )
    MOTION_THRESHOLDS = (20.0, 10.0)

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        sudden_anchor: str = "sudden collision impact crash",
        gradual_anchor: str = "gradual risk build-up near miss",
        rho1: float = 1.0,
        rho2: float = 0.5,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not found. Install with:\n"
                "  pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self.rho1 = rho1
        self.rho2 = rho2

        self.sudden_vec = self.model.encode(sudden_anchor, normalize_embeddings=True)
        self.gradual_vec = self.model.encode(gradual_anchor, normalize_embeddings=True)

        # The output vocabulary is three fixed strings, so their embeddings are
        # three fixed vectors. Encoding the selected one per clip recomputed the
        # same three results 1,500 times per pass; they are computed once here
        # and looked up. Identical values, and one fewer model call per clip.
        self._vec_of = {
            s: self.model.encode(s, normalize_embeddings=True)
            for s in self.MOTION_STRINGS
        }

    def score(self, frames: list) -> np.ndarray:
        """
        Emit the per-frame temporal prior.

        Behaviour is unchanged from the original implementation; only the
        naming and documentation have been corrected.

        Returns
        -------
        scores : np.ndarray, shape (T,), values in [0, 1]
        """
        T = len(frames)
        selected = self._select_string(frames)
        vec = self._vec_of[selected]

        d_sudden = float(vec @ self.sudden_vec)
        d_gradual = float(vec @ self.gradual_vec)

        # Curve position is a function of d_sudden alone, which takes one of
        # three values across the entire corpus.
        t0_frac = 0.5 - 0.3 * d_sudden

        t_arr = np.arange(T, dtype=np.float32)
        logit = self.rho1 * d_sudden + self.rho2 * d_gradual - (t0_frac - t_arr / T)
        return expit(logit * 6.0).astype(np.float32)

    # ------------------------------------------------------------------

    def _select_string(self, frames: list) -> str:
        """
        Pick one of three fixed strings by thresholding mean grayscale motion
        over the last third of the clip.

        This is the whole of the "captioning" step. It shares its input
        statistic with FlowScorer._score_frame_diff, so the two modalities are
        not independent.
        """
        import cv2

        n = len(frames)
        sample = frames[2 * n // 3:]
        if len(sample) < 2:
            return self.MOTION_STRINGS[2]

        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in sample]
        diffs = [np.mean(np.abs(grays[i] - grays[i - 1])) for i in range(1, len(grays))]
        mean_motion = float(np.mean(diffs))

        hi, lo = self.MOTION_THRESHOLDS
        if mean_motion > hi:
            return self.MOTION_STRINGS[0]
        if mean_motion > lo:
            return self.MOTION_STRINGS[1]
        return self.MOTION_STRINGS[2]

    def state_of(self, frames: list) -> int:
        """Which of the three states a clip falls into (0, 1 or 2).

        Provided so the state distribution over a corpus can be reported. If
        one state dominates, the modality is close to a constant.
        """
        return self.MOTION_STRINGS.index(self._select_string(frames))


# Backwards-compatible alias. The original name described an engine that was
# never implemented; it is retained only so existing scripts keep working.
NLPScorer = MotionThresholdPrior
