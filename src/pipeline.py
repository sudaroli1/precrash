"""
PreCrash: Zero-Shot Multimodal Ensemble Accident Anticipation
Six-stage pipeline as described in the paper (Section 3).
"""

from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path
from typing import Union

from .engines.clip_scorer import CLIPScorer
from .engines.flow_scorer import FlowScorer
from .engines.nlp_scorer import NLPScorer
from .ensemble import ensemble_scores
from .postprocess import PostProcessor
from .utils.video_io import load_frames


class PreCrashPipeline:
    """
    Six-stage zero-shot accident anticipation pipeline.

    Stages
    ------
    1. Frame loading & preprocessing
    2. CLIP ViT-L/14 semantic scoring
    3. Optical flow (frame-difference) kinematic scoring
    4. NLP caption temporal prior scoring
    5. Weighted ensemble fusion
    6. Post-processing (Gaussian smooth → power curve →
                        temporal compression → monotone clamp → stability clip)

    Parameters
    ----------
    config : dict
        Parsed YAML configuration (see configs/default.yaml).
    device : str
        PyTorch device string, e.g. "cuda" or "cpu".
    """

    def __init__(self, config: dict, device: str = "cuda"):
        self.cfg = config
        self.device = device

        # Instantiate the three cognitive engines
        self.clip_scorer = CLIPScorer(
            model_name=config["clip"]["model"],
            danger_prompt=config["clip"]["danger_prompt"],
            safe_prompt=config["clip"]["safe_prompt"],
            device=device,
        )
        self.flow_scorer = FlowScorer(
            use_raft=config["flow"].get("use_raft", False),
            device=device,
        )
        self.nlp_scorer = NLPScorer(
            model_name=config["nlp"]["model"],
            sudden_anchor=config["nlp"]["sudden_anchor"],
            gradual_anchor=config["nlp"]["gradual_anchor"],
            rho1=config["nlp"].get("rho1", 1.0),
            rho2=config["nlp"].get("rho2", 0.5),
        )
        self.postprocessor = PostProcessor(config["postprocess"])

        self.w_clip = config["ensemble"]["clip_weight"]
        self.w_flow = config["ensemble"]["flow_weight"]
        self.w_nlp = config["ensemble"]["nlp_weight"]
        assert abs(self.w_clip + self.w_flow + self.w_nlp - 1.0) < 1e-6, \
            "Ensemble weights must sum to 1.0"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: Union[str, Path], device: str = "cuda") -> "PreCrashPipeline":
        """Load pipeline from a YAML config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config, device=device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, video_path: Union[str, Path]) -> np.ndarray:
        """
        Run the full six-stage pipeline on a dashcam video clip.

        Parameters
        ----------
        video_path : str or Path
            Path to the video file (MP4, AVI, etc.).

        Returns
        -------
        risk_scores : np.ndarray, shape (T,)
            Per-frame risk probability in (0.001, 0.999), monotonically
            non-decreasing after threshold crossing (STTA-compliant).
        """
        vcfg = self.cfg["video"]

        # ── Stage 1: Frame loading & preprocessing ────────────────────
        frames = load_frames(
            video_path,
            num_frames=vcfg["num_frames"],
            resize=tuple(vcfg["resize"]),
            interpolation=vcfg.get("interpolation", "bicubic"),
            crop_top_frac=vcfg.get("crop_top_frac", 0.20),
            crop_bottom_frac=vcfg.get("crop_bottom_frac", 0.08),
            normalise=vcfg.get("normalise", True),
        )  # list of np.ndarray, each (H, W, 3) uint8

        # ── Stage 2: CLIP semantic scoring ───────────────────────────
        p_clip = self.clip_scorer.score(frames)        # shape (T,)

        # ── Stage 3: Optical flow kinematic scoring ──────────────────
        p_flow = self.flow_scorer.score(frames)        # shape (T,)

        # ── Stage 4: NLP caption temporal prior ──────────────────────
        p_nlp = self.nlp_scorer.score(frames)          # shape (T,)

        # ── Stage 5: Weighted ensemble fusion (Equation 4) ───────────
        p_raw = ensemble_scores(
            p_clip, p_flow, p_nlp,
            w_clip=self.w_clip,
            w_flow=self.w_flow,
            w_nlp=self.w_nlp,
        )                                              # shape (T,)

        # ── Stage 6: Post-processing ──────────────────────────────────
        p_final = self.postprocessor.process(p_raw)    # shape (T,)

        return p_final

    def predict_batch(self, video_paths: list, show_progress: bool = True) -> list[np.ndarray]:
        """Run predict() over a list of video paths with optional tqdm progress bar."""
        try:
            from tqdm import tqdm
            iterator = tqdm(video_paths, desc="PreCrash inference") if show_progress else video_paths
        except ImportError:
            iterator = video_paths

        return [self.predict(p) for p in iterator]
