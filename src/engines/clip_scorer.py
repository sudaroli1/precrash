"""
Stage 2 — CLIP ViT-L/14 Semantic Scorer.

Computes per-frame danger affinity via cosine similarity between
the frame embedding and opposing text anchors (Equation 2 in the paper).

  p_clip(t) = σ( cos(φ_e(t), f_danger) − cos(φ_e(t), f_safe) )

where φ_e is the ViT-L/14 visual encoder and σ is the logistic function.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import clip
except ImportError:
    raise ImportError(
        "OpenAI CLIP not found. Install with:\n"
        "  pip install git+https://github.com/openai/CLIP.git"
    )


class CLIPScorer:
    """
    Zero-shot frame danger scorer using CLIP ViT-L/14.

    Parameters
    ----------
    model_name : str
        CLIP model variant. Paper uses "ViT-L/14".
    danger_prompt : str
        Text anchor describing a dangerous accident scene.
    safe_prompt : str
        Text anchor describing a normal driving scene.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        model_name: str = "ViT-L/14",
        danger_prompt: str = "dangerous accident collision crash",
        safe_prompt: str = "normal safe driving no hazard",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.device = device
        self.batch_size = int(batch_size)
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        # Encode text anchors once
        with torch.no_grad():
            texts = clip.tokenize([danger_prompt, safe_prompt]).to(device)
            text_features = self.model.encode_text(texts)
            self.text_features = F.normalize(text_features, dim=-1)  # (2, D)

    @torch.no_grad()
    def score(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Compute per-frame danger affinity scores.

        Parameters
        ----------
        frames : list of np.ndarray
            RGB frames, each shape (H, W, 3) uint8.

        Returns
        -------
        scores : np.ndarray, shape (T,)
            Logistic-transformed danger affinity in [0, 1].

        Notes
        -----
        Frames are encoded in batches. The encoder treats each image
        independently, so batching changes throughput and not the quantity
        computed. The previous version ran one frame at a time and called
        `.item()` after each, forcing a device synchronisation per frame and
        leaving the GPU idle between them.
        """
        if not frames:
            return np.zeros(0, dtype=np.float32)

        batch = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames])
        batch = batch.to(self.device)

        logits = []
        for i in range(0, len(batch), self.batch_size):
            feats = self.model.encode_image(batch[i:i + self.batch_size])
            feats = F.normalize(feats, dim=-1)                  # (B, D)
            cos = feats @ self.text_features.T                  # (B, 2)
            logits.append(cos[:, 0] - cos[:, 1])

        # one host transfer for the clip, rather than one per frame
        return torch.sigmoid(torch.cat(logits)).float().cpu().numpy().astype(np.float32)
