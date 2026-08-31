"""
Tests for the two extraction speed-ups, both of which must be exactly
value-preserving.

An optimisation that changes results is not an optimisation, it is a silent
revision of every number in the paper. Both changes here were made to bring a
measured 81 s/clip extraction rate — 20 hours for the dev split alone — down to
something a Colab session can finish, and both are pinned against a reference
implementation of what they replaced.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

cv2 = pytest.importorskip("cv2")

from utils.video_io import load_frames, _normalise_frame  # noqa: E402


# ----------------------------------------------------------------------
# Frame loading: seek once and read forward, rather than seek per frame
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def clip_path(tmp_path_factory):
    """A short synthetic clip with per-frame structure, so a wrong frame index
    shows up as a pixel difference rather than passing unnoticed."""
    path = tmp_path_factory.mktemp("video") / "synth.mp4"
    h, w, n = 120, 160, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    if not writer.isOpened():
        pytest.skip("no mp4 writer available in this OpenCV build")
    rng = np.random.default_rng(0)
    for i in range(n):
        frame = rng.integers(0, 60, size=(h, w, 3), dtype=np.uint8)
        # a bright block whose position encodes the frame index
        x = (i * 3) % (w - 20)
        frame[10:40, x:x + 20] = 240
        writer.write(frame)
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("mp4 writer produced no file")
    return path


def _seek_per_frame(video_path, num_frames, start_frame, end_frame,
                    resize=(224, 224), crop_top_frac=0.20,
                    crop_bottom_frac=0.08, normalise=True):
    """The implementation that was replaced: one POS_FRAMES seek per frame."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo = 0 if start_frame is None else max(0, int(start_frame))
    hi = total if end_frame is None else min(total, int(end_frame))
    indices = np.linspace(lo, hi - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            if frames:
                frames.append(frames[-1].copy())
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h = frame.shape[0]
        frame = frame[int(h * crop_top_frac):h - int(h * crop_bottom_frac), :]
        frame = cv2.resize(frame, (resize[1], resize[0]), interpolation=cv2.INTER_CUBIC)
        if normalise:
            frame = _normalise_frame(frame)
        frames.append(frame)
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else np.zeros((*resize, 3), np.uint8))
    return frames[:num_frames]


@pytest.mark.parametrize("num_frames,start,end", [
    (60, 100, 160),     # contiguous 1:1 window, the case the manifests use
    (30, 100, 160),     # subsampled window
    (60, None, None),   # whole clip
    (40, 0, 40),        # window at the very start
])
def test_sequential_read_matches_seek_per_frame(clip_path, num_frames, start, end):
    fast = load_frames(clip_path, num_frames=num_frames,
                       start_frame=start, end_frame=end)
    ref = _seek_per_frame(clip_path, num_frames, start, end)

    assert len(fast) == len(ref) == num_frames
    for i, (a, b) in enumerate(zip(fast, ref)):
        assert np.array_equal(a, b), f"frame {i} differs"


def test_sparse_request_still_returns_the_right_count(clip_path):
    """Beyond 20x the requested count the loader falls back to seeking. It must
    still return exactly num_frames."""
    frames = load_frames(clip_path, num_frames=5, start_frame=0, end_frame=240)
    assert len(frames) == 5


# ----------------------------------------------------------------------
# CLIP scoring: batched encoding must equal per-frame encoding
# ----------------------------------------------------------------------

def test_batched_clip_scoring_matches_per_frame():
    """Batching is only legitimate because the encoder is per-sample
    independent. This pins that with a stand-in encoder, so the property is
    tested without downloading 900 MB of weights."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from PIL import Image

    from engines.clip_scorer import CLIPScorer

    rng = np.random.default_rng(0)
    D = 8
    proj = torch.tensor(rng.standard_normal((3 * 4 * 4, D)), dtype=torch.float32)

    class StubModel:
        def encode_image(self, batch):                    # (B, 3, 4, 4)
            return batch.reshape(batch.shape[0], -1) @ proj

    scorer = CLIPScorer.__new__(CLIPScorer)               # bypass clip.load
    scorer.device = "cpu"
    scorer.batch_size = 5
    scorer.model = StubModel()
    scorer.preprocess = lambda img: torch.tensor(
        np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    text = torch.tensor(rng.standard_normal((2, D)), dtype=torch.float32)
    scorer.text_features = F.normalize(text, dim=-1)

    frames = [rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8) for _ in range(13)]

    with torch.no_grad():
        got = scorer.score(frames)

        expected = []
        for f in frames:
            t = scorer.preprocess(Image.fromarray(f)).unsqueeze(0)
            v = F.normalize(scorer.model.encode_image(t), dim=-1)
            cos = (v @ scorer.text_features.T).squeeze(0)
            expected.append(torch.sigmoid(cos[0] - cos[1]).item())

    assert got.shape == (13,)
    # Not bitwise equality. A matmul's reduction order depends on the batch
    # dimension, so batching moves float32 results by a few units in the last
    # place — measured at ~3e-8 here. That is far below anything a metric can
    # resolve, but it does mean the cached curves are not bit-reproducible
    # across batch sizes, and the paper should say so rather than claim they
    # are. The tolerance below is the honest bound.
    assert np.allclose(got, np.asarray(expected, dtype=np.float32), atol=1e-6)


def test_batch_size_changes_the_result_only_in_the_last_place():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from engines.clip_scorer import CLIPScorer

    rng = np.random.default_rng(1)
    D = 8
    proj = torch.tensor(rng.standard_normal((3 * 4 * 4, D)), dtype=torch.float32)

    class StubModel:
        def encode_image(self, batch):
            return batch.reshape(batch.shape[0], -1) @ proj

    text = F.normalize(torch.tensor(rng.standard_normal((2, D)),
                                    dtype=torch.float32), dim=-1)
    frames = [rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8) for _ in range(17)]

    def score_with(bs):
        s = CLIPScorer.__new__(CLIPScorer)
        s.device, s.batch_size, s.model = "cpu", bs, StubModel()
        s.preprocess = lambda img: torch.tensor(
            np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        s.text_features = text
        with torch.no_grad():
            return s.score(frames)

    # 17 frames: batch sizes that divide it, do not divide it, and exceed it
    outs = [score_with(bs) for bs in (1, 4, 17, 64)]
    for bs, o in zip((4, 17, 64), outs[1:]):
        # see the note in the test above: last-place float32 movement only
        assert np.allclose(outs[0], o, atol=1e-6), f"batch_size={bs}"
        assert np.max(np.abs(outs[0] - o)) < 1e-6


def test_empty_frame_list_returns_empty():
    pytest.importorskip("torch")
    from engines.clip_scorer import CLIPScorer

    s = CLIPScorer.__new__(CLIPScorer)
    s.device, s.batch_size = "cpu", 8
    assert CLIPScorer.score(s, []).shape == (0,)


# ----------------------------------------------------------------------
# The three-string prior must be unchanged by caching its embeddings
# ----------------------------------------------------------------------

def test_motion_prior_uses_a_fixed_three_string_vocabulary():
    """The embeddings are cached in __init__ rather than recomputed per clip.
    That is only safe because the vocabulary is closed — assert it is."""
    from engines.nlp_scorer import MotionThresholdPrior

    assert len(MotionThresholdPrior.MOTION_STRINGS) == 3
    assert len(set(MotionThresholdPrior.MOTION_STRINGS)) == 3
    assert MotionThresholdPrior.MOTION_THRESHOLDS == (20.0, 10.0)
