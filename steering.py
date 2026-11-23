"""Shared utilities for speaker embedding steering, audio loading, and device selection."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from typing import Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch


def get_best_device() -> str:
    """Return 'mps' on Apple Silicon if available, otherwise 'cpu'."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def get_preferred_dtype(device: str) -> torch.dtype:
    """Return bfloat16 on MPS (and float32 on CPU for broad compatibility)."""
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def load_audio_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> Tuple[np.ndarray, int]:
    """Decode uploaded audio bytes (wav, flac, mp3, ogg, m4a, webm) to mono float32 waveform + sr."""
    # Try soundfile directly first (fastest, handles wav/flac/ogg)
    try:
        with io.BytesIO(audio_bytes) as buf:
            wav, sr = sf.read(buf, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        return wav.astype(np.float32), int(sr)
    except Exception:
        pass

    # Fallback: write to temp file and try librosa or ffmpeg -> wav
    suffix = os.path.splitext(filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        in_path = tmp_in.name

    tmp_wav_path = None
    try:
        try:
            wav, sr = librosa.load(in_path, sr=None, mono=True)
            return wav.astype(np.float32), int(sr)
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                tmp_wav_path = tmp_out.name
            subprocess.run(
                ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-f", "wav", tmp_wav_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wav, sr = sf.read(tmp_wav_path, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = np.mean(wav, axis=-1)
            return wav.astype(np.float32), int(sr)
    finally:
        if os.path.exists(in_path):
            os.unlink(in_path)
        if tmp_wav_path and os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)


def project_embedding(
    embedding: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Project a 2048-D embedding onto K unit-norm PCA components in float32."""
    x = np.asarray(embedding, dtype=np.float32).reshape(-1)
    m = np.asarray(mean, dtype=np.float32).reshape(-1)
    comps = np.asarray(components, dtype=np.float32)
    return (x - m) @ comps.T


def steer_embedding(
    base_embedding: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    targets: Sequence[float],
    preserve_norm: bool = True,
    attribute_vectors: np.ndarray | None = None,
    attribute_deltas: Sequence[float] | None = None,
) -> np.ndarray:
    """Steer a 2048-D speaker embedding to target PCA coordinates + optional attribute deltas in float32.

    1. For each PCA axis i:
         x = x + (t[i] - (x - mean) @ components[i]) * components[i]
    2. For each supervised attribute j (e.g. Gender, Age, Indian, Scottish):
         x = x + delta[j] * attribute_vectors[j]
    3. If preserve_norm is True, rescales the steered vector back to the original L2 norm
       to reduce out-of-distribution distortion when pushing multiple sliders.
    """
    x = np.asarray(base_embedding, dtype=np.float32).reshape(-1).copy()
    m = np.asarray(mean, dtype=np.float32).reshape(-1)
    comps = np.asarray(components, dtype=np.float32)

    orig_norm = float(np.linalg.norm(x))

    for i, target_val in enumerate(targets):
        if i >= comps.shape[0]:
            break
        c = comps[i]
        current_proj = float(np.dot(x - m, c))
        x = x + (float(target_val) - current_proj) * c

    if attribute_vectors is not None and attribute_deltas is not None:
        attrs = np.asarray(attribute_vectors, dtype=np.float32)
        for j, delta_val in enumerate(attribute_deltas):
            if j >= attrs.shape[0]:
                break
            d = float(delta_val)
            if abs(d) > 1e-7:
                x = x + d * attrs[j]

    if preserve_norm and orig_norm > 1e-8:
        new_norm = float(np.linalg.norm(x))
        if new_norm > 1e-8:
            x = x * (orig_norm / new_norm)

    return x.astype(np.float32)
