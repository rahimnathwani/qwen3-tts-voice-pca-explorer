"""Wrapper for the Qwen3 1.7B speaker encoder (2048-D x-vectors).

Supports:
1. Standalone `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` (AutoProcessor + AutoModel).
2. Automatic offline fallback to extracting the identical `speaker_encoder.*` weights
   from a locally cached `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (or `Qwen/...-Base`)
   checkpoint when available, auto-transposing MLX Conv1d weights `(O, K, I) -> (O, I, K)`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModel, AutoProcessor

from steering import get_best_device

SPEAKER_ENCODER_REPO = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
FALLBACK_BASE_REPOS = (
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
)
EXPECTED_DIM = 2048


def _find_local_base_snapshot() -> Optional[Tuple[str, Path]]:
    for repo_id in FALLBACK_BASE_REPOS:
        try:
            snap = snapshot_download(repo_id=repo_id, local_files_only=True)
            snap_path = Path(snap)
            if (snap_path / "config.json").exists() and (snap_path / "model.safetensors").exists():
                return repo_id, snap_path
        except Exception:
            continue
    return None


class SpeakerEncoder:
    """Extracts 2048-D Qwen3-TTS speaker embeddings on MPS or CPU."""

    def __init__(
        self,
        model_id: str = SPEAKER_ENCODER_REPO,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.device = device or get_best_device()
        self.dtype = dtype or torch.float32
        self._mode = "hf"
        self.model_id = model_id
        self.sample_rate = 24000
        self.processor = None
        self.model = None

        # 1) Prefer standalone encoder if already cached locally
        if self._try_load_hf(model_id, local_files_only=True):
            return

        # 2) If not cached, check if a local 1.7B Base checkpoint is already cached (avoids network wait)
        local_base = _find_local_base_snapshot()
        if local_base is not None and os.environ.get("FORCE_STANDALONE_ENCODER", "0") != "1":
            repo_id, snap_path = local_base
            self._load_from_base_snapshot(repo_id, snap_path)
            return

        # 3) Otherwise download standalone encoder from Hugging Face
        self._try_load_hf(model_id, local_files_only=False, raise_on_error=True)

    def _try_load_hf(self, model_id: str, local_files_only: bool, raise_on_error: bool = False) -> bool:
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=local_files_only
            )
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=self.dtype,
                local_files_only=local_files_only,
            ).to(self.device)
            self.model.eval()
            self._mode = "hf"
            self.model_id = model_id

            sr = getattr(self.processor, "sampling_rate", None)
            if sr is None and hasattr(self.processor, "feature_extractor"):
                sr = getattr(self.processor.feature_extractor, "sampling_rate", None)
            if sr is None and hasattr(self.model, "config"):
                sr = getattr(self.model.config, "sample_rate", None) or getattr(
                    self.model.config, "sampling_rate", None
                )
            self.sample_rate = int(sr) if sr is not None else 24000
            return True
        except Exception:
            if raise_on_error:
                raise
            return False

    def _load_from_base_snapshot(self, repo_id: str, snap_path: Path) -> None:
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

        with open(snap_path / "config.json", "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)["speaker_encoder_config"]

        enc_cfg = Qwen3TTSSpeakerEncoderConfig(**cfg_dict)
        enc = Qwen3TTSSpeakerEncoder(enc_cfg)
        expected = enc.state_dict()

        sd = {}
        with safe_open(str(snap_path / "model.safetensors"), framework="pt") as sf_file:
            for k in sf_file.keys():
                if not k.startswith("speaker_encoder."):
                    continue
                sub_k = k[len("speaker_encoder.") :]
                t = sf_file.get_tensor(k)
                # Auto-transpose MLX Conv1d weights (out, k, in) -> PyTorch (out, in, k)
                if (
                    sub_k in expected
                    and t.shape != expected[sub_k].shape
                    and t.ndim == 3
                    and t.permute(0, 2, 1).shape == expected[sub_k].shape
                ):
                    t = t.permute(0, 2, 1).contiguous()
                sd[sub_k] = t

        enc.load_state_dict(sd, strict=True)
        self.model = enc.to(device=self.device, dtype=self.dtype).eval()
        self.sample_rate = int(getattr(enc_cfg, "sample_rate", 24000))
        self._mode = "qwen_base"
        self.model_id = f"{repo_id}::speaker_encoder"

    @torch.no_grad()
    def encode(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract a single (2048,) float32 speaker embedding from a mono waveform."""
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1).astype(np.float32)

        if int(sr) != self.sample_rate:
            wav = librosa.resample(
                y=wav, orig_sr=int(sr), target_sr=self.sample_rate
            ).astype(np.float32)

        if self._mode == "qwen_base":
            from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

            mels = (
                mel_spectrogram(
                    torch.from_numpy(wav).unsqueeze(0),
                    n_fft=1024,
                    num_mels=128,
                    sampling_rate=self.sample_rate,
                    hop_size=256,
                    win_size=1024,
                    fmin=0,
                    fmax=12000,
                )
                .transpose(1, 2)
                .to(device=self.device, dtype=self.dtype)
            )
            emb = self.model(mels)
        else:
            inputs = self.processor(wav, sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = {
                k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                for k, v in inputs.items()
            }
            outputs = self.model(**inputs)

            emb = None
            if isinstance(outputs, torch.Tensor):
                emb = outputs
            else:
                for attr in (
                    "embeddings",
                    "speaker_embeddings",
                    "speaker_embedding",
                    "pooler_output",
                    "last_hidden_state",
                ):
                    val = getattr(outputs, attr, None)
                    if val is not None:
                        emb = val
                        break
                if emb is None and isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    emb = outputs[0]

            if emb is None:
                raise RuntimeError(f"Could not extract embedding tensor from model output: {type(outputs)}")

        emb_np = emb.detach().to(torch.float32).cpu().numpy()
        emb_np = np.squeeze(emb_np)
        if emb_np.ndim > 1:
            emb_np = emb_np.mean(axis=0)

        if emb_np.shape != (EXPECTED_DIM,):
            raise ValueError(
                f"Expected speaker embedding shape ({EXPECTED_DIM},), got {emb_np.shape}"
            )

        return emb_np.astype(np.float32)
