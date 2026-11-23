"""FastAPI web application for Qwen3-TTS Voice PCA Explorer."""

from __future__ import annotations

import asyncio
import gc
import io
import json
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from speaker_encoder import SpeakerEncoder
from steering import (
    get_best_device,
    get_preferred_dtype,
    load_audio_bytes,
    project_embedding,
    steer_embedding,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
PCA_NPZ_PATH = DATA_DIR / "pca_model.npz"
PCA_META_PATH = DATA_DIR / "pca_meta.json"
DEFAULT_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
LOCAL_FALLBACK_TTS_MODELS = (
    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
)

MODEL_CATALOG = [
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "label": "Base (ICL Cloning + PCA Steering)",
        "type": "base",
        "supports_icl": True,
        "supports_instruct": False,
        "has_builtin_speakers": False,
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        "label": "CustomVoice (Instruct + 9 Built-in + PCA)",
        "type": "custom_voice",
        "supports_icl": False,
        "supports_instruct": True,
        "has_builtin_speakers": True,
    },
    {
        "id": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        "label": "VoiceDesign (Instruct + PCA Steering)",
        "type": "voice_design",
        "supports_icl": False,
        "supports_instruct": True,
        "has_builtin_speakers": False,
    },
]


def _is_repo_cached(repo_id: str) -> bool:
    try:
        snapshot_download(repo_id=repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def _get_available_models() -> List[Dict[str, Any]]:
    models = []
    seen = set()
    for entry in MODEL_CATALOG:
        m = dict(entry)
        m["cached"] = _is_repo_cached(entry["id"])
        models.append(m)
        seen.add(entry["id"])
    if TTS_MODEL_REPO not in seen:
        models.insert(
            0,
            {
                "id": TTS_MODEL_REPO,
                "label": f"{TTS_MODEL_REPO} ({state.tts_model_type})",
                "type": state.tts_model_type,
                "supports_icl": state.tts_model_type == "base",
                "supports_instruct": state.tts_model_type != "base",
                "has_builtin_speakers": state.tts_model_type == "custom_voice",
                "cached": True,
            },
        )
    return models


def _resolve_tts_model_repo() -> tuple[str, bool]:
    """Resolve TTS model repo ID and whether to load strictly from local cache."""
    override = os.environ.get("QWEN3_TTS_MODEL")
    if override:
        try:
            snapshot_download(repo_id=override, local_files_only=True)
            return override, True
        except Exception:
            return override, False

    for repo_id in LOCAL_FALLBACK_TTS_MODELS:
        try:
            snapshot_download(repo_id=repo_id, local_files_only=True)
            return repo_id, True
        except Exception:
            continue

    return DEFAULT_TTS_MODEL, False


TTS_MODEL_REPO = DEFAULT_TTS_MODEL


class GenerateRequest(BaseModel):
    embedding_id: str
    targets: List[float]
    attribute_deltas: Optional[List[float]] = None
    text: str = Field(..., min_length=1, max_length=2000)
    instruct: Optional[str] = None
    language: str = "English"
    preserve_norm: bool = True
    use_icl: bool = True
    seed: Optional[int] = 42


class ExportEmbeddingRequest(BaseModel):
    embedding_id: str
    targets: List[float]
    attribute_deltas: Optional[List[float]] = None
    preserve_norm: bool = True
    name: str = "steered_voice"


class SwitchModelRequest(BaseModel):
    model_id: str


class AppState:
    def __init__(self) -> None:
        self.device: str = get_best_device()
        self.dtype: torch.dtype = get_preferred_dtype(self.device)
        self.tts_model: Optional[Qwen3TTSModel] = None
        self.speaker_encoder: Optional[SpeakerEncoder] = None

        self.pca_mean: Optional[np.ndarray] = None
        self.pca_components: Optional[np.ndarray] = None
        self.pca_meta: Dict[str, Any] = {}
        self.attr_vectors: Optional[np.ndarray] = None  # (M, 2048) unit-norm supervised directions
        self.attr_meta: List[Dict[str, Any]] = []

        self.embeddings: Dict[str, np.ndarray] = {}
        self.icl_prompts: Dict[str, Dict[str, Any]] = {}  # embedding_id -> {"ref_code": Tensor, "ref_text": str}
        self.presets_info: List[Dict[str, Any]] = []
        self.supported_languages: List[str] = ["English", "Auto"]
        self.tts_model_type: str = "custom_voice"
        self.lock = asyncio.Lock()


state = AppState()


def _seed_everything(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def _project_attributes(emb: np.ndarray) -> List[float]:
    if state.attr_vectors is None or state.pca_mean is None or len(state.attr_meta) == 0:
        return []
    return project_embedding(emb, state.pca_mean, state.attr_vectors).tolist()


def _build_supervised_attributes(
    preset_list: List[Dict[str, Any]],
    preset_embeddings: Optional[np.ndarray],
) -> None:
    """Compute gender, age, and per-accent steering directions from VCTK speaker embeddings.

    Age and accent directions are orthogonalized against the gender direction so shifting
    an accent or age slider does not accidentally flip the speaker's gender.
    """
    state.attr_vectors = None
    state.attr_meta = []
    if preset_embeddings is None or len(preset_list) < 10 or state.pca_mean is None or state.pca_components is None:
        return

    n = min(len(preset_list), preset_embeddings.shape[0])
    X = preset_embeddings[:n].astype(np.float64)
    mean = state.pca_mean.astype(np.float64)
    Xc = X - mean
    metas = [preset_list[i].get("metadata", {}) for i in range(n)]

    vectors: List[np.ndarray] = []
    attrs: List[Dict[str, Any]] = []

    def _add_attr(
        attr_id: str,
        name: str,
        low_label: str,
        high_label: str,
        vec: np.ndarray,
        count: int,
        pos_mask: Optional[np.ndarray] = None,
        ortho_against: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        v = vec.astype(np.float64).copy()
        if ortho_against is not None:
            v = v - float(np.dot(v, ortho_against)) * ortho_against
        norm = float(np.linalg.norm(v))
        if norm < 1e-8:
            return None
        v = v / norm
        projs = Xc @ v
        std = float(np.std(projs, ddof=1)) if n > 1 else 1.0
        pos_mean_sigma = None
        if pos_mask is not None and np.any(pos_mask) and std > 1e-8:
            pos_mean_sigma = float(np.mean(projs[pos_mask]) / std)

        v32 = v.astype(np.float32)
        pc_weights = (v32 @ state.pca_components.T).tolist()
        vectors.append(v32)
        attrs.append(
            {
                "index": len(attrs),
                "id": attr_id,
                "name": name,
                "low_label": low_label,
                "high_label": high_label,
                "count": int(count),
                "std": std,
                "pos_mean_sigma": pos_mean_sigma,
                "pc_weights": pc_weights,
            }
        )
        return v

    # 1. Gender (Male <-> Female)
    f_mask = np.array([m.get("gender") == "F" for m in metas], dtype=bool)
    m_mask = np.array([m.get("gender") == "M" for m in metas], dtype=bool)
    v_gender = None
    if f_mask.sum() >= 3 and m_mask.sum() >= 3:
        v_gender = _add_attr(
            attr_id="gender",
            name="Gender",
            low_label="Male",
            high_label="Female",
            vec=X[f_mask].mean(axis=0) - X[m_mask].mean(axis=0),
            count=int(f_mask.sum() + m_mask.sum()),
            pos_mask=f_mask,
        )

    # 2. Age (Younger <-> Older), orthogonalized against gender
    age_indices = [i for i, m in enumerate(metas) if m.get("age") is not None]
    if len(age_indices) >= 10:
        ages = np.array([float(metas[i]["age"]) for i in age_indices], dtype=np.float64)
        if np.std(ages) > 1e-6:
            ages_c = ages - ages.mean()
            v_age_raw = Xc[age_indices].T @ ages_c
            _add_attr(
                attr_id="age",
                name="Age",
                low_label="Younger",
                high_label="Older",
                vec=v_age_raw,
                count=len(age_indices),
                ortho_against=v_gender,
            )

    # 3. Accents (all accents with >= 3 speakers, sorted by speaker count desc), orthogonalized against gender
    accent_counts: Dict[str, int] = {}
    for m in metas:
        acc = m.get("accent")
        if acc and acc.lower() != "unknown":
            accent_counts[acc] = accent_counts.get(acc, 0) + 1

    for acc, cnt in sorted(accent_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if cnt < 3:
            continue
        pos_mask = np.array([m.get("accent") == acc for m in metas], dtype=bool)
        neg_mask = ~pos_mask
        if neg_mask.sum() < 3:
            continue
        nice_name = acc.replace("NorthernIrish", "Northern Irish").replace("SouthAfrican", "South African")
        _add_attr(
            attr_id=f"accent_{acc.lower()}",
            name=f"{nice_name} Accent",
            low_label=f"Non-{nice_name}",
            high_label=nice_name,
            vec=X[pos_mask].mean(axis=0) - X[neg_mask].mean(axis=0),
            count=cnt,
            pos_mask=pos_mask,
            ortho_against=v_gender,
        )

    if vectors:
        state.attr_vectors = np.stack(vectors, axis=0).astype(np.float32)
        state.attr_meta = attrs
        print(f"[startup] Computed {len(attrs)} supervised attribute steering vectors (gender-orthogonalized).")


def _load_pca_artifacts() -> None:
    if not PCA_NPZ_PATH.exists() or not PCA_META_PATH.exists():
        print(
            f"[startup] PCA artifacts not found at {DATA_DIR}. "
            "Starting in offline fallback mode (10 orthonormal exploration axes). "
            "Run `uv run python build_pca.py` later for VCTK semantic PCA axes."
        )
        rng = np.random.default_rng(2026)
        q, _ = np.linalg.qr(rng.standard_normal((2048, 10)))
        state.pca_mean = np.zeros(2048, dtype=np.float32)
        state.pca_components = q.T.astype(np.float32)  # (10, 2048) unit-norm orthogonal rows
        state.pca_meta = {
            "mode": "orthogonal_fallback",
            "dataset": "offline (run build_pca.py for VCTK PCA)",
            "num_speakers": 0,
            "num_clips": 0,
            "components": [
                {
                    "index": i,
                    "name": f"Axis {i + 1}",
                    "explained_variance_ratio": 0.1,
                    "min": -5.0,
                    "max": 5.0,
                    "std": 2.0,
                    "gender_r": None,
                    "age_r": None,
                    "accent_low": None,
                    "accent_high": None,
                }
                for i in range(10)
            ],
        }
        state.embeddings["pca_mean"] = state.pca_mean.copy()
        state.presets_info = [
            {
                "id": "pca_mean",
                "label": "Zero Vector (Upload a voice clip to steer)",
                "metadata": {"source": "Offline fallback — upload a .wav/.mp3 or record mic"},
                "projections": [0.0] * 10,
            }
        ]
        return

    npz = np.load(PCA_NPZ_PATH)
    state.pca_mean = npz["mean"].astype(np.float32)
    state.pca_components = npz["components"].astype(np.float32)

    with open(PCA_META_PATH, "r", encoding="utf-8") as f:
        state.pca_meta = json.load(f)

    # Build supervised attribute steering vectors (Gender, Age, Accents) from VCTK speakers
    preset_list = state.pca_meta.get("presets", [])
    preset_embeddings = npz["preset_embeddings"].astype(np.float32) if "preset_embeddings" in npz else None
    _build_supervised_attributes(preset_list, preset_embeddings)

    # Register PCA Mean Voice first
    state.embeddings["pca_mean"] = state.pca_mean.copy()
    mean_proj = project_embedding(state.pca_mean, state.pca_mean, state.pca_components).tolist()
    state.presets_info = [
        {
            "id": "pca_mean",
            "label": "PCA Mean Voice (Centroid)",
            "metadata": {"source": f"Mean of {state.pca_meta.get('num_speakers', '?')} VCTK speakers"},
            "projections": mean_proj,
            "attr_projections": _project_attributes(state.pca_mean),
        }
    ]

    if preset_embeddings is not None:
        for idx, pmeta in enumerate(preset_list):
            if idx >= preset_embeddings.shape[0]:
                break
            emb = preset_embeddings[idx]
            pid = pmeta["id"]
            state.embeddings[pid] = emb
            proj = project_embedding(emb, state.pca_mean, state.pca_components).tolist()
            state.presets_info.append(
                {
                    "id": pid,
                    "label": pmeta["label"],
                    "metadata": pmeta.get("metadata", {}),
                    "projections": proj,
                    "attr_projections": _project_attributes(emb),
                }
            )


def _load_tts_model_by_id(repo_id: str) -> None:
    """Unload any existing TTS model and load `repo_id` onto MPS/CPU."""
    global TTS_MODEL_REPO

    # Remove any old built-in CustomVoice presets before swapping
    state.presets_info = [p for p in state.presets_info if not p["id"].startswith("builtin_")]
    for k in [k for k in list(state.embeddings.keys()) if k.startswith("builtin_")]:
        del state.embeddings[k]

    if state.tts_model is not None:
        del state.tts_model
        state.tts_model = None
        gc.collect()
        if state.device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    local_only = _is_repo_cached(repo_id)
    TTS_MODEL_REPO = repo_id
    print(f"[model] Loading TTS model: {TTS_MODEL_REPO} (local_files_only={local_only})...")
    load_kwargs: Dict[str, Any] = {
        "dtype": state.dtype,
        "attn_implementation": "sdpa",
        "local_files_only": local_only,
    }
    if "mlx-community" in TTS_MODEL_REPO and "Base" in TTS_MODEL_REPO:
        # MLX Base checkpoints store speaker_encoder Conv1d weights as (O, K, I);
        # we already load the properly transposed speaker_encoder separately.
        load_kwargs["ignore_mismatched_sizes"] = True
    if state.device == "mps":
        load_kwargs["device_map"] = "mps"
    else:
        load_kwargs["device_map"] = "cpu"

    state.tts_model = Qwen3TTSModel.from_pretrained(TTS_MODEL_REPO, **load_kwargs)
    state.tts_model_type = str(getattr(state.tts_model.model, "tts_model_type", "custom_voice"))

    langs = state.tts_model.get_supported_languages()
    if langs:
        state.supported_languages = sorted(set(langs))

    # If a CustomVoice model is loaded, extract its 9 built-in speaker embeddings (2048-D)
    spk_id_map = getattr(getattr(state.tts_model.model.config, "talker_config", None), "spk_id", None)
    if isinstance(spk_id_map, dict) and spk_id_map and state.pca_mean is not None and state.pca_components is not None:
        embed_layer = state.tts_model.model.talker.get_input_embeddings()
        builtin_presets = []
        with torch.no_grad():
            for spk_name, tok_id in sorted(spk_id_map.items()):
                tid = torch.tensor(tok_id, device=embed_layer.weight.device, dtype=torch.long)
                vec = embed_layer(tid).detach().to(torch.float32).cpu().numpy().reshape(-1)
                if vec.shape == (2048,):
                    pid = f"builtin_{spk_name}"
                    state.embeddings[pid] = vec
                    proj = project_embedding(vec, state.pca_mean, state.pca_components).tolist()
                    builtin_presets.append(
                        {
                            "id": pid,
                            "label": f"CustomVoice · {spk_name.replace('_', ' ').title()}",
                            "metadata": {"source": f"Built-in {TTS_MODEL_REPO} speaker (id={tok_id})"},
                            "projections": proj,
                            "attr_projections": _project_attributes(vec),
                        }
                    )
        if builtin_presets:
            state.presets_info = state.presets_info[:1] + builtin_presets + state.presets_info[1:]
            print(f"[model] Added {len(builtin_presets)} built-in CustomVoice speakers to presets.")

    print(f"[model] TTS model ready: {TTS_MODEL_REPO} (type={state.tts_model_type})")


def _load_models() -> None:
    print(f"[startup] Device: {state.device}, dtype: {state.dtype}")
    print("[startup] Loading speaker encoder...")
    state.speaker_encoder = SpeakerEncoder(device=state.device, dtype=torch.float32)
    print(f"[startup] Speaker encoder ready: {state.speaker_encoder.model_id}")

    repo_id, _ = _resolve_tts_model_repo()
    _load_tts_model_by_id(repo_id)
    print("[startup] Models loaded and ready.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_pca_artifacts()
    _load_models()
    yield


app = FastAPI(title="Qwen3-TTS Voice PCA Explorer", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


def _build_meta_payload() -> Dict[str, Any]:
    return {
        "device": state.device,
        "dtype": str(state.dtype),
        "tts_model": TTS_MODEL_REPO,
        "tts_model_type": state.tts_model_type,
        "supports_icl": state.tts_model_type == "base",
        "supports_instruct": state.tts_model_type != "base",
        "has_builtin_speakers": state.tts_model_type == "custom_voice",
        "available_models": _get_available_models(),
        "mode": state.pca_meta.get("mode", "pca"),
        "num_speakers": state.pca_meta.get("num_speakers", 0),
        "num_clips": state.pca_meta.get("num_clips", 0),
        "dataset": state.pca_meta.get("dataset", ""),
        "components": state.pca_meta.get("components", []),
        "attributes": state.attr_meta,
        "presets": state.presets_info,
        "supported_languages": state.supported_languages,
    }


@app.get("/api/meta")
async def get_meta():
    return _build_meta_payload()


@app.post("/api/switch-model")
async def switch_model(req: SwitchModelRequest):
    allowed = {m["id"] for m in _get_available_models()}
    if req.model_id not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported model_id: {req.model_id}")

    if req.model_id == TTS_MODEL_REPO and state.tts_model is not None:
        return _build_meta_payload()

    async with state.lock:
        try:
            await asyncio.to_thread(_load_tts_model_by_id, req.model_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to switch model: {exc}") from exc

    return _build_meta_payload()


def _encode_ref_audio_and_codes(
    wav_chunk: np.ndarray, sr: int, ref_text: Optional[str]
) -> tuple[np.ndarray, Optional[torch.Tensor]]:
    assert state.speaker_encoder is not None
    emb = state.speaker_encoder.encode(wav_chunk, sr)
    ref_code = None
    if ref_text and ref_text.strip() and state.tts_model is not None:
        with torch.no_grad():
            enc = state.tts_model.model.speech_tokenizer.encode(wav_chunk, sr=sr)
            ref_code = enc.audio_codes[0].detach().cpu()
    return emb, ref_code


@app.post("/api/encode")
async def encode_audio(
    file: UploadFile = File(...),
    start_sec: Optional[float] = Form(None),
    end_sec: Optional[float] = Form(None),
    ref_text: Optional[str] = Form(None),
):
    if state.speaker_encoder is None or state.pca_mean is None or state.pca_components is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    filename = file.filename or "upload.wav"
    try:
        wav, sr = load_audio_bytes(raw, filename=filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {exc}") from exc

    full_duration_sec = float(len(wav) / max(sr, 1))

    # Optional chunk slicing (e.g. pick a 5-10s segment from the middle of a longer clip)
    s_sec = max(0.0, float(start_sec)) if start_sec is not None else 0.0
    e_sec = min(full_duration_sec, float(end_sec)) if end_sec is not None else full_duration_sec
    if e_sec <= s_sec:
        raise HTTPException(status_code=400, detail=f"Invalid chunk range: {s_sec:.2f}s to {e_sec:.2f}s")

    start_idx = int(round(s_sec * sr))
    end_idx = int(round(e_sec * sr))
    wav_chunk = wav[start_idx:end_idx]

    duration_sec = float(len(wav_chunk) / max(sr, 1))
    if duration_sec < 0.25:
        raise HTTPException(status_code=400, detail="Selected audio chunk is too short (< 0.25s)")

    ref_text_clean = ref_text.strip() if ref_text else ""
    async with state.lock:
        emb, ref_code = await asyncio.to_thread(
            _encode_ref_audio_and_codes, wav_chunk, sr, ref_text_clean or None
        )

    emb_id = f"custom_{uuid.uuid4().hex[:8]}"
    state.embeddings[emb_id] = emb
    has_icl = ref_code is not None and bool(ref_text_clean)
    if has_icl:
        state.icl_prompts[emb_id] = {
            "ref_code": ref_code,
            "ref_text": ref_text_clean,
        }

    projections = project_embedding(emb, state.pca_mean, state.pca_components).tolist()

    chunk_tag = f" [{s_sec:.1f}s–{e_sec:.1f}s]" if (start_sec is not None or end_sec is not None) else ""
    icl_tag = " · ICL" if has_icl else ""
    return {
        "embedding_id": emb_id,
        "label": f"Custom: {filename}{chunk_tag}{icl_tag}",
        "projections": projections,
        "attr_projections": _project_attributes(emb),
        "duration_sec": duration_sec,
        "full_duration_sec": full_duration_sec,
        "clip_start_sec": s_sec,
        "clip_end_sec": e_sec,
        "has_icl": has_icl,
    }


def _run_tts_generation(
    steered_f32: np.ndarray,
    text: str,
    instruct: Optional[str],
    language: str,
    seed: int,
    icl_prompt: Optional[Dict[str, Any]] = None,
) -> tuple[bytes, int, float, str]:
    cv = state.tts_model
    assert cv is not None

    t0 = time.perf_counter()
    _seed_everything(seed, state.device)

    # Cast steered float32 embedding to bfloat16 (or model dtype) on target device
    xvec = torch.from_numpy(steered_f32).to(device=cv.device, dtype=state.dtype)

    ids = cv._tokenize_texts([cv._build_assistant_text(text)])
    instruct_clean = instruct.strip() if instruct else ""
    # Base model does not use instruct_ids; CustomVoice/VoiceDesign do
    use_instruct = bool(instruct_clean) and state.tts_model_type != "base"
    told = [cv._tokenize_texts([cv._build_instruct_text(instruct_clean)])[0]] if use_instruct else None

    if icl_prompt is not None:
        ref_code = icl_prompt["ref_code"]
        ref_ids = [cv._tokenize_texts([cv._build_ref_text(icl_prompt["ref_text"])])[0]]
        voice = {
            "ref_code": [ref_code],
            "ref_spk_embedding": [xvec],
            "x_vector_only_mode": [False],
            "icl_mode": [True],
        }
        gen_mode = "icl+xvec"
    else:
        ref_code = None
        ref_ids = None
        voice = {
            "ref_code": [None],
            "ref_spk_embedding": [xvec],
            "x_vector_only_mode": [True],
            "icl_mode": [False],
        }
        gen_mode = "xvec_only"

    gen_call_kwargs: Dict[str, Any] = dict(
        input_ids=ids,
        voice_clone_prompt=voice,
        languages=[language],
        non_streaming_mode=True,
        **cv._merge_generate_kwargs(temperature=0.7, subtalker_temperature=0.7),
    )
    if told is not None:
        gen_call_kwargs["instruct_ids"] = told
    if ref_ids is not None:
        gen_call_kwargs["ref_ids"] = ref_ids

    with torch.no_grad():
        codes, _ = cv.model.generate(**gen_call_kwargs)
        if ref_code is not None:
            codes_for_decode = torch.cat([ref_code.to(codes[0].device), codes[0]], dim=0)
            wavs_all, rate = cv.model.speech_tokenizer.decode([{"audio_codes": codes_for_decode}])
            ref_len = int(ref_code.shape[0])
            total_len = int(codes_for_decode.shape[0])
            cut = int(ref_len / max(total_len, 1) * wavs_all[0].shape[0])
            wav = np.asarray(wavs_all[0][cut:], dtype=np.float32)
        else:
            wavs, rate = cv.model.speech_tokenizer.decode([{"audio_codes": codes[0]}])
            wav = np.asarray(wavs[0], dtype=np.float32)

    buf = io.BytesIO()
    sf.write(buf, wav, int(rate), format="WAV", subtype="PCM_16")
    elapsed = time.perf_counter() - t0
    return buf.getvalue(), int(rate), elapsed, gen_mode


@app.post("/api/generate")
async def generate_speech(req: GenerateRequest):
    if state.tts_model is None or state.pca_mean is None or state.pca_components is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    base_emb = state.embeddings.get(req.embedding_id)
    if base_emb is None:
        raise HTTPException(status_code=404, detail=f"Unknown embedding_id: {req.embedding_id}")

    steered = steer_embedding(
        base_embedding=base_emb,
        mean=state.pca_mean,
        components=state.pca_components,
        targets=req.targets,
        preserve_norm=req.preserve_norm,
        attribute_vectors=state.attr_vectors,
        attribute_deltas=req.attribute_deltas,
    )

    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    # Only apply ICL mode when the Base model is active
    icl_prompt = (
        state.icl_prompts.get(req.embedding_id)
        if (req.use_icl and state.tts_model_type == "base")
        else None
    )

    async with state.lock:
        try:
            wav_bytes, sr, elapsed, gen_mode = await asyncio.to_thread(
                _run_tts_generation,
                steered,
                req.text,
                req.instruct,
                req.language,
                seed,
                icl_prompt,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"TTS generation error: {exc}") from exc

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Generation-Time": f"{elapsed:.2f}",
            "X-Sample-Rate": str(sr),
            "X-Seed-Used": str(seed),
            "X-Gen-Mode": gen_mode,
        },
    )


@app.post("/api/import-embedding")
async def import_embedding(file: UploadFile = File(...)):
    if state.pca_mean is None or state.pca_components is None:
        raise HTTPException(status_code=503, detail="PCA model not loaded")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    filename = file.filename or "imported.safetensors"
    try:
        tensors = safetensors_load(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid .safetensors file: {exc}") from exc

    tensor = tensors.get("speaker_embedding")
    if tensor is None:
        # Fallback: first tensor with 2048 elements
        for v in tensors.values():
            if v.numel() == 2048:
                tensor = v
                break
    if tensor is None or tensor.numel() != 2048:
        raise HTTPException(
            status_code=400,
            detail="No 2048-D speaker_embedding tensor found in .safetensors file",
        )

    emb = tensor.detach().to(torch.float32).cpu().numpy().reshape(2048)
    emb_id = f"custom_{uuid.uuid4().hex[:8]}"
    state.embeddings[emb_id] = emb
    projections = project_embedding(emb, state.pca_mean, state.pca_components).tolist()

    stem = Path(filename).stem
    return {
        "embedding_id": emb_id,
        "label": f"Imported: {stem}",
        "projections": projections,
        "attr_projections": _project_attributes(emb),
        "norm": float(np.linalg.norm(emb)),
    }


@app.post("/api/export-embedding")
async def export_embedding(req: ExportEmbeddingRequest):
    if state.pca_mean is None or state.pca_components is None:
        raise HTTPException(status_code=503, detail="PCA model not loaded")

    base_emb = state.embeddings.get(req.embedding_id)
    if base_emb is None:
        raise HTTPException(status_code=404, detail=f"Unknown embedding_id: {req.embedding_id}")

    steered = steer_embedding(
        base_embedding=base_emb,
        mean=state.pca_mean,
        components=state.pca_components,
        targets=req.targets,
        preserve_norm=req.preserve_norm,
        attribute_vectors=state.attr_vectors,
        attribute_deltas=req.attribute_deltas,
    )

    tensor = torch.from_numpy(steered.astype(np.float32))
    raw_bytes = safetensors_save({"speaker_embedding": tensor})
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in req.name) or "steered_voice"

    return Response(
        content=raw_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.safetensors"'
        },
    )
