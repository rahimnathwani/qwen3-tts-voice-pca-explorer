"""Offline PCA pipeline for Qwen3-TTS 2048-D speaker embedding space.

Streams multi-speaker VCTK utterances from Hugging Face, extracts 2048-D speaker
embeddings using `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`, averages per
speaker in float32, fits top-K PCA, correlates each PC with VCTK metadata
(gender, age, accent), prints a summary table, and saves:
  - data/pca_model.npz
  - data/pca_meta.json
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from datasets import Audio, get_dataset_split_names, load_dataset
from scipy.stats import pearsonr

from speaker_encoder import EXPECTED_DIM, SpeakerEncoder
from steering import load_audio_bytes

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASET_CANDIDATES = [
    "mteb/cstr-vctk-accent-mini",
    "saeedzou/vctk-16khz",
]


def _decode_dataset_audio(audio_obj: Any) -> Tuple[np.ndarray, int]:
    """Decode a Hugging Face Audio entry whether decoded or raw bytes."""
    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            arr = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj["sampling_rate"])
            if arr.ndim > 1:
                arr = np.mean(arr, axis=-1).astype(np.float32)
            return arr, sr
        if "bytes" in audio_obj and audio_obj["bytes"] is not None:
            path = audio_obj.get("path") or "clip.wav"
            return load_audio_bytes(audio_obj["bytes"], filename=path)
        if "path" in audio_obj and audio_obj["path"]:
            arr, sr = sf.read(audio_obj["path"], dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = np.mean(arr, axis=-1).astype(np.float32)
            return arr, int(sr)
    raise ValueError(f"Unsupported audio object format: {type(audio_obj)}")


def _extract_metadata(row: Dict[str, Any], features: Any) -> Dict[str, Any]:
    """Extract normalized speaker_id, gender, age, accent from a dataset row."""
    # Speaker ID
    spk_id = None
    for key in ("speaker_id", "speaker", "spk_id", "client_id"):
        if key in row and row[key] is not None:
            spk_id = str(row[key]).strip()
            break
    if not spk_id:
        for key in ("audio_id", "id"):
            if key in row and row[key] is not None:
                raw_id = str(row[key]).strip()
                # VCTK audio_id format is typically "p225_001" or "p225_001_mic1"
                spk_id = Path(raw_id).stem.split("_")[0]
                break
    if not spk_id:
        # Try deriving from audio filename (e.g. p225_001.wav -> p225)
        audio = row.get("audio")
        if isinstance(audio, dict) and audio.get("path"):
            stem = Path(audio["path"]).stem
            spk_id = stem.split("_")[0]

    # Gender
    gender = None
    for key in ("gender", "sex"):
        if key in row and row[key] is not None:
            val = row[key]
            if features is not None and key in features and hasattr(features[key], "int2str") and isinstance(val, int):
                val = features[key].int2str(val)
            g = str(val).strip().upper()
            if g in ("F", "FEMALE"):
                gender = "F"
            elif g in ("M", "MALE"):
                gender = "M"
            break

    # Age
    age = None
    for key in ("age",):
        if key in row and row[key] is not None:
            try:
                age = float(row[key])
            except (ValueError, TypeError):
                age = None
            break

    # Accent
    accent = None
    for key in ("accent", "accents", "label", "region"):
        if key in row and row[key] is not None:
            val = row[key]
            if features is not None and key in features and hasattr(features[key], "int2str") and isinstance(val, int):
                val = features[key].int2str(val)
            acc = str(val).strip()
            if acc and acc.lower() != "none":
                accent = acc
            break

    return {
        "speaker_id": spk_id,
        "gender": gender,
        "age": age,
        "accent": accent,
    }


def _open_streaming_dataset(preferred_dataset: Optional[str] = None):
    candidates = [preferred_dataset] + [d for d in DATASET_CANDIDATES if d != preferred_dataset] if preferred_dataset else DATASET_CANDIDATES
    last_err = None
    for ds_name in candidates:
        try:
            print(f"[dataset] Trying streaming dataset: {ds_name}")
            try:
                splits = get_dataset_split_names(ds_name)
            except Exception:
                splits = ["train"]
            # Prefer train, then validation/test if needed
            ordered_splits = [s for s in ("train", "validation", "test") if s in splits] + [
                s for s in splits if s not in ("train", "validation", "test")
            ]
            if not ordered_splits:
                ordered_splits = ["train"]
            # Test opening the first split
            ds_iter = load_dataset(ds_name, split=ordered_splits[0], streaming=True)
            return ds_name, ordered_splits, ds_iter
        except Exception as exc:
            print(f"[dataset] Failed to open {ds_name}: {exc}")
            last_err = exc
    raise RuntimeError(f"Could not open any VCTK streaming dataset. Last error: {last_err}")


def _select_diverse_presets(
    speakers: List[Dict[str, Any]],
    projections: np.ndarray,
    num_presets: int = 6,
) -> List[int]:
    """Select up to `num_presets` diverse speakers spanning gender, accent, and PC1/PC2."""
    n = len(speakers)
    if n <= num_presets:
        return list(range(n))

    selected: List[int] = []
    used_accents: set[str] = set()

    # Try to alternate genders and pick distinct accents spread along PC1
    order_pc1 = np.argsort(projections[:, 0]).tolist()
    # Sample evenly across the PC1 spectrum
    buckets = np.array_split(order_pc1, num_presets)
    for b_idx, bucket in enumerate(buckets):
        want_gender = "F" if b_idx % 2 == 0 else "M"
        chosen = None
        # First pass: match desired gender & unused accent
        for idx in bucket:
            sp = speakers[int(idx)]
            acc = sp.get("accent") or ""
            if sp.get("gender") == want_gender and acc not in used_accents:
                chosen = int(idx)
                break
        # Second pass: unused accent
        if chosen is None:
            for idx in bucket:
                acc = speakers[int(idx)].get("accent") or ""
                if acc not in used_accents:
                    chosen = int(idx)
                    break
        # Fallback: middle of bucket
        if chosen is None and len(bucket) > 0:
            chosen = int(bucket[len(bucket) // 2])
        if chosen is not None and chosen not in selected:
            selected.append(chosen)
            if speakers[chosen].get("accent"):
                used_accents.add(speakers[chosen]["accent"])

    # Fill any remaining slots if needed
    for idx in order_pc1:
        if len(selected) >= num_presets:
            break
        if int(idx) not in selected:
            selected.append(int(idx))

    return selected[:num_presets]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PCA model for Qwen3-TTS speaker embeddings.")
    parser.add_argument("--dataset", type=str, default=None, help="Override Hugging Face dataset ID")
    parser.add_argument("--max-speakers", type=int, default=110, help="Target max unique speakers")
    parser.add_argument("--clips-per-speaker", type=int, default=5, help="Max utterances per speaker")
    parser.add_argument("--num-components", type=int, default=10, help="Number of PCA components (K)")
    parser.add_argument("--max-scan", type=int, default=15000, help="Max streamed rows to scan")
    parser.add_argument(
        "--num-presets",
        type=int,
        default=0,
        help="Number of preset speakers to save (0 = save all collected speakers)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("[encoder] Loading standalone 1.7B Qwen3 speaker encoder...")
    encoder = SpeakerEncoder()
    print(f"[encoder] Loaded {encoder.model_id} on {encoder.device} (target sr={encoder.sample_rate}Hz)")

    ds_name, splits, first_ds = _open_streaming_dataset(args.dataset)
    print(f"[dataset] Using {ds_name} (splits: {splits})")

    speaker_embs: Dict[str, List[np.ndarray]] = defaultdict(list)
    speaker_meta: Dict[str, Dict[str, Any]] = {}
    total_clips = 0
    scanned = 0

    for s_idx, split_name in enumerate(splits):
        if len(speaker_embs) >= args.max_speakers and all(
            len(v) >= args.clips_per_speaker for v in speaker_embs.values()
        ):
            break
        if scanned >= args.max_scan:
            break

        ds = first_ds if s_idx == 0 else load_dataset(ds_name, split=split_name, streaming=True)
        features = getattr(ds, "features", None)
        if features is not None and "audio" in features:
            ds = ds.cast_column("audio", Audio(decode=False))

        for row in ds:
            scanned += 1
            if scanned > args.max_scan:
                break

            meta = _extract_metadata(row, features)
            spk_id = meta["speaker_id"]
            if not spk_id:
                continue

            # If we already have enough unique speakers, skip new ones
            if spk_id not in speaker_embs and len(speaker_embs) >= args.max_speakers:
                # Check if all collected speakers are full
                if all(len(v) >= args.clips_per_speaker for v in speaker_embs.values()):
                    break
                continue

            if len(speaker_embs[spk_id]) >= args.clips_per_speaker:
                continue

            try:
                wav, sr = _decode_dataset_audio(row["audio"])
                if len(wav) < int(0.4 * sr):
                    continue
                emb = encoder.encode(wav, sr)
            except Exception as exc:
                if total_clips < 3:
                    print(f"[warn] Skipping clip for {spk_id}: {exc}")
                continue

            speaker_embs[spk_id].append(emb)
            if spk_id not in speaker_meta:
                speaker_meta[spk_id] = meta
            else:
                # Backfill any missing metadata fields
                for k in ("gender", "age", "accent"):
                    if speaker_meta[spk_id].get(k) is None and meta.get(k) is not None:
                        speaker_meta[spk_id][k] = meta[k]

            total_clips += 1
            if total_clips % 25 == 0:
                print(
                    f"[progress] {total_clips} clips encoded across {len(speaker_embs)} speakers "
                    f"(scanned {scanned} rows)"
                )

    num_speakers = len(speaker_embs)
    if num_speakers < 3:
        raise RuntimeError(f"Only collected {num_speakers} speakers; cannot fit PCA.")

    print(f"\n[pca] Averaging embeddings across {num_speakers} speakers ({total_clips} total clips)...")
    speaker_ids = sorted(speaker_embs.keys())
    speakers_list: List[Dict[str, Any]] = []
    X_list: List[np.ndarray] = []

    for spk_id in speaker_ids:
        avg_emb = np.mean(np.stack(speaker_embs[spk_id], axis=0), axis=0, dtype=np.float64).astype(np.float32)
        assert avg_emb.shape == (EXPECTED_DIM,), f"Unexpected embedding shape: {avg_emb.shape}"
        X_list.append(avg_emb)
        m = speaker_meta[spk_id]
        m["num_clips"] = len(speaker_embs[spk_id])
        speakers_list.append(m)

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, 2048)
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)  # (2048,)
    Xc = (X - mean).astype(np.float32)

    # Fit PCA via exact SVD in float32/float64
    _, S, Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    K = min(args.num_components, Vt.shape[0])
    components = Vt[:K].astype(np.float32)  # (K, 2048), unit-norm orthogonal rows

    # Verify unit-norm
    norms = np.linalg.norm(components, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"PCA components not unit-norm: {norms}"

    eigvals = (S ** 2) / max(num_speakers - 1, 1)
    total_var = float(eigvals.sum())
    explained_variance_ratio = (eigvals[:K] / total_var).astype(np.float32)

    projections = (Xc @ components.T).astype(np.float32)  # (N, K)
    pc_min = projections.min(axis=0).astype(np.float32)
    pc_max = projections.max(axis=0).astype(np.float32)
    pc_std = projections.std(axis=0, ddof=1 if num_speakers > 1 else 0).astype(np.float32)

    # Metadata correlation per PC
    genders_mask = [i for i, sp in enumerate(speakers_list) if sp.get("gender") in ("F", "M")]
    genders_vec = np.array([1.0 if speakers_list[i]["gender"] == "F" else 0.0 for i in genders_mask], dtype=np.float64)

    ages_mask = [i for i, sp in enumerate(speakers_list) if sp.get("age") is not None]
    ages_vec = np.array([float(speakers_list[i]["age"]) for i in ages_mask], dtype=np.float64)

    accent_groups: Dict[str, List[int]] = defaultdict(list)
    for i, sp in enumerate(speakers_list):
        if sp.get("accent"):
            accent_groups[sp["accent"]].append(i)
    # Prefer accents with >= 2 speakers if there are at least 2 such accents
    eligible_accents = {k: v for k, v in accent_groups.items() if len(v) >= 2}
    if len(eligible_accents) < 2:
        eligible_accents = accent_groups

    components_meta: List[Dict[str, Any]] = []
    for k in range(K):
        proj_k = projections[:, k].astype(np.float64)

        gender_r = None
        gender_p = None
        if len(genders_mask) >= 3 and np.std(genders_vec) > 0:
            r, p = pearsonr(proj_k[genders_mask], genders_vec)
            gender_r, gender_p = float(r), float(p)

        age_r = None
        age_p = None
        if len(ages_mask) >= 3 and np.std(ages_vec) > 0:
            r, p = pearsonr(proj_k[ages_mask], ages_vec)
            age_r, age_p = float(r), float(p)

        accent_low = None
        accent_high = None
        accent_low_mean = None
        accent_high_mean = None
        if len(eligible_accents) >= 2:
            acc_means = {acc: float(np.mean(proj_k[idxs])) for acc, idxs in eligible_accents.items()}
            accent_low = min(acc_means, key=acc_means.get)
            accent_high = max(acc_means, key=acc_means.get)
            accent_low_mean = acc_means[accent_low]
            accent_high_mean = acc_means[accent_high]

        components_meta.append(
            {
                "index": k,
                "name": f"PC{k + 1}",
                "explained_variance_ratio": float(explained_variance_ratio[k]),
                "min": float(pc_min[k]),
                "max": float(pc_max[k]),
                "std": float(pc_std[k]),
                "gender_r": gender_r,
                "gender_p": gender_p,
                "age_r": age_r,
                "age_p": age_p,
                "accent_low": accent_low,
                "accent_high": accent_high,
                "accent_low_mean": accent_low_mean,
                "accent_high_mean": accent_high_mean,
            }
        )

    # Print clean summary table
    print("\n" + "=" * 96)
    print(f"Qwen3-TTS 2048-D Speaker Space PCA Summary ({num_speakers} speakers, {total_clips} clips, {ds_name})")
    print(f"Cumulative explained variance (top {K} PCs): {explained_variance_ratio.sum() * 100:.2f}%")
    print("=" * 96)
    header = f"{'PC':<5} {'Var%':>6} {'Std':>7} {'[-2.5σ, +2.5σ]':>18} {'Gender r':>10} {'Age r':>9}  {'Accent Spread (Low <-> High)'}"
    print(header)
    print("-" * 96)
    for cm in components_meta:
        bound = 2.5 * cm["std"]
        g_str = f"{cm['gender_r']:+.3f}" if cm["gender_r"] is not None else "   n/a"
        a_str = f"{cm['age_r']:+.3f}" if cm["age_r"] is not None else "  n/a"
        acc_str = (
            f"{cm['accent_low']} <-> {cm['accent_high']}"
            if cm["accent_low"] and cm["accent_high"]
            else "n/a"
        )
        print(
            f"{cm['name']:<5} {cm['explained_variance_ratio']*100:>5.2f}% {cm['std']:>7.3f} "
            f"[{ -bound:>6.2f}, {bound:>+6.2f}] {g_str:>10} {a_str:>9}  {acc_str}"
        )
    print("=" * 96 + "\n")

    # Select preset speakers (featured diverse 6 first, then remaining speakers if num_presets == 0)
    featured_indices = _select_diverse_presets(speakers_list, projections, num_presets=min(6, num_speakers))
    if args.num_presets and args.num_presets > 0:
        preset_indices = _select_diverse_presets(speakers_list, projections, num_presets=args.num_presets)
    else:
        featured_set = set(featured_indices)
        remaining = [
            i
            for i in range(num_speakers)
            if i not in featured_set
        ]
        remaining.sort(
            key=lambda i: (
                speakers_list[i].get("accent") or "ZZZ",
                speakers_list[i].get("gender") or "Z",
                speakers_list[i].get("speaker_id") or "",
            )
        )
        preset_indices = featured_indices + remaining
    preset_embeddings = np.stack([X[i] for i in preset_indices], axis=0).astype(np.float32)
    presets_meta = []
    for idx in preset_indices:
        sp = speakers_list[idx]
        parts = [sp["speaker_id"]]
        if sp.get("gender"):
            parts.append("Female" if sp["gender"] == "F" else "Male")
        if sp.get("age") is not None:
            parts.append(f"{int(round(sp['age']))}y")
        if sp.get("accent"):
            parts.append(sp["accent"])
        presets_meta.append(
            {
                "id": f"vctk_{sp['speaker_id']}",
                "label": " · ".join(parts),
                "metadata": {
                    "speaker_id": sp["speaker_id"],
                    "gender": sp.get("gender"),
                    "age": sp.get("age"),
                    "accent": sp.get("accent"),
                    "num_clips": sp.get("num_clips"),
                },
            }
        )

    # Save artifacts
    npz_path = DATA_DIR / "pca_model.npz"
    np.savez(
        npz_path,
        mean=mean,
        components=components,
        explained_variance_ratio=explained_variance_ratio,
        pc_min=pc_min,
        pc_max=pc_max,
        pc_std=pc_std,
        preset_embeddings=preset_embeddings,
    )

    meta_out = {
        "dataset": ds_name,
        "encoder_model": encoder.model_id,
        "embedding_dim": EXPECTED_DIM,
        "num_speakers": num_speakers,
        "num_clips": total_clips,
        "num_components": K,
        "cumulative_explained_variance": float(explained_variance_ratio.sum()),
        "metadata_coverage": {
            "gender_count": len(genders_mask),
            "age_count": len(ages_mask),
            "accent_count": sum(len(v) for v in accent_groups.values()),
            "unique_accents": sorted(accent_groups.keys()),
        },
        "components": components_meta,
        "presets": presets_meta,
    }

    meta_path = DATA_DIR / "pca_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    print(f"[saved] {npz_path}")
    print(f"[saved] {meta_path}")


if __name__ == "__main__":
    main()
