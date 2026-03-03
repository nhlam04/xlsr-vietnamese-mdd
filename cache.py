"""
Audio pre-processing and caching.

precompute_audio_to_cache()  — convert raw WAV files to 1D tensors and save to disk.
The resulting .pt file is loaded lazily into RAM just before training starts.
"""

import gc
import time
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import torch
from transformers import AutoFeatureExtractor

from config import MODEL_NAME, INPUT_PATH, OUTPUT_PATH, RESUME_FROM_FOLDER

# Global in-memory cache (populated by train_phoneme_classifier before training)
AUDIO_CACHE: Dict[str, torch.Tensor] = {}


# ============================================================================
# Pre-compute helper
# ============================================================================

def precompute_audio_to_cache(
    audio_paths: List[str],
    processor,
    audio_cache: Dict[str, torch.Tensor],
    save_path: Optional[Path] = None,
    save_every: int = 5_000,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Load and normalise audio files, writing results to *audio_cache*.

    To avoid RAM overflow on large datasets the cache is flushed to disk in
    chunks of *save_every* files.  All chunks are merged into *save_path* at
    the end and chunk files are deleted.

    Args:
        audio_paths:    Unique file paths to pre-process.
        processor:      Wav2Vec2 feature extractor (used for normalisation only).
        audio_cache:    Dict to accumulate tensors into (modified in-place).
        save_path:      If given, save merged cache to this .pt file.
        save_every:     Flush RAM cache to disk every N files.
        show_progress:  Print progress every 1 000 files.

    Returns:
        *audio_cache* (may be empty if chunks were flushed to disk).
    """
    total       = len(audio_paths)
    failed      = []
    saved_chunks: List[Path] = []
    chunk_counter = 0
    start_time = time.time()

    print(f"Pre-processing {total} audio files…")

    for i, audio_path in enumerate(audio_paths):
        try:
            audio_array, _ = librosa.load(audio_path, sr=16000)
            preprocessed = processor(
                audio_array,
                sampling_rate=16000,
                return_tensors='pt',
                padding=False,
            ).input_values[0]
            audio_cache[str(audio_path)] = preprocessed
            del audio_array, preprocessed

            # Flush to disk periodically
            if save_path and len(audio_cache) >= save_every:
                chunk_counter += 1
                chunk_path = save_path.parent / f"{save_path.stem}_chunk_{chunk_counter}.pt"
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(audio_cache, chunk_path)
                saved_chunks.append(chunk_path)
                audio_cache.clear()
                gc.collect()

            if show_progress and (i + 1) % 1_000 == 0:
                elapsed = time.time() - start_time
                rate    = (i + 1) / elapsed
                eta     = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{total}] Rate: {rate:.1f} f/s | ETA: {eta/60:.1f}m")

        except Exception as e:
            failed.append((audio_path, str(e)))

    # Save any remaining items
    if save_path and audio_cache:
        chunk_counter += 1
        chunk_path = save_path.parent / f"{save_path.stem}_chunk_{chunk_counter}.pt"
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(audio_cache, chunk_path)
        saved_chunks.append(chunk_path)
        audio_cache.clear()

    elapsed = time.time() - start_time
    print(f"\n✓ Pre-processing done in {elapsed/60:.1f}min ({total}/{total} files, "
          f"{len(failed)} failures)")

    # Merge chunks
    if save_path and saved_chunks:
        print(f"Merging {len(saved_chunks)} chunk(s) → {save_path.name}…")
        merged: Dict[str, torch.Tensor] = {}
        for cp in saved_chunks:
            merged.update(torch.load(cp, map_location='cpu'))
            del cp
        torch.save(merged, save_path)
        print(f"  ✓ Saved merged cache ({save_path.stat().st_size / 1e9:.2f} GB)")
        for cp in saved_chunks:
            try:
                cp.unlink()
            except Exception:
                pass
        audio_cache.clear()

    if failed:
        print(f"\n⚠  {len(failed)} file(s) failed:")
        for p, e in failed[:5]:
            print(f"  {p}: {e}")

    return audio_cache


# ============================================================================
# Path resolution for cache file
# ============================================================================

def resolve_cache_path() -> Path:
    """
    Return the path to the audio cache .pt file.
    When resuming, prefer INPUT_PATH if a cache already exists there.
    """
    output_cache = OUTPUT_PATH / 'model_phoneme_ctc' / 'audio_cache.pt'
    if RESUME_FROM_FOLDER is not None:
        input_cache = INPUT_PATH / 'model_phoneme_ctc' / 'audio_cache.pt'
        if input_cache.exists():
            return input_cache
    return output_cache


def load_processor_for_caching():
    """Lightweight: load only the feature extractor (no model weights)."""
    print(f"Loading feature extractor from {MODEL_NAME}…")
    proc = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    print("  ✓ Feature extractor ready")
    return proc
