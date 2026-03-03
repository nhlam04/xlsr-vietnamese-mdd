"""
Data collator for CTC training.

Loads pre-processed 1D audio tensors from RAM cache, pads audio and labels,
and (for the linguistic model) also pads canonical_labels.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import librosa
import torch
from transformers import AutoFeatureExtractor

from config import MODEL_ARCHITECTURE, MODEL_NAME
from cache import AUDIO_CACHE


@dataclass
class DataCollatorCTCWithPadding:
    """
    Collate a batch of samples stored as plain dicts.

    Audio is served from the in-memory RAM cache (1-D preprocessed tensors).
    A cache-miss fallback loads + normalises on-the-fly, but this should never
    happen during normal training if the cache was pre-built correctly.
    """

    tokenizer: Any
    audio_cache: Dict[str, torch.Tensor]
    padding: bool = True

    # Internal counters (not part of the dataclass constructor API)
    total_batches: int = field(default=0, init=False, repr=False)
    cache_hits:    int = field(default=0, init=False, repr=False)
    cache_misses:  int = field(default=0, init=False, repr=False)
    _collation_times: List[float] = field(default_factory=list, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_audio(self, audio_path: str) -> torch.Tensor:
        if audio_path in self.audio_cache:
            self.cache_hits += 1
            return self.audio_cache[audio_path]

        # On-the-fly fallback
        print(f"⚠ Cache miss: {audio_path} — preprocessing on the fly")
        if self._processor is None:
            self._processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        audio_array, _ = librosa.load(audio_path, sr=16000)
        tensor = self._processor(
            audio_array, sampling_rate=16000, return_tensors='pt', padding=False
        ).input_values[0]
        self.audio_cache[audio_path] = tensor
        self.cache_misses += 1
        return tensor

    # ── main call ─────────────────────────────────────────────────────────────

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        start = time.time()

        batch_audio: List[torch.Tensor] = []
        batch_labels: List[List[int]]   = []
        batch_canonical: List[List[int]] = [] if MODEL_ARCHITECTURE == 'linguistic' else None  # type: ignore[assignment]

        for item in features:
            batch_audio.append(self._get_audio(item['audio_path']))
            batch_labels.append(item['labels'])
            if MODEL_ARCHITECTURE == 'linguistic':
                batch_canonical.append(item.get('canonical_labels', item['labels']))  # type: ignore[index]

        if not batch_audio:
            raise ValueError("Empty batch in DataCollator")

        # ── pad audio ────────────────────────────────────────────────────────
        max_a = max(len(a) for a in batch_audio)
        padded_audio = torch.zeros(len(batch_audio), max_a, dtype=torch.float32)
        attn_mask    = torch.zeros(len(batch_audio), max_a, dtype=torch.long)
        for i, a in enumerate(batch_audio):
            padded_audio[i, :len(a)] = a
            attn_mask[i, :len(a)]    = 1

        # ── pad labels ───────────────────────────────────────────────────────
        max_l = max(len(l) for l in batch_labels)
        padded_labels = torch.full((len(batch_labels), max_l), -100, dtype=torch.long)
        for i, labels in enumerate(batch_labels):
            t = torch.tensor(labels, dtype=torch.long)
            padded_labels[i, :len(labels)] = t

        # ── pad canonical labels (linguistic model) ───────────────────────────
        padded_canonical = None
        if MODEL_ARCHITECTURE == 'linguistic' and batch_canonical is not None:
            max_c = max(len(c) for c in batch_canonical)
            padded_canonical = torch.zeros(len(batch_canonical), max_c, dtype=torch.long)
            for i, c in enumerate(batch_canonical):
                t = torch.tensor(c, dtype=torch.long)
                padded_canonical[i, :len(c)] = t

        # ── timing / stats ────────────────────────────────────────────────────
        self._collation_times.append(time.time() - start)
        self.total_batches += 1
        if self.total_batches % 50 == 0:
            hit_rate = (self.cache_hits / max(1, self.cache_hits + self.cache_misses)) * 100
            avg_ms   = sum(self._collation_times[-50:]) / min(50, len(self._collation_times)) * 1_000
            print(f"Batch {self.total_batches}: cache={hit_rate:.1f}% hits | "
                  f"collate={avg_ms:.0f}ms/batch")

        batch_dict: Dict[str, torch.Tensor] = {
            'input_values':  padded_audio,
            'attention_mask': attn_mask,
            'labels':         padded_labels,
        }
        if MODEL_ARCHITECTURE == 'linguistic' and padded_canonical is not None:
            batch_dict['canonical_labels'] = padded_canonical

        return batch_dict
