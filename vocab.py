"""
Build the unified ARPAbet + Vietnamese X-SAMPA phoneme vocabulary and provide
helpers for converting phoneme sequences to/from vocabulary indices.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List

from config import (
    LSVSC_CONFIG,
    MODEL_ARCHITECTURE,
    INPUT_PATH,
    OUTPUT_PATH,
    RESUME_FROM_FOLDER,
)
from phonemes import (
    TIMIT_ARPABET_PHONEMES,
    VIETNAMESE_PHONEMES,
    SILENCE_PHONEMES,
    VI_G2P_AVAILABLE,
    get_vietnamese_phoneme_label,
)


# ============================================================================
# Build vocabulary
# ============================================================================

def build_vocab(include_vietnamese: bool = True) -> Dict[str, int]:
    """
    Build a unified phoneme vocabulary.

    Special tokens are always at indices 0-2:
        [PAD] = 0, [UNK] = 1, [CTC] = 2

    Args:
        include_vietnamese: whether to add Vietnamese X-SAMPA labels.

    Returns:
        {label: index} mapping.
    """
    all_phonemes = set(TIMIT_ARPABET_PHONEMES)

    if include_vietnamese and VI_G2P_AVAILABLE and VIETNAMESE_PHONEMES:
        all_phonemes.update(VIETNAMESE_PHONEMES)

    vocab: Dict[str, int] = {
        '[PAD]': 0,
        '[UNK]': 1,
        '[CTC]': 2,   # CTC blank token
    }
    for idx, phoneme in enumerate(sorted(all_phonemes), start=3):
        vocab[phoneme] = idx

    return vocab


# Decide whether to include Vietnamese based on LSVSC usage
_use_lsvsc = (
    LSVSC_CONFIG['use_for_training']
    or LSVSC_CONFIG['use_for_validation']
    or LSVSC_CONFIG['use_for_testing']
)
vocab: Dict[str, int] = build_vocab(include_vietnamese=_use_lsvsc)
id2phoneme: Dict[int, str] = {v: k for k, v in vocab.items()}

print(f"✓ Vocabulary built: {len(vocab)} tokens "
      f"({len(vocab) - 3} phonemes + 3 special tokens)")


# ============================================================================
# Vocabulary persistence helpers
# ============================================================================

def save_vocab(output_dir: Path) -> Path:
    """
    Save the vocabulary to <output_dir>/vocabs/vocab_arpabet_xsampa.json.
    Reuses an existing file from INPUT_PATH when resuming.
    """
    vocab_dir = output_dir / 'vocabs'
    vocab_dir.mkdir(exist_ok=True, parents=True)
    vocab_path = vocab_dir / 'vocab_arpabet_xsampa.json'

    if RESUME_FROM_FOLDER is not None:
        input_vocab_path = INPUT_PATH / 'model_phoneme_ctc' / 'vocabs' / 'vocab_arpabet_xsampa.json'
        if input_vocab_path.exists():
            shutil.copy2(input_vocab_path, vocab_path)
            print(f"✓ Reused vocabulary from checkpoint: {input_vocab_path.name}")
            return vocab_path

    if not vocab_path.exists():
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved vocabulary → {vocab_path}")
    else:
        print(f"✓ Vocabulary already exists at {vocab_path}")

    return vocab_path


# ============================================================================
# Phoneme-to-index conversion
# ============================================================================

def prepare_phoneme_labels(phonemes: List[str], is_vietnamese: bool = False) -> List[int]:
    """
    Convert a phoneme sequence to vocabulary indices.

    Silence phonemes are dropped.  Unknown phonemes map to [UNK].

    Args:
        phonemes:      List of phoneme symbol strings.
        is_vietnamese: If True, applies VI_PHONEME_PREFIX_MAP before lookup.

    Returns:
        List of integer vocabulary indices.
    """
    indices: List[int] = []
    for phoneme in phonemes:
        if phoneme.lower() in SILENCE_PHONEMES:
            continue

        label = get_vietnamese_phoneme_label(phoneme) if is_vietnamese else phoneme.lower()

        if label in vocab:
            indices.append(vocab[label])
        else:
            indices.append(vocab['[UNK]'])
            tag = "Vietnamese" if is_vietnamese else "ARPAbet"
            print(f"Warning: Unknown {tag} phoneme '{phoneme}' (label '{label}'), using [UNK]")

    return indices
