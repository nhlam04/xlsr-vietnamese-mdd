"""
Dataset loading functions for TIMIT, LSVSC, and L2-ARCTIC.

Public API
----------
load_timit_data(path, split)
load_lsvsc_data(path, split, vietnamese_dict)
load_l2_arctic_data(path, speaker_ids, use_suitcase)
split_timit_by_speakers(data, val_ratio, seed)
calculate_audio_duration(data_list)
"""

import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa

from config import (
    TIMIT_PATH,
    LSVSC_PATH,
    L2_ARCTIC_PATH,
    TIMIT_CONFIG,
    LSVSC_CONFIG,
    L2_ARCTIC_CONFIG,
    MODEL_ARCHITECTURE,
)
from phonemes import (
    VIETNAMESE_SYL_DICT,
    get_timit_phoneme_label,
)
from vocab import prepare_phoneme_labels

# ── L2-ARCTIC speaker splits (defined once, used both here and in evaluate) ──
TEST_SPEAKERS  = ['ABA', 'BWC', 'ASI', 'HJK', 'EBVS', 'HQTV']

TRAIN_SPEAKERS = [
    'ERMS', 'MBMPS', 'NJS',
    'SKA',  'YBAA',  'ZHAA',
    'HKK',  'YDCK',  'YKWK',
    'LXC',  'NCC',   'TXHC',
    'RRBI', 'SVBI',  'TNI',
    'PNV',  'THV',   'TLV',
]


# ============================================================================
# TIMIT
# ============================================================================

def load_timit_data(timit_path: Path, split: str = 'TRAIN') -> List[Dict]:
    """
    Load TIMIT dataset with forced-aligned phoneme annotations (.PHN files).
    Applies TIMIT_PHONEME_MAP via get_timit_phoneme_label().

    Returns list of dicts: {audio_path, phonemes, dataset, split, speaker}.
    """
    data: List[Dict] = []
    split_path = timit_path / 'data' / split

    if not split_path.exists():
        print(f"Warning: {split_path} does not exist")
        return data

    wav_files = list(split_path.rglob('*.WAV'))
    print(f"Found {len(wav_files)} WAV files in TIMIT/{split}")
    mapping_stats: Dict[str, int] = defaultdict(int)

    for wav_file in wav_files:
        phn_file = wav_file.with_suffix('.PHN')
        if not phn_file.exists():
            continue

        speaker_id = wav_file.parent.name
        phonemes: List[str] = []

        with open(phn_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                original = parts[2].lower()
                mapped = get_timit_phoneme_label(original)
                phonemes.extend(mapped)
                if original in mapping_stats:   # track mapped symbols only
                    mapping_stats[original] += 1

        if phonemes:
            data.append({
                'audio_path': str(wav_file),
                'phonemes':   phonemes,
                'dataset':    'TIMIT',
                'split':      split,
                'speaker':    speaker_id,
            })

    print(f"Loaded {len(data)} utterances from TIMIT {split}")
    return data


# ============================================================================
# LSVSC
# ============================================================================

def load_lsvsc_data(
    lsvsc_path: Path,
    split: str = 'train',
    vietnamese_dict: Optional[Dict] = None,
    verbose: bool = False,
) -> List[Dict]:
    """
    Load LSVSC (northern-dialect only) with text→X-SAMPA conversion.

    Returns list of dicts: {audio_path, phonemes, dataset, split}.
    Entries with any unknown word are skipped in full.
    """
    if vietnamese_dict is None:
        vietnamese_dict = VIETNAMESE_SYL_DICT

    data: List[Dict] = []
    json_path = lsvsc_path / 'LSVSC' / f'LSVSC_{split}.json'
    wav_dir   = lsvsc_path / 'data'

    if not json_path.exists():
        print(f"Warning: {json_path} does not exist")
        return data
    if not wav_dir.exists():
        print(f"Warning: {wav_dir} does not exist")
        return data

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    skipped_non_northern = skipped_missing = skipped_unknown = skipped_empty = 0
    total_unknown = 0
    unknown_examples: Dict[str, int] = defaultdict(int)

    for ex_id, entry in metadata.items():
        wav_filename = entry['wav']
        text         = entry['text']
        class_info   = entry.get('class', '')

        # Northern dialect filter (region code 'Z' = 3rd character)
        if class_info and len(class_info) >= 5:
            class_str = class_info.strip('[]')
            if len(class_str) >= 3 and class_str[2] != '0':
                skipped_non_northern += 1
                continue

        audio_path = wav_dir / wav_filename
        if not audio_path.exists():
            skipped_missing += 1
            continue

        # Text normalisation
        text_cleaned = text.replace('\ufeff', '').replace('\ufffe', '')
        normalized   = unicodedata.normalize('NFC', text_cleaned.lower())
        words        = normalized.strip().split()

        # First pass: check for unknown words
        has_unknown = False
        for word in words:
            clean = word.strip('.,!?;:""''\u200b\u200c\u200d')
            if clean and clean not in vietnamese_dict:
                total_unknown += 1
                unknown_examples[clean] += 1
                has_unknown = True
                if verbose:
                    print(f"  WARNING ({split}/{ex_id}): unknown '{clean}' in: {text_cleaned[:80]}")

        if has_unknown:
            skipped_unknown += 1
            continue

        # Second pass: build phoneme sequence
        phonemes: List[str] = []
        for word in words:
            clean = word.strip('.,!?;:""''\u200b\u200c\u200d')
            if clean and clean in vietnamese_dict:
                phonemes.extend(vietnamese_dict[clean])

        if phonemes:
            data.append({
                'audio_path': str(audio_path),
                'phonemes':   phonemes,
                'dataset':    'LSVSC',
                'split':      split,
            })
        else:
            skipped_empty += 1

    print(f"\nLoaded {len(data)} utterances from LSVSC/{split} (northern only)")
    print(f"  Skipped: {skipped_non_northern} non-northern, {skipped_missing} missing audio, "
          f"{skipped_unknown} unknown words, {skipped_empty} empty phonemes")
    return data


# ============================================================================
# L2-ARCTIC helpers
# ============================================================================

def _clean_phone(phone: str) -> str:
    """Strip stress markers and diacritics from an ARPAbet phone."""
    return (phone.strip().lower()
            .replace('0', '').replace('1', '').replace('2', '')
            .replace('*', '').replace('_', '').replace('8', '').replace('`', ''))


def parse_textgrid_phones(textgrid_path: Path) -> List[str]:
    """
    Extract PPL (Perceived Phoneme Label) sequence from an L2-ARCTIC TextGrid.
    Annotation format: 'CPL,PPL,error_type' for errors, plain phone for correct.
    """
    phonemes: List[str] = []
    in_phones_tier = False

    with open(textgrid_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines:
        if 'name = "phones"' in line:
            in_phones_tier = True
            continue
        if in_phones_tier and 'item [3]:' in line:
            break
        if in_phones_tier and 'text = ' in line:
            stripped = line.strip()
            if stripped.startswith('text = "') and stripped.endswith('"'):
                phone = stripped[8:-1]
                if not phone or phone in ('', 'sp', 'spn'):
                    continue
                if ',' in phone:
                    parts = phone.split(',')
                    ppl = _clean_phone(parts[1]) if len(parts) > 1 else _clean_phone(parts[0])
                    if ppl and ppl != 'sil':
                        phonemes.append(ppl)
                else:
                    p = _clean_phone(phone)
                    if p and p != 'sil':
                        phonemes.append(p)

    return phonemes


def parse_textgrid_cpl_ppl(textgrid_path: Path) -> Tuple[List[str], List[str]]:
    """
    Extract both CPL and PPL sequences from an L2-ARCTIC TextGrid.
    Returns (cpl_phones, ppl_phones).
    """
    cpl_phones: List[str] = []
    ppl_phones: List[str] = []
    in_phones_tier = False

    with open(textgrid_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines:
        if 'name = "phones"' in line:
            in_phones_tier = True
            continue
        if in_phones_tier and 'item [3]:' in line:
            break
        if in_phones_tier and 'text = ' in line:
            stripped = line.strip()
            if stripped.startswith('text = "') and stripped.endswith('"'):
                phone = stripped[8:-1]
                if not phone or phone in ('', 'sp', 'spn'):
                    continue
                if ',' in phone:
                    parts = phone.split(',')
                    cpl = _clean_phone(parts[0])
                    ppl = _clean_phone(parts[1]) if len(parts) > 1 else cpl
                    if cpl and cpl != 'sil':
                        cpl_phones.append(cpl)
                    if ppl and ppl != 'sil':
                        ppl_phones.append(ppl)
                else:
                    p = _clean_phone(phone)
                    if p and p != 'sil':
                        cpl_phones.append(p)
                        ppl_phones.append(p)

    return cpl_phones, ppl_phones


def load_l2_arctic_data(
    l2_path: Path,
    speaker_ids: List[str],
    use_suitcase: bool = True,
) -> List[Dict]:
    """
    Load scripted (and optionally suitcase) L2-ARCTIC data for the given speakers.
    Extracts CPL+PPL for the linguistic model, PPL only for the standard model.
    """
    data: List[Dict] = []

    for speaker_id in speaker_ids:
        speaker_path = l2_path / speaker_id / speaker_id
        if not speaker_path.exists():
            print(f"Warning: Speaker path {speaker_path} does not exist")
            continue

        annotation_path = speaker_path / 'annotation'
        wav_path         = speaker_path / 'wav'
        if not annotation_path.exists() or not wav_path.exists():
            print(f"Warning: Missing annotation/wav for {speaker_id}")
            continue

        for ann_file in sorted(annotation_path.glob('*.TextGrid')):
            audio_file = wav_path / ann_file.name.replace('.TextGrid', '.wav')
            if not audio_file.exists():
                continue

            if MODEL_ARCHITECTURE == 'linguistic':
                cpl, ppl = parse_textgrid_cpl_ppl(ann_file)
                if cpl and ppl:
                    data.append({
                        'audio_path':        str(audio_file),
                        'phonemes':           ppl,
                        'canonical_phonemes': cpl,
                        'dataset':           'L2-ARCTIC',
                        'speaker':           speaker_id,
                    })
            else:
                phones = parse_textgrid_phones(ann_file)
                if phones:
                    data.append({
                        'audio_path': str(audio_file),
                        'phonemes':   phones,
                        'dataset':    'L2-ARCTIC',
                        'speaker':    speaker_id,
                    })

    # Suitcase corpus (spontaneous speech)
    if use_suitcase:
        suitcase_path = l2_path / 'suitcase_corpus' / 'suitcase_corpus'
        if suitcase_path.exists():
            for speaker_id in speaker_ids:
                if speaker_id.lower() in ('ska', 'asi'):
                    continue
                ann_file   = suitcase_path / 'annotation' / f'{speaker_id.lower()}.TextGrid'
                audio_file = suitcase_path / 'wav'        / f'{speaker_id.lower()}.wav'
                if not ann_file.exists() or not audio_file.exists():
                    continue

                if MODEL_ARCHITECTURE == 'linguistic':
                    cpl, ppl = parse_textgrid_cpl_ppl(ann_file)
                    if cpl and ppl:
                        data.append({'audio_path': str(audio_file), 'phonemes': ppl,
                                     'canonical_phonemes': cpl, 'dataset': 'L2-ARCTIC', 'speaker': speaker_id})
                else:
                    phones = parse_textgrid_phones(ann_file)
                    if phones:
                        data.append({'audio_path': str(audio_file), 'phonemes': phones,
                                     'dataset': 'L2-ARCTIC', 'speaker': speaker_id})

    print(f"Loaded {len(data)} utterances from L2-ARCTIC ({len(speaker_ids)} speakers)")
    return data


# ============================================================================
# Utilities
# ============================================================================

def split_timit_by_speakers(
    timit_data: List[Dict],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Speaker-aware train/val split for TIMIT.
    Ensures no speaker appears in both sets.
    """
    random.seed(seed)
    speaker_utterances: Dict[str, List[Dict]] = defaultdict(list)
    for item in timit_data:
        speaker_utterances[item.get('speaker', 'unknown')].append(item)

    speakers = list(speaker_utterances.keys())
    num_val  = max(1, int(len(speakers) * val_ratio))
    val_set  = set(random.sample(speakers, num_val))

    train_data = [u for s, us in speaker_utterances.items() if s not in val_set for u in us]
    val_data   = [u for s, us in speaker_utterances.items() if s in val_set     for u in us]

    print(f"  TIMIT speaker split: {len(speakers)-num_val} train, {num_val} val speakers "
          f"({len(train_data)} / {len(val_data)} utterances)")
    return train_data, val_data


def calculate_audio_duration(data_list: List[Dict]) -> float:
    """Return total duration in hours for a list of data dicts."""
    try:
        import soundfile as sf
    except ImportError:
        sf = None

    total = 0.0
    for item in data_list:
        path = item['audio_path']
        try:
            if sf:
                total += sf.info(path).duration
            else:
                total += librosa.get_duration(path=path)
        except Exception:
            pass
    return total / 3600


# ============================================================================
# High-level dataset assembly
# ============================================================================

def load_all_datasets():
    """
    Load and combine TIMIT / LSVSC / L2-ARCTIC according to CONFIG flags.

    Returns
    -------
    train_data, val_data, test_data : List[Dict]
    L2_TRAIN_SPEAKERS, L2_VAL_SPEAKERS, L2_TEST_SPEAKERS : List[str]
    """
    timit_train = timit_val = timit_test = []
    lsvsc_train = lsvsc_valid = lsvsc_test = []
    l2_train = l2_val = l2_test = []
    L2_TRAIN_SPEAKERS: List[str] = []
    L2_VAL_SPEAKERS:   List[str] = []
    L2_TEST_SPEAKERS:  List[str] = []
    dataset_label = 'L2-ARCTIC'

    # ── TIMIT ─────────────────────────────────────────────────────────────
    if TIMIT_CONFIG['use_for_training'] or TIMIT_CONFIG['use_for_validation']:
        timit_full = load_timit_data(TIMIT_PATH, 'TRAIN')
        t_train, t_val = split_timit_by_speakers(timit_full, val_ratio=0.15, seed=42)
        if TIMIT_CONFIG['use_for_training']:
            timit_train = t_train
        if TIMIT_CONFIG['use_for_validation']:
            timit_val = t_val

    if TIMIT_CONFIG['use_for_testing']:
        timit_test = load_timit_data(TIMIT_PATH, 'TEST')

    # ── LSVSC ─────────────────────────────────────────────────────────────
    ratio = LSVSC_CONFIG['sample_ratio']

    if LSVSC_CONFIG['use_for_training']:
        full = load_lsvsc_data(LSVSC_PATH, 'train', VIETNAMESE_SYL_DICT)
        lsvsc_train = full[:int(len(full) * ratio)]

    if LSVSC_CONFIG['use_for_validation']:
        full = load_lsvsc_data(LSVSC_PATH, 'valid', VIETNAMESE_SYL_DICT)
        lsvsc_valid = full[:int(len(full) * ratio)]

    if LSVSC_CONFIG['use_for_testing']:
        full = load_lsvsc_data(LSVSC_PATH, 'test', VIETNAMESE_SYL_DICT)
        lsvsc_test = full[:int(len(full) * ratio)]

    # ── L2-ARCTIC ─────────────────────────────────────────────────────────
    if (L2_ARCTIC_CONFIG['use_for_training']
            or L2_ARCTIC_CONFIG['use_for_validation']
            or L2_ARCTIC_CONFIG['use_for_testing']):

        mode = L2_ARCTIC_CONFIG['mode']
        if mode == 'vietnamese_only':
            L2_TRAIN_SPEAKERS = ['HQTV', 'PNV']
            L2_VAL_SPEAKERS   = ['THV']
            L2_TEST_SPEAKERS  = ['TLV']
            dataset_label     = 'L2-ARCTIC-Vietnamese'
        elif mode == 'all_speakers':
            L2_TRAIN_SPEAKERS = ['ABA','SKA','BWC','LXC','ASI','RRBI','HJK','HKK','EBVS','ERMS','HQTV','PNV']
            L2_VAL_SPEAKERS   = ['YBAA','NCC','SVBI','YDCK','MBMPS','THV']
            L2_TEST_SPEAKERS  = ['ZHAA','TXHC','TNI','YKWK','NJS','TLV']
            dataset_label     = 'L2-ARCTIC-All'
        elif mode == 'non_viet_train_viet_test':
            L2_TRAIN_SPEAKERS = ['ABA','SKA','YBAA','BWC','LXC','NCC','ASI','RRBI','SVBI',
                                  'HJK','HKK','YDCK','EBVS','ERMS','MBMPS']
            L2_VAL_SPEAKERS   = ['ZHAA','TXHC','TNI','YKWK','NJS']
            L2_TEST_SPEAKERS  = ['HQTV','PNV','THV','TLV']
            dataset_label     = 'L2-ARCTIC-NonViet'

        if L2_ARCTIC_PATH.exists():
            if L2_ARCTIC_CONFIG['use_for_training']:
                l2_train = load_l2_arctic_data(L2_ARCTIC_PATH, L2_TRAIN_SPEAKERS)
            if L2_ARCTIC_CONFIG['use_for_validation']:
                l2_val = load_l2_arctic_data(L2_ARCTIC_PATH, L2_VAL_SPEAKERS)
            if L2_ARCTIC_CONFIG['use_for_testing']:
                l2_test = load_l2_arctic_data(L2_ARCTIC_PATH, L2_TEST_SPEAKERS)
        else:
            print(f"Warning: L2-ARCTIC not found at {L2_ARCTIC_PATH}")

    train_data = timit_train + lsvsc_train + l2_train
    val_data   = timit_val   + lsvsc_valid + l2_val
    test_data  = timit_test  + lsvsc_test  + l2_test

    print(f"\nCombined splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return (train_data, val_data, test_data,
            L2_TRAIN_SPEAKERS, L2_VAL_SPEAKERS, L2_TEST_SPEAKERS)
