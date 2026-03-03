"""
Evaluation utilities.

Key public API:
    evaluate_classifier(model_dir, test_dataset, map_to_standard_arpabet) → (metrics, preds, refs, canonicals)
    decode_ctc_predictions(logits, processor) → list[str]
    calculate_per(predictions, references) → dict
    calculate_detailed_errors(predictions, references) → dict
    collect_all_errors_with_context(predictions, references, sample_ids) → list[dict]
    write_detailed_errors_to_file(errors, output_path) → None
    map_timit_to_standard_arpabet(seq) → str
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from cache import AUDIO_CACHE
from config import MODEL_ARCHITECTURE, OUTPUT_PATH
from vocab import id2phoneme, vocab


# ============================================================================
# Phoneme mapping helpers
# ============================================================================

# Closures → canonical arpabet (used for TIMIT / L2-ARCTIC evaluation)
_CLOSURE_MAP = {
    'bcl': 'b', 'dcl': 'd', 'gcl': 'g',
    'pcl': 'p', 'tcl': 't', 'kcl': 'k',
    'epi': 'pau',
}
_TIMIT_TO_STD_MAP = {
    'tcl': 't', 'kcl': 'k', 'pcl': 'p',
    # Optionally fold others
}


def map_timit_to_standard_arpabet(phoneme_sequence: str) -> str:
    """
    Map TIMIT closure phones (tcl/kcl/pcl) → t/k/p for evaluation.

    Args:
        phoneme_sequence: Space-separated phoneme string

    Returns:
        Mapped phoneme sequence as space-separated string
    """
    phonemes = phoneme_sequence.split()
    mapped   = [_TIMIT_TO_STD_MAP.get(p, p) for p in phonemes]
    return ' '.join(mapped)


# ============================================================================
# CTC decoding
# ============================================================================

def decode_ctc_predictions(logits: torch.Tensor, processor) -> List[str]:
    """
    Greedy CTC decode.

    Matches the notebook implementation:
    1. Collapse consecutive duplicate tokens (torch.unique_consecutive)
    2. Remove CTC blank token (ID 2)
    3. Decode to phoneme strings joined with spaces

    Args:
        logits: (batch, time, vocab) tensor
        processor: Wav2Vec2Processor (unused, kept for API compatibility)

    Returns:
        List of space-separated phoneme strings, one per sample.
    """
    pred_ids  = torch.argmax(logits, dim=-1)
    ctc_blank = vocab.get('[CTC]', 2)
    results   = []
    for seq in pred_ids:
        # CTC decoding: collapse consecutive duplicates then remove blank
        collapsed = torch.unique_consecutive(seq)
        filtered  = collapsed[collapsed != ctc_blank]
        phonemes  = []
        for token_id in filtered:
            token_id_int = int(token_id.item())
            if token_id_int in id2phoneme:
                phoneme = id2phoneme[token_id_int]
                if phoneme not in ('[PAD]', '[UNK]', '[CTC]'):
                    phonemes.append(phoneme)
        results.append(' '.join(phonemes))
    return results


# ============================================================================
# PER calculation
# ============================================================================

def calculate_per(
    predictions: List[str],
    references:  List[str],
) -> Dict:
    """
    Compute Phoneme Error Rate using Levenshtein distance.

    Args:
        predictions: List of predicted phoneme sequences (space-separated strings)
        references:  List of reference phoneme sequences (space-separated strings)
    """
    try:
        import editdistance
    except ImportError:
        raise ImportError("Install editdistance: pip install editdistance")

    total_distance = 0
    total_reference_length = 0
    correct_sequences = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens  = ref.split()
        distance = editdistance.eval(ref_tokens, pred_tokens)
        total_distance += distance
        total_reference_length += len(ref_tokens)
        if distance == 0:
            correct_sequences += 1

    per = (total_distance / total_reference_length * 100) if total_reference_length > 0 else 0.0
    seq_acc = (correct_sequences / len(predictions) * 100) if predictions else 0.0

    return {
        'per':               per,
        'sequence_accuracy': seq_acc,
        'total_edits':       total_distance,
        'total_phonemes':    total_reference_length,
        'total_sequences':   len(predictions),
        'correct_sequences': correct_sequences,
    }


# ============================================================================
# Detailed error analysis
# ============================================================================

def calculate_detailed_errors(
    predictions: List[str],
    references:  List[str],
) -> Dict:
    """
    Break down PER into substitutions / deletions / insertions.

    Args:
        predictions: List of predicted phoneme sequences (space-separated strings)
        references:  List of reference phoneme sequences (space-separated strings)
    """
    from metrics import align_sequences, _get_ops

    subs = dels = ins = 0
    total_phones = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens  = ref.split()
        ref_a, hyp_a = align_sequences(ref_tokens, pred_tokens)
        ops  = _get_ops(ref_a, hyp_a)
        subs += ops.count('S')
        dels += ops.count('D')
        ins  += ops.count('I')
        total_phones += len(ref_tokens)

    total_edits = subs + dels + ins

    return {
        'substitutions':   subs,
        'deletions':       dels,
        'insertions':      ins,
        'total_edits':     total_edits,
        'total_phonemes':  total_phones,
        'sub_rate':        subs / total_phones * 100 if total_phones else 0,
        'del_rate':        dels / total_phones * 100 if total_phones else 0,
        'ins_rate':        ins  / total_phones * 100 if total_phones else 0,
    }


def collect_all_errors_with_context(
    predictions:  List[str],
    references:   List[str],
    sample_ids:   List,
) -> List[Dict]:
    """
    Return one record per error containing context (surrounding phones).

    Args:
        predictions: List of predicted phoneme sequences (space-separated strings)
        references:  List of reference phoneme sequences (space-separated strings)
        sample_ids:  List of sample identifiers
    """
    from metrics import align_sequences, _get_ops

    records = []
    for pred, ref, sid in zip(predictions, references, sample_ids):
        pred_tokens = pred.split()
        ref_tokens  = ref.split()
        ref_a, hyp_a = align_sequences(ref_tokens, pred_tokens)
        ops  = _get_ops(ref_a, hyp_a)

        for i, op in enumerate(ops):
            ctx_ref  = ref_a[max(0, i-2): i+3]
            ctx_hyp  = hyp_a[max(0, i-2): i+3]
            ref_ph   = ref_a[i]
            hyp_ph   = hyp_a[i]

            if op != 'C':
                records.append({
                    'sample_id':  sid,
                    'operation':  op,
                    'reference':  ref_ph,
                    'prediction': hyp_ph,
                    'context_ref': ' '.join(ctx_ref),
                    'context_hyp': ' '.join(ctx_hyp),
                })
    return records


def write_detailed_errors_to_file(errors: List[Dict], output_path: Path) -> None:
    """Write error records to CSV."""
    if not errors:
        print("No errors to write.")
        return

    fields = ['sample_id', 'operation', 'reference', 'prediction', 'context_ref', 'context_hyp']
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    print(f"  ✓ {len(errors)} error records → {output_path}")


# ============================================================================
# Main evaluation runner
# ============================================================================

def evaluate_classifier(
    model_dir:              str,
    test_dataset,
    map_to_standard_arpabet: bool = False,
) -> Tuple[Dict, List, List, List]:
    """
    Load a trained model from *model_dir* and evaluate on *test_dataset*.

    Returns:
        metrics     – dict with per, sequence_accuracy
        predictions – list[str]  (space-separated phoneme strings)
        references  – list[str]  (space-separated phoneme strings)
        canonicals  – list[str]  (space-separated phoneme strings; empty if not L2-ARCTIC)
    """
    from transformers import Wav2Vec2Processor

    model_dir = Path(model_dir)

    # ── Load processor ───────────────────────────────────────────────────
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))

    # ── Load model ───────────────────────────────────────────────────────
    if MODEL_ARCHITECTURE == 'linguistic':
        from models import Wav2Vec2_Linguistic
        model = Wav2Vec2_Linguistic.from_pretrained(str(model_dir))
    else:
        from transformers import Wav2Vec2ForCTC
        model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    print(f"  ✓ Model loaded ({MODEL_ARCHITECTURE}) on {device}")

    # ── Per-sample inference (matches notebook) ──────────────────────────────
    predictions_all: List[str] = []
    references_all:  List[str] = []
    canonicals_all:  List[str] = []

    id2phoneme_local = {i: p for p, i in vocab.items()}

    print(f"Running inference on {len(test_dataset)} samples...")
    for i, item in enumerate(test_dataset):
        # ── Audio ──────────────────────────────────────────────────────────
        audio_path = item.get('audio_path', '')
        if audio_path and audio_path in AUDIO_CACHE:
            audio_tensor = AUDIO_CACHE[audio_path]
        elif audio_path:
            try:
                import librosa
                audio_array, _ = librosa.load(audio_path, sr=16_000, mono=True)
                audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
            except Exception:
                audio_tensor = torch.tensor(item['input_values'], dtype=torch.float32)
        else:
            audio_tensor = torch.tensor(item['input_values'], dtype=torch.float32)

        input_values = audio_tensor.unsqueeze(0).to(device)

        # ── Model forward ──────────────────────────────────────────────────
        with torch.no_grad():
            if MODEL_ARCHITECTURE == 'linguistic':
                canon_ids = item.get('canonical_labels', item['labels'])
                canon_tensor = torch.tensor([canon_ids], dtype=torch.long).to(device)
                logits = model(input_values=input_values, canonical_labels=canon_tensor).logits
            else:
                logits = model(input_values=input_values).logits

        # ── Decode prediction ──────────────────────────────────────────────
        pred_str = decode_ctc_predictions(logits.cpu(), processor)[0]
        predictions_all.append(pred_str)

        # ── Decode reference (PPL / labels) ───────────────────────────────
        ref_tokens = [
            id2phoneme_local[lid]
            for lid in item['labels']
            if lid >= 0 and lid in id2phoneme_local
            and id2phoneme_local[lid] not in ('[PAD]', '[UNK]', '[CTC]')
        ]
        references_all.append(' '.join(ref_tokens))

        # ── Decode canonical (CPL) ─────────────────────────────────────────
        canon_ids = item.get('canonical_labels', item['labels'])
        canon_tokens = [
            id2phoneme_local[lid]
            for lid in canon_ids
            if lid >= 0 and lid in id2phoneme_local
            and id2phoneme_local[lid] not in ('[PAD]', '[UNK]', '[CTC]')
        ]
        canonicals_all.append(' '.join(canon_tokens))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_dataset)} samples...")

    # ── Optional phone collapse (preds + canonicals ONLY, NOT refs) ───────
    if map_to_standard_arpabet:
        print("  Applying TIMIT to standard ARPAbet mapping...")
        predictions_all = [map_timit_to_standard_arpabet(p) for p in predictions_all]
        canonicals_all  = [map_timit_to_standard_arpabet(c) for c in canonicals_all]

    # ── PER ───────────────────────────────────────────────────────────────
    metrics = calculate_per(predictions_all, references_all)

    print(f"\nEvaluation Results:")
    print(f"  Phoneme Error Rate (PER): {metrics['per']:.2f}%")
    print(f"  Sequence Accuracy: {metrics['sequence_accuracy']:.2f}%")
    print(f"  Total edits: {metrics['total_edits']}")
    print(f"  Total phonemes: {metrics['total_phonemes']}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, predictions_all, references_all, canonicals_all
