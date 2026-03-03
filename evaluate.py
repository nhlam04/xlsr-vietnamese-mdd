"""
Evaluation utilities.

Key public API:
    evaluate_classifier(model_dir, test_dataset, map_to_standard_arpabet) → (metrics, preds, refs, canonicals)
    decode_ctc_predictions(logits, processor) → list[list[str]]
    calculate_per(predictions, references) → dict
    calculate_detailed_errors(predictions, references) → dict
    collect_all_errors_with_context(predictions, references, sample_ids) → list[dict]
    write_detailed_errors_to_file(errors, output_path) → None
    map_timit_to_standard_arpabet(seq) → list[str]
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

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


def map_timit_to_standard_arpabet(phoneme_sequence: List[str]) -> List[str]:
    """Map TIMIT closure phones (tcl/kcl/pcl) → t/k/p for evaluation."""
    return [_TIMIT_TO_STD_MAP.get(p, p) for p in phoneme_sequence]


# ============================================================================
# CTC decoding
# ============================================================================

def decode_ctc_predictions(logits: torch.Tensor, processor) -> List[List[str]]:
    """
    Greedy CTC decode.

    Args:
        logits: (batch, time, vocab) tensor
        processor: Wav2Vec2Processor (for blank_token_id / pad_token_id)

    Returns:
        List of phoneme lists, one per sample.
    """
    pred_ids  = torch.argmax(logits, dim=-1)
    pad_id    = vocab.get('[PAD]', 0)
    ctc_blank = vocab.get('[CTC]', 2)
    results   = []
    for seq in pred_ids:
        phonemes, prev = [], -1
        for idx in seq.tolist():
            if idx == ctc_blank:
                prev = idx
                continue
            if idx == pad_id:
                break
            if idx != prev:
                label = id2phoneme.get(idx, '[UNK]')
                if label not in ('[PAD]', '[UNK]', '[CTC]'):
                    phonemes.append(label)
            prev = idx
        results.append(phonemes)
    return results


# ============================================================================
# PER calculation
# ============================================================================

def calculate_per(
    predictions: List[List[str]],
    references:  List[List[str]],
) -> Dict:
    """Compute Phoneme Error Rate using Levenshtein distance."""
    try:
        import editdistance
    except ImportError:
        raise ImportError("Install editdistance: pip install editdistance")

    total_edits = 0
    total_phones = 0
    exact_match = 0

    for pred, ref in zip(predictions, references):
        edit_dist   = editdistance.eval(pred, ref)
        total_edits += edit_dist
        total_phones += len(ref)
        if pred == ref:
            exact_match += 1

    per = (total_edits / total_phones * 100) if total_phones > 0 else 0.0
    seq_acc = (exact_match / len(references) * 100) if references else 0.0

    return {
        'per':               per,
        'sequence_accuracy': seq_acc,
        'total_edits':       total_edits,
        'total_phonemes':    total_phones,
        'num_samples':       len(references),
    }


# ============================================================================
# Detailed error analysis
# ============================================================================

def calculate_detailed_errors(
    predictions: List[List[str]],
    references:  List[List[str]],
) -> Dict:
    """Break down PER into substitutions / deletions / insertions."""
    from metrics import align_sequences, _get_ops

    subs = dels = ins = 0
    for pred, ref in zip(predictions, references):
        ref_a, hyp_a = align_sequences(ref, pred)
        ops  = _get_ops(ref_a, hyp_a)
        subs += ops.count('S')
        dels += ops.count('D')
        ins  += ops.count('I')

    total_phones = sum(len(r) for r in references)
    total_edits  = subs + dels + ins

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
    predictions:  List[List[str]],
    references:   List[List[str]],
    sample_ids:   List[str],
) -> List[Dict]:
    """
    Return one record per error containing context (surrounding phones).
    """
    from metrics import align_sequences, _get_ops

    records = []
    for pred, ref, sid in zip(predictions, references, sample_ids):
        ref_a, hyp_a = align_sequences(ref, pred)
        ops  = _get_ops(ref_a, hyp_a)

        ref_pos = hyp_pos = 0
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
        metrics     – dict with per, sequence_accuracy, detailed counts
        predictions – list[list[str]]
        references  – list[list[str]]
        canonicals  – list[list[str]] (empty if not L2-ARCTIC)
    """
    from transformers import Wav2Vec2Processor

    model_dir = Path(model_dir)

    # ── Load processor ───────────────────────────────────────────────────
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))

    # ── Load model ───────────────────────────────────────────────────────
    if MODEL_ARCHITECTURE == 'linguistic':
        from models import Wav2Vec2_Linguistic
        from transformers import Wav2Vec2Config
        config = Wav2Vec2Config.from_pretrained(str(model_dir))
        model = Wav2Vec2_Linguistic(config)
        model.load_state_dict(torch.load(model_dir / 'pytorch_model.bin', map_location='cpu'))
    else:
        from transformers import Wav2Vec2ForCTC
        model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    print(f"  ✓ Model loaded ({MODEL_ARCHITECTURE}) on {device}")

    # ── Inference ────────────────────────────────────────────────────────
    predictions_all: List[List[str]] = []
    references_all:  List[List[str]] = []
    canonicals_all:  List[List[str]] = []

    batch_size = 8

    for start in range(0, len(test_dataset), batch_size):
        batch = test_dataset[start: start + batch_size]
        input_values = torch.tensor(np.array(batch['input_values'])).to(device)

        with torch.no_grad():
            if MODEL_ARCHITECTURE == 'linguistic':
                canonical_ids = torch.tensor(batch['canonical_label_ids']).to(device)
                logits = model(input_values=input_values, canonical_ids=canonical_ids).logits
            else:
                logits = model(input_values=input_values).logits

        decoded = decode_ctc_predictions(logits.cpu(), processor)
        predictions_all.extend(decoded)

        for lab in batch['labels']:
            clean = [
                id2phoneme.get(l, '[UNK]') for l in lab
                if l not in (-100, vocab.get('[PAD]', 0))
                and id2phoneme.get(l, '[UNK]') not in ('[PAD]', '[UNK]', '[CTC]')
            ]
            references_all.append(clean)

        if 'canonical_labels' in test_dataset.column_names:
            for can in batch.get('canonical_labels', []):
                canonicals_all.append(can if isinstance(can, list) else [])
        elif not canonicals_all:
            canonicals_all.extend([] for _ in decoded)

        if (start // batch_size) % 10 == 0:
            print(f"  … {start + batch_size}/{len(test_dataset)}")

    # ── Optional phone collapse ───────────────────────────────────────────
    if map_to_standard_arpabet:
        predictions_all = [map_timit_to_standard_arpabet(p) for p in predictions_all]
        references_all  = [map_timit_to_standard_arpabet(r) for r in references_all]

    # ── PER ───────────────────────────────────────────────────────────────
    per_metrics   = calculate_per(predictions_all, references_all)
    error_details = calculate_detailed_errors(predictions_all, references_all)
    metrics = {**per_metrics, **error_details}

    return metrics, predictions_all, references_all, canonicals_all
