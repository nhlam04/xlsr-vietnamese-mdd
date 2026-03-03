"""
main.py — End-to-end pipeline entry point.

Usage:
    python main.py [--train] [--eval] [--no-cache]
"""

import argparse
import gc
from pathlib import Path

import torch


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='XLS-R CTC Phoneme Classifier')
    p.add_argument('--train',    action='store_true', help='Run training')
    p.add_argument('--eval',     action='store_true', help='Run evaluation after training')
    p.add_argument('--no-cache', action='store_true', help='Skip audio caching (load on-the-fly)')
    p.add_argument('--model-dir', default=None,       help='Override model directory for evaluation')
    return p.parse_args()


# ── Step helpers ──────────────────────────────────────────────────────────────

def build_hf_dataset(data_list, vocab_mapping):
    """
    Convert a list of sample dicts into a HuggingFace Dataset.

    Each sample dict must have at minimum:
        'input_values' : np.ndarray  (16 kHz float32 audio)
        'labels'       : list[int]   (phoneme token ids, already filtered for silence)
    """
    from datasets import Dataset
    return Dataset.from_list(data_list)


def print_split_summary(name, dataset):
    if dataset is None or len(dataset) == 0:
        print(f"  {name}: (empty)")
        return
    durations = [len(s.get('input_values', [])) / 16_000 for s in dataset]
    print(f"  {name}: {len(dataset)} samples | {sum(durations)/3600:.2f} hours")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # If neither flag is given, run both
    if not args.train and not args.eval:
        args.train = True
        args.eval  = True

    from config import (
        L2_ARCTIC_CONFIG,
        LSVSC_CONFIG,
        MODEL_ARCHITECTURE,
        OUTPUT_PATH,
        TIMIT_CONFIG,
    )
    from cache import precompute_audio_to_cache, resolve_cache_path
    from datasets_loader import load_all_datasets
    from evaluate import (
        collect_all_errors_with_context,
        evaluate_classifier,
        write_detailed_errors_to_file,
    )
    from metrics import compute_mdd_metrics, print_mdd_metrics
    from vocab import build_vocab, save_vocab

    # ── [1] Load raw data ─────────────────────────────────────────────────
    print('\n' + '='*70)
    print(' [1/6] Loading datasets')
    print('='*70)

    splits = load_all_datasets()
    # load_all_datasets returns a tuple: (train_data, val_data, test_data, ...)
    train_list, val_list, test_list = splits[0], splits[1], splits[2]

    # Assign a meaningful key so downstream logic knows which split this is
    # (controls ARPAbet mapping and MDD metric computation)
    from config import L2_ARCTIC_CONFIG as _L2CFG
    if _L2CFG['use_for_testing']:
        _test_key = 'l2arctic'
    elif test_list:
        _test_key = 'timit'
    else:
        _test_key = 'test'
    test_lists = {_test_key: test_list}

    print('\n  Data splits:')
    print_split_summary('train', train_list)
    print_split_summary('val',   val_list)
    for k, v in test_lists.items():
        print_split_summary(f'test/{k}', v)

    # ── [2] Build vocab ───────────────────────────────────────────────────
    print('\n' + '='*70)
    print(' [2/6] Building vocabulary')
    print('='*70)
    vocab = build_vocab()
    save_vocab()
    print(f'  Vocab size: {len(vocab)} tokens')

    # ── [3] Prepare HuggingFace datasets ──────────────────────────────────
    print('\n' + '='*70)
    print(' [3/6] Preparing HuggingFace datasets')
    print('='*70)
    from datasets_loader import prepare_dataset_with_labels

    train_hf = prepare_dataset_with_labels(train_list)
    val_hf   = prepare_dataset_with_labels(val_list)

    test_hf_dict = {}
    for split_name, split_data in test_lists.items():
        if split_data:
            test_hf_dict[split_name] = prepare_dataset_with_labels(split_data)

    print(f'  train: {len(train_hf)} | val: {len(val_hf)}')

    # ── [4] Audio caching ─────────────────────────────────────────────────
    if not args.no_cache:
        print('\n' + '='*70)
        print(' [4/6] Audio caching')
        print('='*70)
        cache_path = resolve_cache_path()
        if not cache_path.exists():
            all_samples = train_list + val_list
            for v in test_lists.values():
                all_samples += v
            precompute_audio_to_cache(all_samples, cache_path)
        else:
            print(f'  ✓ Cache already exists: {cache_path}')
    else:
        print('\n[4/6] Skipping audio cache (--no-cache)')

    # ── [5] Training ──────────────────────────────────────────────────────
    model_dir = args.model_dir or str(OUTPUT_PATH / 'model_phoneme_ctc')

    if args.train:
        print('\n' + '='*70)
        print(' [5/6] Training')
        print('='*70)
        from train import train_phoneme_classifier
        model_dir = str(train_phoneme_classifier(train_hf, val_hf))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── [6] Evaluation ────────────────────────────────────────────────────
    if args.eval:
        print('\n' + '='*70)
        print(' [6/6] Evaluation')
        print('='*70)

        for split_name, test_hf in test_hf_dict.items():
            print(f'\n  — {split_name.upper()} ({len(test_hf)} samples)')
            use_collapsed = split_name in ('timit', 'l2arctic')
            metrics, preds, refs, cans = evaluate_classifier(
                model_dir, test_hf,
                map_to_standard_arpabet=use_collapsed,
            )
            print(f"    PER:             {metrics['per']:.2f}%")
            print(f"    Seq. accuracy:   {metrics['sequence_accuracy']:.2f}%")
            print(f"    Substitutions:   {metrics.get('substitutions', 'N/A')}")
            print(f"    Deletions:       {metrics.get('deletions', 'N/A')}")
            print(f"    Insertions:      {metrics.get('insertions', 'N/A')}")

            # MDD metrics for L2-ARCTIC (has canonical labels)
            if split_name == 'l2arctic' and cans and any(cans):
                mdd = compute_mdd_metrics(cans, refs, preds)
                print_mdd_metrics(mdd, title=f'MDD Metrics — {split_name}')

            # Write error CSV
            sample_ids = test_hf.get('sample_id', list(range(len(preds))))
            errors = collect_all_errors_with_context(preds, refs, sample_ids)
            out_csv = OUTPUT_PATH / f'errors_{split_name}.csv'
            write_detailed_errors_to_file(errors, out_csv)

    print('\n✓ Done.\n')


if __name__ == '__main__':
    main()
