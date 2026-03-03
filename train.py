"""
Training pipeline.

train_phoneme_classifier(train_dataset, val_dataset)  — main entry point.
find_latest_checkpoint(output_dir, input_dir)          — checkpoint discovery.
"""

import json
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import Trainer, TrainingArguments

from cache import AUDIO_CACHE, resolve_cache_path
from callbacks import DetailedLoggingCallback
from collator import DataCollatorCTCWithPadding
from config import (
    INPUT_PATH,
    OUTPUT_PATH,
    RESUME_FROM_FOLDER,
    TRAINING_CONFIG,
    WANDB_API_KEY,
)
from models import create_model_and_processor


# ============================================================================
# Checkpoint discovery
# ============================================================================

def find_latest_checkpoint(
    output_dir: Path,
    input_dir: Optional[Path] = None,
) -> Tuple[Optional[str], int]:
    """
    Scan *input_dir* (if given) then *output_dir* for 'checkpoint-NNN' dirs.
    Returns (path_str, epoch_number) of the latest valid checkpoint,
    or (None, 0) if none found.
    """
    candidate_dirs = []

    for search_dir in filter(None, [input_dir, output_dir]):
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
        try:
            for item in search_path.iterdir():
                if item.is_dir() and item.name.startswith('checkpoint-'):
                    candidate_dirs.append(item)
        except (PermissionError, OSError):
            pass

    valid = []
    for d in candidate_dirs:
        m = re.search(r'checkpoint-(\d+)', d.name)
        if m and ((d / 'pytorch_model.bin').exists() or (d / 'model.safetensors').exists()):
            valid.append((int(m.group(1)), d))

    if not valid:
        return None, 0

    _, latest = max(valid, key=lambda x: x[0])

    epoch = 0
    state_file = latest / 'trainer_state.json'
    if state_file.exists():
        try:
            with open(state_file) as f:
                epoch = int(json.load(f).get('epoch', 0))
        except Exception:
            pass

    print(f"  ✓ Found checkpoint: {latest.name} (epoch {epoch})")
    return str(latest), epoch


# ============================================================================
# Main training function
# ============================================================================

def train_phoneme_classifier(train_dataset, val_dataset):
    """
    Train the CTC model with auto-resume and best-model selection.

    Steps:
        0. Locate latest checkpoint (if any)
        1. Load audio cache into RAM
        2. Build model + processor
        3. Configure Trainer
        4. Launch training
        5. Save best model
    """
    global AUDIO_CACHE

    output_dir = OUTPUT_PATH / 'model_phoneme_ctc'
    input_dir  = INPUT_PATH  / 'model_phoneme_ctc' if RESUME_FROM_FOLDER is not None else None

    print('\n' + '='*80)
    print(' Training Wav2Vec2 CTC (auto-resume + best model)')
    print('='*80)

    # ── [0] Checkpoint discovery ──────────────────────────────────────────
    print('\n[0/5] Searching for existing checkpoints…')
    checkpoint_path, starting_epoch = find_latest_checkpoint(output_dir, input_dir)
    if checkpoint_path:
        print(f"  → Resuming from epoch {starting_epoch + 1}")
    else:
        print("  → Starting fresh")

    # ── [1] Audio cache ───────────────────────────────────────────────────
    cache_path = resolve_cache_path()
    if not AUDIO_CACHE and cache_path.exists():
        print(f'\n[1/5] Loading audio cache from {cache_path}…')
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        AUDIO_CACHE.update(torch.load(cache_path, map_location='cpu'))
        size_gb = sum(a.element_size() * a.nelement() for a in AUDIO_CACHE.values()) / 1e9
        print(f"  ✓ {len(AUDIO_CACHE)} samples loaded ({size_gb:.2f} GB)")
    else:
        print(f'\n[1/5] Audio cache already in RAM ({len(AUDIO_CACHE)} samples)')

    # ── [2] Model ─────────────────────────────────────────────────────────
    print('\n[2/5] Loading model…')
    model, processor = create_model_and_processor()

    # ── [3] Datasets / collator ───────────────────────────────────────────
    print(f'\n[3/5] Datasets: train={len(train_dataset)}, val={len(val_dataset)}')
    data_collator = DataCollatorCTCWithPadding(
        tokenizer=processor.tokenizer,
        audio_cache=AUDIO_CACHE,
    )

    # ── [4] WandB (optional) ──────────────────────────────────────────────
    wandb_enabled = False
    if WANDB_API_KEY:
        try:
            import wandb, os
            wandb.login(key=WANDB_API_KEY, relogin=True)
            os.environ['WANDB_DISABLED'] = 'false'
            wandb.init(
                project='phoneme-recognition',
                name='xlsr-300m-ctc',
                config={**TRAINING_CONFIG, 'model': 'facebook/wav2vec2-xls-r-300m'},
                resume='allow',
            )
            wandb.watch(model, log='all', log_freq=100)
            wandb_enabled = True
            print('[4/5] WandB enabled')
        except Exception as e:
            print(f'[4/5] WandB skipped: {e}')
    else:
        import os
        os.environ['WANDB_DISABLED'] = 'true'
        print('[4/5] WandB disabled (no API key)')

    # ── [5] Trainer ───────────────────────────────────────────────────────
    training_config = {
        **TRAINING_CONFIG,
        'output_dir':               str(output_dir),
        'report_to':                ['wandb'] if wandb_enabled else [],
        'remove_unused_columns':    False,
        'save_strategy':            'epoch',
        'eval_strategy':            'epoch',
        'save_total_limit':         1,
        'load_best_model_at_end':   True,
        'metric_for_best_model':    'eval_loss',
        'greater_is_better':        False,
    }
    training_args = TrainingArguments(**training_config)

    callback = DetailedLoggingCallback(log_every_n_steps=50, wandb_enabled=wandb_enabled)
    trainer  = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=[callback],
    )

    print(f'\n{"="*80}\nSTARTING TRAINING\n{"="*80}')

    t0 = time.time()
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint_path)
    except Exception as e:
        print(f'\n⚠  Training error: {e}')
        if checkpoint_path and 'checkpoint' in str(e).lower():
            print('  Retrying without checkpoint…')
            result = trainer.train()
        else:
            raise

    elapsed = time.time() - t0
    print(f'\n✓ Best model loaded (lowest validation loss)')
    trainer.save_model()
    processor.save_pretrained(output_dir)

    if wandb_enabled:
        import wandb
        wandb.finish()

    print(f'\n{"="*80}')
    print(f'TRAINING COMPLETE — loss: {result.training_loss:.4f} | time: {elapsed/3600:.1f}h')
    print(f'Model saved to: {output_dir}')
    print('='*80)

    return output_dir
