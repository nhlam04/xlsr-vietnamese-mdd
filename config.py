"""
Configuration: paths, model selection, dataset flags, and training hyperparameters.
Edit this file to change training behaviour without touching any other module.
"""

import os
from pathlib import Path

# ============================================================================
# RESUME TRAINING CONFIGURATION
# ============================================================================
# Point to your existing outputs folder to reuse checkpoints, vocab, and
# audio-cache (saves ~10-20 minutes on resumed Kaggle runs).
# Example:  Path('/kaggle/input/my-training-run/articulatory_models')
# Leave as None for first-time training.
RESUME_FROM_FOLDER = None  # type: Path | None

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================
IS_KAGGLE = os.path.exists('/kaggle')

# ── Output / Input paths ─────────────────────────────────────────────────────
if RESUME_FROM_FOLDER is not None:
    INPUT_PATH = Path(RESUME_FROM_FOLDER)
    OUTPUT_PATH = Path('/kaggle/working/articulatory_models') if IS_KAGGLE else Path('outputs')
else:
    OUTPUT_PATH = Path('/kaggle/working/articulatory_models') if IS_KAGGLE else Path('outputs')
    INPUT_PATH = OUTPUT_PATH

OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# ── Dataset paths ─────────────────────────────────────────────────────────────
if IS_KAGGLE:
    TIMIT_PATH = Path('/kaggle/input/darpa-timit-acousticphonetic-continuous-speech')
    LSVSC_PATH = Path('/kaggle/input/lsvsc-data/LSVSC')
    L2_ARCTIC_PATH = Path('/kaggle/input/l2-arctic-data')
    VIETNAMESE_DICT_PATH = Path('/kaggle/input/vietnamese-g2p/syl_to_phoneme.dict')
else:
    TIMIT_PATH = Path('data/TIMIT')
    LSVSC_PATH = Path('data/LSVSC')
    L2_ARCTIC_PATH = Path('data/l2arctic_release_v5.0')
    VIETNAMESE_DICT_PATH = Path('VietMDD/syl_to_phoneme.dict')
    if not VIETNAMESE_DICT_PATH.exists():
        VIETNAMESE_DICT_PATH = Path('../VietMDD/syl_to_phoneme.dict')

# ============================================================================
# DATASET SELECTION CONFIGURATION
# ============================================================================
TIMIT_CONFIG = {
    'use_for_training': True,
    'use_for_validation': True,
    'use_for_testing': True,
}

LSVSC_CONFIG = {
    'use_for_training': True,
    'use_for_validation': True,
    'use_for_testing': True,
    'sample_ratio': 0.1,   # 0.1 = 10 %, 1.0 = 100 %
}

L2_ARCTIC_CONFIG = {
    'use_for_training': True,
    'use_for_validation': True,
    'use_for_testing': True,
    # Options: 'vietnamese_only', 'all_speakers', 'non_viet_train_viet_test'
    'mode': 'non_viet_train_viet_test',
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# 'standard'  – Wav2Vec2ForCTC with linear CTC head
# 'linguistic' – Wav2Vec2_Linguistic (BiLSTM cross-attention); requires CPL labels
MODEL_ARCHITECTURE = 'linguistic'   # 'standard' or 'linguistic'

# When linguistic model is selected, override datasets to L2-ARCTIC only
if MODEL_ARCHITECTURE == 'linguistic':
    TIMIT_CONFIG.update({'use_for_training': False, 'use_for_validation': False, 'use_for_testing': False})
    LSVSC_CONFIG.update({'use_for_training': False, 'use_for_validation': False, 'use_for_testing': False})
    L2_ARCTIC_CONFIG.update({'use_for_training': True, 'use_for_validation': True, 'use_for_testing': True})

# Pretrained backbone (same for both architectures)
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
TRAINING_CONFIG = {
    'learning_rate': 2e-5,
    'weight_decay': 0.005,
    'num_train_epochs': 100,

    # VRAM optimisation
    'per_device_train_batch_size': 1,
    'per_device_eval_batch_size': 1,
    'gradient_accumulation_steps': 1,
    'gradient_checkpointing': False,

    # Speed optimisation
    'fp16': True,
    'dataloader_num_workers': 4,
    'dataloader_pin_memory': True,
    'ddp_find_unused_parameters': True,
    'optim': 'adamw_8bit',

    # Warmup & evaluation
    'warmup_ratio': 0.1,
    'eval_strategy': 'epoch',
    'save_strategy': 'epoch',
    'logging_steps': 50,
    'logging_first_step': True,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'greater_is_better': False,
    'save_total_limit': 1,
}

# ============================================================================
# WANDB (optional)
# ============================================================================
WANDB_API_KEY = ""   # Set to your key from https://wandb.ai/authorize
