"""
Single-file inference helper.

predict_phonemes(audio_path, model_dir) → str
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def predict_phonemes(
    audio_path: str,
    model_dir:  Optional[str] = None,
    language:   Optional[str] = None,
) -> str:
    """
    Predict phoneme sequence for a single audio file.

    Args:
        audio_path: Path to a WAV (or any librosa-readable) audio file.
        model_dir:  Directory containing the saved model + processor.
                    Defaults to OUTPUT_PATH / 'model_phoneme_ctc'.
        language:   Optional tag, purely informational.

    Returns:
        Space-separated string of predicted phonemes.
    """
    from transformers import Wav2Vec2Processor

    from config import MODEL_ARCHITECTURE, OUTPUT_PATH
    from evaluate import decode_ctc_predictions

    # ── Resolve model directory ───────────────────────────────────────────
    if model_dir is None:
        model_dir = OUTPUT_PATH / 'model_phoneme_ctc'
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # ── Load audio ────────────────────────────────────────────────────────
    try:
        import librosa
        audio, sr = librosa.load(str(audio_path), sr=16_000, mono=True)
    except Exception as e:
        raise RuntimeError(f"Could not load audio file '{audio_path}': {e}")

    # ── Load processor ────────────────────────────────────────────────────
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    inputs    = processor(audio, sampling_rate=16_000, return_tensors='pt', padding=True)

    # ── Load model ────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if MODEL_ARCHITECTURE == 'linguistic':
        from models import Wav2Vec2_Linguistic
        from transformers import Wav2Vec2Config
        config = Wav2Vec2Config.from_pretrained(str(model_dir))
        model  = Wav2Vec2_Linguistic(config)
        model.load_state_dict(
            torch.load(model_dir / 'pytorch_model.bin', map_location='cpu')
        )
    else:
        from transformers import Wav2Vec2ForCTC
        model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))

    model.to(device).eval()

    # ── Inference ─────────────────────────────────────────────────────────
    with torch.no_grad():
        input_values = inputs['input_values'].to(device)
        logits       = model(input_values=input_values).logits

    phonemes_batch = decode_ctc_predictions(logits.cpu(), processor)
    return phonemes_batch[0] if phonemes_batch else ''
