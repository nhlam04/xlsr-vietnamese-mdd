"""
Model architectures:
  - LinguisticEncoder   – 4-layer BiLSTM that encodes canonical phonemes
  - Wav2Vec2_Linguistic  – XLS-R + LinguisticEncoder + cross-attention CTC head
  - create_model_and_processor() – factory used by both train and evaluate

Source: Refining-Linguistic-Information-Utilization-MDD/MDD_model.py
Note: hardcoded 768 replaced with config.hidden_size to support XLS-R 300M (1024).
"""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)
from transformers.modeling_outputs import CausalLMOutput

from config import MODEL_ARCHITECTURE, MODEL_NAME, INPUT_PATH, OUTPUT_PATH, RESUME_FROM_FOLDER
from vocab import vocab


# ============================================================================
# Linguistic encoder
# ============================================================================

class LinguisticEncoder(nn.Module):
    """
    BiLSTM-based encoder: embeds canonical phoneme sequences into (Hk, Hv) tensors
    used as key / value in cross-attention with acoustic features.
    """

    def __init__(self, num_features_out: int = 1024, vocab_size: int = 68):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, 64, padding_idx=vocab_size)
        self.bi_lstm = nn.LSTM(
            input_size=64,
            hidden_size=num_features_out // 2,
            bidirectional=True,
            batch_first=True,
            num_layers=4,
        )
        self.linear = nn.Linear(num_features_out, num_features_out)

    def forward(self, x):
        x = self.embedding(x)                       # batch × length × 64
        out, _ = self.bi_lstm(x)                    # batch × length × hidden_size
        Hk = self.linear(out)                       # projected keys
        Hv = out                                    # values
        return Hk, Hv


# ============================================================================
# Wav2Vec2_Linguistic
# ============================================================================

class Wav2Vec2_Linguistic(transformers.Wav2Vec2PreTrainedModel):
    """
    Wav2Vec2 + LinguisticEncoder CTC model.

    Architecture:
        audio  → Wav2Vec2 CNN+Transformer  → acoustic  (b × t × H)
        CPL    → LinguisticEncoder         → Hk, Hv
        cross-attention(acoustic, Hk, Hv)  → attended  (b × t × H)
        concat([acoustic, attended])        → Linear(2H → vocab_size) → logits
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        H = config.hidden_size  # 1024 for XLS-R 300M
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier_vocab = nn.Linear(H * 2, config.vocab_size)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=H, num_heads=16, dropout=0.2, batch_first=True
        )
        self.linguistic_encoder = LinguisticEncoder(
            num_features_out=H, vocab_size=config.vocab_size
        )
        self.post_init()

    def freeze_feature_encoder(self):
        """Freeze CNN feature extractor; only Transformer + linguistic encoder train."""
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        canonical_labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        acoustic = outputs[0]

        Hk, Hv = self.linguistic_encoder(canonical_labels)
        attended, _ = self.multihead_attention(acoustic, Hk, Hv)
        combined = torch.cat([acoustic, attended], dim=2)
        logits = self.classifier_vocab(combined)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label value >= vocab_size ({self.config.vocab_size})")

            attn = (attention_mask if attention_mask is not None
                    else torch.ones_like(input_values, dtype=torch.long))
            input_lengths  = self._get_feat_extract_output_lengths(attn.sum(-1)).to(torch.long)
            labels_mask    = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flat_targets   = labels.masked_select(labels_mask)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs, flat_targets, input_lengths, target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ============================================================================
# Factory
# ============================================================================

def create_model_and_processor():
    """
    Build processor (tokenizer + feature extractor) and model.

    Both 'standard' and 'linguistic' use the same pretrained weights
    (facebook/wav2vec2-xls-r-300m); only the head wrapper differs.
    """
    # ── Vocab file for tokenizer ──────────────────────────────────────────
    input_vocab = INPUT_PATH / 'model_phoneme_ctc' / 'vocabs' / 'vocab_arpabet_xsampa.json'
    if RESUME_FROM_FOLDER is not None and input_vocab.exists():
        vocab_file = str(input_vocab)
    else:
        tmp_dir = OUTPUT_PATH / 'temp_vocab'
        tmp_dir.mkdir(exist_ok=True, parents=True)
        vocab_file_path = tmp_dir / 'vocab.json'
        with open(vocab_file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)
        vocab_file = str(vocab_file_path)

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_file,
        unk_token='[UNK]',
        pad_token='[PAD]',
        word_delimiter_token='|',
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────
    common_kwargs = dict(
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        layerdrop=0.0,
        ctc_loss_reduction='mean',
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(vocab),
        ctc_zero_infinity=True,
    )

    if MODEL_ARCHITECTURE == 'linguistic':
        config = Wav2Vec2Config.from_pretrained(MODEL_NAME, final_dropout=0.1, **common_kwargs)
        model = Wav2Vec2_Linguistic(config)
        # Load pretrained wav2vec2 weights into the backbone
        pretrained = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        model.wav2vec2.load_state_dict(pretrained.state_dict())
        del pretrained
        print("  Architecture: Wav2Vec2_Linguistic (BiLSTM cross-attention)")
    else:
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, **common_kwargs)
        print("  Architecture: Wav2Vec2ForCTC (standard linear CTC head)")

    model.config.use_cache = False
    total      = sum(p.numel() for p in model.parameters()) / 1e6
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Parameters: {total:.1f}M total, {trainable:.1f}M trainable")
    print(f"  Vocab size: {len(vocab)}")

    return model, processor
