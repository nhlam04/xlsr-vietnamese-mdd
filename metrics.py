"""
MDD sequence alignment and evaluation metrics.

Public API:
    align_sequences(seq1, seq2) → (ref_aligned, hyp_aligned)
    _get_ops(ref_aligned, hyp_aligned) → list[str]
    compute_mdd_metrics(canonicals, references, predictions) → dict
    print_mdd_metrics(metrics, title) → None
"""

from typing import List, Tuple


# ============================================================================
# Needleman–Wunsch global sequence alignment
# ============================================================================

_GAP = '<eps>'
_MATCH    =  1
_MISMATCH = -1
_GAP_COST = -1


def align_sequences(
    seq1: List[str],
    seq2: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Global alignment of two phoneme sequences.

    Returns two equal-length lists with *_GAP* tokens at insertion/deletion
    positions.

    Args:
        seq1: reference sequence
        seq2: hypothesis sequence

    Returns:
        (ref_aligned, hyp_aligned)
    """
    n, m = len(seq1), len(seq2)

    # Fill DP matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * _GAP_COST
    for j in range(m + 1):
        dp[0][j] = j * _GAP_COST

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = _MATCH if seq1[i - 1] == seq2[j - 1] else _MISMATCH
            dp[i][j] = max(
                dp[i - 1][j - 1] + match_score,
                dp[i - 1][j]     + _GAP_COST,
                dp[i][j - 1]     + _GAP_COST,
            )

    # Traceback
    ref_aligned: List[str] = []
    hyp_aligned: List[str] = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_score = _MATCH if seq1[i - 1] == seq2[j - 1] else _MISMATCH
            if dp[i][j] == dp[i - 1][j - 1] + match_score:
                ref_aligned.append(seq1[i - 1])
                hyp_aligned.append(seq2[j - 1])
                i -= 1; j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + _GAP_COST:
            ref_aligned.append(seq1[i - 1])
            hyp_aligned.append(_GAP)
            i -= 1
        else:
            ref_aligned.append(_GAP)
            hyp_aligned.append(seq2[j - 1])
            j -= 1

    ref_aligned.reverse()
    hyp_aligned.reverse()
    return ref_aligned, hyp_aligned


def _get_ops(
    ref_aligned: List[str],
    hyp_aligned: List[str],
) -> List[str]:
    """
    Map aligned sequences to edit operation labels.

    Returns a list parallel to *ref_aligned* / *hyp_aligned*:
        'C'  — correct
        'S'  — substitution
        'D'  — deletion  (ref≠gap, hyp==gap)
        'I'  — insertion (ref==gap, hyp≠gap)
    """
    ops = []
    for r, h in zip(ref_aligned, hyp_aligned):
        if r == _GAP:
            ops.append('I')
        elif h == _GAP:
            ops.append('D')
        elif r == h:
            ops.append('C')
        else:
            ops.append('S')
    return ops


# ============================================================================
# 3-way MDD metrics
# ============================================================================

def compute_mdd_metrics(
    canonicals:  List[List[str]],
    references:  List[List[str]],
    predictions: List[List[str]],
) -> dict:
    """
    Compute three-way MDD evaluation metrics.

    Convention (per VLSP / LingWav2Vec2):
        canonical  — intended / canonical phoneme sequence
        reference  — human-perceived pronunciation (ground truth)
        prediction — model output

    A phoneme is mispronounced when canonical ≠ reference.
    Model detects the mispronunciation when prediction ≠ canonical.

    Counts:
        TA  — True Accept  : canon==ref,  pred==canon   (correct phoneme, model says OK)
        TR  — True Reject  : canon!=ref,  pred!=canon   (mispronounced,   model catches it)
        FA  — False Accept : canon!=ref,  pred==canon   (mispronounced,   model misses it)
        FR  — False Reject : canon==ref,  pred!=canon   (correct phoneme, model flags it)
    """
    TA = TR = FA = FR = 0

    for canonical, ref, pred in zip(canonicals, references, predictions):
        # Align canonical → ref   (detect errors)
        can_ref_r, can_ref_h = align_sequences(canonical, ref)
        can_ref_ops = _get_ops(can_ref_r, can_ref_h)

        # Align canonical → pred  (model output)
        can_pred_r, can_pred_h = align_sequences(canonical, pred)
        can_pred_ops = _get_ops(can_pred_r, can_pred_h)

        min_len = min(len(can_ref_ops), len(can_pred_ops))

        for i in range(min_len):
            is_error     = can_ref_ops[i]  != 'C'
            is_detected  = can_pred_ops[i] != 'C'

            if not is_error and not is_detected:
                TA += 1
            elif is_error and is_detected:
                TR += 1
            elif is_error and not is_detected:
                FA += 1
            else:  # not is_error and is_detected
                FR += 1

    total = TA + TR + FA + FR
    eps   = 1e-9

    recall    = TR / (TR + FA + eps)
    precision = TR / (TR + FR + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    far       = FA / (FA + TA + eps)   # False-Accept rate
    frr       = FR / (FR + TA + eps)   # False-Reject rate
    der       = (FA + FR) / (total + eps)
    det_acc   = (TA + TR) / (total + eps)

    return {
        'TA':                 TA,
        'TR':                 TR,
        'FA':                 FA,
        'FR':                 FR,
        'total':              total,
        'recall':             recall       * 100,
        'precision':          precision    * 100,
        'f1':                 f1           * 100,
        'false_accept_rate':  far          * 100,
        'false_reject_rate':  frr          * 100,
        'detection_error':    der          * 100,
        'detection_accuracy': det_acc      * 100,
    }


def print_mdd_metrics(m: dict, title: str = 'MDD Metrics') -> None:
    """Pretty-print the dict returned by :func:`compute_mdd_metrics`."""
    w = 55
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")
    print(f"  {'TA (True Accept)':<28} {m['TA']:>8}")
    print(f"  {'TR (True Reject)':<28} {m['TR']:>8}")
    print(f"  {'FA (False Accept)':<28} {m['FA']:>8}")
    print(f"  {'FR (False Reject)':<28} {m['FR']:>8}")
    print(f"  {'Total phones evaluated':<28} {m['total']:>8}")
    print(f"  {'-'*40}")
    print(f"  {'Recall':<28} {m['recall']:>7.2f}%")
    print(f"  {'Precision':<28} {m['precision']:>7.2f}%")
    print(f"  {'F1 Score':<28} {m['f1']:>7.2f}%")
    print(f"  {'-'*40}")
    print(f"  {'False Accept Rate (FAR)':<28} {m['false_accept_rate']:>7.2f}%")
    print(f"  {'False Reject Rate (FRR)':<28} {m['false_reject_rate']:>7.2f}%")
    print(f"  {'Detection Error Rate':<28} {m['detection_error']:>7.2f}%")
    print(f"  {'Detection Accuracy':<28} {m['detection_accuracy']:>7.2f}%")
    print(f"{'='*w}")
