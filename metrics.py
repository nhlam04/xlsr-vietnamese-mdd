"""
MDD sequence alignment and evaluation metrics.

Public API:
    align_sequences(seq1, seq2) → (ref_aligned, hyp_aligned)
    _get_ops(ref_aligned, hyp_aligned) → list[str]
    compute_mdd_metrics(canonicals, references, predictions) → dict
        (all three args are List[str] — space-separated phoneme strings)
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
    canonicals:  List[str],
    references:  List[str],
    predictions: List[str],
) -> dict:
    """
    3-way MDD evaluation adopted from metric.ipynb (Kaiqi Fu et al.).

    Three pairwise alignments per utterance:
        A: canonical (CPL) ↔ reference/PPL    → ref_a,    human_a,  op_A
        B: reference/PPL   ↔ prediction (our) → human2_a, our2_a,   op_B
        C: canonical (CPL) ↔ prediction (our) → ref3_a,   our3_a,   op_C

    Deletion detection  — syncs canonical tokens across A and C.
    Cor/Sub/Ins detection — syncs human tokens across A and B.

    Quantities (following paper notation):
        TA  = True  Acceptance  (cor_cor)   : CPL correct, model correct
        TR  = True  Rejection   (detected errors)
        FA  = False Acceptance  (missed errors)
        FR  = False Rejection   (cor_nocor)  : CPL correct, model error

    Metrics:
        Recall    = TR / (TR + FA)
        Precision = TR / (TR + FR)
        F1, FAR, FRR, DER, Detection Accuracy
        Correct / Error Diagnosis rates

    Args:
        canonicals:  List of canonical phoneme sequences (space-separated strings)
        references:  List of reference phoneme sequences (space-separated strings)
        predictions: List of predicted phoneme sequences (space-separated strings)
    """
    cor_cor = cor_nocor = 0
    sub_sub = sub_sub1 = sub_nosub = 0
    ins_ins = ins_ins1 = ins_noins = 0
    del_del = del_del1 = del_nodel = 0

    for canon_str, ref_str, pred_str in zip(canonicals, references, predictions):
        canon = canon_str.split()
        ref   = ref_str.split()
        pred  = pred_str.split()

        # Three alignments
        ref_a,    human_a  = align_sequences(canon, ref)   # A: CPL ↔ PPL
        human2_a, our2_a   = align_sequences(ref,   pred)  # B: PPL ↔ model
        ref3_a,   our3_a   = align_sequences(canon, pred)  # C: CPL ↔ model

        op_A = _get_ops(ref_a,    human_a)
        op_B = _get_ops(human2_a, our2_a)
        op_C = _get_ops(ref3_a,   our3_a)

        # ── Deletion detection (walk CPL tokens through A and C) ──
        flag = 0
        for i in range(len(ref_a)):
            if ref_a[i] == "<eps>":
                continue
            while flag < len(ref3_a) and ref3_a[flag] == "<eps>":
                flag += 1
            if flag < len(ref3_a) and ref_a[i] == ref3_a[flag]:
                if op_A[i] == "D":
                    if op_C[flag] == "D":
                        del_del  += 1          # human deleted AND model deleted  → TR
                    elif op_C[flag] != "C":
                        del_del1 += 1          # human deleted, model wrong sub   → TR (wrong diag)
                    else:
                        del_nodel += 1         # human deleted, model correct     → FA
                flag += 1

        # ── Correct / Substitution / Insertion detection (sync PPL tokens A↔B) ──
        flag = 0
        for i in range(len(human_a)):
            if human_a[i] == "<eps>":
                continue
            while flag < len(human2_a) and human2_a[flag] == "<eps>":
                flag += 1
            if flag < len(human2_a) and human_a[i] == human2_a[flag]:
                # Correct phoneme (no error in human)
                if op_A[i] == "C":
                    if op_B[flag] == "C":
                        cor_cor  += 1          # model also correct  → TA
                    else:
                        cor_nocor += 1         # model wrong         → FR

                # Substitution (human produced wrong phoneme)
                if op_A[i] == "S":
                    if op_B[flag] == "C":
                        sub_sub  += 1          # model predicts correct phoneme → TR correct diag
                    elif ref_a[i] != our2_a[flag]:
                        sub_sub1 += 1          # model predicts different sub   → TR wrong diag
                    else:
                        sub_nosub += 1         # model agrees with human error  → FA

                # Insertion by human (no CPL reference)
                if op_A[i] == "I":
                    if op_B[flag] == "C":
                        ins_ins  += 1          # model drops it (correct)  → TR correct diag
                    elif op_B[flag] != "D":
                        ins_ins1 += 1          # model maps to wrong phone → TR wrong diag
                    else:
                        ins_noins += 1         # model also inserts        → FA

                flag += 1

    # ── Aggregate counters ──
    TR = sub_sub + sub_sub1 + del_del + del_del1 + ins_ins + ins_ins1
    FR = cor_nocor
    FA = sub_nosub + ins_noins + del_nodel
    TA = cor_cor

    total        = TA + TR + FA + FR
    err_count    = sub_sub + sub_sub1 + sub_nosub + ins_ins + ins_ins1 + ins_noins + del_del + del_del1 + del_nodel
    Correct_Diag = sub_sub + ins_ins + del_del
    Error_Diag   = sub_sub1 + ins_ins1 + del_del1
    false_accept = sub_nosub + ins_noins + del_nodel

    recall    = TR / (TR + FA)              if (TR + FA) > 0              else 0.0
    precision = TR / (TR + FR)              if (TR + FR) > 0              else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    FAR       = 1 - recall
    FRR       = FR / (TA + FR)              if (TA + FR) > 0              else 0.0
    DER       = Error_Diag / (Correct_Diag + Error_Diag) if (Correct_Diag + Error_Diag) > 0 else 0.0
    det_acc   = (TA + TR) / total           if total > 0                  else 0.0

    return {
        # Raw counters
        'TA': TA, 'TR': TR, 'FA': FA, 'FR': FR,
        'cor_cor': cor_cor, 'cor_nocor': cor_nocor,
        'sub_sub': sub_sub, 'sub_sub1': sub_sub1, 'sub_nosub': sub_nosub,
        'ins_ins': ins_ins, 'ins_ins1': ins_ins1, 'ins_noins': ins_noins,
        'del_del': del_del, 'del_del1': del_del1, 'del_nodel': del_nodel,
        'err_count': err_count,
        'false_accept': false_accept,
        'correct_diag': Correct_Diag,
        'error_diag': Error_Diag,
        # Rates (0-1)
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'FAR': FAR,
        'FRR': FRR,
        'DER': DER,
        'detection_accuracy': det_acc,
        'true_acceptance_rate': TA / (TA + FR) if (TA + FR) > 0 else 0.0,
        'false_acceptance_rate_err': false_accept / err_count if err_count > 0 else 0.0,
        'correct_diagnosis_rate': Correct_Diag / (Correct_Diag + Error_Diag) if (Correct_Diag + Error_Diag) > 0 else 0.0,
    }


def print_mdd_metrics(m: dict, title: str = 'MDD Metrics') -> None:
    """Pretty-print the dict returned by :func:`compute_mdd_metrics`."""
    print(f"\n{'─'*60}")
    print(f"{title}")
    print(f"{'─'*60}")
    print(f"  Recall    (True Rejection Rate) : {m['recall']*100:.2f}%")
    print(f"  Precision                       : {m['precision']*100:.2f}%")
    print(f"  F1-Score                        : {m['f1']*100:.2f}%")
    print(f"  Detection Accuracy (TA+TR/total): {m['detection_accuracy']*100:.2f}%")
    print(f"  False Acceptance Rate (FAR)     : {m['FAR']*100:.2f}%")
    print(f"  False Rejection Rate (FRR)      : {m['FRR']*100:.2f}%")
    print(f"  Diagnosis Error Rate (DER)      : {m['DER']*100:.2f}%")
    print(f"  True Acceptance Rate            : {m['true_acceptance_rate']*100:.2f}%")
    print(f"  False Acceptance Rate (errors)  : {m['false_acceptance_rate_err']*100:.2f}%")
    print(f"  Correct Diagnosis Rate          : {m['correct_diagnosis_rate']*100:.2f}%")
    print(f"\n  Counts — TA:{m['TA']}  TR:{m['TR']}  FA:{m['FA']}  FR:{m['FR']}")
    print(f"  Error breakdown:")
    print(f"    Sub  → detected correctly (sub_sub) : {m['sub_sub']}")
    print(f"    Sub  → detected as other  (sub_sub1): {m['sub_sub1']}")
    print(f"    Sub  → missed             (sub_nosub): {m['sub_nosub']}")
    print(f"    Del  → detected correctly (del_del) : {m['del_del']}")
    print(f"    Del  → detected as other  (del_del1): {m['del_del1']}")
    print(f"    Del  → missed             (del_nodel): {m['del_nodel']}")
    print(f"    Ins  → detected correctly (ins_ins) : {m['ins_ins']}")
    print(f"    Ins  → detected as other  (ins_ins1): {m['ins_ins1']}")
    print(f"    Ins  → missed             (ins_noins): {m['ins_noins']}")
