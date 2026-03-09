"""
Phoneme definitions, conflict-resolution mappings, Vietnamese dictionary loader,
and helper functions for ARPAbet / X-SAMPA label conversion.
"""

import unicodedata
from pathlib import Path
from typing import Dict, List

from config import VIETNAMESE_DICT_PATH

# ============================================================================
# ARPAbet phoneme set (TIMIT original)
# ============================================================================
TIMIT_ARPABET_ORIGINAL = {
    # Vowels
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 'ax-h', 'ay',
    'eh', 'el', 'em', 'en', 'eng', 'er', 'ey',
    'ih', 'ix', 'iy',
    'ow', 'oy',
    'uh', 'uw', 'ux',
    # Plosives
    'b', 'p', 't', 'd', 'k', 'g', 'q',
    # Closures
    'bcl', 'dcl', 'gcl', 'kcl', 'pcl', 'tcl',
    # Affricates
    'ch', 'jh',
    # Fricatives
    'f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh', 'hh', 'hv',
    # Nasals (incl. syllabic)
    'm', 'n', 'ng', 'nx',
    # Approximants (incl. syllabic)
    'l', 'r', 'w', 'y',
    # Flap / other
    'dx', 'wh',
}

# ── TIMIT normalisation map (applied during dataset loading) ─────────────────
TIMIT_PHONEME_MAP: Dict[str, str] = {
    'ax-h': 'ax',       # Reduced vowel variant → standard reduced vowel
    'bcl':  'b',        # /b/ closure
    'dcl':  'd',        # /d/ closure
    'eng':  'ax ng',    # Syllabic /ŋ/ → splits into two labels
    'gcl':  'g',        # /g/ closure
    'hv':   'hh',       # Voiced /h/
}

# ── Vietnamese X-SAMPA original symbols from syl_to_phoneme.dict ────────────
VIETNAMESE_XSAMPA_ORIGINAL = {
    '7', '7_X', 'a', 'a_X', 'b', 'd', 'dZ', 'e', 'E', 'E_X',
    'f', 'G', 'h', 'i', 'ie', 'j', 'J', 'k', 'kcl', 'kp', 'l',
    'm', 'M', 'M7', 'n', 'N', 'Nm', 'o', 'O', 'O_X', 'p', 'pcl',
    'r', 's', 'S', 't', 'tcl', 't_h', 'tS', 'ts_', 'u', 'uo', 'v',
    'w', 'wp', 'x', 'z',
}

# Conflict-resolution mapping for Vietnamese X-SAMPA symbols that overlap with
# ARPAbet labels but represent different phonemes.
VI_PHONEME_PREFIX_MAP: Dict[str, str] = {
    'b':   'b*',    # Vietnamese /ɓ/ implosive  vs ARPAbet /b/
    'd':   'd*',    # Vietnamese /ɗ/ implosive  vs ARPAbet /d/
    'h':   'hh',
    'N':   'ng',
    'a':   'aa',
    'E':   'eh',
    'i':   'ih',
    'O':   'ao',
    'u':   'uw',
    'j':   'y',
    'r':   'z',
    'ts_': 'tS',
    'S':   's',
    'dZ':  'z',
    'G':   'g*',
    'e':   'eh*',
    'J':   'n*',
    'Nm':  'n*',
    'o':   'uh*',
    'M':   'uw*',
    'a_X': 'aa*',
    'E_X': 'aa*',
    '7':   'ah*',
    '7_X': 'ah*',
    'O_X': 'ao*',
    'ie':  'ih ax',  # Vietnamese diphthong /iə/ → two labels
    'kp':  'k*',    #TODO overlapping with x
    'M7':  'uw* ax',  # Vietnamese diphthong /ɯə/ → two labels
    't_h': 't*',
    'tS':  't sh',  #TODO this is the canonical only, ppl needed
    'uo':  'uw ax',
    'wp':  'w*',
    'x':   'k*',
}

# Silence / non-speech labels that should be filtered from all datasets
SILENCE_PHONEMES = {'pau', 'h#', 'epi', 'sil'}

# Closures-to-stop mapping used when evaluating against L2-ARCTIC
# (L2-ARCTIC ARPAbet does not include closure notation)
CLOSURES_TO_ARPABET: Dict[str, str] = {
    'tcl': 't',
    'kcl': 'k',
    'pcl': 'p',
}


# ============================================================================
# Label helpers
# ============================================================================

def get_timit_phoneme_label(arpabet_symbol: str) -> List[str]:
    """
    Apply TIMIT_PHONEME_MAP to a single ARPAbet symbol.
    Returns a list because 'eng' expands to ['ax', 'ng'].
    """
    symbol = arpabet_symbol.strip().lower()
    if symbol in TIMIT_PHONEME_MAP:
        mapped = TIMIT_PHONEME_MAP[symbol]
        return mapped.split() if ' ' in mapped else [mapped]
    return [symbol]


def get_vietnamese_phoneme_label(xsampa_symbol: str) -> str:
    """
    Map a Vietnamese X-SAMPA symbol to its unique vocabulary label.
    Preserves case (X-SAMPA is case-sensitive).
    """
    symbol = xsampa_symbol.strip()
    return VI_PHONEME_PREFIX_MAP.get(symbol, symbol)


# ============================================================================
# Compute final phoneme sets (after applying maps)
# ============================================================================

def _build_timit_phonemes() -> frozenset:
    result = set()
    for sym in TIMIT_ARPABET_ORIGINAL:
        result.update(get_timit_phoneme_label(sym))
    return frozenset(result)


def _build_vietnamese_phonemes() -> frozenset:
    result = set()
    for sym in VIETNAMESE_XSAMPA_ORIGINAL:
        result.add(get_vietnamese_phoneme_label(sym))
    return frozenset(result)


TIMIT_ARPABET_PHONEMES: frozenset = _build_timit_phonemes()
VIETNAMESE_PHONEMES: frozenset = _build_vietnamese_phonemes()


# ============================================================================
# Vietnamese dictionary loader
# ============================================================================

def load_vietnamese_dict(dict_path: Path) -> Dict[str, List[str]]:
    """
    Load Vietnamese syllable → X-SAMPA phoneme dictionary.

    File format per line:  syllable|phoneme1 phoneme2 ... _tone
    Final p/t/k are converted to pcl/tcl/kcl (unreleased stops).

    Returns:
        {normalized_syllable: [phoneme, ...]}
    """
    syl_to_phonemes: Dict[str, List[str]] = {}

    if not dict_path.exists():
        print(f"Warning: Vietnamese dictionary not found at {dict_path}")
        return syl_to_phonemes

    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue

            syllable, phonemes_str = line.split('|', 1)
            phoneme_list = [p for p in phonemes_str.split() if not p.startswith('_')]

            # Convert final unreleased stops
            if phoneme_list:
                final = phoneme_list[-1]
                if final == 'p':
                    phoneme_list[-1] = 'pcl'
                elif final == 't':
                    phoneme_list[-1] = 'tcl'
                elif final == 'k':
                    phoneme_list[-1] = 'kcl'

            normalized = unicodedata.normalize('NFC', syllable.lower())
            syl_to_phonemes[normalized] = phoneme_list

    return syl_to_phonemes


def vietnamese_text_to_phonemes(text: str, syl_dict: Dict[str, List[str]]) -> List[str]:
    """
    Convert a Vietnamese text string to a list of X-SAMPA phonemes via the
    syllable dictionary.  Unknown syllables produce a warning.
    """
    phonemes: List[str] = []
    words = text.lower().split()
    for word in words:
        word = word.strip('.,!?;:"\'()[]')
        if word in syl_dict:
            phonemes.extend(syl_dict[word])
        else:
            print(f"Warning: Unknown Vietnamese syllable '{word}'")
    return phonemes


# ============================================================================
# Module-level dictionary (loaded once)
# ============================================================================
print("Loading Vietnamese syllable-to-phoneme dictionary…")
VIETNAMESE_SYL_DICT = load_vietnamese_dict(VIETNAMESE_DICT_PATH)
VI_G2P_AVAILABLE = bool(VIETNAMESE_SYL_DICT)
print(f"  {'✓' if VI_G2P_AVAILABLE else '✗'} {len(VIETNAMESE_SYL_DICT)} syllable mappings loaded")
