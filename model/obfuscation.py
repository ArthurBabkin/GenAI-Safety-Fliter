"""
Text obfuscation and deobfuscation for robustness stress testing.

Obfuscation applies deterministic transforms (leetspeak, char spacing,
char repetition, mixed case) to simulate adversarial evasion attempts.

Deobfuscation applies rule-based normalization to reverse common evasion
techniques before model inference.
"""

import re
import random
import unicodedata
from typing import List

# --- Leetspeak mappings ---

LEET_MAP = {
    'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7',
    'A': '4', 'E': '3', 'I': '1', 'O': '0', 'S': '5', 'T': '7',
}

LEET_REVERSE = {
    '4': 'a', '3': 'e', '1': 'i', '0': 'o', '5': 's', '7': 't',
    '@': 'a', '!': 'i',
}

# Cyrillic ↔ Latin homoglyphs (common in Russian-language evasion)
CYRILLIC_TO_LATIN = {
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c',
    'у': 'y', 'х': 'x', 'А': 'A', 'Е': 'E', 'О': 'O',
    'Р': 'P', 'С': 'C', 'У': 'Y', 'Х': 'X',
}

LATIN_TO_CYRILLIC = {v: k for k, v in CYRILLIC_TO_LATIN.items()}


def _is_cyrillic_word(word: str) -> bool:
    """Check if a word is predominantly Cyrillic."""
    cyrillic = sum(1 for c in word if '\u0400' <= c <= '\u04ff')
    return cyrillic > len(word) / 2


# --- Obfuscation transforms ---

def _leetspeak(word: str, rng: random.Random) -> str:
    """Replace eligible characters with leetspeak substitutions."""
    result = []
    for ch in word:
        if ch in LEET_MAP and rng.random() < 0.5:
            result.append(LEET_MAP[ch])
        elif _is_cyrillic_word(word) and ch in CYRILLIC_TO_LATIN and rng.random() < 0.4:
            result.append(CYRILLIC_TO_LATIN[ch])
        else:
            result.append(ch)
    return ''.join(result)


def _char_spacing(word: str, rng: random.Random) -> str:
    """Insert spaces between characters: 'hate' -> 'h a t e'."""
    return ' '.join(word)


def _char_repeat(word: str, rng: random.Random) -> str:
    """Double or triple random characters: 'hate' -> 'haatte'."""
    result = []
    for ch in word:
        result.append(ch)
        if ch.isalpha() and rng.random() < 0.35:
            result.append(ch * rng.randint(1, 2))
    return ''.join(result)


def _mixed_case(word: str, rng: random.Random) -> str:
    """Randomize case: 'hate' -> 'hAtE'."""
    return ''.join(
        ch.upper() if rng.random() < 0.5 else ch.lower()
        for ch in word
    )


_TRANSFORMS = [
    (_leetspeak, 0.30),
    (_char_spacing, 0.25),
    (_char_repeat, 0.25),
    (_mixed_case, 0.20),
]


def obfuscate_text(text: str, rng: random.Random, word_prob: float = 0.4) -> str:
    """
    Apply random obfuscation transforms to words in text.

    Each word has `word_prob` chance of being transformed. The specific
    transform is chosen by weighted random selection.

    Args:
        text: Input text.
        rng: Seeded Random instance for determinism.
        word_prob: Probability of transforming each word.

    Returns:
        Obfuscated text.
    """
    words = text.split()
    result = []
    transforms, weights = zip(*_TRANSFORMS)

    for word in words:
        if len(word) >= 2 and rng.random() < word_prob:
            fn = rng.choices(transforms, weights=weights, k=1)[0]
            result.append(fn(word, rng))
        else:
            result.append(word)

    return ' '.join(result)


def obfuscate_dataset(texts: List[str], seed: int = 42,
                      word_prob: float = 0.4) -> List[str]:
    """
    Apply deterministic obfuscation to a list of texts.

    Each text gets its own RNG seeded as (seed + index) for reproducibility
    and independence.

    Args:
        texts: List of input texts.
        seed: Base random seed.
        word_prob: Probability of transforming each word.

    Returns:
        List of obfuscated texts.
    """
    obfuscated = []
    for i, text in enumerate(texts):
        rng = random.Random(seed + i)
        obfuscated.append(obfuscate_text(text, rng, word_prob))
    return obfuscated


# --- Deobfuscation transforms ---

# Regex for spaced-out characters: sequences of single chars separated by spaces
# e.g., "h a t e" but not "I am ok" (common single-char words excluded)
_SPACED_PATTERN = re.compile(
    r'(?<!\w)'           # not preceded by word char
    r'([^\s])'           # first char
    r'((?:\s[^\s]){2,})' # space+char repeated 2+ times (total 3+ chars)
    r'(?!\w)',           # not followed by word char
)

# Repeated chars (3+)
_REPEAT_PATTERN = re.compile(r'(.)\1{2,}')


def _collapse_spaced_chars(text: str) -> str:
    """Collapse spaced-out characters: 'h a t e' -> 'hate'."""
    def _collapse_match(m):
        full = m.group(0)
        return full.replace(' ', '')

    return _SPACED_PATTERN.sub(_collapse_match, text)


def _collapse_repeats(text: str) -> str:
    """Collapse 3+ repeated characters to 1: 'haaate' -> 'hate'."""
    return _REPEAT_PATTERN.sub(r'\1', text)


def _reverse_leetspeak(text: str) -> str:
    """Reverse common leetspeak substitutions."""
    result = []
    for ch in text:
        if ch in LEET_REVERSE:
            result.append(LEET_REVERSE[ch])
        else:
            result.append(ch)
    return ''.join(result)


def _normalize_unicode(text: str) -> str:
    """Normalize Unicode: strip combining marks, normalize to NFC."""
    # NFD decomposition, then strip combining marks
    nfd = unicodedata.normalize('NFD', text)
    stripped = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', stripped)


def deobfuscate_text(text: str) -> str:
    """
    Apply rule-based deobfuscation to reverse common evasion techniques.

    Pipeline order matters:
    1. Unicode normalization (handles homoglyphs/combining marks)
    2. Collapse spaced characters ('h a t e' -> 'hate')
    3. Lowercase
    4. Reverse leetspeak (4->a, 3->e, etc.)
    5. Collapse repeated characters (3+ -> 1)

    Args:
        text: Potentially obfuscated text.

    Returns:
        Deobfuscated text.
    """
    text = _normalize_unicode(text)
    text = _collapse_spaced_chars(text)
    text = text.lower()
    text = _reverse_leetspeak(text)
    text = _collapse_repeats(text)
    return text


def deobfuscate_dataset(texts: List[str]) -> List[str]:
    """
    Apply deobfuscation to a list of texts.

    Args:
        texts: List of potentially obfuscated texts.

    Returns:
        List of deobfuscated texts.
    """
    return [deobfuscate_text(t) for t in texts]
