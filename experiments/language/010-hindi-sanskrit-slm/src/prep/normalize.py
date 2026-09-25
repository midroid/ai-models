"""Devanagari-aware text cleanup.

NFC, whitespace, and a script-ratio check. Nukta letters are left as they are.
"""

import re
import unicodedata

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F
MIN_CHARS = 20
MAX_CHARS = 4000


def normalize_text(text):
    """Return cleaned text, or None when the line should be dropped."""
    if not isinstance(text, str):
        return None
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return None
    if _repeated(text):
        return None
    return text


def devanagari_ratio(text):
    """Share of letters that are Devanagari. Digits and punctuation are ignored."""
    letters = [char for char in text if char.isalpha() or _is_devanagari(char)]
    if not letters:
        return 0.0
    return sum(1 for char in letters if _is_devanagari(char)) / len(letters)


def passes_script(text, minimum):
    return devanagari_ratio(text) >= minimum


def _is_devanagari(char):
    return DEVANAGARI_START <= ord(char) <= DEVANAGARI_END


def _repeated(text):
    words = text.split()
    if len(words) >= 8 and len(set(words)) / len(words) < 0.3:
        return True
    return False
