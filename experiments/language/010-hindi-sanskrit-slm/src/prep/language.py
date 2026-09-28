"""Lightweight Hindi versus Sanskrit decision from function words.

Generic language id confuses the two because both use Devanagari. This is the
baseline the training filter uses. It is not a measured 98% classifier.
"""

HINDI = {
    "का",
    "की",
    "के",
    "में",
    "है",
    "हैं",
    "को",
    "से",
    "ने",
    "यह",
    "वह",
    "और",
    "नहीं",
    "एक",
    "के",
}
SANSKRIT = {
    "च",
    "वा",
    "हि",
    "एव",
    "इति",
    "अस्ति",
    "सन्ति",
    "तथा",
    "अपि",
    "च",
    "वा",
}


def classify(text):
    """Return (label, confidence) where label is hi, sa, mixed, or unknown."""
    words = [_strip(word) for word in text.split()]
    words = [word for word in words if word]
    if not words:
        return "unknown", 0.0
    hindi = sum(1 for word in words if word in HINDI)
    sanskrit = sum(1 for word in words if word in SANSKRIT)
    if text.count("ः") / len(words) >= 0.12:
        sanskrit += 2
    total = hindi + sanskrit
    if total == 0:
        return "unknown", 0.0
    if hindi >= sanskrit * 2 and hindi >= 1:
        return "hi", hindi / total
    if sanskrit >= hindi * 2 and sanskrit >= 1:
        return "sa", sanskrit / total
    return "mixed", max(hindi, sanskrit) / total


def _strip(word):
    return word.strip("।॥.,;:!?\"'()[]")
