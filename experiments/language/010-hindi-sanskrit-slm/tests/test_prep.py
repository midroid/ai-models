"""Normalization, language tags, dedupe, and leakage."""

from src.prep.dedup import dedupe, hash_set, leakage
from src.prep.language import classify
from src.prep.normalize import normalize_text
from src.prep.run import align_pairs


def test_nukta_is_preserved():
    text = normalize_text("मुझे क़ानून की ज़रूरत है और यह वाक्य काफी लंबा है।")
    assert text is not None
    assert "क़" in text
    assert "ज़" in text


def test_short_lines_drop():
    assert normalize_text("छोटा") is None


def test_hindi_and_sanskrit_labels():
    assert classify("भारत में अनेक भाषाएँ बोली जाती हैं और यह एक देश है।")[0] == "hi"
    assert classify("भारते अनेकाः भाषाः भाष्यन्ते तथा अस्ति इति वा।")[0] == "sa"


def test_exact_dedupe_and_leakage():
    rows = [{"text": "एक"}, {"text": "एक"}, {"text": "दो"}]
    kept, dropped = dedupe(rows, lambda row: row["text"])
    assert dropped == 1
    assert len(kept) == 2
    banned = hash_set(["दो"])
    kept, leaks = leakage(kept, banned, lambda row: row["text"])
    assert leaks == 1
    assert [row["text"] for row in kept] == ["एक"]


def test_parallel_alignment_keeps_both_directions():
    left = ["भारत में अनेक भाषाएँ बोली जाती हैं और लोग उन्हें पढ़ते हैं।"]
    right = ["भारते अनेकाः भाषाः भाष्यन्ते तथा जनाः ताः पठन्ति इति।"]
    rows, rejected = align_pairs(left, right, "fixture", "gold", "research-only", "translate_hi_sa", "translate_sa_hi")
    assert rejected == 0
    assert {row["task"] for row in rows} == {"translate_hi_sa", "translate_sa_hi"}
