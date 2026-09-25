"""SentencePiece round-trip and the Devanagari vocab scan."""

from pathlib import Path

from src.data import balanced_pretrain_text, read_jsonl
from src.tokenizer import RESERVED, Tokenizer, train_tokenizer

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixture" / "pretrain.jsonl"


def _train(tmp_path):
    rows = read_jsonl(FIXTURE)
    text = balanced_pretrain_text(rows)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    chars = {ch for ch in text if not ch.isspace()}
    vocab = max(64, len(chars) + 32)
    return Tokenizer(train_tokenizer(corpus, tmp_path / "sp", vocab))


def test_roundtrip_nukta_and_danda(tmp_path):
    tokenizer = _train(tmp_path)
    samples = [
        "मुझे क़ानून की ज़रूरत है।",
        "धर्मो रक्षति रक्षितः॥",
        "चाँदनी रात में ख़त आया।",
    ]
    for text in samples:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocab_has_no_arabic_or_latin(tmp_path):
    tokenizer = _train(tmp_path)
    allowed = set(RESERVED)
    for piece in tokenizer.pieces():
        if piece in allowed or piece.startswith("<"):
            continue
        for char in piece:
            assert not ("A" <= char <= "Z" or "a" <= char <= "z")
            assert not ("\u0600" <= char <= "\u06ff")
