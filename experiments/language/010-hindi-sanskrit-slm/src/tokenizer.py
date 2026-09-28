"""SentencePiece Unigram tokenizer for Devanagari text.

Control pieces are reserved. Normalization is identity so nukta is preserved.
"""

from pathlib import Path

import sentencepiece as spm

CONTROL = ("<unk>", "<pad>", "<bos>", "<eos>")
USER_DEFINED = (
    "<hi>",
    "<sa>",
    "<standard_hi>",
    "<sanskritized_hi>",
    "<translate_hi_sa>",
    "<translate_sa_hi>",
    "<correct_hi>",
    "<correct_sa>",
    "<src>",
    "<tgt>",
)
RESERVED = CONTROL + USER_DEFINED
REQUIRED_CHARS = "ंःँऽ।॥़क़ख़ग़ज़फ़ड़ढ़"


def train_tokenizer(corpus_path, model_prefix, vocab_size):
    """Train Unigram SentencePiece. `model_prefix` is a path without a suffix."""
    prefix = Path(model_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        byte_fallback=False,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        unk_id=0,
        pad_id=1,
        bos_id=2,
        eos_id=3,
        unk_piece="<unk>",
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        user_defined_symbols=list(USER_DEFINED),
        required_chars=REQUIRED_CHARS,
    )
    return prefix.with_suffix(".model")


class Tokenizer:
    """Loaded SentencePiece model."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))

    @property
    def vocab_size(self):
        return self.sp.vocab_size()

    @property
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode([int(i) for i in ids])

    def id_of(self, piece):
        return self.sp.piece_to_id(piece)

    def pieces(self):
        return [self.sp.id_to_piece(i) for i in range(self.vocab_size)]

    def model_bytes(self):
        return self.model_path.read_bytes()


def write_model_bytes(blob, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path
