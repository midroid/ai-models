"""Loss masks and non-empty fixture rows."""

from pathlib import Path

from src.data import encode_sft, read_jsonl, supervised_count

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixture"


class FakeTokenizer:
    def __init__(self):
        pieces = [
            "<unk>",
            "<pad>",
            "<bos>",
            "<eos>",
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
        ]
        self._pieces = {piece: index for index, piece in enumerate(pieces)}
        self.eos_id = self._pieces["<eos>"]
        self.pad_id = self._pieces["<pad>"]

    def encode(self, text):
        ids = []
        index = 0
        specials = sorted(self._pieces, key=len, reverse=True)
        while index < len(text):
            matched = False
            for piece in specials:
                if text.startswith(piece, index):
                    ids.append(self._pieces[piece])
                    index += len(piece)
                    matched = True
                    break
            if not matched:
                ids.append(100 + ord(text[index]))
                index += 1
        return ids


def test_fixture_rows_are_nonempty():
    pretrain = read_jsonl(FIXTURE / "pretrain.jsonl")
    sft = read_jsonl(FIXTURE / "sft.jsonl")
    assert pretrain and sft
    for row in pretrain:
        assert row["text"]
        assert row["language"] in ("hi", "sa")
    for row in sft:
        if row["task"] == "standard_hi":
            assert row["text"]
        else:
            assert row["source"] and row["target"]


def test_loss_mask_starts_after_tgt():
    tokenizer = FakeTokenizer()
    ids, labels = encode_sft(
        {
            "task": "translate_hi_sa",
            "source": "भारत",
            "target": "भारतम्",
        },
        tokenizer,
    )
    tgt = tokenizer._pieces["<tgt>"]
    tgt_at = ids.index(tgt)
    assert labels[: tgt_at + 1] == [-100] * (tgt_at + 1)
    assert labels[tgt_at + 1] == ids[tgt_at + 1]
    assert labels[-1] == tokenizer.eos_id
    assert supervised_count(labels) > 0


def test_standard_hi_is_fully_supervised():
    tokenizer = FakeTokenizer()
    ids, labels = encode_sft({"task": "standard_hi", "text": "जानकारी"}, tokenizer)
    assert labels == ids
    assert -100 not in labels
