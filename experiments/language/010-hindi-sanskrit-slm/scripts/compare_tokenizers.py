"""Compare 6k, 8k, and 12k Unigram tokenizers on the cleaned corpus.

    uv run python scripts/compare_tokenizers.py

Trains each size on a 50/50 Hindi-Sanskrit character budget and writes
tokenizers/compare.json.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer import REQUIRED_CHARS, Tokenizer, train_tokenizer

CLEAN = Path("data/clean/pretrain.jsonl")
OUT = Path("tokenizers")
SIZES = (6000, 8000, 12000)
PROBES = (
    "मुझे क़ानून की ज़रूरत है।",
    "धर्मो रक्षति रक्षितः॥",
    "चाँदनी रात में ख़त आया।",
    "अन्तर्राष्ट्रीयकरणम्",
    REQUIRED_CHARS,
)


def split_corpus():
    """Hold out every 50th paragraph. Train on equal character budgets."""
    train, held = {"hi": [], "sa": []}, {"hi": [], "sa": []}
    seen = {"hi": 0, "sa": 0}
    with CLEAN.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            language = row.get("language")
            text = (row.get("text") or "").replace("\n", " ").strip()
            if language not in train or not text:
                continue
            seen[language] += 1
            bucket = held if seen[language] % 50 == 0 else train
            bucket[language].append(text)
    target = min(sum(len(text) for text in train[lang]) for lang in train)
    lines = []
    kept = {}
    for language in ("hi", "sa"):
        taken = 0
        chosen = []
        for text in train[language]:
            if taken >= target:
                break
            chosen.append(text)
            taken += len(text)
        kept[language] = taken
        lines.extend(chosen)
    random.Random(0).shuffle(lines)
    return lines, held, kept


def fertility(tokenizer, lines):
    tokens = words = chars = unk = 0
    for text in lines:
        ids = tokenizer.encode(text)
        tokens += len(ids)
        words += max(len(text.split()), 1)
        chars += len(text)
        unk += sum(1 for index in ids if index == tokenizer.sp.unk_id())
    return {
        "lines": len(lines),
        "tokens_per_word": tokens / max(words, 1),
        "tokens_per_char": tokens / max(chars, 1),
        "unk_rate": unk / max(tokens, 1),
    }


def roundtrip(tokenizer):
    samples = [
        "मुझे क़ानून की ज़रूरत है।",
        "धर्मो रक्षति रक्षितः॥",
        "चाँदनी रात में ख़त आया।",
        REQUIRED_CHARS,
    ]
    return all(tokenizer.decode(tokenizer.encode(text)) == text for text in samples)


def main():
    lines, held, kept = split_corpus()
    corpus = OUT / "corpus.txt"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {"train_chars": kept, "sizes": {}}
    for size in SIZES:
        prefix = OUT / f"sp_{size // 1000}k"
        print(f"training {size}", flush=True)
        model_path = train_tokenizer(corpus, prefix, size)
        tokenizer = Tokenizer(model_path)
        stats = {
            "vocab": tokenizer.vocab_size,
            "roundtrip": roundtrip(tokenizer),
            "hi": fertility(tokenizer, held["hi"]),
            "sa": fertility(tokenizer, held["sa"]),
            "probes": {
                text: tokenizer.decode(tokenizer.encode(text)) == text for text in PROBES
            },
        }
        report["sizes"][str(size)] = stats
        print(size, json.dumps(stats), flush=True)
    (OUT / "compare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT / "compare.json")


if __name__ == "__main__":
    main()
