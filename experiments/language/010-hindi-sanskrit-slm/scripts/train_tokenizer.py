"""Train a SentencePiece Unigram tokenizer.

    uv run python scripts/train_tokenizer.py --vocab-size 8000 --corpus data/fixture/pretrain.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import balanced_pretrain_text, read_jsonl
from src.tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tokenizers/sp_8k"))
    args = parser.parse_args()
    text = balanced_pretrain_text(read_jsonl(args.corpus))
    corpus_path = args.output.parent / "corpus.txt"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(text, encoding="utf-8")
    model_path = train_tokenizer(corpus_path, args.output, args.vocab_size)
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
