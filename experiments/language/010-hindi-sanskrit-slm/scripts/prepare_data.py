"""Prepare the Hindi-Sanskrit training files.

    uv run python scripts/prepare_data.py --hi-words 1000000 --sa-words 1000000

Writes data/clean/pretrain.jsonl, data/clean/sft.jsonl, and
data/manifests/sources.csv. Benchmarks are downloaded and then banned from
the training files.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prep.run import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hi-words", type=int, default=1_000_000)
    parser.add_argument("--sa-words", type=int, default=1_000_000)
    parser.add_argument("--skip-wikipedia", action="store_true")
    args = parser.parse_args()
    report = prepare(
        hi_words=args.hi_words,
        sa_words=args.sa_words,
        include_wikipedia=not args.skip_wikipedia,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
