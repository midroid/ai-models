"""Write the high-plus-medium training set.

High is Samasāmayik train. Medium is the BPCC English pivot. Wikipedia, IN22,
and FLORES stay out. Benchmarks stay in data/clean-high.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "clean-high-medium"

KEEP = {
    "samasamayik": "high",
    "bpcc-pivot": "medium",
}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows():
    rows = []
    with (CLEAN / "sft.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            quality = KEEP.get(row.get("dataset"))
            if quality is None:
                continue
            row = dict(row)
            row["split"] = "train"
            row["quality"] = quality
            rows.append(row)
    return rows


def pretrain_from_pairs(rows):
    seen = set()
    mono = []
    for row in rows:
        if row["task"] != "translate_hi_sa":
            continue
        for language, text in (("hi", row["source"]), ("sa", row["target"])):
            key = (language, text)
            if key in seen:
                continue
            seen.add(key)
            mono.append(
                {
                    "text": text,
                    "language": language,
                    "source": row["dataset"],
                    "license": row.get("license", ""),
                    "quality": row["quality"],
                    "split": "train",
                }
            )
    return mono


def main():
    rows = load_rows()
    mono = pretrain_from_pairs(rows)
    write_jsonl(OUT / "sft.jsonl", rows)
    write_jsonl(OUT / "pretrain.jsonl", mono)
    counts = {}
    for row in rows:
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
    print(json.dumps({"sft_rows": len(rows), "pretrain_rows": len(mono), "by_dataset": counts}, indent=2))


if __name__ == "__main__":
    main()
