"""Write the human-checked training set and the held-out benchmarks.

Training is Samasāmayik train only. Wikipedia, the BPCC pivot, IN22, and
FLORES stay out of these files. The official test, IN22, and FLORES are
written as benchmark JSONL for evaluation.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prep.run import align_pairs, read_lines

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "clean-high"
BENCH = ROOT / "data" / "benchmarks"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_gold():
    rows = []
    with (CLEAN / "sft.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") == "samasamayik":
                row = dict(row)
                row["split"] = "train"
                row["quality"] = "high"
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
                    "source": "samasamayik",
                    "license": "research-only",
                    "quality": "high",
                    "split": "train",
                }
            )
    return mono


def benchmark(name, hi_path, sa_path, license_name):
    pairs, rejected = align_pairs(
        read_lines(hi_path),
        read_lines(sa_path),
        name,
        "high",
        license_name,
        "translate_hi_sa",
        "translate_sa_hi",
    )
    for row in pairs:
        row["split"] = "test"
    return pairs, rejected


def main():
    gold = load_gold()
    mono = pretrain_from_pairs(gold)
    write_jsonl(OUT / "sft.jsonl", gold)
    write_jsonl(OUT / "pretrain.jsonl", mono)
    reports = {"sft_rows": len(gold), "pretrain_rows": len(mono)}
    specs = (
        ("samasamayik-test", BENCH / "samasamayik_test_hi.txt", BENCH / "samasamayik_test_sa.txt", "research-only"),
        ("in22", BENCH / "in22_hi.txt", BENCH / "in22_sa.txt", "CC-BY-SA-4.0"),
        ("flores-200", BENCH / "flores_hi.txt", BENCH / "flores_sa.txt", "CC-BY-SA-4.0"),
    )
    for name, hi_path, sa_path, license_name in specs:
        rows, rejected = benchmark(name, hi_path, sa_path, license_name)
        write_jsonl(OUT / f"benchmark-{name}.jsonl", rows)
        reports[name] = {"rows": len(rows), "rejected": rejected}
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
