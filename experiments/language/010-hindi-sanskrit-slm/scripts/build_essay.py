"""Build essay SFT rows that fit the current 512-token context.

Hindi passages come from the MIDAS public-domain story sentences, joined back
into stories. Sanskrit passages come from the Devanagari Mahabharata and
Ramayana distributed by bombay.indology.info. A matching number of Samasamayik
translation rows is mixed in so this fine-tune still sees the translation tasks.

Wikipedia and Sangraha are not used. The official test, IN22, and FLORES are not used.
"""

import json
import random
import re
from pathlib import Path

from src.data import TASK_TAGS
from src.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
LONGFORM = ROOT / "data" / "raw" / "longform"
HIGH_SFT = ROOT / "data" / "clean-high" / "sft.jsonl"
TOKENIZER = ROOT / "tokenizers" / "sp_hm.model"
OUTPUT = ROOT / "data" / "raw" / "essay" / "sft.jsonl"
BLOCK_LIMIT = 496
MIN_TARGET_WORDS = 40
MAX_SANSKRIT = 8000
VERSE_ID = re.compile(r"^\d+[A-Za-z]*\s+")


def local_sentence_report():
    print("current training files, one line each")
    for name in (
        "samasamayik_train_hi.txt",
        "samasamayik_train_sa.txt",
        "bpcc_train_hi.txt",
        "bpcc_train_sa.txt",
    ):
        path = ROOT / "data" / "raw" / name
        lengths = []
        for line in path.open(encoding="utf-8"):
            text = line.strip()
            if text:
                lengths.append(len(text.split()))
        lengths.sort()
        def pct(p):
            return lengths[min(len(lengths) - 1, int(p * (len(lengths) - 1)))]
        long = sum(1 for n in lengths if n >= 80)
        print(
            f"  {name}: rows {len(lengths)}  words p50 {pct(0.5)}  "
            f"p99 {pct(0.99)}  max {lengths[-1]}  lines>=80w {long}"
        )


def epic_units(path):
    units = []
    for line in path.open(encoding="utf-8"):
        text = line.strip()
        if not text or text.startswith("%"):
            continue
        text = VERSE_ID.sub("", text).strip()
        if text:
            units.append(text)
    return units


def story_documents(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = list(payload.values())
    else:
        rows = payload
    grouped = {}
    for index, row in enumerate(rows):
        story = row.get("Story_no", row.get("story_no"))
        sentence = (row.get("Sentence") or row.get("sentence") or "").strip()
        if story is None or not sentence:
            continue
        grouped.setdefault(story, []).append((index, sentence))
    documents = []
    for story in sorted(grouped):
        documents.append([sentence for _, sentence in grouped[story]])
    return documents


def unit_costs(tokenizer, units):
    costs = []
    for index, unit in enumerate(units):
        piece = unit if index == 0 else " " + unit
        costs.append(len(tokenizer.encode(piece)))
    return costs


def passages(units, tokenizer, tag, block_limit=BLOCK_LIMIT):
    if len(units) < 2:
        return []
    costs = unit_costs(tokenizer, units)
    rows = []
    index = 0
    while index < len(units) - 1:
        prefix = f"{tag}<src>{units[index]}<tgt>"
        used = len(tokenizer.encode(prefix)) + 1
        target = []
        target_cost = 0
        cursor = index + 1
        while cursor < len(units) and used + target_cost + costs[cursor] <= block_limit:
            target.append(units[cursor])
            target_cost += costs[cursor]
            cursor += 1
        while target:
            text = " ".join(target)
            ids = tokenizer.encode(f"{tag}<src>{units[index]}<tgt>{text}")
            if len(ids) + 1 <= block_limit:
                break
            target.pop()
        text = " ".join(target)
        if len(text.split()) >= MIN_TARGET_WORDS:
            rows.append((units[index], text))
            index += 1 + len(target)
        else:
            index += 1
    return rows


def translation_replay(count):
    chosen = []
    seen = 0
    rng = random.Random(7)
    for line in HIGH_SFT.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("dataset") != "samasamayik":
            continue
        if row.get("task") not in ("translate_hi_sa", "translate_sa_hi"):
            continue
        seen += 1
        if len(chosen) < count:
            chosen.append(row)
        else:
            swap = rng.randrange(seen)
            if swap < count:
                chosen[swap] = row
    return chosen


def write_rows(rows):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    local_sentence_report()
    tokenizer = Tokenizer(TOKENIZER)
    hindi_rows = []
    documents = story_documents(LONGFORM / "discourse_dataset.json")
    tag = TASK_TAGS["essay_hi"]
    for units in documents:
        for source, target in passages(units, tokenizer, tag):
            hindi_rows.append(
                {
                    "task": "essay_hi",
                    "source": source,
                    "target": target,
                    "dataset": "hindi-discourse",
                    "quality": "high",
                    "license": "research-only",
                    "split": "train",
                }
            )
    sanskrit_rows = []
    tag = TASK_TAGS["essay_sa"]
    files = sorted((LONGFORM / "mbh").glob("MBh*.txt")) + sorted(
        (LONGFORM / "ramayana").glob("Ram*.txt")
    )
    for path in files:
        dataset = "mahabharata" if path.name.startswith("MBh") else "ramayana"
        for source, target in passages(epic_units(path), tokenizer, tag):
            sanskrit_rows.append(
                {
                    "task": "essay_sa",
                    "source": source,
                    "target": target,
                    "dataset": dataset,
                    "quality": "high",
                    "license": "bori-electronic-text",
                    "split": "train",
                }
            )
    hindi_unique = len(hindi_rows)
    hindi_rows = hindi_rows * 3
    cap = min(len(sanskrit_rows), max(hindi_unique * 3, 1), MAX_SANSKRIT)
    if len(sanskrit_rows) > cap:
        rng = random.Random(7)
        sanskrit_rows = rng.sample(sanskrit_rows, cap)
    replay = translation_replay(len(hindi_rows) + len(sanskrit_rows))
    rows = hindi_rows + sanskrit_rows + replay
    rng = random.Random(7)
    rng.shuffle(rows)
    write_rows(rows)
    print(
        f"essay_hi {len(hindi_rows)}  essay_sa {len(sanskrit_rows)}  "
        f"translation_replay {len(replay)}  total {len(rows)}"
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
