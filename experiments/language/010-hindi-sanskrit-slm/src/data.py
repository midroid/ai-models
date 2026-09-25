"""JSONL rows, language packing, and target-only labels.

Pretraining packs one language at a time. Supervised rows put -100 through
the `<tgt>` token so the shifted loss starts at the first target token.
"""

import hashlib
import json
from pathlib import Path

TASK_TAGS = {
    "translate_hi_sa": "<translate_hi_sa>",
    "translate_sa_hi": "<translate_sa_hi>",
    "sanskritize_hindi": "<sanskritized_hi>",
    "correct_hi": "<correct_hi>",
    "correct_sa": "<correct_sa>",
    "standard_hi": "<standard_hi>",
}


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_pretrain_text(rows):
    """Half the characters are Hindi and half are Sanskrit."""
    groups = {"hi": [], "sa": []}
    for row in rows:
        language = row.get("language")
        text = row.get("text", "")
        if language not in groups or not text:
            raise ValueError("pretrain rows need language hi|sa and non-empty text")
        groups[language].append(text)
    if not groups["hi"] or not groups["sa"]:
        raise ValueError("pretrain corpus needs both Hindi and Sanskrit")
    target = max(sum(len(text) for text in groups["hi"]), sum(len(text) for text in groups["sa"]))
    lines = []
    for language in ("hi", "sa"):
        bucket = groups[language]
        taken = 0
        index = 0
        while taken < target:
            text = bucket[index % len(bucket)]
            lines.append(text)
            taken += len(text)
            index += 1
    return "\n".join(lines) + "\n"


def _repeat(rows, minimum_tokens):
    tiled = []
    total = 0
    while total < minimum_tokens:
        for row in rows:
            tiled.append(row)
            total += len(row)
            if total >= minimum_tokens:
                break
    return tiled


def pack_ids(sequences, block_size):
    """Non-overlapping chunks of block_size + 1 so each position has a next token."""
    stream = [token for sequence in sequences for token in sequence]
    width = block_size + 1
    if len(stream) < width:
        stream = [token for copy in _repeat([stream], width) for token in copy]
    blocks = []
    for start in range(0, len(stream) - width + 1, block_size):
        blocks.append(stream[start : start + width])
    if not blocks:
        raise ValueError(f"need at least {width} tokens to pack")
    return blocks


def pretrain_examples(rows, tokenizer, block_size):
    """Pack each language on its own stream. Returns (input, label) pairs."""
    grouped = {"hi": [], "sa": []}
    for row in rows:
        language = row["language"]
        if language not in grouped:
            raise ValueError(f"unknown language {language}")
        text = row["text"]
        if not text:
            raise ValueError("empty pretrain text")
        ids = tokenizer.encode(f"<{language}>{text}")
        ids.append(tokenizer.eos_id)
        grouped[language].append(ids)
    examples = []
    for sequences in grouped.values():
        if not sequences:
            continue
        for chunk in pack_ids(sequences, block_size):
            # Labels align with the tokens. The model shifts once for next-token loss.
            window = chunk[:-1]
            examples.append((window, window))
    return examples


def encode_sft(row, tokenizer):
    """Return token ids and labels of the same length."""
    task = row["task"]
    if task not in TASK_TAGS:
        raise ValueError(f"unknown task {task}")
    tag = TASK_TAGS[task]
    eos = tokenizer.eos_id
    if task == "standard_hi":
        text = row.get("text") or ""
        if not text:
            raise ValueError("standard_hi row needs text")
        ids = tokenizer.encode(tag + text)
        ids.append(eos)
        return ids, list(ids)
    source = row.get("source") or ""
    target = row.get("target") or ""
    if not source or not target:
        raise ValueError(f"{task} row needs source and target")
    prefix = tokenizer.encode(f"{tag}<src>{source}<tgt>")
    completion = tokenizer.encode(target)
    completion.append(eos)
    ids = prefix + completion
    labels = [-100] * len(prefix) + completion
    return ids, labels


def sft_examples(rows, tokenizer, block_size, repeat_to=1):
    """Truncate each supervised row to block_size. Padding happens per batch."""
    if repeat_to > len(rows):
        rows = [rows[index % len(rows)] for index in range(repeat_to)]
    examples = []
    for row in rows:
        ids, labels = encode_sft(row, tokenizer)
        if len(ids) > block_size:
            ids = ids[:block_size]
            labels = labels[:block_size]
        if len(ids) < 2:
            continue
        examples.append((ids, labels))
    if not examples:
        raise ValueError("no supervised rows fit in block_size")
    return examples


def supervised_count(labels):
    """Positions that enter the shifted cross-entropy."""
    return sum(1 for label in labels[1:] if label != -100)
