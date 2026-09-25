"""Download licensed sources, clean them, and write trainer JSONL.

Monolingual text comes from Hindi and Sanskrit Wikipedia (CC BY-SA 4.0).
Parallel text comes from the Samasāmayik GitHub release. That repository has
no LICENSE file; the paper limits the release to research use, so the manifest
marks commercial use and redistribution as false. IN22 and FLORES are
CC-BY-SA-4.0, so they enter training as silver pairs. The official
Samasāmayik test stays held out. Sangraha stays out until its licence is clear.
"""

import csv
import hashlib
import json
import urllib.request
from datetime import date
from pathlib import Path

from src.prep.dedup import content_hash, dedupe, hash_set, leakage
from src.prep.language import classify
from src.prep.normalize import devanagari_ratio, normalize_text, passes_script

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
RAW = EXPERIMENT_DIR / "data" / "raw"
CLEAN = EXPERIMENT_DIR / "data" / "clean"
BENCHMARKS = EXPERIMENT_DIR / "data" / "benchmarks"
MANIFESTS = EXPERIMENT_DIR / "data" / "manifests"

SAMASAMAYIK = "https://raw.githubusercontent.com/karthika95/samasaamayik/main/final_data"
PARALLEL_SCRIPT_MIN = 0.6
MONO_SCRIPT_MIN = 0.9

RAW_FILES = {
    "samasamayik_train_hi": f"{SAMASAMAYIK}/train/samasaamayik/combn_train.hi",
    "samasamayik_train_sa": f"{SAMASAMAYIK}/train/samasaamayik/combn_train.sa",
    "samasamayik_test_hi": f"{SAMASAMAYIK}/test/samasaamayik/combn_test.hi",
    "samasamayik_test_sa": f"{SAMASAMAYIK}/test/samasaamayik/combn_test.sa",
    "bpcc_train_hi": f"{SAMASAMAYIK}/train/bpcc/hindi.txt",
    "bpcc_train_sa": f"{SAMASAMAYIK}/train/bpcc/sanskrit.txt",
    "in22_hi": f"{SAMASAMAYIK}/test/IN22/hindi.txt",
    "in22_sa": f"{SAMASAMAYIK}/test/IN22/sanskrit.txt",
    "flores_hi": f"{SAMASAMAYIK}/test/flores/hindi.txt",
    "flores_sa": f"{SAMASAMAYIK}/test/flores/sanskrit.txt",
}


def download_raw():
    """Fetch the GitHub release files once. Existing files are kept."""
    RAW.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, url in RAW_FILES.items():
        path = RAW / f"{name}.txt"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        written[name] = path
    return written


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def licensed_benchmarks(files, prepared_date="2026-09-25"):
    """IN22 and FLORES are CC-BY-SA, so both directions enter training as silver."""
    rows = []
    rejected = 0
    for dataset, hi_key, sa_key in (
        ("in22", "in22_hi", "in22_sa"),
        ("flores-200", "flores_hi", "flores_sa"),
    ):
        part, dropped = align_pairs(
            read_lines(files[hi_key]),
            read_lines(files[sa_key]),
            dataset,
            "silver",
            "CC-BY-SA-4.0",
            "translate_hi_sa",
            "translate_sa_hi",
        )
        for row in part:
            row["prepared_date"] = prepared_date
            row["snapshot_date"] = prepared_date
        rows.extend(part)
        rejected += dropped
    return rows, rejected


def align_pairs(left_lines, right_lines, dataset, quality, license_name, task_forward, task_backward):
    """Zip aligned lines into both translation directions."""
    rows = []
    rejected = 0
    for left, right in zip(left_lines, right_lines):
        pair = _clean_pair(left, right)
        if pair is None:
            rejected += 1
            continue
        source, target = pair
        rows.append(_parallel_row(task_forward, source, target, dataset, quality, license_name))
        rows.append(_parallel_row(task_backward, target, source, dataset, quality, license_name))
    return rows, rejected


def _clean_pair(left, right):
    source = normalize_text(left)
    target = normalize_text(right)
    if source is None or target is None:
        return None
    if not passes_script(source, PARALLEL_SCRIPT_MIN) or not passes_script(target, PARALLEL_SCRIPT_MIN):
        return None
    ratio = len(source) / max(len(target), 1)
    if ratio < 0.3 or ratio > 3.0:
        return None
    return source, target


def _parallel_row(task, source, target, dataset, quality, license_name):
    return {
        "task": task,
        "source": source,
        "target": target,
        "dataset": dataset,
        "quality": quality,
        "license": license_name,
        "language_source": classify(source)[0],
        "language_target": classify(target)[0],
    }


def mono_rows(paragraphs, language, dataset, license_name):
    rows = []
    for text in paragraphs:
        cleaned = normalize_text(text)
        if cleaned is None or not passes_script(cleaned, MONO_SCRIPT_MIN):
            continue
        label, confidence = classify(cleaned)
        if label not in (language, "unknown"):
            continue
        rows.append(
            {
                "text": cleaned,
                "language": language,
                "source": dataset,
                "license": license_name,
                "language_predicted": label,
                "language_confidence": round(confidence, 3),
                "script_ratio": round(devanagari_ratio(cleaned), 3),
                "quality": "silver",
            }
        )
    return rows


def wikipedia_paragraphs(language, word_cap):
    """Stream one Wikipedia snapshot until word_cap words are kept."""
    from datasets import load_dataset

    config = f"20231101.{language}"
    dataset = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    words = 0
    paragraphs = []
    for article in dataset:
        text = article.get("text") or ""
        for piece in text.split("\n"):
            piece = piece.strip()
            if len(piece) < 20:
                continue
            paragraphs.append(piece)
            words += len(piece.split())
            if words >= word_cap:
                return paragraphs
    return paragraphs


def held_out_hashes(files):
    texts = []
    for path in files.values():
        for line in read_lines(path):
            cleaned = normalize_text(line)
            if cleaned:
                texts.append(cleaned)
    return hash_set(texts)


def assign_split(row, key, val_percent=1):
    bucket = int(content_hash(key(row))[:8], 16) % 100
    row["split"] = "val" if bucket < val_percent else "train"
    return row


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "url",
        "version",
        "language",
        "license",
        "commercial_use",
        "redistribution_allowed",
        "exclude_from_training",
        "number_of_documents",
        "number_of_sentences",
        "download_date",
        "checksum",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def file_checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(hi_words=1_000_000, sa_words=1_000_000, include_wikipedia=True):
    """Run the full preparation and return a short count report."""
    files = download_raw()
    # The official test stays out. IN22 and FLORES are CC-BY-SA, so they train.
    banned = held_out_hashes(
        {
            "samasamayik_test_hi": files["samasamayik_test_hi"],
            "samasamayik_test_sa": files["samasamayik_test_sa"],
        }
    )
    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    for name in ("samasamayik_test_hi", "samasamayik_test_sa", "in22_hi", "in22_sa", "flores_hi", "flores_sa"):
        target = BENCHMARKS / files[name].name
        if not target.exists():
            target.write_bytes(files[name].read_bytes())

    gold, gold_rejected = align_pairs(
        read_lines(files["samasamayik_train_hi"]),
        read_lines(files["samasamayik_train_sa"]),
        "samasamayik",
        "gold",
        "research-only",
        "translate_hi_sa",
        "translate_sa_hi",
    )
    silver, silver_rejected = align_pairs(
        read_lines(files["bpcc_train_hi"]),
        read_lines(files["bpcc_train_sa"]),
        "bpcc-pivot",
        "silver",
        "CC0",
        "translate_hi_sa",
        "translate_sa_hi",
    )
    benchmarks, benchmark_rejected = licensed_benchmarks(files)
    parallel = gold + silver + benchmarks
    parallel, parallel_dupes = dedupe(parallel, lambda row: row["task"] + "\n" + row["source"] + "\n" + row["target"])
    parallel, parallel_leaks = leakage(
        parallel,
        banned,
        lambda row: row["source"],
    )
    parallel, target_leaks = leakage(parallel, banned, lambda row: row["target"])
    parallel = [assign_split(row, lambda item: item["task"] + "\n" + item["source"]) for row in parallel]
    prepared = "2026-09-25"
    for row in parallel:
        row["prepared_date"] = prepared
        row.setdefault("snapshot_date", prepared)

    mono = []
    if include_wikipedia:
        mono.extend(mono_rows(wikipedia_paragraphs("hi", hi_words), "hi", "wikipedia-hi", "CC-BY-SA-4.0"))
        mono.extend(mono_rows(wikipedia_paragraphs("sa", sa_words), "sa", "wikipedia-sa", "CC-BY-SA-4.0"))
    mono, mono_dupes = dedupe(mono, lambda row: row["text"])
    mono, mono_leaks = leakage(mono, banned, lambda row: row["text"])
    mono = [assign_split(row, lambda item: item["text"]) for row in mono]
    for row in mono:
        row["prepared_date"] = "2026-09-25"
        row["snapshot_date"] = "2023-11-01"

    write_jsonl(CLEAN / "pretrain.jsonl", mono)
    write_jsonl(CLEAN / "sft.jsonl", parallel)
    today = date.today().isoformat()
    records = _manifest_rows(files, today, len(gold), len(silver), len(mono))
    write_manifest(MANIFESTS / "sources.csv", records)
    report = {
        "pretrain_rows": len(mono),
        "pretrain_hi": sum(1 for row in mono if row["language"] == "hi"),
        "pretrain_sa": sum(1 for row in mono if row["language"] == "sa"),
        "sft_rows": len(parallel),
        "sft_train": sum(1 for row in parallel if row["split"] == "train"),
        "sft_val": sum(1 for row in parallel if row["split"] == "val"),
        "gold_rejected": gold_rejected,
        "silver_rejected": silver_rejected,
        "benchmark_train_rows": len(benchmarks),
        "benchmark_rejected": benchmark_rejected,
        "parallel_dupes": parallel_dupes,
        "parallel_leaks": parallel_leaks + target_leaks,
        "mono_dupes": mono_dupes,
        "mono_leaks": mono_leaks,
        "benchmark_hashes": len(banned),
    }
    (MANIFESTS / "prepare_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _manifest_rows(files, today, gold_rows, silver_rows, mono_rows_n):
    def row(dataset, url, language, license_name, commercial, redistribute, exclude, sentences, notes, path=None):
        marks = {
            "samasamayik-train": ("gold", today),
            "samasamayik-test": ("gold", today),
            "bpcc-pivot": ("silver", today),
            "in22": ("silver", today),
            "flores-200": ("silver", today),
            "wikipedia-hi": ("silver", "2023-11-01"),
            "wikipedia-sa": ("silver", "2023-11-01"),
            "sangraha": ("unreviewed", ""),
        }
        quality, snapshot = marks.get(dataset, ("", today))
        return {
            "dataset": dataset,
            "snapshot_date": snapshot,
            "quality": quality,
            "url": url,
            "version": "2026-09-25",
            "language": language,
            "license": license_name,
            "commercial_use": commercial,
            "redistribution_allowed": redistribute,
            "exclude_from_training": exclude,
            "number_of_documents": "",
            "number_of_sentences": sentences,
            "download_date": today,
            "checksum": file_checksum(path) if path else "",
            "notes": notes,
        }

    return [
        row(
            "samasamayik-train",
            RAW_FILES["samasamayik_train_hi"],
            "hi-sa",
            "research-only",
            "false",
            "false",
            "false",
            gold_rows,
            "Paper ethics statement limits use to research. No LICENSE file in the GitHub repo. Official test split is excluded.",
            files["samasamayik_train_hi"],
        ),
        row(
            "samasamayik-test",
            RAW_FILES["samasamayik_test_hi"],
            "hi-sa",
            "research-only",
            "false",
            "false",
            "true",
            "",
            "Official held-out test. Used only as a leakage ban list.",
            files["samasamayik_test_hi"],
        ),
        row(
            "bpcc-pivot",
            RAW_FILES["bpcc_train_hi"],
            "hi-sa",
            "CC0",
            "true",
            "true",
            "false",
            silver_rows,
            "Hindi-Sanskrit pairs pivoted through English from BPCC, as shipped beside Samasamayik. Silver quality.",
            files["bpcc_train_hi"],
        ),
        row(
            "in22",
            RAW_FILES["in22_hi"],
            "hi-sa",
            "CC-BY-SA-4.0",
            "true",
            "true",
            "false",
            "",
            "CC-BY-SA-4.0. Silver. Prepared 2026-09-25. Included in training.",
            files["in22_hi"],
        ),
        row(
            "flores-200",
            RAW_FILES["flores_hi"],
            "hi-sa",
            "CC-BY-SA-4.0",
            "true",
            "true",
            "false",
            "",
            "CC-BY-SA-4.0. Silver. Prepared 2026-09-25. Included in training.",
            files["flores_hi"],
        ),
        row(
            "wikipedia-hi",
            "https://huggingface.co/datasets/wikimedia/wikipedia",
            "hi",
            "CC-BY-SA-4.0",
            "true",
            "true",
            "false",
            mono_rows_n,
            "Snapshot 20231101.hi. ShareAlike applies to derivatives.",
        ),
        row(
            "wikipedia-sa",
            "https://huggingface.co/datasets/wikimedia/wikipedia",
            "sa",
            "CC-BY-SA-4.0",
            "true",
            "true",
            "false",
            "",
            "Snapshot 20231101.sa. ShareAlike applies to derivatives.",
        ),
        row(
            "sangraha",
            "https://huggingface.co/datasets/ai4bharat/sangraha",
            "hi-sa",
            "unclear",
            "false",
            "false",
            "true",
            "",
            "Not downloaded. License review is still open.",
        ),
    ]
