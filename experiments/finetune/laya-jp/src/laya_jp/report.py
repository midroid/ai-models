"""Assemble the before-column from saved predictions.

    USE_TF=0 uv run python -m laya_jp.report
"""

from __future__ import annotations

import json
from collections import defaultdict

from laya_jp.metrics import score_choice, score_noul, score_ordinal
from laya_jp.pins import (
    DRIFT_TOLERANCE,
    PUBLISHED_BENCH_JA,
    PUBLISHED_SNSK_MLX,
    RESULTS,
)
from laya_jp.sokudan_questions import DEPARTMENTS_EN, DEPARTMENTS_JA, URGENCY_EN, URGENCY_JA


def read_jsonl(name: str) -> list[dict]:
    path = RESULTS / name
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sokudan_block(rows: list[dict], departments: dict, urgency: list[str]) -> dict:
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["qtype"]].append(row)
    return {
        "choice": score_choice(by_type["choice"], list(departments)),
        "score": score_ordinal(by_type["score"], len(urgency)),
        "noul": score_noul(by_type["noul"]),
    }


def drift(block: dict) -> dict:
    published = PUBLISHED_BENCH_JA
    gaps = {
        "choice_accuracy": block["choice"]["accuracy"] - published["choice_accuracy"],
        "score_accuracy": block["score"]["accuracy"] - published["score_accuracy"],
        "noul_accuracy": block["noul"]["accuracy"] - published["noul_accuracy"],
        "noul_auroc": None if block["noul"]["auroc"] is None else block["noul"]["auroc"] - published["noul_auroc"],
    }
    flagged = [
        name for name, gap in gaps.items()
        if gap is not None and abs(gap) > DRIFT_TOLERANCE
    ]
    return {
        "tolerance": DRIFT_TOLERANCE,
        "gaps": {name: None if gap is None else round(gap, 6) for name, gap in gaps.items()},
        "exceeds_tolerance": flagged,
        "note": (
            "Published bench_ja numbers are CUDA fp32 on laya 0.3.4. "
            "This run is the pinned Hub snapshot on the local device. "
            "Prompts were not edited to close a gap."
        ),
    }


def rate(rows: list[dict], key: str = "consistent") -> dict:
    usable = [row for row in rows if row.get(key) is not None]
    if not usable:
        return {"n": 0, "rate": None}
    hits = sum(1 for row in usable if row[key])
    return {"n": len(usable), "rate": round(hits / len(usable), 6)}


def diagnostic_block(rows: list[dict]) -> dict:
    by_suite = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_suite[row["suite"]][row["probe"]].append(row)
    summary = {}
    for suite, probes in by_suite.items():
        summary[suite] = {}
        for probe, group in probes.items():
            item = rate(group)
            if probe.startswith("score_"):
                slot = [row["slot0"] for row in group if row.get("slot0") is not None]
                item["slot0_rate"] = round(sum(slot) / len(slot), 6) if slot else None
            if probe == "noul_labels_ab":
                deltas = [abs(row["delta_p_true"]) for row in group if row.get("delta_p_true") is not None]
                item["mean_abs_delta_p_true"] = round(sum(deltas) / len(deltas), 6) if deltas else None
            logits = [row.get("logits") or {} for row in group]
            item["logits_available"] = sum(1 for item_logits in logits if item_logits.get("available"))
            summary[suite][probe] = item
    return summary


def fmt(value) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def write_readme(payload: dict) -> None:
    ja = payload["bench_ja"]
    published = PUBLISHED_BENCH_JA
    snsk = payload["snsk"]["composite"]
    drift_names = ", ".join(payload["bench_ja_drift"]["exceeds_tolerance"]) or "none"
    lines = [
        "# Original laya-multilingual baseline",
        "",
        f"Checkpoint `{payload['environment'].get('model_id')}` at `{payload['environment'].get('revision')}`.",
        f"Device `{payload['environment'].get('device')}`, dtype `{payload['environment'].get('dtype')}`, laya `{payload['environment'].get('laya')}`.",
        "",
        "The published bench_ja column is CUDA fp32 on laya 0.3.4. The published 36.9 composite is MLX FP16, a different checkpoint. This file is the local PyTorch run.",
        "",
        "## bench_ja",
        "",
        "| metric | this run | published CUDA |",
        "| --- | ---: | ---: |",
        f"| choice accuracy | {fmt(ja['choice']['accuracy'])} | {published['choice_accuracy']:.3f} |",
        f"| score accuracy | {fmt(ja['score']['accuracy'])} | {published['score_accuracy']:.3f} |",
        f"| score slot 0 count | {ja['score']['slot0_count']} | {published['score_slot0_count']} |",
        f"| noul accuracy | {fmt(ja['noul']['accuracy'])} | {published['noul_accuracy']:.3f} |",
        f"| noul AUROC | {fmt(ja['noul']['auroc'])} | {published['noul_auroc']:.3f} |",
        f"| noul balanced accuracy | {fmt(ja['noul']['balanced_accuracy'])} | |",
        f"| score MAE argmax | {fmt(ja['score']['mae_argmax'])} | |",
        f"| score MAE expectation | {fmt(ja['score']['mae_expectation'])} | |",
        "",
        f"Accuracy gaps larger than {DRIFT_TOLERANCE:.2f}: {drift_names}.",
        "",
        "## bench_en",
        "",
        f"Choice accuracy {fmt(payload['bench_en']['choice']['accuracy'])}. "
        f"Score accuracy {fmt(payload['bench_en']['score']['accuracy'])}, slot 0 count {payload['bench_en']['score']['slot0_count']}. "
        f"Noul accuracy {fmt(payload['bench_en']['noul']['accuracy'])}, AUROC {fmt(payload['bench_en']['noul']['auroc'])}.",
        "",
        "## 40-question business bench",
        "",
        f"Composite {snsk['points']:.1f} / 100. "
        f"Bootstrap 95% interval over cases: {snsk['bootstrap']['percentile_2_5']:.1f} to {snsk['bootstrap']['percentile_97_5']:.1f}. "
        f"Repeats identical: {snsk['answers_identical_across_repeats']}. "
        f"Published MLX FP16 composite: {PUBLISHED_SNSK_MLX['composite']}.",
        "",
        "Per-type pass counts are in `snsk.summary.json`.",
        "",
        "## Diagnostics",
        "",
        "Consistency is the fraction of items whose semantic answer matches the batched baseline. Score slot 0 is how often the first presented level wins under that probe.",
        "",
        "On a 3-level score list, shuffling with seed 2026 is exactly the reversal, so `score_permuted` and `score_reversed` are the same probe on bench_ja and bench_en. Longer rubrics in the 40-question bench get a different order.",
        "",
    ]
    for suite, probes in payload["diagnostics"].items():
        lines.append(f"### {suite}")
        lines.append("")
        lines.append("| probe | n | consistency | slot 0 | mean abs delta P(true) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for probe, item in probes.items():
            lines.append(
                f"| {probe} | {item['n']} | {fmt(item['rate'])} | {fmt(item.get('slot0_rate'))} | {fmt(item.get('mean_abs_delta_p_true'))} |"
            )
        lines.append("")
    (RESULTS / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    environment = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
    bench_ja = sokudan_block(read_jsonl("bench_ja.predictions.jsonl"), DEPARTMENTS_JA, URGENCY_JA)
    bench_en = sokudan_block(read_jsonl("bench_en.predictions.jsonl"), DEPARTMENTS_EN, URGENCY_EN)
    snsk_summary = json.loads((RESULTS / "snsk.summary.json").read_text(encoding="utf-8"))
    # The evaluated list is large and already implied by the predictions file.
    snsk_summary.pop("evaluated", None)
    payload = {
        "environment": environment,
        "bench_ja": bench_ja,
        "bench_ja_drift": drift(bench_ja),
        "bench_en": bench_en,
        "snsk": {
            "composite": {
                "points": snsk_summary["total_score"]["points"],
                "answers_identical_across_repeats": snsk_summary["answers_identical_across_repeats"],
                "bootstrap": snsk_summary["bootstrap"],
                "published_mlx_fp16": PUBLISHED_SNSK_MLX["composite"],
            },
            "groups": snsk_summary["groups"],
            "valid": snsk_summary["valid"],
            "planned": snsk_summary["planned"],
        },
        "diagnostics": diagnostic_block(read_jsonl("diagnostics.jsonl")),
    }
    (RESULTS / "baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(payload)
    print(f"wrote {RESULTS / 'baseline.json'}", flush=True)
    if payload["bench_ja_drift"]["exceeds_tolerance"]:
        print("bench_ja drift vs published CUDA:", ", ".join(payload["bench_ja_drift"]["exceeds_tolerance"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
