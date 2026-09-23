"""Noul-label, score-order, and call-grouping probes on the frozen items.

Reads the baseline predictions and does not replace them.

    USE_TF=0 uv run python -m laya_jp.diagnostics
"""

from __future__ import annotations

import importlib.util
import json
import random

from laya_jp.eval_sokudan import load_items
from laya_jp.lock import verify_lock
from laya_jp.pins import EXTERNAL, RESULTS, SCORE_PERM_SEED
from laya_jp.predict import predict_with_logits
from laya_jp.sokudan_questions import bench_questions_en, bench_questions_ja


def load_snsk():
    path = EXTERNAL / "snsk" / "bench.py"
    spec = importlib.util.spec_from_file_location("snsk_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def baseline_index(path) -> dict[tuple, dict]:
    index = {}
    for row in read_jsonl(path):
        index[(row["item_id"], row.get("repeat", 1), row["question_id"])] = row
    return index


def trim_logits(captured: dict) -> dict:
    if not captured.get("available"):
        return {"available": False, "reason": captured.get("reason", "unavailable")}
    logits = captured.get("logits") or []
    mask = captured.get("marker_mask")
    trimmed = []
    for row, values in enumerate(logits):
        width = len(values)
        if mask is not None and row < len(mask):
            width = int(sum(1 for bit in mask[row] if bit))
        trimmed.append([round(float(v), 4) for v in values[:width]])
    return {"available": True, "marker_logits": trimmed}


def presented_score_label(answer: dict, levels: list[str]) -> str | None:
    probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
    if not probs:
        return None
    index = int(max(probs, key=probs.get))
    if index >= len(levels):
        return None
    return levels[index]


def noul_choice(spec: dict, true_text: str, false_text: str) -> dict:
    return {
        "type": "choice",
        "instructions": spec["instructions"],
        "criteria": {"A": true_text, "B": false_text},
    }


def permute(levels: list[str]) -> list[str]:
    order = list(levels)
    random.Random(SCORE_PERM_SEED).shuffle(order)
    return order


def write_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def probe_questions(handle, suite: str, item_id: str, state, questions: dict, baseline: dict,
                    true_text: str, false_text: str, repeat: int = 1) -> None:
    batched_ids = list(questions)
    for qid in batched_ids:
        spec = questions[qid]
        base = baseline.get((item_id, repeat, qid))
        single_questions = {qid: spec}
        response, latency_ms, captured = predict_with_logits(state, single_questions)
        answer = (response.get("answers") or {}).get(qid, {})
        row = {
            "suite": suite,
            "item_id": item_id,
            "repeat": repeat,
            "probe": "single_question",
            "question_id": qid,
            "qtype": spec["type"],
            "latency_ms": round(latency_ms, 3),
            "logits": trim_logits(captured),
            "answer": answer,
        }
        if spec["type"] == "score":
            label = presented_score_label(answer, spec["criteria"])
            row["pred_label"] = label
            row["consistent"] = base is not None and label == base.get("pred_label")
        elif spec["type"] == "noul":
            p_true = answer.get("noul")
            pred = None if p_true is None else bool(float(p_true) >= 0.5)
            row["p_true"] = None if p_true is None else float(p_true)
            row["pred"] = pred
            row["consistent"] = base is not None and pred == base.get("pred")
        else:
            row["pred"] = answer.get("choice")
            row["consistent"] = base is not None and row["pred"] == base.get("pred")
        write_row(handle, row)

        if spec["type"] == "noul":
            labeled = dict(spec)
            labeled["labels"] = {"false": "B", "true": "A"}
            response, latency_ms, captured = predict_with_logits(state, {qid: labeled})
            answer = (response.get("answers") or {}).get(qid, {})
            p_true = answer.get("noul")
            pred = None if p_true is None else bool(float(p_true) >= 0.5)
            base_p = None if base is None else base.get("p_true")
            write_row(handle, {
                "suite": suite,
                "item_id": item_id,
                "repeat": repeat,
                "probe": "noul_labels_ab",
                "question_id": qid,
                "qtype": "noul",
                "p_true": None if p_true is None else float(p_true),
                "pred": pred,
                "baseline_p_true": base_p,
                "delta_p_true": None if p_true is None or base_p is None else round(float(p_true) - float(base_p), 4),
                "consistent": base is not None and pred == base.get("pred"),
                "latency_ms": round(latency_ms, 3),
                "logits": trim_logits(captured),
            })
            choice = noul_choice(spec, true_text, false_text)
            response, latency_ms, captured = predict_with_logits(state, {qid: choice})
            answer = (response.get("answers") or {}).get(qid, {})
            probs = {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}
            chosen = answer.get("choice")
            write_row(handle, {
                "suite": suite,
                "item_id": item_id,
                "repeat": repeat,
                "probe": "noul_choice_ab",
                "question_id": qid,
                "qtype": "choice",
                "pred": chosen,
                "p_true": probs.get("A"),
                "probs": probs,
                "consistent": base is not None and (chosen == "A") == bool(base.get("pred")),
                "latency_ms": round(latency_ms, 3),
                "logits": trim_logits(captured),
            })

        if spec["type"] == "score":
            for probe, levels in (
                ("score_reversed", list(reversed(spec["criteria"]))),
                ("score_permuted", permute(spec["criteria"])),
            ):
                variant = dict(spec)
                variant["criteria"] = levels
                response, latency_ms, captured = predict_with_logits(state, {qid: variant})
                answer = (response.get("answers") or {}).get(qid, {})
                label = presented_score_label(answer, levels)
                probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
                slot0 = int(max(probs, key=probs.get)) == 0 if probs else None
                write_row(handle, {
                    "suite": suite,
                    "item_id": item_id,
                    "repeat": repeat,
                    "probe": probe,
                    "question_id": qid,
                    "qtype": "score",
                    "presented": levels,
                    "pred_label": label,
                    "baseline_pred_label": None if base is None else base.get("pred_label"),
                    "consistent": base is not None and label == base.get("pred_label"),
                    "slot0": slot0,
                    "latency_ms": round(latency_ms, 3),
                    "logits": trim_logits(captured),
                })


def run_sokudan(handle, suite: str, items_path, questions: dict, true_text: str, false_text: str) -> None:
    baseline = baseline_index(RESULTS / f"{suite}.predictions.jsonl")
    items = load_items(items_path)
    print(f"diagnostics {suite}: {len(items)}", flush=True)
    for index, item in enumerate(items, start=1):
        probe_questions(
            handle, suite, item["item_id"], {"body": item["state"]}, questions,
            baseline, true_text, false_text,
        )
        if index == 1 or index % 25 == 0 or index == len(items):
            print(f"  {index}/{len(items)}", flush=True)


def run_snsk(handle) -> None:
    bench = load_snsk()
    questions = bench.load_yaml(EXTERNAL / "snsk" / "pilot.questions.yaml")
    baseline = baseline_index(RESULTS / "snsk.predictions.jsonl")
    cases = questions["cases"]
    print(f"diagnostics snsk: {len(cases)}", flush=True)
    for index, case in enumerate(cases, start=1):
        payload = dict(case["input"]["questions"])
        true_text, false_text = "yes, the statement holds", "no, the statement does not hold"
        for spec in payload.values():
            if spec["type"] == "noul":
                crit = spec.get("criteria") or {}
                true_text = crit.get("true") or true_text
                false_text = crit.get("false") or false_text
        probe_questions(
            handle, "snsk_jp_business", case["id"], case["input"]["state"], payload,
            baseline, true_text, false_text, repeat=1,
        )
        if index == 1 or index % 10 == 0 or index == len(cases):
            print(f"  {index}/{len(cases)}", flush=True)


def main() -> int:
    verify_lock()
    for name in ("bench_ja.predictions.jsonl", "bench_en.predictions.jsonl", "snsk.predictions.jsonl"):
        if not (RESULTS / name).is_file():
            raise SystemExit(f"missing baseline {name}; run eval_sokudan and eval_snsk first")
    out = RESULTS / "diagnostics.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        run_sokudan(
            handle, "bench_ja", EXTERNAL / "sokudan" / "bench_ja.jsonl", bench_questions_ja(),
            "送信者は解約・契約終了を示唆している", "送信者は解約・契約終了を示唆していない",
        )
        run_sokudan(
            handle, "bench_en", EXTERNAL / "sokudan" / "bench_en.jsonl", bench_questions_en(),
            "the sender is hinting that they may stop using the service",
            "the sender is not hinting that they may stop using the service",
        )
        run_snsk(handle)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
