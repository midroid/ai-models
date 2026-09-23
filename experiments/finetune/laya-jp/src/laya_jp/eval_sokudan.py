"""Run the original checkpoint on vendored bench_ja and bench_en.

    USE_TF=0 uv run python -m laya_jp.eval_sokudan
"""

from __future__ import annotations

import json
from pathlib import Path

from laya_jp.lock import verify_lock
from laya_jp.pins import EXTERNAL, RESULTS
from laya_jp.predict import load_agent, predict, write_environment
from laya_jp.sokudan_questions import (
    DEPARTMENTS_EN,
    DEPARTMENTS_JA,
    URGENCY_EN,
    URGENCY_JA,
    bench_questions_en,
    bench_questions_ja,
)


def load_items(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def _top(probs: dict) -> float | None:
    if not probs:
        return None
    return max(float(v) for v in probs.values())


def decision_rows(suite: str, item: dict, questions: dict, response: dict, latency_ms: float,
                  departments: dict, urgency: list[str]) -> list[dict]:
    answers = response.get("answers") or {}
    request = {"state": {"body": item["state"]}, "questions": questions}
    rows = []
    specs = (
        ("department", "choice", item["department"], departments),
        ("urgency", "score", item["urgency"], urgency),
        ("churn", "noul", bool(item["churn"]), None),
    )
    for qid, qtype, gold, space in specs:
        answer = answers.get(qid) or {}
        row = {
            "suite": suite,
            "item_id": item["item_id"],
            "question_id": qid,
            "qtype": qtype,
            "call_group": "all_three",
            "gold": gold,
            "request": request,
            "response": response,
            "latency_ms": round(latency_ms, 3),
            "confidence": answer.get("confidence"),
        }
        if qtype == "choice":
            probs = {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}
            row.update(pred=answer.get("choice"), probs=probs, p_top=_top(probs))
        elif qtype == "score":
            probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
            pred_index = max(probs, key=probs.get) if probs else None
            row.update(
                gold_index=int(gold),
                gold_label=space[int(gold)],
                pred_index=int(pred_index) if pred_index is not None else None,
                pred_label=space[int(pred_index)] if pred_index is not None else None,
                expected_score=answer.get("score"),
                probs=probs,
                p_top=_top(probs),
            )
        else:
            p_true = answer.get("noul")
            row.update(
                p_true=None if p_true is None else float(p_true),
                pred=None if p_true is None else bool(float(p_true) >= 0.5),
                probs={"false": None if p_true is None else 1.0 - float(p_true),
                       "true": None if p_true is None else float(p_true)},
                p_top=None if p_true is None else max(float(p_true), 1.0 - float(p_true)),
            )
        rows.append(row)
    return rows


def run_suite(name: str, path: Path, questions: dict, departments: dict, urgency: list[str]) -> Path:
    items = load_items(path)
    out = RESULTS / f"{name}.predictions.jsonl"
    print(f"{name}: {len(items)} items", flush=True)
    with out.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(items, start=1):
            response, latency_ms = predict({"body": item["state"]}, questions)
            for row in decision_rows(name, item, questions, response, latency_ms, departments, urgency):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index == 1 or index % 25 == 0 or index == len(items):
                print(f"  {index}/{len(items)} {latency_ms:.0f} ms", flush=True)
    return out


def main() -> int:
    verify_lock()
    RESULTS.mkdir(parents=True, exist_ok=True)
    agent = load_agent()
    env = write_environment(agent)
    print(f"loaded {env['model_id']} @ {env['revision']} on {env.get('device')} {env.get('dtype')}", flush=True)
    run_suite(
        "bench_ja",
        EXTERNAL / "sokudan" / "bench_ja.jsonl",
        bench_questions_ja(),
        DEPARTMENTS_JA,
        URGENCY_JA,
    )
    run_suite(
        "bench_en",
        EXTERNAL / "sokudan" / "bench_en.jsonl",
        bench_questions_en(),
        DEPARTMENTS_EN,
        URGENCY_EN,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
