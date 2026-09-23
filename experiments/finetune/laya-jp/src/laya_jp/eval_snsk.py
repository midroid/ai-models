"""Score the frozen 40-question bench with the vendored snsk scorer.

    USE_TF=0 uv run python -m laya_jp.eval_snsk
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

from laya_jp.lock import verify_lock
from laya_jp.pins import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, EXTERNAL, MODEL_ID, RESULTS
from laya_jp.predict import load_agent, predict, write_environment


def load_snsk():
    path = EXTERNAL / "snsk" / "bench.py"
    spec = importlib.util.spec_from_file_location("snsk_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _top(probs: dict) -> float | None:
    if not probs:
        return None
    return max(float(v) for v in probs.values())


def decision_rows(case: dict, repeat: int, response: dict, latency_ms: float, request: dict) -> list[dict]:
    rows = []
    answers = (response or {}).get("answers") or {}
    for qid, spec in case["input"]["questions"].items():
        answer = answers.get(qid) or {}
        qtype = spec["type"]
        row = {
            "suite": "snsk_jp_business",
            "item_id": case["id"],
            "repeat": repeat,
            "question_id": qid,
            "qtype": qtype,
            "call_group": "case",
            "group": case["group"],
            "request": request,
            "response": response,
            "latency_ms": round(latency_ms, 3),
            "confidence": answer.get("confidence"),
        }
        if qtype == "choice":
            probs = {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}
            row.update(probs=probs, p_top=_top(probs), pred=answer.get("choice"))
        elif qtype == "score":
            probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
            pred_index = max(probs, key=probs.get) if probs else None
            levels = spec["criteria"]
            row.update(
                probs=probs,
                p_top=_top(probs),
                pred_index=int(pred_index) if pred_index is not None else None,
                pred_label=levels[int(pred_index)] if pred_index is not None else None,
                expected_score=answer.get("score"),
            )
        else:
            p_true = answer.get("noul")
            row.update(
                p_true=None if p_true is None else float(p_true),
                pred=None if p_true is None else bool(float(p_true) >= 0.5),
                p_top=None if p_true is None else max(float(p_true), 1.0 - float(p_true)),
            )
        rows.append(row)
    return rows


def answers_match(records: list[dict]) -> bool:
    by_case: dict[str, list] = {}
    for record in records:
        if record.get("error"):
            return False
        by_case.setdefault(record["case_id"], []).append(record["response"]["answers"])
    for answers in by_case.values():
        first = json.dumps(answers[0], ensure_ascii=False, sort_keys=True)
        if any(json.dumps(other, ensure_ascii=False, sort_keys=True) != first for other in answers[1:]):
            return False
    return True


def bootstrap_composite(bench, questions, key, records, config) -> dict:
    latest = {(r["case_id"], r["repeat"]): r for r in records}
    passes = {}
    for case in questions["cases"]:
        hits = 0
        for repeat in range(1, config["repeats"] + 1):
            scored = bench.evaluate_case(case, key, latest.get((case["id"], repeat)), config)
            hits += int(bool(scored["passed"]))
        passes[case["id"]] = hits
    weights = {case["id"]: 1.5 if case["group"] == "composite" else 1.0 for case in questions["cases"]}
    ids = [case["id"] for case in questions["cases"]]
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        drawn = rng.choices(ids, k=len(ids))
        weight = sum(weights[i] for i in drawn)
        earned = sum(weights[i] * passes[i] / config["repeats"] for i in drawn)
        samples.append(100.0 * earned / weight)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "case",
        "percentile_2_5": round(lo, 2),
        "percentile_97_5": round(hi, 2),
    }


def main() -> int:
    verify_lock()
    RESULTS.mkdir(parents=True, exist_ok=True)
    bench = load_snsk()
    root = EXTERNAL / "snsk"
    questions = bench.load_yaml(root / "pilot.questions.yaml")
    key = bench.load_yaml(root / "pilot.answers.yaml")
    config = bench.load_config(root / "run_config.json")
    bench.validate_fixtures(questions, key, config)

    agent = load_agent()
    write_environment(agent)
    rng = random.Random(config["shuffle_seed"])
    records = []
    rows = []
    out = RESULTS / "snsk.predictions.jsonl"
    print(f"snsk: {len(questions['cases'])} cases x {config['repeats']} repeats", flush=True)
    with out.open("w", encoding="utf-8") as handle:
        for repeat in range(1, config["repeats"] + 1):
            cases = list(questions["cases"])
            rng.shuffle(cases)
            for case in cases:
                request = bench.payload_for(case, MODEL_ID)
                try:
                    response, latency_ms = predict(request["state"], request["questions"])
                    error = None
                except Exception as exc:
                    response, latency_ms, error = None, 0.0, f"{type(exc).__name__}: {exc}"
                record = {
                    "case_id": case["id"],
                    "repeat": repeat,
                    "elapsed_seconds": latency_ms / 1000.0,
                    "request": request,
                    "response": response,
                    "error": error,
                    "attempts": [],
                }
                records.append(record)
                if response is not None:
                    for row in decision_rows(case, repeat, response, latency_ms, request):
                        rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  {case['id']} repeat {repeat} {latency_ms:.0f} ms", flush=True)

    summary = bench.summarize(questions, key, records, config)
    summary["answers_identical_across_repeats"] = answers_match(records)
    summary["bootstrap"] = bootstrap_composite(bench, questions, key, records, config)
    summary["checkpoint"] = MODEL_ID
    (RESULTS / "snsk.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    points = summary["total_score"]["points"]
    print(f"snsk composite {points:.1f} / 100  identical_repeats={summary['answers_identical_across_repeats']}", flush=True)
    return 0 if summary["valid"] == summary["planned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
