#!/usr/bin/env python3
"""Run the frozen Jev fixtures and render an offline report. No API keys in artifacts."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
from decimal import Decimal
import hashlib
import html
import json
import math
import operator
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parent
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
TYPES = {"choice": "Choice", "score": "Score", "noul": "Noul", "composite": "複合"}
OPS = {"==": operator.eq, "!=": operator.ne, ">=": operator.ge, "<=": operator.le,
       ">": operator.gt, "<": operator.lt}
RULE = re.compile(r"([a-z_]+)\.(choice|score|noul)\s*(!=|==|>=|<=|>|<)\s*(?:'([^']*)'|(\d+(?:\.\d+)?))")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    value = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in value:
            raise ValueError(f"YAMLに重複キーがあります: {key}")
        value[key] = loader.construct_object(value_node, deep=deep)
    return value


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_yaml(path):
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueLoader)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in ("repeats", "timeout_seconds", "max_retries", "shuffle_seed"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"設定値が不正です: {name}")
    if not 1 <= config["repeats"] <= 100 or not 1 <= config["timeout_seconds"] <= 60:
        raise ValueError("repeatsは1〜100、timeout_secondsは1〜60にしてください。")
    if config["max_retries"] > 5:
        raise ValueError("max_retriesは5以下にしてください。")
    if config["score_tolerance"] != 0.5 or config["noul_threshold"] != 0.5:
        raise ValueError("この版では採点原本に合わせて閾値を0.5に固定しています。")
    if not isinstance(config["model"], str) or not config["model"].startswith("jev-"):
        raise ValueError("TypeSafeのJevモデル名を指定してください。")
    for name in ("price_usd_per_million_input_tokens", "price_usd_per_million_output_tokens"):
        if not number(config[name]) or config[name] < 0:
            raise ValueError(f"単価が不正です: {name}")
    return config


def get_key(path):
    # Deliberately not a shell source operation: no interpolation or command execution.
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key and Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.partition("=")[2].strip()
                if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
                    key = key[1:-1]
    if key.lower() in {"", "your_api_key", "your-key-here", "replace_me"}:
        raise ValueError(f"APIキーが未設定です。{Path(path).resolve()} の TYPESAFE_API_KEY= に設定してください。")
    if any(c.isspace() for c in key):
        raise ValueError("APIキーに空白が含まれています。設定ファイルを確認してください。")
    return key


def redact(value, key):
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, dict):
        return {redact(k, key): redact(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(x, key) for x in value]
    return value


def payload_for(case, model):
    return {"model": model, "state": case["input"]["state"], "questions": case["input"]["questions"]}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def rounding_interval(value):
    # A declared evaluation-side assumption, not a claim about provider internals.
    # At most 0.005 of rounding, narrowed when a returned value has more precision.
    digits = max(2, -Decimal(str(value)).as_tuple().exponent)
    half = 0.5 * 10 ** -digits
    return value - half, value + half


def probability_bounds(probs):
    return {k: (max(0, rounding_interval(p)[0]), min(1, rounding_interval(p)[1])) for k,p in probs.items()}


def weighted_bounds(bounds):
    # Extremal means of all normalized distributions inside the rounding intervals.
    base = sum(int(k) * lo for k,(lo,hi) in bounds.items())
    remaining = 1 - sum(lo for lo,hi in bounds.values())
    extrema = []
    for reverse in (False, True):
        weight, value = remaining, base
        for k in sorted(bounds, key=int, reverse=reverse):
            lo, hi = bounds[k]
            allocated = min(max(0, weight), hi - lo)
            value += int(k) * allocated
            weight -= allocated
        extrema.append(value)
    return extrema


def validate_answer(spec, answer):
    if not isinstance(answer, dict) or answer.get("type") != spec["type"]:
        return "回答のtypeが設問と一致しません。"
    kind = spec["type"]
    if kind == "noul":
        return None if number(answer.get("noul")) and 0 <= answer["noul"] <= 1 else "noulが0〜1の有限数ではありません。"
    if not number(answer.get("confidence")) or not 0 <= answer["confidence"] <= 1:
        return "confidenceが0〜1の有限数ではありません。"
    expected_keys = set(spec["criteria"]) if kind == "choice" else {str(i) for i in range(len(spec["criteria"]))}
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != expected_keys:
        return "確率分布のキーが選択肢・段階と一致しません。"
    if not all(number(p) and 0 <= p <= 1 for p in probs.values()):
        return "確率分布の範囲または合計が不正です。"
    bounds = probability_bounds(probs)
    if sum(lo for lo,hi in bounds.values()) > 1 + 1e-9 or sum(hi for lo,hi in bounds.values()) < 1 - 1e-9:
        return "丸め誤差を考慮しても確率の合計が1になりません。"
    if kind == "choice":
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in expected_keys:
            return "choiceが選択肢に含まれません。"
        if max(probs.values()) - probs[choice] > 0.00001:
            return "choiceが最大確率の選択肢ではありません。"
    else:
        score = answer.get("score")
        if not number(score) or not 0 <= score <= len(spec["criteria"]) - 1:
            return "scoreが段階の範囲内の有限数ではありません。"
        minimum, maximum = weighted_bounds(bounds)
        score_lo, score_hi = rounding_interval(score)
        if score_hi < minimum - 1e-9 or score_lo > maximum + 1e-9:
            return "丸め誤差を考慮してもscoreと確率加重平均が一致しません。"
        legend = answer.get("legend")
        if not isinstance(legend, dict) or set(legend) != expected_keys:
            return "legendの段階が設問と一致しません。"
    return None


def numerical_notes(spec, answer):
    if validate_answer(spec, answer) or spec["type"] == "noul":
        return []
    notes = []
    probs = answer["probabilities"]
    if abs(sum(probs.values()) - 1) > 1e-9:
        notes.append(f"表示確率の合計は{sum(probs.values()):.6f}です。丸めを仮定すると合計1と整合します。")
    if spec["type"] == "score":
        computed = sum(int(i) * p for i,p in probs.items())
        if abs(answer["score"] - computed) > 1e-9:
            notes.append(f"返却Score={answer['score']:.4f}、表示確率の加重平均={computed:.4f}。差は丸めを仮定した整合範囲内です。採点には返却Scoreを使用しました。")
    return notes


def score_answer(spec, gold, answer, config):
    error = validate_answer(spec, answer)
    result = {"type": spec["type"], "valid": error is None, "passed": False, "error": error}
    if error:
        return result
    result["notes"] = numerical_notes(spec, answer)
    if spec["type"] == "choice":
        result.update(passed=answer["choice"] == gold["expected_choice"], predicted=answer["choice"],
                      expected=gold["expected_choice"], nll=-math.log(max(answer["probabilities"][gold["expected_choice"]], 1e-12)))
    elif spec["type"] == "score":
        error = abs(answer["score"] - gold["target_level"])
        result.update(passed=error <= config["score_tolerance"], predicted=answer["score"],
                      expected=gold["target_level"], absolute_error=error)
    else:
        predicted = answer["noul"] >= config["noul_threshold"]
        result.update(passed=predicted == gold["expected_boolean"], predicted=predicted,
                      expected=gold["expected_boolean"], probability=answer["noul"],
                      brier=(answer["noul"] - int(gold["expected_boolean"])) ** 2)
    return result


def compose(composition, answers):
    # A tiny comparison grammar; never eval instructions or YAML as Python code.
    for rule in composition["ordered_rules"]:
        if rule["action"] == "invalid":
            continue
        if rule["when"] == "otherwise":
            return rule["action"]
        match = RULE.fullmatch(rule["when"])
        if not match:
            raise ValueError("未対応の合成ルール: " + rule["when"])
        question, field, op, string_value, numeric = match.groups()
        right = string_value if string_value is not None else float(numeric)
        if OPS[op](answers[question][field], right):
            return rule["action"]
    raise ValueError("合成ルールに既定アクションがありません。")


def validate_fixtures(questions, key, config):
    if questions["benchmark_id"] != key["benchmark_id"] or questions["version"] != key["version"]:
        raise ValueError("設問と想定解の版が一致しません。")
    cases = questions["cases"]
    if len({c["id"] for c in cases}) != len(cases) or {c["id"] for c in cases} != set(key["answers"]):
        raise ValueError("設問IDと想定解IDが一致しないか、重複しています。")
    if collections.Counter(c["group"] for c in cases) != questions["pilot_case_counts"]:
        raise ValueError("型ごとの設問数がメタデータと一致しません。")
    if sum(len(c["input"]["questions"]) for c in cases) != questions["atomic_question_count"]:
        raise ValueError("型付き判断の総数が一致しません。")
    if set(key["compositions"]) != {c["id"] for c in cases if c["group"] == "composite"}:
        raise ValueError("複合ケースの合成ルールが一致しません。")
    for case in cases:
        if set(case["input"]) != {"state", "questions"}:
            raise ValueError("inputにstate/questions以外のデータがあります。")
        specs = case["input"]["questions"]
        if set(specs) != set(key["answers"][case["id"]]):
            raise ValueError("質問IDと想定解が一致しません: " + case["id"])
        if len(specs) != (3 if case["group"] == "composite" else 1):
            raise ValueError("ケース内の質問数が不正です。")
        gold_values = {}
        for qid, spec in specs.items():
            gold = key["answers"][case["id"]][qid]
            if set(spec) != {"type", "instructions", "criteria"} or spec["type"] not in {"choice", "score", "noul"}:
                raise ValueError("質問のスキーマが不正です。")
            if not isinstance(spec["instructions"], str) or not spec["instructions"].strip():
                raise ValueError("instructionsが空です。")
            kind = spec["type"]
            if case["group"] != "composite" and case["group"] != kind:
                raise ValueError("ケースと質問の型が一致しません。")
            if kind == "choice":
                if not isinstance(spec["criteria"], dict) or not 2 <= len(spec["criteria"]) <= 255 or gold["expected_choice"] not in spec["criteria"]:
                    raise ValueError("Choiceの想定解・選択肢が不正です。")
                gold_values[qid] = {"choice": gold["expected_choice"]}
            elif kind == "score":
                if not isinstance(spec["criteria"], list) or not 2 <= len(spec["criteria"]) <= 10 or type(gold["target_level"]) is not int or not 0 <= gold["target_level"] < len(spec["criteria"]):
                    raise ValueError("Scoreの段階・想定解が不正です。")
                gold_values[qid] = {"score": gold["target_level"]}
            else:
                if set(spec["criteria"]) != {"true", "false"} or type(gold["expected_boolean"]) is not bool:
                    raise ValueError("Noulの基準・想定解が不正です。")
                gold_values[qid] = {"noul": float(gold["expected_boolean"])}
        if case["group"] == "composite":
            composition = key["compositions"][case["id"]]
            if collections.Counter(s["type"] for s in specs.values()) != {"choice": 1, "score": 1, "noul": 1}:
                raise ValueError("複合内の型の構成が不正です。")
            for rule in composition["ordered_rules"][1:]:
                if rule["when"] != "otherwise":
                    match = RULE.fullmatch(rule["when"])
                    if not match or match[1] not in gold_values or match[2] not in gold_values[match[1]]:
                        raise ValueError("合成ルールの参照が不正です。")
            if compose(composition, gold_values) != composition["expected_action"]:
                raise ValueError("想定解から合成したアクションが一致しません。")


def evaluate_case(case, key, record, config):
    result = {"case_id": case["id"], "group": case["group"], "repeat": record["repeat"] if record else None,
              "status": "not_run", "passed": False, "questions": {}}
    if not record:
        return result
    if record.get("error"):
        result.update(status="api_error", error=record["error"])
        return result
    response = record.get("response")
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        result.update(status="invalid", error="answersがありません。")
        return result
    specs = case["input"]["questions"]
    if set(response["answers"]) != set(specs):
        result.update(status="invalid", error="応答とリクエストの質問IDが一致しません。")
        return result
    for qid, spec in specs.items():
        result["questions"][qid] = score_answer(spec, key["answers"][case["id"]][qid], response["answers"][qid], config)
    valid = all(x["valid"] for x in result["questions"].values())
    passed = valid and all(x["passed"] for x in result["questions"].values())
    result.update(status=("pass" if passed else "fail") if valid else "invalid", passed=passed)
    if case["group"] == "composite":
        result["atomic_pass"] = passed
        result["derived_action"] = compose(key["compositions"][case["id"]], response["answers"]) if valid else "invalid"
        result["action_pass"] = valid and result["derived_action"] == key["compositions"][case["id"]]["expected_action"]
        result["passed"] = passed and result["action_pass"]
        if valid:
            result["status"] = "pass" if result["passed"] else "fail"
    return result


def mean(values):
    return statistics.mean(values) if values else None


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * quantile
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def latest_records(records):
    return {(r["case_id"], r["repeat"]): r for r in records}


def summarize(questions, key, records, config):
    latest = latest_records(records)
    evaluated = [evaluate_case(c, key, latest.get((c["id"], rep)), config)
                 for c in questions["cases"] for rep in range(1, config["repeats"] + 1)]
    groups = {}
    for kind in TYPES:
        rows = [r for r in evaluated if r["group"] == kind]
        valid = [r for r in rows if r["status"] in {"pass", "fail"}]
        atomics = [x for r in valid for x in r["questions"].values()]
        item = {"planned": len(rows), "valid": len(valid), "passed": sum(r["passed"] for r in rows),
                "api_errors": sum(r["status"] == "api_error" for r in rows),
                "invalid": sum(r["status"] == "invalid" for r in rows),
                "not_run": sum(r["status"] == "not_run" for r in rows)}
        item["pass_rate_valid"] = item["passed"] / len(valid) if valid else None
        if kind == "choice":
            item["nll"] = mean([x["nll"] for x in atomics])
        elif kind == "score":
            item["mae"] = mean([x["absolute_error"] for x in atomics])
        elif kind == "noul":
            item["brier"] = mean([x["brier"] for x in atomics])
        elif kind == "composite":
            item["atomic_passed"] = sum(r.get("atomic_pass", False) for r in valid)
            item["action_passed"] = sum(r.get("action_pass", False) for r in valid)
        groups[kind] = item
    times = [r["elapsed_seconds"] for r in latest.values() if not r.get("error")]
    usage_in, usage_out, missing_usage = 0, 0, 0
    models = set()
    for record in records:
        response = record.get("response") or {}
        if not record.get("error"):
            if isinstance(response.get("model"), str):
                models.add(response["model"])
            usage = response.get("usage", {})
            if isinstance(usage, dict) and type(usage.get("input_tokens")) is int and type(usage.get("output_tokens")) is int and usage["input_tokens"] >= 0 and usage["output_tokens"] >= 0:
                usage_in += usage["input_tokens"]
                usage_out += usage["output_tokens"]
            else:
                missing_usage += 1
    stable, eligible = 0, 0
    for case in questions["cases"]:
        rows = [r for r in evaluated if r["case_id"] == case["id"]]
        if config["repeats"] < 2 or any(r["status"] not in {"pass", "fail"} for r in rows):
            continue
        for qid, spec in case["input"]["questions"].items():
            values = [r["questions"][qid]["predicted"] for r in rows]
            if spec["type"] == "score":
                values = [math.floor(v + 0.5) for v in values]
            eligible += 1
            stable += len(set(values)) == 1
    estimated_cost = (usage_in * config["price_usd_per_million_input_tokens"] + usage_out * config["price_usd_per_million_output_tokens"]) / 1e6
    note_count = sum(bool(s.get("notes")) for r in evaluated for s in r["questions"].values())
    total_weight = sum(1.5 if c["group"] == "composite" else 1.0 for c in questions["cases"])
    earned_weight = math.fsum((1.5 if r["group"] == "composite" else 1.0) * r["passed"] / config["repeats"] for r in evaluated)
    total_points = 100 * earned_weight / total_weight if records else None
    return {"groups": groups, "evaluated": evaluated, "planned": len(evaluated), "numerical_note_count": note_count,
            "total_score": {"points": total_points, "maximum": 100, "single_weight": 1, "composite_weight": 1.5,
                            "weight_sum": total_weight, "earned_weight": earned_weight,
                            "provisional": any(r["status"] in {"not_run", "api_error", "invalid"} for r in evaluated),
                            "formula": "100 * sum(case_weight * case_pass_count / repeats) / sum(case_weight)"},
            "recorded": len(latest), "valid": sum(r["status"] in {"pass", "fail"} for r in evaluated),
            "observed_models": sorted(models), "latency_median": mean([]) if not times else statistics.median(times),
            "latency_p95": percentile(times, .95), "latency_count": len(times),
            "input_tokens": usage_in, "output_tokens": usage_out, "missing_usage": missing_usage,
            "estimated_cost_usd": estimated_cost if records and not missing_usage else None,
            "cost_observed_tokens_usd": estimated_cost, "stable_questions": stable, "stability_eligible": eligible,
            "http_attempts": sum(len(r.get("attempts", [])) for r in records),
            "retry_attempts": sum(max(0, len(r.get("attempts", [])) - 1) for r in records)}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def call_api(payload, api_key, config):
    opener = urllib.request.build_opener(NoRedirect)
    started, attempts = time.perf_counter(), []
    result = {"started_at": now(), "attempts": attempts}
    for attempt in range(config["max_retries"] + 1):
        tick = time.perf_counter()
        request = urllib.request.Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + api_key,
                                                  "Content-Type": "application/json", "User-Agent": "jev-jp-benchmark/0.1"}, method="POST")
        retry, retry_after = False, 0
        try:
            with opener.open(request, timeout=config["timeout_seconds"]) as response:
                raw = response.read(2_000_001)
                status = response.status
            if len(raw) > 2_000_000:
                raise ValueError("response_size")
            data = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite_json")))
            if not isinstance(data, dict):
                raise ValueError("response_shape")
            attempts.append({"status": status, "elapsed_seconds": time.perf_counter() - tick})
            result["response"] = redact(data, api_key)
            result["error"] = None
            break
        except urllib.error.HTTPError as exc:
            # Never print/store request headers, credentials, or arbitrary server error bodies.
            attempts.append({"status": exc.code, "elapsed_seconds": time.perf_counter() - tick})
            result["error"] = {"kind": "http", "status": exc.code, "message": f"HTTP {exc.code}"}
            retry = exc.code in {429, 500, 502, 503, 504, 529}
            header = exc.headers.get("Retry-After", "")
            if header.isdigit():
                retry_after = min(int(header), 30)
            exc.close()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            attempts.append({"status": "network_error", "elapsed_seconds": time.perf_counter() - tick})
            result["error"] = {"kind": "network", "message": f"ネットワーク接続に失敗しました ({type(exc).__name__})"}
            # Network errors may have happened after dispatch; do not silently replay billed requests.
            break
        except (ValueError, UnicodeError):
            attempts.append({"status": "invalid_json", "elapsed_seconds": time.perf_counter() - tick})
            result["error"] = {"kind": "protocol", "message": "応答が有効なJSONオブジェクトではありません。"}
            break
        if not retry or attempt == config["max_retries"]:
            break
        time.sleep(max(retry_after, min(2 ** attempt, 8)))
    result["elapsed_seconds"] = time.perf_counter() - started
    result["finished_at"] = now()
    return result


def load_records(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def h(value):
    return html.escape(str(value), quote=True)


def f(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def percentage(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def expected_text(spec, gold):
    if spec["type"] == "choice":
        choice = gold["expected_choice"]
        return choice + " — " + spec["criteria"][choice]
    if spec["type"] == "score":
        level = gold["target_level"]
        return f"段階 {level} — " + spec["criteria"][level]
    return str(gold["expected_boolean"]).lower()


def value_text(scored):
    if not scored or not scored.get("valid"):
        return "—"
    if scored["type"] == "noul":
        return f"P(true)={scored['probability']:.4f} → {str(scored['predicted']).lower()}"
    if scored["type"] == "score":
        return f"{scored['predicted']:.4f}（誤差 {scored['absolute_error']:.4f}）"
    return scored["predicted"]


def answer_of(record, qid):
    response = (record or {}).get("response")
    answers = response.get("answers") if isinstance(response, dict) else None
    answer = answers.get(qid) if isinstance(answers, dict) else None
    return answer if isinstance(answer, dict) else {}


def probability_html(spec, answer):
    if validate_answer(spec, answer):
        return ""
    probs = {"true": answer["noul"], "false": 1 - answer["noul"]} if spec["type"] == "noul" else answer["probabilities"]
    rows = []
    for label, p in probs.items():
        rows.append(f'<div class="prob-row"><span>{h(label)}</span><span class="prob-track" aria-hidden="true"><span style="width:{p*100:.3f}%"></span></span><strong>{p:.4f}</strong></div>')
    return '<div class="probabilities">' + "".join(rows) + "</div>"


def render_report(questions, key, config, manifest, records, output):
    summary = summarize(questions, key, records, config)
    latest = latest_records(records)
    complete = summary["recorded"] == summary["planned"] and summary["valid"] == summary["planned"]
    state = "実測完了" if complete else ("一部実測・未完了" if records else "未実行・APIキー設定待ち")
    metric_defs = [("choice", "正解率", "pass_rate_valid", "percent"), ("score", "平均絶対誤差", "mae", "number"),
                   ("noul", "Brierスコア", "brier", "number"), ("composite", "完全一致率", "pass_rate_valid", "percent")]
    metrics = []
    for kind, label, name, fmt in metric_defs:
        group = summary["groups"][kind]
        val = percentage(group.get(name)) if fmt == "percent" else f(group.get(name), 4)
        foot = "高いほど良い" if fmt == "percent" else "小さいほど良い・理想値0"
        metrics.append(f'<article class="metric {kind}"><p class="eyebrow">{TYPES[kind]}</p><h2>{label}</h2><p class="metric-value">{val}</p><p>{foot}</p><small>有効 {group["valid"]} / 予定 {group["planned"]} 回</small></article>')
    group_rows = []
    for kind, group in summary["groups"].items():
        extra = ("NLL " + f(group.get("nll"), 4)) if kind == "choice" else (
            "許容差内 " + str(group["passed"]) + " 回" if kind == "score" else (
                "分類正解率 " + percentage(group["pass_rate_valid"]) if kind == "noul" else f'個別判断一致 {group.get("atomic_passed", 0)}・最終対応一致 {group.get("action_passed", 0)} 回'))
        group_rows.append(f'<tr><th scope="row">{TYPES[kind]}</th><td>{group["planned"]}</td><td>{group["valid"]}</td><td>{group["passed"]}</td><td>{group["api_errors"]}</td><td>{group["invalid"]}</td><td>{group["not_run"]}</td><td>{extra}</td></tr>')
    case_html, issue_links = [], []
    all_pass = all_fail = partial = pending = 0
    for case in questions["cases"]:
        trials = [evaluate_case(case, key, latest.get((case["id"], rep)), config) for rep in range(1, config["repeats"] + 1)]
        valid_trials = [r for r in trials if r["status"] in {"pass", "fail"}]
        passed = sum(r["passed"] for r in trials)
        if all(r["status"] == "not_run" for r in trials):
            outcome, label = "pending", "未実行"
            pending += 1
        elif any(r["status"] in {"invalid", "api_error", "not_run"} for r in trials):
            outcome, label = "incomplete", "未完了 / エラーあり"
            partial += 1
        elif passed == len(trials):
            outcome, label = "pass", "全試行合格"
            all_pass += 1
        else:
            outcome, label = "fail", "不合格あり"
            all_fail += 1
        if outcome in {"fail", "incomplete"}:
            issue_links.append(f'<li><a href="#case-{case["id"]}"><b>{case["id"]}</b> {h(case["title"])}</a><span>{passed}/{config["repeats"]} 合格</span></li>')
        parts = [f'<details class="case" id="case-{case["id"]}" data-kind="{case["group"]}" data-outcome="{outcome}"><summary><span class="case-id">{case["id"]}</span><span class="case-title">{h(case["title"])}<small>{h(case["japan_focus"])}</small></span><span class="badge {outcome}">{label}</span><span class="case-count">{passed}/{config["repeats"]}</span></summary><div class="case-body">',
                 '<h3>提示した状況とルール</h3><div class="state-text">' + h(case["input"]["state"]) + '</div>']
        for qid, spec in case["input"]["questions"].items():
            gold = key["answers"][case["id"]][qid]
            parts.append(f'<section class="question"><h3>{TYPES[spec["type"]]} · {h(qid)}</h3><p class="instruction">{h(spec["instructions"])}</p><p class="expected"><b>想定解</b> {h(expected_text(spec, gold))}</p><p><b>作問者の採点根拠：</b>{h(gold["rationale"])}</p>')
            criteria = spec["criteria"].items() if isinstance(spec["criteria"], dict) else enumerate(spec["criteria"])
            parts.append('<details class="nested"><summary>選択肢・判断基準を見る</summary><dl>' + ''.join(f'<dt>{h(k)}</dt><dd>{h(v)}</dd>' for k,v in criteria) + '</dl></details>')
            parts.append('<div class="table-scroll" tabindex="0"><table><caption>試行別の実測値</caption><thead><tr><th scope="col">試行</th><th scope="col">実測値</th><th scope="col">判定</th><th scope="col">confidence</th></tr></thead><tbody>')
            for rep, trial in enumerate(trials, 1):
                record = latest.get((case["id"], rep))
                scored = trial["questions"].get(qid)
                answer = answer_of(record, qid)
                judgement = ("合格" if scored["passed"] else "不合格") if scored and scored.get("valid") else {"not_run":"未実行", "api_error":"APIエラー"}.get(trial["status"], "回答不正")
                confidence = f(answer.get("confidence"), 4) if scored and scored.get("valid") and spec["type"] != "noul" else "—"
                parts.append(f'<tr><th scope="row">{rep}</th><td>{h(value_text(scored))}</td><td>{judgement}</td><td>{confidence}</td></tr>')
            parts.append('</tbody></table></div>')
            for rep, trial in enumerate(trials, 1):
                scored = trial["questions"].get(qid, {})
                for note in scored.get("notes", []):
                    parts.append(f'<p class="note">試行{rep}・数値整合性の注記：{h(note)}</p>')
                if scored.get("error"):
                    parts.append(f'<p class="note">試行{rep}・回答不正：{h(scored["error"])}</p>')
            for rep in range(1, config["repeats"] + 1):
                record = latest.get((case["id"], rep))
                answer = answer_of(record, qid)
                if answer:
                    parts.append(f'<details class="nested"><summary>試行{rep}の確率分布</summary>{probability_html(spec, answer)}</details>')
            parts.append('</section>')
        if case["group"] == "composite":
            composition = key["compositions"][case["id"]]
            parts.append('<section class="composition"><h3>合成した最終対応</h3><p class="expected">想定：' + h(composition["expected_action"]) + '</p><p class="state-text">' + h(composition["expected_business_behavior"]) + '</p><ul>')
            parts.extend(f'<li>試行{i}: {h(t.get("derived_action", "未判定"))}</li>' for i,t in enumerate(trials,1))
            parts.append('</ul></section>')
        for rep in range(1, config["repeats"] + 1):
            record = latest.get((case["id"], rep))
            if record:
                parts.append(f'<details class="nested raw"><summary>試行{rep}のAPI応答・計測記録（{f(record["elapsed_seconds"])}秒）</summary><pre tabindex="0"><code>{h(json.dumps({k:v for k,v in record.items() if k != "request"},ensure_ascii=False,indent=2))}</code></pre></details>')
        parts.append('</div></details>')
        case_html.append(''.join(parts))
    if not records:
        lead = "実測値はまだありません。APIキーを設定して実行すると、このレポートに40問の結果が反映されます。"
    else:
        lead = f'{len(questions["cases"])}ケースを各{config["repeats"]}回評価。全試行合格は{all_pass}ケース、不合格があるものは{all_fail}ケース、未完了・未実行は{partial + pending}ケースです。'
    issues = '<ul class="issue-list">' + ''.join(issue_links) + '</ul>' if issue_links else '<p class="muted">' + ('全ケースを実行した後、ここに確認対象を表示します。' if not records else '記録済みの試行に不合格・エラーはありません。') + '</p>'
    model_text = ', '.join(summary["observed_models"]) or "未取得"
    stability = percentage(summary["stable_questions"] / summary["stability_eligible"]) if summary["stability_eligible"] else "—"
    hashes = manifest.get("hashes", {})
    tokens = {
        "STATUS": h(state), "LEAD": h(lead), "RUN_ID": h(manifest.get("run_id", "preview")),
        "DATE": h(manifest.get("finished_at") or manifest.get("started_at") or "未実行"),
        "MODEL": h(model_text), "REQUESTED_MODEL": h(config["model"]), "VERSION": h(questions["version"]),
        "REPEATS": str(config["repeats"]), "CASE_COUNT": str(len(questions["cases"])),
        "PLANNED": str(summary["planned"]), "VALID": str(summary["valid"]),
        "METRICS": ''.join(metrics), "GROUP_ROWS": ''.join(group_rows), "CASES": ''.join(case_html), "ISSUES": issues,
        "P50": f(summary["latency_median"]), "P95": f(summary["latency_p95"]), "LATENCY_COUNT": str(summary["latency_count"]),
        "STABILITY": stability, "STABLE": str(summary["stable_questions"]), "STABILITY_N": str(summary["stability_eligible"]),
        "COST": f(summary["estimated_cost_usd"], 6), "INPUT_TOKENS": f'{summary["input_tokens"]:,}',
        "OUTPUT_TOKENS": f'{summary["output_tokens"]:,}', "RETRIES": str(summary["retry_attempts"]),
        "ATTEMPTS": str(summary["http_attempts"]), "SEED": str(config["shuffle_seed"]),
        "PRICE": str(config["price_usd_per_million_input_tokens"]), "PRICE_DATE": h(config["price_checked_at"]),
        "QUESTION_HASH": h(hashes.get("questions", "未取得")), "ANSWER_HASH": h(hashes.get("answers", "未取得")),
        "RUNNER_HASH": h(hashes.get("runner", "未取得")),
        "ANALYSIS_HASH": h(digest(__file__)), "NUMERIC_NOTES": str(summary["numerical_note_count"]),
        "TOTAL_SCORE": f(summary["total_score"]["points"], 1),
        "SCORE_QUALIFIER": "暫定点・未完了分は未加点" if records and summary["total_score"]["provisional"] else ("実測結果" if records else "未実測"),
        "ERROR_NOTICE": h(manifest.get("stop_reason", "")),
    }
    template = (ROOT / "report_template.html").read_text(encoding="utf-8")
    rendered = re.sub(r"@@([A-Z0-9_]+)@@", lambda m: tokens[m[1]], template)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(rendered, encoding="utf-8")
    return summary


def publish_current_report(directory, manifest):
    """Only promote results whose fixtures match the active benchmark exactly."""
    hashes = manifest.get("hashes", {})
    if (hashes.get("questions") != digest(ROOT / "pilot.questions.yaml") or
            hashes.get("answers") != digest(ROOT / "pilot.answers.yaml")):
        return False
    (ROOT / "report.html").write_bytes((directory / "report.html").read_bytes())
    write_json(ROOT / "results" / "latest.json", {"run_id": manifest["run_id"], "directory": str(directory.resolve())})
    return True


def run(args):
    api_key = get_key(args.env_file)
    if args.resume:
        directory = Path(args.resume).resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, file in [("questions", "questions.snapshot.yaml"), ("answers", "answers.snapshot.yaml"), ("config", "config.json")]:
            if digest(directory / file) != manifest["hashes"][name]:
                raise ValueError("再開対象のスナップショットが変更されています。")
        if digest(__file__) != manifest["hashes"]["runner"]:
            raise ValueError("実行コードが変更されました。同じ実行へ混在させず、新しい実行を開始してください。")
        questions, key = load_yaml(directory / "questions.snapshot.yaml"), load_yaml(directory / "answers.snapshot.yaml")
        config = load_config(directory / "config.json")
        manifest.setdefault("resumed_at", []).append(now())
    else:
        questions, key = load_yaml(ROOT / "pilot.questions.yaml"), load_yaml(ROOT / "pilot.answers.yaml")
        config = load_config(args.config)
        validate_fixtures(questions, key, config)
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        directory = ROOT / "results" / run_id
        directory.mkdir(parents=True)
        for source, destination in [(ROOT / "pilot.questions.yaml", directory / "questions.snapshot.yaml"),
                                    (ROOT / "pilot.answers.yaml", directory / "answers.snapshot.yaml"),
                                    (Path(args.config), directory / "config.json"),
                                    (Path(__file__), directory / "runner.snapshot.py")]:
            destination.write_bytes(source.read_bytes())
        manifest = {"run_id": run_id, "started_at": now(), "endpoint": ENDPOINT,
                    "benchmark_id": questions["benchmark_id"], "benchmark_version": questions["version"],
                    "hashes": {"questions": digest(directory / "questions.snapshot.yaml"), "answers": digest(directory / "answers.snapshot.yaml"),
                               "config": digest(directory / "config.json"), "runner": digest(__file__)},
                    "measurement": "sequential, per-case HTTP round trip including retries/backoff; no warmup; no answer cache",
                    "status": "running"}
    validate_fixtures(questions, key, config)
    manifest.update(status="running", stop_reason="")
    manifest.pop("finished_at", None)
    write_json(directory / "manifest.json", manifest)
    records = load_records(directory / "responses.jsonl")
    latest = latest_records(records)
    rng = random.Random(config["shuffle_seed"])
    schedule = []
    for rep in range(1, config["repeats"] + 1):
        cases = list(questions["cases"])
        rng.shuffle(cases)
        schedule.extend((rep, c) for c in cases)
    stopped = False
    print(f"実行先: {directory}", flush=True)
    try:
        with (directory / "responses.jsonl").open("a", encoding="utf-8") as log:
            for index, (rep, case) in enumerate(schedule, 1):
                prior = latest.get((case["id"], rep))
                if prior and not prior.get("error"):
                    continue
                payload = payload_for(case, config["model"])
                record = call_api(payload, api_key, config)
                record.update(case_id=case["id"], repeat=rep, request=payload)
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                os.fsync(log.fileno())
                records.append(record)
                result = evaluate_case(case, key, record, config)
                print(f"[{index}/{len(schedule)}] {case['id']} 試行{rep} {result['status']} {record['elapsed_seconds']:.3f}s", flush=True)
                error = record.get("error")
                if error and (error.get("kind") != "http" or error.get("status") in {400,401,402,403,404,422}):
                    manifest["stop_reason"] = error["message"] + "。残るリクエストは未実行です。"
                    stopped = True
                    break
    except KeyboardInterrupt:
        manifest["stop_reason"] = "実行が中断されました。記録済みの結果のみ表示しています。"
        stopped = True
    finally:
        manifest.update(finished_at=now(), status="partial" if stopped else "finished")
        write_json(directory / "manifest.json", manifest)
        summary = render_report(questions, key, config, manifest, records, directory / "report.html")
        write_json(directory / "summary.json", summary)
        published = publish_current_report(directory, manifest)
        print(f"HTML: {ROOT / 'report.html' if published else directory / 'report.html'}", flush=True)
    return 4 if stopped else (0 if summary["valid"] == summary["planned"] else 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "preview", "run"):
        command = sub.add_parser(name)
        command.add_argument("--config", default=str(ROOT / "run_config.json"))
        if name == "run":
            command.add_argument("--env-file", default=str(ROOT.parent / ".env.local"))
            command.add_argument("--resume")
    report = sub.add_parser("report")
    report.add_argument("run_directory")
    args = parser.parse_args()
    try:
        if args.command == "run":
            return run(args)
        if args.command == "report":
            directory = Path(args.run_directory)
            manifest = json.loads((directory / "manifest.json").read_text())
            for name, filename in [("questions", "questions.snapshot.yaml"), ("answers", "answers.snapshot.yaml"), ("config", "config.json")]:
                if digest(directory / filename) != manifest["hashes"][name]:
                    raise ValueError("スナップショットのハッシュが一致しません。")
            q, a = load_yaml(directory / "questions.snapshot.yaml"), load_yaml(directory / "answers.snapshot.yaml")
            config = load_config(directory / "config.json")
            validate_fixtures(q, a, config)
            manifest.setdefault("analysis_history", []).append({"generated_at": now(), "analysis_runner_sha256": digest(__file__),
                "total_score_policy": {"single_weight": 1, "composite_weight": 1.5, "maximum": 100, "unit": "case_pass_fraction_over_repeats"},
                "note": "返却値の表示精度を考慮した整合性検証。確率・Scoreは最低小数2桁の四捨五入を仮定し、より細かい値にはその精度を適用。想定解・合否閾値・生応答は変更しない。"})
            write_json(directory / "manifest.json", manifest)
            summary = render_report(q, a, config, manifest, load_records(directory / "responses.jsonl"), directory / "report.html")
            write_json(directory / "summary.json", summary)
            published = publish_current_report(directory, manifest)
            print("HTMLを保存済みの応答から再生成しました。API呼び出しはありません。")
            if not published:
                print("現在の設問・想定解と異なるため、実行ディレクトリ内にのみ保存しました。")
            return 0
        q, a, config = load_yaml(ROOT / "pilot.questions.yaml"), load_yaml(ROOT / "pilot.answers.yaml"), load_config(args.config)
        validate_fixtures(q, a, config)
        if args.command == "preview":
            render_report(q, a, config, {"hashes": {"questions": digest(ROOT / "pilot.questions.yaml"), "answers": digest(ROOT / "pilot.answers.yaml"), "runner": digest(__file__)}}, [], ROOT / "report.preview.html")
            print(f"未実行プレビュー: {ROOT / 'report.preview.html'}")
        else:
            print(f"検証OK: {len(q['cases'])}ケース / {q['atomic_question_count']}判断 / {config['repeats']}反復")
        return 0
    except (ValueError, KeyError, OSError, yaml.YAMLError) as exc:
        print(f"実行できません: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
