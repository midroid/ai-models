"""Payload locks. These tests do not load the checkpoint."""

from __future__ import annotations

import importlib.util
import json
import sys
import types

from laya_jp.lock import verify_lock
from laya_jp.pins import EXTERNAL
from laya_jp.sokudan_questions import bench_questions_en, bench_questions_ja


def load_module(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_matches_vendored_files():
    verify_lock()


def test_snsk_evaluation_lock_matches_vendored_hashes():
    lock = json.loads((EXTERNAL / "snsk" / "evaluation_lock.json").read_text(encoding="utf-8"))
    from laya_jp.lock import sha256_file

    root = EXTERNAL / "snsk"
    for name in ("pilot.questions.yaml", "pilot.answers.yaml", "run_config.json", "bench.py"):
        assert sha256_file(root / name) == lock["hashes"][name]


def test_ja_questions_match_vendored_runner():
    vendored = load_module("vendored_bench_ja", EXTERNAL / "sokudan" / "bench_ja.py")
    assert bench_questions_ja() == vendored.bench_questions()


def test_en_questions_match_vendored_runner():
    ja = load_module("vendored_bench_ja_for_en", EXTERNAL / "sokudan" / "bench_ja.py")
    pkg = types.ModuleType("sokudan")
    evaluation = types.ModuleType("sokudan.eval")
    pkg.eval = evaluation
    sys.modules["sokudan"] = pkg
    sys.modules["sokudan.eval"] = evaluation
    sys.modules["sokudan.eval.bench_ja"] = ja
    vendored = load_module("vendored_bench_en", EXTERNAL / "sokudan" / "bench_en.py")
    assert bench_questions_en() == vendored.bench_questions()


def test_snsk_payload_matches_published_request():
    bench = load_module("snsk_bench_for_test", EXTERNAL / "snsk" / "bench.py")
    questions = bench.load_yaml(EXTERNAL / "snsk" / "pilot.questions.yaml")
    case = questions["cases"][0]
    assert case["id"] == "C01"
    published = None
    with (EXTERNAL / "snsk" / "laya.responses.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["case_id"] == "C01" and record["repeat"] == 1:
                published = record["request"]
                break
    assert published is not None
    assert bench.payload_for(case, published["model"]) == published
