"""Checkpoint boundaries and a fresh SFT optimizer."""

from pathlib import Path

import torch

from src.train import boundary_name, train

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "fixture" / "pretrain.jsonl"
SFT = Path(__file__).resolve().parents[1] / "data" / "fixture" / "sft.jsonl"


def tiny_overrides():
    return {
        "vocab_size": 8000,
        "n_embd": 32,
        "n_layer": 2,
        "n_head": 4,
        "n_kv_head": 2,
        "head_dim": 8,
        "intermediate": 64,
        "block_size": 16,
        "dropout": 0.0,
    }


def test_boundary_names():
    assert boundary_name(100_000_000) == "tok-100m"
    assert boundary_name(32) == "tok-32"


def test_two_boundaries_and_sft(tmp_path):
    tokenizer = tmp_path / "fixture.model"
    runs = tmp_path / "runs"
    common = dict(
        data_path=FIXTURE,
        runs_dir=runs,
        tokenizer_path=tokenizer,
        model_overrides=tiny_overrides(),
        train_overrides={"batch_size": 2, "eval_interval": 10**9, "warmup_ratio": 0.02},
        batch_size=2,
        block_size=16,
        enforce_parameter_count=False,
        device="cpu",
        seed=1,
    )
    train(stage="pretrain", until_tokens=32, save_every_tokens=32, max_steps=4, **common)
    first = runs / "pretrain" / "tok-32"
    assert (first / "checkpoint.pt").exists()
    assert not (runs / "pretrain" / "tok-64").exists()
    train(
        stage="pretrain",
        until_tokens=64,
        save_every_tokens=32,
        resume=first,
        max_steps=4,
        **common,
    )
    second = runs / "pretrain" / "tok-64"
    assert (second / "checkpoint.pt").exists()
    resumed = torch.load(second / "checkpoint.pt", weights_only=False)
    assert resumed["tokens_seen"] >= 64
    assert resumed["stage"] == "pretrain"
    train(
        stage="sft",
        data_path=SFT,
        runs_dir=runs,
        until_tokens=32,
        tokenizer_path=tokenizer,
        model_overrides=tiny_overrides(),
        train_overrides={"batch_size": 2, "eval_interval": 10**9},
        init=second,
        max_steps=1,
        batch_size=2,
        block_size=64,
        enforce_parameter_count=False,
        device="cpu",
    )
    sft_path = runs / "sft" / "from-tok-64" / "checkpoint.pt"
    sft = torch.load(sft_path, weights_only=False)
    assert sft["init_checkpoint"].endswith("tok-64")
    assert sft["init_tokens_seen"] == resumed["tokens_seen"]
    assert sft["optimizer_step"] == 1
    assert sft["optimizer_step"] < resumed["optimizer_step"]


def test_latest_checkpoint_resumes(tmp_path):
    tokenizer = tmp_path / "fixture.model"
    runs = tmp_path / "runs"
    common = dict(
        data_path=FIXTURE,
        runs_dir=runs,
        tokenizer_path=tokenizer,
        model_overrides=tiny_overrides(),
        train_overrides={"batch_size": 2, "eval_interval": 10**9, "save_every_steps": 2},
        save_every_tokens=10**9,
        until_tokens=10**9,
        batch_size=2,
        block_size=16,
        enforce_parameter_count=False,
        device="cpu",
        seed=1,
    )
    first = train(stage="pretrain", max_steps=2, **common)
    latest = runs / "pretrain" / "latest" / "checkpoint.pt"
    assert latest.exists()
    saved = torch.load(latest, weights_only=False)
    assert saved["optimizer_step"] == 2
    assert saved["schedule"]["step"] == 2
    second = train(stage="pretrain", max_steps=1, resume=latest.parent, **common)
    assert second["step"] == first["step"] + 1
    assert second["tokens_seen"] > first["tokens_seen"]


def test_overfit_loss_falls(tmp_path):
    overrides = tiny_overrides()
    overrides["vocab_size"] = 128
    result = train(
        stage="overfit",
        data_path=SFT,
        runs_dir=tmp_path / "runs",
        tokenizer_path=tmp_path / "fixture.model",
        model_overrides=overrides,
        train_overrides={"batch_size": 8, "eval_interval": 10**9, "peak_lr": 3e-3, "min_lr": 1e-3},
        until_tokens=10**9,
        batch_size=8,
        block_size=32,
        max_steps=200,
        loss_stop=0.5,
        enforce_parameter_count=False,
        device="cpu",
        seed=1,
    )
    assert result["loss"] < 0.75
