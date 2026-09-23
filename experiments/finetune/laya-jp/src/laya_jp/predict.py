"""The only inference path. Suites call predict(); they do not construct an Agent."""

from __future__ import annotations

import json
import os
import platform
import time
from typing import Any

from laya_jp.pins import CACHE, DEVICE, MODEL_ID, MODEL_REVISION, RESULTS

_agent = None


def load_agent():
    global _agent
    if _agent is not None:
        return _agent
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download
    import laya

    CACHE.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=str(CACHE))
    _agent = laya.load(local, device=DEVICE)
    return _agent


def environment(agent=None) -> dict[str, Any]:
    os.environ.setdefault("USE_TF", "0")
    import laya
    import torch
    import transformers

    agent = agent if agent is not None else _agent
    payload = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "laya": laya.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": DEVICE,
    }
    if agent is not None:
        payload["device"] = str(agent.device)
        payload["dtype"] = str(agent.dtype)
    return payload


def write_environment(agent=None) -> dict[str, Any]:
    payload = environment(agent)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "environment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def predict(state, questions) -> tuple[dict, float]:
    agent = load_agent()
    started = time.perf_counter()
    result = agent.predict(state, questions)
    return result, (time.perf_counter() - started) * 1000.0


def predict_with_logits(state, questions) -> tuple[dict, float, dict]:
    """One forward pass. Logits are the marker scores before temperature."""
    agent = load_agent()
    captured: dict[str, Any] = {"available": False}
    original = getattr(agent, "_forward", None)
    if original is None or not hasattr(agent, "model"):
        result, elapsed = predict(state, questions)
        captured["reason"] = "agent has no _forward"
        return result, elapsed, captured

    def wrapped(batch):
        logits, act = original(batch)
        mask = batch.get("marker_mask")
        captured["available"] = True
        captured["logits"] = logits.tolist() if hasattr(logits, "tolist") else logits
        if mask is not None and hasattr(mask, "detach"):
            captured["marker_mask"] = mask.detach().cpu().tolist()
        return logits, act

    agent._forward = wrapped
    try:
        result, elapsed = predict(state, questions)
    finally:
        agent._forward = original
    return result, elapsed, captured
