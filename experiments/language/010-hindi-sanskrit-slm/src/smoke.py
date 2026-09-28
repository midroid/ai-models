"""Check a tied decoder, then take one optimizer step.

    uv run python -m src.smoke --model 50m
    uv run python -m src.smoke --model 10m

The cache check fails if prefill or decode logits differ from a full forward
by more than 1e-4. The parameter check fails if the tied model is any other
size. Dropout is on during the training step and off during the cache check.
"""

import argparse

import torch

from src.model import (
    MODEL_10M,
    MODEL_50M,
    MODEL_HIGH,
    PARAMETER_COUNT,
    PARAMETER_COUNT_50M,
    PARAMETER_COUNT_HIGH,
    build_model,
    unique_parameters,
)
from src.train import GAP_LIMIT, build_optimizer, cache_check, pick_device

MODELS = {
    "10m": (MODEL_10M, PARAMETER_COUNT),
    "50m": (MODEL_50M, PARAMETER_COUNT_50M),
    "high": (MODEL_HIGH, PARAMETER_COUNT_HIGH),
}


def smoke(name, device=None):
    device = device or pick_device()
    base, expected = MODELS[name]
    config = dict(base)
    model = build_model(config).to(device)
    count = model.num_parameters()
    if count != expected:
        raise SystemExit(f"parameter count {count} is not {expected:,}")
    if model.lm_head.weight.data_ptr() != model.wte.weight.data_ptr():
        raise SystemExit("embedding and lm_head do not share storage")
    report = cache_check(model)
    if report["prefill_gap"] > GAP_LIMIT or report["decode_gap"] > GAP_LIMIT:
        raise SystemExit(f"cache gap {report}")
    model.train()
    optimizer = build_optimizer(model, 3e-4, 0.1, 0.9, 0.95)
    inputs = torch.randint(0, config["vocab_size"], (2, 16), device=device)
    labels = inputs.clone()
    _, loss = model(inputs, labels)
    loss.backward()
    if not torch.isfinite(loss):
        raise SystemExit("loss is not finite")
    for param in unique_parameters(model):
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise SystemExit("gradient is not finite")
    query_shape = model.blocks[0].attn.last_q_shape
    key_shape = model.blocks[0].attn.last_k_shape
    expected_query = (2, 16, config["n_head"], config["head_dim"])
    expected_key = (2, 16, config["n_kv_head"], config["head_dim"])
    if query_shape != expected_query or key_shape != expected_key:
        raise SystemExit(f"attention shapes query {query_shape} key {key_shape}")
    optimizer.step()
    print(
        f"smoke ok  model {name}  device {device}  parameters {count:,}  "
        f"prefill gap {report['prefill_gap']:.3e}  decode gap {report['decode_gap']:.3e}  "
        f"loss {loss.item():.4f}"
    )
    return {"parameters": count, "loss": float(loss.detach()), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default="50m")
    args = parser.parse_args()
    smoke(args.model)


if __name__ == "__main__":
    main()
