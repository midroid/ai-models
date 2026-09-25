"""Check the 10,942,720-parameter decoder, then take one optimizer step.

    uv run python -m src.smoke

The cache check fails if prefill or decode logits differ from a full forward
by more than 1e-4. The parameter check fails if the tied model is any other
size. Dropout is on during the training step and off during the cache check.
"""

import torch

from src.model import MODEL_10M, PARAMETER_COUNT, build_model, unique_parameters
from src.train import GAP_LIMIT, build_optimizer, cache_check, pick_device


def main():
    device = pick_device()
    config = dict(MODEL_10M)
    model = build_model(config).to(device)
    count = model.num_parameters()
    if count != PARAMETER_COUNT:
        raise SystemExit(f"parameter count {count} is not {PARAMETER_COUNT:,}")
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
    if query_shape != (2, 16, 8, 32) or key_shape != (2, 16, 2, 32):
        raise SystemExit(f"attention shapes query {query_shape} key {key_shape}")
    optimizer.step()
    print(
        f"smoke ok  device {device}  parameters {count:,}  "
        f"prefill gap {report['prefill_gap']:.3e}  decode gap {report['decode_gap']:.3e}  "
        f"loss {loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
