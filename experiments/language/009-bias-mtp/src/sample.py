"""Continue a prompt from a checkpoint, using the main stack and its KV cache.

    uv run python -m src.sample --checkpoint runs/mtp-laptop.pt --prompt "ROMEO:"

Prefill runs the prompt once. Each new character is one decode step. The
second-token head is not called. The script prints token-layers and seconds
for each phase.
"""

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from src.model import build_model
from src.train import CharTokenizer, pick_device


def load_checkpoint(path, device=None):
    device = device or pick_device()
    # The file is a dict of weights, vocabulary, and config, not a bare tensor.
    # Older PyTorch builds do not accept weights_only.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    tokenizer = CharTokenizer.from_state(checkpoint["tokenizer"])
    model = build_model(tokenizer.vocab_size, checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, checkpoint, device


def continue_prompt(checkpoint_path, prompt, max_new_tokens=300, device=None):
    """Return the prompt plus up to max_new_tokens new characters.

    An empty prompt starts from a newline, which is token id 0 in this vocabulary.
    Generation stops at block_size because the rotary table has that many rows.
    """
    model, tokenizer, checkpoint, device = load_checkpoint(checkpoint_path, device=device)
    if not prompt:
        prompt = "\n"
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt encoded to no tokens")
    if len(ids) > model.block_size:
        ids = ids[-model.block_size :]
    room = model.block_size - len(ids)
    steps = min(max_new_tokens, room)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    started = time.perf_counter()
    logits, cache = model.prefill(idx)
    prefill_seconds = time.perf_counter() - started
    print(
        f"prefill tokens {idx.size(1)}  "
        f"token-layers {model.stats.token_layers}  "
        f"kv bytes/token {model.kv_bytes_per_token()}  "
        f"({prefill_seconds:.3f}s)"
    )

    decode_token_layers = 0
    started = time.perf_counter()
    for _ in range(steps):
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        logits, cache = model.decode_step(idx_next, cache)
        decode_token_layers += model.stats.token_layers
    decode_seconds = time.perf_counter() - started
    print(
        f"decode tokens {steps}  "
        f"token-layers {decode_token_layers}  "
        f"({decode_seconds:.3f}s)"
    )
    return tokenizer.decode(idx[0].tolist()), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()
    text, _ = continue_prompt(args.checkpoint, args.prompt, args.max_new_tokens)
    print(text)


if __name__ == "__main__":
    main()
