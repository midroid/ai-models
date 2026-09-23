"""Continue a prompt from a minGPT checkpoint.

    uv run python -m src.sample --checkpoint runs/mingpt-smoke.pt --prompt "ROMEO:"
"""

import argparse
from pathlib import Path

import torch

from src.model import build_model
from src.train import CharDataset
from src.trainer import pick_device


def load_checkpoint(path, device=None):
    device = device or pick_device()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    chars = checkpoint["tokenizer"]["chars"]
    dataset = CharDataset("".join(chars), checkpoint["config"]["block_size"])
    # The checkpoint vocabulary is the source of ids. Rebuild maps from it so a
    # short dummy string cannot change the order.
    dataset.stoi = {ch: i for i, ch in enumerate(chars)}
    dataset.itos = {i: ch for i, ch in enumerate(chars)}
    model = build_model(len(chars), checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, dataset, checkpoint, device


def continue_prompt(checkpoint_path, prompt, max_new_tokens=200, device=None):
    model, dataset, checkpoint, device = load_checkpoint(checkpoint_path, device=device)
    if not prompt:
        prompt = "\n"
    ids = dataset.encode(prompt)
    if not ids:
        raise ValueError("prompt encoded to no tokens")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=1.0, top_k=10)[0].tolist()
    return dataset.decode(out), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()
    text, _ = continue_prompt(args.checkpoint, args.prompt, args.max_new_tokens)
    print(text)


if __name__ == "__main__":
    main()
