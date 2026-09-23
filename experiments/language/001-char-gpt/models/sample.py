"""Continue a prompt from a saved character-level checkpoint."""

import argparse
from pathlib import Path

import torch

from train import build_model, pick_device
from tokenizer import CharTokenizer


def load_checkpoint(path, device=None):
    device = device or pick_device()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    tokenizer = CharTokenizer.from_state(checkpoint["tokenizer"])
    model = build_model(checkpoint["model_name"], tokenizer.vocab_size, checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, checkpoint, device


def continue_prompt(checkpoint_path, prompt, max_new_tokens=300, device=None):
    model, tokenizer, checkpoint, device = load_checkpoint(checkpoint_path, device=device)
    if not prompt:
        prompt = "\n"
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt encoded to no tokens")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    return tokenizer.decode(out), checkpoint


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
