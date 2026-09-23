"""Continue a prompt from a nanoGPT checkpoint.

    uv run python -m src.sample --checkpoint runs/gpt-laptop.pt --prompt "ROMEO:"
"""

import argparse
from pathlib import Path

import torch

from src.model import build_model
from src.train import CharTokenizer, pick_device


def load_checkpoint(path, device=None):
    device = device or pick_device()
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
    """Return the prompt plus max_new_tokens new characters.

    An empty prompt starts from a newline, which is token id 0 in this vocabulary.
    Generation crops the context to block_size inside the model.
    """
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
