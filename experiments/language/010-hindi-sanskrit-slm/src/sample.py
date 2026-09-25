"""Generate from a checkpoint for one task tag.

    uv run python -m src.sample --checkpoint runs/overfit --task translate_hi_sa --prompt "भारत एक विशाल देश है।"
"""

import argparse
from pathlib import Path

import torch

from src.data import TASK_TAGS
from src.model import build_model
from src.tokenizer import Tokenizer, write_model_bytes
from src.train import pick_device


def load_checkpoint(directory, device=None):
    device = device or pick_device()
    directory = Path(directory)
    path = directory / "checkpoint.pt" if directory.is_dir() else directory
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    tokenizer_path = path.parent / "tokenizer.model"
    if not tokenizer_path.exists():
        write_model_bytes(checkpoint["tokenizer_model"], tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.tie_weights()
    model.to(device)
    model.eval()
    return model, tokenizer, checkpoint, device


def continue_prompt(
    checkpoint_path,
    task,
    prompt,
    max_new_tokens=32,
    temperature=0.0,
    device=None,
    model=None,
    tokenizer=None,
):
    if model is None or tokenizer is None:
        model, tokenizer, checkpoint, device = load_checkpoint(checkpoint_path, device=device)
    else:
        checkpoint = None
        device = device or next(model.parameters()).device
    if task == "standard_hi":
        prefix = TASK_TAGS[task] + prompt
    else:
        prefix = f"{TASK_TAGS[task]}<src>{prompt}<tgt>"
    ids = tokenizer.encode(prefix)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, eos_id=tokenizer.eos_id, temperature=temperature)
    return tokenizer.decode(out[0].tolist()), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASK_TAGS), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    text, _ = continue_prompt(args.checkpoint, args.task, args.prompt, args.max_new_tokens)
    print(text)


if __name__ == "__main__":
    main()
