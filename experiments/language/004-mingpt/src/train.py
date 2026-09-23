"""Train a character-level minGPT on Tiny Shakespeare.

From the experiment directory:

    uv run python -m src.train --preset smoke

`smoke` is chargpt turned down: gpt-micro, 400 steps, dropout 0. Upstream
chargpt uses gpt-mini and does not set max_iters. torch.compile stays off.
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.model import build_model
from src.trainer import Trainer, pick_device

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = EXPERIMENT_DIR / "data" / "input.txt"
DEFAULT_RUNS = EXPERIMENT_DIR / "runs"

PRESETS = {
    "smoke": {
        "model_type": "gpt-micro",
        "block_size": 64,
        "batch_size": 12,
        "dropout": 0.0,
        "learning_rate": 5e-4,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "grad_norm_clip": 1.0,
        "num_workers": 0,
        "max_iters": 400,
        "log_every": 50,
        "sample_tokens": 200,
        "prompt": "ROMEO:",
    },
}


class CharDataset(Dataset):
    """Chunks of block_size + 1 characters. The vocabulary is the whole file."""

    def __init__(self, text, block_size):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.data = text

    @property
    def vocab_size(self):
        return len(self.stoi)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        ids = [self.stoi[ch] for ch in chunk]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def state_dict(self):
        return {"chars": [self.itos[i] for i in range(self.vocab_size)]}


def load_text(path):
    return Path(path).read_text(encoding="utf-8")


def train(
    preset="smoke",
    data_path=DEFAULT_DATA,
    runs_dir=DEFAULT_RUNS,
    max_iters=None,
    tag=None,
    seed=3407,
    device=None,
):
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset}")
    config = dict(PRESETS[preset])
    if max_iters is not None:
        config["max_iters"] = max_iters
    device = device or pick_device()
    started = time.perf_counter()

    torch.manual_seed(seed)
    text = load_text(data_path)
    dataset = CharDataset(text, config["block_size"])
    model = build_model(dataset.vocab_size, config)
    transformer_params = model.transformer_parameters()
    total_params = model.num_parameters()
    print(
        f"model mingpt  preset {preset}  device {device}  "
        f"type {config['model_type']}  "
        f"parameters {transformer_params:,} transformer, {total_params:,} with head  "
        f"steps {config['max_iters']}  lr {config['learning_rate']:g}"
    )

    history = []

    def on_batch(trainer):
        step = trainer.iter_num
        if step == 1 or step % config["log_every"] == 0 or step == config["max_iters"]:
            loss = trainer.loss.item()
            history.append({"step": step, "train_loss": loss})
            print(f"step {step}: train loss {loss:.4f}")

    trainer = Trainer(model, dataset, config, device=device)
    trainer.add_callback(on_batch)
    trainer.run()

    model.eval()
    prompt = config["prompt"]
    ids = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=trainer.device)
    generated = model.generate(ids, max_new_tokens=config["sample_tokens"], temperature=1.0, top_k=10)
    sample = dataset.decode(generated[0].tolist())
    print("sample:")
    print(sample)

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = tag or f"mingpt-{preset}"
    checkpoint_path = runs_dir / f"{checkpoint_name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": dataset.state_dict(),
            "config": config,
            "preset": preset,
            "history": history,
            "device_trained": trainer.device,
            "sample": sample,
        },
        checkpoint_path,
    )
    elapsed = time.perf_counter() - started
    print(f"wrote {checkpoint_path}  ({elapsed:.1f}s)")
    return {
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "vocab_size": dataset.vocab_size,
        "num_parameters": total_params,
        "transformer_parameters": transformer_params,
        "elapsed_seconds": elapsed,
        "device": trainer.device,
        "config": config,
        "sample": sample,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    train(
        preset=args.preset,
        data_path=args.data,
        runs_dir=args.runs_dir,
        max_iters=args.max_iters,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
