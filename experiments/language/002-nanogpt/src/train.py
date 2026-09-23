"""Train a character-level nanoGPT on Tiny Shakespeare.

From the experiment directory:

    uv run python -m src.train --preset laptop

`laptop` is the small run for this Mac. `shakespeare` matches upstream
config/train_shakespeare_char.py and is meant for a GPU. The learning rate
warms up linearly, then follows a cosine down to min_lr. Gradients are clipped
to 1.0. torch.compile stays off so the same script runs on MPS.
"""

import argparse
import math
import time
from pathlib import Path

import torch

from src.model import build_model

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = EXPERIMENT_DIR / "data" / "input.txt"
DEFAULT_RUNS = EXPERIMENT_DIR / "runs"

PRESETS = {
    "laptop": {
        "batch_size": 12,
        "block_size": 64,
        "max_iters": 2000,
        "eval_interval": 200,
        "eval_iters": 50,
        "n_embd": 128,
        "n_head": 4,
        "n_layer": 4,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "min_lr": 1e-4,
        "warmup_iters": 100,
        "weight_decay": 0.1,
    },
    "shakespeare": {
        "batch_size": 64,
        "block_size": 256,
        "max_iters": 5000,
        "eval_interval": 250,
        "eval_iters": 200,
        "n_embd": 384,
        "n_head": 6,
        "n_layer": 6,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "min_lr": 1e-4,
        "warmup_iters": 100,
        "weight_decay": 0.1,
    },
}

GRAD_CLIP = 1.0


class CharTokenizer:
    """Sorted character vocabulary. Index 0 is the newline in Tiny Shakespeare."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @classmethod
    def from_state(cls, state):
        return cls(state["chars"])

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def state_dict(self):
        return {"chars": self.chars}


def pick_device():
    """CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(step, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    """Linear warmup, then cosine decay down to min_lr."""
    if step < warmup_iters:
        return learning_rate * (step + 1) / (warmup_iters + 1)
    if step >= lr_decay_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def load_text(path):
    return Path(path).read_text(encoding="utf-8")


def split_data(encoded, train_fraction=0.9):
    """First 90% of the character stream is train. The cut is in time, not shuffled.

    The vocabulary is still built on the whole file, so a rare character keeps
    one id even if it appears only in the held-out tail.
    """
    n = int(train_fraction * len(encoded))
    return encoded[:n], encoded[n:]


def get_batch(split_name, train_data, val_data, batch_size, block_size, device):
    """x[b, t] is the input character. y[b, t] is the next one."""
    data = train_data if split_name == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    """Mean train and validation loss. Dropout is off for the measurement."""
    model.eval()
    out = {}
    for split_name in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split_name, train_data, val_data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


def train(
    preset="laptop",
    data_path=DEFAULT_DATA,
    runs_dir=DEFAULT_RUNS,
    max_iters=None,
    tag=None,
    seed=1337,
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
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_data(encoded)

    model = build_model(tokenizer.vocab_size, config).to(device)
    num_parameters = model.num_parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
    )
    print(
        f"model gpt  preset {preset}  device {device}  "
        f"parameters {num_parameters:,}  steps {config['max_iters']}  "
        f"lr {config['learning_rate']:g} -> {config['min_lr']:g}"
    )

    history = []
    for step in range(config["max_iters"]):
        lr = cosine_lr(
            step,
            config["warmup_iters"],
            config["max_iters"],
            config["learning_rate"],
            config["min_lr"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Loss is measured at the start of the step, before this step's update.
        if step % config["eval_interval"] == 0 or step == config["max_iters"] - 1:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                config["batch_size"],
                config["block_size"],
                config["eval_iters"],
                device,
            )
            history.append(
                {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "lr": lr,
                }
            )
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, lr {lr:.6f}"
            )

        xb, yb = get_batch(
            "train",
            train_data,
            val_data,
            config["batch_size"],
            config["block_size"],
            device,
        )
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = tag or f"gpt-{preset}"
    checkpoint_path = runs_dir / f"{checkpoint_name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "config": config,
            "preset": preset,
            "history": history,
            "device_trained": device,
        },
        checkpoint_path,
    )
    elapsed = time.perf_counter() - started
    print(f"wrote {checkpoint_path}  ({elapsed:.1f}s)")
    return {
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "vocab_size": tokenizer.vocab_size,
        "num_parameters": num_parameters,
        "elapsed_seconds": elapsed,
        "device": device,
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="laptop",
        help="laptop: small MPS/CPU run. shakespeare: upstream GPU hyperparameters",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="UTF-8 text file")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS, help="directory for the checkpoint")
    parser.add_argument("--max-iters", type=int, default=None, help="override the preset step count")
    parser.add_argument("--tag", default=None, help="checkpoint filename stem; default is gpt-{preset}")
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
