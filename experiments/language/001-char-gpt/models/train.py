"""Train a character-level bigram or GPT on Tiny Shakespeare.

Run from the repository root so the local imports resolve:

    .venv/bin/python experiments/language/001-char-gpt/models/train.py --model gpt

`laptop` is the small preset used on this Mac. `lecture` matches the GPU
hyperparameters in karpathy/ng-video-lecture gpt.py. The bigram uses the same
batch, block size, and step count, and ignores layers, heads, and embedding
width. Its learning rate is higher because the lookup table is tiny.
"""

import argparse
import time
from pathlib import Path

import torch

from bigram import BigramLanguageModel
from gpt import GPTLanguageModel
from tokenizer import CharTokenizer

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = EXPERIMENT_DIR / "data" / "input.txt"
DEFAULT_RUNS = EXPERIMENT_DIR / "runs"

# n_embd, n_head, n_layer, and dropout apply only to the GPT.
# The bigram is a vocab-by-vocab table and does not read those four fields.
PRESETS = {
    "laptop": {
        "batch_size": 12,  # sequences per optimizer step
        "block_size": 64,  # characters of context in each sequence
        "max_iters": 2000,
        "eval_interval": 200,  # print train and val loss every this many steps
        "eval_iters": 50,  # batches averaged into each printed loss
        "n_embd": 128,
        "n_head": 4,
        "n_layer": 4,
        "dropout": 0.0,
    },
    "lecture": {
        "batch_size": 64,
        "block_size": 256,
        "max_iters": 5000,
        "eval_interval": 500,
        "eval_iters": 200,
        "n_embd": 384,
        "n_head": 6,
        "n_layer": 6,
        "dropout": 0.2,
    },
}

LEARNING_RATES = {
    "bigram": 1e-2,
    "gpt": 3e-4,
}


def pick_device():
    """CUDA, then Apple MPS, then CPU. This Mac selects MPS."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_text(path):
    return Path(path).read_text(encoding="utf-8")


def split_data(encoded, train_fraction=0.9):
    """First 90% of the character stream is train, the rest is validation.

    The split is a cut in time, not a shuffle, so validation text is later in
    the plays than the training text. The character vocabulary is still built
    on the whole file, matching the lecture: a rare character should keep a
    stable id even if it shows up only in the held-out tail.
    """
    n = int(train_fraction * len(encoded))
    return encoded[:n], encoded[n:]


def get_batch(split_name, train_data, val_data, batch_size, block_size, device):
    """Random chunks of `block_size` characters.

    x[b, t] is the input token. y[b, t] is the next character, x[b, t + 1].
    Both tensors have shape (batch_size, block_size).
    """
    data = train_data if split_name == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    """Mean train and validation loss over `eval_iters` fresh batches.

    Dropout is turned off for the measurement, then training mode is restored.
    """
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


def build_model(model_name, vocab_size, config):
    if model_name == "bigram":
        return BigramLanguageModel(vocab_size)
    if model_name == "gpt":
        return GPTLanguageModel(
            vocab_size=vocab_size,
            block_size=config["block_size"],
            n_embd=config["n_embd"],
            n_head=config["n_head"],
            n_layer=config["n_layer"],
            dropout=config["dropout"],
        )
    raise ValueError(f"unknown model {model_name}")


def train(
    model_name,
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
    if model_name not in LEARNING_RATES:
        raise ValueError(f"unknown model {model_name}")

    config = dict(PRESETS[preset])
    if max_iters is not None:
        config["max_iters"] = max_iters
    learning_rate = LEARNING_RATES[model_name]
    device = device or pick_device()
    started = time.perf_counter()

    torch.manual_seed(seed)
    text = load_text(data_path)
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_data(encoded)

    model = build_model(model_name, tokenizer.vocab_size, config).to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print(
        f"model {model_name}  preset {preset}  device {device}  "
        f"parameters {num_parameters:,}  steps {config['max_iters']}  lr {learning_rate:g}"
    )

    history = []
    for step in range(config["max_iters"]):
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
            history.append({"step": step, "train_loss": losses["train"], "val_loss": losses["val"]})
            print(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
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
        optimizer.step()

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = tag or f"{model_name}-{preset}"
    checkpoint_path = runs_dir / f"{checkpoint_name}.pt"
    torch.save(
        {
            "model_name": model_name,
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "config": config,
            "learning_rate": learning_rate,
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
        "learning_rate": learning_rate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bigram", "gpt"), required=True, help="bigram baseline or GPT")
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="laptop",
        help="laptop: small MPS/CPU run. lecture: the GPU hyperparameters from gpt.py",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="UTF-8 text file")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS, help="directory for the checkpoint")
    parser.add_argument("--max-iters", type=int, default=None, help="override the preset step count")
    parser.add_argument("--tag", default=None, help="checkpoint filename stem; default is {model}-{preset}")
    args = parser.parse_args()
    train(
        model_name=args.model,
        preset=args.preset,
        data_path=args.data,
        runs_dir=args.runs_dir,
        max_iters=args.max_iters,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
