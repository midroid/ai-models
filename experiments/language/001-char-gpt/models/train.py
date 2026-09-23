"""Train a character-level bigram or GPT on Tiny Shakespeare."""

import argparse
from pathlib import Path

import torch

from bigram import BigramLanguageModel
from gpt import GPTLanguageModel
from tokenizer import CharTokenizer

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
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_text(path):
    return Path(path).read_text(encoding="utf-8")


def split_data(encoded, train_fraction=0.9):
    n = int(train_fraction * len(encoded))
    return encoded[:n], encoded[n:]


def get_batch(split_name, train_data, val_data, batch_size, block_size, device):
    data = train_data if split_name == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
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

    torch.manual_seed(seed)
    text = load_text(data_path)
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = split_data(encoded)

    model = build_model(model_name, tokenizer.vocab_size, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history = []
    for step in range(config["max_iters"]):
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
    print(f"wrote {checkpoint_path}")
    return {
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "vocab_size": tokenizer.vocab_size,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "device": device,
        "config": config,
        "learning_rate": learning_rate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bigram", "gpt"), required=True)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="laptop")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--tag", default=None)
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
