"""Train a dense GPT or a Mixtral-style mixture of experts on Tiny Shakespeare.

From the experiment directory:

    uv run python -m src.train --model gpt --preset laptop
    uv run python -m src.train --model moe --preset laptop

The learning rate is a constant 3e-4. AdamW decays matrices only. Gradients
are clipped to 1.0. The printed loss is next-character cross-entropy. The
mixture adds `0.01 * aux` to the optimizer step and prints that auxiliary
loss beside the cross-entropy. The text is the Tiny Shakespeare file from
experiment 002.
"""

import argparse
import time
from pathlib import Path

import torch

from src.model import build_model, verify_cache

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = EXPERIMENT_DIR.parent / "002-nanogpt" / "data" / "input.txt"
DEFAULT_RUNS = EXPERIMENT_DIR / "runs"

PRESETS = {
    "laptop": {
        "batch_size": 12,  # sequences per optimizer step
        "block_size": 64,  # characters of context, and the rotary-table length
        "max_iters": 2000,
        "eval_interval": 200,  # print train and val loss every this many steps
        "eval_iters": 50,  # batches averaged into each printed loss
        "n_embd": 128,
        "n_head": 4,
        "n_layer": 4,
        "n_expert": 8,  # Mixtral's expert count
        "top_k": 2,  # experts run for each token
        "aux_loss_coef": 0.01,  # Switch load-balancing weight; 0 in effect for the dense GPT
        "learning_rate": 3e-4,  # held fixed; experiment 002 cosines down from 1e-3
        "weight_decay": 0.1,  # applied to matrices only
    },
}

GRAD_CLIP = 1.0
MODEL_NAMES = ("gpt", "moe")


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


def build_optimizer(model, learning_rate, weight_decay):
    """AdamW that decays matrices only. Vectors and biases are left alone."""
    decay, no_decay = [], []
    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


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
    """Mean train and validation cross-entropy, and the mean auxiliary loss."""
    was_training = model.training
    model.eval()
    out = {}
    for split_name in ("train", "val"):
        losses = torch.zeros(eval_iters)
        auxes = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split_name, train_data, val_data, batch_size, block_size, device)
            _, loss, aux = model(x, y)
            losses[k] = loss.item()
            auxes[k] = aux.item()
        out[split_name] = losses.mean().item()
        out[f"{split_name}_aux"] = auxes.mean().item()
    if was_training:
        model.train()
    return out


@torch.no_grad()
def expert_load(model, train_data, val_data, batch_size, block_size, device):
    """Per-layer expert fractions on one validation batch. Empty for the dense GPT.

    Each row sums to 1. A row is the fraction of (token, slot) assignments,
    with two slots per token.
    """
    if not model.moe:
        return []
    was_training = model.training
    model.eval()
    x, _ = get_batch("val", train_data, val_data, batch_size, block_size, device)
    model(x)
    fractions = [[float(value) for value in row.tolist()] for row in model.expert_fractions]
    if was_training:
        model.train()
    return fractions


def format_fractions(fractions):
    lines = []
    for layer, row in enumerate(fractions):
        body = " ".join(f"{value:.3f}" for value in row)
        lines.append(f"layer {layer} experts {body}")
    return "\n".join(lines)


def print_verification(model):
    """Log the cache check before any weight update.

    The check is about the forward, not about Shakespeare. It uses random
    token ids and a private generator so the training seed still starts at 1337.
    """
    report = verify_cache(model)
    report["parameters"] = model.num_parameters()
    report["active_parameters"] = model.num_active_parameters()
    print(
        f"verify prefill gap {report['prefill_gap']:.3e}  "
        f"decode gap {report['decode_gap']:.3e}  "
        f"prefill token-layers {report['prefill_encoder_token_layers']}  "
        f"decode token-layers {report['decode_token_layers']}  "
        f"kv bytes/token {report['kv_bytes_per_token']}  "
        f"parameters {report['parameters']:,}  "
        f"active {report['active_parameters']:,}"
    )
    return report


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
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model {model_name}")

    config = dict(PRESETS[preset])
    config["moe"] = model_name == "moe"
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
    active_parameters = model.num_active_parameters()
    learning_rate = config["learning_rate"]
    optimizer = build_optimizer(model, learning_rate, config["weight_decay"])
    print(
        f"model {model_name}  preset {preset}  device {device}  "
        f"parameters {num_parameters:,}  active {active_parameters:,}  "
        f"steps {config['max_iters']}  lr {learning_rate:g}"
    )
    verification = print_verification(model)

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
            fractions = expert_load(
                model,
                train_data,
                val_data,
                config["batch_size"],
                config["block_size"],
                device,
            )
            row = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "train_aux": losses["train_aux"],
                "val_aux": losses["val_aux"],
                "expert_fractions": fractions,
            }
            history.append(row)
            if model.moe:
                print(
                    f"step {step}: train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}, "
                    f"train aux {losses['train_aux']:.4f}, val aux {losses['val_aux']:.4f}"
                )
                print(format_fractions(fractions))
            else:
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
        _, cross_entropy, aux = model(xb, yb)
        # The printed loss stays cross-entropy so it compares with experiment 005.
        # The auxiliary term is what keeps the router from collapsing.
        loss = cross_entropy + config["aux_loss_coef"] * aux
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
            "verification": verification,
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
        "active_parameters": active_parameters,
        "elapsed_seconds": elapsed,
        "device": device,
        "config": config,
        "learning_rate": learning_rate,
        "verification": verification,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True, help="dense gpt or moe")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="laptop")
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
