"""Train the Hindi-Sanskrit decoder.

From the experiment directory:

    uv run python -m src.train --stage overfit
    uv run python -m src.train --stage pretrain --until-tokens 100000000
    uv run python -m src.train --stage pretrain --resume runs/pretrain/tok-100m --until-tokens 200000000
    uv run python -m src.train --stage sft --init runs/pretrain/tok-100m

Pretraining saves `runs/pretrain/tok-100m` each time `tokens_seen` crosses a
multiple of `--save-every-tokens` (default 100M). It also rewrites
`runs/pretrain/latest` every `--save-every-steps` and again at the end, so a
run can resume before the next 100M boundary. Supervised fine-tuning loads
weights only and starts a new AdamW. `tokens_seen` counts target positions
whose label is not -100.
"""

import argparse
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

from src.data import file_sha256, pretrain_examples, read_jsonl, sft_examples, supervised_count
from src.model import MODEL_10M, PARAMETER_COUNT, build_model, unique_parameters, verify_cache
from src.tokenizer import Tokenizer, train_tokenizer, write_model_bytes

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = EXPERIMENT_DIR / "configs"
FIXTURE_PRETRAIN = EXPERIMENT_DIR / "data" / "fixture" / "pretrain.jsonl"
FIXTURE_SFT = EXPERIMENT_DIR / "data" / "fixture" / "sft.jsonl"
CLEAN_PRETRAIN = EXPERIMENT_DIR / "data" / "clean" / "pretrain.jsonl"
CLEAN_SFT = EXPERIMENT_DIR / "data" / "clean" / "sft.jsonl"
DEFAULT_RUNS = EXPERIMENT_DIR / "runs"
DEFAULT_TOKENIZER = EXPERIMENT_DIR / "tokenizers" / "sp.model"
GAP_LIMIT = 1e-4


def _format_duration(seconds):
    if seconds == float("inf") or seconds < 0:
        return "unknown"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def pick_device():
    """CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def boundary_name(tokens):
    if tokens % 1_000_000 == 0:
        return f"tok-{tokens // 1_000_000}m"
    return f"tok-{tokens}"


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=EXPERIMENT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


class CosineSchedule:
    """Linear warmup, then cosine down to the floor. Step is an optimizer step."""

    def __init__(self, peak, floor, warmup_steps, total_steps, step=0):
        self.peak = peak
        self.floor = floor
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, 1)
        self.step = step

    def lr(self):
        if self.step < self.warmup_steps:
            return self.peak * float(self.step + 1) / float(max(self.warmup_steps, 1))
        span = self.total_steps - self.warmup_steps
        if span <= 0:
            return self.floor
        progress = min(1.0, max(0.0, (self.step - self.warmup_steps) / span))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.floor + (self.peak - self.floor) * cosine

    def advance(self):
        self.step += 1

    def state_dict(self):
        return {
            "peak": self.peak,
            "floor": self.floor,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "step": self.step,
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            peak=state["peak"],
            floor=state["floor"],
            warmup_steps=state["warmup_steps"],
            total_steps=state["total_steps"],
            step=state["step"],
        )


def build_optimizer(model, learning_rate, weight_decay, beta1, beta2):
    """AdamW that decays matrices only."""
    decay, no_decay = [], []
    for param in unique_parameters(model):
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
        betas=(beta1, beta2),
    )


def capture_rng():
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def ensure_fixture_tokenizer(model_path, corpus_rows, vocab_size):
    model_path = Path(model_path)
    if model_path.exists():
        return Tokenizer(model_path)
    corpus = model_path.parent / "fixture_corpus.txt"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    from src.data import balanced_pretrain_text

    corpus.write_text(balanced_pretrain_text(corpus_rows), encoding="utf-8")
    trained = train_tokenizer(corpus, model_path.with_suffix(""), vocab_size)
    return Tokenizer(trained)


def _batch(examples, start, batch_size, device, pad_id):
    chosen = []
    index = start
    for _ in range(batch_size):
        chosen.append(examples[index % len(examples)])
        index += 1
    width = max(len(row[0]) for row in chosen)
    input_rows = []
    label_rows = []
    for ids, labels in chosen:
        pad_n = width - len(ids)
        input_rows.append(ids + [pad_id] * pad_n)
        label_rows.append(labels + [-100] * pad_n)
    inputs = torch.tensor(input_rows, dtype=torch.long, device=device)
    labels = torch.tensor(label_rows, dtype=torch.long, device=device)
    return inputs, labels, index


def _save(directory, payload, tokenizer):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_model_bytes(tokenizer.model_bytes(), directory / "tokenizer.model")
    torch.save(payload, directory / "checkpoint.pt")
    print(f"wrote {directory}")


def _load_checkpoint(directory):
    directory = Path(directory)
    path = directory / "checkpoint.pt" if directory.is_dir() else directory
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    tokenizer_path = path.parent / "tokenizer.model"
    if not tokenizer_path.exists():
        write_model_bytes(checkpoint["tokenizer_model"], tokenizer_path)
    return checkpoint, Tokenizer(tokenizer_path), path.parent


def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train(
    stage,
    data_path,
    runs_dir=DEFAULT_RUNS,
    until_tokens=None,
    save_every_tokens=None,
    save_every_steps=None,
    resume=None,
    init=None,
    tokenizer_path=DEFAULT_TOKENIZER,
    model_overrides=None,
    train_overrides=None,
    max_steps=None,
    batch_size=None,
    block_size=None,
    loss_stop=None,
    enforce_parameter_count=True,
    seed=1337,
    device=None,
):
    if stage not in ("overfit", "pretrain", "sft"):
        raise ValueError(f"unknown stage {stage}")
    device = device or pick_device()
    model_config = dict(MODEL_10M)
    model_config.update(load_yaml(CONFIG_DIR / "model_10m.yaml"))
    train_config = load_yaml(CONFIG_DIR / ("sft.yaml" if stage == "sft" else "pretrain.yaml"))
    if model_overrides:
        model_config.update(model_overrides)
    if train_overrides:
        train_config.update(train_overrides)
    if until_tokens is None:
        until_tokens = train_config["until_tokens"]
    if save_every_tokens is None:
        save_every_tokens = train_config.get("save_every_tokens", until_tokens)
    if save_every_steps is None:
        save_every_steps = train_config.get("save_every_steps")
    if batch_size is not None:
        train_config["batch_size"] = batch_size
    if block_size is not None:
        model_config["block_size"] = block_size

    if stage == "overfit":
        model_config["dropout"] = 0.0
        loss_stop = 0.1 if loss_stop is None else loss_stop
        data_path = data_path or FIXTURE_SFT

    if data_path is None:
        if stage == "overfit":
            data_path = FIXTURE_SFT
        elif stage == "sft":
            data_path = CLEAN_SFT if CLEAN_SFT.exists() else FIXTURE_SFT
        else:
            data_path = CLEAN_PRETRAIN if CLEAN_PRETRAIN.exists() else FIXTURE_PRETRAIN
    data_path = Path(data_path)
    rows = read_jsonl(data_path)
    extra = CLEAN_SFT.parent / "sft_cc_by_sa.jsonl"
    if stage == "sft" and extra.exists():
        seen = {(row.get("task"), row.get("source"), row.get("target")) for row in rows}
        for row in read_jsonl(extra):
            key = (row.get("task"), row.get("source"), row.get("target"))
            if key not in seen:
                rows.append(row)
                seen.add(key)
    if stage == "sft":
        trainable = [row for row in rows if row.get("split", "train") != "val"]
        if trainable:
            rows = trainable
    data_hash = file_sha256(data_path)

    resume_checkpoint = None
    init_checkpoint = None
    if resume:
        resume_checkpoint, tokenizer, _ = _load_checkpoint(resume)
        if resume_checkpoint["stage"] != stage:
            raise ValueError(f"{stage} resume expects a {stage} checkpoint")
        if resume_checkpoint["data_sha256"] != data_hash:
            raise ValueError("data file changed since the checkpoint")
        model_config = resume_checkpoint["model_config"]
    else:
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            if tokenizer_path.name != "fixture.model":
                raise FileNotFoundError(
                    f"missing tokenizer {tokenizer_path}. "
                    "Train one with scripts/compare_tokenizers.py."
                )
            corpus_rows = read_jsonl(FIXTURE_PRETRAIN)
            from src.data import balanced_pretrain_text

            text = balanced_pretrain_text(corpus_rows)
            chars = {ch for ch in text if not ch.isspace()}
            vocab = len(src_tokenizer_vocab(chars))
            ensure_fixture_tokenizer(tokenizer_path, corpus_rows, vocab)
        tokenizer = Tokenizer(tokenizer_path)

    if tokenizer.vocab_size > model_config["vocab_size"]:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} exceeds model vocab {model_config['vocab_size']}"
        )

    torch.manual_seed(seed)
    model = build_model(model_config).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        model.tie_weights()
    elif init:
        init_checkpoint, _, init_dir = _load_checkpoint(init)
        if init_checkpoint["stage"] == "sft":
            raise ValueError("sft --init expects a pretrain checkpoint")
        model.load_state_dict(init_checkpoint["model"])
        model.tie_weights()
    else:
        init_checkpoint = None
        init_dir = None

    parameter_count = model.num_parameters()
    full_model = model_config["vocab_size"] == MODEL_10M["vocab_size"] and not model_overrides
    if enforce_parameter_count and full_model and parameter_count != PARAMETER_COUNT:
        raise SystemExit(f"parameter count {parameter_count} is not {PARAMETER_COUNT:,}")

    tokens_per_step_guess = train_config["batch_size"] * model_config["block_size"]
    if stage == "overfit":
        total_steps = max_steps or 300
        warmup_steps = 1
    else:
        total_steps = max(1, math.ceil(until_tokens / tokens_per_step_guess))
        warmup_steps = max(1, int(train_config["warmup_ratio"] * total_steps))
    schedule = CosineSchedule(
        peak=train_config["peak_lr"],
        floor=train_config["min_lr"],
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    optimizer = build_optimizer(
        model,
        schedule.lr(),
        train_config["weight_decay"],
        train_config["beta1"],
        train_config["beta2"],
    )

    tokens_seen = 0
    block_index = 0
    if resume_checkpoint is not None:
        tokens_seen = resume_checkpoint["tokens_seen"]
        block_index = resume_checkpoint["block_index"]
        schedule = CosineSchedule.from_state(resume_checkpoint["schedule"])
        schedule.total_steps = max(schedule.total_steps, total_steps)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        _optimizer_to_device(optimizer, device)
        restore_rng(resume_checkpoint["rng"])

    log_path = Path(runs_dir) / stage / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    log(
        f"stage {stage}  device {device}  parameters {parameter_count:,}  "
        f"vocab {tokenizer.vocab_size}  tokenizer {tokenizer.model_path}"
    )
    log(
        f"data {data_path}  sha256 {data_hash}  rows {len(rows)}  "
        f"batch {train_config['batch_size']}  block {model_config['block_size']}"
    )
    log(
        f"until {until_tokens}  warmup_steps {schedule.warmup_steps}  "
        f"total_steps {schedule.total_steps}  peak_lr {train_config['peak_lr']}  "
        f"min_lr {train_config['min_lr']}  save_every_steps {save_every_steps}  "
        f"save_every_tokens {save_every_tokens}"
    )
    log("packing examples")
    if stage == "pretrain":
        examples = pretrain_examples(rows, tokenizer, model_config["block_size"])
    else:
        repeat_to = 100 if stage == "overfit" else 1
        examples = sft_examples(rows, tokenizer, model_config["block_size"], repeat_to=repeat_to)
    log(f"packed {len(examples)} examples  tokens_seen {tokens_seen}")
    if stage != "overfit" and resume_checkpoint is None and examples:
        sample = examples[:32]
        mean_supervised = sum(supervised_count(labels) for _, labels in sample) / len(sample)
        per_step = max(1.0, mean_supervised * train_config["batch_size"])
        schedule.total_steps = max(1, math.ceil(until_tokens / per_step))
        schedule.warmup_steps = max(1, int(train_config["warmup_ratio"] * schedule.total_steps))
        log(
            f"schedule fit  supervised_per_step {per_step:.0f}  "
            f"warmup_steps {schedule.warmup_steps}  total_steps {schedule.total_steps}"
        )

    init_path = str(init) if init else ""
    init_tokens = None if init_checkpoint is None else init_checkpoint["tokens_seen"]

    def snapshot(stage_name, init_ckpt, init_tokens_seen):
        return _payload(
            stage=stage_name,
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            tokens_seen=tokens_seen,
            block_index=block_index,
            data_hash=data_hash,
            tokenizer=tokenizer,
            model_config=model_config,
            train_config=train_config,
            init_checkpoint=init_ckpt,
            init_tokens_seen=init_tokens_seen,
        )

    def save_latest():
        if stage == "sft":
            payload = snapshot("sft", init_path, init_tokens)
        else:
            payload = snapshot(stage, None, None)
        _save(Path(runs_dir) / stage / "latest", payload, tokenizer)

    model.train()
    step = schedule.step
    started = time.perf_counter()
    tokens_at_start = tokens_seen
    last_log_time = started
    last_log_tokens = tokens_seen
    last_loss = None
    nan_streak = 0
    next_boundary = (tokens_seen // save_every_tokens + 1) * save_every_tokens
    try:
        while True:
            if stage != "overfit" and tokens_seen >= until_tokens:
                break
            if max_steps is not None and step >= (resume_checkpoint or {}).get("optimizer_step", 0) + max_steps:
                break
            if stage == "overfit" and max_steps is not None and step >= max_steps:
                break
            inputs = labels = None
            try:
                inputs, labels, block_index = _batch(
                    examples, block_index, train_config["batch_size"], device, tokenizer.pad_id
                )
                learning_rate = schedule.lr()
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                _, loss = model(inputs, labels)
                if not torch.isfinite(loss):
                    nan_streak += 1
                    optimizer.zero_grad(set_to_none=True)
                    log(
                        f"non-finite loss at step {step}  tokens {tokens_seen}  "
                        f"batch {tuple(inputs.shape)}  streak {nan_streak}"
                    )
                    if nan_streak >= 3:
                        raise RuntimeError(f"loss is not finite: {loss.detach()}")
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    unique_parameters(model), train_config["grad_clip"]
                )
                if not torch.isfinite(grad_norm):
                    nan_streak += 1
                    optimizer.zero_grad(set_to_none=True)
                    log(
                        f"non-finite gradient at step {step}  tokens {tokens_seen}  "
                        f"batch {tuple(inputs.shape)}  streak {nan_streak}"
                    )
                    if nan_streak >= 3:
                        raise RuntimeError(f"gradient norm is not finite: {grad_norm}")
                    continue
                nan_streak = 0
                optimizer.step()
            except Exception as exc:
                shape = tuple(inputs.shape) if inputs is not None else None
                log(
                    f"error step {step}  tokens {tokens_seen}  batch {shape}  "
                    f"{type(exc).__name__}: {exc}"
                )
                log(traceback.format_exc().rstrip())
                raise
            schedule.advance()
            step = schedule.step
            capacity = inputs.shape[0] * max(inputs.shape[1] - 1, 0)
            gained = int((labels[:, 1:] != -100).detach().to("cpu").sum().item())
            if gained < 0 or gained > capacity:
                log(
                    f"error step {step}  tokens {tokens_seen}  "
                    f"supervised count {gained} outside 0..{capacity}"
                )
                raise RuntimeError(f"supervised token count {gained} is outside 0..{capacity}")
            if gained == 0:
                log(f"error step {step}  tokens {tokens_seen}  batch {tuple(inputs.shape)} has no supervised labels")
                raise RuntimeError("batch has no supervised labels; increase block_size")
            tokens_seen += gained
            if device == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            last_loss = float(loss.detach())
            grad_value = float(grad_norm.detach()) if torch.is_tensor(grad_norm) else float(grad_norm)
            if step % train_config["eval_interval"] == 0 or step == 1:
                now = time.perf_counter()
                elapsed = now - started
                gained_since_start = tokens_seen - tokens_at_start
                rate = gained_since_start / elapsed if elapsed > 0 else 0.0
                interval = now - last_log_time
                interval_rate = (tokens_seen - last_log_tokens) / interval if interval > 0 else 0.0
                remaining = max(0, until_tokens - tokens_seen)
                eta = remaining / rate if rate > 0 else float("inf")
                percent = 100.0 * tokens_seen / until_tokens if until_tokens else 0.0
                log(
                    f"step {step}  tokens {tokens_seen}/{until_tokens} ({percent:.2f}%)  "
                    f"loss {last_loss:.4f}  lr {learning_rate:.3e}  "
                    f"grad {grad_value:.3f}  batch {tuple(inputs.shape)}  "
                    f"supervised {gained}  tok/s {interval_rate:.0f}  "
                    f"avg_tok/s {rate:.0f}  elapsed {_format_duration(elapsed)}  "
                    f"eta {_format_duration(eta)}"
                )
                last_log_time = now
                last_log_tokens = tokens_seen
            if stage == "pretrain":
                while tokens_seen >= next_boundary and next_boundary <= until_tokens:
                    boundary = Path(runs_dir) / "pretrain" / boundary_name(next_boundary)
                    _save(boundary, snapshot("pretrain", None, None), tokenizer)
                    log(f"saved boundary {boundary}  step {step}  tokens {tokens_seen}")
                    next_boundary += save_every_tokens
            if save_every_steps and stage != "overfit" and step % int(save_every_steps) == 0:
                save_latest()
                log(f"saved latest  step {step}  tokens {tokens_seen}")
            if stage == "overfit" and last_loss <= loss_stop:
                log(f"overfit loss {last_loss:.4f}")
                break
            if stage == "overfit" and step >= (max_steps or 300):
                break
    finally:
        log_file.close()

    if stage == "pretrain" and tokens_seen > 0:
        save_latest()
    if stage == "sft":
        init_name = Path(init).name if init else "scratch"
        _save(
            Path(runs_dir) / "sft" / f"from-{init_name}",
            snapshot("sft", init_path, init_tokens),
            tokenizer,
        )
        save_latest()
    if stage == "overfit":
        _save(
            Path(runs_dir) / "overfit",
            _payload(
                stage="overfit",
                model=model,
                optimizer=optimizer,
                schedule=schedule,
                tokens_seen=tokens_seen,
                block_index=block_index,
                data_hash=data_hash,
                tokenizer=tokenizer,
                model_config=model_config,
                train_config=train_config,
                init_checkpoint=None,
                init_tokens_seen=None,
            ),
            tokenizer,
        )
    elapsed = time.perf_counter() - started
    log_file = log_path.open("a", encoding="utf-8")
    try:
        log(
            f"done stage {stage}  loss {last_loss}  tokens {tokens_seen}  "
            f"steps {step}  elapsed {_format_duration(elapsed)}"
        )
    finally:
        log_file.close()
    return {"loss": last_loss, "tokens_seen": tokens_seen, "parameters": parameter_count, "step": step}


def src_tokenizer_vocab(chars):
    """Vocab large enough for the fixture characters plus the reserved pieces."""
    return range(max(64, len(chars) + 32))


def _payload(
    stage,
    model,
    optimizer,
    schedule,
    tokens_seen,
    block_index,
    data_hash,
    tokenizer,
    model_config,
    train_config,
    init_checkpoint,
    init_tokens_seen,
):
    return {
        "stage": stage,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": schedule.state_dict(),
        "tokens_seen": tokens_seen,
        "block_index": block_index,
        "optimizer_step": schedule.step,
        "data_sha256": data_hash,
        "tokenizer_model": tokenizer.model_bytes(),
        "model_config": model_config,
        "train_config": train_config,
        "init_checkpoint": init_checkpoint,
        "init_tokens_seen": init_tokens_seen,
        "rng": capture_rng(),
        "git_sha": git_sha(),
        "python": sys.version,
        "torch": torch.__version__,
    }


def cache_check(model):
    report = verify_cache(model)
    if report["prefill_gap"] > GAP_LIMIT or report["decode_gap"] > GAP_LIMIT:
        raise AssertionError(
            f"cache gap prefill {report['prefill_gap']:.3e} decode {report['decode_gap']:.3e}"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("overfit", "pretrain", "sft"), required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--until-tokens", type=int, default=None)
    parser.add_argument("--save-every-tokens", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loss-stop", type=float, default=None)
    args = parser.parse_args()
    train(
        stage=args.stage,
        data_path=args.data,
        runs_dir=args.runs_dir,
        until_tokens=args.until_tokens,
        save_every_tokens=args.save_every_tokens,
        save_every_steps=args.save_every_steps,
        resume=args.resume,
        init=args.init,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_steps=args.max_steps,
        loss_stop=args.loss_stop,
    )


if __name__ == "__main__":
    main()
