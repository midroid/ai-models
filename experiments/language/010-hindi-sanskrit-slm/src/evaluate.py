"""Fixture cross-entropy and chrF++ against a copy baseline.

    uv run python -m src.evaluate --checkpoint runs/sft/from-tok-100m
"""

import argparse
from pathlib import Path

import torch

from src.data import encode_sft, read_jsonl
from src.sample import continue_prompt, load_checkpoint
from src.train import FIXTURE_SFT, pick_device


def copy_chrf(pairs):
    import sacrebleu

    hypotheses = [pair["source"] for pair in pairs]
    references = [pair["target"] for pair in pairs]
    return sacrebleu.corpus_chrf(hypotheses, [references]).score


def _translation_pairs(rows):
    return [row for row in rows if row["task"] in ("translate_hi_sa", "translate_sa_hi")]


def cross_entropy(model, tokenizer, rows, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for row in rows:
            ids, labels = encode_sft(row, tokenizer)
            if len(ids) < 2 or len(ids) > model.block_size:
                continue
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            targets = torch.tensor([labels], dtype=torch.long, device=device)
            _, loss = model(inputs, targets)
            if torch.isfinite(loss):
                losses.append(loss.item())
    if not losses:
        raise ValueError("no rows produced a loss")
    return sum(losses) / len(losses)


def model_chrf(model, tokenizer, rows, device):
    import sacrebleu

    hypotheses = []
    references = []
    for row in rows:
        text, _ = continue_prompt(
            None,
            row["task"],
            row["source"],
            max_new_tokens=32,
            temperature=0.0,
            device=device,
            model=model,
            tokenizer=tokenizer,
        )
        marker = "<tgt>"
        generated = text.split(marker, 1)[1] if marker in text else text
        hypotheses.append(generated)
        references.append(row["target"])
    return sacrebleu.corpus_chrf(hypotheses, [references]).score


def evaluate(checkpoint, data_path=FIXTURE_SFT, device=None):
    device = device or pick_device()
    rows = read_jsonl(data_path)
    if any(row.get("split") == "val" for row in rows):
        rows = [row for row in rows if row.get("split") == "val"]
    model, tokenizer, _, device = load_checkpoint(checkpoint, device=device)
    loss = cross_entropy(model, tokenizer, rows, device)
    pairs = _translation_pairs(rows)
    report = {"rows": len(rows), "cross_entropy": loss, "copy_chrf": copy_chrf(pairs)}
    try:
        report["model_chrf"] = model_chrf(model, tokenizer, pairs, device)
    except Exception as exc:
        report["model_chrf"] = None
        report["model_chrf_error"] = str(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=FIXTURE_SFT)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.data)
    for key, value in report.items():
        print(f"{key} {value}")


if __name__ == "__main__":
    main()
