# 010 — Hindi–Sanskrit decoder

A 10,942,720-parameter decoder for Hindi and Sanskrit. Width 256, 12 layers, 8 query heads and 2 key-value heads, SwiGLU width 640, tied embeddings, RMSNorm, and rotary positions with the same pair rotation as experiment 008. Vocabulary is 12,000, the Unigram size with the lowest tokens per word on the cleaned Hindi and Sanskrit holdout.

`tokens_seen` counts target positions whose label is not -100. Pretraining writes `runs/pretrain/tok-100m`, then `tok-200m`, and so on. Each of those directories can be supervised-finetuned on its own. The fine-tune loads weights only and starts a new AdamW.

## Setup

This experiment has its own uv project. From this directory:

```bash
uv sync
uv run python -m src.smoke
uv run pytest -v
uv run python -m src.train --stage overfit --max-steps 40 --block-size 32 --batch-size 4
```

The first real pretrain uses the cleaned JSONL and `tokenizers/sp.model`:

```bash
uv run python scripts/compare_tokenizers.py
uv run python -m src.train --stage pretrain --until-tokens 100000000
uv run python -m src.train --stage pretrain --resume runs/pretrain/latest --until-tokens 200000000
uv run python -m src.train --stage sft --init runs/pretrain/tok-100m
uv run python -m src.evaluate --checkpoint runs/sft/from-tok-100m
```

`compare_tokenizers.py` trains Unigram models at 6k, 8k, and 12k on a balanced slice of `data/clean/pretrain.jsonl` and writes `tokenizers/compare.json`. The chosen model is `tokenizers/sp.model` (12k). Pretraining rewrites `runs/pretrain/latest` every 500 steps and again when the run stops, and still writes `tok-100m`, `tok-200m`, and so on at each 100M-token boundary. Resume either a boundary directory or `latest`.

The fixture corpus is too small for a 12,000-piece tokenizer. Tests that pass `fixture.model` train a small tokenizer on first use. Smoke always checks the full 10,942,720-parameter model.

Prepare the training files from Wikipedia and the Samasāmayik release:

```bash
uv run python scripts/prepare_data.py --hi-words 1000000 --sa-words 1000000
```

That writes `data/clean/pretrain.jsonl`, `data/clean/sft.jsonl`, and `data/manifests/sources.csv`. Every source is marked with a snapshot or prepared date and a quality (`gold`, `silver`, or `unreviewed`). IN22 and FLORES are CC-BY-SA-4.0, so a later prepare includes them as silver pairs. The official Samasāmayik test stays held out. Sangraha stays out until its license is reviewed. The file already in use by the current fine-tune is left as-is; the new pairs are in `data/clean/sft_cc_by_sa.jsonl` and are loaded on the next supervised run.

## Smoke

`src.smoke` fails if the parameter count is not 10,942,720, if the embedding and the output head do not share storage, if a cached prefill or decode differs from a full forward by more than 1e-4, if query or key shapes are wrong, or if the loss or a gradient is non-finite.
