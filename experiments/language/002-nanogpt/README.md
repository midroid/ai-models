# 002 — nanoGPT

A character-level GPT trained on Tiny Shakespeare, in the style of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (MIT). It continues text one character at a time.

Experiment 001 builds the same idea as separate attention heads with ReLU and an untied output layer. This experiment uses nanoGPT's training model:

- one linear map produces query, key, and value, then splits into heads
- GELU in the feed-forward network
- the token embedding and the output head share weights
- linear warmup, then cosine decay, and gradient clipping at 1.0

OpenWebText and the multi-GPU GPT-2 reproduction stay upstream. This run is Tiny Shakespeare only.

## Setup

This experiment has its own uv project. From this directory:

```bash
uv sync
uv run python -m src.train --preset laptop
uv run python -m src.sample --checkpoint runs/gpt-laptop.pt --prompt "ROMEO:"
```

## Presets

| preset | block | batch | layers | heads | width | steps | dropout | learning rate | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| laptop | 64 | 12 | 4 | 4 | 128 | 2000 | 0.0 | 1e-3 down to 1e-4, warmup 100 | This machine (MPS or CPU) |
| shakespeare | 256 | 64 | 6 | 6 | 384 | 5000 | 0.2 | 1e-3 down to 1e-4, warmup 100 | Upstream `train_shakespeare_char.py` |

`shakespeare` is the config that reaches about 1.47 validation loss in a few minutes on an A100. It is not the run recorded below.

## Results

Laptop preset, Apple MPS, seed 1337, 2000 steps, 29.7 seconds. The model has 809,856 parameters. Tiny Shakespeare is 1,115,394 characters, vocabulary 65, so a uniform guess scores `ln(65)` = 4.17. The split is 1,003,854 train tokens and 111,540 validation tokens.

| step | train loss | val loss | learning rate |
| --- | --- | --- | --- |
| 0 | 4.209 | 4.203 | 1e-5 |
| 1999 | 1.794 | 1.919 | 1e-4 |

The learning rate warms up to 1e-3 and then cosines down to 1e-4. Validation stays a little above training at the end. Checkpoint: `runs/gpt-laptop.pt`.

Continuation from `ROMEO:`, 220 new characters:

```text
ROMEO:
MyRYTUSS:
Nay:
Who have love ais your coming, go?

Nyecp, and businet: thou we worby
Your live thas stords mity mather fid, it you for,
Ansermans, you my of off wropetys,
Of it quidy, whe no rom in fraiusht?

SLEO:
Obbl
```

The words are invented, and speaker labels and line breaks already show up. The walkthrough notebook trains the same preset for 400 steps, plots the curve, and writes `runs/gpt-walkthrough.pt`. On that shorter schedule validation loss ends at 2.36.
