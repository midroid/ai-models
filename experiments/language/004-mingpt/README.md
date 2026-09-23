# 004 minGPT

A character-level language model in the style of [karpathy/minGPT](https://github.com/karpathy/minGPT) (MIT). The source here is our rewrite of `mingpt/model.py` and `mingpt/trainer.py`, trained the way `projects/chargpt` trains: next character on a text file. The file is the same Tiny Shakespeare as experiment 001.

minGPT is the earlier library. Experiment 002 is the later nanoGPT character model. This one keeps the pieces that library is built around: a learned position embedding, an untied output head, explicit attention, the tanh GELU, and a `Trainer`.

## What is implemented

| module | role |
| --- | --- |
| `src/model.py` | `wte`, `wpe`, pre-norm blocks, `NewGELU`, untied `lm_head` |
| `src/trainer.py` | batches, AdamW, grad clip 1.0, device selection |
| `src/train.py` | character dataset and the smoke preset |
| `src/sample.py` | continue a prompt from a checkpoint |

`--model_type` is the size dial. `gpt-micro` is 4 layers, 4 heads, 128-d. Upstream chargpt uses `gpt-mini` (6 layers, 6 heads, 192-d) and never sets `max_iters`. That preset is not the recorded run.

Weight decay 0.1 applies to `Linear` weights. Biases, LayerNorm, and the embedding tables are not decayed. Residual `c_proj` weights are initialized at `0.02/sqrt(2*n_layer)`.

Device order is CUDA, then Apple MPS, then CPU. `pin_memory` is on only for CUDA. The smoke uses `num_workers=0`. `torch.compile` stays off.

## Left out

`GPT.from_pretrained` copies Hugging Face GPT-2 weights and needs `transformers`. `mingpt/bpe.py` exists for that path. `projects/adder` trains a model to add numbers. None of those run here.

## Setup

```bash
cd experiments/language/004-mingpt
uv sync
uv run python -m src.train --preset smoke
uv run python -m src.sample --checkpoint runs/mingpt-smoke.pt --prompt "ROMEO:"
```

The checkpoint is gitignored.

## Smoke settings

| knob | upstream chargpt | this smoke |
| --- | --- | --- |
| model | `gpt-mini` | `gpt-micro` |
| block | 128 | 64 |
| batch | 64 | 12 |
| dropout | 0.1 | 0 |
| learning rate | 5e-4 | 5e-4 |
| steps | until stopped | 400 |
| data workers | 4 | 0 |

## Recorded smoke

Run on this Mac. The script printed `device mps`. Wall clock was 9.5 seconds. `gpt-micro` is 809,856 transformer parameters and 818,176 once the untied head is counted. Vocabulary is 65 characters.

| step | train loss |
| --- | --- |
| 1 | 4.2144 |
| 200 | 2.4489 |
| 400 | 2.3653 |

Checkpoint: `runs/mingpt-smoke.pt`.

Prompt `ROMEO:`, 200 new characters, temperature 1, top-k 10:

```text
ROMEO:
Hy watis'd, ther thasth as at this thest seangls thouet t conins one fato thou
Chicksang, shere orit or wereson at myooung, shirt.

Whyast stte man anout soure thal warrant by oumard than wit.

Wowns
```

The sample is rough. Four hundred steps are enough to show the loss falling and a generation coming back.
