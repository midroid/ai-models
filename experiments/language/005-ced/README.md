# 005 — causal encoder-decoder

A character-level language model on Tiny Shakespeare. One module trains two modes: a decoder-only GPT, and the same stack split into a causal encoder and a decoder.

The inference shape is reimplemented from [karpathy/nanochat](https://github.com/karpathy/nanochat): separate query, key, and value projections, rotary positions, RMSNorm, SDPA, and a KV cache. Experiment 003 pins that repository and does not change it. This experiment does not vendor that checkout.

The architecture split follows DeepSeek-V4.1-Flash, [section 2.2](https://arxiv.org/html/2609.19969). The bottom half of the layers is the encoder. Each decoder layer keeps its own query, and projects its keys and values from the encoder output `H`:

```text
K_l, V_l = H W_l
```

Training runs every layer on the whole block, because every character needs a loss. Prefill runs the encoder on the whole prompt and the decoder only on the last character. Decode runs all four layers on the new character. There is no sliding window, so those cached logits match a full forward. The paper's window would make that match approximate.

Left out, because they do not change this measurement: FlashAttention 3, Muon, value embeddings, smear, residual lambdas, logit softcap, BPE, ClimbMix, SFT, and the chat CLI. The feed-forward network is GELU. The text, the 90/10 cut, and AdamW follow experiment 002. The learning rate here is a constant `3e-4`, with weight decay 0.1 on matrices and gradient clipping at 1.0.

## Reading the code

| file | what to look at |
| --- | --- |
| `src/model.py` | rotary attention, the encoder/decoder split, `prefill` and `decode_step` |
| `src/train.py` | 90/10 cut, the cache check printed before the first update, loss printed before the step's update |
| `src/sample.py` | prefill once, then one decode step per new character, with token-layers and seconds |
| `src/smoke.py` | logit gaps, token-layer counts, then 20 training steps |

A token-layer is one token through one block. Projecting decoder keys for the rest of the prompt is not a token-layer. KV bytes per token match in the two modes: each layer still stores its own keys and values. The paper's smaller cache is CSA2 and FP4, which are not in this model.

## Prefill and decode

A prompt is one forward. Each new character is a smaller forward that reads the cache.

| phase | GPT | CED |
| --- | --- | --- |
| prompt of N characters | all 4 layers on all N | encoder (layers 0 and 1) on all N, then decoder (layers 2 and 3) on the last character only |
| each new character | all 4 layers on that character | all 4 layers on that character |

`GPT.forward` is the training path and the reference the smoke compares against. It runs the decoder on every character, because every character needs a next-character loss. `GPT._infer` is the cache path. In CED mode it still writes a decoder key and value for every prompt character, by multiplying `H` by that layer's key and value weights. It runs the decoder attention and MLP only at the last character.

Those two paths agree at that last character. A decoder query reads keys made from `H`. It does not read the decoder residual at any earlier character, so skipping those characters does not change the logits. The smoke measures that gap and fails above `1e-4`.

Rotary position is applied to queries and keys, not to values. The table has one row per character up to `block_size`, which is why sampling stops there.

## Setup

This experiment has its own uv project. From this directory:

```bash
uv sync
uv run python -m src.smoke
uv run python -m src.train --model gpt --preset laptop
uv run python -m src.train --model ced --preset laptop
uv run python -m src.sample --checkpoint runs/gpt-laptop.pt --prompt "ROMEO:"
```

The text file is `../002-nanogpt/data/input.txt`. Checkpoints are gitignored.

## Smoke

`src.smoke` checks both modes, then trains each for 20 steps. It exits non-zero if a logit gap is above `1e-4`, or if a CED prefill runs the decoder on more than the last character.

Run on this Mac, device `mps`. The prompt in the check is 8 random characters. Both modes have 803,072 parameters and 4,096 KV bytes per token.

| mode | prefill gap | decode gap | prefill token-layers | decode token-layers |
| --- | --- | --- | --- | --- |
| gpt | 0 | 1.8e-7 | enc 32, dec 0 | 4 |
| ced | 9.7e-8 | 1.5e-7 | enc 16, dec 2 | 4 |

Encoder token-layers for the GPT are `4 * 8`. For CED they are `2 * 8`, and the decoder adds `2 * 1`. A decode step is 4 token-layers in both modes. The 20-step tails reached validation loss about 3.20. The script printed `smoke ok`.

## Results

Laptop preset, Apple MPS, seed 1337, 2000 steps. Both modes have 803,072 parameters. Tiny Shakespeare is 1,115,394 characters, vocabulary 65, so a uniform guess scores `ln(65)` = 4.17. The split is 1,003,854 train tokens and 111,540 validation tokens.

| model | step | train loss | val loss |
| --- | --- | --- | --- |
| gpt | 0 | 4.211 | 4.212 |
| gpt | 1999 | 1.773 | 1.900 |
| ced | 0 | 4.211 | 4.211 |
| ced | 1999 | 1.769 | 1.884 |

The GPT run took 23.1 seconds. The CED run took 23.4 seconds.

Validation stays a little above training at the last step. CED is the same size as the GPT and lands in the same loss range. The prefill count is the compute difference: on a 64-character prompt the GPT runs 256 token-layers and CED runs 130 (128 encoder, 2 decoder). After a few warmup calls, those prefills measured 1.82 ms and 1.10 ms on this Mac.

Checkpoints: `runs/gpt-laptop.pt`, `runs/ced-laptop.pt`.

Sampling stops at `block_size` (64), because the rotary table has that many rows. From `ROMEO:` (6 characters) that is 58 new characters.

GPT:

```text
ROMEO:
Come to olGives, my shis we my venges mike he your,
Towar
```

CED:

```text
ROMEO:
Therefore, nayny for tquors whith awel heart that most;
O
```

The words are invented. The lines already break like verse.
