# 008 — Multi-head latent attention

A character-level language model on Tiny Shakespeare. One module trains two attention modes on the dense GELU GPT from experiment 006. `mha` is that experiment's multi-head attention. `mla` replaces it with the low-rank key-value map from [DeepSeek-V2](https://arxiv.org/abs/2405.04434), section 2.1.

Multi-head attention caches a key and a value for every head. Latent attention compresses both into one vector per token, `c_kv`, of rank 32, then up-projects that vector back to per-head content keys and values. The content keys are not rotated. Rotary position is a second key of width 16, shared by all four heads, and that key is what the cache stores beside the latent. A query is the content query concatenated with a per-head rotary query, so the score is the sum of the two dot products and the scale is `1/sqrt(48)`.

```text
c_kv = W_DKV(x)
content key, value = W_UK(c_kv), W_UV(c_kv)
score = content query · content key + rotary query · rotary key
```

The cache holds `c_kv` and the already-rotated shared key. Decode up-projects every cached latent. Absorbing `W_UK` into the query and `W_UV` into the output projection is the same result with less decode work, and it is left out. Query compression is left out too: it does not change the cache. The feed-forward network stays one GELU MLP of hidden size 512.

## The cache

Multi-head attention stores both a key and a value for each of 4 heads, and each of those is 32 wide. That is `2 * 4 * 32` = 256 floats per layer per token. Latent attention stores the rank-32 latent and one rotary key of width 16, which is 48 floats. Four layers, in float32, are 4,096 bytes against 768.

The latent is not rotated. A rotation depends on the absolute position of the token. If that rotation were folded into `c_kv`, the cached vector would be valid only at the position where it was written, and up-projecting it later would not recover the content key. The rotary key is computed from the residual stream, not from the latent, rotated once at write time, and shared by every head. Each head still has its own rotary query, so the heads can use that one key differently.

```text
per token, per layer:  c_kv (32)  +  rotated shared key (16)
content key and value are rebuilt:  W_UK(c_kv),  W_UV(c_kv)
```

The low-rank map also has fewer parameters. Multi-head Q, K, V, and the output projection are `4 * 128 * 128` = 65,536 per layer. The latent projections are 55,296: content query 16,384, rotary query 8,192, down-projection 4,096, key up-projection 4,096, value up-projection 4,096, rotary key 2,048, output projection 16,384. Four layers drop 40,960 parameters, so the model goes from 803,072 to 762,112.

## Reading the code

| file | what to look at |
| --- | --- |
| `src/model.py` | the latent, the shared rotary key, the cache that stores those two and up-projects on read |
| `src/train.py` | cross-entropy, parameter count, and KV bytes for `mha` and `mla` |
| `src/sample.py` | prefill once, then one decode step per new character |
| `src/smoke.py` | cache gaps, byte and parameter counts, latent reconstruction, decoupled rotary position, 20 training steps |

A token-layer is one token through one block. Compression does not add a pass.

## Setup

This experiment has its own uv project. From this directory:

```bash
uv sync
uv run python -m src.smoke
uv run python -m src.train --model mha --preset laptop
uv run python -m src.train --model mla --preset laptop
uv run python -m src.sample --checkpoint runs/mla-laptop.pt --prompt "ROMEO:"
```

The text file is `../002-nanogpt/data/input.txt`. Checkpoints are gitignored.

## Smoke

`src.smoke` checks both modes, then trains each for 20 steps. It exits non-zero if a logit gap is above `1e-4`, if the cache is not 4,096 bytes for multi-head attention and 768 for latent attention, if up-projecting a cached latent disagrees with a direct projection, or if a content key changes with position.

Run on this Mac, device `mps`. The prompt in the cache check is 8 random characters.

| mode | prefill gap | decode gap | parameters | kv bytes/token |
| --- | --- | --- | --- | --- |
| mha | 0 | 1.8e-7 | 803,072 | 4,096 |
| mla | 0 | 1.4e-7 | 762,112 | 768 |

Prefill is 32 token-layers and a decode step is 4, in both modes. The same latent produced the same content key at two positions. The shared rotary key moved by 0.221. Rebuilding keys from the cache matched the direct projection exactly. The 20-step tails reached validation loss 3.20 for multi-head attention and 3.17 for latent attention. The script printed `smoke ok`.

## Results

Laptop preset, Apple MPS, seed 1337, 2000 steps. Tiny Shakespeare is 1,115,394 characters, vocabulary 65, so a uniform guess scores `ln(65)` = 4.17. The split is 1,003,854 train tokens and 111,540 validation tokens.

| model | step | train loss | val loss | parameters | kv bytes/token |
| --- | --- | --- | --- | --- | --- |
| mha | 0 | 4.211 | 4.212 | 803,072 | 4,096 |
| mha | 1999 | 1.773 | 1.900 | 803,072 | 4,096 |
| mla | 0 | 4.213 | 4.215 | 762,112 | 768 |
| mla | 1999 | 1.841 | 1.977 | 762,112 | 768 |

The multi-head run took 23.3 seconds and lands on the experiment 006 dense GPT (validation loss 1.900 there). Latent attention took 25.8 seconds. Its validation loss is 0.077 higher, and the cache is 768 bytes per token instead of 4,096.

Checkpoints: `runs/mha-laptop.pt`, `runs/mla-laptop.pt`.

Sampling stops at `block_size` (64), because the rotary table has that many rows. From `ROMEO:` (6 characters) that is 58 new characters. Prefill is 24 token-layers. Decode is 232. The latent cache is 768 bytes per token on both phases.

```text
ROMEO:
DefF OF mines that?
Pried do cusuns worm. Old cunson in a
```

The words are invented. The line already breaks like verse.
