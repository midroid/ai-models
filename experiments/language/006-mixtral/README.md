# 006 — Mixtral-style mixture of experts

A character-level language model on Tiny Shakespeare. One module trains two modes: the dense GPT from experiment 005, and the same stack with each feed-forward network replaced by eight experts. A token runs two of them.

The gate is the one in [Mixtral of Experts](https://arxiv.org/abs/2401.04088), section 2. The top two router logits are kept, the rest are treated as negative infinity, and softmax over those two multiplies the expert outputs. There is no expert capacity and no dropped token. The experts are the dense GELU network, four times the embedding width, so the comparison with the GPT is the router rather than a new block. Grouped-query attention and SwiGLU stay out.

The load-balancing term is the Switch auxiliary loss. For one layer, `f` is the fraction of (token, slot) assignments on each expert, `P` is the mean router probability before the top-2 mask, and the loss is `8 * sum(f * P)`. That value is 1 when `P` is uniform. `f` is detached. The optimizer steps on cross-entropy plus `0.01` times the mean of that loss across layers. The printed loss is still cross-entropy.

Active parameters are the embedding, the attention, the router, and two of the eight experts. Total parameters count all eight. One expert is `2 * 128 * 512` = 131,072 parameters. The router is `128 * 8` per layer, 4,096 across four layers. Active feed-forward compute is twice the dense GPT, because two full experts run.

## Reading the code

| file | what to look at |
| --- | --- |
| `src/model.py` | the top-2 gate, sparse dispatch, the auxiliary loss, active parameter count |
| `src/train.py` | cross-entropy printed beside the auxiliary loss, expert fractions on one validation batch |
| `src/sample.py` | prefill once, then one decode step per new character |
| `src/smoke.py` | cache gaps, gate sparsity, dispatch, gradients, auxiliary loss, 20 training steps |

A token-layer is one token through one block. Two active experts still count as one block. KV bytes per token match the dense GPT, because the cache stores keys and values, not expert outputs.

## Setup

This experiment has its own uv project. From this directory:

```bash
uv sync
uv run python -m src.smoke
uv run python -m src.train --model gpt --preset laptop
uv run python -m src.train --model moe --preset laptop
uv run python -m src.sample --checkpoint runs/moe-laptop.pt --prompt "ROMEO:"
```

The text file is `../002-nanogpt/data/input.txt`. Checkpoints are gitignored.

## Smoke

`src.smoke` checks both modes, then trains each for 20 steps. It exits non-zero if a logit gap is above `1e-4`, if a token does not have exactly two positive gate weights that sum to 1, if sparse dispatch disagrees with running every expert, if an unused expert gets a gradient, or if a flat router does not score auxiliary loss 1.

Run on this Mac, device `mps`. The prompt in the cache check is 8 random characters. Both modes have 4,096 KV bytes per token.

| mode | prefill gap | decode gap | parameters | active |
| --- | --- | --- | --- | --- |
| gpt | 0 | 1.5e-7 | 803,072 | 803,072 |
| moe | 0 | 1.6e-7 | 4,477,184 | 1,331,456 |

Prefill is 32 token-layers and a decode step is 4, in both modes. Sparse dispatch matched the all-expert mix within `3.6e-7`. A flat router scored auxiliary loss 1.000, and a router peaked on one expert scored 4.000. The 20-step tails reached validation loss 3.20 for the GPT and 3.13 for the mixture. The script printed `smoke ok`.

## Results

Laptop preset, Apple MPS, seed 1337, 2000 steps. Tiny Shakespeare is 1,115,394 characters, vocabulary 65, so a uniform guess scores `ln(65)` = 4.17. The split is 1,003,854 train tokens and 111,540 validation tokens.

| model | step | train loss | val loss | val aux | parameters | active |
| --- | --- | --- | --- | --- | --- | --- |
| gpt | 0 | 4.211 | 4.212 | 0 | 803,072 | 803,072 |
| gpt | 1999 | 1.773 | 1.900 | 0 | 803,072 | 803,072 |
| moe | 0 | 4.208 | 4.209 | 1.030 | 4,477,184 | 1,331,456 |
| moe | 1999 | 1.632 | 1.818 | 1.041 | 4,477,184 | 1,331,456 |

The GPT run took 24.2 seconds and lands on the experiment 005 GPT (validation loss 1.900 there). The mixture took 145.4 seconds. Its validation loss is lower because each token runs two full feed-forward networks, so the active model is larger than the dense GPT. The auxiliary loss stays near 1, which is the balanced value of that term.

Expert fractions at step 1999, on one validation batch. Each row sums to 1. A uniform router would put 0.125 on every expert. The busiest expert on this batch is 0.200 and the quietest is 0.051, so the router did not collapse onto one expert.

| layer | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.137 | 0.113 | 0.103 | 0.111 | 0.148 | 0.192 | 0.051 | 0.145 |
| 1 | 0.105 | 0.100 | 0.115 | 0.100 | 0.122 | 0.200 | 0.084 | 0.173 |
| 2 | 0.116 | 0.166 | 0.094 | 0.143 | 0.160 | 0.089 | 0.122 | 0.110 |
| 3 | 0.157 | 0.095 | 0.089 | 0.135 | 0.069 | 0.154 | 0.105 | 0.197 |

Checkpoints: `runs/gpt-laptop.pt`, `runs/moe-laptop.pt`.

Sampling stops at `block_size` (64), because the rotary table has that many rows. From `ROMEO:` (6 characters) that is 58 new characters. Prefill is 24 token-layers. Decode is 232.

GPT:

```text
ROMEO:
Nuris your they buserang unsetoone, and Yare.
Thour heard
```

MoE:

```text
ROMEO:
O muth envesty heart, I point York you chould:
Ayneeednin
```

The words are invented. The lines already break like verse.
