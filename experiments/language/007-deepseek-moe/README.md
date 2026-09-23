# 007 — DeepSeekMoE

A character-level language model on Tiny Shakespeare. One module trains two feed-forward mixes on the experiment 006 attention stack. `mixtral` is that experiment's block: eight experts, top-2. `dsmoe` segments each expert in half and isolates one of them as a shared expert, following [DeepSeekMoE](https://arxiv.org/abs/2401.06066) sections 2.1 and 2.2.

The shared expert is an ordinary GELU feed-forward network. It runs on every token and is added with coefficient 1. It is not a row of the router. The other fifteen experts use the Mixtral gate: the top three logits are kept, the rest are treated as negative infinity, and softmax over those three multiplies their outputs. Four half-width experts match the two full experts Mixtral runs, so the active feed-forward cost is the same. The router is the extra cost: `128 × 15` instead of `128 × 8`, which is 3,584 parameters across four layers.

Device-limited routing is left out. It decides which machine an expert lives on, and this run is one process.

The load-balancing term is the Switch auxiliary loss on the routed experts only. For one layer, `f` is the fraction of (token, slot) assignments on each routed expert, `P` is the mean router probability before the top-3 mask, and the loss is `15 * sum(f * P)`. That value is 1 when `P` is uniform over the routed experts. `f` is detached. The optimizer steps on cross-entropy plus `0.01` times the mean of that loss across layers. The printed loss is still cross-entropy.

One half-width expert is `2 * 128 * 256` = 65,536 parameters. One Mixtral expert is `2 * 128 * 512` = 131,072. Both modes hold 1,048,576 expert parameters per layer and activate 262,144 of them.

## The split

Mixtral picks 2 experts out of 8. That is 28 possible pairs. DeepSeekMoE first replaces each of those 8 with two experts of half the width, which is 16 experts, and would activate 4 of them to keep the same multiply count. One of the 4 active slots is then reserved for a shared expert that is removed from the menu. What remains is 15 routed experts, of which 3 are chosen: 455 combinations, and the shared expert is in every one of them.

```text
y = shared(x) + sum of the three routed experts, each scaled by its gate
```

`shared(x)` has no gate. The three weights come from softmax over the top three router logits and sum to 1. Active parameters subtract only the 12 routed experts a token does not run. Subtracting the shared expert would make this block look cheaper than Mixtral when the feed-forward work is the same.

## Reading the code

| file | what to look at |
| --- | --- |
| `src/model.py` | the shared expert, the top-3 routed gate, active parameters with shared experts kept |
| `src/train.py` | cross-entropy beside the auxiliary loss, shared fraction and routed fractions |
| `src/sample.py` | prefill once, then one decode step per new character |
| `src/smoke.py` | cache gaps, matched feed-forward cost, dispatch, gradients, auxiliary loss, 20 training steps |

A token-layer is one token through one block. The shared expert and the three routed experts still count as one block. KV bytes per token match, because the cache stores keys and values, not expert outputs.

## Setup

This experiment has its own uv project. From this directory:

```bash
uv sync
uv run python -m src.smoke
uv run python -m src.train --model mixtral --preset laptop
uv run python -m src.train --model dsmoe --preset laptop
uv run python -m src.sample --checkpoint runs/dsmoe-laptop.pt --prompt "ROMEO:"
```

The text file is `../002-nanogpt/data/input.txt`. Checkpoints are gitignored.

## Smoke

`src.smoke` checks both modes, then trains each for 20 steps. It exits non-zero if a logit gap is above `1e-4`, if the active feed-forward cost drifts from Mixtral, if a token does not have exactly three positive routed weights that sum to 1, if the shared expert appears in the router, if sparse dispatch disagrees with running every routed expert, or if a flat router does not score auxiliary loss 1.

Run on this Mac, device `mps`. The prompt in the cache check is 8 random characters. Both modes have 4,096 KV bytes per token and 262,144 active feed-forward parameters per layer.

| mode | prefill gap | decode gap | parameters | active |
| --- | --- | --- | --- | --- |
| mixtral | 0 | 1.8e-7 | 4,477,184 | 1,331,456 |
| dsmoe | 0 | 1.6e-7 | 4,480,768 | 1,335,040 |

The two active counts differ by 3,584, which is the router. Prefill is 32 token-layers and a decode step is 4, in both modes. Sparse dispatch matched the all-expert mix within `2.1e-7`. Zeroing the shared expert left only the routed mix. A flat router scored auxiliary loss 1.000, and a peaked router scored 5.000. The 20-step tails reached validation loss 3.13 for Mixtral and 3.14 for DeepSeekMoE. The script printed `smoke ok`.

## Results

Laptop preset, Apple MPS, seed 1337, 2000 steps. Tiny Shakespeare is 1,115,394 characters, vocabulary 65, so a uniform guess scores `ln(65)` = 4.17. The split is 1,003,854 train tokens and 111,540 validation tokens.

| model | step | train loss | val loss | val aux | parameters | active |
| --- | --- | --- | --- | --- | --- | --- |
| mixtral | 0 | 4.208 | 4.209 | 1.030 | 4,477,184 | 1,331,456 |
| mixtral | 1999 | 1.633 | 1.814 | 1.041 | 4,477,184 | 1,331,456 |
| dsmoe | 0 | 4.196 | 4.196 | 1.042 | 4,480,768 | 1,335,040 |
| dsmoe | 1999 | 1.638 | 1.804 | 1.033 | 4,480,768 | 1,335,040 |

The Mixtral run took 143.9 seconds and lands on the experiment 006 mixture (validation loss 1.818 there). DeepSeekMoE took 223.0 seconds. Its validation loss is in the same range at the same active feed-forward cost. The auxiliary loss stays near 1. The shared expert's token fraction is 1 on every layer.

Routed fractions at step 1999, on one validation batch. Each row sums to 1. A uniform router would put about 0.067 on every expert. The busiest expert on this batch is 0.114 and the quietest is 0.034, so the router did not collapse.

| layer | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | e10 | e11 | e12 | e13 | e14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.076 | 0.061 | 0.090 | 0.066 | 0.034 | 0.061 | 0.071 | 0.064 | 0.088 | 0.035 | 0.048 | 0.064 | 0.059 | 0.095 | 0.089 |
| 1 | 0.047 | 0.079 | 0.075 | 0.114 | 0.043 | 0.065 | 0.061 | 0.069 | 0.049 | 0.082 | 0.060 | 0.061 | 0.067 | 0.059 | 0.068 |
| 2 | 0.082 | 0.050 | 0.071 | 0.054 | 0.049 | 0.065 | 0.079 | 0.114 | 0.080 | 0.058 | 0.058 | 0.076 | 0.058 | 0.037 | 0.070 |
| 3 | 0.061 | 0.086 | 0.064 | 0.061 | 0.082 | 0.060 | 0.045 | 0.063 | 0.062 | 0.059 | 0.064 | 0.112 | 0.071 | 0.044 | 0.066 |

Checkpoints: `runs/mixtral-laptop.pt`, `runs/dsmoe-laptop.pt`.

Sampling stops at `block_size` (64), because the rotary table has that many rows. From `ROMEO:` (6 characters) that is 58 new characters. Prefill is 24 token-layers. Decode is 232.

```text
ROMEO:
I'll contart even them!
And the beseech unants bone, thee
```

The words are invented. The lines already break like verse.
