# 009 — Router bias and multi-token prediction

A character-level language model on Tiny Shakespeare. The block is the DeepSeekMoE from experiment 007: one shared expert, fifteen routed experts of hidden 256, top-3, on that experiment's multi-head attention. Three modes change the training, not the experts. The two new modes are a router bias that replaces the auxiliary loss, and that bias plus a second predicted character. The folder is named for those two, `bias-mtp`.

`dsmoe` is experiment 007. The optimizer steps on next-character cross-entropy plus `0.01` times the Switch auxiliary loss. `bias` drops that term. A buffer on each routed expert is added only when choosing the top three, following Wang et al., [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664), as used in [DeepSeek-V3](https://arxiv.org/abs/2412.19437) section 2.1.2. `mtp` keeps the bias and also predicts the character after the next one, the sequential form of Gloeckle et al., [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737), from V3 section 2.2.

DeepSeek-V3 also trains in FP8, pipelines the forward and backward passes, and uses a 128k context. Those are left out, along with the tiny sequence-wise auxiliary term that paper keeps beside the bias. Latent attention stays in experiment 008, so a `dsmoe` run can still be compared with experiment 007.

## The bias

The router still emits one logit per routed expert. The shared expert is still not a row. Selection is

```text
index = topk(logits + bias, k=3)
weights = softmax(logits[index])
```

`weights` sum to 1. The bias is a buffer of 15 zeros, not a parameter, so AdamW does not train it. After each training step, and not during evaluation:

```text
bias <- bias + 0.001 * sign(1/15 - f)
```

`f` is the fraction of (token, slot) assignments on that expert. It sums to 1. An expert above `1/15` loses `0.001`. An expert below it gains `0.001`. Putting the bias inside the softmax would let it scale the expert outputs and would give it a gradient. A zero bias reproduces top-3 on the raw logits, so `dsmoe` and `bias` match at step 0.

## The second token

`get_batch` for `mtp` reads one character further. `x` is the input, `y` is the next character, and `z` is the character after `y`. The main head still predicts `y`. The extra head concatenates the main hidden state with the embedding of `y`, projects `256 -> 128`, and runs one dense block: multi-head attention plus a GELU MLP of hidden 512. The block is dense so the bias update still counts assignments from the four main layers only. A separate head scores `z`, which leaves the main head the same width as experiment 007. The optimizer adds `0.1` times that cross-entropy. The printed loss stays the next-character loss, so the three modes can be compared on the same number.

The extra projection is 32,768 parameters, the dense block is 196,608, and the extra head is 8,320. Training parameters are 4,718,464. Prefill, decode, and sampling call the main stack only, so inference stays 4,480,768 parameters, 1,335,040 of them active, and 4,096 KV bytes per token. A token-layer is still one token through one main block.

## Reading the code

| file | what to look at |
| --- | --- |
| `src/model.py` | the bias added only to top-k, the update from assignment fractions, the second-token block |
| `src/train.py` | cross-entropy, the Switch term only in `dsmoe`, the bias step after the optimizer, the second-token loss |
| `src/sample.py` | prefill once, then one decode step per new character, main stack only |
| `src/smoke.py` | cache gaps, parameter counts, raw-logit gate, one bias update, 20 training steps |

## Setup

This experiment has its own uv project. From this directory:

```bash
uv sync
uv run python -m src.smoke
uv run python -m src.train --model dsmoe --preset laptop
uv run python -m src.train --model bias --preset laptop
uv run python -m src.train --model mtp --preset laptop
uv run python -m src.sample --checkpoint runs/mtp-laptop.pt --prompt "ROMEO:"
```

The text file is `../002-nanogpt/data/input.txt`. Checkpoints are gitignored.

## Smoke

`src.smoke` checks all three modes, then trains each for 20 steps. It exits non-zero if a logit gap is above `1e-4`, if the inference parameter count leaves 4,480,768, if a zero bias does not match top-3 on the raw logits, if a forced assignment does not move the bias by `0.001`, or if `bias` or `mtp` adds the Switch term.

Run on this Mac, device `mps`. The prompt in the cache check is 8 random characters. Every mode has 4,096 KV bytes per token.

| mode | prefill gap | decode gap | training parameters | inference | active |
| --- | --- | --- | --- | --- | --- |
| dsmoe | 0 | 1.3e-7 | 4,480,768 | 4,480,768 | 1,335,040 |
| bias | 0 | 1.8e-7 | 4,480,768 | 4,480,768 | 1,335,040 |
| mtp | 0 | 2.0e-7 | 4,718,464 | 4,480,768 | 1,335,040 |

Prefill is 32 token-layers and a decode step is 4, in every mode. A zero bias matched raw top-3. One assignment of every token to expert 0 set that bias to `-0.001` and the other fourteen to `+0.001`. The second-token head received a gradient, and decode stayed on the main stack. The 20-step tails reached validation loss 3.14 for `dsmoe` and `bias`, and 3.13 for `mtp`. Step 0 of `dsmoe` and `bias` matched. The script printed `smoke ok`.

## Results

Laptop preset, Apple MPS, seed 1337, 2000 steps. Tiny Shakespeare is 1,115,394 characters, vocabulary 65, so a uniform guess scores `ln(65)` = 4.17. The split is 1,003,854 train tokens and 111,540 validation tokens. The printed loss is the next-character cross-entropy. `val mtp` is the second-token loss. `bias` is the min and max of the router buffer at that step.

| model | step | train loss | val loss | val aux | val mtp | bias | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dsmoe | 0 | 4.196 | 4.196 | 1.042 |  |  | 4,480,768 |
| dsmoe | 1999 | 1.633 | 1.794 | 1.035 |  |  | 4,480,768 |
| bias | 0 | 4.196 | 4.196 | 1.042 |  | 0 | 4,480,768 |
| bias | 1999 | 1.629 | 1.805 | 1.006 |  | -0.675..0.745 | 4,480,768 |
| mtp | 0 | 4.195 | 4.195 | 1.042 | 4.208 | 0 | 4,718,464 |
| mtp | 1999 | 1.622 | 1.794 | 1.001 | 1.843 | -0.779..0.873 | 4,718,464 |

`dsmoe` took 209.2 seconds. Its step-0 loss matches experiment 007, and the run lands on that validation loss (1.804 there). `bias` took 212.3 seconds. The Switch value falls from 1.042 to 1.006 without being added to the optimizer, and the validation loss stays with `dsmoe`. `mtp` took 213.6 seconds. The next-character loss is in the same range. The second-token loss ends at 1.843.

Routed fractions at step 1999, on one validation batch. Each row sums to 1. A uniform router puts about 0.067 on every expert. Under the auxiliary loss the busiest expert is 0.110 and the quietest is 0.028. Under the bias the busiest is 0.087 and the quietest is 0.046.

| layer | e0 | e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 | e9 | e10 | e11 | e12 | e13 | e14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.054 | 0.071 | 0.069 | 0.071 | 0.066 | 0.059 | 0.056 | 0.066 | 0.065 | 0.079 | 0.046 | 0.079 | 0.077 | 0.069 | 0.073 |
| 1 | 0.078 | 0.061 | 0.064 | 0.061 | 0.087 | 0.060 | 0.068 | 0.063 | 0.074 | 0.048 | 0.068 | 0.056 | 0.057 | 0.071 | 0.085 |
| 2 | 0.059 | 0.068 | 0.068 | 0.083 | 0.079 | 0.069 | 0.065 | 0.061 | 0.060 | 0.060 | 0.071 | 0.062 | 0.074 | 0.054 | 0.066 |
| 3 | 0.073 | 0.067 | 0.064 | 0.075 | 0.056 | 0.071 | 0.069 | 0.066 | 0.068 | 0.053 | 0.079 | 0.072 | 0.058 | 0.064 | 0.065 |

The shared expert's token fraction is 1 on every layer, in every mode.

Checkpoints: `runs/dsmoe-laptop.pt`, `runs/bias-laptop.pt`, `runs/mtp-laptop.pt`.

Sampling stops at `block_size` (64), because the rotary table has that many rows. From `ROMEO:` (6 characters) that is 58 new characters. Prefill is 24 token-layers. Decode is 232. The second-token head is not used. The cache is 4,096 bytes per token.

```text
ROMEO:
Trumban, butkmont anameft the arn while!
A made any filel
```

The words are invented. The line already breaks like verse.
