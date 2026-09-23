# 003 nanochat

A pinned checkout of [karpathy/nanochat](https://github.com/karpathy/nanochat), plus a short Apple MPS smoke of the chat pipeline. This folder does not reimplement the model. `upstream/` is the source. The notes, the notebook, and `smoke_mps.sh` are ours.

`cache/` is `NANOCHAT_BASE_DIR` (shards, tokenizer, checkpoints, the smoke log). It is gitignored.

## What the pipeline is

nanochat trains a chat model in stages. `--depth` is the single size dial: width is `depth * aspect_ratio`, rounded up to a multiple of `--head-dim`.

| stage | module | what it does |
| --- | --- | --- |
| data | `nanochat.dataset` | downloads ClimbMix parquet shards |
| tokenizer | `scripts.tok_train`, `scripts.tok_eval` | trains a rustbpe BPE tokenizer, then checks compression |
| pretrain | `scripts.base_train` | next-token training on the shards |
| SFT | `scripts.chat_sft` | conversation fine-tune (SmolTalk, plus MMLU and GSM8K in the full script) |
| chat | `scripts.chat_cli` | one prompt, one reply |

Device selection is already in upstream: CUDA, else MPS, else CPU (`nanochat.common.autodetect_device_type`). The smoke passes `--device-type=mps` so a CPU-only fallback cannot hide.

## Setup

From `upstream/`, the project uses uv. The CPU extra is what `runs/runcpu.sh` uses. On this Mac that wheel still reports MPS.

```bash
cd experiments/language/003-nanochat/upstream
uv sync --extra cpu
```

`smoke_mps.sh` does that sync, sets `NANOCHAT_BASE_DIR` to `../cache`, and sets `WANDB_RUN=dummy`.

## Smoke

`smoke_mps.sh` is `runs/runcpu.sh` with the knobs turned down, run from `upstream/` so `python -m scripts...` resolves.

| knob | full Mac demo (`runs/runcpu.sh`) | this smoke |
| --- | --- | --- |
| shards | 8 train shards | 2 |
| tokenizer | 2B characters, vocab 32768 | 200k characters, vocab 512 |
| pretrain | depth 6, seq 512, batch 32, 5000 steps | depth 4, seq 512, batch 1, 20 steps |
| window | `L` | `L` (SDPA has no sliding window) |
| CORE metric | off (`-1`) | off |
| SFT | 1500 steps, full mixture | 5 steps, SmolTalk only |
| chat | commented out in `runcpu.sh` | one prompt |

`TORCHDYNAMO_DISABLE=1` keeps `torch.compile` in `base_train` and `chat_sft` from running Dynamo on MPS. Upstream source is unchanged.

```bash
bash experiments/language/003-nanochat/smoke_mps.sh
```

The log is `cache/smoke.log`. A checkpoint lands under `cache/base_checkpoints/` and `cache/sft_checkpoints/` (or the upstream names those scripts use). The reply can be nonsense. The smoke is finished when it prints a loss, writes a checkpoint, and prints a `chat_cli` reply.

## Runs we do not execute

`bash runs/runcpu.sh` is the unmodified Mac demo: tokenizer on about 2B characters, then about half an hour of pretraining on an M3 Max, then SFT. `runs/speedrun.sh` is the 8×H100 GPT-2 CORE speedrun. Both stay in upstream. This experiment only runs the smoke.

## MPS workaround

`nanochat/optim.py` keeps the AdamW and Muon step scalars as 0-D CPU tensors so `torch.compile` does not recompile when they change. Eager MPS does not promote those scalars inside `lerp_`, and the first optimizer step raised `Tensor for argument #3 'weight' is on CPU, but expected it to be on GPU`. `mps_optim.patch` moves those scalars onto the parameter device when the device is MPS. `smoke_mps.sh` applies the patch if it is not already in the checkout. Upstream is otherwise the pinned commit. `TORCHDYNAMO_DISABLE=1` keeps the model and the fused steps in eager mode.

A sequence length of 128 also failed SFT: with vocab 512, every SmolTalk conversation was longer than the row, the loss mask was empty, and the loss was `nan`. Seq 512 is long enough that some conversations fit. The final SFT step still runs validation even when `--eval-every=-1`, so `--eval-tokens=512` keeps that to one batch.

## Recorded smoke

Run on this Mac with `--device-type=mps` (upstream `compute_init` asserts MPS is available). Compute dtype is float32. The model is depth 4, width 256, 4 heads, 3,670,146 parameters. Wall clock for the successful script was 17 seconds, with the ClimbMix shards, the tokenizer, and SmolTalk already in `cache/` from the earlier attempts.

| stage | result |
| --- | --- |
| pretrain step 0 | loss 6.238682, validation bpb 4.201388 |
| pretrain step 19 | loss 6.286864 |
| pretrain end | validation bpb 4.242748, checkpoint `cache/base_checkpoints/d4/model_000020.pt` |
| SFT last logged step | loss 5.146000, validation bpb 4.0501 |
| SFT checkpoint | `cache/chatsft_checkpoints/d4/model_000004.pt` |
| chat prompt | What is the capital of France? |
| chat reply | `oxc._j for':ingur  these otherrecur`n re:ifindcomedyn' m m forur_indt opth`d` |

The reply is nonsense. Twenty pretrain steps and a handful of SFT steps are there to prove the pipeline reaches a loss, a checkpoint, and a generation.
