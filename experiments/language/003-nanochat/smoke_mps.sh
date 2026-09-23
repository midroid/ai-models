#!/usr/bin/env bash
# Short slice of upstream runs/runcpu.sh, forced onto Apple MPS.
# The full Mac demo and runs/speedrun.sh stay documented in the README.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/upstream"

# Eager MPS does not promote the 0-D CPU scalars inside adamw_step_fused / muon_step_fused.
if ! grep -q 'def _scalar_on' nanochat/optim.py; then
  patch -p1 < "$ROOT/mps_optim.patch"
fi

export NANOCHAT_BASE_DIR="$ROOT/cache"
export WANDB_RUN=dummy
export PYTHONUNBUFFERED=1
# base_train and chat_sft call torch.compile. Dynamo is unreliable on MPS,
# so this smoke stays in eager mode without editing upstream source.
export TORCHDYNAMO_DISABLE=1

mkdir -p "$NANOCHAT_BASE_DIR"
LOG="$NANOCHAT_BASE_DIR/smoke.log"

{
  echo "smoke start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  SECONDS=0

  uv sync --extra cpu

  # A couple of ClimbMix train shards, plus the pinned validation shard.
  uv run python -m nanochat.dataset -n 2

  # Tiny BPE: 200k characters, vocab 512 (9 special tokens sit on top of >=256 bytes).
  uv run python -m scripts.tok_train --max-chars=200000 --doc-cap=2000 --vocab-size=512
  uv run python -m scripts.tok_eval

  # depth=4 is the size dial. One sequence per step, 20 steps, no CORE metric.
  # seq 512 is long enough that some SmolTalk rows fit during SFT. At 128, every
  # conversation was longer than the row, the loss mask was empty, and the loss was nan.
  uv run python -m scripts.base_train \
    --depth=4 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=512 \
    --eval-every=10 \
    --eval-tokens=512 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --num-iterations=20 \
    --warmup-steps=1 \
    --device-type=mps \
    --run="$WANDB_RUN"

  # A handful of SFT steps on SmolTalk only, same sequence length.
  # The final step still runs validation, so eval-tokens stays at one batch.
  uv run python -m scripts.chat_sft \
    --device-type=mps \
    --num-iterations=5 \
    --max-seq-len=512 \
    --device-batch-size=1 \
    --total-batch-size=512 \
    --eval-every=-1 \
    --eval-tokens=512 \
    --chatcore-every=-1 \
    --mmlu-epochs=0 \
    --gsm8k-epochs=0 \
    --load-optimizer=0 \
    --run="$WANDB_RUN"

  uv run python -m scripts.chat_cli \
    --device-type=mps \
    -p "What is the capital of France?"

  echo "smoke elapsed_s: $SECONDS"
  echo "smoke end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$LOG"
