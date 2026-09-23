# Laya Japanese baseline

Locked evaluation of `convaiinnovations/laya-multilingual` before any Japanese business fine-tune. This directory is not part of the numbered from-scratch language experiments.

The vendored benches under `data/external/` are immutable. `data/manifest.json` stores their sha256 hashes, and the eval commands refuse to run if a file changes. Nothing here is training data.

## Run

From this directory:

```bash
uv sync
USE_TF=0 uv run pytest
USE_TF=0 uv run python -m laya_jp.eval_sokudan
USE_TF=0 uv run python -m laya_jp.eval_snsk
USE_TF=0 uv run python -m laya_jp.diagnostics
USE_TF=0 uv run python -m laya_jp.report
```

`eval_sokudan` and `eval_snsk` download the pinned Hub snapshot into `.cache/` (gitignored) and write `results/baseline/`. The report reads those files and writes `baseline.json`.

## Sources

- Sokudan `bench_ja` and `bench_en`, CC BY 4.0, commit `e1bcfbb5884c43762d6f6ab6df5b78cc75746fb9`. Evaluation only. https://github.com/hiroki-abe-58/sokudan
- Snsk 40-question bench, commit `743e233f69e898f4d576f31b0ba4668df7f26765`. Scoring uses their `bench.py`. https://github.com/snsk/jev-laya-japanese-business-benchmark

The published bench_ja table is CUDA fp32 on laya 0.3.4. The published composite 36.9 is an MLX FP16 conversion. This project loads `convaiinnovations/laya-multilingual` at Hub revision `82d57fc4f2d1be3d2caac494045f2ec51d0842f3` with PyTorch.
