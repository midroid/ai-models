"""Pins for the original-checkpoint baseline. Do not point these at a fine-tune."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "data" / "external"
MANIFEST = ROOT / "data" / "manifest.json"
RESULTS = ROOT / "results" / "baseline"
CACHE = ROOT / ".cache" / "laya-multilingual"

MODEL_ID = "convaiinnovations/laya-multilingual"
# Hub commit resolved 2026-09-24. laya.load has no revision argument, so the
# snapshot is downloaded at this commit and loaded from disk.
MODEL_REVISION = "82d57fc4f2d1be3d2caac494045f2ec51d0842f3"
DEVICE = "mps"

PROB_FLOOR = 5e-5
ECE_BINS = 15
DRIFT_TOLERANCE = 0.02
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 2026
SNSK_SHUFFLE_SEED = 20260921
SCORE_PERM_SEED = 2026

# sokudan docs/baseline_ja.md, CUDA fp32, laya 0.3.4, 2026-09-20.
PUBLISHED_BENCH_JA = {
    "choice_accuracy": 0.747,
    "score_accuracy": 0.443,
    "score_slot0_count": 0,
    "noul_accuracy": 0.543,
    "noul_auroc": 0.523,
    "n": 300,
    "runtime": "CUDA fp32, laya 0.3.4",
}

# snsk report, 2026-09-21. Different artifact from MODEL_ID.
PUBLISHED_SNSK_MLX = {
    "composite": 36.9,
    "checkpoint": "aac6fef/laya-multilingual-mlx",
    "dtype": "float16",
}

SOKUDAN_COMMIT = "e1bcfbb5884c43762d6f6ab6df5b78cc75746fb9"
SNSK_COMMIT = "743e233f69e898f4d576f31b0ba4668df7f26765"
