# ai-models

Small models trained as separate experiments. Each experiment has its own folder. This file is the index.

This repository continues text, images, audio, and video one experiment at a time. The first language model predicts the next character in Tiny Shakespeare.

Setup and the first training run, from the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiments/language/001-char-gpt/models/train.py --model gpt
```

The script prints the device (CUDA, then Apple MPS, then CPU), the parameter count, the loss every 200 steps, and the wall-clock time. Details, the bigram baseline, and how to read the modules are in the experiment README.

Experiment 002 has its own uv project. From `experiments/language/002-nanogpt`:

```bash
uv sync
uv run python -m src.train --preset laptop
```

Experiment 003 pins [karpathy/nanochat](https://github.com/karpathy/nanochat) and runs a short MPS smoke. From `experiments/language/003-nanochat`:

```bash
bash smoke_mps.sh
```

Experiment 004 has its own uv project. From `experiments/language/004-mingpt`:

```bash
uv sync
uv run python -m src.train --preset smoke
```

Experiment 005 has its own uv project. From `experiments/language/005-ced`:

```bash
uv sync
uv run python -m src.smoke
```

| id | modality | name | status | goal | compute | folder |
| --- | --- | --- | --- | --- | --- | --- |
| 001 | language | char-gpt | done | Character-level language model on Tiny Shakespeare | laptop / MPS, about a minute | [experiments/language/001-char-gpt](experiments/language/001-char-gpt) |
| 002 | language | nanogpt | done | nanoGPT-style character model on Tiny Shakespeare | laptop / MPS, about 30 seconds | [experiments/language/002-nanogpt](experiments/language/002-nanogpt) |
| 003 | language | nanochat | done | Chat pipeline smoke on a tiny nanochat model | laptop / MPS, about 20 seconds once data is cached | [experiments/language/003-nanochat](experiments/language/003-nanochat) |
| 004 | language | mingpt | done | minGPT-style character model on Tiny Shakespeare | laptop / MPS, about 10 seconds | [experiments/language/004-mingpt](experiments/language/004-mingpt) |
| 005 | language | ced | done | Causal encoder-decoder character model on Tiny Shakespeare | laptop / MPS, smoke plus about 25 seconds per mode | [experiments/language/005-ced](experiments/language/005-ced) |
