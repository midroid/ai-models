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

Experiment 006 has its own uv project. From `experiments/language/006-mixtral`:

```bash
uv sync
uv run python -m src.smoke
```

Experiment 007 has its own uv project. From `experiments/language/007-deepseek-moe`:

```bash
uv sync
uv run python -m src.smoke
```

Experiment 008 has its own uv project. From `experiments/language/008-mla`:

```bash
uv sync
uv run python -m src.smoke
```

Experiment 009 has its own uv project. From `experiments/language/009-bias-mtp`:

```bash
uv sync
uv run python -m src.smoke
```

Experiment 010 has its own uv project. From `experiments/language/010-hindi-sanskrit-slm`:

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
| 006 | language | mixtral | done | Mixtral-style top-2 mixture of experts on Tiny Shakespeare | laptop / MPS, smoke plus about 25 seconds dense and 2.5 minutes for the mixture | [experiments/language/006-mixtral](experiments/language/006-mixtral) |
| 007 | language | deepseek-moe | done | DeepSeekMoE shared expert and fine-grained router on Tiny Shakespeare | laptop / MPS, smoke plus about 2.5 minutes Mixtral and 4 minutes DeepSeekMoE | [experiments/language/007-deepseek-moe](experiments/language/007-deepseek-moe) |
| 008 | language | mla | done | Multi-head latent attention versus multi-head attention on Tiny Shakespeare | laptop / MPS, smoke plus about 25 seconds per mode | [experiments/language/008-mla](experiments/language/008-mla) |
| 009 | language | bias-mtp | done | Router bias and a second predicted character on Tiny Shakespeare | laptop / MPS, smoke plus about 3.5 minutes per mode | [experiments/language/009-bias-mtp](experiments/language/009-bias-mtp) |
| 010 | language | hindi-sanskrit | ready | 10M Hindi–Sanskrit decoder, pretrain checkpoints every 100M tokens | laptop / MPS, smoke is one forward of the 10M model | [experiments/language/010-hindi-sanskrit-slm](experiments/language/010-hindi-sanskrit-slm) |
