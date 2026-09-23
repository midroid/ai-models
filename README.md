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

| id | modality | name | status | goal | compute | folder |
| --- | --- | --- | --- | --- | --- | --- |
| 001 | language | char-gpt | done | Character-level language model on Tiny Shakespeare | laptop / MPS, about a minute | [experiments/language/001-char-gpt](experiments/language/001-char-gpt) |
| 002 | language | nanogpt | done | nanoGPT-style character model on Tiny Shakespeare | laptop / MPS, about 30 seconds | [experiments/language/002-nanogpt](experiments/language/002-nanogpt) |
