# 001 — character GPT

A character-level language model trained on Tiny Shakespeare. It continues text one character at a time.

The architecture follows Andrej Karpathy’s nanoGPT lecture:

- [Neural Networks: Zero to Hero — GPT lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Colab notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
- [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture) (MIT)

The modules here are a reimplementation of that lecture: a bigram baseline, then a small decoder-only Transformer. Upstream code is MIT licensed; this experiment keeps the same model shape and cites that source.

## Layout

| path | role |
| --- | --- |
| `models/tokenizer.py` | Character vocabulary, encode and decode |
| `models/bigram.py` | Embedding lookup that predicts the next character |
| `models/gpt.py` | Decoder-only Transformer |
| `models/train.py` | 90/10 split, AdamW, train and validation loss |
| `models/sample.py` | Continue a prompt from a saved checkpoint |
| `models/walkthrough.ipynb` | The lecture notebook: data, bigram, self-attention derivation, then the full training cell |
| `data/input.txt` | Tiny Shakespeare (~1 MB) |
| `runs/` | Checkpoints (gitignored) |

## Setup

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Train

From the repository root:

```bash
.venv/bin/python experiments/language/001-char-gpt/models/train.py --model bigram
.venv/bin/python experiments/language/001-char-gpt/models/train.py --model gpt
```

`--preset` is `laptop` by default.

| preset | block | batch | layers | heads | embedding | steps | dropout | intended use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| laptop | 64 | 12 | 4 | 4 | 128 | 2000 | 0.0 | This machine (MPS or CPU) |
| lecture | 256 | 64 | 6 | 6 | 384 | 5000 | 0.2 | The lecture’s GPU run |

The bigram model ignores layers, heads, and embedding size. Its learning rate is `1e-2`. The GPT learning rate is `3e-4`. Both use AdamW.

```bash
.venv/bin/python experiments/language/001-char-gpt/models/train.py --model gpt --preset lecture
```

## Sample

```bash
.venv/bin/python experiments/language/001-char-gpt/models/sample.py --checkpoint experiments/language/001-char-gpt/runs/gpt-laptop.pt --prompt "ROMEO:"
```

## Results

Laptop preset, Apple MPS, seed 1337, 2000 steps. Tiny Shakespeare is 1,115,394 characters, vocabulary 65, so a uniform guess scores `ln(65)` = 4.17. The split is 1,003,854 train tokens and 111,540 validation tokens. The GPT has 816,705 parameters.

| model | learning rate | train loss | val loss |
| --- | --- | --- | --- |
| bigram, step 0 | 1e-2 | 4.724 | 4.730 |
| bigram, step 1999 | 1e-2 | 2.469 | 2.491 |
| gpt, step 0 | 3e-4 | 4.218 | 4.214 |
| gpt, step 1999 | 3e-4 | 1.752 | 1.869 |

Train and validation stay close for the bigram. The GPT is still improving at the last step, with validation about 0.12 above training.

Checkpoints: `runs/bigram-laptop.pt`, `runs/gpt-laptop.pt`.

Bigram continuation from `ROMEO:` is character salad with spaces and a few word-shaped clumps. GPT continuation from the same prompt, 400 new characters:

```text
ROMEO:
Lost wads rehtsentry mightand net of ouths;
And pliving: any to nope reain;
By that dove daid in the cut his honour;
Shirer liy?
If as the will to hrow, by go hant and
Ind I will you coods idain
Wark to thing darrung too, korrow be chere
Then will wour made; I thing you her.

SABILSA:
My you.

HENRY RALTHNGNo!
Thyd Mussee, now fut,
All yet roward, his nuke of quing of the
Ill no.

GLOkIquing hold
```

The words are invented, and the lines already have the shape of a speech and a speaker label. The walkthrough notebook is the lecture Colab: it derives attention in the notebook, then trains the Colab's smaller finished model in the last cell (4 layers, 64-d, block 32, 5000 steps).
