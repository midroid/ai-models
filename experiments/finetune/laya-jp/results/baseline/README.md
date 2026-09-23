# Original laya-multilingual baseline

Checkpoint `convaiinnovations/laya-multilingual` at `82d57fc4f2d1be3d2caac494045f2ec51d0842f3`.
Device `mps`, dtype `torch.float32`, laya `0.3.11`.

The published bench_ja column is CUDA fp32 on laya 0.3.4. The published 36.9 composite is MLX FP16, a different checkpoint. This file is the local PyTorch run.

## bench_ja

| metric | this run | published CUDA |
| --- | ---: | ---: |
| choice accuracy | 0.747 | 0.747 |
| score accuracy | 0.447 | 0.443 |
| score slot 0 count | 0 | 0 |
| noul accuracy | 0.547 | 0.543 |
| noul AUROC | 0.523 | 0.523 |
| noul balanced accuracy | 0.528 | |
| score MAE argmax | 0.617 | |
| score MAE expectation | 0.687 | |

Accuracy gaps larger than 0.02: none.

## bench_en

Choice accuracy 0.817. Score accuracy 0.266, slot 0 count 0. Noul accuracy 0.641, AUROC 0.355.

## 40-question business bench

Composite 36.9 / 100. Bootstrap 95% interval over cases: 21.8 to 52.4. Repeats identical: True. Published MLX FP16 composite: 36.9.

Per-type pass counts are in `snsk.summary.json`.

## Diagnostics

Consistency is the fraction of items whose semantic answer matches the batched baseline. Score slot 0 is how often the first presented level wins under that probe.

On a 3-level score list, shuffling with seed 2026 is exactly the reversal, so `score_permuted` and `score_reversed` are the same probe on bench_ja and bench_en. Longer rubrics in the 40-question bench get a different order.

### bench_ja

| probe | n | consistency | slot 0 | mean abs delta P(true) |
| --- | ---: | ---: | ---: | ---: |
| single_question | 900 | 1.000 |  |  |
| score_reversed | 300 | 0.097 | 0.000 |  |
| score_permuted | 300 | 0.097 | 0.000 |  |
| noul_labels_ab | 300 | 0.843 |  | 0.196 |
| noul_choice_ab | 300 | 0.533 |  |  |

### bench_en

| probe | n | consistency | slot 0 | mean abs delta P(true) |
| --- | ---: | ---: | ---: | ---: |
| single_question | 870 | 1.000 |  |  |
| score_reversed | 290 | 0.003 | 0.000 |  |
| score_permuted | 290 | 0.003 | 0.000 |  |
| noul_labels_ab | 290 | 0.910 |  | 0.145 |
| noul_choice_ab | 290 | 0.797 |  |  |

### snsk_jp_business

| probe | n | consistency | slot 0 | mean abs delta P(true) |
| --- | ---: | ---: | ---: | ---: |
| single_question | 48 | 1.000 |  |  |
| score_reversed | 16 | 0.250 | 0.000 |  |
| score_permuted | 16 | 0.125 | 0.062 |  |
| noul_labels_ab | 16 | 0.688 |  | 0.426 |
| noul_choice_ab | 16 | 0.438 |  |  |
