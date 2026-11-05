# GPT-2 Language Modeling on Amazon Reviews

[Portfolio Page](https://app.notion.com/p/3e968564df5a815bbd9ff17299b214cd)

A language modeling project fine-tuning **GPT-2** on Amazon review text and comparing training configurations using **Perplexity (PPL)** and training time.


## Project Overview

The task is causal language modeling: predicting the next token in review text. Three experiments vary learning rate, epoch count, and maximum sequence length to examine model loss and computational cost.

**Technologies:** Python, PyTorch, Hugging Face Transformers, GPT-2, Matplotlib.

## Approach

- Reads UTF-8 review text with one non-empty review per line.
- Uses the GPT-2 tokenizer with the EOS token as the padding token.
- Fine-tunes `GPT2LMHeadModel` with next-token labels, masking padding positions with `-100`.
- Uses AdamW, validation-loss checkpoint selection, and early stopping.
- Saves model weights and a training-loss/validation-PPL plot.
- Separates dataset preparation, training, evaluation, and experiment settings.

### Data Split

For datasets with at least 500 reviews, the training function assigns **400 reviews to training**, **50 to validation**, and the remainder to a third partition. Smaller datasets use approximately 80% training and 10% validation, with the remainder reserved. Split randomness is controlled by the configuration seed.

The updated training function returns the reserved **test partition**. After restoring the best validation checkpoint, `main.py` evaluates only that held-out partition with token-weighted cross-entropy and reports test PPL. At least 10 non-empty reviews are required.

## Results

| Setting | Learning Rate | Maximum Epochs | Max Length | Training Time | Reported full-data PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| #1 | 3e-5 | 10 | 256 | 72.35 min | 3.1366 |
| #2 | 2e-5 | 15 | 512 | 133.61 min | **1.8683** |
| #3 | 5e-5 | 8 | 128 | **34.64 min** | 6.7798 |

Batch size is **4** across configurations. The assignment settings specify **100 warmup steps**, although the current training loop does not implement a warmup scheduler.

### Interpretation

**Evaluation caveat:** the table contains historical, full-dataset PPL values from the original assignment; they are **not held-out test PPL** and cannot be directly compared with results from the corrected pipeline.

Setting #2 has the lowest reported PPL and the longest training time, while Setting #3 is fastest and has the highest reported PPL. This comparison illustrates the observed tradeoff across the submitted configurations.

Learning rate, maximum epochs, and sequence length change together, so the experiment does not isolate the effect of any single hyperparameter. Early stopping can also make the actual epoch count lower than the configured maximum.

## Repository Structure

```text
.
├── config/
│   ├── setting1.yaml
│   ├── setting2.yaml
│   └── setting3.yaml
├── src/
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
├── main.py
├── requirements.txt
└── README.md
```

## Usage

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Prepare `data/amazon.txt` as UTF-8 text with one review per line, or supply another file using `--data`:

```bash
python main.py --config config/setting2.yaml --data data/amazon.txt
```

Use `config/setting1.yaml` or `config/setting3.yaml` for the other experiments. GPT-2 weights and tokenizer are loaded through Hugging Face; initial execution requires access to them or a local cache. The source review corpus and trained checkpoints are not included.

For Setting #2, outputs are written to `outputs/setting2/`:

- `best_model.pt` — checkpoint selected by validation loss
- `training_plot.png` — training loss and validation PPL

## Limitations

- The corrected code masks padding labels, evaluates the held-out test partition, and weights evaluation loss by valid target-token count. The historical results above were obtained before these changes and have not been reproduced.
- Warmup is configured but not applied in the training loop.
- Datasets with fewer than 10 reviews are rejected. Very short individual reviews may not contain enough next-token targets for meaningful perplexity.

The updated evaluation improves metric validity, but a controlled comparison still requires documenting the corpus, random seeds, software environment, and newly reproduced scores.

## Review

This project provided practice with generative language modeling, experiment configuration, validation monitoring, and checkpoint selection. It also highlights why token masking and evaluation partitions matter when comparing perplexity across configurations.

## Data and Third-party Resources

- **Dataset:** the assignment's Amazon review corpus is not bundled; original provider access and redistribution terms apply.
- **Model:** pretrained GPT-2 is downloaded from its publisher. Displayed PPL values are historical assignment results, not scores produced by the corrected held-out evaluator.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
