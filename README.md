# GPT-2 Language Modeling on Amazon Reviews

A language modeling project fine-tuning **GPT-2** on Amazon review text and comparing training configurations using **Perplexity (PPL)** and training time.

[Portfolio](https://incredible-march-0ef.notion.site/GPT-2-Language-Modeling-on-Amazon-Reviews-3e968564df5a815bbd9ff17299b214cd)

## Project Overview

The task is causal language modeling: predicting the next token in review text. Three experiments vary learning rate, epoch count, and maximum sequence length to examine model loss and computational cost.

**Technologies:** Python, PyTorch, Hugging Face Transformers, GPT-2, Matplotlib.

## Experiment Settings & Reported Results

| Setting | Learning Rate | Maximum Epochs | Max Length | Training Time | Reported PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| #1 | 3e-5 | 10 | 256 | 72.35 min | 3.1366 |
| #2 | 2e-5 | 15 | 512 | 133.61 min | **1.8683** |
| #3 | 5e-5 | 8 | 128 | **34.64 min** | 6.7798 |

Batch size is **4** across configurations. The assignment settings specify **100 warmup steps**, although the current training loop does not implement a warmup scheduler.

The PPL and training-time values are transcribed from the submitted assignment evaluation sheet; they have not been reproduced in this documentation update.

## Implementation

- Reads UTF-8 review text with one non-empty review per line.
- Uses the GPT-2 tokenizer with the EOS token as the padding token.
- Fine-tunes `GPT2LMHeadModel` with input token IDs as labels.
- Uses AdamW, validation-loss checkpoint selection, and early stopping.
- Saves model weights and a training-loss/validation-PPL plot.
- Separates dataset preparation, training, evaluation, and experiment settings.

### Data Split

For datasets with at least 500 reviews, the training function assigns **400 reviews to training**, **50 to validation**, and the remainder to a third partition. Smaller datasets use approximately 80% training and 10% validation, with the remainder reserved. Split randomness is controlled by the configuration seed.

The third partition is currently discarded by the training function. After restoring the best checkpoint, `main.py` evaluates the **full dataset**, including training and validation examples. Its final printed PPL therefore is not a held-out test metric.

## Findings

Setting #2 has the lowest reported PPL and the longest training time, while Setting #3 is fastest and has the highest reported PPL. This comparison illustrates the observed tradeoff across the submitted configurations.

Learning rate, maximum epochs, and sequence length change together, so the experiment does not isolate the effect of any single hyperparameter. Early stopping can also make the actual epoch count lower than the configured maximum.

## Project Structure

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

## Run

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

## Evaluation Limitations

- Padding positions remain in the labels rather than being masked with `-100`. Consequently, the loss includes padded EOS tokens, and different sequence lengths can change how much padding contributes.
- Final evaluation uses the full dataset, rather than the reserved partition.
- Evaluation averages batch losses equally instead of aggregating negative log-likelihood by valid target-token count.
- Warmup is configured but not applied in the training loop.
- Very small datasets can produce an empty training or validation split.

These details limit interpretation of the reported PPL. A stronger evaluation would mask padding labels, use a held-out test partition, aggregate loss over valid target tokens, and document the corpus and experimental environment. Any corrected rerun should be reported separately from the historical assignment results.

## Review

This project provided practice with generative language modeling, experiment configuration, validation monitoring, and checkpoint selection. It also highlights why token masking and evaluation partitions matter when comparing perplexity across configurations.
