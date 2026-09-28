# GPT-2 Language Modeling on Amazon Reviews

A language modeling project that fine-tunes **GPT-2** on Amazon review text and compares hyperparameter settings using **Perplexity (PPL)**.

## Overview

Three experiments vary the learning rate, number of epochs, and maximum sequence length.

| Setting | Learning Rate | Epochs | Max Length | Training Time | PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| #1 | 3e-5 | 10 | 256 | 72.35 min | 3.1366 |
| #2 | 2e-5 | 15 | 512 | 133.61 min | **1.8683** |
| #3 | 5e-5 | 8 | 128 | 34.64 min | 6.7798 |

Batch size 4 and 100 warmup steps were used across all three settings.

## What I Implemented

- Loaded Amazon review text as a causal language modeling dataset.
- Tokenized and padded sequences with the GPT-2 tokenizer.
- Fine-tuned `GPT2LMHeadModel` with labels equal to the input token IDs.
- Monitored validation perplexity during training.
- Applied early stopping and saved the best checkpoint.
- Compared three hyperparameter settings by PPL and training time.

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

```bash
python main.py --config config/setting2.yaml
```

## Notes

The PPL and training-time values above are taken from the submitted assignment evaluation sheet. The portfolio repository reorganizes the supplied implementation while preserving its core GPT-2 training and evaluation workflow.
