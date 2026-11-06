# GPT-2 Language Modeling on Amazon Reviews

[Portfolio](https://app.notion.com/p/3e968564df5a815bbd9ff17299b214cd)

Fine-tuning **GPT-2** on Amazon review text and comparing three training configurations using **Perplexity (PPL)** and training time.

## Project Overview

- **Task:** Causal Language Modeling (next-token prediction)
- **Model:** GPT-2 (`GPT2LMHeadModel`)
- **Data:** Amazon review text
- **Tech Stack:** Python, PyTorch, Hugging Face Transformers, Matplotlib

## Implementation

- Tokenizes one review per line with the GPT-2 tokenizer and uses the EOS token for padding.
- Fine-tunes with AdamW, masking padding labels with `-100`.
- Uses training and validation samples for checkpoint selection and early stopping.
- Computes PPL over the full input dataset after training.
- Saves a model checkpoint and a training-loss/validation-PPL plot.

The training code samples 400 reviews for training and 50 for validation when at least 500 reviews are available. For smaller datasets, it uses approximately 80% for training and 10% for validation.

## Results

| Setting | Learning Rate | Max Epochs | Max Length | PPL ↓ | Training Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| #1 | 3e-5 | 10 | 256 | 3.1366 | 72.35 min |
| #2 | 2e-5 | 15 | 512 | **1.8683** | 133.61 min |
| #3 | 5e-5 | 8 | 128 | 6.7798 | **34.64 min** |

All configurations use a **batch size of 4**.

Setting #2 produced the lowest reported PPL with the longest training time. Setting #3 trained fastest but recorded the highest PPL. These are comparisons across complete configurations, not isolated effects of individual hyperparameters.

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

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare `data/amazon.txt` as UTF-8 text with one review per line, then run:

```bash
python main.py --config config/setting2.yaml --data data/amazon.txt
```

Use the other YAML files for Settings #1 and #3. Outputs include `best_model.pt` and `training_plot.png` in the corresponding `outputs/setting*/` folder.

**Data:** The original Amazon review corpus is not distributed in this repository. The pretrained GPT-2 model and tokenizer are loaded from Hugging Face.

**Configuration note:** The YAML files retain `warmup_steps: 100` from the assignment settings; the published training loop does not apply a warmup scheduler.
