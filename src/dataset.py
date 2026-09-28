import torch
from torch.utils.data import Dataset


class AmazonReviewDataset(Dataset):
    """Tokenize Amazon review lines for causal language modeling."""

    def __init__(self, file_path, tokenizer, max_length):
        with open(file_path, "r", encoding="utf-8") as file:
            self.texts = [line.strip() for line in file if line.strip()]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
