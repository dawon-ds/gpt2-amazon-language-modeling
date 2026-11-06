import math

import torch
from torch.utils.data import DataLoader


def evaluate(model, dataset, batch_size, device):
    """Return token-weighted causal LM loss and perplexity for a dataset."""
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset must not be empty.")
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    loss_sum = 0.0
    token_count = 0

    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            valid_targets = (inputs["labels"][:, 1:] != -100).sum().item()
            if valid_targets == 0:
                continue
            loss = model(**inputs).loss
            loss_sum += loss.item() * valid_targets
            token_count += valid_targets

    if token_count == 0:
        raise ValueError("No valid next-token targets for evaluation.")
    average_loss = loss_sum / token_count
    return {
        "loss": average_loss,
        "perplexity": math.exp(average_loss),
    }
