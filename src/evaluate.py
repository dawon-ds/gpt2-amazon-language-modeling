import math

import torch
from torch.utils.data import DataLoader


def evaluate(model, dataset, batch_size, device):
    """Evaluate a causal language model and return loss and perplexity."""
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            total_loss += model(**inputs).loss.item()

    average_loss = total_loss / len(loader)
    return {
        "loss": average_loss,
        "perplexity": math.exp(average_loss),
    }
