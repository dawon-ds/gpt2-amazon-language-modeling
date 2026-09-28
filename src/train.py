import math
import os

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def train_model(model, dataset, config, device):
    """Train GPT-2 and save the checkpoint with the best validation loss."""
    total = len(dataset)
    if total == 0:
        raise ValueError("The dataset is empty.")

    if total >= 500:
        train_size, validation_size, test_size = 400, 50, total - 450
    else:
        train_size = int(total * 0.8)
        validation_size = int(total * 0.1)
        test_size = total - train_size - validation_size

    generator = torch.Generator().manual_seed(config["seed"])
    train_data, validation_data, _ = random_split(
        dataset,
        [train_size, validation_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config["batch_size"],
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    best_validation_loss = float("inf")
    no_improvement = 0
    train_losses = []
    validation_ppls = []

    os.makedirs(config["output_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["output_dir"], "best_model.pt")

    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**inputs).loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                inputs = {
                    key: value.to(device)
                    for key, value in batch.items()
                }
                total_validation_loss += model(**inputs).loss.item()

        validation_loss = total_validation_loss / len(validation_loader)
        validation_ppl = math.exp(validation_loss)
        train_losses.append(train_loss)
        validation_ppls.append(validation_ppl)

        print(
            f"Epoch {epoch + 1}: "
            f"loss={train_loss:.4f}, val_ppl={validation_ppl:.4f}"
        )

        if validation_loss < best_validation_loss - config["min_delta"]:
            best_validation_loss = validation_loss
            no_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            no_improvement += 1
            if no_improvement >= config["early_stop"]:
                break

    save_training_plot(
        train_losses,
        validation_ppls,
        config["output_dir"],
    )
    return checkpoint_path


def save_training_plot(train_losses, validation_ppls, output_dir):
    plt.figure()
    plt.plot(train_losses, marker="o", label="Train Loss")
    plt.plot(validation_ppls, marker="s", label="Validation PPL")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_plot.png"))
    plt.close()
