import argparse

import torch
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.dataset import AmazonReviewDataset
from src.evaluate import evaluate
from src.train import train_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/amazon.txt")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    args = parse_args()
    config = load_config(args.config)

    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    dataset = AmazonReviewDataset(
        args.data,
        tokenizer,
        config["max_length"],
    )

    model = GPT2LMHeadModel.from_pretrained(config["model_name"]).to(device)
    checkpoint_path = train_model(model, dataset, config, device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    metrics = evaluate(model, dataset, config["batch_size"], device)

    print(f"Evaluation loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")


if __name__ == "__main__":
    main()
