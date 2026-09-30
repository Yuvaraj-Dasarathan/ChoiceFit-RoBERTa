"""Local or single-instance SageMaker training entry point."""
import argparse
import json
import os
import random
from pathlib import Path

import torch
from transformers import set_seed

from .model import ChoiceModel, read_data


def targets(rows, device):
    return torch.tensor([list(r["choices"]).index(r["label"]) for r in rows], device=device)


@torch.inference_mode()
def evaluate(model, rows, batch_size):
    model.model.eval()
    correct = total_loss = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        logits = model.logits(batch)
        labels = targets(batch, model.device)
        total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(-1) == labels).sum().item()
    return {"accuracy": correct / len(rows), "loss": total_loss / len(rows), "examples": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.path.join(os.getenv("SM_CHANNEL_TRAIN", "data"), "train.jsonl"))
    parser.add_argument("--validation", default=os.path.join(os.getenv("SM_CHANNEL_VALIDATION", "data"), "validation.jsonl"))
    parser.add_argument("--output", default=os.getenv("SM_MODEL_DIR", "artifacts/choicefit-roberta"))
    parser.add_argument("--model", default="FacebookAI/roberta-base")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4, help="Messages per batch; each expands to all its choices")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be positive")
    set_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    train, validation = read_data(args.train), read_data(args.validation)
    if {r["message"] for r in train} & {r["message"] for r in validation}:
        raise ValueError("Training and validation messages overlap")
    model = ChoiceModel(args.model, args.device, args.max_length)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    history = [{"epoch": 0, "validation": evaluate(model, validation, args.batch_size)}]
    print(json.dumps(history[-1]), flush=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.model.train()
        random.shuffle(train)
        total_loss = 0
        for start in range(0, len(train), args.batch_size):
            batch = train[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model.logits(batch), targets(batch, model.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)
        metrics = evaluate(model, validation, args.batch_size)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train), "validation": metrics})
        print(json.dumps(history[-1]), flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            model.save(args.output)
    Path(args.output, "training_metrics.json").write_text(json.dumps({"arguments": vars(args), "history": history}, indent=2))


if __name__ == "__main__":
    main()
