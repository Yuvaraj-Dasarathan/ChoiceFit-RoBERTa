"""Evaluate a saved choice model on a held-out JSONL split."""
import argparse
import json
from contextlib import nullcontext

import torch

from .model import ChoiceModel, iter_data
from .train import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=None,
                        help="Tokens per pair; default is the model's saved choice_max_length "
                             "(use 256 for the untrained base model to match fine-tuning)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seeds the random scoring head of an untrained base model")
    parser.add_argument("--predictions", help="Write per-example predictions to a new JSONL file")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = ChoiceModel(args.model, device=args.device, max_length=args.max_length)
    with (open(args.predictions, "x") if args.predictions else nullcontext()) as sink:
        print(json.dumps(evaluate(model, iter_data(args.data), args.batch_size,
                                  args.pair_batch_size, prediction_sink=sink), indent=2))


if __name__ == "__main__":
    main()
