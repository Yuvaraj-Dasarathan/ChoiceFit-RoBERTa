import argparse
import json
from .model import ChoiceModel, DEFAULT_QUESTION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/typed-local-v1")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--message")
    inputs.add_argument("--input-json", help="Prepared example with message, question, and choices")
    parser.add_argument("--choices", help="JSON file mapping labels to descriptions")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pair-batch-size", type=int, default=16, help="Maximum candidate pairs per encoder forward pass")
    parser.add_argument("--debug", action="store_true", help="Print input pairs, tensor shapes, and choice scores to stderr")
    args = parser.parse_args()
    if args.message is not None and not args.choices:
        parser.error("--choices is required with --message; or use --input-json")
    if args.input_json:
        with open(args.input_json) as source:
            row = json.load(source)
        message, choices, question = row["message"], row["choices"], row.get("question", DEFAULT_QUESTION)
    else:
        message, question = args.message, args.question
        with open(args.choices) as source:
            choices = json.load(source)
    print(json.dumps(ChoiceModel(args.model, args.device).predict(
        message, choices, question, debug=args.debug,
        pair_batch_size=args.pair_batch_size), indent=2))

if __name__ == "__main__":
    main()
