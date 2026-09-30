import argparse
import json
from .model import ChoiceModel, DEFAULT_QUESTION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/choicefit-roberta")
    parser.add_argument("--message", required=True)
    parser.add_argument("--choices", default="data/choices.json", help="JSON file mapping labels to descriptions")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    with open(args.choices) as source:
        choices = json.load(source)
    print(json.dumps(ChoiceModel(args.model, args.device).predict(args.message, choices, args.question), indent=2))


if __name__ == "__main__":
    main()
