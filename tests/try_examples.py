"""Manually try the trained ChoiceFit model on your own examples (not a pytest test).

Edit tests/manual_examples.json, then run from the repo root:

    .venv/bin/python tests/try_examples.py
    .venv/bin/python tests/try_examples.py --examples my_file.json --model artifacts/sagemaker/<job>

Each example needs "message" and "choices" (label -> description, 2 or more options).
"question" is optional. "expected" (a choice label) is optional and, if present, is scored.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import DEFAULT_QUESTION, ChoiceModel  # noqa: E402

TRAINED_MAX_CHOICES = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--examples", default=str(ROOT / "tests" / "manual_examples.json"))
    parser.add_argument("--model", default=str(ROOT / "artifacts/sagemaker/choicefit-l4-full-b8-001"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--pair-batch-size", type=int, default=16)
    args = parser.parse_args()

    examples = json.loads(Path(args.examples).read_text())
    if isinstance(examples, dict):
        examples = [examples]
    model = ChoiceModel(args.model, device=args.device)
    graded = correct = 0
    for number, example in enumerate(examples, 1):
        choices = example["choices"]
        result = model.predict(example["message"], choices, example.get("question", DEFAULT_QUESTION),
                               pair_batch_size=args.pair_batch_size)
        print(f"\n=== Example {number} ===\nMessage : {example['message']}")
        print(f"Question: {example.get('question', DEFAULT_QUESTION)}")
        if len(choices) > TRAINED_MAX_CHOICES:
            print(f"NOTE: {len(choices)} options; training used at most {TRAINED_MAX_CHOICES}, "
                  "so expect lower accuracy.")
        for label, probability in sorted(result["probabilities"].items(), key=lambda item: -item[1]):
            marker = "<-- picked" if label == result["choice"] else ""
            print(f"  {probability * 100:6.2f}%  {label}: {choices[label]}  {marker}")
        if "expected" in example:
            graded += 1
            hit = result["choice"] == example["expected"]
            correct += hit
            print(f"Expected: {example['expected']}  ->  {'CORRECT' if hit else 'WRONG'}")
    if graded:
        print(f"\nScored examples: {correct}/{graded} correct")


if __name__ == "__main__":
    main()
