"""Convert a customer-message intent dataset into ChoiceFit's labeled JSONL test format.

Each message becomes a choice question: the true intent plus (N-1) random distractor intents,
in shuffled order. Outputs the same fields as data/typed-full-8x256/test.jsonl, so
evaluate / evaluate_jev / compare_all work unchanged.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

QUESTION = "Which category best describes what the customer wants?"

# name -> (hub id, config, text column, label-name column or ClassLabel column, test split)
DATASETS = {
    "banking77": ("mteb/banking77", None, "text", "label_text"),
    "clinc_oos": ("clinc/clinc_oos", "plus", "text", "intent"),
}


def describe(label):
    return label.replace("_", " ").strip()


def label_names(dataset, column):
    feature = dataset.features[column]
    if hasattr(feature, "names"):
        return lambda value: feature.names[value]
    return str


def convert(name, split, num_choices, limit, seed):
    hub_id, config, text_column, label_column = DATASETS[name]
    data = load_dataset(hub_id, config, split=split) if config else load_dataset(hub_id, split=split)
    to_name = label_names(data, label_column)
    labels = sorted({to_name(value) for value in data[label_column]} - {"oos"})
    if not 2 <= num_choices <= len(labels):
        raise ValueError(f"--choices must be between 2 and {len(labels)}")
    rng = random.Random(seed)
    indexes = [i for i in range(len(data)) if to_name(data[i][label_column]) != "oos"]
    rng.shuffle(indexes)
    for position, index in enumerate(indexes[:limit or None]):
        row = data[index]
        truth = to_name(row[label_column])
        options = rng.sample([label for label in labels if label != truth], num_choices - 1) + [truth]
        rng.shuffle(options)
        keys = [f"option_{i}" for i in range(num_choices)]
        identifier = f"{name}:{split}:{index}"
        yield {
            "message": row[text_column],
            "question": QUESTION,
            "choices": {key: describe(label) for key, label in zip(keys, options)},
            "target": {key: float(label == truth) for key, label in zip(keys, options)},
            "id": identifier, "group_id": identifier, "question_id": "decision",
            "source": name, "variant": f"{num_choices}-way", "split": split,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--choices", type=int, default=4, help="Options per question (true intent + distractors)")
    parser.add_argument("--limit", type=int, default=500, help="Rows to keep after shuffling; 0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="New JSONL file")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("x") as sink:
        for row in convert(args.dataset, args.split, args.choices, args.limit, args.seed):
            sink.write(json.dumps(row) + "\n")
            count += 1
    print(json.dumps({"output": str(output), "rows": count, "choices": args.choices}))


if __name__ == "__main__":
    main()
