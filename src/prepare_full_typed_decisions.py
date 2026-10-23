"""Stream the full supported typed-decisions splits to training JSONL files."""
import argparse
import json
from collections import Counter
from itertools import groupby
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from .data import message_key
from .model import validate
from .prepare_typed_decisions import DATASET, REVISION, convert_choice, fits_token_budget


def write_split(rows, destination, tokenizer, limit, max_choices, max_length,
                used_groups, used_messages, direct_only):
    """Write eligible complete groups without retaining the split in memory."""
    stats, sources, selected, soft_targets = Counter(), Counter(), 0, 0
    first = None
    with destination.open("x") as output:
        for group_id, members in groupby(rows, key=lambda row: row["group_id"]):
            members = list(members)
            stats["scanned"] += len(members)
            if group_id in used_groups:
                stats["overlapping_group"] += len(members)
                continue
            candidates = []
            for raw in members:
                if raw.get("kind") != "choice":
                    stats["not_choice"] += 1
                    continue
                if str(raw.get("source", "")).startswith("multilingual/"):
                    stats["multilingual_source"] += 1
                    continue
                if direct_only and raw.get("variant") != "direct":
                    stats["not_direct"] += 1
                    continue
                try:
                    row = convert_choice(raw)
                except (ValueError, TypeError):
                    stats["invalid"] += 1
                    continue
                if len(row["choices"]) > max_choices:
                    stats["too_many_choices"] += 1
                    continue
                if not fits_token_budget(row, tokenizer, max_length):
                    stats["too_long"] += 1
                    continue
                candidates.append(row)
            if not candidates or any(message_key(row["message"]) in used_messages for row in candidates):
                if candidates:
                    stats["overlapping_message"] += len(candidates)
                continue
            if limit and selected + len(candidates) > limit:
                stats["group_exceeds_limit"] += len(candidates)
                # A bounded pilot should finish promptly. Stopping at this
                # complete group keeps the split just below its requested cap
                # without scanning the rest of a million-row source split.
                break
            for row in candidates:
                output.write(json.dumps(row) + "\n")
                selected += 1
                sources[row["source"]] += 1
                soft_targets += int(not any(value == 1.0 for value in row["target"].values()))
                if first is None:
                    first = row
            used_groups.add(group_id)
            used_messages.update(message_key(row["message"]) for row in candidates)
    stats["selected"] = selected
    stats["soft_targets"] = soft_targets
    if not selected:
        raise ValueError(f"No eligible rows for {destination.name}")
    return dict(stats), dict(sources), first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/typed-full")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--config", default="default")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--tokenizer", default="FacebookAI/roberta-base")
    parser.add_argument("--train-limit", type=int, default=1_000_000,
                        help="0 means all eligible training rows")
    parser.add_argument("--validation-limit", type=int, default=0,
                        help="0 means all eligible validation rows")
    parser.add_argument("--test-limit", type=int, default=0,
                        help="0 means all eligible test rows")
    parser.add_argument("--max-choices", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()
    if min(args.train_limit, args.validation_limit, args.test_limit) < 0:
        parser.error("Limits must be zero or positive")
    if not 2 <= args.max_choices <= 255 or not 16 <= args.max_length <= 512:
        parser.error("Require 2–255 choices and 16–512 tokens")
    output = Path(args.output)
    if output.exists():
        parser.error("Output directory already exists; choose a new output path")
    output.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    used_groups, used_messages, report, example = set(), set(), {}, None
    for split, direct_only in (("train", False), ("validation", True), ("test", True)):
        print(json.dumps({"event": "preparing_split", "split": split}), flush=True)
        rows = load_dataset(args.dataset, name=args.config, revision=args.revision,
                            split=split, streaming=True)
        stats, sources, first = write_split(
            rows, output / f"{split}.jsonl", tokenizer, getattr(args, f"{split}_limit"),
            args.max_choices, args.max_length, used_groups, used_messages, direct_only)
        report[split] = {"counts": stats, "sources": sources}
        if split == "test":
            example = first
        print(json.dumps({"event": "prepared_split", "split": split,
                          "rows": stats["selected"], "sources": len(sources)}), flush=True)
    (output / "example.json").write_text(json.dumps(example, indent=2))
    (output / "preparation.json").write_text(json.dumps({"arguments": vars(args), "splits": report,
        "filtering": "choice only; <=4 choices; complete pairs <=192 tokens; multilingual excluded; direct-only evaluation",
        "targets": "Original probability distributions preserved; no argmax conversion"}, indent=2))


if __name__ == "__main__":
    main()
