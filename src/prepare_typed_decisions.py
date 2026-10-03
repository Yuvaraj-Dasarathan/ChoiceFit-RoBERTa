"""Prepare bounded local choice samples from Tasksource's typed decisions.

The upstream target is a distribution over the ordered options. IDs are local
bookkeeping only; target probabilities stay attached to their option text.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

from transformers import AutoTokenizer

from .data import message_key
from .model import validate

DATASET = "tasksource/tasksource-jev-typed-decisions"
# Pin the inspected release; --revision lets the user explicitly select another.
REVISION = "d2ab1d12be4463fb7ac1f877ed2193b3bb59642a"


def convert_choice(row):
    if row.get("kind") != "choice":
        raise ValueError("Expected a choice record")
    options, target = row.get("options"), row.get("target")
    if not isinstance(options, list) or not isinstance(target, list) or len(options) != len(target):
        raise ValueError("options and target must be equally sized lists")
    ids = [f"option_{i}" for i in range(len(options))]
    converted = {"message": row.get("state"), "question": row.get("question"),
                 "choices": dict(zip(ids, options)), "target": dict(zip(ids, target))}
    for key in ("id", "group_id", "question_id", "source", "variant", "split", "license", "license_use"):
        if key in row:
            converted[key] = row[key]
    validate(converted, labeled=True)
    return converted


def fits_token_budget(row, tokenizer, max_length):
    context = f"Question: {row['question']}\nMessage: {row['message']}"
    # Same formatting and special tokens as encode(), but no truncation.
    tokens = tokenizer([context] * len(row["choices"]),
                       [f"Candidate description: {text}" for text in row["choices"].values()],
                       truncation=False, padding=False, verbose=False)
    return all(len(ids) <= max_length for ids in tokens["input_ids"])


def sample_split(rows, tokenizer, limit, scan_limit, max_choices, max_length, seed,
                 excluded_groups=None, excluded_messages=None, reservoir_groups=32, direct_only=False):
    """Reservoir-sample groups per source, then interleave sources.

    Scan limits bound download/CPU work. This is a small diverse development
    sample, not an unbiased sample of the entire upstream mixture.
    """
    rng = random.Random(seed)
    reservoirs, seen_per_source = defaultdict(list), Counter()
    stats = Counter()
    excluded_groups = excluded_groups or set()
    excluded_messages = excluded_messages or set()
    # Consume complete groups; stop only between groups so the scan cap is soft.
    for group_id, members in groupby(rows, key=lambda row: row["group_id"]):
        if stats["scanned"] >= scan_limit:
            break
        members = list(members)
        stats["scanned"] += len(members)
        if group_id in excluded_groups:
            stats["excluded_group"] += len(members)
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
                converted = convert_choice(raw)
            except (ValueError, TypeError):
                stats["invalid"] += 1
                continue
            if len(converted["choices"]) > max_choices:
                stats["too_many_choices"] += 1
                continue
            if not fits_token_budget(converted, tokenizer, max_length):
                stats["too_long"] += 1
                continue
            candidates.append(converted)
        # Exclude an entire eligible group if any member overlaps earlier splits.
        if any(message_key(r["message"]) in excluded_messages for r in candidates):
            stats["overlapping_message"] += len(candidates)
            continue
        if not candidates:
            continue
        source = candidates[0]["source"]
        seen_per_source[source] += 1
        pool = reservoirs[source]
        if len(pool) < reservoir_groups:
            pool.append(candidates)
        else:
            slot = rng.randrange(seen_per_source[source])
            if slot < reservoir_groups:
                pool[slot] = candidates
    for pool in reservoirs.values():
        rng.shuffle(pool)
    selected = []
    active = list(reservoirs)
    while active and len(selected) < limit:
        rng.shuffle(active)
        remaining = []
        for source in active:
            group = reservoirs[source].pop()
            if len(selected) + len(group) <= limit:
                selected.extend(group)
            else:
                stats["group_exceeds_remaining_limit"] += len(group)
            if reservoirs[source]:
                remaining.append(source)
        active = remaining
    rng.shuffle(selected)
    stats["selected"] = len(selected)
    stats["soft_targets"] = sum(not any(p == 1.0 for p in row["target"].values()) for row in selected)
    return selected, dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/typed-local")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--config", default="default")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--tokenizer", default="FacebookAI/roberta-base")
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--validation-limit", type=int, default=100)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--scan-limit", type=int, default=10000, help="Maximum source rows scanned per split (complete final group)")
    parser.add_argument("--max-choices", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--reservoir-groups", type=int, default=32, help="Bounded candidate group buffer per source")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-dir", help="Optional raw train/validation/test JSONL files for offline conversion")
    args = parser.parse_args()
    if min(args.train_limit, args.validation_limit, args.test_limit, args.scan_limit, args.reservoir_groups) < 1:
        parser.error("Limits must be positive")
    if not 2 <= args.max_choices <= 255 or not 16 <= args.max_length <= 512:
        parser.error("Require 2–255 choices and 16–512 tokens")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output is not empty; choose a new --output")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prepared, report = {}, {}
    used_groups, used_messages = set(), set()
    for index, split in enumerate(("train", "validation", "test")):
        print(json.dumps({"preparing": split, "scan_limit": args.scan_limit}), flush=True)
        if args.source_dir:
            source = open(Path(args.source_dir) / f"{split}.jsonl")
            rows = (json.loads(line) for line in source if line.strip())
        else:
            from datasets import load_dataset
            source = None
            rows = load_dataset(args.dataset, name=args.config, revision=args.revision,
                                split=split, streaming=True)
        try:
            selected, stats = sample_split(
                rows, tokenizer, getattr(args, f"{split}_limit"), args.scan_limit,
                args.max_choices, args.max_length, args.seed + index,
                used_groups, used_messages, args.reservoir_groups, direct_only=split != "train")
        finally:
            if source is not None:
                source.close()
        if not selected:
            raise ValueError(f"No eligible {split} records; inspect filtering or increase --scan-limit")
        used_groups.update(row["group_id"] for row in selected)
        used_messages.update(message_key(row["message"]) for row in selected)
        prepared[split] = selected
        report[split] = {"counts": stats, "sources": dict(Counter(row["source"] for row in selected))}
        print(json.dumps({"split": split, **stats, "sources": len(report[split]["sources"])}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in prepared.items():
        with (output / f"{split}.jsonl").open("w") as destination:
            for row in rows:
                destination.write(json.dumps(row) + "\n")
    (output / "example.json").write_text(json.dumps(prepared["test"][0], indent=2))
    manifest = {"arguments": vars(args), "splits": report,
                "sampling": "Per-source group reservoirs within a bounded scan; source-interleaved selection. Not a benchmark sample.",
                "filtering": "choice only; multilingual/ sources excluded (not language detection); full token pairs must fit; direct-only evaluation",
                "targets": "Original probabilities preserved; no argmax conversion",
                "license": "Per-source upstream terms apply; per-record license metadata retained"}
    (output / "preparation.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
