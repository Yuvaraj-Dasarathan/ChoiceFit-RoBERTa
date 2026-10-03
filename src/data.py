"""Bounded-memory JSONL iteration, shuffling, and split auditing."""
import hashlib
import random
import sqlite3
import tempfile
from pathlib import Path

from .model import iter_data


def message_key(message):
    return hashlib.sha256(" ".join(message.casefold().split()).encode("utf-8")).digest()


def audit_splits(train_path, validation_path, max_pairs):
    """Keep overlap hashes on disk, rather than keeping all messages in RAM."""
    counts = {}
    with tempfile.TemporaryDirectory(prefix="choicefit-audit-") as directory:
        with sqlite3.connect(Path(directory) / "messages.db") as connection:
            connection.execute("CREATE TABLE messages (digest BLOB PRIMARY KEY)")
            connection.execute("CREATE TABLE groups (id TEXT PRIMARY KEY)")
            for split, path in (("train", train_path), ("validation", validation_path)):
                count = 0
                for row in iter_data(path):
                    digest = message_key(row["message"])
                    group = row.get("group_id")
                    if split == "train":
                        if len(row["choices"]) > max_pairs:
                            raise ValueError("A training example exceeds --max-pairs; sample fewer "
                                             "training choices or increase the budget")
                        connection.execute("INSERT OR IGNORE INTO messages VALUES (?)", (digest,))
                        if group:
                            connection.execute("INSERT OR IGNORE INTO groups VALUES (?)", (group,))
                    else:
                        if connection.execute("SELECT 1 FROM messages WHERE digest=?", (digest,)).fetchone():
                            raise ValueError("Training and validation messages overlap after whitespace/case normalization")
                        if group and connection.execute("SELECT 1 FROM groups WHERE id=?", (group,)).fetchone():
                            raise ValueError("Training and validation group_id values overlap")
                    count += 1
                counts[split] = count
    return counts


def buffered_shuffle(rows, buffer_size, seed):
    if buffer_size < 1:
        raise ValueError("shuffle buffer must be positive")
    rng = random.Random(seed)
    buffer = []
    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
        else:
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = row
    rng.shuffle(buffer)
    yield from buffer


def batches(rows, batch_size, max_pairs=None):
    if batch_size < 1 or (max_pairs is not None and max_pairs < 1):
        raise ValueError("Batch and pair limits must be positive")
    batch, pairs = [], 0
    for row in rows:
        size = len(row["choices"])
        if max_pairs is not None and size > max_pairs:
            raise ValueError("A single example exceeds the training pair budget")
        if batch and (len(batch) >= batch_size or (max_pairs is not None and pairs + size > max_pairs)):
            yield batch
            batch, pairs = [], 0
        batch.append(row)
        pairs += size
    if batch:
        yield batch
