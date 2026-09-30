"""A shared RoBERTa scorer, trained with cross entropy across supplied choices."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_QUESTION = "Which category best matches this message?"


def validate(row, labeled=False):
    if not isinstance(row, dict):
        raise ValueError("Each example must be a JSON object")
    for field in ("message", "question"):
        value = row.get(field, DEFAULT_QUESTION if field == "question" else None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string")
    choices = row.get("choices")
    if not isinstance(choices, dict) or not 2 <= len(choices) <= 255:
        raise ValueError("choices must map 2–255 labels to descriptions")
    if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
           for k, v in choices.items()):
        raise ValueError("Choice labels and descriptions must be nonempty strings")
    if labeled and row.get("label") not in choices:
        raise ValueError("label must be one of the supplied choices")


def read_data(path):
    rows = []
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate(row, labeled=True)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def encode(tokenizer, rows, max_length):
    contexts, candidates, sizes = [], [], []
    for row in rows:
        validate(row)
        sizes.append(len(row["choices"]))
        for description in row["choices"].values():
            contexts.append(f"Question: {row.get('question', DEFAULT_QUESTION)}\nMessage: {row['message']}")
            # IDs deliberately stay out of the encoder: descriptions define meaning.
            candidates.append(f"Candidate description: {description}")
    tokens = tokenizer(contexts, candidates, padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    return tokens, sizes


def grouped_logits(scores, sizes):
    return torch.nn.utils.rnn.pad_sequence(
        scores.flatten().split(sizes), batch_first=True, padding_value=float("-inf"))


class ChoiceModel:
    def __init__(self, checkpoint, device=None, max_length=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                               "mps" if torch.backends.mps.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1).to(self.device)
        self.max_length = max_length or getattr(self.model.config, "choice_max_length", 192)
        if not 16 <= self.max_length <= 512:
            raise ValueError("max_length must be between 16 and 512")

    def logits(self, rows):
        tokens, sizes = encode(self.tokenizer, rows, self.max_length)
        scores = self.model(**{k: v.to(self.device) for k, v in tokens.items()}).logits
        return grouped_logits(scores, sizes)

    @torch.inference_mode()
    def predict(self, message, choices, question=DEFAULT_QUESTION):
        row = dict(message=message, choices=choices, question=question)
        validate(row)
        self.model.eval()
        probabilities = self.logits([row])[0].softmax(-1).cpu().tolist()
        labels = list(choices)
        best = max(range(len(labels)), key=probabilities.__getitem__)
        return {"type": "choice", "choice": labels[best],
                "probabilities": dict(zip(labels, probabilities)),
                "top_probability": probabilities[best]}

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.config.choice_max_length = self.max_length
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
