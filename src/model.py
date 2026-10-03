"""A shared RoBERTa scorer, trained with cross entropy across supplied choices."""
import json
import math
import sys
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
    if labeled:
        if "target" in row:
            target = row["target"]
            if "label" in row:
                raise ValueError("Supply target probabilities or label, not both")
            if not isinstance(target, dict) or set(target) != set(choices):
                raise ValueError("target must map every choice ID to a probability")
            if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
                   not math.isfinite(p) or p < 0 for p in target.values()):
                raise ValueError("target probabilities must be finite and nonnegative")
            if not math.isclose(sum(target.values()), 1.0, abs_tol=1e-5):
                raise ValueError("target probabilities must sum to 1")
        elif row.get("label") not in choices:
            raise ValueError("label must be one of the supplied choices")


def iter_data(path):
    """Validate and yield JSONL records without retaining the whole dataset."""
    count = 0
    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate(row, labeled=True)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            count += 1
            yield row
    if not count:
        raise ValueError(f"Empty dataset: {path}")


def read_data(path):
    return list(iter_data(path))


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

    def logits(self, rows, debug=False, pair_batch_size=None):
        if pair_batch_size is not None:
            if pair_batch_size < 1:
                raise ValueError("pair_batch_size must be positive")
            if self.model.training or torch.is_grad_enabled():
                raise ValueError("Pair chunking is for eval mode with gradients disabled")
        tokens, sizes = encode(self.tokenizer, rows, self.max_length)
        hooks = []

        def show(name, tensor):
            sample = tensor.detach().reshape(-1)[:8].cpu().tolist()
            print(f"[debug] {name}: shape={tuple(tensor.shape)}, first values={sample}", file=sys.stderr)

        if debug:
            print(f"[debug] choices per message={sizes}; device={self.device}; "
                  f"training={self.model.training}", file=sys.stderr)
            for index, ids in enumerate(tokens["input_ids"]):
                print(f"[debug] pair {index}: {self.tokenizer.decode(ids, skip_special_tokens=False)}",
                      file=sys.stderr)
            for name, tensor in tokens.items():
                show(name, tensor)

        try:
            if debug:
                # Observe the actual forward pass without recomputing or replacing it.
                head = self.model.classifier

                def head_input(module, inputs):
                    show("RoBERTa output / head input", inputs[0])
                    show("first-token <s> vectors", inputs[0][:, 0, :])

                hooks.append(head.register_forward_pre_hook(head_input))
                hooks.append(head.dense.register_forward_hook(
                    lambda module, inputs, output: show("head dense output (before tanh)", output)))
                hooks.append(head.out_proj.register_forward_pre_hook(
                    lambda module, inputs: show("head tanh + second dropout / out_proj input", inputs[0])))
            # This calls BOTH the RoBERTa encoder and its scalar classification head.
            chunk_size = pair_batch_size or sum(sizes)
            scores = torch.cat([
                self.model(**{k: v[start:start + chunk_size].to(self.device)
                              for k, v in tokens.items()}).logits
                for start in range(0, sum(sizes), chunk_size)
            ])
        finally:
            for hook in hooks:
                hook.remove()
        logits = grouped_logits(scores, sizes)
        if debug:
            show("head out_proj output (one raw score per pair)", scores)
            print(f"[debug] grouped logits={logits.detach().cpu().tolist()}", file=sys.stderr)
        return logits

    @torch.inference_mode()
    def predict(self, message, choices, question=DEFAULT_QUESTION, debug=False, pair_batch_size=16):
        row = dict(message=message, choices=choices, question=question)
        validate(row)
        self.model.eval()
        probabilities = self.logits([row], debug=debug, pair_batch_size=pair_batch_size)[0].softmax(-1).cpu().tolist()
        labels = list(choices)
        best = max(range(len(labels)), key=probabilities.__getitem__)
        if debug:
            print(f"[debug] softmax across choices={dict(zip(labels, probabilities))}", file=sys.stderr)
            print(f"[debug] winning index={best}; returned label={labels[best]!r}", file=sys.stderr)
        return {"type": "choice", "choice": labels[best],
                "probabilities": dict(zip(labels, probabilities)),
                "top_probability": probabilities[best]}

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.config.choice_max_length = self.max_length
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
