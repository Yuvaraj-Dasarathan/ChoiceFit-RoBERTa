"""Publish the ChoiceFit model and demo Space to the Hugging Face Hub.

Prerequisite: a free account and a WRITE token (https://huggingface.co/settings/tokens).
Authenticate once with `.venv/bin/python -m huggingface_hub.commands.huggingface_cli login`
or export HF_TOKEN in your shell (never commit it).

    .venv/bin/python scripts/publish_to_hf.py --username YOUR_HF_NAME --dry-run
    .venv/bin/python scripts/publish_to_hf.py --username YOUR_HF_NAME
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILES = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "special_tokens_map.json"]

MODEL_CARD = """---
license: other
license_name: research-and-educational-use-only
language: en
base_model: FacebookAI/roberta-base
datasets:
- tasksource/tasksource-jev-typed-decisions
tags:
- multiple-choice
- roberta
---

# ChoiceFit RoBERTa

RoBERTa-base (125M parameters) fine-tuned to pick the best option for a message. Each (question + message, option
description) pair gets one score from a single-output classification head; a softmax across a question's options
gives probabilities. Trained on about 486k examples (2-8 options, 256 tokens) from
[tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions),
a collection of roughly 400 source datasets with mixed licenses (some non-commercial).

**Intended use: research and education only. Not for commercial use.** This is a personal learning project.

## Results (hard accuracy)

| Test set | Untrained RoBERTa | ChoiceFit | JEV (TypeSafe) |
| --- | ---: | ---: | ---: |
| Held-out mixed benchmark (6,597 rows) | 37.0% | 66.7% | 78.3% |
| banking77, 4 options (500 rows) | 12.2% | 67.0% | 96.2% |
| clinc_oos, 4 options (500 rows) | 16.2% | 82.8% | 99.4% |

## Limitations
- Trained on 2-8 options; more options are untested and likely less accurate.
- Weak on knowledge-heavy and many-similar-label questions.
- Overconfident: stated probabilities run higher than actual accuracy.

Code: https://github.com/Yuvaraj-Dasarathan/ChoiceFit-RoBERTa
"""

SPACE_README = """---
title: ChoiceFit RoBERTa
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: other
---

Demo of ChoiceFit, a RoBERTa-base multiple-choice model. For research and educational use only.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--model-dir", default=str(ROOT / "artifacts/sagemaker/choicefit-l4-full-b8-001"))
    parser.add_argument("--model-name", default="choicefit-roberta")
    parser.add_argument("--space-name", default="choicefit-roberta-demo")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded and stop")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    missing = [name for name in MODEL_FILES if not (model_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing model files in {model_dir}: {missing}")
    model_repo = f"{args.username}/{args.model_name}"
    space_repo = f"{args.username}/{args.space_name}"
    space_files = {"app.py": ROOT / "demo/app.py", "requirements.txt": ROOT / "demo/requirements.txt",
                   "model.py": ROOT / "src/model.py"}
    print(f"Model repo : {model_repo}\n  files: {MODEL_FILES} + README.md")
    print(f"Space repo : {space_repo}\n  files: {list(space_files)} + README.md")
    if args.dry_run:
        return

    api = HfApi()
    print("Logged in as:", api.whoami()["name"])
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    for name in MODEL_FILES:
        api.upload_file(path_or_fileobj=str(model_dir / name), path_in_repo=name, repo_id=model_repo)
    api.upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md", repo_id=model_repo)

    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    for name, path in space_files.items():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=space_repo, repo_type="space")
    api.upload_file(path_or_fileobj=SPACE_README.encode(), path_in_repo="README.md",
                    repo_id=space_repo, repo_type="space")
    api.add_space_variable(space_repo, "CHOICEFIT_MODEL", model_repo)
    print(f"Done.\nModel: https://huggingface.co/{model_repo}\nSpace: https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
