"""Publish the ChoiceFit model and its model card (docs/model_card.md) to the Hugging Face Hub.

Prerequisite: a free account and a WRITE token (https://huggingface.co/settings/tokens).
Authenticate once with `.venv/bin/python -m huggingface_hub.commands.huggingface_cli login`
or export HF_TOKEN in your shell (never commit it).

    .venv/bin/python scripts/publish_to_hf.py --username YOUR_HF_NAME --dry-run
    .venv/bin/python scripts/publish_to_hf.py --username YOUR_HF_NAME              # weights + card
    .venv/bin/python scripts/publish_to_hf.py --username YOUR_HF_NAME --card-only  # card text only
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILES = ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "special_tokens_map.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--model-dir", default=str(ROOT / "artifacts/sagemaker/choicefit-l4-full-b8-001"))
    parser.add_argument("--model-name", default="choicefit-roberta")
    parser.add_argument("--card", default=str(ROOT / "docs/model_card.md"))
    parser.add_argument("--card-only", action="store_true", help="Upload only the model card (README.md)")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded and stop")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not args.card_only:
        missing = [name for name in MODEL_FILES if not (model_dir / name).exists()]
        if missing:
            raise SystemExit(f"Missing model files in {model_dir}: {missing}")
    model_repo = f"{args.username}/{args.model_name}"
    files = [] if args.card_only else MODEL_FILES
    print(f"Model repo : {model_repo}\n  files: {files} + README.md (from {args.card})")
    if args.dry_run:
        return

    api = HfApi()
    print("Logged in as:", api.whoami()["name"])
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    for name in files:
        api.upload_file(path_or_fileobj=str(model_dir / name), path_in_repo=name, repo_id=model_repo)
    api.upload_file(path_or_fileobj=args.card, path_in_repo="README.md", repo_id=model_repo,
                    commit_message="Update model card")
    print(f"Done.\nModel: https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
