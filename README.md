# ChoiceFit-RoBERTa

Fine-tune RoBERTa to choose among descriptions supplied with each request. The encoder sees the question, message, and candidate description; label IDs are only handles for returning the selected choice. Both RoBERTa and the shared scalar scoring head are trained.

The current dataset is [Tasksource typed decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions). We use `choice` records and preserve their target probability distributions. See [the local M1 training guide](docs/typed-decisions-local.md) for preparation, filtering, measurements, and limitations.

## Setup

Run from the repository root. Python 3.10+ is required; this workspace already has `.venv`.

```bash
source .venv/bin/activate
python -m pip install -e '.[data,test]'
```

## Current data

- `data/typed-local/`: 1,000 training, 100 validation, and 100 test examples, plus a sample input and preparation metadata.
- `data/typed-smoke/`: 32 training, 16 validation, and 16 test examples for pipeline verification.

These generated datasets are ignored by Git. To prepare a fresh sample, use `python -m src.prepare_typed_decisions --scan-limit 5000`. The output directory must be empty; preparation defaults to `data/typed-local/`.

## Train on the M1 Mac

```bash
HF_HOME="$PWD/.cache/huggingface" python -m src.train \
  --epochs 1 \
  --batch-size 1 \
  --max-pairs 4 \
  --gradient-accumulation-steps 4 \
  --eval-pair-batch-size 2 \
  --gradient-checkpointing \
  --device mps \
  --log-every 25
```

Training defaults to `data/typed-local/train.jsonl` and `data/typed-local/validation.jsonl`, with output in `artifacts/typed-local-v1/`. Choose a new `--output` for additional runs; existing checkpoints are protected. The full input pair is limited to 192 tokens by default (512 maximum). The local preparation step filters out pairs that exceed its configured limit.

## Evaluate and infer

After training:

```bash
python -m src.evaluate \
  --model artifacts/typed-local-v1 \
  --data data/typed-local/test.jsonl \
  --batch-size 1 --pair-batch-size 2 --device mps \
  --predictions artifacts/typed-local-v1/test-predictions.jsonl

python -m src.infer \
  --input-json data/typed-local/example.json \
  --pair-batch-size 2 --device mps
```

To try the already-verified smoke model, add `--model artifacts/typed-mps-verified`. For custom inputs, supply `--message`, `--question`, and an explicit `--choices` JSON mapping. Add `--debug` to inspect tensor shapes and intermediate scores.

Each training record contains `message`, `question`, `choices`, and a `target` mapping choice IDs to probabilities. A single `label` is also supported for hard-target examples. Inference returns the winning ID and probabilities over all supplied choices. These probabilities are not calibrated confidence, and the model always selects a supplied choice.

## Validation and deployment scope

Run `python -m pytest -q`. Local MPS training, checkpoint reload, and inference have been verified; details are in the guide. A small smoke run validates the pipeline, not generalization to arbitrary descriptions.

The trainer respects `SM_CHANNEL_TRAIN`, `SM_CHANNEL_VALIDATION`, and `SM_MODEL_DIR`. After installation, use `python -m choicefit_roberta.train` outside the checkout. SageMaker deployment, optimizer-state resume, and periodic recovery checkpoints are not yet implemented. No AWS resources have been created.
