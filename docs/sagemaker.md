# L4 SageMaker training

Use the existing `yuvaraj-choicefit-training` bucket and `SageMakerChoiceFit` role in `us-east-1`. These commands use On-Demand capacity, not Spot.

## 1. Prepare the 8-choice, 256-token L4 pilot data

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m src.prepare_full_typed_decisions \
  --output data/typed-l4-pilot \
  --train-limit 20000 --validation-limit 1000 --test-limit 1000 \
  --max-choices 8 --max-length 256
```

## 2. Dry run the L4 pilot

```bash
export AWS_PROFILE=choicefit
export AWS_SDK_UA_APP_ID=AWSSkill-SageMaker

.venv/bin/python -m cloud.launch \
  --instance-type ml.g6.2xlarge \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-l4-pilot \
  --job-name choicefit-l4-pilot-001 \
  --max-run 7200
```

## 3. Launch the L4 pilot

Run the same command with `--launch`:

```bash
.venv/bin/python -m cloud.launch \
  --instance-type ml.g6.2xlarge \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-l4-pilot \
  --job-name choicefit-l4-pilot-001 \
  --max-run 7200 --launch
```

Check CloudWatch. Confirm it reports `cuda`, `precision: bf16`, no out-of-memory error, and the expected progress speed. The pilot uses batch 4, 32 candidate pairs, accumulation 16, length 256, learning rate `2e-5`, and 10% linear warmup.

## 4. Prepare the full L4 data

Only after the pilot finishes successfully:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m src.prepare_full_typed_decisions \
  --output data/typed-full-8x256 \
  --train-limit 1000000 --validation-limit 0 --test-limit 0 \
  --max-choices 8 --max-length 256
```

Read the actual retained counts:

```bash
.venv/bin/python -m json.tool data/typed-full-8x256/preparation.json
```

## 5. Dry run the full job

```bash
.venv/bin/python -m cloud.launch \
  --instance-type ml.g6.2xlarge \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-full-8x256 \
  --job-name choicefit-l4-full-001 \
  --max-run 86400
```

## 6. Launch the full job

```bash
.venv/bin/python -m cloud.launch \
  --instance-type ml.g6.2xlarge \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-full-8x256 \
  --job-name choicefit-l4-full-001 \
  --max-run 86400 --launch
```

The job uses BF16, linear warmup/decay, and saves recovery checkpoints. Keep the test split only for final reporting; validation loss controls the saved model and early stopping.
