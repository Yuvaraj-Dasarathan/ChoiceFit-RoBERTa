# Full typed-decisions SageMaker run

This guide uses the existing `yuvaraj-choicefit-training` bucket, `SageMakerChoiceFit` role, and `us-east-1`. Run commands from this repository on the Mac. The launcher uploads data, the base model, and code to S3, then starts the managed training job. No Docker image, Lambda, notebook, or endpoint is needed for training.

## 1. Prepare full training, validation, and test files

The pinned Hugging Face dataset has 999,858 train rows, 15,000 validation rows, and 15,000 test rows before filtering. This command streams all splits and does not retain a million rows in memory. It writes every eligible row, including the full eligible validation and test splits.

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m src.prepare_full_typed_decisions \
  --output data/typed-full \
  --train-limit 1000000 --validation-limit 0 --test-limit 0 \
  --max-choices 4 --max-length 192
```

`0` means all eligible rows. The source train split is slightly below one million, and unsupported rows are filtered out because this model accepts at most four choices and 192-token input pairs. Read the actual final counts from:

```bash
.venv/bin/python -m json.tool data/typed-full/preparation.json
```

Keep all files in `data/typed-full/`; the launcher uploads these exact files.

## 2. Full-run schedule

One optimizer update normally represents four rows: batch size 1 and four gradient-accumulation steps. The launcher uses:

| Setting | Value | Meaning |
| --- | ---: | --- |
| Learning rate | `1e-5` | The selected local rate. |
| Validation interval | 10,000 updates | About 40,000 training rows; each pass reads the complete validation file. |
| Minimum validations | 3 | Early stopping cannot occur before three checks. |
| Patience | 5 | Stop after five later non-improving checks. |
| Minimum improvement | `0.0001` | Smaller changes do not reset patience. |
| Recovery checkpoint | 5,000 updates | Required for Managed Spot restart recovery. |
| Maximum epochs | 3 | Ceiling, not a promise to run all three. |

CloudWatch prints one progress row every 10,000 updates, each validation result, any `early_stop`, and the final test result. The detailed trace is `convergence.jsonl` in the model output.

## 3. Dry run — no AWS action or cost

```bash
export AWS_PROFILE=choicefit
export AWS_SDK_UA_APP_ID=AWSSkill-SageMaker

.venv/bin/python -m cloud.launch \
  --bucket yuvaraj-choicefit-training \
  --role arn:aws:iam::612242785639:role/SageMakerChoiceFit \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-full \
  --job-name choicefit-full-001 \
  --spot --max-run 86400 --max-wait 172800
```

This checks all rows, split/group/message separation, token limits, and request shape. It does not upload or create anything. The base-model path must contain `model.safetensors` and tokenizer files.

## 4. Start the paid job

After the dry run succeeds, add `--launch`:

```bash
.venv/bin/python -m cloud.launch \
  --bucket yuvaraj-choicefit-training \
  --role arn:aws:iam::612242785639:role/SageMakerChoiceFit \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-full \
  --job-name choicefit-full-001 \
  --spot --max-run 86400 --max-wait 172800 \
  --launch
```

Open **SageMaker AI → Training jobs → choicefit-full-001** and CloudWatch. `Downloading` copies S3 inputs, `Training` runs Python, and `Uploading` sends final artifacts to S3. The final archive is under:

```text
s3://yuvaraj-choicefit-training/choicefit/choicefit-full-001/output/model.tar.gz
```

It contains the best validation model, tokenizer, `training_metrics.json`, and `test_metrics.json`. Test metrics are calculated after selection and never control early stopping.

## 5. Validate the completed model

Download the job's **Model artifacts** archive from the SageMaker console and extract it into `artifacts/choicefit-full-001/`. Then export final test predictions:

```bash
.venv/bin/python -m src.evaluate \
  --model artifacts/choicefit-full-001 \
  --data data/typed-full/test.jsonl \
  --batch-size 1 --pair-batch-size 2 --device mps \
  --predictions artifacts/choicefit-full-001/test-predictions.jsonl
```

Choose training settings using validation loss only. Read test metrics once afterward to report final quality.

## 6. Later: free demo UI

After a final model is selected, a Hugging Face Gradio Space is the simplest free demo host. Create a Space, upload the extracted model and a small `app.py` wrapper around `ChoiceModel`, and test it with `data/typed-full/example.json`. It is appropriate for a learning demo; verify free storage/runtime limits before relying on it publicly.

## 7. Later: compare with Jev

Use the same frozen test rows for both systems. Do not train from test results.

1. Keep the local model's `test-predictions.jsonl` from the command above.
2. Obtain a Jev key and confirm its current request contract from the official docs. Keep it only in `JEV_API_KEY`, never in Git, a UI, logs, or output JSONL.
3. Build a rate-limited benchmark client that sends each row's message as state, question as Choice instructions, and choices as criteria. Save Jev choice/probabilities in a separate JSONL keyed by row `id`.
4. Join both prediction files with targets and report hard-label accuracy, cross-entropy, Brier score, failures, model/API versions, date, row count, and Jev cost. Break down errors by source and soft versus one-hot targets.

Jev Choice returns a selected option and probabilities, so it can be compared against this model's output. Read the official [TypeSafe introduction](https://docs.typesafe.ai/introduction) before implementing the benchmark client.
