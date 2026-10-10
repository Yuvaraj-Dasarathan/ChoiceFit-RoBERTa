# Low-cost SageMaker training

Use **Amazon SageMaker AI → Training jobs**, one **ml.g4dn.xlarge**, in **us-east-1 (N. Virginia)**. Submit from this Mac; a SageMaker notebook, Studio GPU session, endpoint, and hyperparameter tuning job are not needed.

This is a prepared training integration, not an AWS-verified deployment. No paid job has been launched. Local MPS and CPU tests cannot prove that account permissions, quotas, AWS container dependencies, CUDA, or S3 configuration work. Run the short cloud pilot below before a longer run. No code can guarantee a zero-failure cloud job.

## Where the code and data run

Run `python -m cloud.launch ...` in the VS Code terminal on your Mac, from this repository. With `--launch`, it uploads the code, original RoBERTa base model, and dataset to your S3 bucket, then asks SageMaker to start a managed training job. Manual dataset uploads are unnecessary with this launcher; it uses its own per-job S3 paths.

SageMaker provisions the GPU machine, starts the AWS-provided PyTorch container, installs `cloud/requirements.txt`, and executes `cloud/entrypoint.py`, which calls `src/train.py`. The job saves model output to S3 and releases the training compute when it ends. You monitor it in the SageMaker AI console under Training jobs. You do not build a Docker image or configure Lambda for this workflow. S3 stores files; the SageMaker training instance executes the code.

The initial pilot uses `--data-dir data/typed-smoke`:

| Local file | Records | Purpose |
| --- | ---: | --- |
| `data/typed-smoke/train.jsonl` | 32 | Training |
| `data/typed-smoke/validation.jsonl` | 16 | Validation during training |
| `data/typed-smoke/test.jsonl` | 16 | Evaluation after reloading the saved model |

This pilot verifies AWS compatibility, not model quality. It starts from the original base-model snapshot in `.cache/huggingface`, not from a local fine-tuned artifact.

## Retained local files

- `artifacts/typed-local-recovery-v1/` (~480 MB): latest trained weights, tokenizer/configuration, training and test metrics, and test predictions. Keep this for local inference and inspecting errors. The weights and tokenizer/configuration are needed together to load the model; the metrics and predictions document its results.
- `checkpoints/typed-local-recovery-v1/` (~3.3 GB): full training recovery state, including optimizer and random states. These are separate from inference artifacts and are not uploaded by the cloud launcher. They have been retained for recovery inspection; they are optional once the completed run no longer needs recovery.
- `.cache/huggingface/`: downloaded original base weights used to initialize new training. The launcher needs the selected snapshot here.

Removed obsolete artifact folders: `choicefit-roberta` (old demo), `typed-smoke-mps` (old baseline diagnostic), `typed-mps-verified` (superseded smoke model), and `typed-local-recovered-check` (verified identical duplicate). The latest model and both dataset directories remain.

## Actual local result

The completed M1 run trained on 1,000 examples for one epoch (250 optimizer updates), with 100 validation and 100 held-out test examples. Training took 343 seconds; training plus final validation took 350 seconds, excluding startup and initial validation. The saved model is `artifacts/typed-local-recovery-v1`.

| Measurement | Result |
| --- | ---: |
| Validation accuracy before training | 33% |
| Validation accuracy after training | 31% |
| Validation loss before → after | 1.02769 → 1.02779 |
| Reloaded saved-model test accuracy | 38% |
| Expected accuracy from uniform random choices, validation / test | 37.75% / 37.08% |

These are pipeline checks, not satisfactory model quality: this small run shows no validation improvement and test accuracy is close to random-choice expectation. The random baseline accounts for each example's two, three, or four options. It is an expectation, not a separately sampled model. With only 100 evaluation examples, small differences are inconclusive. Before expensive scaling, inspect errors and run controlled local experiments that demonstrate learning on a small, coherent subset. A short AWS pilot remains useful solely to verify infrastructure.

Training history, held-out test metrics, and per-example predictions are saved beside the model as `training_metrics.json`, `test_metrics.json`, and `test-predictions.jsonl`.

The inference CLI also passed on MPS. Reloading the real recovery checkpoint into a fresh output directory reproduced the original exported model's SHA-256 hash. The automated suite passed all 29 tests, including a simulated mid-epoch interruption; these checks still do not verify AWS execution.

## Cost choice

AWS's public SageMaker price files checked during this implementation show these On-Demand training rates (USD/hour):

| Instance | N. Virginia | Oregon | Ohio | Mumbai |
| --- | ---: | ---: | ---: | ---: |
| ml.g4dn.xlarge | 0.736 | 0.736 | 0.736 | 0.811 |
| ml.g6.xlarge | 1.127 | 1.127 | 1.127 | 1.353 |
| ml.g5.xlarge | 1.408 | 1.408 | 1.2575 | 1.690 |

Choose g4dn in N. Virginia as a low hourly-cost GPU option among these comparisons. This does not establish the cheapest total training cost across all instances/regions; faster GPUs can finish sooner. Spot capacity and effective savings vary, so a worldwide cheapest Spot region cannot be guaranteed.

Source: [AWS price list for us-east-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.json), with matching regional files for `us-west-2`, `us-east-2`, and `ap-south-1`. See [SageMaker pricing](https://aws.amazon.com/sagemaker/ai/pricing/).

The pilot defaults to a 900-second (15-minute) runtime limit. Fifteen minutes at the quoted On-Demand compute rate is **$0.184**. This is a planning estimate, not a total-bill cap: S3, storage, logs, artifact transfer, and AWS billing details are additional. Runtime limits are not dollar-denominated budgets. A two-hour On-Demand compute allowance would be $1.472 at that rate.

Recommended progression:

1. Finish local code/recovery checks.
2. Run a 32-example **On-Demand pilot**, capped at 15 minutes. The purpose is verifying this exact AWS image and permissions without Spot interruptions.
3. If the job completes and exported model reload/test metrics exist, use **Managed Spot** for a larger experiment. Spot is optional even for the pilot: add `--spot` if minimum cost matters more than waiting or interruptions.
4. Keep the first larger job at 1,000 examples and one epoch. Measure throughput and quality before increasing to 5,000 or more.

## What prevents wasted runs

- `cloud.launch` is a dry run unless `--launch` is explicitly supplied. It checks all local rows, train/validation/test overlap, group overlap, choice count, token length, and the AWS request schema before uploading anything.
- A small allowlisted code archive is uploaded, rather than the workspace, virtual environment, caches, or unrelated models.
- The already-downloaded base model is uploaded as an S3 input channel. Training does not download weights or datasets from Hugging Face.
- The AWS container and text-model dependencies are pinned. The image is AWS's published `pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker`; this image has not been executed in this account yet. See [AWS available images](https://aws.github.io/deep-learning-containers/reference/available_images/).
- The entrypoint fails immediately if CUDA is unavailable; it does not silently run slow CPU training.
- Recovery checkpoints include model, AdamW optimizer, random states, epoch, completed update position, partial loss statistics, history, and the best trained weights. Saves happen every 50 optimizer updates and after epochs. Two checkpoints are retained; a corrupt latest checkpoint falls back to the previous readable one.
- Recovery rejects changed data hashes or incompatible training settings. Same-data ordering is reconstructed deterministically, then completed updates are skipped. Very large streams still need rereading to reach the recovery position.
- The best model is restored after a restarted job, even when its previous `/opt/ml/model` folder was lost. A final saved-model reload and test evaluation must succeed before the script exits successfully.
- Recovery tests simulate interruption and verify that restored AdamW/dropout training matches an uninterrupted CPU run exactly. This does not simulate every S3 sync or GPU failure.

SageMaker syncs `/opt/ml/checkpoints` to the per-job S3 checkpoint prefix and restores it for Managed Spot restarts. Only a successfully synced checkpoint can be recovered. Work since the last usable checkpoint can be lost. The launcher uses a fresh prefix for each job; manually starting a new job does not automatically reuse another job's checkpoints. [AWS checkpoint documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html), [Managed Spot](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html).

## 1. Connect this terminal to AWS

The bucket `yuvaraj-choicefit-training` and role `SageMakerChoiceFit` already exist, so do not recreate them. Keep the console in **US East (N. Virginia)** and wait for the `ml.g4dn.xlarge for training job usage` quota to become at least 1.

Sign in with the browser-based temporary-credential profile:

```bash
aws login --profile choicefit-login --region us-east-1
```

Configure the profile used by Python/boto3. The region is included in both places because the credential subprocess also needs it:

```bash
aws configure set credential_process 'aws configure export-credentials --profile choicefit-login --region us-east-1 --format process' --profile choicefit
aws configure set region us-east-1 --profile choicefit
export AWS_PROFILE=choicefit
aws sts get-caller-identity --region us-east-1
```

The returned `Account` must be `612242785639`. Repeat `export AWS_PROFILE=choicefit` in each new Bash terminal. If the session expires, repeat only the `aws login` command. The login identity needs S3 access, SageMaker create/describe/stop permissions, and `iam:PassRole` for `arn:aws:iam::612242785639:role/SageMakerChoiceFit`. The execution role's policies remain in `cloud/` for comparison with the console configuration.

## 2. Install the launch dependency

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -e '.[aws,test]'
python -m pytest -q
```

Do not install `requirements-lock.txt` into the AWS container: it records the local Mac environment. `cloud/requirements.txt` is the separate training dependency list; CUDA-enabled Torch comes from the AWS image. The default public-network training configuration needs package-index access during container startup. A private VPC without NAT/endpoints needs separate networking work; do not add it to the pilot blindly.

## 3. Review the job without spending on training

The base-model snapshot already exists in this workspace at the path below. It is the original RoBERTa checkpoint, not the 32-example fine-tuned model.

```bash
python3 -m cloud.launch \
  --bucket yuvaraj-choicefit-training \
  --role arn:aws:iam::612242785639:role/SageMakerChoiceFit \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-smoke \
  --job-name choicefit-pilot-001
```

This validates locally and prints the complete request. It does not contact AWS, upload, or create resources. In a fresh checkout, download the pinned base snapshot locally first, using `huggingface_hub.snapshot_download`; include config, tokenizer files, and `model.safetensors`.

## 4. Launch the short cloud pilot

After the dry run succeeds, run the same command with **`--launch`**:

```bash
python3 -m cloud.launch \
  --bucket yuvaraj-choicefit-training \
  --role arn:aws:iam::612242785639:role/SageMakerChoiceFit \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-smoke \
  --job-name choicefit-pilot-001 \
  --launch
```

This command starts a paid job and returns after submission. It uploads roughly 500 MB of model files plus small data/code files. The job runs on AWS even after this terminal is closed.

Monitor **SageMaker AI → Training → Training jobs → choicefit-pilot-001**, then open its CloudWatch logs. Look for device `cuda`, epoch metrics, and `reloaded_checkpoint_test`. A successful job produces `model.tar.gz` in the S3 output location, containing model weights, tokenizer, training metrics, and test metrics. Do not treat `InProgress` or a successful submission as training success.

Check or stop explicitly:

```bash
aws sagemaker describe-training-job --region us-east-1 --training-job-name choicefit-pilot-001
aws sagemaker stop-training-job --region us-east-1 --training-job-name choicefit-pilot-001
```

If this pilot fails, inspect `FailureReason` and logs before retrying. Do not increase runtime to hide dependency, permission, CUDA, or data errors. If only startup/download exceeds the pilot cap, assess that timing before choosing a modest larger limit.

## 5. Run the 1,000-example Spot experiment

Only after the pilot and model reload succeed:

```bash
python -m cloud.launch \
  --bucket yuvaraj-choicefit-training \
  --role arn:aws:iam::612242785639:role/SageMakerChoiceFit \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-local \
  --job-name choicefit-1000-spot-001 \
  --spot --max-run 3600 --max-wait 7200 \
  --launch
```

One hour of running time and two hours including waiting are the maximums configured here. They are not predictions of runtime. The job uses one epoch, one message per microbatch, up to four candidate pairs, four-step gradient accumulation, and a 192-token limit. To increase these, change and revalidate the configuration deliberately.

Do not automatically promote to millions of records based on one accuracy number. The local sample covers many tasks with few examples each; inspect validation loss, per-task errors, held-out source families, and description-sensitive cases. The current implementation still has a 512-token architectural limit and scores each candidate independently.

## 6. Test the result locally without a paid endpoint

Use the `ModelArtifacts.S3ModelArtifacts` URI returned by `describe-training-job` to download your own `model.tar.gz`. Extract into a new directory, then run:

```bash
python -m src.infer \
  --model artifacts/sagemaker-result \
  --input-json data/typed-local/example.json \
  --pair-batch-size 2 --device mps
```

No endpoint is needed to test the model on the Mac. Completed training stops its training compute; S3 artifacts/checkpoints and CloudWatch logs remain and can continue to incur storage charges. Remove obsolete job prefixes when their artifacts are no longer needed. Keep at least the final model and experiment metrics.
