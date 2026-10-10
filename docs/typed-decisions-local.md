# Tasksource typed decisions: local M1 validation

The current training source is [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions). We use `kind == "choice"` records. `score` and `noul` are not implemented by this choice scorer.

The converter pins revision `d2ab1d12be4463fb7ac1f877ed2193b3bb59642a`. At this inspected revision the `default` configuration has 999,858 training records; `full` has 2,500,000. Both have 15,000 validation and 15,000 test records across decision kinds. A streaming scan is used instead of downloading and loading the whole collection.

## Already prepared and verified

This Mac is an M1 MacBook Air with 16 GiB unified memory. PyTorch's MPS device worked outside the restricted execution sandbox.

| Stage | Train | Validation | Test | Status |
| --- | ---: | ---: | ---: | --- |
| Code smoke test | 32 | 16 | 16 available; 100 evaluated | Completed on MPS |
| First learning run | 1,000 | 100 | 100 | Completed; 31% validation, 38% test accuracy |
| Later local experiment | 5,000 | 500 | 500 | Defer until smaller experiments show learning |

Prepared samples: `data/typed-local/`. The training sample spans 317 sources and includes 12 soft-target examples. Validation spans 69 sources and test spans 67. These particular validation/test samples have only one-hot targets. This means they do not adequately evaluate soft-target generalization; use an additional soft-target evaluation subset for that later.

The small smoke sample is in `data/typed-smoke/`. It deliberately includes the soft examples to exercise probability-target training. Its older trained models were removed after verification; the input files remain for the AWS pilot. The retained model is `artifacts/typed-local-recovery-v1/`, trained on 1,000 examples.

Observed verification:

- 29 automated tests pass, including soft-target validation, finite padded loss and gradients, accumulation, group overlap checks, export, and training/save/reload.
- The verified full RoBERTa-base smoke run took about 20 seconds wall time; its 32-message training portion took 12 seconds. The earlier run took 17 seconds for training. These are short-run measurements, not sustained throughput promises.
- The verified run reported a peak memory footprint of 4.36 GB (about 4.1 GiB), using batch size 1, up to four choices, 192 tokens, and gradient checkpointing.
- Checkpoint comparison confirmed changes in both the classification head and an encoder attention matrix.
- Reloaded MPS evaluation completed on 100 held-out examples: 39% accuracy, cross entropy 1.0455. After only 32 training examples this is a code check, not evidence of useful model quality.
- A separate CLI inference call completed using `--input-json`.

## Data representation

Converted JSONL keeps arbitrary choice IDs paired with the original option text:

```json
{"message":"Example state","question":"Which description fits?","choices":{"option_0":"Description A","option_1":"Description B"},"target":{"option_0":0.25,"option_1":0.75}}
```

`target` maps every choice ID to its probability. The encoder sees the descriptions, not `option_0` or `option_1`. Probabilities are not converted to a winning label. Legacy examples containing a single `label` still work and are converted to one-hot targets internally. Supply either `target` or `label`, not both.

The loss is cross entropy against the full target distribution. Padded nonexistent choices have zero target probability; the loss masks their log probabilities before multiplication to avoid `0 * -inf` NaNs. Inference continues to score all options and apply one softmax across the combined scores.

Evaluation reports:

- `accuracy`: accuracy on exactly one-hot target rows; null if none exist.
- `hard_examples` and `soft_examples`: counts for interpreting that accuracy.
- `top_target_accuracy`: whether the prediction is one of the highest-target-probability choices (ties accepted).
- `mean_selected_target_probability`: mean gold probability assigned to the predicted option.
- `loss`: cross entropy across every row.
- `brier_score`: mean sum of squared differences between predicted and target probabilities.

## Run the prepared 1,000-record experiment

From the project root:

```bash
source .venv/bin/activate
HF_HOME="$PWD/.cache/huggingface" python -m src.train \
  --train data/typed-local/train.jsonl \
  --validation data/typed-local/validation.jsonl \
  --output artifacts/typed-local-v1 \
  --epochs 1 \
  --batch-size 1 \
  --max-pairs 4 \
  --gradient-accumulation-steps 4 \
  --eval-pair-batch-size 2 \
  --gradient-checkpointing \
  --max-length 192 \
  --device mps \
  --log-every 25
```

Start with one epoch, inspect validation and predictions, then decide whether a fresh longer run is useful. Based only on the short smoke test, linear extrapolation gives roughly 6–9 minutes of training per 1,000 messages; allow more time for differing lengths, validation, model saving, and thermal slowdown. No 1,000-example run has been timed yet.

The output directory must be new or empty. The best **trained** epoch is saved; the untrained baseline is recorded in metrics but cannot replace a trained checkpoint. Existing demo artifacts are not overwritten.

Evaluate the saved checkpoint and export all 100 predictions:

```bash
python -m src.evaluate \
  --model artifacts/typed-local-v1 \
  --data data/typed-local/test.jsonl \
  --batch-size 1 \
  --pair-batch-size 2 \
  --device mps \
  --predictions artifacts/typed-local-v1/test-predictions.jsonl
```

The predictions file must not already exist. It includes input, question, descriptions, target, predicted ID, winning description, and probabilities so results can be inspected without another forward pass.

Run one prediction (replace the model with `artifacts/typed-local-recovery-v1` to try the retained 1,000-example model immediately):

```bash
python -m src.infer \
  --model artifacts/typed-local-v1 \
  --input-json data/typed-local/example.json \
  --pair-batch-size 2 \
  --device mps
```

For new inputs, use the existing `--message`, `--question`, and `--choices` arguments. Each dataset row supplies its own choices. With `--message`, an explicit `--choices` file is required.

## Reproduce or grow the sample

The existing workspace already has dependencies and data. For a fresh setup:

```bash
python -m pip install -e '.[data,test]'
HF_HOME="$PWD/.cache/huggingface" python -m src.prepare_typed_decisions \
  --output data/typed-local-new \
  --train-limit 1000 --validation-limit 100 --test-limit 100 \
  --scan-limit 5000 --max-choices 4 --max-length 192
```

To try a larger local sample later:

```bash
HF_HOME="$PWD/.cache/huggingface" python -m src.prepare_typed_decisions \
  --output data/typed-local-5k \
  --train-limit 5000 --validation-limit 500 --test-limit 500 \
  --scan-limit 50000 --max-choices 4 --max-length 192
```

Limits are upper bounds: whole eligible groups are selected, so selection can produce fewer records. Check `preparation.json` for actual counts. Increasing sample size mostly increases runtime, not peak training memory; memory is controlled primarily by candidate count, sequence length, optimizer state, and microbatch size. On an 8 GiB Mac, these 16 GiB measurements do not establish fit; start with the smoke test and close memory-heavy applications.

Sampling details:

- Scan a bounded number of upstream records per original split, completing the last contiguous group.
- Keep only valid choice records with 2–4 options whose complete formatted input pairs fit the token budget. Do not truncate or sample away choices from an accepted row.
- Exclude sources prefixed `multilingual/`; this is a source filter, not language detection.
- Reservoir-sample complete eligible groups per source, then interleave sources. Memory is bounded by scan/group reservoir limits rather than the full dataset. This changes the upstream mixture and is intended for development, not a published benchmark.
- Use only `direct` variants for validation/test. Keep original split assignments and remove cross-split selected-message/group overlap.
- Preserve IDs, group IDs, source, variants, original split, and license metadata. Original source terms continue to apply.
- Long-input filtering deliberately limits task coverage. The current model's maximum is 512 tokens per pair; even increasing to 512 will not fit every upstream example.

The official validation/test sources may also occur in training. These splits measure new examples, not necessarily generalization to unseen task families. Add source-family holdouts before claiming arbitrary-description generalization. Keep any original option references and their meanings intact when converting; IDs are only a returned handle.

## Before SageMaker

The package now installs as `choicefit_roberta`; both `python -m src.train` from the checkout and `python -m choicefit_roberta.train` after installation work. The trainer retains `SM_CHANNEL_TRAIN`, `SM_CHANNEL_VALIDATION`, and `SM_MODEL_DIR` defaults. No AWS resources have been created.

Recovery checkpoints and a SageMaker entrypoint/launcher have now been added; see [the SageMaker guide](sagemaker.md). Use `--checkpoint-dir` to enable full-state recovery. Loading `--model` alone still loads only model weights. The exact AWS image, CUDA execution, account permissions, and S3 restoration need a short cloud pilot before a costly run.
