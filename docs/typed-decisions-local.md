# Tasksource typed decisions: local M1 validation

The current training source is [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions). We use `kind == "choice"` records. `score` and `noul` are not implemented by this choice scorer.

The converter pins revision `d2ab1d12be4463fb7ac1f877ed2193b3bb59642a`. At this inspected revision the `default` configuration has 999,858 training records; `full` has 2,500,000. Both have 15,000 validation and 15,000 test records across decision kinds. A streaming scan is used instead of downloading and loading the whole collection.

## Already prepared and verified

This Mac is an M1 MacBook Air with 16 GiB unified memory. PyTorch's MPS device worked outside the restricted execution sandbox.

| Stage | Train | Validation | Test | Status |
| --- | ---: | ---: | ---: | --- |
| Code smoke test | 32 | 16 | 16 available; 100 evaluated | Completed on MPS |
| First learning run | 1,000 | 100 | 100 | Completed; 31% validation, 38% test accuracy |
| Next local experiment | 5,000 | 500 | 500 | Ready to prepare and run |

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

## Watch convergence locally

Each experiment makes 250 optimizer updates per epoch (1,000 messages / four accumulated messages), records every message, and evaluates the fixed validation split after every 25 updates. Use only the three complete commands below; the earlier one-epoch command was removed because it did not create convergence logs or set an explicit learning rate.

`convergence.jsonl` is deliberately compact:

- `example`: the choice IDs, gold target distribution, scalar scores, softmax probabilities, predicted choice, and loss **before** its upcoming optimizer update. It deliberately excludes source, group, and record metadata.
- `update`: the average loss for the accumulated four messages, running train loss, learning rate, gradient norm before clipping, and L2 norms for the four classification-head parameters **after** the update.
- `validation`: validation loss, accuracy, Brier score, and target-probability metrics at that update. This is the line used to choose the best checkpoint.

It does not contain the 125 million individual RoBERTa weights or 768-value token vectors. Those would make the file many gigabytes and would not show a usable learning trend. The classifier-head norms and validation values show the relevant changes.

On this M1 Air, expect roughly 6 minutes for one 1,000-row epoch without interval validation, based on the measured run above. With validation every 25 updates, allow about 7 minutes per epoch and about 35 minutes for the five-epoch command; thermal throttling can make it longer. The JSONL trace itself adds little work.

Do not choose an epoch count in advance for a full run. In this experiment, select the **update with the lowest validation loss**, not necessarily the final epoch. If it keeps falling at update 1,250, run a longer fresh experiment. If it flattens or rises for several validation checks, stop; more epochs are unlikely to help that setting. Keep the test split untouched while making this decision.

For a practical learning sequence, first make a 16–32 hard-label example memorization run and verify that its training loss falls close to zero. Then run the five-epoch 1,000-row experiment at `5e-6`, `1e-5`, and `2e-5` with the same seed and splits. Compare each run's lowest validation loss and the update where it occurred. A tiny set spanning hundreds of sources can fail to improve validation even with correct code, so this teaches the mechanics and rejects unstable learning rates; it does not predict the final large-dataset accuracy.

Run each command once. Every run has its own output folder, so the trainer never overwrites an earlier model or trace.

```bash
# Learning-rate run 1: 5e-6
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 python3 -m src.train \
  --train data/typed-local/train.jsonl \
  --validation data/typed-local/validation.jsonl \
  --output artifacts/convergence-1k-lr5e-6 \
  --epochs 5 \
  --batch-size 1 --max-pairs 4 --gradient-accumulation-steps 4 \
  --eval-pair-batch-size 2 --gradient-checkpointing --max-length 192 --device mps \
  --learning-rate 5e-6 --log-every 25 --eval-every 25 \
  --convergence-jsonl artifacts/convergence-1k-lr5e-6/convergence.jsonl
```

```bash
# Learning-rate run 2: 1e-5
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 python3 -m src.train \
  --train data/typed-local/train.jsonl \
  --validation data/typed-local/validation.jsonl \
  --output artifacts/convergence-1k-lr1e-5-v2 \
  --epochs 5 \
  --batch-size 1 --max-pairs 4 --gradient-accumulation-steps 4 \
  --eval-pair-batch-size 2 --gradient-checkpointing --max-length 192 --device mps \
  --learning-rate 1e-5 --log-every 25 --eval-every 25 \
  --convergence-jsonl artifacts/convergence-1k-lr1e-5-v2/convergence.jsonl
```

```bash
# Learning-rate run 3: 2e-5
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 python3 -m src.train \
  --train data/typed-local/train.jsonl \
  --validation data/typed-local/validation.jsonl \
  --output artifacts/convergence-1k-lr2e-5 \
  --epochs 5 \
  --batch-size 1 --max-pairs 4 --gradient-accumulation-steps 4 \
  --eval-pair-batch-size 2 --gradient-checkpointing --max-length 192 --device mps \
  --learning-rate 2e-5 --log-every 25 --eval-every 25 \
  --convergence-jsonl artifacts/convergence-1k-lr2e-5/convergence.jsonl
```

After two or more runs complete, generate a browser-ready graph that compares both losses. The first graph is training loss after every weight update; the second is validation loss after every validation check. Both use the actual JSONL values:

```bash
python3 -m src.plot_convergence \
  --run lr-5e-6=artifacts/convergence-1k-lr5e-6/convergence.jsonl \
  --run lr-1e-5=artifacts/convergence-1k-lr1e-5-v2/convergence.jsonl \
  --run lr-2e-5=artifacts/convergence-1k-lr2e-5/convergence.jsonl \
  --output artifacts/convergence-comparison.html --overwrite
```

Open `artifacts/convergence-comparison.html` in a browser. Use the same data splits, seed, maximum epochs, batch size, and evaluation interval in every learning-rate run. First select a learning rate; only then experiment with the effective batch size.

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
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m src.prepare_typed_decisions \
  --output data/typed-local-new \
  --train-limit 1000 --validation-limit 100 --test-limit 100 \
  --scan-limit 5000 --max-choices 4 --max-length 192
```

Prepare the 5,000-row local experiment. This creates a separate folder and leaves the completed 1,000-row experiment unchanged:

```bash
HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -m src.prepare_typed_decisions \
  --output data/typed-local-5k \
  --train-limit 5000 --validation-limit 500 --test-limit 500 \
  --scan-limit 50000 --max-choices 4 --max-length 192
```

Run the 5,000-row training experiment. `1e-5` is now the trainer default and is written explicitly here so the run is easy to identify. Validation runs every 250 weight updates. The run stops after four consecutive validation checks that do not improve validation loss by at least `0.0001`; its maximum remains five epochs.

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv/bin/python -m src.train \
  --train data/typed-local-5k/train.jsonl \
  --validation data/typed-local-5k/validation.jsonl \
  --output artifacts/convergence-5k-lr1e-5 \
  --epochs 5 \
  --batch-size 1 --max-pairs 4 --gradient-accumulation-steps 4 \
  --eval-pair-batch-size 2 --gradient-checkpointing --max-length 192 --device mps \
  --learning-rate 1e-5 --log-every 125 --eval-every 250 \
  --early-stopping-patience 4 --early-stopping-min-delta 0.0001 \
  --convergence-jsonl artifacts/convergence-5k-lr1e-5/convergence.jsonl
```

When it finishes, inspect the training and validation curves from this run only:

```bash
.venv/bin/python -m src.plot_convergence \
  --run 5k-lr1e-5=artifacts/convergence-5k-lr1e-5/convergence.jsonl \
  --output artifacts/convergence-5k-lr1e-5/convergence.html
```

Then evaluate the saved best-validation checkpoint on the untouched test split:

```bash
.venv/bin/python -m src.evaluate \
  --model artifacts/convergence-5k-lr1e-5 \
  --data data/typed-local-5k/test.jsonl \
  --batch-size 1 --pair-batch-size 2 --device mps \
  --predictions artifacts/convergence-5k-lr1e-5/test-predictions.jsonl
```

The generated model folder holds the best observed trained weights, rather than necessarily the weights from the last update. `training_metrics.json` records the epoch summaries, and `convergence.jsonl` records every update plus each validation check. Do not use test loss or test accuracy to decide when to stop.

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
