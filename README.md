# ChoiceFit-RoBERTa

[GitHub repository](https://github.com/yuvarajd2588/ChoiceFit-RoBERTa)

RoBERTa fine-tuning with dynamic described choices.

Fine-tunes **FacebookAI/roberta-base** on `(question + message, candidate description)` pairs. A shared scalar scoring head produces one logit per candidate; cross entropy trains the correct candidate against the other supplied choices. The entire encoder is fine-tuned. At inference, softmax normalizes the candidate scores and the winning label is returned.

Inspired by [Jev's official Choice documentation](https://docs.typesafe.ai/primitives/choice): message/state, question/instructions, described choices/criteria, selected choice and probabilities. This is an independent implementation, not Jev's architecture, weights, API client, or training recipe. Only string descriptions and one choice question per prediction are supported. Unlike Jev, label IDs are not encoded: changing an ID leaves its meaning unchanged. `top_probability` is an uncalibrated softmax value, not Jev's separate confidence measure. See [Jev confidence](https://docs.typesafe.ai/confidence) and [Hugging Face RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta).

## Setup and training

Python 3.10+; this workspace already contains `.venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
HF_HOME="$PWD/.cache/huggingface" python -m choicefit_roberta.train
```

The initial run downloads RoBERTa-base (roughly 500 MB). CUDA, Apple MPS, then CPU are selected automatically; use `--device cpu` to override. Defaults: 8 epochs, learning rate 2e-5, 4 messages per batch, 192 tokens per pair. A batch expands to the total number of candidate pairs. Inputs exceeding the token budget are truncated using longest-first truncation; long instructions, descriptions, and messages may lose relevant text. Maximum supported length is 512.

The demo includes 40 manually authored training examples and 8 separate validation examples across billing, HR, technical support, and shipping, with 2–4 choices per record. The user example is not in training. The checkpoint with lowest validation loss is saved to `artifacts/choicefit-roberta/`, together with tokenizer, config, and `training_metrics.json`. Validation selects the checkpoint, so it is not an independent final test set.

## Try inference

```bash
source .venv/bin/activate
python -m choicefit_roberta.infer --message "My card was charged twice."
```

The default choices file contains the two options in the original request. Pass `--choices path/to/choices.json` for another mapping and `--question "Which team should handle this message?"` for custom instructions.

```python
from choicefit_roberta import ChoiceModel

model = ChoiceModel("artifacts/choicefit-roberta")
result = model.predict(
    message="My card was charged twice.",
    choices={
        "billing": "Payments, charges and billing discrepancies",
        "HR": "Employee benefits and workplace questions",
    },
    question="Which team should handle this message?",
)
print(result)
```

The result contains `type`, `choice`, `probabilities` (one per supplied label), and `top_probability`. Model loading happens once; reuse the instance. Predictions always select one supplied option. Unknown inputs are not automatically rejected; an explicit other option needs appropriate training examples. Because candidates are scored independently, relative “none of these” semantics are a limitation.

## Your larger dataset

Use JSONL, one complete decision per line:

```json
{"message":"My invoice is incorrect.","question":"Which team should handle this?","choices":{"billing":"Payments and billing problems","HR":"Employee benefits"},"label":"billing"}
```

```bash
python -m choicefit_roberta.train --train data/train.jsonl --validation data/validation.jsonl --epochs 8 --batch-size 4 --output artifacts/my-model
python -m pytest -q
```

Each target must match a choice key. Provide realistic incorrect alternatives, varied descriptions, choice counts, and balanced targets. Split by source/customer/template to avoid leakage; the script rejects exact message overlap. Keep an untouched test set, including new descriptions, unseen categories, ambiguous messages, and changed questions for the same input. This tiny routing dataset does **not** teach general instruction following or demonstrate reliable zero-shot classification. Learning arbitrary question semantics requires diverse question-conditioned training examples. Evaluate accuracy and calibration separately before using probabilities for automation.

## Later: SageMaker

No AWS resources are created. The training CLI already respects `SM_CHANNEL_TRAIN`, `SM_CHANNEL_VALIDATION`, and `SM_MODEL_DIR`, expecting `train.jsonl` and `validation.jsonl` in their channels. A future single-instance SageMaker container can install this package and run `python -m choicefit_roberta.train`; saved Hugging Face artifacts go to the model directory. This is not yet a SageMaker deployment or distributed training implementation. Add streaming/sharding, checkpoint resume, and an inference handler when the dataset and hosting requirements are known.

## Verified demo run

The local RoBERTa-base run completed 8 epochs: validation accuracy improved from 1/8 to 6/8, and validation cross entropy from 1.2434 to 0.6179. Eight automated tests passed. Reloaded checkpoint inference selected `billing` for “My card was charged twice.” (softmax 0.9889), and `HR` for a dental-benefits enrollment question (0.9905). Reordering choices and renaming their IDs preserved probabilities within 1e-5. Run `python evaluate_checkpoint.py` to repeat those checks. These two examples and eight validation records are a smoke check, not a reliable performance estimate. Installed dependency versions are recorded in `requirements-lock.txt` (captured on macOS ARM).
