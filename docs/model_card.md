---
license: other
license_name: research-and-educational-use-only
language: en
base_model: FacebookAI/roberta-base
datasets:
- tasksource/tasksource-jev-typed-decisions
metrics:
- accuracy
tags:
- multiple-choice
- roberta
- decision-making
---

# ChoiceFit RoBERTa

A **RoBERTa-base** model fine-tuned to pick the best option for a message from **2 to 8 candidate options**.
Trained on about **half a million (485,945)** multiple-choice examples.

- **Try it live, no install:** https://yuvaraj-dasarathan.github.io/ChoiceFit-RoBERTa/
- **Code, training recipe and full write-up:** https://github.com/Yuvaraj-Dasarathan/ChoiceFit-RoBERTa

> The live demo runs on the author's laptop, so it can be offline. If it is, use the code below to try the model yourself.

## Inspired by TypeSafe.ai JEV

This project is inspired by [TypeSafe.ai](https://typesafe.ai)'s JEV classification: instead of a fixed label set,
every request brings its own options and the model chooses among them. ChoiceFit is a small, open, learning-project
take on that idea. JEV (a commercial API) was used as a reference to measure how close a small fine-tuned model can get.

## How it works

Each (question + message, option description) pair is encoded by RoBERTa and receives **one score** from a shared
single-output head. A softmax across the options of one request turns the scores into probabilities. Option text is
what the model reads, so new options never need retraining.

## Training data

- Source: [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions),
  `choice` records only, with target probability distributions preserved.
- **485,945 training**, 6,651 validation and 6,597 test examples.
- Options per example: 2 (38%), 3 (26%), 4 (23%), 5-8 (12%).
- Topics (approximate): reasoning and logic 34%, science/math/medical 11%, commonsense and reading comprehension 8%,
  safety, hate speech and prompt-injection checks 8%, sentiment and emotion 7%, other mixed tasks 32%.
- General-purpose data, not built for any one industry.

## Training setup

| | |
| --- | --- |
| Hardware | AWS SageMaker `ml.g6.2xlarge`, 1x NVIDIA L4 (24 GB) |
| Training time | about 7 billed hours on an L4 (20+ GPU hours in total across all SageMaker runs), stopped early at about 2.5 of 3 epochs |
| Precision | bf16 |
| Learning rate | 2e-5, 10% warmup, linear decay |
| Max length | 256 tokens, up to 8 options |
| Batch | about 64 messages per optimizer update |
| Selection | lowest validation loss (0.764, validation accuracy 67.1%), checked every 625 updates |

## Results (hard accuracy: the top-scored option is the correct one)

**These are not random or AI-generated percentages.** Every number was measured on held-out data the model never trained on, the same rows were run through both systems, and the raw reports and per-row predictions are public so you can check them: https://github.com/Yuvaraj-Dasarathan/ChoiceFit-RoBERTa/tree/main/results

| Test set | ChoiceFit | JEV (TypeSafe.ai) |
| --- | ---: | ---: |
| Held-out mixed benchmark (6,597 rows) | 66.7% | 78.3% |
| banking77, 4 options (500 rows) | 67.0% | 96.2% |
| clinc_oos, 4 options (500 rows) | 82.8% | 99.4% |

JEV is stronger on every set. This is a learning project, not a JEV replacement.

## Use it

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "Yuvaraj-ml/choicefit-roberta"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

question = "Which category best describes what the customer wants?"
message = "I was charged twice for the same coffee this morning."
options = ["Report a duplicate charge", "Reset a forgotten password", "Open a new bank account"]

batch = tok([f"Question: {question}\nMessage: {message}"] * len(options),
            [f"Candidate description: {o}" for o in options],
            padding=True, truncation=True, max_length=256, return_tensors="pt")
with torch.no_grad():
    probs = model(**batch).logits.squeeze(-1).softmax(-1)
for o, p in sorted(zip(options, probs.tolist()), key=lambda x: -x[1]):
    print(f"{p:6.1%}  {o}")
```

## Limitations

- Trained on 2-8 options; more options are untested and likely less accurate.
- Weak on knowledge-heavy questions and on many near-identical labels.
- Overconfident: stated probabilities run higher than actual accuracy.
- **Research and education only. Not for commercial use.** The source data mixes licenses (some non-commercial).
  Do not use it for real decisions or with sensitive data.
