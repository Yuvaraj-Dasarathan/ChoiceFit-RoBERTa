# ChoiceFit-RoBERTa

A RoBERTa-base model, fine-tuned to **choose the best option for a message from 2-8 options supplied with each request**, with no fixed label set.

- **Try it:** https://yuvaraj-dasarathan.github.io/ChoiceFit-RoBERTa/ (runs on my laptop, so it can be offline)
- **Model:** https://huggingface.co/Yuvaraj-ml/choicefit-roberta
- **Inspired by:** [TypeSafe.ai](https://typesafe.ai)'s JEV classification

## Why I built this

I wanted to learn the full lifecycle of fine-tuning a transformer end to end: data preparation, training on a laptop, scaling to a cloud GPU, evaluating honestly, and shipping a public demo. TypeSafe.ai's JEV idea made a good target: instead of a classifier with a fixed set of labels, every request brings its own options (for example "report a duplicate charge / reset password / open an account"), and the model picks one. I wanted to see how close a small open model could get, and I measured it against the real JEV API.

## What is trained

Each (question + message, option description) pair goes through RoBERTa and gets **one score** from a shared single-output head. A softmax over the options of one request gives the probabilities. Option IDs are only handles; the encoder reads option *text*, so new options need no retraining. Both RoBERTa and the scoring head are trained, with cross-entropy against the target distribution (soft targets are preserved).

## Results

**These are not random or AI-generated percentages.** Every number was measured by scripts in this repo on held-out data the model never trained on, and the raw reports and per-row predictions are committed in [`results/`](results/README.md) so you can check or reproduce them.

Hard accuracy = the top-scored option is the correct one.

| Test set | ChoiceFit | TypeSafe JEV |
| --- | ---: | ---: |
| Held-out mixed test set (6,597 rows, 2-8 options) | 66.7% | 78.3% |
| banking77, 4 options (500 rows) | 67.0% | 96.2% |
| clinc_oos, 4 options (500 rows) | 82.8% | 99.4% |

JEV is stronger on every set, as expected for a commercial system. ChoiceFit is a learning project. Weak spots: knowledge-heavy questions, many near-identical labels, and overconfident probabilities (mean confidence 71.8% against 66.7% accuracy on the mixed test set). Full side-by-side reports (Brier score, log loss, rows only one system got right, accuracy by option count) are in [`results/`](results/README.md).

## Data

- Source: [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions), `choice` records only (about 400 underlying datasets, mixed licenses).
- **485,945 training**, 6,651 validation, 6,597 test examples (about half a million).
- Options per example: 2 (38%), 3 (26%), 4 (23%), 5-8 (12%).
- Topics (approximate, from my own grouping of the source tags): reasoning/entailment 34%, science/math/medical 11%, commonsense and reading comprehension 8%, safety/hate/prompt-injection 8%, sentiment/emotion 7%, other 32%. It is general-purpose, not built for one industry.
- Preparation streams the dataset, filters examples that exceed the token limit and keeps groups together to avoid train/test overlap. See `src/prepare_full_typed_decisions.py`.

## How I trained it

**1. Pilot runs on an M1 MacBook Air (16 GB).** I first proved the pipeline on tiny samples (32, then 1,000 and 5,000 rows) on Apple MPS, with batch size 1, gradient checkpointing and 192 tokens (peak memory about 4.4 GB). A 1,000-row epoch takes roughly 6-7 minutes, and a five-epoch run about 35 minutes. The 1,000-row model scored about chance (31% validation, 38% test). That was the point: the code worked, and data scale was the bottleneck.

**2. Choosing the learning rate.** The trainer logs every update's loss and periodic validation loss (`convergence.jsonl`), and `src/plot_convergence.py` overlays runs. I compared 5e-6, 1e-5 and 2e-5 on the same seed, splits and epochs, and picked the rate with stable loss decrease and the lowest validation loss. The full run used **2e-5** with 10% warmup and linear decay.

**3. Full training on AWS SageMaker.** Instance `ml.g6.2xlarge` with **one NVIDIA L4 GPU (24 GB)**, on-demand, bf16, 256 tokens, up to 8 options, about 64 messages per optimizer update, learning rate 2e-5, up to 3 epochs. The trainer validated every 625 updates, kept the weights with the **lowest validation loss**, and stopped early after 5 checks without improvement. SageMaker billed about 7 hours (25,089 s) for the final run on the L4.

**Convergence, as measured** (see `results/training_metrics.json`):
- Validation accuracy rose from 37.9% (before training) to about 67% by epoch 2.
- The lowest validation loss (0.764) came early in epoch 3, at about 2.1 epochs. The saved model is that checkpoint (validation accuracy 67.1%, test 66.7%).
- After that, validation loss stopped improving (it bounced between about 0.77 and 0.80) while training loss kept falling (0.89, 0.70, 0.62 by epoch). That gap is the start of overfitting, so early stopping ended the run at about 2.5 of 3 epochs.

**GPU use.** Raising batch size and token length took L4 memory use only from about 14% to about 20%, so the run was limited by something other than memory (gradient checkpointing and compute are the likely causes). The lesson: measure utilization instead of assuming that bigger batches speed things up.

**Time and cost.** About 50+ hours of total effort, counting laptop experiments, cloud runs, evaluation and building the demo. SageMaker billing: **20+ GPU hours in total** across all runs. About 7 hours on the L4 for the main run (plus a 23-minute L4 pilot), and about 16 hours on `ml.g4dn.xlarge` (NVIDIA T4), mostly one 15.4-hour run at learning rate 1e-5 (batch 1, 192 tokens, about 392K rows; 63.0% validation accuracy, 58.8% test) Compared with those, the final L4 setup (learning rate 2e-5, bigger batch) reached 68.4% validation accuracy. Total spend was about **$50**, all inclusive: those runs, small pilots, and the paid JEV API calls used for testing.

## Evaluate against JEV

`src/evaluate.py` scores any labeled JSONL file; `src/evaluate_jev.py` sends the same rows to the JEV API; `src/compare_all.py` builds the side-by-side report. `scripts/run_eval.sh` runs it all with one command and takes a new dataset name each time. The API key is read from an environment variable and never stored. Details: [docs/sagemaker.md](docs/sagemaker.md).

## Use the model

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

## Public demo (React + FastAPI + ngrok)

- `ui/`: mobile-first React app, deployed to GitHub Pages by `.github/workflows/pages.yml`. It shows a "server temporarily down" banner when the API is unreachable.
- `server/api.py`: FastAPI service with input validation (2-8 options, length caps), per-IP and global rate limits (per minute and per day), a queue cap, a kill switch and an origin allow-list. It reads the real client IP from the ngrok `X-Forwarded-For` header, so a spoofed header does not bypass the limits.
- `scripts/start_demo.sh`: starts the API plus an ngrok tunnel and keeps the Mac awake. Setup and settings: [docs/sagemaker.md](docs/sagemaker.md), section 14.

## Repository map

| Path | What |
| --- | --- |
| `src/` | Data preparation, model, training, evaluation, JEV client, comparison reports, convergence plots |
| `cloud/` | SageMaker launcher, training entrypoint, IAM policy templates |
| `server/`, `ui/`, `scripts/` | Demo API, React UI, start/eval/publish scripts |
| `tests/` | 37 tests (data validation, soft targets, recovery, scaling, evaluation tools) |
| `docs/` | Guides: [SageMaker, evaluation and demo](docs/sagemaker.md), [local M1 guide](docs/typed-decisions-local.md), [HF model card](docs/model_card.md) |

## Setup

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[data,test]'
python -m pytest -q
```

## Limitations and licence

Research and education only; not for commercial use. The source data mixes licenses (about 12% non-commercial). The model is overconfident, is trained on 2-8 options, and must not be used for real decisions or with sensitive data.
