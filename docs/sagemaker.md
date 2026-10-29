aws login --profile choicefit-login --region us-east-1

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
  --job-name choicefit-l4-pilot-b8-001 \
  --max-run 7200
```

## 3. Launch the L4 pilot

Run the same command with `--launch`:

```bash
.venv/bin/python -m cloud.launch \
  --instance-type ml.g6.2xlarge \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-l4-pilot \
  --job-name choicefit-l4-pilot-b8-001 \
  --max-run 7200 --launch
```

Check CloudWatch. Confirm it reports `cuda`, `precision: bf16`, no out-of-memory error, and the expected progress speed. This pilot uses batch 8, 64 candidate pairs, accumulation 8, length 256, learning rate `2e-5`, and 10% linear warmup. Its effective batch remains 64 messages.

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
  --job-name choicefit-l4-full-b8-001 \
  --max-run 86400
```

## 6. Launch the full job

```bash
.venv/bin/python -m cloud.launch \
  --instance-type ml.g6.2xlarge \
  --base-model .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b \
  --data-dir data/typed-full-8x256 \
  --job-name choicefit-l4-full-b8-001 \
  --max-run 86400 --launch
```

The job uses BF16, linear warmup/decay, and saves recovery checkpoints. Keep the test split only for final reporting; validation loss controls the saved model and early stopping.

## 7. Download the completed model and evaluate the untouched test set

Set `JOB_NAME` to the completed SageMaker training-job name. This downloads the one final best-validation model that SageMaker exported, not the recovery checkpoints.

```bash
export AWS_PROFILE=choicefit
export AWS_SDK_UA_APP_ID=AWSSkill-SageMaker
JOB_NAME='replace-with-completed-job-name'
MODEL_DIR="artifacts/sagemaker/$JOB_NAME"

mkdir -p "$MODEL_DIR"
aws s3 cp --only-show-errors \
  "s3://yuvaraj-choicefit-training/choicefit/$JOB_NAME/output/model.tar.gz" \
  "$MODEL_DIR/model.tar.gz"
tar -xzf "$MODEL_DIR/model.tar.gz" -C "$MODEL_DIR"
rm "$MODEL_DIR/model.tar.gz"
```

Nothing in steps 7–9 runs on SageMaker; they all run on your laptop. The only SageMaker-side test is automatic: at the end of the training job, `cloud/entrypoint.py` evaluates the reloaded model on `test.jsonl` once and saves `test_metrics.json` (aggregate numbers only). This local command re-runs the same test and also creates the per-example prediction file the comparison needs. The test file is passed with `--data`; change `--data` and `--predictions` for another labeled JSONL file (or use `scripts/run_eval.sh`, section 11).

```bash
JOB_NAME=choicefit-l4-full-b8-001   # set again in case this is a new terminal
MODEL_DIR="artifacts/sagemaker/$JOB_NAME"

.venv/bin/python -m src.evaluate \
  --model "$MODEL_DIR" \
  --data data/typed-full-8x256/test.jsonl \
  --batch-size 8 --pair-batch-size 64 \
  --predictions "artifacts/sagemaker/$JOB_NAME/choicefit-test.jsonl"
```

## 8. Run the same labeled file through JEV

Create a TypeSafe API key, then set it only in your shell. Do not put it in a file or commit it.

```bash
export TYPESAFE_API_KEY='paste-your-key-here'

.venv/bin/python -m src.evaluate_jev \
  --data data/typed-full-8x256/test.jsonl \
  --predictions "artifacts/sagemaker/$JOB_NAME/jev-test.jsonl" \
  --model jev-latest \
  --workers 8
```

The script sends one API request per JSONL row (TypeSafe has no batch-file endpoint, and each row has its own choices). Add `--workers 8` to send 8 requests concurrently with threads; results are still written in input order. Start with 4–8 and check your org's limits at console.typesafe.ai/rate-limits; HTTP 429 responses are retried automatically. If it is interrupted, run the same command again with `--resume`; completed rows are skipped and are not billed again. Use `--limit 100` first if you want a small API-cost and format check.

## 9. Compare ChoiceFit and JEV

```bash
.venv/bin/python -m src.compare_predictions \
  --choicefit "artifacts/sagemaker/$JOB_NAME/choicefit-test.jsonl" \
  --jev "artifacts/sagemaker/$JOB_NAME/jev-test.jsonl" \
  --output-dir "artifacts/sagemaker/$JOB_NAME/comparison-test"
```

Open `comparison-test/report.md` for aggregate results and representative cases where one model selected a target-best choice and the other did not. `examples.jsonl` has every row for deeper inspection. Both inputs must contain the same unique `id` values and a ground-truth `target` or `label`.

## 10. Reference: what is in `artifacts/sagemaker/<job>/`

`artifacts/` is git-ignored. Files from SageMaker's `model.tar.gz`:

| File | Meaning |
| --- | --- |
| `model.safetensors`, `config.json` | Best-validation fine-tuned weights and config (what `--model` loads) |
| `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, `special_tokens_map.json` | RoBERTa tokenizer |
| `training_metrics.json` | Training arguments plus validation metrics per epoch |
| `convergence.jsonl` | Large step-by-step training/validation log (use `src.plot_convergence`) |
| `test_metrics.json` | Aggregate test result SageMaker computed once after training |

Files you generate locally:

| File | Meaning |
| --- | --- |
| `choicefit-test.jsonl` | One row per test example from our model: `id`, `message`, `choices`, `target`, predicted `choice`, probabilities |
| `jev-test.jsonl` | Same format, from the TypeSafe JEV API |
| `comparison-test/` | `report.md` (summary), `report.json` (numbers), `examples.jsonl` (per-row outcomes) |

The aggregate numbers from the local run are printed to the terminal by `src.evaluate` (they are not saved to a file; redirect with `> metrics.json` if wanted). The local run of job `choicefit-l4-full-b8-001` matched SageMaker's `test_metrics.json`: accuracy 0.6669 on 6,597 examples (6,289 hard, 308 soft).

## 11. Quick start: test any labeled file

One command runs ChoiceFit locally, calls the JEV API from your laptop, and builds the report in `artifacts/sagemaker/<job>/runs/<NAME>/`:

```bash
export TYPESAFE_API_KEY='paste-your-key-here'
scripts/run_eval.sh data/typed-full-8x256/test.jsonl test 100   # first 100 rows (cheap check)
scripts/run_eval.sh path/to/new-file.jsonl my-new-set           # whole file
```

The same steps by hand:

Everything is driven by file paths, so for a new dataset only change `DATA` and `NAME`. New data must be in the same JSONL format as `test.jsonl` (`id`, `message`, `question`, `choices`, `target`).

```bash
export AWS_PROFILE=choicefit
JOB_NAME=choicefit-l4-full-b8-001
OUT="artifacts/sagemaker/$JOB_NAME"
DATA=data/typed-full-8x256/test.jsonl   # change me
NAME=test                               # change me (used in output names)

# 1. ChoiceFit (about 8 minutes for 6.6k rows on Apple MPS)
.venv/bin/python -m src.evaluate --model "$OUT" --data "$DATA" \
  --batch-size 8 --pair-batch-size 64 --predictions "$OUT/choicefit-$NAME.jsonl"

# 2. JEV. The API key goes ONLY in this environment variable (never in a file).
export TYPESAFE_API_KEY='paste-your-key-here'
.venv/bin/python -m src.evaluate_jev --data "$DATA" \
  --predictions "$OUT/jev-$NAME.jsonl" --limit 100     # smoke test first
# Full run: drop --limit. If interrupted, re-run with --resume.

# 3. Compare (both prediction files must cover the same ids; use the same --limit
#    subset only if ChoiceFit predictions also cover those ids)
.venv/bin/python -m src.compare_predictions \
  --choicefit "$OUT/choicefit-$NAME.jsonl" --jev "$OUT/jev-$NAME.jsonl" \
  --output-dir "$OUT/comparison-$NAME"
```

Notes:
- Output files must not already exist (`evaluate` and `compare` refuse to overwrite); pick a new `NAME`.
- `compare_predictions` requires identical id sets. For a `--limit 100` JEV run, either create a 100-row copy of the data (`head -n 100 "$DATA" > data/small.jsonl`) and run both tools on it, or run JEV on the full file.
- Each JEV row is one billed API call.

## 12. Untrained RoBERTa baseline and three-way comparison

The baseline is the original `roberta-base` with a randomly initialized scoring head (seeded, `--seed 42`), run through the same code and test file. It shows how much the fine-tuning added. The base model is already cached under `.cache/huggingface/hub/models--FacebookAI--roberta-base`; if it is missing, `HF_HOME="$PWD/.cache/huggingface" .venv/bin/python -c "from huggingface_hub import snapshot_download as d; print(d('FacebookAI/roberta-base'))"` downloads it. Use `--max-length 256` so the baseline sees the same 256 tokens as the fine-tuned model.

```bash
JOB_NAME=choicefit-l4-full-b8-001
OUT="artifacts/sagemaker/$JOB_NAME"
BASE=$(echo .cache/huggingface/hub/models--FacebookAI--roberta-base/snapshots/*)

.venv/bin/python -m src.evaluate --model "$BASE" --max-length 256 --seed 42 \
  --data data/typed-full-8x256/test.jsonl --batch-size 8 --pair-batch-size 64 \
  --predictions "$OUT/baseline-test.jsonl"

.venv/bin/python -m src.compare_all \
  --system baseline="$OUT/baseline-test.jsonl" \
  --system choicefit="$OUT/choicefit-test.jsonl" \
  --system jev="$OUT/jev-test.jsonl" \
  --output-dir "$OUT/comparison-all-test"
```

`compare_all` accepts any number of `--system NAME=PATH` files (all must cover the same ids). `report.md` shows accuracy, Brier score, log loss and confidence per system, how many rows each system alone got right, how often systems picked the same option, and accuracy by number of choices and by source. `scripts/run_eval.sh` (section 11) now runs the baseline, ChoiceFit and JEV and writes both `comparison/` (ChoiceFit vs JEV) and `comparison-all/`.

Reference result on `test.jsonl` (6,597 rows): baseline 37.0% hard accuracy, ChoiceFit 66.7%, JEV 78.3%.

## 13. Simpler real-world test sets (banking77, clinc_oos)

`typed-full-8x256` is a very hard, mixed benchmark. These two customer-intent sets are easier and more realistic. Neither appears in the training sources. `src/prepare_intent_eval.py` converts a public dataset into the same JSONL format: each message becomes a question whose options are the true intent plus random distractor intents (shuffled, seeded).

### Already done (do NOT re-run)

| Step | Result |
| --- | --- |
| Created the datasets | `data/eval/banking77-4way-500.jsonl`, `data/eval/clinc-4way-500.jsonl` (500 rows each, 4 options) |
| Ran untrained baseline | `artifacts/sagemaker/<job>/runs/<NAME>/baseline.jsonl` |
| Ran ChoiceFit | `artifacts/sagemaker/<job>/runs/<NAME>/choicefit.jsonl` |

Results so far (chance is 25%): banking77 ChoiceFit 67.0% / baseline 12.2%; clinc_oos ChoiceFit 82.8% / baseline 16.2%.

### To do: add JEV and build the three-way report

```bash
export TYPESAFE_API_KEY='paste-your-key-here'
J=artifacts/sagemaker/choicefit-l4-full-b8-001
for N in banking77-4way-500 clinc-4way-500; do
  .venv/bin/python -m src.evaluate_jev --data data/eval/$N.jsonl --predictions $J/runs/$N/jev.jsonl --workers 8
  .venv/bin/python -m src.compare_all --system baseline=$J/runs/$N/baseline.jsonl \
    --system choicefit=$J/runs/$N/choicefit.jsonl --system jev=$J/runs/$N/jev.jsonl \
    --output-dir $J/runs/$N/comparison-all
done
```

Reports: `$J/runs/<NAME>/comparison-all/report.md` (about 1,000 JEV calls total).

### Optional: new datasets later

Make another file (`--choices` sets options per question, try 8 for harder; `--limit 0` keeps every row; `--seed` changes sampling):

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python -m src.prepare_intent_eval --dataset banking77 --choices 8 --limit 500 \
  --output data/eval/banking77-8way-500.jsonl
```

Then run everything (baseline, ChoiceFit, JEV, reports) with one command. It fails if `runs/<NAME>/` outputs already exist, so use a new NAME:

```bash
export TYPESAFE_API_KEY='paste-your-key-here'
scripts/run_eval.sh data/eval/banking77-8way-500.jsonl banking77-8way-500
```

For any other dataset, write a converter that produces rows with `message`, `question`, `choices` (`{"option_0": "description", ...}`), `target` (`{"option_0": 1.0, "option_1": 0.0, ...}`) and a unique `id`; `src/prepare_intent_eval.py` is a short template.

## 14. Public demo on Hugging Face (free)

The demo is a Gradio app, [demo/app.py](../demo/app.py), hosted as a Hugging Face Space. The Space loads the model from a Hub model repo and runs on the free CPU tier. It limits input to 2-8 options (the trained range) and shows a research/education-only notice.

### Try it locally first

```bash
.venv/bin/python -m pip install "gradio>=5,<6"     # demo-only dependency
.venv/bin/python demo/app.py                       # open http://127.0.0.1:7860
```

It loads `artifacts/sagemaker/choicefit-l4-full-b8-001` automatically. Set `CHOICEFIT_MODEL` to use another folder or Hub repo id.

### Publish

1. Create a free account at https://huggingface.co/join.
2. Create a **write** token at https://huggingface.co/settings/tokens and log in (the token is stored on your machine, never in the repo):
   ```bash
   .venv/bin/python -m huggingface_hub.commands.huggingface_cli login
   ```
3. Preview, then upload. This creates the model repo (`<you>/choicefit-roberta`, about 500 MB) and the Space (`<you>/choicefit-roberta-demo`) and links them:
   ```bash
   .venv/bin/python scripts/publish_to_hf.py --username YOUR_HF_NAME --dry-run
   .venv/bin/python scripts/publish_to_hf.py --username YOUR_HF_NAME
   ```
4. Wait a few minutes for the Space to build, then open `https://huggingface.co/spaces/YOUR_HF_NAME/choicefit-roberta-demo`. The free tier sleeps when idle, so the first visit can take about a minute.

The script uploads only the weights, config and tokenizer files (not `convergence.jsonl` or checkpoints). Re-running it updates the repos. It writes the model card, which names the source dataset and carries the research/education-only notice.

## 15. Public demo from your laptop (GitHub Pages + ngrok)

The Hugging Face Gradio Space needs a PRO plan (section 14), so the demo runs like this instead:

- **UI:** React app in `ui/`, deployed to GitHub Pages by `.github/workflows/pages.yml` on every push to `training` that touches `ui/`. Live at `https://yuvaraj-dasarathan.github.io/ChoiceFit-RoBERTa/`. In the repo's Settings > Pages, **Source** must be **GitHub Actions**.
- **API:** `server/api.py` (FastAPI, `POST /predict`, `GET /health`) on your laptop, exposed by ngrok at your fixed dev domain.
- If the laptop is off, the page still loads and shows "The server is temporarily down."

### Start the server (each time)

```bash
cd /Users/yuvaraj/Desktop/AI/choicefit_roberta
scripts/start_demo.sh
```

This starts the API (CORS allowed only for `https://yuvaraj-dasarathan.github.io`), starts `ngrok http 8000`, keeps the Mac awake, and prints the public URL. Press Ctrl+C to stop both. The page checks `/health` once when it opens (to show the "temporarily down" banner); after that only a failed submit reports an error, so refresh the page after starting the server.

Test it locally at `http://127.0.0.1:8000` (the API also serves the built UI from `web/` if it exists).

### Settings

| What | Where |
| --- | --- |
| Public API address used by the hosted page | `REMOTE_API_URL` in `ui/src/config.js` (your ngrok dev domain; it does not change on restart) |
| Allowed web origin | `CHOICEFIT_ORIGINS` env var (default in `scripts/start_demo.sh`) |
| Per-visitor limits | `CHOICEFIT_RATE_PER_MIN` (20) and `CHOICEFIT_IP_PER_DAY` (200) |
| All visitors together | `CHOICEFIT_GLOBAL_PER_MIN` (60) and `CHOICEFIT_GLOBAL_PER_DAY` (2000), the last protects the ngrok monthly quota |
| Queue limit | `CHOICEFIT_MAX_QUEUE` (5): extra requests get 503 instead of waiting |
| Kill switch | `CHOICEFIT_DISABLED=1` makes `/predict` return 503 |
| Input limits | `server/api.py`: 2-8 options, 2000-char message, 300-char options |

Changing `ui/` or `config.js` and pushing redeploys the page (about a minute). Build locally with `cd ui && npm install && npm run build` (output goes to `web/`, git-ignored). Node 14 works with the pinned Vite 2.

### Limits and shutdown

- ngrok free plan: 1 GB and 20,000 requests per month.
- Anyone with the URL can call the API, within the limits above. The visitor IP is the last `X-Forwarded-For` entry, because ngrok appends the real address and callers can fake earlier ones. Limits reset when the server restarts. Do not run it with AWS or TypeSafe keys in your environment.
- To end the demo: stop the script, then delete or unpublish the Pages site (Settings > Pages) or revert `config.js`.
