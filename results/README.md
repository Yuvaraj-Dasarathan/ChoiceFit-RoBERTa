# Measured results

Every number in the project README, the demo UI and the Hugging Face model card comes from the files in this folder. They are real outputs of the evaluation scripts in `src/`, not estimates.

| Folder or file | What it is |
| --- | --- |
| [mixed-test/report.md](mixed-test/report.md) | 6,597 held-out rows, never used for training or model selection. ChoiceFit vs JEV vs untrained baseline. Accuracy, Brier score, log loss, confidence. |
| [banking77-4way-500/](banking77-4way-500/report.md) | 500 banking-intent rows, 4 options each, from a different source dataset. Includes every per-row prediction (`choicefit.jsonl`, `jev.jsonl`). |
| [clinc-4way-500/](clinc-4way-500/report.md) | 500 everyday-intent rows, 4 options each, same layout. |
| [training_metrics.json](training_metrics.json) | Training arguments, per-epoch train loss and validation metrics, epoch times. |
| [test_metrics.json](test_metrics.json) | The one-time test result computed right after training. |

## How the numbers were produced

- **Test data was untouched.** The 6,597 test rows were not used to train or to choose the checkpoint (that used the separate validation split).
- **Same rows for every system.** `src/compare_all.py` refuses to compare systems unless they cover identical row IDs.
- **JEV was the real API.** `src/evaluate_jev.py` sent each row to TypeSafe.ai's JEV endpoint. The 500-row files contain JEV's actual answers.
- **Accuracy** means the top-scored option is the target-best option.

## Reproduce

```bash
python -m src.evaluate --model <model dir> --data <labeled.jsonl> --predictions out.jsonl
python -m src.evaluate_jev --data <labeled.jsonl> --predictions jev.jsonl   # needs your own JEV API key
python -m src.compare_all --system choicefit=out.jsonl --system jev=jev.jsonl
```

The full 6,597-row prediction files are not committed (they contain third-party dataset text). Regenerate them with the commands above; see [docs/sagemaker.md](../docs/sagemaker.md), sections 7 to 13.
