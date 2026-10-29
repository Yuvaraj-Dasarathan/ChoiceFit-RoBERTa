# System comparison

Examples compared: 6597

| Metric | baseline | choicefit | jev |
| --- | ---: | ---: | ---: |
| hard accuracy | 37.00% | 66.69% | 78.25% |
| top target accuracy | 36.73% | 65.94% | 77.54% |
| mean selected target probability | 36.71% | 65.68% | 76.98% |
| brier score | 0.6063 | 0.4237 | 0.3111 |
| log loss | 1.0500 | 0.7811 | 0.8240 |
| mean confidence | 37.34% | 71.82% | 83.50% |

Brier score and log loss: lower is better.

| Outcome | Rows |
| --- | ---: |
| all systems correct | 1337 |
| no system correct | 498 |
| only baseline correct | 252 |
| only choicefit correct | 411 |
| only jev correct | 984 |
| same pick: baseline~choicefit | 2464 |
| same pick: baseline~jev | 2336 |
| same pick: choicefit~jev | 4142 |

## Accuracy by number of choices

| num_choices | rows | baseline | choicefit | jev |
| --- | ---: | ---: | ---: | ---: |
| 2 | 2988 | 47.16% | 74.03% | 80.29% |
| 4 | 1464 | 26.16% | 56.97% | 78.28% |
| 3 | 1402 | 34.24% | 67.97% | 74.25% |
| 5 | 441 | 23.81% | 47.39% | 75.96% |
| 6 | 148 | 17.57% | 52.03% | 62.16% |
| 8 | 86 | 13.95% | 38.37% | 84.88% |
| 7 | 68 | 11.76% | 47.06% | 42.65% |

## Accuracy by data source

| source | rows | baseline | choicefit | jev |
| --- | ---: | ---: | ---: | ---: |
| tasksource_dpo_pairs | 203 | 48.77% | 76.85% | 79.80% |
| dynasent/r1_votes | 113 | 34.51% | 68.14% | 77.88% |
| dynasent/r2_votes | 112 | 35.71% | 70.54% | 74.11% |
| dataset_train_nli | 83 | 51.81% | 83.13% | 73.49% |
| spartqa-mchoice | 78 | 30.77% | 51.28% | 58.97% |
| chaos-mnli-ambiguity/votes | 78 | 34.62% | 65.38% | 69.23% |
| proto_qa/proto_qa | 78 | 23.08% | 20.51% | 48.72% |
| WANLI | 71 | 38.03% | 70.42% | 78.87% |
| riddle_sense | 61 | 29.51% | 44.26% | 88.52% |
| medmcqa | 60 | 16.67% | 40.00% | 78.33% |
| swag/regular | 58 | 24.14% | 51.72% | 77.59% |
| wiqa | 58 | 22.41% | 60.34% | 56.90% |
| discosense | 57 | 29.82% | 24.56% | 45.61% |
| ekar_english | 55 | 20.00% | 30.91% | 54.55% |
| cloth | 54 | 27.78% | 48.15% | 85.19% |
| cycic_multiplechoice | 54 | 20.37% | 53.70% | 88.89% |
| head_qa/en | 53 | 13.21% | 32.08% | 90.57% |
| logiqa | 53 | 30.19% | 26.42% | 83.02% |
| cosmos_qa | 52 | 21.15% | 63.46% | 76.92% |
| math_qa | 52 | 19.23% | 26.92% | 73.08% |
| PARADISE | 51 | 39.22% | 92.16% | 80.39% |
| cos_e/v1.0 | 51 | 41.18% | 68.63% | 92.16% |
| hellaswag | 51 | 37.25% | 43.14% | 78.43% |
| mutual | 50 | 14.00% | 68.00% | 96.00% |
| social_i_qa | 50 | 38.00% | 58.00% | 88.00% |
| commonsense_qa | 49 | 14.29% | 65.31% | 85.71% |
| definite_pronoun_resolution | 45 | 46.67% | 62.22% | 88.89% |
| implicatures | 45 | 55.56% | 77.78% | 91.11% |
| art | 44 | 45.45% | 52.27% | 77.27% |
| cicero | 44 | 25.00% | 61.36% | 77.27% |
| quartz | 44 | 40.91% | 65.91% | 93.18% |
| winogrande/winogrande_xl | 44 | 50.00% | 38.64% | 86.36% |
| e-CARE | 43 | 46.51% | 76.74% | 90.70% |
| fig-qa | 43 | 55.81% | 76.74% | 93.02% |
| moral_stories/full | 43 | 39.53% | 81.40% | 88.37% |
| MedQA-USMLE-4-options-hf | 42 | 33.33% | 28.57% | 92.86% |
| ReSQ | 42 | 45.24% | 76.19% | 73.81% |
| piqa | 42 | 50.00% | 64.29% | 92.86% |
| arct | 41 | 51.22% | 70.73% | 70.73% |

(Top 40 sources shown; see examples.jsonl for all.)
