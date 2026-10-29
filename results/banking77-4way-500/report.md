# System comparison

Examples compared: 500

| Metric | baseline | choicefit | jev |
| --- | ---: | ---: | ---: |
| hard accuracy | 12.20% | 67.00% | 96.20% |
| top target accuracy | 12.20% | 67.00% | 96.20% |
| mean selected target probability | 12.20% | 67.00% | 96.20% |
| brier score | 0.7504 | 0.4587 | 0.0553 |
| log loss | 1.3872 | 0.8676 | 0.1930 |
| mean confidence | 25.06% | 69.15% | 97.65% |

Brier score and log loss: lower is better.

| Outcome | Rows |
| --- | ---: |
| all systems correct | 30 |
| no system correct | 11 |
| only baseline correct | 2 |
| only choicefit correct | 6 |
| only jev correct | 123 |
| same pick: baseline~choicefit | 67 |
| same pick: baseline~jev | 61 |
| same pick: choicefit~jev | 335 |

## Accuracy by number of choices

| num_choices | rows | baseline | choicefit | jev |
| --- | ---: | ---: | ---: | ---: |
| 4 | 500 | 12.20% | 67.00% | 96.20% |

## Accuracy by data source

| source | rows | baseline | choicefit | jev |
| --- | ---: | ---: | ---: | ---: |
| banking77 | 500 | 12.20% | 67.00% | 96.20% |

(Top 40 sources shown; see examples.jsonl for all.)
