# System comparison

Examples compared: 500

| Metric | baseline | choicefit | jev |
| --- | ---: | ---: | ---: |
| hard accuracy | 16.20% | 82.80% | 99.40% |
| top target accuracy | 16.20% | 82.80% | 99.40% |
| mean selected target probability | 16.20% | 82.80% | 99.40% |
| brier score | 0.7503 | 0.2610 | 0.0085 |
| log loss | 1.3869 | 0.5031 | 0.0177 |
| mean confidence | 25.05% | 80.44% | 98.93% |

Brier score and log loss: lower is better.

| Outcome | Rows |
| --- | ---: |
| all systems correct | 45 |
| no system correct | 0 |
| only baseline correct | 1 |
| only choicefit correct | 0 |
| only jev correct | 52 |
| same pick: baseline~choicefit | 62 |
| same pick: baseline~jev | 78 |
| same pick: choicefit~jev | 413 |

## Accuracy by number of choices

| num_choices | rows | baseline | choicefit | jev |
| --- | ---: | ---: | ---: | ---: |
| 4 | 500 | 16.20% | 82.80% | 99.40% |

## Accuracy by data source

| source | rows | baseline | choicefit | jev |
| --- | ---: | ---: | ---: | ---: |
| clinc_oos | 500 | 16.20% | 82.80% | 99.40% |

(Top 40 sources shown; see examples.jsonl for all.)
