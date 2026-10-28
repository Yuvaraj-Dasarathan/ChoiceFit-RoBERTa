"""Compare any number of prediction files (e.g. baseline RoBERTa, ChoiceFit, JEV) on the same labeled set."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .compare_predictions import aggregate, mean_of, read_predictions, score


def percent(value):
    return "n/a" if value is None else f"{value * 100:.2f}%"


def group_table(title, rows, names, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    lines = ["", f"## {title}", "", f"| {key} | rows | " + " | ".join(names) + " |",
             "| --- | ---: | " + " | ".join("---:" for _ in names) + " |"]
    for name, members in sorted(groups.items(), key=lambda item: -len(item[1])):
        cells = [percent(sum(m["scores"][n]["is_modal"] for m in members) / len(members)) for n in names]
        lines.append(f"| {name} | {len(members)} | " + " | ".join(cells) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", required=True, metavar="NAME=PATH",
                        help="Repeat for each system, e.g. --system baseline=a.jsonl --system choicefit=b.jsonl")
    parser.add_argument("--output-dir", required=True, help="New directory for report files")
    parser.add_argument("--max-source-rows", type=int, default=40)
    args = parser.parse_args()
    systems = {}
    for item in args.system:
        name, _, path = item.partition("=")
        if not name or not path or name in systems:
            parser.error(f"Bad or duplicate --system {item!r}; use NAME=PATH")
        systems[name] = read_predictions(path)
    names = list(systems)
    reference = set(systems[names[0]])
    for name in names[1:]:
        if set(systems[name]) != reference:
            raise ValueError(f"{name} covers different ids than {names[0]}")
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    output.mkdir(parents=True)

    rows = []
    for identifier in systems[names[0]]:
        first = systems[names[0]][identifier]
        for name in names[1:]:
            other = systems[name][identifier]
            if other["choices"] != first["choices"] or other.get("target") != first.get("target"):
                raise ValueError(f"{name} disagrees about choices or target for {identifier!r}")
        rows.append({"id": identifier, "message": first["message"], "source": first.get("source") or "unknown",
                     "num_choices": len(first["choices"]), "target": first["target"],
                     "choices": {n: systems[n][identifier]["choice"] for n in names},
                     "scores": {n: score(systems[n][identifier]) for n in names}})
    for row in rows:
        right = [n for n in names if row["scores"][n]["is_modal"]]
        row["correct_systems"] = right

    summary = {"examples": len(rows), "systems": {}}
    for name in names:
        wrapped = [{name: row["scores"][name]} for row in rows]
        summary["systems"][name] = {**aggregate(wrapped, name),
                                    "mean_confidence": mean_of(wrapped, name, "confidence")}
        for field in ("brier_score", "log_loss"):
            summary["systems"][name][field] = mean_of(wrapped, name, field.replace("_score", ""))
    summary["all_correct"] = sum(len(r["correct_systems"]) == len(names) for r in rows)
    summary["none_correct"] = sum(not r["correct_systems"] for r in rows)
    summary["only_correct"] = {n: sum(r["correct_systems"] == [n] for r in rows) for n in names}
    summary["agreement_with"] = {f"{a}~{b}": sum(r["choices"][a] == r["choices"][b] for r in rows)
                                 for i, a in enumerate(names) for b in names[i + 1:]}

    lines = ["# System comparison", "", f"Examples compared: {len(rows)}", "",
             "| Metric | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for _ in names) + " |"]
    for field, fmt in (("hard_accuracy", percent), ("top_target_accuracy", percent),
                       ("mean_selected_target_probability", percent),
                       ("brier_score", lambda v: "n/a" if v is None else f"{v:.4f}"),
                       ("log_loss", lambda v: "n/a" if v is None else f"{v:.4f}"),
                       ("mean_confidence", percent)):
        lines.append(f"| {field.replace('_', ' ')} | " +
                     " | ".join(fmt(summary["systems"][n][field]) for n in names) + " |")
    lines += ["", "Brier score and log loss: lower is better.", "", "| Outcome | Rows |", "| --- | ---: |",
              f"| all systems correct | {summary['all_correct']} |",
              f"| no system correct | {summary['none_correct']} |"]
    lines += [f"| only {n} correct | {c} |" for n, c in summary["only_correct"].items()]
    lines += [f"| same pick: {pair} | {count} |" for pair, count in summary["agreement_with"].items()]
    lines += group_table("Accuracy by number of choices", rows, names, "num_choices")
    source_lines = group_table("Accuracy by data source", rows, names, "source")
    lines += source_lines[:4 + args.max_source_rows] + ["", f"(Top {args.max_source_rows} sources shown; see examples.jsonl for all.)"]

    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
