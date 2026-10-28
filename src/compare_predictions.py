"""Compare ChoiceFit and JEV predictions exported from the same labeled JSONL set."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_predictions(path):
    records = {}
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{path}:{line_number}: prediction has no usable id")
            if identifier in records:
                raise ValueError(f"{path}:{line_number}: duplicate id {identifier!r}")
            records[identifier] = record
    if not records:
        raise ValueError(f"No predictions in {path}")
    return records


def target_probabilities(record):
    target = record.get("target")
    choices = record["choices"]
    if isinstance(target, str):
        return {key: float(key == target) for key in choices}
    if isinstance(target, dict) and set(target) == set(choices):
        return target
    raise ValueError(f"Prediction {record.get('id')!r} has an invalid target")


def score(record):
    target = target_probabilities(record)
    selected = target[record["choice"]]
    best = max(target.values())
    hard = sum(value == 1 for value in target.values()) == 1
    result = {"target_probability": selected, "is_modal": selected >= best - 1e-6,
              "is_hard_correct": hard and selected == 1, "is_hard": hard}
    predicted = record.get("probabilities")
    if predicted:
        result["brier"] = sum((predicted.get(key, 0.0) - target[key]) ** 2 for key in target)
        result["log_loss"] = -sum(target[key] * math.log(max(predicted.get(key, 0.0), 1e-12))
                                  for key in target if target[key] > 0)
        result["confidence"] = predicted.get(record["choice"], max(predicted.values()))
    return result


def mean_of(rows, prefix, field):
    values = [row[prefix][field] for row in rows if field in row[prefix]]
    return sum(values) / len(values) if values else None


def aggregate(rows, prefix):
    hard = [row for row in rows if row[prefix]["is_hard"]]
    return {
        "brier_score": mean_of(rows, prefix, "brier"),
        "log_loss": mean_of(rows, prefix, "log_loss"),
        "mean_confidence": mean_of(rows, prefix, "confidence"),
        "examples": len(rows),
        "hard_examples": len(hard),
        "hard_accuracy": (sum(row[prefix]["is_hard_correct"] for row in hard) / len(hard)
                          if hard else None),
        "top_target_accuracy": sum(row[prefix]["is_modal"] for row in rows) / len(rows),
        "mean_selected_target_probability": sum(
            row[prefix]["target_probability"] for row in rows) / len(rows),
    }


CONFIDENCE_BINS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def breakdown(examples, key):
    groups = defaultdict(list)
    for row in examples:
        groups[row[key]].append(row)
    result = []
    for name, rows in groups.items():
        result.append({
            key: name, "examples": len(rows),
            "choicefit_accuracy": sum(r["choicefit"]["is_modal"] for r in rows) / len(rows),
            "jev_accuracy": sum(r["jev"]["is_modal"] for r in rows) / len(rows),
            "choicefit_brier": mean_of(rows, "choicefit", "brier"),
            "jev_brier": mean_of(rows, "jev", "brier"),
            "choicefit_only_modal": sum(r["outcome"] == "choicefit_only_modal" for r in rows),
            "jev_only_modal": sum(r["outcome"] == "jev_only_modal" for r in rows)})
    return sorted(result, key=lambda item: -item["examples"])


def calibration(examples, prefix):
    """Accuracy per confidence bucket; a calibrated model's accuracy is close to its confidence."""
    rows = [r for r in examples if "confidence" in r[prefix]]
    buckets, error = [], 0.0
    for low, high in zip(CONFIDENCE_BINS, CONFIDENCE_BINS[1:]):
        inside = [r for r in rows if low <= r[prefix]["confidence"] < high]
        if not inside:
            continue
        confidence = sum(r[prefix]["confidence"] for r in inside) / len(inside)
        accuracy = sum(r[prefix]["is_modal"] for r in inside) / len(inside)
        error += len(inside) / len(rows) * abs(confidence - accuracy)
        buckets.append({"range": f"{low:.1f}-{min(high, 1.0):.1f}", "examples": len(inside),
                        "mean_confidence": confidence, "accuracy": accuracy})
    return {"expected_calibration_error": error if rows else None, "buckets": buckets}


def render_markdown(summary, examples):
    def percent(value):
        return "n/a" if value is None else f"{value * 100:.2f}%"
    lines = ["# ChoiceFit vs JEV", "", f"Examples compared: {summary['examples']}", "",
             "| Metric | ChoiceFit | JEV |", "| --- | ---: | ---: |"]
    for field in ("hard_accuracy", "top_target_accuracy", "mean_selected_target_probability"):
        label = field.replace("_", " ")
        lines.append(f"| {label} | {percent(summary['choicefit'][field])} | {percent(summary['jev'][field])} |")
    lines += ["", "Lower is better for Brier score and log loss. Mean confidence is the average "
              "probability each model gave its own pick.", "",
              "| Metric | ChoiceFit | JEV |", "| --- | ---: | ---: |"]
    for field in ("brier_score", "log_loss", "mean_confidence"):
        ours, theirs = summary["choicefit"][field], summary["jev"][field]
        fmt = (lambda v: "n/a" if v is None else f"{v:.4f}")
        lines.append(f"| {field.replace('_', ' ')} | {fmt(ours)} | {fmt(theirs)} |")
    lines += ["", "| Outcome | Count |", "| --- | ---: |"]
    for field in ("both_modal", "choicefit_only_modal", "jev_only_modal", "neither_modal", "same_choice"):
        lines.append(f"| {field.replace('_', ' ')} | {summary[field]} |")
    def table(title, rows, key, limit=None):
        out = ["", f"## {title}", "",
               f"| {key} | rows | ChoiceFit acc | JEV acc | ChoiceFit Brier | JEV Brier | CF-only right | JEV-only right |",
               "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in rows[:limit]:
            brier = lambda v: "n/a" if v is None else f"{v:.3f}"
            out.append(f"| {r[key]} | {r['examples']} | {percent(r['choicefit_accuracy'])} | "
                       f"{percent(r['jev_accuracy'])} | {brier(r['choicefit_brier'])} | "
                       f"{brier(r['jev_brier'])} | {r['choicefit_only_modal']} | {r['jev_only_modal']} |")
        return out
    lines += table("By number of choices", summary["by_num_choices"], "num_choices")
    lines += table("By data source (40 largest; all in report.json)", summary["by_source"], "source", 40)
    lines += ["", "## Calibration (is stated confidence trustworthy?)", ""]
    for label, prefix in (("ChoiceFit", "choicefit"), ("JEV", "jev")):
        cal = summary["calibration"][prefix]
        ece = cal["expected_calibration_error"]
        lines += [f"**{label}** expected calibration error: {'n/a' if ece is None else f'{ece:.4f}'} "
                  "(0 is perfect)", "", "| confidence | rows | mean confidence | actual accuracy |",
                  "| --- | ---: | ---: | ---: |"]
        lines += [f"| {b['range']} | {b['examples']} | {percent(b['mean_confidence'])} | "
                  f"{percent(b['accuracy'])} |" for b in cal["buckets"]] + [""]
    for heading, key in (("ChoiceFit-only correct examples", "choicefit_only_modal"),
                         ("JEV-only correct examples", "jev_only_modal")):
        lines += ["", f"## {heading}", ""]
        selected = [row for row in examples if row["outcome"] == key][:10]
        if not selected:
            lines.append("None.")
        for row in selected:
            message = row["message"].replace("\n", " ")[:180]
            lines.append(f"- `{row['id']}` — {message}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choicefit", required=True, help="ChoiceFit prediction JSONL")
    parser.add_argument("--jev", required=True, help="JEV prediction JSONL")
    parser.add_argument("--output-dir", required=True, help="New directory for report files")
    args = parser.parse_args()
    choicefit, jev = read_predictions(args.choicefit), read_predictions(args.jev)
    if set(choicefit) != set(jev):
        raise ValueError(f"Prediction id sets differ: ChoiceFit-only={len(set(choicefit) - set(jev))}, "
                         f"JEV-only={len(set(jev) - set(choicefit))}")
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    output.mkdir(parents=True)
    examples = []
    for identifier in choicefit:
        ours, theirs = choicefit[identifier], jev[identifier]
        if ours["choices"] != theirs["choices"] or ours.get("target") != theirs.get("target"):
            raise ValueError(f"Predictions disagree about choices or target for {identifier!r}")
        ours_score, theirs_score = score(ours), score(theirs)
        if ours_score["is_modal"] and theirs_score["is_modal"]:
            outcome = "both_modal"
        elif ours_score["is_modal"]:
            outcome = "choicefit_only_modal"
        elif theirs_score["is_modal"]:
            outcome = "jev_only_modal"
        else:
            outcome = "neither_modal"
        examples.append({"id": identifier, "message": ours["message"], "question": ours.get("question"),
                         "target": ours["target"], "source": ours.get("source") or "unknown",
                         "num_choices": len(ours["choices"]), "choicefit_choice": ours["choice"],
                         "jev_choice": theirs["choice"], "choicefit": ours_score, "jev": theirs_score,
                         "outcome": outcome})
    summary = {"examples": len(examples), "choicefit": aggregate(examples, "choicefit"),
               "jev": aggregate(examples, "jev"),
               "both_modal": sum(row["outcome"] == "both_modal" for row in examples),
               "choicefit_only_modal": sum(row["outcome"] == "choicefit_only_modal" for row in examples),
               "jev_only_modal": sum(row["outcome"] == "jev_only_modal" for row in examples),
               "neither_modal": sum(row["outcome"] == "neither_modal" for row in examples),
               "by_num_choices": breakdown(examples, "num_choices"),
               "by_source": breakdown(examples, "source"),
               "calibration": {"choicefit": calibration(examples, "choicefit"),
                               "jev": calibration(examples, "jev")},
               "same_choice": sum(row["choicefit_choice"] == row["jev_choice"] for row in examples)}
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in examples))
    (output / "report.md").write_text(render_markdown(summary, examples))
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
