"""Create a browser-ready loss comparison from convergence JSONL files."""
import argparse
import html
import json
from pathlib import Path


def load_run(specification):
    try:
        name, raw_path = specification.split("=", 1)
    except ValueError as exc:
        raise ValueError("Each --run must be NAME=PATH") from exc
    if not name or not raw_path:
        raise ValueError("Each --run must be NAME=PATH")
    path = Path(raw_path)
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    metadata = next((event for event in events if event.get("event") == "run"), None)
    updates = [event["update"] for event in events if event.get("event") == "update"]
    if not updates:
        raise ValueError(f"{path} has no completed optimizer updates")
    updates_per_epoch = max(updates)
    training_points, validation_points = [], []
    for event in events:
        if event.get("event") == "update" and "mean_train_loss" in event:
            total_update = (event["epoch"] - 1) * updates_per_epoch + event["update"]
            training_points.append([total_update, event["mean_train_loss"]])
        if event.get("event") != "validation":
            continue
        epoch, update = event["epoch"], event["update"]
        if event.get("end_of_epoch") and epoch:
            continue
        total_update = 0 if epoch == 0 else (epoch - 1) * updates_per_epoch + update
        validation_points.append([total_update, event["validation"]["loss"]])
    if not validation_points:
        raise ValueError(f"{path} has no validation records")
    learning_rate = metadata["arguments"]["learning_rate"] if metadata else None
    return {"name": name, "learning_rate": learning_rate,
            "training_points": training_points, "validation_points": validation_points}


def chart_svg(runs, field, title):
    selected = [run for run in runs if run[field]]
    if not selected:
        return f"<p>{html.escape(title)}: no values recorded.</p>"
    width, height, left, right, top, bottom = 900, 440, 78, 26, 54, 64
    plot_width, plot_height = width - left - right, height - top - bottom
    points = [point for run in selected for point in run[field]]
    x_max = max(point[0] for point in points) or 1
    values = [point[1] for point in points]
    low, high = min(values), max(values)
    padding = max((high - low) * .1, .0001)
    y_min, y_max = low - padding, high + padding
    x = lambda value: left + value * plot_width / x_max
    y = lambda value: top + (y_max - value) * plot_height / (y_max - y_min)
    marks = [f'<text x="{left}" y="20">{html.escape(title)}</text>',
             f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="currentColor" opacity=".35"/>']
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4
        position = y(value)
        marks.append(f'<path d="M{left} {position:.1f}H{width - right}" stroke="currentColor" opacity=".15"/>')
        marks.append(f'<text x="8" y="{position + 4:.1f}">{value:.5f}</text>')
        update = x_max * index / 4
        marks.append(f'<text x="{x(update):.1f}" y="{height - 38}" text-anchor="middle">{update:.0f}</text>')
    colors = ("#1769aa", "#c45a00", "#218739", "#7b4fb7")
    for index, run in enumerate(selected):
        path = " ".join(f"{'M' if point_index == 0 else 'L'}{x(update):.1f},{y(loss):.1f}"
                        for point_index, (update, loss) in enumerate(run[field]))
        label = run["name"] if run["learning_rate"] is None else f"{run['name']} (lr={run['learning_rate']:g})"
        color = colors[index % len(colors)]
        marks.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        marks.append(f'<text x="{left + index * 190}" y="38" fill="{color}">{html.escape(label)}</text>')
    marks.append(f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle">optimizer update</text>')
    marks.append(f'<text x="20" y="{height / 2}" text-anchor="middle" transform="rotate(-90 20 {height / 2})">loss</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">{"".join(marks)}</svg>'


def render_html(runs):
    return f"""<!doctype html>
<meta charset="utf-8">
<title>ChoiceFit loss comparison</title>
<main>
  <h1>Training and validation loss comparison</h1>
  <p>Each line is one run. Use validation loss to choose the best run and stopping point.</p>
  {chart_svg(runs, "training_points", "Training loss — after every weight update")}
  {chart_svg(runs, "validation_points", "Validation loss — every validation check")}
</main>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #202124; }}
svg {{ display: block; width: min(100%, 1000px); height: auto; margin-block: 24px; }}
svg text {{ fill: currentColor; font-size: 12px; }}
@media (prefers-color-scheme: dark) {{ body {{ background: #202124; color: #e8eaed; }} }}
</style>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="Comparison input as NAME=PATH")
    parser.add_argument("--output", default="artifacts/convergence-explainer.html")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        runs = [load_run(specification) for specification in args.run]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; choose another --output or pass --overwrite")
    output.write_text(render_html(runs))
    print(json.dumps({"output": str(output), "runs": [run["name"] for run in runs]}), flush=True)


if __name__ == "__main__":
    main()
