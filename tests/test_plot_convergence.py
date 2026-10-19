import json

from src.plot_convergence import load_run, render_html


def test_comparison_uses_interval_validation_and_learning_rate(tmp_path):
    trace = tmp_path / "trace.jsonl"
    events = [
        {"event": "run", "arguments": {"learning_rate": 1e-5}},
        {"event": "validation", "epoch": 0, "update": 0, "validation": {"loss": 1.2}, "end_of_epoch": True},
        {"event": "update", "epoch": 1, "update": 25, "mean_train_loss": 0.9},
        {"event": "validation", "epoch": 1, "update": 25, "validation": {"loss": 1.0}},
        {"event": "validation", "epoch": 1, "update": 25, "validation": {"loss": 1.0}, "end_of_epoch": True},
    ]
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    run = load_run(f"small={trace}")
    assert run == {"name": "small", "learning_rate": 1e-5,
                   "training_points": [[25, 0.9]], "validation_points": [[0, 1.2], [25, 1.0]]}
    page = render_html([run])
    assert "Training loss — after every weight update" in page
    assert 'd="M78.0,' in page
