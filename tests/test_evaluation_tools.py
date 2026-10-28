import json

from src.compare_predictions import aggregate, score, target_probabilities
from src.evaluate_jev import payload_for


def record(target):
    return {"id": "one", "message": "hello", "question": "Which?",
            "choices": {"a": "first", "b": "second"}, "target": target, "choice": "b"}


def test_jev_payload_uses_question_as_instructions_and_descriptions_as_criteria():
    payload = payload_for(record({"a": 1.0, "b": 0.0}), "jev-latest")
    assert payload["state"] == "hello"
    assert payload["questions"]["decision"] == {
        "type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": "second"}}


def test_comparison_supports_hard_and_soft_targets():
    hard = score(record({"a": 0.0, "b": 1.0}))
    soft = score(record({"a": 0.25, "b": 0.75}))
    assert hard["is_hard_correct"] is True
    assert soft["is_hard"] is False
    assert soft["is_modal"] is True
    metrics = aggregate([{"choicefit": hard}, {"choicefit": soft}], "choicefit")
    assert metrics["hard_accuracy"] == 1
    assert metrics["top_target_accuracy"] == 1


def test_target_probabilities_converts_hard_label_to_distribution():
    value = record("a")
    assert target_probabilities(value) == {"a": 1.0, "b": 0.0}


def test_score_reports_brier_log_loss_and_confidence():
    value = record({"a": 0.0, "b": 1.0})
    value["probabilities"] = {"a": 0.25, "b": 0.75}
    result = score(value)
    assert abs(result["brier"] - 0.125) < 1e-9
    assert abs(result["log_loss"] + __import__("math").log(0.75)) < 1e-9
    assert result["confidence"] == 0.75


def test_compare_all_reports_three_systems(tmp_path, monkeypatch):
    from src import compare_all

    def write(name, choice):
        row = record({"a": 0.0, "b": 1.0})
        row.update(choice=choice, probabilities={"a": 1.0 - (choice == "b"), "b": float(choice == "b")})
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        return f"{name}={path}"
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["x", "--system", write("one", "a"), "--system", write("two", "b"),
                                     "--system", write("three", "b"), "--output-dir", str(out)])
    compare_all.main()
    summary = json.loads((out / "report.json").read_text())
    assert summary["systems"]["one"]["hard_accuracy"] == 0
    assert summary["systems"]["two"]["hard_accuracy"] == 1
    assert summary["agreement_with"]["two~three"] == 1


def test_intent_converter_builds_valid_choice_rows(monkeypatch):
    from datasets import Dataset
    from src import prepare_intent_eval
    from src.model import validate

    names = ["lost_card", "pin_change", "refund", "oos"]
    data = Dataset.from_dict({"text": ["a", "b", "c", "d"], "label_text": [names[i] for i in range(4)]})
    monkeypatch.setattr(prepare_intent_eval, "load_dataset", lambda *args, **kwargs: data)
    rows = list(prepare_intent_eval.convert("banking77", "test", 3, 0, 1))
    assert len(rows) == 3
    for row in rows:
        validate(row, labeled=True)
        assert len(row["choices"]) == 3 and sum(row["target"].values()) == 1.0
