import copy

import pytest
import torch

from src.model import grouped_logits, validate
from src.prepare_typed_decisions import convert_choice, sample_split
from src.train import choice_loss, evaluate, targets


def raw(group="g", source="task", target=None, state="message"):
    return {"kind": "choice", "state": state, "question": "Which description fits?",
            "options": ["Description A", "Description B"], "target": target or [0.25, 0.75],
            "source": source, "group_id": group, "id": group, "variant": "direct"}


def test_conversion_preserves_probabilities_and_option_order():
    row = convert_choice(raw())
    assert row["target"] == {"option_0": 0.25, "option_1": 0.75}
    assert "label" not in row
    reordered = copy.deepcopy(row)
    reordered["choices"] = dict(reversed(list(row["choices"].items())))
    assert targets([reordered], "cpu").tolist() == [[0.75, 0.25]]


@pytest.mark.parametrize("probabilities", [[0.2, 0.2], [-0.1, 1.1], [float("nan"), 1], [float("inf"), 0], [True, 0]])
def test_invalid_target_rejected(probabilities):
    with pytest.raises(ValueError):
        convert_choice(raw(target=probabilities))


def test_soft_loss_with_padding_matches_independent_examples_and_gradients():
    rows = [convert_choice(raw()), {"message": "another", "choices": {"a": "A", "b": "B", "c": "C"}, "label": "b"}]
    scores = torch.tensor([1., 2., 3., 4., 5.], requires_grad=True)
    logits = grouped_logits(scores, [2, 3])
    loss = choice_loss(logits, targets(rows, "cpu")).sum()
    expected = -(torch.tensor([0.25, 0.75]) * scores[:2].log_softmax(-1)).sum() - scores[2:].log_softmax(-1)[1]
    assert torch.allclose(loss, expected)
    expected_grad = torch.autograd.grad(expected, scores, retain_graph=True)[0]
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.allclose(scores.grad, expected_grad)


def test_soft_metrics_are_not_reported_as_hard_accuracy():
    class Fake:
        device = "cpu"
        model = torch.nn.Linear(1, 1)
        def logits(self, rows, pair_batch_size=None):
            return torch.tensor([[0., 1.] for _ in rows])
    result = evaluate(Fake(), [convert_choice(raw())], 1)
    assert result["accuracy"] is None
    assert result["hard_examples"] == 0
    assert result["soft_examples"] == 1
    assert result["mean_selected_target_probability"] == 0.75
    assert result["top_target_accuracy"] == 1


def test_sampling_filters_whole_groups_and_keeps_source_diversity():
    def tokenizer(contexts, candidates, **kwargs):
        return {"input_ids": [[1] * (100 if "long" in text else 4) for text in contexts]}
    rows = [raw("blocked"), raw("a", "first"), raw("a", "first"),
            raw("b", "second", state="second"), raw("long", state="long"),
            {**raw("noul"), "kind": "noul"}, raw("foreign", "multilingual/example")]
    selected, stats = sample_split(rows, tokenizer, limit=3, scan_limit=100,
                                   max_choices=4, max_length=16, seed=42,
                                   excluded_groups={"blocked"})
    assert len(selected) == 3
    assert {r["source"] for r in selected} == {"first", "second"}
    assert sum(r["group_id"] == "a" for r in selected) == 2
    assert stats["too_long"] == 1
    assert stats["not_choice"] == 1
    assert stats["multilingual_source"] == 1


def test_group_audit_catches_reformatted_state(tmp_path):
    import json
    from src.data import audit_splits
    train = tmp_path / 'train.jsonl'
    validation = tmp_path / 'validation.jsonl'
    train.write_text(json.dumps(convert_choice(raw(state='first format'))) + '\n')
    validation.write_text(json.dumps(convert_choice(raw(state='second format'))) + '\n')
    with pytest.raises(ValueError, match='group_id'):
        audit_splits(train, validation, max_pairs=4)


def test_prediction_export_preserves_description_and_target():
    import io
    import json
    class Fake:
        device = 'cpu'
        model = torch.nn.Linear(1, 1)
        def logits(self, rows, pair_batch_size=None):
            return torch.tensor([[0., 1.] for _ in rows])
    output = io.StringIO()
    evaluate(Fake(), [convert_choice(raw())], 1, prediction_sink=output)
    result = json.loads(output.getvalue())
    assert result['choice'] == 'option_1'
    assert result['description'] == 'Description B'
    assert result['target']['option_1'] == 0.75
