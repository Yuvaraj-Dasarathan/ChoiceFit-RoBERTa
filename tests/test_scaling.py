import copy
import json

import pytest
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification

from src.data import audit_splits, batches, buffered_shuffle, message_key
from src.model import ChoiceModel, iter_data
from src.train import train_epoch


def row(message, size=2):
    return {"message": message, "choices": {str(i): f"description {i}" for i in range(size)}, "label": "0"}


def test_streaming_validation_and_overlap(tmp_path):
    train, validation = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train.write_text(json.dumps(row("Hello world")) + "\n")
    validation.write_text(json.dumps(row("  HELLO   world ")) + "\n")
    with pytest.raises(ValueError, match="overlap"):
        audit_splits(train, validation, 4)
    validation.write_text(json.dumps(row("Different")) + "\n")
    assert audit_splits(train, validation, 4) == {"train": 1, "validation": 1}
    train.write_text(json.dumps(row("valid")) + "\ninvalid JSON\n")
    stream = iter_data(train)
    assert next(stream)["message"] == "valid"
    with pytest.raises(ValueError, match="train.jsonl:2"):
        next(stream)


def test_shuffle_and_pair_budgets():
    rows = [row(str(i), size) for i, size in enumerate([2, 3, 4, 2, 2])]
    shuffled = list(buffered_shuffle(iter(rows), 3, 42))
    assert shuffled == list(buffered_shuffle(iter(rows), 3, 42))
    assert sorted(r["message"] for r in shuffled) == ["0", "1", "2", "3", "4"]
    grouped = list(batches(rows, batch_size=3, max_pairs=5))
    assert [len(b) for b in grouped] == [2, 1, 2]
    with pytest.raises(ValueError, match="single example"):
        list(batches([row("too big", 6)], 4, 5))


def test_chunked_real_encoder_matches_unchunked():
    torch.manual_seed(42)
    wrapper = ChoiceModel.__new__(ChoiceModel)
    wrapper.device = torch.device("cpu")
    wrapper.max_length = 16
    wrapper.model = RobertaForSequenceClassification(RobertaConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, num_labels=1)).eval()
    def tokenizer(contexts, candidates, **kwargs):
        return {"input_ids": torch.tensor([[0, 4 + i, 2] for i in range(len(contexts))]),
                "attention_mask": torch.ones(len(contexts), 3, dtype=torch.long)}
    wrapper.tokenizer = tokenizer
    with torch.inference_mode():
        whole = wrapper.logits([row("one", 3), row("two", 2)])
        chunked = wrapper.logits([row("one", 3), row("two", 2)], pair_batch_size=2)
    assert torch.allclose(whole, chunked, atol=1e-6)
    assert chunked.softmax(-1)[1, 2] == 0


def test_accumulation_matches_message_weighted_updates():
    class TinyModel:
        device = torch.device("cpu")
        def __init__(self):
            self.model = torch.nn.Linear(1, 2)
        def logits(self, rows):
            return self.model(torch.tensor([[float(r["message"])] for r in rows]))
    torch.manual_seed(12)
    full = TinyModel()
    accumulated = TinyModel()
    accumulated.model.load_state_dict(copy.deepcopy(full.model.state_dict()))
    rows = [row(str(i)) for i in range(7)]
    for wrapper, batch_size, steps in ((full, 4, 1), (accumulated, 3, 2)):
        optimizer = torch.optim.SGD(wrapper.model.parameters(), lr=0.1)
        # Pair limit makes the accumulated microbatches [2,2,2,1].
        train_epoch(wrapper, optimizer, iter(rows), batch_size,
                    8 if steps == 1 else 4, steps, log_every=0)
    for expected, actual in zip(full.model.parameters(), accumulated.model.parameters()):
        assert torch.allclose(expected, actual, atol=1e-6)


def test_training_cli_saves_reloadable_checkpoint(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from src.train import main, evaluate

    backend = Tokenizer(WordLevel({"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>",
                                        pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    checkpoint = tmp_path / "initial"
    RobertaForSequenceClassification(RobertaConfig(
        vocab_size=4, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, num_labels=1, type_vocab_size=2)).save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train.write_text("".join(json.dumps(row(str(i))) + "\n" for i in range(5)))
    validation.write_text(json.dumps(row("held out")) + "\n")
    output = tmp_path / "trained"
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setattr("sys.argv", ["train", "--model", str(checkpoint), "--train", str(train),
                                    "--validation", str(validation), "--output", str(output),
                                    "--epochs", "1", "--batch-size", "2", "--max-pairs", "4",
                                    "--gradient-accumulation-steps", "2", "--gradient-checkpointing",
                                    "--max-length", "16", "--device", "cpu",
                                    "--checkpoint-dir", str(checkpoints), "--checkpoint-every", "1"])
    main()
    metrics = json.loads((output / "training_metrics.json").read_text())
    assert metrics["history"][1]["updates"] == 2
    restored = ChoiceModel(output, device="cpu")
    result = evaluate(restored, iter_data(validation), batch_size=1, pair_batch_size=1)
    assert result["examples"] == 1
    assert torch.isfinite(torch.tensor(result["loss"]))
    # Completed recovery must re-export the best model in a fresh output dir.
    original_weights = {k: v.clone() for k, v in restored.model.state_dict().items()}
    import sys
    recovered_output = tmp_path / "recovered"
    sys.argv[sys.argv.index("--output") + 1] = str(recovered_output)
    main()
    recovered = ChoiceModel(recovered_output, device="cpu")
    assert all(torch.equal(value, recovered.model.state_dict()[key]) for key, value in original_weights.items())
    assert len(json.loads((recovered_output / "training_metrics.json").read_text())["history"]) == 2
