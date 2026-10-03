import pytest
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification
from src.model import validate, grouped_logits, ChoiceModel


def test_variable_choices_loss():
    scores = torch.tensor([1., 3., 4., 2., 0.], requires_grad=True)
    logits = grouped_logits(scores, [2, 3])
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0]))
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert logits.argmax(-1).tolist() == [1, 0]
    assert logits.softmax(-1)[0, 2] == 0


@pytest.mark.parametrize('choices', [{}, {'a':'one'}, {'a':'', 'b':'two'}])
def test_invalid_choices(choices):
    with pytest.raises(ValueError):
        validate({'message':'hello','choices':choices})


def test_missing_target():
    with pytest.raises(ValueError):
        validate({'message':'hello','choices':{'a':'one','b':'two'},'label':'c'}, True)


def test_real_encoder_gradient_and_roundtrip(tmp_path):
    config = RobertaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                          num_attention_heads=2, intermediate_size=32, num_labels=1)
    model = RobertaForSequenceClassification(config)
    inputs = torch.randint(3,32,(5,8))
    logits = grouped_logits(model(inputs).logits,[2,3])
    torch.nn.functional.cross_entropy(logits,torch.tensor([1,2])).backward()
    assert model.roberta.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    model.eval()
    model.save_pretrained(tmp_path)
    restored=RobertaForSequenceClassification.from_pretrained(tmp_path).eval()
    assert torch.allclose(model(inputs).logits,restored(inputs).logits)


def test_pair_encoding_uses_question_and_description_not_id():
    from src.model import encode
    captured = {}
    def tokenizer(contexts, candidates, **kwargs):
        captured.update(contexts=contexts, candidates=candidates)
        return {}
    encode(tokenizer, [{'message':'charged twice', 'question':'Which team?',
                        'choices':{'opaque_1':'Payments', 'opaque_2':'Benefits'}}], 128)
    assert captured['contexts'] == ['Question: Which team?\nMessage: charged twice'] * 2
    assert captured['candidates'] == ['Candidate description: Payments', 'Candidate description: Benefits']


def test_non_object_rejected():
    with pytest.raises(ValueError, match='JSON object'):
        validate([])
