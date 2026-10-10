import copy
import json

import pytest
import torch

from src.checkpoint import load_recovery, restore_rng, rng_state, save_recovery
from src.data import buffered_shuffle
from src.train import train_epoch


class Tiny:
    device = 'cpu'
    def __init__(self):
        self.model = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 2))
    def logits(self, rows):
        return self.model(torch.tensor([[float(r['message']) / 10] for r in rows]))


def test_interrupted_adam_and_dropout_match_uninterrupted(tmp_path):
    torch.manual_seed(7)
    full, interrupted = Tiny(), Tiny()
    interrupted.model.load_state_dict(copy.deepcopy(full.model.state_dict()))
    rows = [{'message': str(i), 'choices': {'a': 'A', 'b': 'B'}, 'target': {'a': .25, 'b': .75}} for i in range(11)]
    full_opt = torch.optim.AdamW(full.model.parameters(), lr=.01)
    partial_opt = torch.optim.AdamW(interrupted.model.parameters(), lr=.01)
    torch.manual_seed(99)
    expected = train_epoch(full, full_opt, buffered_shuffle(rows, 4, 42), 2, 4, 2, 0)
    def interrupt(progress):
        save_recovery(tmp_path, {'version': 1, 'model': interrupted.model.state_dict(),
                                'optimizer': partial_opt.state_dict(), 'progress': progress, 'rng': rng_state()})
        raise InterruptedError('simulated Spot interruption')
    torch.manual_seed(99)
    with pytest.raises(InterruptedError):
        train_epoch(interrupted, partial_opt, buffered_shuffle(rows, 4, 42), 2, 4, 2, 0,
                    checkpoint_every=1, checkpoint_callback=interrupt)
    state = load_recovery(tmp_path)
    restored = Tiny()
    restored.model.load_state_dict(state['model'])
    restored_opt = torch.optim.AdamW(restored.model.parameters(), lr=.01)
    restored_opt.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    actual = train_epoch(restored, restored_opt, buffered_shuffle(rows, 4, 42), 2, 4, 2, 0, progress=state['progress'])
    assert actual['updates'] == expected['updates'] == 3
    assert actual['examples'] == 11
    assert actual['train_loss'] == pytest.approx(expected['train_loss'])
    for left, right in zip(full.model.parameters(), restored.model.parameters()):
        assert torch.equal(left, right)
    assert restored_opt.state_dict()['state'][0]['step'] == 3


def test_corrupt_latest_checkpoint_falls_back(tmp_path):
    first = save_recovery(tmp_path, {'version': 1, 'value': 1})
    second = save_recovery(tmp_path, {'version': 1, 'value': 2})
    second.write_bytes(b'truncated upload')
    assert load_recovery(tmp_path)['value'] == 1
    assert first.exists()
