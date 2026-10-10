"""Rotating recovery checkpoints for trusted, private training runs."""
import hashlib
import os
import pickle
import random
from pathlib import Path

import torch


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rng_state():
    result = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        result['mps'] = torch.mps.get_rng_state()
    return result


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if 'mps' in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])


def save_recovery(directory, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    previous = sorted(directory.glob('recovery-*.pt'))
    sequence = int(previous[-1].stem.split('-')[1]) + 1 if previous else 1
    destination = directory / f'recovery-{sequence:09d}.pt'
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as output:
        torch.save(payload, output)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(destination)
    # Keep the preceding complete checkpoint if an upload or new write fails.
    for old in previous[:-1]:
        old.unlink()
    return destination


def load_recovery(directory):
    if not directory:
        return None
    candidates = sorted(Path(directory).glob('recovery-*.pt'), reverse=True)
    for path in candidates:
        try:
            state = torch.load(path, map_location='cpu', weights_only=True)
            if state.get('version') != 1:
                raise ValueError('Unsupported recovery checkpoint version')
            return state
        except (OSError, RuntimeError, EOFError, ValueError, IndexError, pickle.UnpicklingError) as exc:
            print(f'Cannot read recovery checkpoint {path.name}: {exc}', flush=True)
    if candidates:
        raise ValueError('No readable recovery checkpoint; refusing to restart silently')
    return None
