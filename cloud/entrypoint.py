"""AWS PyTorch DLC script-mode entrypoint; run configuration comes from the launcher."""
import json
import os
import sys
from pathlib import Path

import torch

from src.model import ChoiceModel, iter_data
from src.train import main, evaluate


def run():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; refusing an unexpectedly slow CPU training job')
    os.environ['HF_HUB_OFFLINE'] = '1'
    options = json.loads(os.environ['CHOICEFIT_CONFIG'])
    argv = ['train', '--model', os.environ['SM_CHANNEL_BASE'], '--device', 'cuda',
            '--checkpoint-dir', '/opt/ml/checkpoints']
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                argv.append('--' + key)
        else:
            argv.extend(['--' + key, str(value)])
    sys.argv = argv
    main()
    # Exercise the exported model path before marking the job successful.
    model = ChoiceModel(os.environ['SM_MODEL_DIR'], device='cuda')
    test = Path(os.environ['SM_CHANNEL_TEST']) / 'test.jsonl'
    metrics = evaluate(model, iter_data(test), batch_size=1, pair_batch_size=2)
    (Path(os.environ['SM_MODEL_DIR']) / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({'reloaded_checkpoint_test': metrics}), flush=True)


if __name__ == '__main__':
    run()
