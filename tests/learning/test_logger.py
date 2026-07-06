import os
import sys
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.learning.logger.train_logs import TrainingLogger


def test_save_checkpoint():
    print('=== test_save_checkpoint ===')
    with tempfile.TemporaryDirectory() as tmpdir:
        model = nn.Linear(8, 4)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        logger = TrainingLogger(log_dir=tmpdir)

        logger.save_checkpoint(model, optimizer, step=1)
        checkpoint_path = os.path.join(tmpdir, 'checkpoints', 'step_1.pt')
        print('checkpoint path:', checkpoint_path)

        assert os.path.exists(checkpoint_path), 'Checkpoint file was not created.'
        checkpoint = torch.load(checkpoint_path)
        assert 'model_state' in checkpoint
        assert 'optimizer' in checkpoint


if __name__ == '__main__':
    test_save_checkpoint()
    print('test_logger.py completed successfully.')
