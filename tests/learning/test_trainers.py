import os
import sys
import types
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Create a fake src.losses module so the trainer import resolves.
import src.learning.losses.losses as lossmod
mock_losses = types.ModuleType('src.losses')
mock_losses.combined_surface_loss_torch = lossmod.combined_surface_loss
mock_losses.kl_divergence_loss_torch = lossmod.kl_divergence_loss
sys.modules['src.losses'] = mock_losses

orig_cuda = nn.Module.cuda
nn.Module.cuda = lambda self: self
try:
    from src.learning.trainers.E3_end2end import TrainingStepper, TrainingOrchestrator
finally:
    nn.Module.cuda = orig_cuda


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, graph):
        batch_size = 1
        mu = torch.zeros(batch_size, 2)
        logvar = torch.zeros(batch_size, 2)
        R = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        t = torch.zeros(batch_size, 3)
        return mu, logvar, R, t


class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inv):
        return torch.zeros(1, 4, 3, dtype=torch.float32)


class DummyLogger:
    def __init__(self):
        self.metrics = []
        self.saved_steps = []
        self.visualized_steps = []

    def log_metrics(self, metrics, step):
        print(f'Logging metrics at step {step}:', metrics)
        self.metrics.append((step, metrics))

    def save_checkpoint(self, state, step):
        print(f'Saving checkpoint at step {step} with state type {type(state)}')
        self.saved_steps.append(step)

    def visualize_results(self, aux, step):
        print(f'Visualizing results at step {step} with aux {aux}')
        self.visualized_steps.append(step)


class DummyDataloader:
    def __init__(self):
        self.calls = 0

    def get_next(self):
        self.calls += 1
        return None, torch.zeros(1, 4, 3, dtype=torch.float32), torch.ones(1, 4, dtype=torch.bool)


def test_training_stepper():
    print('=== test_training_stepper ===')
    encoder = DummyEncoder()
    decoder = DummyDecoder()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-4)

    loss, output = stepper.train_step(None, torch.zeros(1, 4, 3, dtype=torch.float32), torch.ones(1, 4, dtype=torch.bool))
    print('TrainingStepper loss:', loss)
    print('TrainingStepper output shape:', output.shape)

    assert isinstance(loss, float)
    assert output.shape == (1, 4, 3)


def test_training_orchestrator():
    print('=== test_training_orchestrator ===')
    from types import SimpleNamespace

    dummy_stepper = SimpleNamespace(train_step=lambda batch: ('state', 0.123, {'example': True}))
    logger = DummyLogger()
    dataloader = DummyDataloader()

    orchestrator = TrainingOrchestrator(stepper=dummy_stepper, logger=logger, dataloader=dataloader)
    orchestrator.run(num_steps=3, log_every=1, save_every=2)

    assert logger.metrics[0][0] == 0
    assert logger.metrics[1][0] == 1
    assert logger.saved_steps == [0, 2]
    assert logger.visualized_steps == [0, 2]


if __name__ == '__main__':
    test_training_stepper()
    test_training_orchestrator()
    print('test_trainers.py completed successfully.')
