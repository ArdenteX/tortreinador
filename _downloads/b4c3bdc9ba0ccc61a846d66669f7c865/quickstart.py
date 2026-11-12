"""Run from the repository root: python docs/examples/quickstart.py."""
import torch
from torch import nn
from tortreinador.train import TorchTrainer, config_generator
from tortreinador.utils.preprocessing import get_dataloader
from tortreinador.utils.Recorder import MetricDefine, MetricManager


class RegressionTrainer(TorchTrainer):
    def calculate(self, x, y, mode=1):
        prediction = self.model(x)
        loss = self.criterion(prediction, y)
        return self._standard_return(mode=mode, update_values=[loss])


def main():
    torch.manual_seed(42)
    x = torch.linspace(-1, 1, 160).reshape(-1, 1)
    y = 2 * x + 1
    indices = torch.randperm(len(x))
    train_idx, val_idx = indices[:128], indices[128:]
    train = get_dataloader(x[train_idx], y[train_idx], 16, True, num_workers=0)
    val = get_dataloader(x[val_idx], y[val_idx], 16, False, num_workers=0)
    model = nn.Linear(1, 1)
    metrics = MetricManager([
        MetricDefine('train_loss', metric_mode=1, use_as_criterion=True),
        MetricDefine('val_loss', metric_mode=2, use_as_valloss=True, use_for_log=True),
    ])
    trainer = RegressionTrainer(is_gpu=False, epoch=20, model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=nn.MSELoss(), metric_manager=metrics)
    config = config_generator(best_loss=float('inf'))
    history = trainer.fit(train, val, tqdm_disable=True, **config)
    validation_losses = history[0].recorder_for_epoch['val_loss'].val
    print(f'Final validation MSE: {validation_losses[-1].item():.6f}')
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor([[0.5]]))
    print(f'Prediction for x=0.5: {prediction.item():.4f}')
    return validation_losses, prediction


if __name__ == '__main__':
    main()
