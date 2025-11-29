import os.path

import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Dict, Literal
import numpy as np

def _check_type(x, y):
    """Ensure two tensors share dtype before aggregation."""
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x.dtype == y.dtype

    else:
        return False


class Recorder:
    def __init__(self, device):
        """Track running sums for a metric on the specified device."""
        super().__init__()
        self.device = device
        self.val = torch.tensor(0.0).to(self.device)
        self.count = 0
        self.is_check = False

    def update(self, val):
        """Accumulate a metric value and keep dtype consistent after first update."""
        if not self.is_check:
            if not _check_type(self.val, val):
                self.val = self.val.to(val.dtype)
                self.is_check = True

            else:
                self.is_check = True

        self.val += val
        self.count += 1

    def avg(self):
        """Return average value or zero tensor when no samples are recorded."""
        if self.count == 0:
            return torch.tensor(0.0).to(self.device)

        return (self.val / self.count).to(self.device)

    def reset(self):
        """Clear stored values while keeping device placement."""
        self.val = torch.tensor(0.0).to(self.device)
        self.count = 0


class RecorderForEpoch:
    def __init__(self, device):
        """Store metric values for each epoch to retain per-epoch history."""
        super().__init__()
        self.device = device
        self.val = torch.Tensor([]).to(self.device)
        self.is_check = False

    def update(self, val):
        """Append a new value to the history, aligning dtype on first use."""
        if not self.is_check:
            if not _check_type(self.val, val):
                self.val = self.val.to(val.dtype)
                self.is_check = True

            else:
                self.is_check = True

        val = val.unsqueeze(0)
        self.val = torch.cat((self.val, val), 0)

    def avg(self):
        """Return the mean across all recorded epochs."""
        return torch.mean(self.val.mean(), dim=0).unsqueeze(0)

    def reset(self):
        """Reset history to an empty tensor on the same device."""
        self.val = torch.Tensor([]).to(self.device)


class CheckpointRecorder:
    """
    Recording training information including 'total epoch', 'current epoch', 'optimizer parameters', 'best model(path)',
    'current model(path), 'config', 'log_dir' '
    """
    def __init__(self, checkpoint=None, total_epoch: int = None, config: dir = None, log_dir: str = None, mode: str = 'new', system: str = None):
        """
        Create a checkpoint recorder for persisting model and optimizer state.

        Args:
            checkpoint: Path to save checkpoint (mode='new') or loaded checkpoint dict (mode='reload').
            total_epoch (int): Training epochs used to seed the checkpoint metadata.
            config (dict): Training configuration stored alongside state dicts.
            log_dir (str): Location of csv logger when using csv mode.
            mode (str): 'new' to create a fresh checkpoint, 'reload' to resume.
            system (str): Name of current OS used for path handling.
        """
        # Two modes, initial checkpoint or load checkpoint
        mode_list = ['new', 'reload']
        if mode not in mode_list:
            raise ValueError("Please specify mode to 'new' or 'reload'")

        self.checkpoint_path = None
        self.checkpoint = None

        if mode == 'new':
            self.checkpoint_path = checkpoint
            self.initial(self.checkpoint_path, total_epoch, config, log_dir)
        else:
            self.checkpoint = checkpoint
            self.checkpoint_path = self.checkpoint['config']['checkpoint_path']

        self.file_time = self.checkpoint_path.split('/')[-1].split('_')[-1].split('.')[0]

    def initial(self, checkpoint_path, total_epoch, config, log_dir):
        """Initialize checkpoint content on disk."""
        config['checkpoint_path'] = checkpoint_path
        config['train_mode'] = 'reload'
        config['log_dir'] = log_dir
        torch.save({
            'total_epoch': total_epoch,
            'current_epoch': 0,
            'config': config
        }, checkpoint_path)
        self.checkpoint = torch.load(checkpoint_path)

    def reload(self, model, optimizer):
        """Load model and optimizer state from an existing checkpoint record."""
        self.checkpoint['config']['start_epoch'] = self.checkpoint['current_epoch']
        optimizer.load_state_dict(self.checkpoint['optimizer'])
        try:
            model.load_state_dict(torch.load(self.checkpoint['best_model']))

        except KeyError:
            model.load_state_dict(torch.load(self.checkpoint['current_model']))

    def update(self, current_epoch, model, current_optimizer_sd, mode: str = 'current'):
        """Persist current model/optimizer states and training progress to disk."""
        self.checkpoint['current_epoch'] = current_epoch
        # model_save_path = self.checkpoint['config']['m_p'] + '{}_{}_model.pth'.format(self.file_time, mode)
        # optimizer_save_path = self.checkpoint['config']['m_p'] + '{}_{}_optimizer.pth'.format(self.file_time, mode)

        model_save_path = os.path.join( self.checkpoint['config']['m_p'], '{}_{}_model.pth'.format(self.file_time, mode))
        optimizer_save_path = os.path.join( self.checkpoint['config']['m_p'], '{}_{}_optimizer.pth'.format(self.file_time, mode))

        torch.save(model, model_save_path)
        torch.save(current_optimizer_sd, optimizer_save_path)

        self.checkpoint['{}_model'.format(mode)] = model_save_path
        self.checkpoint['{}_optimizer'.format(mode)] = optimizer_save_path

        self._save_pth()

    def update_by_condition(self, condition, b_m, b_l):
        """Update tracked best metric/loss values according to training condition."""
        if condition == 0:
            self.checkpoint['config']['b_m'] = b_m

        if condition == 1:
            self.checkpoint['config']['b_l'] = b_l

        if condition == 2:
            self.checkpoint['config']['b_m'] = b_m
            self.checkpoint['config']['b_l'] = b_l

    def _save_pth(self):
        """Write the checkpoint dictionary to the configured path."""
        torch.save(self.checkpoint, self.checkpoint_path)


@dataclass
class MetricDefine:
    """
    Define the metrics that display on tqdm
    Attributes
        metric_name: Name of the metric, default as 'Unknown Metric'
        metric_value: Value of the metric
        metric_mode:
            - 0: Display this metric in All the stage (Training and Validation)
            - 1: Display this metric in Training
            - 2: Display this metric in Validation
    """
    metric_name: str = 'Unknown Metric'
    metric_value: Union[torch.Tensor, None] = torch.tensor(0.0)
    metric_mode: Literal[0, 1, 2, None] = None
    use_as_baseline: bool = False
    use_as_criterion: bool = False

    def update(self, v):
        self.metric_value = v


class MetricManager:

    def __init__(self, metrics: List[
        MetricDefine
    ]):
        """Register metric definitions and expose helpers for training/validation loops."""
        self.metric_list = []
        self.metric_names = []
        self.baseline_metric_idx = None
        self.criterion_idx = None

        for idx, m in enumerate(metrics):
            self.metric_list.append(m)
            self.metric_names.append(m.metric_name)

            if m.use_as_baseline:
                self.baseline_metric_idx = idx

            if m.use_as_criterion:
                self.criterion_idx = idx

        self.metric_list = np.array(self.metric_list)
        self.metric_names = np.array(self.metric_names)

        self._check_criterion_exist()

    def _check_criterion_exist(self):
        """Ensure at least one metric is marked as criterion for model selection."""
        if self.criterion_idx is None:
            raise ValueError('It seems that none of the registered metrics used as criterion, it will cause the training to fail')

    def get_metrics_by_mode(self, mode: int = 0, idx: bool = False):
        """
        Retrieve metrics filtered by their mode.

        Args:
            mode (int): 0 for both train/val, 1 for train-only, 2 for val-only.
            idx (bool): Return indices instead of MetricDefine objects when True.
        """
        metrics_by_mode = []
        metrics_idx = []
        for m_idx in range(len(self.metric_list)):
            current_metric = self.metric_list[m_idx]
            if current_metric.metric_mode == mode or current_metric.metric_mode == 0:
                if not idx:
                    metrics_by_mode.append(current_metric)

                if idx:
                    metrics_idx.append(m_idx)
                # yield m_idx
        if idx:
            return metrics_idx

        else:
            return metrics_by_mode

    def check_metric_exist(self, name):
        """Return True if a metric with the given name substring is registered."""
        for n in self.metric_list:
            if name.lower() in n.metric_name.lower():
                return True

    def get_metrics_by_name(self, name, idx: bool = False):
        """Fetch metrics whose names contain the provided substring."""
        metrics_by_name = []
        idx_by_name = []
        for m_idx in range(len(self.metric_list)):
            current_metric = self.metric_list[m_idx]
            if name.lower() in current_metric.metric_name.lower():
                if not idx:
                    metrics_by_name.append(current_metric)

                if idx:
                    idx_by_name.append(m_idx)

        if not idx:
            return metrics_by_name

        elif idx:
            return idx_by_name

    def update(self, update_pair: List[torch.Tensor] = None, mode: int = None):
        """Update all metrics belonging to a particular phase with fresh values."""
        current_metrics = self.get_metrics_by_mode(mode)
        for n_v, update_v in zip(current_metrics, update_pair):
            n_v.update(update_v)

