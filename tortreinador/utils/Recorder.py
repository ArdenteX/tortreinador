import torch


def _check_type(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x.dtype == y.dtype

    else:
        return False


class Recorder:
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.val = torch.tensor(0.0).to(self.device)
        self.count = 0
        self.is_check = False

    def update(self, val):
        if not self.is_check:
            if not _check_type(self.val, val):
                self.val = self.val.to(val.dtype)
                self.is_check = True

            else:
                self.is_check = True

        self.val += val
        self.count += 1

    def avg(self):
        if self.count == 0:
            return torch.tensor(0.0).to(self.device)

        return (self.val / self.count).to(self.device)

    def reset(self):
        self.val = torch.tensor(0.0).to(self.device)
        self.count = 0


class RecorderForEpoch:
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.val = torch.Tensor([]).to(self.device)
        self.is_check = False

    def update(self, val):
        if not self.is_check:
            if not _check_type(self.val, val):
                self.val = self.val.to(val.dtype)
                self.is_check = True

            else:
                self.is_check = True

        val = val.unsqueeze(0)
        self.val = torch.cat((self.val, val), 0)

    def avg(self):
        return torch.mean(self.val.mean(), dim=0).unsqueeze(0)

    def reset(self):
        self.val = torch.Tensor([]).to(self.device)


class CheckpointRecorder:
    """
    Recording training information including 'total epoch', 'current epoch', 'optimizer parameters', 'best model(path)',
    'current model(path), 'config', 'log_dir' '
    """
    def __init__(self, checkpoint=None, total_epoch: int = None, config: dir = None, log_dir: str = None, mode: str = 'new', system: str = None):
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
        # Initial PTH file
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
        self.checkpoint['config']['start_epoch'] = self.checkpoint['current_epoch']
        optimizer.load_state_dict(self.checkpoint['optimizer'])
        try:
            model.load_state_dict(torch.load(self.checkpoint['best_model']))

        except KeyError:
            model.load_state_dict(torch.load(self.checkpoint['current_model']))

    def update(self, current_epoch, model, current_optimizer_sd, mode: str = 'current'):
        self.checkpoint['current_epoch'] = current_epoch
        self.checkpoint['optimizer'] = current_optimizer_sd
        model_save_path = self.checkpoint['config']['m_p'] + '{}_{}_model.pth'.format(self.file_time, mode)
        torch.save(model, model_save_path)
        self.checkpoint['{}_model'.format(mode)] = model_save_path
        self._save_pth()

    def update_by_condition(self, condition, b_m, b_l):
        if condition == 0:
            self.checkpoint['config']['b_m'] = b_m

        if condition == 1:
            self.checkpoint['config']['b_l'] = b_l

        if condition == 2:
            self.checkpoint['config']['b_m'] = b_m
            self.checkpoint['config']['b_l'] = b_l

    def _save_pth(self):
        torch.save(self.checkpoint, self.checkpoint_path)



