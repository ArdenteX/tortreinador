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




