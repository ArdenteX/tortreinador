import torch


class Recorder:
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.val = torch.Tensor([]).to(self.device)

    def update(self, val):
        val = val.unsqueeze(0)
        self.val = torch.cat((self.val, val), 0)

    def avg(self):
        return torch.mean(self.val.mean(), dim=0).unsqueeze(0)

    def reset(self):
        self.val = torch.Tensor([]).to(self.device)