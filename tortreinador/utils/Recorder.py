import torch


class Recorder:
    def __init__(self, device):
        """
        Metrics Recorde e.g. loss and accuracy

            Args:
                - device (string): Device used e.g. 'cpu' or 'cuda'

            Variables:
                - self.val: Pre-created empty tensor, used to concat the new tensor
                - self.is_check: This is used to avoid the Runtime error like "expected scalar type double but found float"
                    - 0: Waiting for check
                    - 1: Dtype of input data is different with 'self.val'
                    - 2: Dtype of input data is the same as 'self.val'
        """
        super().__init__()
        self.device = device
        self.val = torch.Tensor([]).to(self.device)
        self.is_check = 0

    def update(self, val):
        if self.is_check == 0:
            if not _check_type(self.val, val):
                self.val.to(val.dtype)
                self.is_check = 1

            else:
                self.is_check = 2

        elif self.is_check == 1:
            self.val.to(val.dtype)

        val = val.unsqueeze(0)
        self.val = torch.cat((self.val, val), 0)

    def avg(self):
        return torch.mean(self.val.mean(), dim=0).unsqueeze(0)

    def reset(self):
        self.val = torch.Tensor([]).to(self.device)


def _check_type(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x.dtype == y.dtype

    else:
        return False

