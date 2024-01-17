import torch


def _check_is_tensor(x):
    return isinstance(x, torch.Tensor)


def _check_in_same_device(arg1, arg2):
    if _check_is_tensor(arg1) is not True or _check_is_tensor(arg2) is not True:
        msg = 'Type different than Tensor detected, x: {}, y: {}'.format(type(arg1), type(arg2))
        raise TypeError(msg)

    return arg1.device == arg2.device


def r2_score(y_ture, y_pred):
    """
    r2 score function for the data in GPU
    :params y_ture:
    :params y_pred:
    """
    if _check_in_same_device(y_ture, y_pred) is not True:
        msg = 'Two arguments in different device'
        raise RuntimeError(msg)

    ss_tot = torch.sum(((y_ture - torch.mean(y_ture, dim=0)) ** 2), dim=0, dtype=torch.float64)
    ss_res = torch.sum(((y_ture - y_pred) ** 2), dim=0, dtype=torch.float64)
    r2 = 1 - ss_res / ss_tot

    return torch.mean(r2)

