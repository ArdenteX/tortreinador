import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score
from sklearn.metrics import r2_score as r2_
from torch.distributions import Categorical, Normal
import numpy as np


def _check_is_tensor(x):
    return isinstance(x, torch.Tensor)


def _check_in_same_device(arg1, arg2):
    if _check_is_tensor(arg1) is not True or _check_is_tensor(arg2) is not True:
        msg = 'Type different than Tensor detected, x: {}, y: {}'.format(type(arg1), type(arg2))
        raise TypeError(msg)

    return arg1.device == arg2.device


def r2_score(y_true, y_pred):
    """
    r2 score function for the data in GPU

        Args:
            - y_ture: true
            - y_pred: predicted
    """
    if _check_in_same_device(y_true, y_pred) is not True:
        msg = 'Two arguments in different device'
        raise RuntimeError(msg)

    ss_tot = torch.sum(((y_true - torch.mean(y_true, dim=0)) ** 2), dim=0, dtype=torch.float64)
    ss_res = torch.sum(((y_true - y_pred) ** 2), dim=0, dtype=torch.float64)
    r2 = 1 - ss_res / ss_tot

    return torch.mean(r2)


def mixture(pi, mu, sigma, sample_for='train'):
    cat = Categorical(logits=pi)

    select_idx = cat.sample()

    # Advance Indexing
    mu_selected = mu[torch.arange(mu.shape[0]), select_idx, :]
    sigma_selected = sigma[torch.arange(sigma.shape[0]), select_idx, :]

    pdf = Normal(loc=mu_selected, scale=sigma_selected)

    if sample_for == 'train':
        return pdf

    else:
        return pdf, select_idx, mu_selected, sigma_selected

def evaluation(y_true, y_pred):
    r2 = r2_(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    rmse = mean_squared_error(y_true, y_pred, squared=False)

    medae = median_absolute_error(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    explained_variance = explained_variance_score(y_true, y_pred)

    print(f"R-Square: {r2}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"MedAE: {medae}")
    # print(f"Max Error: {max_err}")
    print(f"MAPE: {mape}")
    print(f"Explained Variance: {explained_variance}")
    # print(f"MSLE: {msle}")

    return r2, mse, mae, rmse, medae, mape, explained_variance