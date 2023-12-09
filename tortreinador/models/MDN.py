import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical, Normal


class mdn(nn.Module):
    def __init__(self, input_size, output_size, num_gaussian, num_hidden):
        super(mdn, self).__init__()
        self.i_s = input_size
        self.o_s = output_size
        self.n_g = num_gaussian
        self.n_h = num_hidden

        self.root_layer = nn.Sequential(
            # nn.BatchNorm1d(self.i_s),
            nn.Linear(self.i_s, self.n_h),
            nn.SiLU(),
            # nn.Dropout(),
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            # nn.Dropout(),
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU()
        ).double()

        self.pi = nn.Sequential(
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            nn.Linear(self.n_h, self.n_g)
        ).double()

        self.mu = nn.Sequential(
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            nn.Linear(self.n_h, self.o_s * self.n_g)
        ).double()

        self.sigma = nn.Sequential(
            nn.Linear(self.n_h, self.n_h),
            nn.ELU(),
            nn.Linear(self.n_h, self.o_s * self.n_g)
        ).double()

    def forward(self, x, eps=1e-6):
        parameters = self.root_layer(x).double()

        pi = torch.log_softmax(self.pi(parameters), dim=-1)

        mu = self.mu(parameters)

        sigma = self.sigma(parameters)
        sigma = torch.exp(sigma + eps)

        return pi, mu.reshape(-1, self.n_g, self.o_s), sigma.reshape(-1, self.n_g, self.o_s)


class NLLLoss(nn.Module):
    """
        Implementation of NLLLoss using probability density function

        Using probability density function to calculate the NLLLoss will more suitable for regression task, it will not
        break the straight relationship between loss function and weight of model. Besides, the implementation method of
        firstly sampling than calculate the loss function will increase the uncertainly and break the straight relationship.
    """
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, pi, mu, sigma, y):
        z_score = (torch.unsqueeze(y, dim=1) - mu) / sigma

        normal_loglik = (-0.5 * torch.einsum("bij, bij->bi", z_score, z_score)) - torch.sum(torch.log(sigma), dim=-1)

        loglik = -torch.logsumexp(pi + normal_loglik, dim=-1)

        return loglik.mean()


class Mixture(nn.Module):
    def __init__(self):
        super(Mixture, self).__init__()


    def forward(self, pi, mu, sigma, sample_for='train'):
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


class NLLLoss_Version_2(nn.Module):
    def __init__(self):
        super(NLLLoss_Version_2, self).__init__()

    def forward(self, pdf, y_true):
        log_prob = pdf.log_prob(y_true)
        log_prob = log_prob.negative()
        log_prob = log_prob.mean()

        return log_prob


class RelativeError(nn.Module):
    def __init__(self):
        super(RelativeError, self).__init__()

    def forward(self, y_ture, samples, eps=1e-6):
        relative_error = np.divide((np.abs(samples - y_ture)), np.abs(y_ture + eps)).mean()

        return relative_error






