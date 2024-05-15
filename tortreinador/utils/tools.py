import numpy as np
import torch.nn as nn
from tortreinador.utils.View import split_weights, init_weights


def check_outlier(validation_set, stride, windows_size, z_score_threshold):
    window = np.array([])
    for i in range(0, len(validation_set) - windows_size + stride, stride):
        curr_size = len(validation_set) - i
        if curr_size < windows_size:
            tmp = np.array(validation_set[i: i + curr_size])

        else:
            tmp = np.array(validation_set[i: i + windows_size])

        mu = np.mean(tmp)
        sig = np.std(tmp)
        z_score = np.abs((tmp - mu) / sig)
        indices = np.where(z_score > z_score_threshold)
        if np.any(indices):
            window = np.append(window, np.array([indices[j] + i for j in range(len(indices))]))

    window = window.astype(np.int64)
    count = {}
    for i in range(len(window)):
        if window[i] not in count.keys():
            count[window[i]] = 1

        else:
            count[window[i]] += 1

    outlier = [key for key, value in count.items() if value > 1]

    return outlier


def xavier_init(net: nn.Module):
    return split_weights(init_weights(net))

