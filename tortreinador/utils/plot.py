import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


def plot_line_2(y_1: str, y_2: str, df: pd.DataFrame, output_path: str, fig_size: tuple = (10, 6),
                dpi: int = 300):
    """
    Plot Merge Line (2 Lines) using Seaborn

        Args:

            - param y_1: Name of Line 1
            - param y_2: Name of Line 2
            - param df: Dataframe
            - param fig_size:
            - param output_path:
            - param dpi:

        return: Show Line picture and save to the specific location
    """
    fig = plt.figure(figsize=fig_size)
    sns.lineplot(x='epoch', y=y_1, data=df)
    sns.lineplot(x='epoch', y=y_2, data=df)
    plt.show()
    fig.savefig(output_path, dpi=dpi)


def calculate_GMM(p, m, s, y_label):
    """
    Calculate the probability density function of the Gaussian Mixture Model

        Args:
            - param p: pi
            - param m: mean
            - param s: standard deviation
            - param y_label: e.g. np.arange(0, 1, 0.001)

    """
    if len(y_label.shape) == 1:
        y_label = y_label.reshape(-1, 1)

    y_label_ = y_label[:, np.newaxis, np.newaxis, :]

    mu_sub_T = np.transpose(m, (0, 2, 1))
    sigma_sub_T = np.transpose(s, (0, 2, 1))

    # shape(1000, 6, 100, 10) 1000 data, 6 type, 100 rows with 10 columns every type
    exponent = np.exp(
        -1 / 2 * np.square(np.transpose((y_label_ - mu_sub_T), (1, 2, 0, 3)) / sigma_sub_T[:, :, np.newaxis, :]))
    factors = 1 / math.sqrt(2 * math.pi) / sigma_sub_T[:, :, np.newaxis, :]
    GMM_PDF = np.sum(p[:, np.newaxis, np.newaxis, :] * factors * exponent, axis=-1)
    GMM_PDF = GMM_PDF.reshape(GMM_PDF.shape[0] * GMM_PDF.shape[1], GMM_PDF.shape[-1]).transpose((-1, 0))
    # f = e.transpose((-1, 0))
    return GMM_PDF



