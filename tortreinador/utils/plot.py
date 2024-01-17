import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_line_2(y_1: str, y_2: str, df: pd.DataFrame, output_path: str, fig_size: tuple = (10, 6),
                dpi: int = 300):
    """
    Plot Merge Line (2 Lines) using Seaborn
    :param y_1: Name of Line 1
    :param y_2: Name of Line 2
    :param df: Dataframe
    :param fig_size:
    :param output_path:
    :param dpi:
    :return: Show Line picture and save to the specific location
    """
    fig = plt.figure(figsize=fig_size)
    sns.lineplot(x='epoch', y=y_1, data=df)
    sns.lineplot(x='epoch', y=y_2, data=df)
    plt.show()
    fig.savefig(output_path, dpi=dpi)


