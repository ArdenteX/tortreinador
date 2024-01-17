from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(data: pd.DataFrame, input_parameters: list, output_parameters: list,
              feature_range=None, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1,
              if_normal: bool = True, if_shuffle: bool = True, n_workers: int = 8, batch_size: int = 256):
    """
    Load Data and Normalization
    :return:
        Train DataLoader, Validation DataLoader, Test X, Test Y, Scaler X, Scaler Y
    """
    if (train_size + test_size + val_size) != 1:
        raise ValueError("train_size + test_size + val_size must equals 1")

    # Data Normalization
    if feature_range is None:
        feature_range = [0, 1]
    scaler_x = MinMaxScaler(feature_range=feature_range)
    scaler_y = MinMaxScaler(feature_range=feature_range)

    train_x = None
    train_y = None
    val_x = None
    val_y = None
    test_x = None
    test_y = None

    data_x = eval("data.{}[:, input_parameters]".format('iloc' if type(input_parameters[0]) == int else 'loc'))
    data_y = eval("data.{}[:, output_parameters]".format('iloc' if type(output_parameters[0]) == int else 'loc'))

    v_t_sum = (val_size + test_size)

    val_rate = val_size / v_t_sum

    if if_normal:
        data_x_nor = pd.DataFrame(scaler_x.fit_transform(data_x))
        data_y_nor = pd.DataFrame(scaler_y.fit_transform(data_y))

        if if_shuffle:
            train_x = data_x_nor.sample(frac=train_size)
            train_y = data_y_nor.iloc[train_x.index]

            val_x = data_x_nor.drop(train_x.index).sample(frac=val_rate)
            val_y = data_y_nor.iloc[val_x.index]

            test_x = data_x_nor.drop(train_x.index).drop(val_x.index)
            test_y = data_y_nor.iloc[test_x.index]

            train_x.reset_index(inplace=True, drop=True)
            train_y.reset_index(inplace=True, drop=True)
            val_x.reset_index(inplace=True, drop=True)
            val_y.reset_index(inplace=True, drop=True)
            test_x.reset_index(inplace=True, drop=True)
            test_y.reset_index(inplace=True, drop=True)

        elif not if_shuffle:
            train_x = data_x_nor.iloc[:int(len(data) * train_size), :]
            train_y = data_y_nor.iloc[:int(len(data) * train_size), :]

            val_x = data_x_nor.iloc[int(len(data) * train_size):int(len(data) * (train_size + val_size)), :]
            val_y = data_y_nor.iloc[int(len(data) * train_size):int(len(data) * (train_size + val_size)), :]

            test_x = data_x_nor.iloc[int(len(data) * (train_size + val_size)):, :]
            test_y = data_y_nor.iloc[int(len(data) * (train_size + val_size)):, :]

    else:
        if if_shuffle:
            train_x = data_x.sample(frac=train_size)
            train_y = data_y.iloc[train_x.index]

            val_x = data_x.drop(train_x.index).sample(frac=val_rate)
            val_y = data_y.iloc[val_x.index]

            test_x = data_x.drop(train_x.index).drop(val_x.index)
            test_y = data_y.iloc[test_x.index]

            train_x.reset_index(inplace=True, drop=True)
            train_y.reset_index(inplace=True, drop=True)
            val_x.reset_index(inplace=True, drop=True)
            val_y.reset_index(inplace=True, drop=True)
            test_x.reset_index(inplace=True, drop=True)
            test_y.reset_index(inplace=True, drop=True)

        elif not if_shuffle:
            train_x = data_x.iloc[:int(len(data) * train_size), :]
            train_y = data_y.iloc[:int(len(data) * train_size), :]

            val_x = data_x.iloc[int(len(data) * train_size):int(len(data) * (train_size + val_size)), :]
            val_y = data_y.iloc[int(len(data) * train_size):int(len(data) * (train_size + val_size)), :]

            test_x = data_x.iloc[int(len(data) * (train_size + val_size)):, :]
            test_y = data_y.iloc[int(len(data) * (train_size + val_size)):, :]

    train_x = torch.from_numpy(train_x.to_numpy()).double()
    train_y = torch.from_numpy(train_y.to_numpy()).double()
    val_x = torch.from_numpy(val_x.to_numpy()).double()
    val_y = torch.from_numpy(val_y.to_numpy()).double()
    test_x = torch.from_numpy(test_x.to_numpy()).double()
    test_y = torch.from_numpy(test_y.to_numpy()).double()

    t_set = TensorDataset(train_x, train_y)
    train_loader = DataLoader(t_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    v_set = TensorDataset(val_x, val_y)
    validation_loader = DataLoader(v_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    return train_loader, validation_loader, test_x, test_y, scaler_x, scaler_y