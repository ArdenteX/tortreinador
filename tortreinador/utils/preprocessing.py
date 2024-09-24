from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class _FunctionController:
    """
    Call function dynamically

        Args:
            - requirement_dict (list): Combination of requirement, [[_normal, _shuffle], [_normal, _not_shuffle], [_not_normal, _shuffle], [_not_normal, _not_shuffle]]

    """
    def __init__(self, requirement_list: list, train_size, val_size, random_state, x, y, scaler_x, scaler_y):
        self.r_d = requirement_list
        self.t_size = train_size
        self.v_size = val_size
        self.random_state = random_state
        # Original Data
        self.x = x
        self.y = y
        self.s_x = scaler_x
        self.s_y = scaler_y

    def _normal(self):
        x_normal = pd.DataFrame(self.s_x.fit_transform(self.x))
        y_normal = pd.DataFrame(self.s_y.fit_transform(self.y))

        return [x_normal, y_normal]

    def _not_normal(self):
        return [self.x, self.y]

    def _shuffle(self, *args):
        train_x = args[0].sample(frac=self.t_size, random_state=self.random_state)
        train_y = args[1].loc[train_x.index]

        test_x = args[0].drop(train_x.index)
        test_y = args[1].loc[test_x.index]

        val_x = train_x.sample(frac=self.v_size, random_state=self.random_state)
        val_y = train_y.loc[val_x.index]

        train_x = train_x.drop(val_x.index)
        train_y = train_y.drop(val_y.index)

        train_x.reset_index(inplace=True, drop=True)
        train_y.reset_index(inplace=True, drop=True)
        val_x.reset_index(inplace=True, drop=True)
        val_y.reset_index(inplace=True, drop=True)
        test_x.reset_index(inplace=True, drop=True)
        test_y.reset_index(inplace=True, drop=True)

        return train_x, train_y, val_x, val_y, test_x, test_y

    def _not_shuffle(self, *args):
        train_x = args[0].iloc[:int(len(args[0]) * self.t_size), :]
        train_y = args[1].iloc[:int(len(args[1]) * self.t_size), :]

        val_x = args[0].iloc[int(len(args[0]) * self.t_size):int(len(args[0]) * (self.t_size + self.v_size)), :]
        val_y = args[1].iloc[int(len(args[1]) * self.t_size):int(len(args[1]) * (self.t_size + self.v_size)), :]

        test_x = args[0].iloc[int(len(args[0]) * (self.t_size + self.v_size)):, :]
        test_y = args[1].iloc[int(len(args[1]) * (self.t_size + self.v_size)):, :]

        return train_x, train_y, val_x, val_y, test_x, test_y

    def exec(self):
        return getattr(self, self.r_d[1])(*getattr(self, self.r_d[0])())


def load_data(data: pd.DataFrame, input_parameters: list, output_parameters: list,
              feature_range=None, train_size: float = 0.8, val_size: float = 0.1, if_normal: bool = True,
              if_shuffle: bool = True, n_workers: int = 8, batch_size: int = 256, random_state=42, if_double: bool = False):
    """
    Load Data and Normalize for Regression Tasks: This function preprocesses data specifically for regression tasks by handling data splitting, optional shuffling, normalization, and DataLoader creation.

    Args:
        data (pd.DataFrame): The complete dataset in a Pandas DataFrame.
        input_parameters (list of str or int): Column names or indices representing the input features.
        output_parameters (list of str or int): Column names or indices representing the target variables.
        feature_range (tuple of (float, float), optional): The range (min, max) used by the MinMaxScaler for scaling data. Defaults to (0, 1).
        train_size (float): The proportion of the dataset to include in the train split (0 to 1).
        val_size (float): The proportion of the training data to use as validation data (0 to 1).
        if_normal (bool): Flag to determine whether to normalize the data using MinMaxScaler.
        if_shuffle (bool): Flag to determine whether to shuffle the data before splitting into training, validation, and test sets.
        n_workers (int): The number of subprocesses to use for data loading. More workers can increase the loading speed but consume more CPU cores.
        batch_size (int): Number of samples per batch to load.
        random_state (int, optional): A seed used by the random number generator for reproducibility. Defaults to None.
        if_double (bool): Flag to determine whether to convert data to double precision (float64) format.

    Returns:
        tuple: Contains Train DataLoader, Validation DataLoader, Test X, Test Y, Scaler for X, and Scaler for Y.
        - Train DataLoader (torch.utils.data.DataLoader): DataLoader containing the training data.
        - Validation DataLoader (torch.utils.data.DataLoader): DataLoader containing the validation data.
        - Test X (np.array): Features of the test dataset.
        - Test Y (np.array): Targets of the test dataset.
        - Scaler X (sklearn.preprocessing.MinMaxScaler): Scaler object used for the input features.
        - Scaler Y (sklearn.preprocessing.MinMaxScaler): Scaler object used for the output targets.
    """
    if val_size >= 0.5:
        raise Warning("The percentage of validation data too high will let the training data not enough to train the powerful model. "
                      "Usually set the percentage of validation dataset at 0.1-0.3")

    scaler_x = None
    scaler_y = None

    # Data Normalization
    if feature_range is None:
        feature_range = [0, 1]

    # 8/12 Fixed
    success = False
    while success is False:
        try:
            scaler_x = MinMaxScaler(feature_range=feature_range)
            scaler_y = MinMaxScaler(feature_range=feature_range)
            success = True

        except:
            feature_range = tuple(feature_range)

    if 'scaler_x' not in locals() or 'scaler_y' not in locals():
        raise ValueError("Can not define scaler_x or scaler_y, please check the input feature range.")

    train_x = None
    train_y = None
    val_x = None
    val_y = None
    test_x = None
    test_y = None

    data_x = eval("data.{}[:, input_parameters]".format('iloc' if type(input_parameters[0]) == int else 'loc'))
    data_y = eval("data.{}[:, output_parameters]".format('iloc' if type(output_parameters[0]) == int else 'loc'))

    requirement_list = ['_normal' if if_normal is True else '_not_normal', '_shuffle' if if_shuffle is True else '_not_shuffle']

    controller = _FunctionController(requirement_list, train_size, val_size, random_state, data_x, data_y, scaler_x, scaler_y)

    train_x, train_y, val_x, val_y, test_x, test_y = controller.exec()

    train_x = eval('torch.from_numpy(train_x.to_numpy()){}'.format('.double()' if if_double is True else ''))
    train_y = eval('torch.from_numpy(train_y.to_numpy()){}'.format('.double()' if if_double is True else ''))
    val_x = eval('torch.from_numpy(val_x.to_numpy()){}'.format('.double()' if if_double is True else ''))
    val_y = eval('torch.from_numpy(val_y.to_numpy()){}'.format('.double()' if if_double is True else ''))
    test_x = eval('torch.from_numpy(test_x.to_numpy()){}'.format('.double()' if if_double is True else ''))
    test_y = eval('torch.from_numpy(test_y.to_numpy()){}'.format('.double()' if if_double is True else ''))

    t_set = TensorDataset(train_x, train_y)
    train_loader = DataLoader(t_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    v_set = TensorDataset(val_x, val_y)
    validation_loader = DataLoader(v_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    return train_loader, validation_loader, test_x, test_y, scaler_x, scaler_y


def noise_generator(error_rate, input_features, data):
    columns_mean = data[input_features].mean()
    variance = (columns_mean * error_rate) ** 2
    corr = data[input_features].corr()

    cov = []
    for i in range(len(variance)):
        var = variance.iloc[i]
        col = input_features[i]
        tmp = []
        for j in range(len(corr)):
            current_col = input_features[j]
            if j == i:
                tmp.append(var)

            else:
                p = corr.loc[col, current_col]
                current_var = variance[current_col]
                tmp.append(p * (var * current_var))

        cov.append(tmp)

    np_cov = np.array(cov)
    return np_cov


