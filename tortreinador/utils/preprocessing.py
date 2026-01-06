import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List
import torch
from typing import Union, List, Dict, Tuple


@dataclass
class ScalerConfig:
    """
    Configuration for feature/target normalization.

    Attributes:
        on (bool): Whether to enable normalization.
        method (str): Either 'MinMaxScaler'/'minmax' or 'StandardScaler'/'standard'.
        normal_y (bool): Whether to normalize target columns in addition to features.
        feature_range (tuple): Range used by MinMaxScaler.
    """
    on: bool = True
    method: str = 'MinMaxScaler'
    normal_y: bool = False
    feature_range: Union[Tuple, List, None] = (0, 1)

    def validate(self):
        """Validate scaler configuration before use."""
        if self.on is True and self.method not in ['MinMaxScaler', 'StandardScaler', 'minmax', 'standard']:
            raise ValueError(
                "Currently only 'MinMaxScaler' and 'StandardScaler' are supported for normalization methods ('minmax' and 'standard' are allowed).")

    def select_scaler(self):
        """
        Instantiate sklearn scaler objects for X and Y.

        Returns:
            tuple: A pair of scalers (scaler_x, scaler_y) matching the requested method.
        """
        if self.feature_range is None:
            self.feature_range = (0, 1)

        scaler_x = None
        scaler_y = None

        success = False
        while success is False:
            try:
                scaler_x = MinMaxScaler(feature_range=self.feature_range) if self.method in ['MinMaxScaler',
                                                                                                 'minmax'] else StandardScaler()
                scaler_y = MinMaxScaler(feature_range=self.feature_range) if self.method in ['MinMaxScaler',
                                                                                                 'minmax'] else StandardScaler()
                success = True

            except:
                self.feature_range = tuple(self.feature_range)

        return scaler_x, scaler_y


class _FunctionController:
    """
    Call function dynamically

        Args:
            - requirement_dict (list): Combination of requirement, [[_normal, _shuffle], [_normal, _not_shuffle], [_not_normal, _shuffle], [_not_normal, _not_shuffle]]

    """

    def __init__(self, requirement_list: dict, train_size, val_size, random_state, x, y, scaler_x, scaler_y):
        """
        Args:
            requirement_list (dict): Functions to call for normalization and shuffling.
            train_size (float): Fraction used for training split.
            val_size (float): Fraction of training split reserved for validation.
            random_state (int): Seed for deterministic sampling.
            x (pd.DataFrame): Feature dataframe.
            y (pd.DataFrame): Target dataframe.
            scaler_x: Scaler used for features.
            scaler_y: Scaler used for targets.
        """
        self.r_d = requirement_list
        self.t_size = train_size
        self.v_size = val_size
        self.random_state = random_state
        # Original Data
        self.x = x
        self.y = y
        self.s_x = scaler_x
        self.s_y = scaler_y

    def _normal(self, x_, y_, first=False, normal_y=True):
        """
        Normalize features (and optionally targets) using pre-selected scalers.

        Args:
            x_ (pd.DataFrame): Feature slice.
            y_ (pd.DataFrame): Target slice.
            first (bool): If True, fit scalers; otherwise transform only.
            normal_y (bool): Whether to normalize targets.
        """

        if first:
            x_ = pd.DataFrame(self.s_x.fit_transform(x_))
            if normal_y:
                y_ = pd.DataFrame(self.s_y.fit_transform(y_))

        else:
            x_ = pd.DataFrame(self.s_x.transform(x_))
            if normal_y:
                y_ = pd.DataFrame(self.s_y.transform(y_))

        return [x_, y_]

    def _not_normal(self):
        """Return raw feature/target data without normalization."""
        return [self.x, self.y]

    def _shuffle(self, args):
        """
        Shuffle dataset rows and split into train/val/test with optional normalization.

        Args:
            args (str): Either '_normal' or '_not_normal' indicating normalization path.
        """

        train_x = self.x.sample(frac=self.t_size, random_state=self.random_state)
        train_y = self.y.loc[train_x.index]

        test_x = self.x.drop(train_x.index)
        test_y = self.y.loc[test_x.index]

        val_x = train_x.sample(frac=self.v_size, random_state=self.random_state)
        val_y = train_y.loc[val_x.index]

        if args == '_normal':
            train_x, train_y = self._normal(train_x, train_y, first=True, normal_y=self.r_d['normal_y'])
            test_x, test_y = self._normal(test_x, test_y, first=False, normal_y=self.r_d['normal_y'])
            val_x, val_y = self._normal(val_x, val_y, first=False, normal_y=self.r_d['normal_y'])

        train_x = train_x.drop(val_x.index)
        train_y = train_y.drop(val_y.index)

        train_x.reset_index(inplace=True, drop=True)
        train_y.reset_index(inplace=True, drop=True)
        val_x.reset_index(inplace=True, drop=True)
        val_y.reset_index(inplace=True, drop=True)
        test_x.reset_index(inplace=True, drop=True)
        test_y.reset_index(inplace=True, drop=True)

        return train_x, train_y, val_x, val_y, test_x, test_y

    def _not_shuffle(self, args):
        """
        Deterministically split dataset without shuffling, preserving original order.

        Args:
            args (str): Either '_normal' or '_not_normal' indicating normalization path.
        """
        train_x = self.x.iloc[:int(len(self.x) * self.t_size), :]
        train_y = self.y.iloc[:int(len(self.y) * self.t_size), :]

        val_x = self.x.iloc[int(len(self.x) * self.t_size):int(len(self.x) * (self.t_size + self.v_size)), :]
        val_y = self.y.iloc[int(len(self.y) * self.t_size):int(len(self.y) * (self.t_size + self.v_size)), :]

        test_x = self.x.iloc[int(len(self.x) * (self.t_size + self.v_size)):, :]
        test_y = self.y.iloc[int(len(self.y) * (self.t_size + self.v_size)):, :]

        if args == '_normal':
            train_x, train_y = self._normal(train_x, train_y, first=True)
            test_x, test_y = self._normal(test_x, test_y, first=False)
            val_x, val_y = self._normal(val_x, val_y, first=False)

        return train_x, train_y, val_x, val_y, test_x, test_y

    def exec(self):
        """Dispatch to shuffle/not-shuffle and normalize/not-normalize implementations."""
        return getattr(self, self.r_d['shuffle_function'])(self.r_d['normal_function'])


def load_data(data: pd.DataFrame, input_parameters: list, output_parameters: list,
              train_size: float = 0.8, val_size: float = 0.1, normal: ScalerConfig = None,
              if_shuffle: bool = True, num_workers: Union[List, Dict[str, int], Tuple, int] = (0, 0), batch_size: int = 256, random_state=42,
              if_double: bool = False, add_noise: bool = False, error_rate: list = None, only_noise=True, save_path: str = None,
              grid_search: bool = False, log_injection: bool = False, legal_check: bool = True):
    """
    Load Data and Normalize for Regression Tasks: This function preprocesses data specifically for regression tasks by handling data splitting, optional shuffling, normalization, and DataLoader creation.

    Args:
        data (pd.DataFrame): The complete dataset in a Pandas DataFrame.
        input_parameters (list of str or int): Column names or indices representing the input features.
        output_parameters (list of str or int): Column names or indices representing the target variables.

        train_size (float): The proportion of the dataset to include in the train split (0 to 1).
        val_size (float): The proportion of the training data to use as validation data (0 to 1).

        if_shuffle (bool): Flag to determine whether to shuffle the data before splitting into training, validation, and test sets.
        num_workers (int): The number of subprocesses to use for data loading. More workers can increase the loading speed but consume more CPU cores.
        batch_size (int): Number of samples per batch to load.
        random_state (int, optional): A seed used by the random number generator for reproducibility. Defaults to None.
        if_double (bool): Flag to determine whether to convert data to double precision (float64) format.
        add_noise(bool): Flag to determine whether to add noise to origin dataset.
        error_rate(list): List of error rates to calculate for covariance reflection. Defaults to None.
        only_noise(bool): Flag to determine whether to add noise to dataset or add noised data to dataset.
        save_path(str): Path to save .npy file
        grid_search(bool): Only return the train loader and validation loader if using Grid Search function, defaults to False.
        log_injection(bool): Whether to use log space noise injection techniques. Defaults to False.
        legal_check(bool): Flag to determine whether to check for legal noised datas. Defaults to True.

        normal(ScalerConfig, optional): Normalization method to use. Defaults to MinMaxScaler()
            - on (bool): Flag to determine whether to normalize the data using MinMaxScaler.
            - method (str): MinMaxScaler or StandardScaler
            - feature_range (tuple of (float, float), optional): The range (min, max) used by the MinMaxScaler for scaling data. Defaults to (0, 1).
            - normal_y(bool): Flag to determine whether to normalize the y using MinMaxScaler.

    Returns:
        tuple: Contains Train DataLoader, Validation DataLoader, Test X, Test Y, Scaler for X, and Scaler for Y.
        - Train DataLoader (torch.utils.data.DataLoader): DataLoader containing the training data.
        - Validation DataLoader (torch.utils.data.DataLoader): DataLoader containing the validation data.
        - Test X (np.array): Features of the test dataset.
        - Test Y (np.array): Targets of the test dataset.
        - Scaler X (sklearn.preprocessing.MinMaxScaler): Scaler object used for the input features.
        - Scaler Y (sklearn.preprocessing.MinMaxScaler): Scaler object used for the output targets.
    """
    train_x = None
    train_y = None
    val_x = None
    val_y = None
    test_x = None
    test_y = None
    scaler_x = None
    scaler_y = None

    if val_size >= 0.5:
        print(
            "Warning: The percentage of validation data too high will let the training data not enough to train the powerful model. "
            "Usually set the percentage of validation dataset at 0.1-0.3")

    if add_noise and error_rate is None:
        raise ValueError("Please specify the error rate list (e.g. [0.01, 0.01, 0.01]) when using the add_noise flag")

    if normal is None:
        normal = ScalerConfig(on=False)

    if normal.on:
        normal.validate()
        scaler_x, scaler_y = normal.select_scaler()

        if 'scaler_x' not in locals() or 'scaler_y' not in locals():
            raise ValueError("Can not define scaler_x or scaler_y, please check the input feature range.")

    data_x = eval("data.{}[:, input_parameters]".format('iloc' if type(input_parameters[0]) == int else 'loc'))
    data_y = eval("data.{}[:, output_parameters]".format('iloc' if type(output_parameters[0]) == int else 'loc'))

    if add_noise:
        if log_injection:

            if legal_check:
                data_x_noise, l_mask = log_space_noise_injection(data_x, error_rate, legal_check=legal_check)
                data_x_noise = pd.DataFrame(data_x_noise, columns=data_x.columns)
                data_x_noise = data_x_noise[l_mask]
                data_y = data_y[l_mask]

            else:
                data_x_noise = log_space_noise_injection(data_x, error_rate, legal_check=legal_check)
                data_x_noise = pd.DataFrame(data_x_noise, columns=data_x.columns)

        else:
            if legal_check:

                data_x_noise, l_mask = noise_injection(error_rate, data_x, legal_check=legal_check)
                data_x_noise = pd.DataFrame(data_x_noise, columns=data_x.columns)
                data_x_noise = data_x_noise[l_mask]
                data_y = data_y[l_mask]

            else:
                data_x_noise = noise_injection(error_rate, data_x, legal_check=legal_check)

        # error_rate = np.array(error_rate)
        # adj_cov_x = noise_generator(error_rate, data_x)
        # obs_error = np.random.multivariate_normal(mean=[0] * adj_cov_x.shape[-1], cov=adj_cov_x, size=data_x.shape[0])
        # data_x_noise = data_x + obs_error

        if only_noise:
            data_x = data_x_noise

        else:
            for i in input_parameters:
                data_x[i + "_noise"] = data_x_noise.loc[:, i]

    # requirement_list = ['_normal' if if_normal is True else '_not_normal',
    #                     '_shuffle' if if_shuffle is True else '_not_shuffle']

    requirement_list = {
        'normal_function': '_normal' if normal.on is True else '_not_normal',
        'shuffle_function': '_shuffle' if if_shuffle is True else '_not_shuffle',
        'normal_y': normal.normal_y
    }

    controller = _FunctionController(requirement_list, train_size, val_size, random_state, data_x, data_y, scaler_x,
                                     scaler_y)

    train_x, train_y, val_x, val_y, test_x, test_y = controller.exec()

    # if add_noise:
    #     error_rate = np.array(error_rate)
    #     adjusted_cov_train_x = noise_generator(error_rate, train_x)
    #
    #     adjusted_cov_val_x = noise_generator(error_rate, val_x)
    #     adjusted_cov_test_x = noise_generator(error_rate, test_x)
    #
    #     noise_train_x = np.random.multivariate_normal(mean=[0] * adjusted_cov_train_x.shape[-1], cov=adjusted_cov_train_x,
    #                                                   size=train_x.shape[0])
    #     noise_val_x = np.random.multivariate_normal(mean=[0] * adjusted_cov_val_x.shape[-1], cov=adjusted_cov_val_x, size=val_x.shape[0])
    #     noise_test_x = np.random.multivariate_normal(mean=[0] * adjusted_cov_test_x.shape[-1], cov=adjusted_cov_test_x, size=test_x.shape[0])
    #
    #     if not only_noise:
    #         train_x_noise = train_x + noise_train_x
    #         val_x_noise = val_x + noise_val_x
    #         test_x_noise = test_x + noise_test_x
    #
    #     else:
    #         train_x_noise = noise_train_x
    #         val_x_noise = noise_val_x
    #         test_x_noise = noise_test_x
    #
    #     for i in range(len(input_parameters)):
    #         train_x[i + len(input_parameters) + 1] = train_x_noise.loc[:, i] if not only_noise else train_x_noise[:, i]
    #         val_x[i + len(input_parameters) + 1] = val_x_noise.loc[:, i] if not only_noise else val_x_noise[:, i]
    #         test_x[i + len(input_parameters) + 1] = test_x_noise.loc[:, i] if not only_noise else test_x_noise[:, i]

        # print(train_x.head())
    if save_path is not None:
        save_npy(path=save_path, df=train_x, name='train_x')
        save_npy(path=save_path, df=train_y, name='train_y')
        save_npy(path=save_path, df=val_x, name='val_x')
        save_npy(path=save_path, df=val_y, name='val_y')
        save_npy(path=save_path, df=test_x, name='test_x')
        save_npy(path=save_path, df=test_y, name='test_y')

        save_scaler(scaler_x, path=save_path, name='scaler_x')
        save_scaler(scaler_y, path=save_path, name='scaler_y')

    train_x = eval('torch.from_numpy(train_x.to_numpy()){}'.format('.double()' if if_double is True else '.float()'))
    train_y = eval('torch.from_numpy(train_y.to_numpy()){}'.format('.double()' if if_double is True else '.float()'))
    val_x = eval('torch.from_numpy(val_x.to_numpy()){}'.format('.double()' if if_double is True else '.float()'))
    val_y = eval('torch.from_numpy(val_y.to_numpy()){}'.format('.double()' if if_double is True else '.float()'))
    test_x = eval('torch.from_numpy(test_x.to_numpy()){}'.format('.double()' if if_double is True else '.float()'))
    test_y = eval('torch.from_numpy(test_y.to_numpy()){}'.format('.double()' if if_double is True else '.float()'))

    # t_set = TensorDataset(train_x, train_y)
    # train_loader = DataLoader(t_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    if isinstance(num_workers, list) or isinstance(num_workers, tuple):
        train_loader = get_dataloader(x=train_x, y=train_y, batch_size=batch_size, shuffle=False, num_workers=num_workers[0])

        # v_set = TensorDataset(val_x, val_y)
        # validation_loader = DataLoader(v_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        validation_loader = get_dataloader(val_x, val_y, batch_size=batch_size, shuffle=False, num_workers=num_workers[1])

    if isinstance(num_workers, int):
        train_loader = get_dataloader(x=train_x, y=train_y, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers)

        # v_set = TensorDataset(val_x, val_y)
        # validation_loader = DataLoader(v_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        validation_loader = get_dataloader(val_x, val_y, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if isinstance(num_workers, dict):
        train_loader = get_dataloader(x=train_x, y=train_y, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers['train_num_workers'])

        # v_set = TensorDataset(val_x, val_y)
        # validation_loader = DataLoader(v_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        validation_loader = get_dataloader(val_x, val_y, batch_size=batch_size, shuffle=False, num_workers=num_workers['val_num_workers'])

    if grid_search:
        return train_loader, validation_loader

    else:
        return train_loader, validation_loader, test_x, test_y, scaler_x, scaler_y


def save_npy(df, path, name):
    """
    Save a dataframe as a NumPy binary file.

    Raises:
        FileNotFoundError: When the destination directory does not exist.
    """
    if os.path.exists(path):
        df_to_np = df.to_numpy()
        np.save(os.path.join(path, name + ".npy"), df_to_np)

    else:
        raise FileNotFoundError()

def save_scaler(scaler, path, name):
    """
    Persist a fitted scaler with joblib.

    Raises:
        FileNotFoundError: When the destination directory does not exist.
    """
    if os.path.exists(path):
        joblib.dump(scaler, os.path.join(path, name + ".save"))

    else:
        raise FileNotFoundError()

def get_dataloader(x, y, batch_size, shuffle, num_workers=8):
    """Wrap tensors into a DataLoader with consistent arguments used across the package."""
    tensor_dataset = TensorDataset(x, y)
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

def cov_decompose(cov):
    """Extract diagonal variances and upper-triangular covariances from a covariance matrix."""
    S = [cov[i, i] for i in range(len(cov))]
    U = [[cov[i, j] for j in range(i + 1, len(cov))] for i in range(int(len(cov) - 1))]
    return S, U


def p_calculation(s, u):
    """Convert covariance components into correlation-like coefficients used for noise synthesis."""
    P_ij = []
    for i in range(len(u)):
        tmp_u = u[i]
        tmp_s = s[i]
        tmp_pij = []
        for j in range(len(tmp_u)):
            tmp_s_next = s[i + j + 1]
            s_s_next_sqrt = np.sqrt(tmp_s * tmp_s_next)
            p_ij = tmp_u[j] / s_s_next_sqrt
            tmp_pij.append(p_ij)

        P_ij.append(tmp_pij)

    return P_ij


def cov_adj_compose(pij, sadj):
    """Recompose an adjusted covariance matrix from correlation coefficients and adjusted variances."""
    cov_adj = np.zeros((4, 4))
    for i in range(len(pij)):
        tmp_pij = pij[i]
        tmp_s = sadj[i]
        for j in range(len(tmp_pij)):
            tmp_s_next = sadj[i + j + 1]
            s_s_next_sqrt = np.sqrt(tmp_s * tmp_s_next)
            covij = tmp_pij[j] * s_s_next_sqrt
            cov_adj[i, i + j + 1] = covij
            cov_adj[i + j + 1, i] = covij

    cov_adj = cov_adj + np.diag(sadj)
    return cov_adj


def noise_generator(n_r, x):
    """
    Generate an adjusted covariance matrix to synthesize noise scaled by provided error rates.

    Args:
        n_r (np.ndarray): Error rate multipliers per feature.
        df (pd.DataFrame): Data used to compute the base covariance.

    Returns:
        np.ndarray: Adjusted covariance matrix.
    """
    cov = np.cov(x.to_numpy().T)

    try:
        m = x.to_numpy().mean(axis=0) * n_r

    except AttributeError:
        m = x.mean(axis=0) * n_r

    S, U = cov_decompose(cov)

    P_ij = p_calculation(S, U)
    S_adj = m ** 2
    cov_adj = cov_adj_compose(P_ij, S_adj)
    return cov_adj


def noise_injection(error_rate, data_x, legal_check: bool = True):
    if isinstance(error_rate, list):
        error_rate = np.array(error_rate)

    error_rate = np.array(error_rate)
    adj_cov_x = noise_generator(error_rate, data_x)
    obs_error = np.random.multivariate_normal(mean=[0] * adj_cov_x.shape[-1], cov=adj_cov_x, size=data_x.shape[0])

    data_x_noisy = data_x + obs_error
    if legal_check:
        l_m = legal_log_space(data_x_noisy)

        return data_x_noisy, l_m

    return data_x_noisy


def legal_global(noisy_dataset):
    noisy_dataset.iloc[:, [2, 3]] = noisy_dataset.iloc[:, [2, 3]].clip(lower=1e-5)

    return noisy_dataset


def legal_log_space(noisy_dataset):
    if not isinstance(noisy_dataset, pd.DataFrame):
        noisy_dataset_df = pd.DataFrame(noisy_dataset)

    else:
        noisy_dataset_df = noisy_dataset

    legal_mask = (
            noisy_dataset_df.iloc[:, 0].between(0.0, 12.0) &
            noisy_dataset_df.iloc[:, 1].between(0.0, 2.4) &
            noisy_dataset_df.iloc[:, 2].between(0.0, 6.1) &
            noisy_dataset_df.iloc[:, 3].between(0.0, 2.4)
    )

    return legal_mask

def log_space_noise_injection(X, error_rate, legal_check: bool = True):
    """
    Injects physics-informed, multivariate log-normal noise into a dataset.

    This function simulates realistic observational uncertainties for exoplanetary
    parameters (e.g., Mass, Radius, Abundances) by employing a hybrid noise model
    that operates in log-space. It preserves the covariance structure of the
    original population to account for systematic correlations between parameters
    (e.g., M-R relation, stellar-planetary compositional links).

    The noise model combines two components:
    1. **Noise Floor (Additive):** Dominates at low values (low SNR), anchored to
       the population median. Simulates instrumental detection limits.
    2. **Relative Noise (Multiplicative):** Dominates at high values (high SNR).
       Simulates calibration or model uncertainties.

    The injection is performed via the Cholesky decomposition of the reconstructed
    covariance matrix for each sample, ensuring that the injected noise respects
    the inter-parameter correlations of the dataset.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The clean, synthetic input data. Shape (N_samples, N_features).
        Values must be strictly positive (physical quantities like Mass, Radius).
    error_rate : list or np.ndarray
        A list of error rates (fractions) for each feature.
        Example: [0.14, 0.04, 0.12, 0.13] implies 14% error for feature 0, etc.
        Note: These rates scale both the relative noise component and the
        median-anchored noise floor.

    Returns
    -------
    noisy_dataset : np.ndarray
        The dataset with injected noise. Shape (N_samples, N_features).
        Guaranteed to be positive due to log-normal sampling.

    Mathematical Formulation
    ------------------------
    For each feature $j$ and sample $i$:
    $$ \sigma_{floor, j} = \text{Median}(X_j) \times \text{rate}_j $$
    $$ \sigma_{curr, ij} = X_{ij} \times \text{rate}_j $$
    $$ \sigma_{total, ij} = \sqrt{\sigma_{floor, j}^2 + \sigma_{curr, ij}^2} $$

    The effective relative error is mapped to log-space variance:
    $$ \sigma^2_{log, ij} \approx [\ln(1 + \frac{\sigma_{total, ij}}{X_{ij}})]^2 $$

    Noise $\eta$ is sampled from $\mathcal{N}(0, \Sigma_{local})$, where $\Sigma_{local}$
    retains the correlation structure of the input population.
    $$ X_{noisy} = X_{clean} \cdot e^{\eta} $$
    """
    if isinstance(X, pd.DataFrame):
        X =X.to_numpy()

    cov = np.cov(np.log(X).T)

    X_median = np.median(X, axis=0)
    noise_floor = X_median * error_rate
    noise_current = X * error_rate

    sigma_total = np.sqrt(noise_floor ** 2 + noise_current ** 2)
    r_eff = sigma_total / X

    log_sigma = np.log1p(r_eff) ** 2

    S, U = cov_decompose(cov)
    P_ij = p_calculation(S, U)

    # lens = np.array([len(x) for x in P_ij], dtype=int)
    pij_vec = np.concatenate([np.asarray(x, dtype=float) for x in P_ij], axis=0)

    d = log_sigma.shape[-1]
    r, c = np.triu_indices(d, k=1)
    diag_r, diag_c = [i for i in range(d)], [i for i in range(d)]
    scale = np.sqrt(log_sigma[:, r] * log_sigma[:, c])
    cov_vec = pij_vec * scale

    cov_adjs = np.zeros((X.shape[0], d, d), dtype=float)
    cov_adjs[:, r, c] = cov_vec
    cov_adjs[:, c, r] = cov_vec
    cov_adjs[:, diag_r, diag_c] = log_sigma

    rng = np.random.default_rng(42)

    L = np.linalg.cholesky(cov_adjs + 1e-10 * np.eye(d)[None, :, :])  # (N,d,d)
    z = rng.standard_normal((X.shape[0], d))  # (N,d)
    eta = (L @ z[..., None]).squeeze(-1)

    noisy_dataset = X * np.exp(eta)

    if legal_check:
        legal_mask = legal_log_space(noisy_dataset)
        return noisy_dataset, legal_mask

    else:
        return noisy_dataset

