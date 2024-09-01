import torch
import torch.nn as nn
from torch.optim import Optimizer
from tortreinador.utils.metrics import r2_score, mixture
from tqdm import tqdm
from tortreinador.utils.Recorder import Recorder, RecorderForEpoch
from tortreinador.utils.WarmUpLR import WarmUpLR
from tortreinador.utils.View import visualize_lastlayer, visualize_train_loss, visualize_test_loss
from tensorboardX import SummaryWriter
from datetime import datetime
import csv
import os
import platform


class TorchTrainer:
    """
    A class to implement the training and validation loop based on PyTorch.

    Attributes:
        epoch (int): The number of epochs to train the model.
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer used for training the model.
        criterion (nn.Module): The loss function used for training.
        data_save_mode (str): The mode to save data ('recorder' or 'csv').
        device (torch.device): The device (CPU or GPU) on which the model will be trained.
        writer (SummaryWriter): TensorBoard writer for visualizing training metrics.
        train_loss_recorder (Recorder): Recorder for tracking training loss.
        val_loss_recorder (Recorder): Recorder for tracking validation loss.
        train_metric_recorder (Recorder): Recorder for tracking training metrics.
        val_metric_recorder (Recorder): Recorder for tracking validation metrics.
        epoch_train_loss (RecorderForEpoch): Recorder for tracking loss across epochs (only if `data_save_mode` is 'recorder').
        epoch_val_loss (RecorderForEpoch): Recorder for tracking validation loss across epochs (only if `data_save_mode` is 'recorder').
        epoch_train_metric (RecorderForEpoch): Recorder for tracking training metrics across epochs (only if `data_save_mode` is 'recorder').
        epoch_val_metric (RecorderForEpoch): Recorder for tracking validation metrics across epochs (only if `data_save_mode` is 'recorder').
        epoch_extra_metric (RecorderForEpoch, optional): Recorder for tracking additional metrics across epochs.
        csv_filename (str, optional): The filename for saving data in CSV format (only if `data_save_mode` is 'csv').

    """
    def __init__(self,
                 is_gpu: bool = True,
                 epoch: int = 150, log_dir: str = None, model: nn.Module = None,
                 optimizer: Optimizer = None, extra_metric: nn.Module = None, criterion: nn.Module = None, data_save_mode: str = 'recorder'):
        """
        Initializes the TorchTrainer with model, optimizer, criterion, and other training settings.

        Args:
            is_gpu (bool): Flag indicating whether to use GPU if available.
            epoch (int): Number of epochs to train the model.
            log_dir (str, optional): Directory to save TensorBoard logs.
            model (nn.Module): The model to be trained.
            optimizer (Optimizer): The optimizer used for training the model.
            extra_metric (nn.Module, optional): Additional metric for evaluation.
            criterion (nn.Module): The loss function used for training.
            data_save_mode (str): The mode to save data ('recorder' or 'csv').

        Raises:
            ValueError: If model, optimizer, or criterion is not provided or is of incorrect type.
            ValueError: If data_save_mode is not 'recorder' or 'csv'.

        """

        if not isinstance(model, nn.Module) or not isinstance(optimizer, Optimizer) or not isinstance(criterion,
                                                                                                      nn.Module) or epoch is None:
            raise ValueError("Please provide the correct type of model, optimizer, criterion and the not none epoch")

        data_save_mode_list = ['recorder', 'csv']
        if data_save_mode not in data_save_mode_list:
            raise ValueError(
                "Unexpected value for data_save_mode: {}, please input 'recorder' or 'csv', defaults to 'recorder'.".format(
                    data_save_mode))

        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.data_save_mode = data_save_mode
        self.system = platform.system()
        self.file_time = datetime.now().strftime('%Y-%m-%d %H:%M').replace(":", "").replace("-", '').replace(" ", '')

        self.device = torch.device("cuda" if torch.cuda.is_available() and is_gpu else "cpu")

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else log_dir

        self.train_loss_recorder = Recorder(self.device.type)
        self.val_loss_recorder = Recorder(self.device.type)
        self.train_metric_recorder = Recorder(self.device.type)
        self.val_metric_recorder = Recorder(self.device.type)

        if extra_metric is not None:
            self.extra_metric = extra_metric
            self.extra_recorder = Recorder(self.device.type)

        if self.data_save_mode == 'recorder':
            self.epoch_train_loss = RecorderForEpoch(self.device.type)
            self.epoch_val_loss = RecorderForEpoch(self.device.type)
            self.epoch_train_metric = RecorderForEpoch(self.device.type)
            self.epoch_val_metric = RecorderForEpoch(self.device.type)
            self.epoch_extra_metric = None

            if 'extra_metric' in self.__dict__:
                self.epoch_extra_metric = RecorderForEpoch(self.device.type)

        elif self.data_save_mode == 'csv':
            current_path = os.getcwd()
            filepath = current_path + "\\train_log" if self.system == 'Windows' else current_path + "/train_log"

            self.csv_filename = filepath + '\\log_{}.csv'.format(self.file_time) if self.system == 'Windows' else filepath + '/log_{}.csv'.format(self.file_time)

            if not os.path.exists(filepath):
                os.mkdir(filepath)

            if not os.path.isfile(self.csv_filename):
                if 'extra_metric' not in self.__dict__:
                    with open(self.csv_filename, 'w') as file:
                        writer = csv.writer(file)
                        writer.writerow(['epoch', 'train_loss', 'train_metrics', 'val_loss', 'val_metrics'])

                elif 'extra_metric' in self.__dict__:
                    with open(self.csv_filename, 'w') as file:
                        writer = csv.writer(file)
                        writer.writerow(['epoch', 'train_loss', 'train_metrics', 'val_loss', 'val_metrics', 'val_extra_metrics'])

        print("Epoch:{}, Device: {}".format(epoch, self.device))

    def calculate(self, x, y, mode='t'):
        """
        Performs a forward pass of the model, calculates the loss and metrics.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target data.
            mode (str): Mode of operation - 't' for training, 'v' for validation.

        Returns:
            list: A list containing the loss, metric, and other relevant information based on the mode.

        """
        pi, mu, sigma = self.model(x)

        loss = self.criterion(pi, mu, sigma, y)

        pdf = mixture(pi, mu, sigma)

        y_pred = pdf.sample()

        metric_per = r2_score(y, y_pred)

        return self._standard_return(loss, metric_per, mode, y, y_pred)

    def _standard_return(self, loss, metric_per, mode, y, y_pred):
        """
        Formats the return values based on the operation mode.

        Args:
            loss (torch.Tensor): The calculated loss.
            metric_per (torch.Tensor): The performance metric.
            mode (str): Mode of operation - 't' for training, 'v' for validation.
            y (torch.Tensor): Target data.
            y_pred (torch.Tensor): Predicted data.

        Returns:
            list: A list containing the loss, metric, and other relevant information based on the mode.

        """
        if mode == 't':
            return [loss, metric_per, 't']

        elif mode == 'v' and 'extra_metric' in self.__dict__:
            return [loss, metric_per, y, y_pred, 'v']

        elif mode == 'v':
            return [loss, metric_per, 'v']

    def cal_result(self, *args):
        """
        Updates the loss and metric recorders based on the results from the current epoch.

        Args:
            *args: A variable-length argument list containing the results from the epoch.

        Returns:
            dict: A dictionary containing the updated loss and metrics.

        """
        if args[-1] == 't':
            self.train_loss_recorder.update(args[0])
            self.train_metric_recorder.update(args[1])
            return {
                # 'loss': (self.train_loss_recorder.val[-1].item(), '.4f'),
                'loss_avg': (self.train_loss_recorder.avg().item(), '.4f'),
                'train_metric': (self.train_metric_recorder.avg().item(), '.4f')
            }, args[0]

        elif args[-1] == 'v':
            self.val_loss_recorder.update(args[0])
            self.val_metric_recorder.update(args[1])
            if len(args[:-1]) == 4:
                self.extra_recorder.update(self.extra_metric(args[2], args[3]))
                return {
                    # 'loss': (self.val_loss_recorder.val[-1].item(), '.4f'),
                    'loss_avg': (self.val_loss_recorder.avg().item(), '.4f'),
                    'val_metric': (self.val_metric_recorder.avg().item(), '.4f'),
                    'extra_metric': (self.extra_recorder.avg().item(), '.4f')
                }

            else:
                return {
                    # 'loss': (self.val_loss_recorder.val[-1].item(), '.4f'),
                    'loss_avg': (self.val_loss_recorder.avg().item(), '.4f'),
                    'val_metric': (self.val_metric_recorder.avg().item(), '.4f'),
                }

    def _check_best_metric_for_regression(self, b_m):
        """
        Checks whether the best metric for regression is within valid bounds.

        Args:
            b_m (float): The best metric value for comparison.

        Returns:
            bool: True if the best metric is within valid bounds, False otherwise.

        """
        if b_m >= 1.0:
            return False

        else:
            return True

    def _check_param_exist(self, b_m):
        if b_m is not None:
            return True

        else:
            return False

    def fit(self, t_l, v_l, **kwargs):
        """
        Trains and validates a machine learning model using the provided training and validation data loaders,
        applying specific learning rate schedules and warm-up periods. The function also handles the saving of the model
        based on validation performance metrics.

        Args:
            t_l (DataLoader): DataLoader containing the training data.
            v_l (DataLoader): DataLoader containing the validation data.
            **kwargs: A dictionary of additional keyword arguments:
                m_p (str): Path where the model should be saved.
                w_e (int, optional): Number of initial epochs during which the learning rate is increased linearly.
                l_m (list of int, optional): Epoch indices at which the learning rate should be decreased.
                gamma (float, optional): Multiplicative factor by which the learning rate is decayed at each milestone.
                b_m (float, optional): Threshold for a performance metric (e.g., accuracy) to determine the best model.
                b_l (float, optional): Threshold for the loss value to determine the best model.
                condition(int): Decided by b_m and b_l. If b_m and b_l are not None, condition=2, if b_m is not None and b_l is None, condition=0, if b_m is None and b_l is not None, condition=1, note that this parameter can not be specified.

        Process:
            1. The function initializes required devices and settings for training.
            2. It enters a training loop for the specified number of epochs, handling both training and validation phases.
            3. During the training phase, it optionally applies a warm-up schedule and adjusts the learning rate based on milestones.
            4. Post each epoch, it checks if the current model outperforms previous metrics using the `b_m` value and potentially saves the model.
            5. Outputs training progress and validation metrics after each epoch.

        Returns:
            A tuple of collected metrics over epochs:
            - epoch_train_loss (MetricTracker): Tracker for training loss across epochs.
            - epoch_val_loss (MetricTracker): Tracker for validation loss across epochs.
            - epoch_val_metric (MetricTracker): Tracker for validation metric specified by `b_m`.
            - epoch_train_metric (MetricTracker): Tracker for training metric.
            - epoch_extra_metric (MetricTracker, optional): Additional metrics tracker if `extra_metric` is True.

        Raises:
            ValueError: If required parameters in `kwargs` like `b_m` are missing or if validation metrics exceed expected boundaries.

        """
        if not self._check_param_exist(kwargs['b_m']):
            raise ValueError('Best metric does not exist')

        else:
            if not self._check_best_metric_for_regression(kwargs['b_m']):
                raise ValueError("Best metric can't higher than 1.0")

        IS_WARMUP = False
        IS_LR_MILESTONE = False

        IF_SAVE = False
        CONDITION = kwargs['condition']

        self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.criterion.to(self.device)

        # Schedular 1
        if 'w_e' in kwargs.keys():
            IS_WARMUP = True
            warmup = WarmUpLR(self.optimizer, len(t_l) * kwargs['w_e'])

        # Schedular 2
        if 'l_m' in kwargs.keys():
            IS_LR_MILESTONE = True
            lr_schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                milestones=kwargs['l_m']['s_l'],
                                                                gamma=kwargs['l_m']['gamma'])

        for name, parameters in self.model.named_parameters():
            print(name, ':', parameters.size())

        # Epoch
        for e in range(self.epoch):

            self.model.train()

            if IS_WARMUP is True and IS_LR_MILESTONE is True and e >= kwargs['w_e']:
                lr_schedular.step()

            # lr_schedular.step()

            i = 0

            with tqdm(t_l, unit='batch') as t_epoch:
                for x, y in t_epoch:
                    if IS_WARMUP is True and e < kwargs['w_e']:
                        warmup.step()
                    t_epoch.set_description(f"Epoch {e + 1} Training")

                    mini_batch_x = x.to(self.device)
                    mini_batch_y = y.to(self.device)
                    self.optimizer.zero_grad()

                    param_options, loss = self.cal_result(*self.calculate(mini_batch_x, mini_batch_y, mode='t'))

                    param_options['lr'] = (self.optimizer.state_dict()['param_groups'][0]['lr'], '.6f')

                    params = {key: "{value:{format}}".format(value=value, format=f)
                              for key, (value, f) in param_options.items()}

                    loss.backward()

                    self.optimizer.step()

                    if self.writer is not None:
                        n_iter = (e - 1) * len(t_l) + i + 1
                        visualize_lastlayer(self.writer, self.model, n_iter)
                        visualize_train_loss(self.writer, loss.item(), n_iter)

                    t_epoch.set_postfix(**params)

                # epoch_train_metric.append(self.train_metric_recorder.avg)
                # epoch_train_loss.append(self.train_loss_recorder.avg)

            with torch.no_grad():
                self.model.eval()

                with tqdm(v_l, unit='batch') as v_epoch:
                    v_epoch.set_description(f"Epoch {e + 1} Validating")

                    for v_x, v_y in v_epoch:
                        val_batch_x = v_x.to(self.device)
                        val_batch_y = v_y.to(self.device)

                        param_options = self.cal_result(*self.calculate(val_batch_x, val_batch_y, mode='v'))

                        params = {key: "{value:{format}}".format(value=value, format=f)
                                  for key, (value, f) in param_options.items()}

                        v_epoch.set_postfix(**params)

                if self.data_save_mode == 'recorder':
                    self.epoch_train_metric.update(self.train_metric_recorder.avg().detach())
                    self.epoch_train_loss.update(self.train_loss_recorder.avg().detach())
                    self.epoch_val_loss.update(self.val_loss_recorder.avg().detach())
                    self.epoch_val_metric.update(self.val_metric_recorder.avg().detach())

                    if self.epoch_extra_metric is not None:
                        self.epoch_extra_metric.update(self.extra_recorder.avg().detach())

                elif self.data_save_mode == 'csv':
                    with open(self.csv_filename, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([e + 1, self.train_loss_recorder.avg().detach().item(), self.train_metric_recorder.avg().detach().item(), self.val_loss_recorder.avg().detach().item(), self.val_metric_recorder.avg().detach().item()])

                if self.writer is not None:
                    visualize_test_loss(self.writer, self.epoch_val_loss.val[-1], e)

                if CONDITION == 0:
                    if self.val_metric_recorder.avg().item() >= kwargs['b_m']:
                        kwargs['b_m'] = self.val_metric_recorder.avg().item()
                        IF_SAVE = True

                elif CONDITION == 1:
                    if self.val_loss_recorder.avg().item() <= kwargs['b_l']:
                        kwargs['b_l'] = self.val_loss_recorder.avg().item()
                        IF_SAVE = True

                elif CONDITION == 2:
                    if self.val_loss_recorder.avg().item() <= kwargs['b_l'] and self.val_metric_recorder.avg().item() >= \
                            kwargs['b_m']:
                        kwargs['b_m'] = self.val_metric_recorder.avg().item()
                        kwargs['b_l'] = self.val_loss_recorder.avg().item()
                        IF_SAVE = True

                if 'm_p' in kwargs.keys() and IF_SAVE is True:
                    torch.save(self.model.state_dict(), '{}best_model_{}.pth'.format(kwargs['m_p'], self.file_time))

                    print(
                        "Save Best model: Metric:{:.4f}, Loss Avg:{:.4f}".format(self.val_metric_recorder.avg().item(),
                                                                                 self.val_loss_recorder.avg().item()))
                    IF_SAVE = False

                # else:
                #     print("Best model: R2:{:.4f}, Loss Avg:{:.4f}".format(self.val_metric_recorder.avg().item(),
                #                                                           self.val_loss_recorder.avg().item()))

            self.train_loss_recorder.reset()
            self.val_loss_recorder.reset()
            self.train_metric_recorder.reset()
            self.val_metric_recorder.reset()
            if 'extra_metric' in self.__dict__:
                self.extra_recorder = self.extra_recorder.reset()

            if IS_WARMUP is False and IS_LR_MILESTONE is True:
                lr_schedular.step()

        if 'extra_metric' in self.__dict__ and self.data_save_mode == 'recorder':
            return self.epoch_train_loss, self.epoch_val_loss, self.epoch_val_metric, self.epoch_train_metric, self.epoch_extra_metric

        elif 'extra_metric' not in self.__dict__ and self.data_save_mode == 'recorder':
            return self.epoch_train_loss, self.epoch_val_loss, self.epoch_val_metric, self.epoch_train_metric

        elif self.data_save_mode == 'csv':
            return 'OK'


def config_generator(model_save_path: str, warmup_epochs: int = None, lr_milestones: list = None,
                     lr_decay_rate: float = None, best_metric: float = None, best_loss: float = None, validation_rate: float = 0.2,
                     ):
    """
    Generates a configuration dictionary for model training based on specified parameters.

    Parameters:
        model_save_path (str): Path where the model should be saved.
        warmup_epochs (int, optional): Number of initial epochs during which the learning rate is increased linearly.
        lr_milestones (list of int, optional): Epoch indices at which the learning rate should be decreased.
        lr_decay_rate (float, optional): Multiplicative factor by which the learning rate is decayed at each milestone.
        best_metric (float, optional): Threshold for a performance metric (e.g., accuracy) to determine the best model.
        best_loss (float, optional): Threshold for the loss value to determine the best model.
        validation_rate(float): Fraction of the validation set which split by training data(developing)

    Raises:
        ValueError: If `model_save_path` is None or if `lr_milestones` is set but `lr_decay_rate` is not provided.

    Returns:
        dict: Configuration dictionary containing all settings necessary for model training, including paths,
              learning rate schedules, and performance thresholds.

    This function constructs a dictionary that includes settings for saving the model, applying warmup periods and
    learning rate schedules, and monitoring specific performance metrics or loss values to save the best model during training.

    """
    config = {}

    if model_save_path is None:
        raise ValueError("Please specify the path to save the model, this is used to save the best performance model")

    if validation_rate >= 0.5:
        raise Warning("The validation rate usually is less than 0.5.")

    # config['validation_rate'] = validation_rate

    config['m_p'] = model_save_path

    if best_metric is not None and best_loss is not None:
        config['b_m'] = best_metric
        config['b_l'] = best_loss
        config['condition'] = 2

    elif best_metric is not None and best_loss is None:
        config['b_m'] = best_metric
        config['condition'] = 0

    elif best_metric is None and best_loss is not None:
        config['b_l'] = best_loss
        config['condition'] = 1

    if warmup_epochs is not None:
        config['w_e'] = warmup_epochs

    if lr_milestones is not None and lr_decay_rate is None:
        raise ValueError("Please specify the lr decay rate e.g. 0.7 if you want to use lr decay schedule")

    elif lr_milestones is not None and lr_decay_rate is not None:
        child_dict = {
            's_l': lr_milestones,
            'gamma': lr_decay_rate
        }
        config['l_m'] = child_dict

    return config

