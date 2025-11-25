import warnings
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tortreinador.utils.metrics import r2_score, mixture
from tqdm import tqdm
from tortreinador.utils.Recorder import Recorder, RecorderForEpoch, CheckpointRecorder, MetricManager, MetricDefine
from tortreinador.utils.WarmUpLR import WarmUpLR
from tortreinador.utils.View import visualize_lastlayer, visualize_train_loss, visualize_test_loss
from tensorboardX import SummaryWriter
from datetime import datetime
import csv
import os
import platform
import numpy as np
from typing import Union, List, Tuple, Dict, Literal


class TorchTrainer:
    """
    High level training utility that wires a PyTorch model, optimizer, and loss into a
    metric-aware training/validation loop with optional checkpointing and logging.

    The trainer keeps a `MetricManager` plus per-metric `Recorder` instances that gather batch
    statistics, and it can either persist those numbers in memory (`recorder` mode) or to CSV
    files. Learning-rate warmup, multi-step decay, cosine restarts, and two mix strategies are
    supported out of the box.
    """

    def __init__(self,
                 is_gpu: bool = True,
                 epoch: int = 150, log_dir: str = None, model: nn.Module = None,
                 optimizer: Optimizer = None, metric_manager: MetricManager = None, criterion: nn.Module = None,
                 data_save_mode: str = 'recorder'):
        """
        Build a trainer with the core components required for the optimization loop.

        Args:
            is_gpu: Whether CUDA should be used when available.
            epoch: Maximum number of epochs to run.
            log_dir: Destination directory for TensorBoard summaries. When ``None`` logging is disabled.
            model: PyTorch module to train.
            optimizer: Optimizer configured for ``model``.
            metric_manager: MetricManager describing which metrics should be tracked. A default manager
                that only tracks loss is created when ``None`` is supplied.
            criterion: Callable loss object that matches the model outputs.
            data_save_mode: Selects how epoch statistics are persisted. ``'recorder'`` keeps the data in
                memory, ``'csv'`` saves into ``train_log/log_<timestamp>.csv``.

        Raises:
            ValueError: When mandatory components (model, optimizer, criterion, or epoch) are missing or
                when ``data_save_mode`` falls outside the supported values.
        """

        if not isinstance(model, nn.Module) or not isinstance(optimizer, Optimizer) or not isinstance(criterion,
                                                                                                      nn.Module) or epoch is None:
            raise ValueError("Please provide the correct type of model, optimizer, criterion and the not none epoch")

        data_save_mode_list = ['recorder', 'csv', None]
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

        self.device = torch.device("cuda" if torch.cuda.is_available() and is_gpu else "cpu")

        self.writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else log_dir

        self.checkpoint_recorder = None

        # mix strategy 1
        self.current_error_rate = None

        # mix strategy 2
        self.event_occurs = False
        self.lambda_factor = None


        # self.train_loss_recorder = Recorder(self.device.type)
        # self.val_loss_recorder = Recorder(self.device.type)
        # self.train_metric_recorder = Recorder(self.device.type)
        # self.val_metric_recorder = Recorder(self.device.type)

        if metric_manager is not None:
            self.metric_manager = metric_manager

        else:
            self.metric_manager = MetricManager([MetricDefine('loss', torch.tensor(0.0), 0)])

        self.recorders = [
            Recorder(self.device.type) for i in range(len(self.metric_manager.metric_list))
        ]
        self.recorders = np.array(self.recorders)

        if self.data_save_mode == 'recorder':
            self.recorder_for_epoch = [
            RecorderForEpoch(self.device.type) for i in range(len(self.metric_manager.metric_list))
        ]
            self.recorder_for_epoch = np.array(self.recorder_for_epoch)

            # self.epoch_train_loss = RecorderForEpoch(self.device.type)
            # self.epoch_val_loss = RecorderForEpoch(self.device.type)
            # self.epoch_train_metric = RecorderForEpoch(self.device.type)
            # self.epoch_val_metric = RecorderForEpoch(self.device.type)
            # self.epoch_extra_metric = None

        print("Epoch:{}, Device: {}".format(epoch, self.device))

    def calculate(self, x, y, mode=1):
        """
        Forward pass helper that computes metrics and updates the MetricManager cache.

        Args:
            x: Batch of inputs for the model.
            y: Batch of targets that match ``x``.
            mode: 1 when called from the training loop and 2 for validation. The mode determines which
                metric slots get updated.

        Returns:
            int: The ``mode`` value, propagated so callers can feed it back into ``cal_result``.
        """
        pi, mu, sigma = self.model(x)

        loss = self.criterion(pi, mu, sigma, y)

        pdf = mixture(pi, mu, sigma)

        y_pred = pdf.sample()

        r2_result = r2_score(y, y_pred)
        metric_return = [loss, r2_result]

        return self._standard_return(mode=mode, update_values=metric_return)

    def _standard_return(self, mode: int = None, update_values: list = None):
        """
        Update the metric manager for a given mode and return the mode id.

        Args:
            mode: 1 for training or 2 for validation. Any other value results in a ValueError.
            update_values: Iterable that matches ``metric_manager.metric_list`` for the provided ``mode``.
        """
        if mode not in [1, 2]:
            raise ValueError("Unexpected value for mode: {}, please input either 1 or 2.".format(mode))

        if not isinstance(mode, int):
            raise ValueError("Unexpected type for mode: {}, please input correct type (int).".format(type(mode)))

        self.metric_manager.update(update_values, mode=mode)
        return mode

    def _dict_return(self, return_dict):
        """
        Returns a metric dictionary without modification.

        Args:
            return_dict (dict): Dictionary that already contains formatted metric values.

        Returns:
            dict: The same dictionary passed in, enabling a consistent return signature.
        """
        return return_dict

    def cal_result(self, mode):
        """
        Consolidate per-batch metrics into averaged statistics for the requested mode.

        Args:
            mode: 1 for training or 2 for validation. Determines which subset of metrics get read.

        Returns:
            tuple: (formatted_metrics, criterion_value) where ``formatted_metrics`` is a dict that maps
            metric names to (value, print_format) tuples and ``criterion_value`` is the metric selected
            as the optimization target.
        """

        current_mode_idx = self.metric_manager.get_metrics_by_mode(mode, idx=True)
        current_mode_metrics = self.metric_manager.get_metrics_by_mode(mode)

        # print(current_mode_idx)

        for c_i, c_m in zip(current_mode_idx, current_mode_metrics):
            self.recorders[c_i].update(c_m.metric_value)

        return {
            '{}'.format(k): (v.avg().item(), '.4f') for k, v in zip(self.metric_manager.metric_names[current_mode_idx], self.recorders[current_mode_idx])
        }, self.metric_manager.metric_list[self.metric_manager.criterion_idx].metric_value

    def _check_best_metric_for_regression(self, b_m):
        """
        Validate that the stored "best metric" sits inside expected regression bounds.

        Args:
            b_m: Candidate best metric value.

        Returns:
            bool: True if the metric is < 1.0, False otherwise.
        """
        if b_m >= 1.0:
            return False

        else:
            return True

    def _check_param_exist(self, b_m):
        """
        Check whether an optional metric baseline parameter has been provided.

        Args:
            b_m: Best metric threshold or ``None``.

        Returns:
            bool: True when the parameter is not None.
        """
        if b_m is not None:
            return True

        else:
            return False

    def _random_event(self, event_rate):
        """
        Samples a Bernoulli-style random event to decide whether mix strategy 2 should trigger.

        Args:
            event_rate (float): Probability threshold between 0 and 1.

        Side Effects:
            Updates `self.event_occurs` to True when the sampled value is below the event rate.
        """
        event_ = np.random.rand()

        if event_ <= event_rate:
            self.event_occurs = True

        else:
            self.event_occurs = False

    def _initial_csv_mode(self):
        """
        Prepare CSV logging by creating a timestamped file path and header.

        Returns:
            tuple: (csv_filename, file_time) where `csv_filename` is the path to the log file and
            `file_time` is the timestamp suffix used for consistent naming.
        """
        file_time = datetime.now().strftime('%Y-%m-%d %H:%M').replace(":", "").replace("-", '').replace(
            " ", '')
        current_path = os.getcwd()

        filepath = os.path.join(current_path, 'train_log')

        csv_filename = os.path.join(filepath, 'log_{}.csv'.format(
            file_time))

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        if not os.path.isfile(csv_filename):
            if 'extra_metric' not in self.__dict__:
                with open(csv_filename, 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(['epoch', 'train_loss', 'train_metrics', 'val_loss', 'val_metrics'])

            elif 'extra_metric' in self.__dict__:
                with open(csv_filename, 'w') as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ['epoch', 'train_loss', 'train_metrics', 'val_loss', 'val_metrics', 'val_extra_metrics'])
        return csv_filename, file_time

    def fit(self, t_l, v_l, checkpoint_=None, **kwargs):
        """
        Execute the training/validation loop with optional checkpoint reloads and LR schedules.

        Args:
            t_l: Training dataloader.
            v_l: Validation dataloader.
            checkpoint_: Serialized checkpoint dictionary. When provided, its ``config`` field overwrites
                ``kwargs`` and the checkpoint weights/optimizer state are restored automatically.
            **kwargs: Runtime configuration. Important keys include:
                - ``train_mode`` (``'new'`` or ``'reload'``): Selects whether to start fresh or continue.
                - ``m_p``: Directory used to save checkpoints.
                - ``start_epoch``: Epoch index to begin training from.
                - ``w_e``: Warmup epochs for :class:`WarmUpLR`.
                - ``l_m``: Dict containing ``'s_l'`` milestones and ``'gamma'`` decay for MultiStepLR.
                - ``lr_restart``: Dict describing cosine restart hyper-parameters and ``mode`` (batch/epoch).
                - ``mix``: Dict describing mix strategy parameters. Two modes are supported: ramping the
                  target error (condition 1) and event-based sampling (condition 2).
                - ``b_m`` / ``b_l``: Validation metric and loss thresholds that control best-model saving.
                - ``condition``: Encodes which of ``b_m``/``b_l`` is considered (0 metric, 1 loss, 2 both).
                - ``auto_save``: Interval (epochs) for periodic checkpointing.
                - ``val_cycle``: Validate every ``val_cycle`` epochs instead of each epoch.
                - ``train_mode`` specific keys like ``log_dir`` when ``data_save_mode`` is ``'csv'``.

        Returns:
            list[RecorderForEpoch] | str: Recorder list with epoch level summaries when ``data_save_mode``
            is ``'recorder'``. When ``'csv'`` mode is used the literal string ``'OK'`` indicates success.
        """

        # TRAIN_INIT
        if checkpoint_ is not None:
            kwargs = checkpoint_['config']

        # if not self._check_param_exist(kwargs['b_m']):
        #     raise ValueError('Best metric does not exist')

        # if self._check_param_exist(kwargs['b_m']):
        #     if not self._check_best_metric_for_regression(kwargs['b_m']):
        #         raise ValueError("Best metric can't higher than 1.0")

        csv_filename = None
        file_time = None
        if self.data_save_mode == 'csv' and kwargs['train_mode'] == 'new':
            csv_filename, file_time = self._initial_csv_mode()

        if kwargs['train_mode'] == 'new' and 'm_p' in kwargs.keys():
            CHECK_POINT_PATH = os.path.join(kwargs['m_p'], 'check_point_{}.pth'.format(file_time))
            self.checkpoint_recorder = CheckpointRecorder(CHECK_POINT_PATH, self.epoch, kwargs, csv_filename,
                                                          mode='new',
                                                          system=self.system)

        elif kwargs['train_mode'] == 'reload':
            self.checkpoint_recorder = CheckpointRecorder(checkpoint_, mode='reload')
            csv_filename = kwargs['log_dir']
            self.checkpoint_recorder.reload(self.model, self.optimizer)

        INIT_RATE = None
        FINAL_RATE = None

        EVENT_RATE = None
        if 'mix' in kwargs.keys():
            if kwargs['mix']['condition'] == 1:
                INIT_RATE = kwargs['mix']['initial_rate']
                FINAL_RATE = kwargs['mix']['final_rate']

            elif kwargs['mix']['condition'] == 2:
                EVENT_RATE = kwargs['mix']['event_rate']
                self.current_error_rate = kwargs['mix']['final_rate']
                self.lambda_factor = kwargs['mix']['lambda_factor']

        IS_WARMUP = False
        IS_LR_MILESTONE = False
        IS_LR_RESTART = False

        IF_SAVE = False
        CONDITION = kwargs['condition']
        START_EPOCH = kwargs['start_epoch']
        AUTO_SAVE = kwargs['auto_save']
        AUTO_COUNT = 1

        VAL_COUNT = 1
        VAL_CYCLE = kwargs['val_cycle']

        # self.model = nn.DataParallel(self.model)

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

        if 'lr_restart' in kwargs.keys():
            IS_LR_RESTART = True
            restart_schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                     T_0=kwargs['lr_restart']['t_0'],
                                                                                     T_mult=kwargs['lr_restart'][
                                                                                         't_mult'],
                                                                                     eta_min=kwargs['lr_restart'][
                                                                                         'eta_min'])

        # for name, parameters in self.model.named_parameters():
        #     print(name, ':', parameters.size())


        for e in range(START_EPOCH, self.epoch):
            # TRAIN_EPOCH_START
            self.model.train()

            if IS_WARMUP and IS_LR_MILESTONE is True and e >= kwargs['w_e']:
                lr_schedular.step()

            if IS_WARMUP and IS_LR_RESTART is True and e >= kwargs['w_e'] and kwargs['lr_restart']['mode'] == 'epoch':
                restart_schedular.step()

            # lr_schedular.step()

            i = 0
            if 'mix' in kwargs.keys():
                if kwargs['mix']['condition'] == 1 and e <= kwargs['mix']['warmup_error']:
                    self.current_error_rate = (FINAL_RATE - INIT_RATE) * (
                                e / (kwargs['mix']['warmup_error'] - 1)) + INIT_RATE

                elif kwargs['mix']['condition'] == 2 and e >= kwargs['mix']['warmup_error']:

                    self._random_event(EVENT_RATE)

            with tqdm(t_l, unit='batch') as t_epoch:
                for batch_idx, (x, y) in enumerate(t_epoch):
                    # TRAIN_BATCH_START

                    t_epoch.set_description(f"Epoch {e + 1} Training")

                    mini_batch_x = x.to(self.device)
                    mini_batch_y = y.to(self.device)
                    self.optimizer.zero_grad()

                    cal = self.calculate(mini_batch_x, mini_batch_y, mode=1)
                    # TRAIN_BATCH_CALCULATION_END

                    param_options, loss = self.cal_result(mode=cal)

                    param_options['lr'] = (self.optimizer.state_dict()['param_groups'][0]['lr'], '.6f')

                    params = {key: "{value:{format}}".format(value=value, format=f)
                              for key, (value, f) in param_options.items()}
                    # TRAIN_BATCH_METRIC_COLLECTION_COMPLETE
                    if 'mix' in kwargs.keys() and kwargs['mix']['condition'] == 2:
                        params['event'] = ("Occurs" if self.event_occurs else "Not Occurs")

                    loss.backward()

                    self.optimizer.step()

                    # TRAIN_BATCH_END
                    if IS_WARMUP is True and e < kwargs['w_e']:
                        warmup.step()

                    if not IS_WARMUP and IS_LR_RESTART and kwargs['lr_restart']['mode'] == 'batch':
                        restart_schedular.step(e + batch_idx / len(t_l))

                    if IS_WARMUP and IS_LR_RESTART is True and e >= kwargs['w_e'] and kwargs['lr_restart'][
                        'mode'] == 'batch':
                        restart_schedular.step(e - kwargs['w_e'] + 1 + batch_idx / len(t_l))

                    if self.writer is not None:
                        n_iter = (e - 1) * len(t_l) + i + 1
                        visualize_lastlayer(self.writer, self.model, n_iter)
                        visualize_train_loss(self.writer, loss.item(), n_iter)

                    t_epoch.set_postfix(**params)

                # epoch_train_metric.append(self.train_metric_recorder.avg)
                # epoch_train_loss.append(self.train_loss_recorder.avg)
            # TRAIN_EPOCH_END
            if VAL_COUNT % VAL_CYCLE == 0:

                VAL_COUNT = 1

                with torch.no_grad():
                    self.model.eval()
                    # VALIDATION_START
                    with tqdm(v_l, unit='batch') as v_epoch:
                        # VALIDATION_BATCH_START
                        v_epoch.set_description(f"Epoch {e + 1} Validating")

                        for v_x, v_y in v_epoch:
                            val_batch_x = v_x.to(self.device)
                            val_batch_y = v_y.to(self.device)

                            param_options, _ = self.cal_result(self.calculate(val_batch_x, val_batch_y, mode=2))

                            params = {key: "{value:{format}}".format(value=value, format=f)
                                      for key, (value, f) in param_options.items()}

                            # VALIDATION_BATCH_METRIC_COLLECTION_COMPLETE
                            v_epoch.set_postfix(**params)

                            # VALIDATION_BATCH_END

                    # VALIDATION_END
                    if self.data_save_mode == 'recorder':

                        for i in range(len(self.recorder_for_epoch)):
                            self.recorder_for_epoch[i].update(self.recorders[i].avg().detach())

                    elif self.data_save_mode == 'csv':
                        with open(csv_filename, 'a', newline='') as file:
                            writer = csv.writer(file)
                            # writer.writerow([e + 1, self.train_loss_recorder.avg().detach().item(),
                            #                  self.train_metric_recorder.avg().detach().item(),
                            #                  self.val_loss_recorder.avg().detach().item(),
                            #                  self.val_metric_recorder.avg().detach().item()])

                            writer.writerow([e + 1] + [r.avg().detach().item() for r in self.recorders])

                    val_loss_recorder = self.recorders[self.metric_manager.criterion_idx]
                    val_baseline_metric = self.recorders[self.metric_manager.baseline_metric_idx]

                    val_loss = val_loss_recorder.avg().item()
                    val_metric = val_baseline_metric.avg().item()

                    if self.writer is not None:
                        visualize_test_loss(self.writer, val_loss_recorder.val[-1], e)

                    if CONDITION == 0:
                        if val_metric > kwargs['b_m']:
                            kwargs['b_m'] = val_metric
                            IF_SAVE = True

                    elif CONDITION == 1:
                        if val_loss < kwargs['b_l']:
                            kwargs['b_l'] = val_loss
                            IF_SAVE = True

                    elif CONDITION == 2:
                        if val_loss < kwargs[
                            'b_l'] and val_metric > \
                                kwargs['b_m']:
                            kwargs['b_m'] = val_metric
                            kwargs['b_l'] = val_loss
                            IF_SAVE = True

                        elif val_loss < kwargs[
                            'b_l'] and val_metric < \
                                kwargs['b_m']:
                            abs_dis = np.abs((kwargs['b_m'] - val_metric) / kwargs['b_m'])
                            if 0.001 < abs_dis < 0.003:
                                kwargs['b_m'] = val_metric
                                kwargs['b_l'] = val_loss
                                IF_SAVE = True

                    if 'm_p' in kwargs.keys() and IF_SAVE is True:
                        self.checkpoint_recorder.update_by_condition(CONDITION,
                                                                     b_m=kwargs['b_m'] if 'b_m' in kwargs.keys() else None,
                                                                     b_l=kwargs['b_l'] if 'b_l' in kwargs.keys() else None)
                        self.checkpoint_recorder.update(e, model=self.model.state_dict(),
                                                        current_optimizer_sd=self.optimizer.state_dict(), mode='best')
                        AUTO_COUNT = 1
                        print(
                            "Save Best model: Metric:{:.4f}, Loss Avg:{:.4f}".format(
                                val_metric,
                                val_loss))
                        IF_SAVE = False

                    if IF_SAVE:
                        print(
                            "Best model Detected: Metric:{:.4f}, Loss Avg:{:.4f}".format(
                                val_metric,
                                val_loss))

            else:
                VAL_COUNT += 1

            if 'm_p' in kwargs.keys():
                if AUTO_COUNT % AUTO_SAVE == 0:
                    self.checkpoint_recorder.update(e, model=self.model.state_dict(),
                                                    current_optimizer_sd=self.optimizer.state_dict())
                    AUTO_COUNT = 1
                else:
                    AUTO_COUNT += 1

                # else:
                #     print("Best model: R2:{:.4f}, Loss Avg:{:.4f}".format(self.val_metric_recorder.avg().item(),
                #                                                           self.val_loss_recorder.avg().item()))

            for recorder in self.recorders:
                recorder.reset()

            if IS_WARMUP is False and IS_LR_MILESTONE is True:
                lr_schedular.step()

            if IS_WARMUP is False and IS_LR_RESTART and kwargs['lr_restart']['mode'] == 'epoch':
                restart_schedular.step()

        # TRAIN_COMPLETE
        if self.data_save_mode == 'recorder':
            return self.recorder_for_epoch

        elif self.data_save_mode == 'csv':
            return 'OK'

    def continue_fit(self, t_l, v_l, checkpoint_path):
        """
        Resume training from an on-disk checkpoint produced by :meth:`fit`.

        Args:
            t_l: Training dataloader.
            v_l: Validation dataloader.
            checkpoint_path: Path to the ``.pth`` checkpoint produced by :class:`CheckpointRecorder`.

        Returns:
            Same as :meth:`fit`. The checkpoint config is injected so no additional kwargs are required.
        """
        checkpoint_ = torch.load(checkpoint_path)
        self.fit(t_l, v_l, checkpoint_)


def config_generator(model_save_path: str=None, warmup_epochs: int = None, lr_milestones: list = None,
                     lr_decay_rate: float = None, best_metric: float = None, best_loss: float = None,
                     validation_cycle: int = 1,
                     auto_save: int = None, if_mix: bool = False, initial_rate: float = None, final_rate: float = None,
                     warmup_error: int = None, event_rate: float = None,
                     lambda_factor: float = 0.7,
                     lr_restart: bool = False, T_0: int = 10, t_mult: float = 1, eta_min: float = 0.00001,
                     restart_mode='batch'):
    """
    Build a ``fit`` configuration dictionary using explicit keyword arguments.

    Args:
        model_save_path: Directory for checkpoints. Required when ``auto_save`` is used.
        warmup_epochs: Number of warmup epochs for :class:`WarmUpLR`.
        lr_milestones: Epoch indices for MultiStepLR.
        lr_decay_rate: Multiplicative decay applied at each milestone.
        best_metric: Initial threshold for the validation metric used to detect improvements.
        best_loss: Initial threshold for validation loss.
        validation_cycle: Validate every ``validation_cycle`` epochs.
        auto_save: Save a checkpoint every ``auto_save`` epochs. Only active when ``model_save_path`` is set.
        if_mix: Toggle mix strategy use. Additional keyword arguments define the chosen strategy.
        initial_rate: Starting error rate for mix strategy 1.
        final_rate: Target error rate for mix strategies.
        warmup_error: Number of epochs before mix strategy transitions to steady state.
        event_rate: Probability that mix strategy 2 triggers a synthetic mix event.
        lambda_factor: Mixing coefficient for strategy 2.
        lr_restart: Whether to configure cosine annealing restarts.
        T_0: Number of iterations for the first cosine period.
        t_mult: Multiplicative increase applied to ``T_0`` for every restart.
        eta_min: Minimum learning rate for cosine restart schedules.
        restart_mode: ``'batch'`` or ``'epoch'``; determines when the restart scheduler steps.

    Returns:
        dict: Fully-populated configuration dictionary ready to be passed to :meth:`TorchTrainer.fit`.

    Raises:
        ValueError: If mutually exclusive mix parameters are provided or when learning-rate milestones
        are given without a decay rate.
    """
    config = {}

    if restart_mode not in ['batch', 'epoch']:
        raise ValueError("Restart mode must be either 'batch' or 'epoch'.")

    # config['validation_rate'] = validation_rate

    if model_save_path is not None:
        config['m_p'] = model_save_path

    config['start_epoch'] = 0
    config['train_mode'] = 'new'
    config['auto_save'] = auto_save
    config['if_mix'] = if_mix
    config['val_cycle'] = validation_cycle

    if model_save_path is None and auto_save is not None:
        warnings.warn("model_save_path does not exist while specifying auto_save. This will not cause any crash, but the auto save function will not effect.")

    if if_mix:
        config['mix'] = {}

        if initial_rate is None and final_rate is None and event_rate is None:
            raise ValueError(
                'Please specify initial_rate, final_rate and warmup_error as mix strategy 1 or specify event_rate as mix strategy 2')

        if initial_rate is not None and final_rate is not None and event_rate is not None:
            raise ValueError("Can't specify both (initial_rate and final_rate) and event_rate at the same time")

        if initial_rate is not None and final_rate is not None and event_rate is None:
            config['mix']['initial_rate'] = initial_rate
            config['mix']['final_rate'] = final_rate
            config['mix']['warmup_error'] = 10 if warmup_error is None else warmup_error
            # config['mix']['error'] = error
            config['mix']['condition'] = 1

        elif initial_rate is None and event_rate is not None:
            config['mix']['event_rate'] = event_rate
            config['mix']['condition'] = 2
            config['mix']['warmup_error'] = 5 if warmup_error is None else warmup_error
            # config['mix']['error'] = error
            config['mix']['lambda_factor'] = lambda_factor
            if final_rate is None:
                config['mix']['final_rate'] = 0.8
                print(
                    "Warning: The mix data training strategy 2 is turn on, however the final rate is unspecified, set it as 0.8")

            else:
                config['mix']['final_rate'] = final_rate


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

    if lr_restart:
        child_dict = {
            't_0': T_0,
            't_mult': t_mult,
            'eta_min': eta_min,
            'mode': restart_mode
        }
        config['lr_restart'] = child_dict

    return config




