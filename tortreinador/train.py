# Copyright (c) 2025 ArdenteX
# Author: ArdenteX (https://github.com/ArdenteX)
# License: MIT License
# Package Name: Tortreinador
import logging
import warnings
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tortreinador.utils.metrics import r2_score, mixture
from tqdm import tqdm
from tortreinador.utils.Recorder import Recorder, RecorderForEpoch, CheckpointRecorder, MetricManager, MetricDefine
from typing import List, Union, Dict
from tortreinador.utils.View import visualize_lastlayer, visualize_train_loss, visualize_test_loss
from tensorboardX import SummaryWriter
from datetime import datetime
import csv
import os
import platform
import numpy as np
from tortreinador.Events.event_system import EventManager, EventType
from tortreinador.Events.config_register_event import ConfigRegisterEvent


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
        self.current_epoch = None
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

        if metric_manager is not None:
            self.metric_manager = metric_manager

        else:
            self.metric_manager = MetricManager([MetricDefine('loss', torch.tensor(0.0), 0)])

        self.recorders = {}
        for metric in self.metric_manager.metric_names.tolist():
            self.recorders[metric] = Recorder(self.device.type)

        self.event_manager = EventManager()
        self.subscribe(event_type=EventType.TRAIN_INIT, event=ConfigRegisterEvent())

        self.timestamp = self.get_current_time()

        print("Epoch:{}, Device: {}".format(epoch, self.device))

    def subscribe(self, event_type: Union[EventType, List[EventType]], event):
        self.event_manager.subscribe(event_type, event)

    def trigger(self, event_type: EventType, **kwargs):
        for event in self.event_manager.listener[event_type]:
            event.on_fire(event_type, trainer=self, **kwargs)

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

    def _standard_return(self, mode: int = None, update_values: Union[List, Dict] = None):
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

        current_mode_metrics, current_mode_idx = self.metric_manager.get_metrics_by_mode(mode, both=True)

        # print(current_mode_idx)

        for c_i, c_m in zip(current_mode_idx, current_mode_metrics):
            current_value = c_m.metric_value.detach()
            self.recorders[self.metric_manager.metric_names[c_i]].update(current_value)

        return {
            '{}'.format(k): (v.avg().item(), '.4f') for k, v in zip(self.metric_manager.metric_names[current_mode_idx], [self.recorders[self.metric_manager.metric_names[c_idx]] for c_idx in current_mode_idx])
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

    def get_current_time(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M').replace(":", "").replace("-", '').replace(
            " ", '')

    def _initial_csv_mode(self):
        """
        Prepare CSV logging by creating a timestamped file path and header.

        Returns:
            tuple: (csv_filename, file_time) where `csv_filename` is the path to the log file and
        """
        # file_time = self.get_current_time()
        current_path = os.getcwd()

        filepath = os.path.join(current_path, 'train_log')

        csv_filename = os.path.join(filepath, 'log_{}.csv'.format(
            self.timestamp))

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
        return csv_filename

    def fit(self, t_l, v_l, checkpoint_=None, tqdm_disable=False, **kwargs):
        """
        Execute the training/validation loop with optional checkpoint reloads and LR schedules.

        Args:
            t_l: Training dataloader.
            v_l: Validation dataloader.
            checkpoint_: Serialized checkpoint dictionary. When provided, its ``config`` field overwrites
                ``kwargs`` and the checkpoint weights/optimizer state are restored automatically.
            tqdm_disable: Turn on tqdm or not
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
        kwargs['lr_schedule']['dataset_length'] = len(t_l)

        self.event_manager.trigger(EventType.TRAIN_INIT, trainer=self, **kwargs)
        if checkpoint_ is not None:
            kwargs = checkpoint_['config']


        csv_filename = None
        # file_time = self.get_current_time()
        # if self.data_save_mode == 'csv' and kwargs['train_mode'] == 'new':
        #     csv_filename = self._initial_csv_mode()

        if kwargs['train_mode'] == 'new' and 'm_p' in kwargs.keys():
            CHECK_POINT_PATH = os.path.join(kwargs['m_p'], 'check_point_{}.pth'.format(self.timestamp))
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

        START_EPOCH = kwargs['start_epoch']

        VAL_COUNT = 1
        VAL_CYCLE = kwargs['val_cycle']

        # self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.criterion.to(self.device)

        # for name, parameters in self.model.named_parameters():
        #     print(name, ':', parameters.size())

        for e in range(START_EPOCH, self.epoch):
            # TRAIN_EPOCH_START
            self.model.train()
            self.current_epoch = e
            self.event_manager.trigger(EventType.TRAIN_EPOCH_START, trainer=self)

            i = 0
            if 'mix' in kwargs.keys():
                if kwargs['mix']['condition'] == 1 and e <= kwargs['mix']['warmup_error']:
                    self.current_error_rate = (FINAL_RATE - INIT_RATE) * (
                                e / (kwargs['mix']['warmup_error'] - 1)) + INIT_RATE

                elif kwargs['mix']['condition'] == 2 and e >= kwargs['mix']['warmup_error']:

                    self._random_event(EVENT_RATE)

            with tqdm(t_l, unit='batch', disable=tqdm_disable) as t_epoch:
                for batch_idx, (x, y) in enumerate(t_epoch):
                    # TRAIN_BATCH_START

                    t_epoch.set_description(f"Epoch {e + 1} Training")
                    try:
                        mini_batch_x = x.to(self.device)
                        mini_batch_y = y.to(self.device)

                    except AttributeError:
                        mini_batch_x = x
                        mini_batch_y = y
                        # self.trigger(event_type=EventType.INFO,
                        #                 **{'msg': "Seems that there was something wrong when transferring the x or y to CUDA, the device of x and y, which will convert to the calculate() is CPU", 'prefix': 'Trainer'})
                    self.optimizer.zero_grad()

                    cal = self.calculate(mini_batch_x, mini_batch_y, mode=1)

                    param_options, loss = self.cal_result(mode=cal)
                    # TRAIN_BATCH_CALCULATION_END

                    param_options['lr'] = (self.optimizer.state_dict()['param_groups'][0]['lr'], '.6f')

                    params = {key: "{value:{format}}".format(value=value, format=f)
                              for key, (value, f) in param_options.items()}
                    # TRAIN_BATCH_METRIC_COLLECTION_COMPLETE
                    if 'mix' in kwargs.keys() and kwargs['mix']['condition'] == 2:
                        params['event'] = ("Occurs" if self.event_occurs else "Not Occurs")

                    loss.backward()

                    self.optimizer.step()

                    # TRAIN_BATCH_END
                    self.event_manager.trigger(EventType.TRAIN_BATCH_END, trainer=self, batch_idx=batch_idx)

                    if self.writer is not None:
                        n_iter = (e - 1) * len(t_l) + i + 1
                        visualize_lastlayer(self.writer, self.model, n_iter)
                        visualize_train_loss(self.writer, loss.item(), n_iter)

                    t_epoch.set_postfix(**params)

                # epoch_train_metric.append(self.train_metric_recorder.avg)
                # epoch_train_loss.append(self.train_loss_recorder.avg)
            # VALIDATION_START
            if VAL_COUNT % VAL_CYCLE == 0:

                VAL_COUNT = 1

                with torch.no_grad():
                    self.model.eval()

                    with tqdm(v_l, unit='batch', disable=tqdm_disable) as v_epoch:
                        # VALIDATION_BATCH_START
                        v_epoch.set_description(f"Epoch {e + 1} Validating")

                        for v_x, v_y in v_epoch:
                            try:
                                val_batch_x = v_x.to(self.device)
                                val_batch_y = v_y.to(self.device)

                            except AttributeError:
                                val_batch_x = x
                                val_batch_y = y

                            param_options, _ = self.cal_result(self.calculate(val_batch_x, val_batch_y, mode=2))

                            params = {key: "{value:{format}}".format(value=value, format=f)
                                      for key, (value, f) in param_options.items()}

                            # VALIDATION_BATCH_METRIC_COLLECTION_COMPLETE
                            v_epoch.set_postfix(**params)

                            # VALIDATION_BATCH_END

                    # VALIDATION_END
                    self.event_manager.trigger(EventType.VALIDATION_END, trainer=self)

            else:
                VAL_COUNT += 1

            # TRAIN_EPOCH_END
            self.event_manager.trigger(EventType.TRAIN_EPOCH_END, trainer=self)

            # if self.writer is not None:
            #     visualize_test_loss(self.writer, val_loss_recorder.val[-1], e)

            for recorder in self.recorders.values():
                recorder.reset()

        # TRAIN_COMPLETE
        if self.data_save_mode == 'recorder':
            return self.event_manager.listener[EventType.TRAIN_EPOCH_END_RECORD]

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
                     restart_mode='batch', logger_on: bool = False, logger: logging.Logger = None, logger_level: int = None,
                     logger_file_max_bytes: int = 10 * 1024 * 1024, logger_file_backup_count: int = 5, logger_dir: str = None):
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
        logger_on:
        logger:
        logger_level:
        logger_file_max_bytes:
        logger_file_backup_count:
        logger_dir:

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

    config['best_model_detection'] = {
        'b_m': 0.0,
        'b_l': 0.0,
        'condition': -1
    }

    if best_metric is not None and best_loss is not None:
        config['best_model_detection']['b_m'] = best_metric
        config['best_model_detection']['b_l'] = best_loss
        config['best_model_detection']['condition'] = 2

    elif best_metric is not None and best_loss is None:
        config['best_model_detection']['b_m'] = best_metric
        config['best_model_detection']['condition'] = 0

    elif best_metric is None and best_loss is not None:
        config['best_model_detection']['b_l'] = best_loss
        config['best_model_detection']['condition'] = 1

    elif best_loss is None and best_metric is None:
        warnings.warn("Seems you don't assign either the best metric or the best loss, so that the best model will not be detected while training.")

    if lr_milestones is not None and lr_decay_rate is None:
        raise ValueError("Please specify the lr decay rate e.g. 0.7 if you want to use lr decay schedule")

    config['lr_schedule'] = {
        'on': True if warmup_epochs is not None or lr_milestones is not None or lr_restart else False,
        'warmup': {
            'on': True if warmup_epochs is not None else False,
            'warmup_epochs': warmup_epochs,
        },
        'lr_milestones': {
            'on': True if lr_milestones is not None else False,
            'stone_list': lr_milestones,
            'gamma': lr_decay_rate,
        },
        'lr_restart': {
            'on': True if lr_restart else False,
            't_0': T_0,
            't_mult': t_mult,
            'eta_min': eta_min,
            'mode': restart_mode
        }
    }

    config['logger'] = {
        'on': logger_on,
        'logger': logger,
        'level': logger_level,
        'logger_dir': logger_dir,
        'logger_file_max_bytes': logger_file_max_bytes,
        'logger_file_backup_count': logger_file_backup_count
    }

    return config


