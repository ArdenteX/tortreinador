from typing import Union, Dict, List
import torch
import optuna
from optuna import Trial
from tortreinador.train import config_generator
from tortreinador.utils.Recorder import MetricManager
from tortreinador.utils.preprocessing import load_data, get_dataloader
import pandas as pd
import gc
from tortreinador.ParamOptimization.seach_task import Task
from tortreinador.ParamOptimization.hyperparam import IntParam, FloatParam, LogFloatParam, ChoiceParam, HyperParam
import logging
from tortreinador.Events.event_system import EventManager, EventType
from tortreinador.Events.logger_event import LoggerEvent


def _optuna_suggestion_mapping(trial, param_name, param_obj):
    if isinstance(param_obj, IntParam):
        return trial.suggest_int(param_name, param_obj.low, param_obj.high)

    elif isinstance(param_obj, FloatParam):
        return trial.suggest_float(param_name, param_obj.low, param_obj.high)

    elif isinstance(param_obj, LogFloatParam):
        return trial.suggest_float(param_name, param_obj.low, param_obj.high, log=True)

    elif isinstance(param_obj, ChoiceParam):
        return trial.suggest_categorical(param_name, param_obj.choices)

    else:
        return param_obj

class TaskManager:
    def __init__(self, tasks: Union[Task, List[Task]], level: int = logging.INFO, log_dir: str = None,
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):

        self.tasks = tasks
        self.dataset_needs_search = False
        self.model_needs_search = False
        self.optimizer_needs_search = False
        self.trainer_config_needs_search = False

        self.hpo_event_manager = EventManager()
        self.hpo_event_manager.subscribe(event_type=EventType.INFO, event=LoggerEvent(logging.getLogger('Tortreinador.HPO'),
                                                                                      level=level, log_dir=log_dir, max_bytes=max_bytes, backup_count=backup_count))

        if not isinstance(tasks, list):
            self.tasks = [tasks]

    def check_dataset_type(self, task: Task):
        return type(task.dataset)

    def check_hps_needs_search(self, hps: Dict[str, Union[int, float, List, IntParam, FloatParam, LogFloatParam, ChoiceParam, str, bool, MetricManager]]):
        for k, v in hps.items():
            if isinstance(v, HyperParam):
                return True
        return False

    def check_all_hps(self, task: Task):
        self.dataset_needs_search = self.check_hps_needs_search(task.dataset_hps)
        self.model_needs_search = self.check_hps_needs_search(task.model_hps)
        self.trainer_config_needs_search = self.check_hps_needs_search(task.trainer_hps)
        self.optimizer_needs_search = self.check_hps_needs_search(task.optimizer_hps)

        self.hpo_event_manager.trigger(event_type=EventType.INFO, **{
            'msg': "dataset: {}, model: {}, optimizer: {}, trainer_configs: {}".format(self.dataset_needs_search, self.model_needs_search, self.optimizer_needs_search, self.trainer_config_needs_search),
            'prefix': "HPO-Search Status"
        })

        # print("Search Status: dataset: {}, model: {}, optimizer: {}, trainer_configs: {}".format(self.dataset_needs_search, self.model_needs_search, self.optimizer_needs_search, self.trainer_config_needs_search))

    def check_tasks_num(self):
        return len(self.tasks)

    def cache_clean(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_current_task(self, idx):
        return self.tasks[idx]

    def get_hps(self, trial, hps):
        hp = {}
        for k, v in hps.items():
            if isinstance(v, HyperParam):
                param_name = k
                hp[k] = _optuna_suggestion_mapping(trial, param_name, v)

            else:
                hp[k] = v

        return hp

    def get_processed_loaders_trial(self, trial, task: Task):
        # Data Loading
        train_dataloader = None
        validation_dataloader = None
        dataset_type = self.check_dataset_type(task)
        if dataset_type == dict:
            train_x, train_y = task.dataset['train_x'], task.dataset['train_y']
            val_x, val_y = task.dataset['val_x'], task.dataset['val_y']
            train_dataloader = get_dataloader(train_x,
                                              train_y, **self.get_hps(trial, task.dataset_hps) if self.dataset_needs_search else task.dataset_hps)
            validation_dataloader = get_dataloader(val_x,
                                                   val_y, **self.get_hps(trial, task.dataset_hps) if self.dataset_needs_search else task.dataset_hps)

        elif dataset_type == pd.DataFrame:
            train_dataloader, validation_dataloader, test_x, test_y, s_x, s_y = load_data(data=task.dataset,
                                                                                          **self.get_hps(trial, task.dataset_hps) if self.dataset_needs_search else task.dataset_hps)
        return train_dataloader, validation_dataloader

    def get_processed_loaders(self, task: Task):
        # Data Loading
        train_dataloader = None
        validation_dataloader = None
        dataset_type = self.check_dataset_type(task)
        if dataset_type == dict:
            train_x, train_y = task.dataset['train_x'], task.dataset['train_y']
            val_x, val_y = task.dataset['val_x'], task.dataset['val_y']
            train_dataloader = get_dataloader(train_x,
                                              train_y, **task.dataset_hps)
            validation_dataloader = get_dataloader(val_x,
                                                   val_y, **task.dataset_hps)

        elif dataset_type == pd.DataFrame:
            train_dataloader, validation_dataloader, test_x, test_y, s_x, s_y = load_data(data=task.dataset,
                                                                                          **task.dataset_hps)
        return train_dataloader, validation_dataloader

    """
     - Only Support the RecorderForEpoch for now
    """
    def search(self, target_param_key: str = None, direction: str = 'maximize'):

        for task in self.tasks:
            # CHECK_ALL_HYPERPARAMETERS
            self.check_all_hps(task)
            if not self.dataset_needs_search:
                fixed_train_dataloader, fixed_validation_dataloader = self.get_processed_loaders(task)
                self.hpo_event_manager.trigger(event_type=EventType.INFO, **{
                    'msg': "No need to change the DataLoader",
                    'prefix': 'HPO'
                })

            search_times = task.search_times

            self.hpo_event_manager.trigger(event_type=EventType.INFO, **{
                'msg': "Current Task Info: Search Times: {}".format(search_times),
                'prefix': 'HPO'
            })
            # for i in range(search_times):
            # COMPONENTS_START_LOADING
            # Model, Optimizer, Criterion Loading
            def objective(trial: Trial):
                model = task.model_class(**self.get_hps(trial, task.model_hps) if self.model_needs_search else task.model_hps)

                optimizer = task.optimizer_class(model.parameters(), **self.get_hps(trial, task.optimizer_hps) if self.optimizer_needs_search else task.optimizer_hps)

                criterion = task.criterion()

                current_trainer = task.trainer(model=model, optimizer=optimizer, criterion=criterion, **self.get_hps(trial, task.trainer_hps) if self.trainer_config_needs_search else task.trainer_hps)

                self.hpo_event_manager.trigger(event_type=EventType.INFO, **{
                    'msg': "All Components loaded",
                    'prefix': 'HPO'
                })
                train_dataloader = fixed_train_dataloader
                validation_dataloader = fixed_validation_dataloader

                # COMPONENTS_LOADING_COMPLETE
                if self.dataset_needs_search:
                    train_dataloader, validation_dataloader = self.get_processed_loaders_trial(trial, task)

                # ROUND_SEARCH_START

                result = current_trainer.fit(train_dataloader, validation_dataloader, **config_generator(**self.get_hps(trial, task.train_configs) if self.trainer_config_needs_search else task.train_configs))
                self.hpo_event_manager.trigger(event_type=EventType.HPO_ROUND_SEARCH_FINISHED, **{
                    'validation_dataloader': validation_dataloader,
                    'model': model,
                    'result': result,
                    'current_params': trial.params
                })

                maximize_param = result[target_param_key].val.detach().cpu().numpy()[-1]
                # ROUND_SEARCH_FINISHED
                del current_trainer, model, optimizer

                if self.dataset_needs_search:
                    del train_dataloader, validation_dataloader
                self.cache_clean()

                return maximize_param

            sampler = optuna.samplers.TPESampler()
            study = optuna.create_study(direction=direction, sampler=sampler)

            self.hpo_event_manager.trigger(event_type=EventType.INFO, **{
                'msg': "Optuna setting ready, beginning hyperparameter optimization",
                'prefix': 'HPO'
            })
            study.optimize(objective, n_trials=search_times)

            self.hpo_event_manager.trigger(event_type=EventType.INFO, **{
                'msg': "Hyperparameter optimization Complete, best {}: {} \n best param: {}".format(target_param_key, study.best_value, study.best_params),
                'prefix': 'HPO'
            })