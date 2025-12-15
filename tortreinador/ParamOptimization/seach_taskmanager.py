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
from tortreinador.Events.csv_event_hpo import CsvEventForHPO
import os
from torch.utils.data import DataLoader
from tortreinador.utils.View import init_weights


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
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5, csv_on: bool = True):

        self.tasks = tasks
        self.dataset_needs_search = False
        self.model_needs_search = False
        self.optimizer_needs_search = False
        self.trainer_config_needs_search = False
        self.criterion_needs_search = False

        self.hpo_event_manager = EventManager()
        self.hpo_event_manager.subscribe(event_type=EventType.INFO, event=LoggerEvent(logging.getLogger('Tortreinador.HPO'),
                                                                                      level=level, log_dir=log_dir, max_bytes=max_bytes, backup_count=backup_count))
        self.hpo_event_manager.subscribe(event_type=[EventType.HPO_COMPONENTS_LOADING_COMPLETE, EventType.HPO_ROUND_SEARCH_FINISHED], event=CsvEventForHPO())

        self.optimize_params = []
        self.csv_on = csv_on
        self.csv_init = False

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
        self.criterion_needs_search = self.check_hps_needs_search(task.criterion_hps)

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
                self.optimize_params.append(param_name)
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

            if 'Dataset' in task.dataset.keys():
                train_dataloader = DataLoader(task.dataset['Dataset']['train_dataset'], **task.dataset_hps)
                validation_dataloader = DataLoader(task.dataset['Dataset']['val_dataset'], **task.dataset_hps)
                return train_dataloader, validation_dataloader

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

    def get_processed_result(self, rfe):
        result = {}
        for k, v in rfe.items():
            result[k] = v.val.detach().cpu().numpy()[-1].item()

        return result

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
                init_weights(model)

                optimizer = task.optimizer_class(model.parameters(), **self.get_hps(trial, task.optimizer_hps) if self.optimizer_needs_search else task.optimizer_hps)

                criterion = task.criterion(**self.get_hps(trial, task.criterion_hps) if self.criterion_needs_search else task.criterion_hps)

                current_trainer = task.trainer(model=model, optimizer=optimizer, criterion=criterion, **self.get_hps(trial, task.trainer_hps) if self.trainer_config_needs_search else task.trainer_hps)

                train_config = self.get_hps(trial, task.train_configs) if self.trainer_config_needs_search else task.train_configs

                self.hpo_event_manager.trigger(event_type=EventType.INFO, **{
                    'msg': "All Components loaded",
                    'prefix': 'HPO'
                })
                train_dataloader = fixed_train_dataloader
                validation_dataloader = fixed_validation_dataloader

                # COMPONENTS_LOADING_COMPLETE
                if self.dataset_needs_search:
                    train_dataloader, validation_dataloader = self.get_processed_loaders_trial(trial, task)

                if self.csv_init is False:
                    self.hpo_event_manager.trigger(event_type=EventType.HPO_COMPONENTS_LOADING_COMPLETE, trainer=current_trainer, **{
                        'columns': ['trial_id', 'model_id'] + self.optimize_params + current_trainer.metric_manager.metric_names.tolist(),
                        'timestamp': current_trainer.timestamp,
                        'm_p': None if 'model_save_path' not in train_config.keys() else train_config['model_save_path'],
                        'task_name': task.task_name
                    })

                    self.csv_init = True

                # ROUND_SEARCH_START
                result = current_trainer.fit(train_dataloader, validation_dataloader, **config_generator(**train_config))

                if current_trainer.data_save_mode == 'recorder':
                    processed_result = self.get_processed_result(result[0].recorder_for_epoch)

                else:
                    if 'model_save_path' in task.train_configs.keys():
                        csv_path = os.path.join(task.train_configs['model_save_path'], 'train_log', 'log_{}.csv'.format(current_trainer.timestamp))

                    else:
                        csv_path = os.path.join(os.getcwd(), 'train_log', 'log_{}.csv'.format(current_trainer.timestamp))

                    result_csv = pd.read_csv(csv_path)
                    result_csv = result_csv.iloc[-1, :].to_dict()
                    processed_result = {k: v for k, v in result_csv.items() if k != 'epoch'}


                maximize_param = processed_result[target_param_key]

                self.hpo_event_manager.trigger(event_type=EventType.HPO_ROUND_SEARCH_FINISHED, trainer=current_trainer,
                                               **{
                                                   **processed_result,
                                                   **trial.params,
                                                   'trial_id': trial.number,
                                                   'model_id': current_trainer.timestamp
                                               })

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

            self.csv_init = False
            # self.hpo_event_manager.trigger(event_type=EventType.INFO, **{
            #     'msg': "Hyperparameter optimization Complete, best {}: {} \n best param: {}".format(target_param_key, study.best_value, study.best_params),
            #     'prefix': 'HPO'
            # })