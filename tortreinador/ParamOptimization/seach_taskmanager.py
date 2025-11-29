from typing import Union, Dict, List
import torch
from tortreinador.train import config_generator
from tortreinador.utils.Recorder import MetricManager
from tortreinador.utils.preprocessing import load_data, get_dataloader
import pandas as pd
import gc
from tortreinador.ParamOptimization.seach_task import Task
from tortreinador.ParamOptimization.hyperparam import IntParam, FloatParam, LogFloatParam, ChoiceParam, HyperParam

class TaskManager:
    def __init__(self, tasks: Union[Task, List[Task]]):
        self.tasks = tasks
        self.dataset_needs_search = False
        self.model_needs_search = False
        self.optimizer_needs_search = False
        self.trainer_config_needs_search = False

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

        print("Search Status: dataset: {}, model: {}, optimizer: {}, trainer_configs: {}".format(self.dataset_needs_search, self.model_needs_search, self.optimizer_needs_search, self.trainer_config_needs_search))

    def check_tasks_num(self):
        return len(self.tasks)

    def cache_clean(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def get_current_task(self, idx):
        return self.tasks[idx]

    def get_hps(self, hps):
        hp = {}
        for k, v in hps.items():
            hp[k] = v.sample() if isinstance(v, HyperParam) else v

        return hp

    def get_processed_loaders(self, task: Task):
        # Data Loading
        train_dataloader = None
        validation_dataloader = None
        dataset_type = self.check_dataset_type(task)
        if dataset_type == dict:
            train_x, train_y = task.dataset['train_x'], task.dataset['train_y']
            val_x, val_y = task.dataset['val_x'], task.dataset['val_y']
            train_dataloader = get_dataloader(train_x,
                                              train_y, **self.get_hps(task.dataset_hps) if self.dataset_needs_search else task.dataset_hps)
            validation_dataloader = get_dataloader(val_x,
                                                   val_y, **self.get_hps(task.dataset_hps) if self.dataset_needs_search else task.dataset_hps)

        elif dataset_type == pd.DataFrame:
            train_dataloader, validation_dataloader, test_x, test_y, s_x, s_y = load_data(data=task.dataset,
                                                                                          **self.get_hps(task.dataset_hps) if self.dataset_needs_search else task.dataset_hps)
        return train_dataloader, validation_dataloader

    """
    TODO:
     - Design the event system
     - Design the Logger system
     - Collect the best hyperparameters and metrics one times
     - Collect the best hyperparameters and metrics of current task
    """
    def search(self):

        for task in self.tasks:
            # CHECK_ALL_HYPERPARAMETERS
            self.check_all_hps(task)
            if not self.dataset_needs_search:
                train_dataloader, validation_dataloader = self.get_processed_loaders(task)
                print("Current Task Info: No need to change the DataLoader")

            search_times = task.search_times
            print("Current Task Info: Search Times: {}".format(search_times))
            for i in range(search_times):
                # COMPONENTS_START_LOADING
                # Model, Optimizer, Criterion Loading
                model = task.model_class(**self.get_hps(task.model_hps) if self.model_needs_search else task.model_hps)
                print("Current Task Info: Model loaded")

                optimizer = task.optimizer_class(model.parameters(), **self.get_hps(task.optimizer_hps) if self.optimizer_needs_search else task.optimizer_hps)
                print("Current Task Info: Optimizer loaded")

                criterion = task.criterion()
                print("Current Task Info: Criterion loaded")

                current_trainer = task.trainer(model=model, optimizer=optimizer, criterion=criterion, **self.get_hps(task.trainer_hps) if self.trainer_config_needs_search else task.trainer_hps)
                print("Current Task Info: Trainer loaded")

                # COMPONENTS_LOADING_COMPLETE
                if self.dataset_needs_search:
                    train_dataloader, validation_dataloader = self.get_processed_loaders(task)

                # ROUND_SEARCH_START
                result = current_trainer.fit(train_dataloader, validation_dataloader, **config_generator(**self.get_hps(task.train_configs) if self.trainer_config_needs_search else task.train_configs))
                print("Current Task Info: Training finished")
                # ROUND_SEARCH_FINISHED
                del current_trainer, model, optimizer

                if self.dataset_needs_search:
                    del train_dataloader, validation_dataloader
                self.cache_clean()