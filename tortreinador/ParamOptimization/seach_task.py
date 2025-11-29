from dataclasses import dataclass, field
from typing import Union, Dict, List
import torch
from tortreinador.utils.Recorder import MetricManager
from tortreinador.utils.preprocessing import ScalerConfig
import pandas as pd
from tortreinador.ParamOptimization.hyperparam import IntParam, FloatParam, LogFloatParam, ChoiceParam


@dataclass
class Task:

    # Model
    model_class: type = None
    model_hps: Dict[str, Union[int, float, List, IntParam, FloatParam]] = field(default_factory=dict)

    # Criterion
    criterion: type = None

    # Optimizer
    optimizer_class: type = None
    optimizer_hps: Dict[str, Union[int, float, List, IntParam, FloatParam, LogFloatParam]] = field(default_factory=dict)

    # Trainer
    search_times: int = 10
    task_name: Union[str, None] = 'Unknown Task'

    dataset: Union[pd.DataFrame, Dict[str, torch.Tensor]] = None
    dataset_hps: Dict[str, Union[pd.DataFrame, int, ScalerConfig, bool, list, str, IntParam, ChoiceParam]] = field(default_factory=dict)
    # batch_size_set: Union[ChoiceParam, int] = None
    # dataset_random_seed: int = 42

    trainer: type = None
    trainer_hps: Dict[str, Union[bool, int, MetricManager]] = field(default_factory=dict)
    train_configs: Dict[str, Union[IntParam, FloatParam, LogFloatParam, ChoiceParam, str, bool, int, float]] = field(default_factory=dict)