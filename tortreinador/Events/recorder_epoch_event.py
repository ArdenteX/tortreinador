from tortreinador.Events.event_system import Event, EventType
from tortreinador.utils.Recorder import RecorderForEpoch
from typing import List, Union
import numpy as np

class RecorderEpochEvent(Event):
    def __init__(self, metric_names: Union[List, np.ndarray], device: str):
        super().__init__()

        self.recorder_for_epoch = {}

        if isinstance(metric_names, np.ndarray):
            metric_names = metric_names.tolist()

        for metric in metric_names:
            self.recorder_for_epoch[metric] = RecorderForEpoch(device)

    def on_fire(self, event_type: Union[EventType], trainer, **kwargs):
        for k in self.recorder_for_epoch.keys():
            self.recorder_for_epoch[k].update(trainer.recorders[k].avg().detach())