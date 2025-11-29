from enum import Enum, auto
from typing import Union

from tortreinador.Events.event_system import Event, EventType


class LossStage(Enum):
    STABLE = auto()
    TURBULENCE = auto()

class LossMaintainer(Event):
    def __init__(self, window_size: int, intervention_epoch: int):
        super().__init__()
        self.current_stage = LossStage.STABLE
        self.window_size = window_size
        self.intervention_epoch = intervention_epoch
        self.window = []

    def add(self, loss):
        self.window.append(loss)

    def pop(self):
        self.window.pop(0)

    def on_fire(self, event_type: Union[EventType], trainer, **kwargs):
        if trainer.current_epoch < self.intervention_epoch:
            return "Not the timing of intervention"

        self.add(kwargs['loss'])
        if len(self.window) < self.window_size:
            return "Collecting data"

    def get_slope(self):
        pass

    def get_volatility(self):
        pass

    def stable_score(self):
        pass

    def turbulence_score(self):
        pass
