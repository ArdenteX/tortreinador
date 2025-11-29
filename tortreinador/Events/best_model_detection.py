from typing import Union
import numpy as np

from tortreinador.Events.event_system import EventType, Event

class BestModelDetection(Event):
    def __init__(self, **kwargs):
        super().__init__()
        self.best_metric = kwargs['b_m']
        self.best_loss = kwargs['b_l']
        self.condition = kwargs['condition']
        self.SAVE = False

    def on_fire(self, event_type: Union[EventType], trainer, **kwargs):
        if self.condition == 0:
            if kwargs['val_metric'] > self.best_metric:
                self.best_metric = kwargs['val_metric']
                self.SAVE = True

        elif self.condition == 1:
            if kwargs['val_loss'] < self.best_loss:
                self.best_loss = kwargs['val_loss']
                self.SAVE = True

        elif self.condition == 2:
            if kwargs['val_loss'] < self.best_loss and kwargs['val_metric'] > \
                    self.best_metric:
                self.best_metric = kwargs['val_metric']
                self.best_loss = kwargs['val_loss']
                self.SAVE = True

            elif kwargs['val_loss'] < self.best_loss and kwargs['val_metric'] < \
                    self.best_metric:
                abs_dis = np.abs((self.best_metric - kwargs['val_metric']) / self.best_metric)
                if 0.001 < abs_dis < 0.003:
                    self.best_metric = kwargs['val_metric']
                    self.best_loss = kwargs['val_loss']
                    self.SAVE = True

        if self.SAVE:
            trainer.event_manager.trigger(EventType.BEST_MODEL_DETECTED, trainer=trainer, val_metric=kwargs['val_metric'],
                            val_loss=kwargs['val_loss'], condition=self.condition, best_metric=self.best_metric, best_loss=self.best_loss)

            self.SAVE = False