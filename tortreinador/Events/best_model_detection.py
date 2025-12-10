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
        val_loss_recorder = trainer.recorders[
            trainer.metric_manager.metric_names[trainer.metric_manager.criterion_idx]]
        val_loss_recorder = val_loss_recorder.avg().item()

        try:
            val_baseline_metric = trainer.recorders[
                trainer.metric_manager.metric_names[trainer.metric_manager.baseline_metric_idx]]
            val_baseline_metric = val_baseline_metric.avg().item()

        except:
            val_baseline_metric = 'Not Found'

        if self.condition == 0:

            if val_baseline_metric > self.best_metric:
                self.best_metric = val_baseline_metric
                self.SAVE = True

        elif self.condition == 1:

            if val_loss_recorder < self.best_loss:
                self.best_loss = val_loss_recorder
                self.SAVE = True

        elif self.condition == 2:
            val_baseline_metric = trainer.recorders[
                trainer.metric_manager.metric_names[trainer.metric_manager.baseline_metric_idx]]
            val_baseline_metric = val_baseline_metric.avg().item()
            
            if val_loss_recorder < self.best_loss and val_baseline_metric > \
                    self.best_metric:
                self.best_metric = val_baseline_metric
                self.best_loss = val_loss_recorder
                self.SAVE = True

            elif val_loss_recorder < self.best_loss and val_baseline_metric < \
                    self.best_metric:
                abs_dis = np.abs((self.best_metric - val_baseline_metric) / self.best_metric)
                if 0.001 < abs_dis < 0.003:
                    self.best_metric = val_baseline_metric
                    self.best_loss = val_loss_recorder
                    self.SAVE = True

        if self.SAVE:
            trainer.event_manager.trigger(EventType.BEST_MODEL_DETECTED, trainer=trainer, val_metric=val_baseline_metric,
                            val_loss=val_loss_recorder, condition=self.condition, best_metric=self.best_metric, best_loss=self.best_loss)

            self.SAVE = False