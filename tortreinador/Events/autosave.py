from typing import Union
from tortreinador.Events.event_system import Event, EventType

class AutoSave(Event):
    def __init__(self, **kwargs):
        super().__init__()
        self.AUTO_SAVE_EPOCH = kwargs['auto_save']
        self.MODEL_SAVE_PATH = kwargs['m_p']
        self.AUTO_COUNT = 1

    def on_fire(self, event_type: Union[EventType], trainer, **kwargs):

        if event_type == EventType.BEST_MODEL_DETECTED:
            if self.MODEL_SAVE_PATH is not None:
                trainer.checkpoint_recorder.update_by_condition(kwargs['condition'],
                                                             b_m=kwargs['best_metric'] if 'best_metric' in kwargs.keys() else None,
                                                             b_l=kwargs['best_loss'] if 'best_loss' in kwargs.keys() else None)
                trainer.checkpoint_recorder.update(trainer.current_epoch, model=trainer.model.state_dict(),
                                                current_optimizer_sd=trainer.optimizer.state_dict(), mode='best')
                self.AUTO_COUNT = 1
                trainer.trigger(event_type=EventType.INFO, **{'msg': "Save Best model: Metric:{:.4f}, Loss Avg:{:.4f}".format(
                    kwargs['val_metric'],
                    kwargs['val_loss']),
                'prefix': 'BMD'})

            else:

                trainer.trigger(event_type=EventType.INFO, **{'msg': "Best model Detected: Metric:{:.4f}, Loss Avg:{:.4f}".format(
                kwargs['val_metric'],
                kwargs['val_loss']), 'prefix': 'BMD'})

        elif event_type == EventType.TRAIN_EPOCH_END:
            if self.MODEL_SAVE_PATH is not None:
                if self.AUTO_COUNT % self.AUTO_SAVE_EPOCH == 0:
                    trainer.checkpoint_recorder.update(trainer.current_epoch, model=trainer.model.state_dict(),
                                                    current_optimizer_sd=trainer.optimizer.state_dict())
                    trainer.trigger(event_type=EventType.INFO,
                                    **{'msg': 'Auto save model and optimizer', 'prefix': 'Auto Save'})
                    self.AUTO_COUNT = 1
                else:
                    self.AUTO_COUNT += 1