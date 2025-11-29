from tortreinador.Events.event_system import Event, EventType
from tortreinador.Events.lr_schedule_event import LRSchedule
from tortreinador.Events.best_model_detection import BestModelDetection
from tortreinador.Events.autosave import AutoSave
# from tortreinador.train import TorchTrainer

class ConfigRegisterEvent(Event):
    def __init__(self):
        super(ConfigRegisterEvent, self).__init__()

    def on_fire(self, event_type, trainer, **kwargs):
        if kwargs['lr_schedule']['on']:
            trainer.subscribe([EventType.TRAIN_EPOCH_START, EventType.TRAIN_BATCH_END, EventType.TRAIN_EPOCH_END], LRSchedule(trainer, **kwargs['lr_schedule']))

        if kwargs['best_model_detection']['condition'] in [0, 1, 2]:
            trainer.subscribe(EventType.VALIDATION_END, BestModelDetection(**kwargs['best_model_detection']))

        if kwargs['auto_save'] is not None and kwargs['m_p'] is not None:
            trainer.subscribe([EventType.TRAIN_EPOCH_END, EventType.BEST_MODEL_DETECTED], AutoSave(**{'m_p': kwargs['m_p'], 'auto_save': kwargs['auto_save']}))