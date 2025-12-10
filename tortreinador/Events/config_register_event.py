from tortreinador.Events.event_system import Event, EventType
from tortreinador.Events.lr_schedule_event import LRSchedule
from tortreinador.Events.best_model_detection import BestModelDetection
from tortreinador.Events.logger_event import LoggerEvent
from tortreinador.Events.autosave import AutoSave
from tortreinador.Events.csv_event import CsvEvent
from tortreinador.Events.recorder_epoch_event import RecorderEpochEvent
import logging
# from tortreinador.train import TorchTrainer

class ConfigRegisterEvent(Event):
    def __init__(self):
        super(ConfigRegisterEvent, self).__init__()

    def on_fire(self, event_type, trainer, **kwargs):
        if kwargs['logger']['on']:
            trainer.subscribe(EventType.INFO, LoggerEvent(logger=kwargs['logger']['logger'], level=kwargs['logger']['level'],
                                                          log_dir=kwargs['logger']['logger_dir'], max_bytes=kwargs['logger']['logger_file_max_bytes'], backup_count=kwargs['logger']['logger_file_backup_count']))

        if kwargs['lr_schedule']['on']:
            trainer.subscribe([EventType.TRAIN_EPOCH_START, EventType.TRAIN_BATCH_END, EventType.TRAIN_EPOCH_END], LRSchedule(trainer, **kwargs['lr_schedule']))

        if kwargs['best_model_detection']['condition'] in [0, 1, 2]:
            trainer.subscribe(EventType.VALIDATION_END, BestModelDetection(**kwargs['best_model_detection']))

        if kwargs['auto_save'] is not None and 'm_p' in kwargs.keys():
            trainer.subscribe([EventType.TRAIN_EPOCH_END, EventType.BEST_MODEL_DETECTED], AutoSave(**{'m_p': kwargs['m_p'], 'auto_save': kwargs['auto_save']}))

        if trainer.data_save_mode == 'recorder':
            trainer.subscribe([EventType.VALIDATION_END, EventType.TRAIN_EPOCH_END_RECORD],
                              RecorderEpochEvent(trainer.metric_manager.metric_names.tolist(), trainer.device.type))

        if trainer.data_save_mode == 'csv':
            trainer.subscribe([EventType.VALIDATION_END, EventType.TRAIN_EPOCH_END_RECORD],
                              CsvEvent(trainer.timestamp, trainer.metric_manager.metric_names.tolist(), None if 'm_p' not in kwargs.keys() else kwargs['m_p']))

        trainer.trigger(EventType.INFO, **{
            'prefix': 'Register',
            'msg': "Built-in functions: \n Logger: {}, \n Learning Rate Schedule Strategy : {}, \n Condition of Best Model Detection: {} \n Data Save Mode: {}".format('on' if kwargs['logger']['on'] else 'off', 'on' if kwargs['lr_schedule']['on'] else 'off',
                                                                                                                                                    kwargs['best_model_detection']['condition'] if kwargs['best_model_detection']['condition'] in [0, 1, 2] else 'off',
                                                                                                                                                                       trainer.data_save_mode),
        })







