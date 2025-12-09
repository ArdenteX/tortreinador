from enum import Enum, auto
from typing import Union, List

class EventType(Enum):
    TRAIN_INIT = auto()
    TRAIN_EPOCH_START = auto()
    TRAIN_BATCH_START = auto()
    TRAIN_BATCH_CALCULATION_END = auto()
    TRAIN_BATCH_METRIC_COLLECTION_COMPLETE = auto()
    TRAIN_BATCH_END = auto()
    TRAIN_EPOCH_END = auto()
    TRAIN_EPOCH_END_RECORD = auto()
    VALIDATION_START = auto()
    VALIDATION_BATCH_START = auto()
    VALIDATION_BATCH_METRIC_COLLECTION_COMPLETE = auto()
    VALIDATION_BATCH_END = auto()
    VALIDATION_END = auto()
    TRAIN_COMPLETE = auto()
    HPO_CHECK_ALL_HYPERPARAMETERS = auto()
    HPO_COMPONENTS_START_LOADING = auto()
    HPO_COMPONENTS_LOADING_COMPLETE = auto()
    HPO_ROUND_SEARCH_START = auto()
    HPO_ROUND_SEARCH_FINISHED = auto()
    BEST_MODEL_DETECTED = auto()
    INFO = auto()


class Event:
    def __init__(self, **kwargs):
        pass

    def on_fire(self, event_type: Union[EventType], trainer, **kwargs):
        raise NotImplementedError


class EventManager:
    def __init__(self):
        self.listener = {
            e_t: [] for e_t in EventType
        }

    def subscribe(self, event_type: Union[List[EventType], EventType], event: Event):
        if isinstance(event_type, list):
            for e_t in event_type:
                self.listener[e_t].append(event)

        else:
            self.listener[event_type].append(event)

    def trigger(self, event_type: EventType, **kwargs):
        for event in self.listener[event_type]:
            event.on_fire(event_type, **kwargs)



# event_manager = EventManager()















