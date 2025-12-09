from typing import Union
from tortreinador.Events.event_system import Event, EventType
import os
import csv

class CsvEventForHPO(Event):
    def __init__(self, ):
        super().__init__()
        self.TIMESTAMP = None
        self.COL = None
        self.MODEL_SAVE_PATH = None
        self.CSV_FILENAME = None

    def _initial_csv_mode(self, **kwargs):
        """
        Prepare CSV logging by creating a timestamped file path and header.

        Returns:
            tuple: (csv_filename, file_time) where `csv_filename` is the path to the log file and
        """
        # file_time = self.get_current_time()
        self.TIMESTAMP = kwargs['timestamp']
        self.COL = kwargs['columns']
        self.MODEL_SAVE_PATH = kwargs['m_p']
        # print(self.MODEL_SAVE_PATH)

        current_path = None

        if self.MODEL_SAVE_PATH is not None:
            current_path = self.MODEL_SAVE_PATH

        elif self.MODEL_SAVE_PATH is None:
            current_path = os.getcwd()

        filepath = os.path.join(current_path, 'hpo_log')
        csv_filename = os.path.join(filepath, '{}_log_{}.csv'.format(
            kwargs['task_name'], self.TIMESTAMP))

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w', encoding='utf-8', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.COL)
                writer.writeheader()

        self.CSV_FILENAME = csv_filename

    def on_fire(self, event_type: Union[EventType], trainer, **kwargs):
        if event_type == EventType.HPO_COMPONENTS_LOADING_COMPLETE:
            self._initial_csv_mode(**kwargs)

        elif event_type == EventType.HPO_ROUND_SEARCH_FINISHED:
            with open(self.CSV_FILENAME, 'a', newline='') as file:
                writer = csv.DictWriter(file, self.COL)

                writer.writerow(kwargs)
