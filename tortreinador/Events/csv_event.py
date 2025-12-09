from typing import Union
from tortreinador.Events.event_system import Event, EventType
import os
import csv

class CsvEvent(Event):
    def __init__(self, timestamp, columns, m_p=None):
        super().__init__()
        self.TIMESTAMP = timestamp
        self.COL = columns
        self.MODEL_SAVE_PATH = m_p
        self.CSV_FILENAME = self._initial_csv_mode()

    def _initial_csv_mode(self) -> str:
        """
        Prepare CSV logging by creating a timestamped file path and header.

        Returns:
            tuple: (csv_filename, file_time) where `csv_filename` is the path to the log file and
        """
        # file_time = self.get_current_time()
        current_path = None

        if self.MODEL_SAVE_PATH is not None:
            current_path = self.MODEL_SAVE_PATH

        elif self.MODEL_SAVE_PATH is None:
            current_path = os.getcwd()

        filepath = os.path.join(current_path, 'train_log')
        csv_filename = os.path.join(filepath, 'log_{}.csv'.format(
            self.TIMESTAMP))

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        if not os.path.isfile(csv_filename):
            with open(csv_filename, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['epoch'] + self.COL)

        return csv_filename

    def on_fire(self, event_type: Union[EventType], trainer, **kwargs):

        with open(self.CSV_FILENAME, 'a', newline='') as file:
            writer = csv.writer(file)

            writer.writerow([trainer.current_epoch + 1] + [r.avg().detach().item() for r in trainer.recorders.values()])