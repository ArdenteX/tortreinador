from datetime import datetime
from typing import Union
import logging
from logging.handlers import RotatingFileHandler
from tortreinador.Events.event_system import Event, EventType
from typing import Union
from pathlib import Path

class LoggerEvent(Event):
    def __init__(self,
                 logger: Union[logging.Logger, None] = None,
                 level: int = logging.INFO,
                 log_dir: str = None,
                 max_bytes: int = 10 * 1024 * 1024,
                 backup_count: int = 5):
        super().__init__()

        self.fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(tag)s] %(message)s"
        self.datefmt = "%Y-%m-%d %H:%M:%S"
        self.formatter = logging.Formatter(self.fmt, self.datefmt)

        self.logger = logger if logger is not None else logging.getLogger('Tortreinador')
        self.level = level if level is not None else logging.INFO
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()

        self.add_stream_handler()

        self.log_dir = log_dir
        self.max_bytes = max_bytes if max_bytes is not None else 10 * 1024 * 1024
        self.backup_count = backup_count if backup_count is not None else 5

        if self.log_dir is not None:
            self.add_file_handler()

        # self.logger.log(self.level, "Logger Event: on \n Logger Level: {}".format(self.level))

    def on_fire(self, event_type: Union[EventType], **kwargs):
        if 'prefix' not in kwargs.keys():
            kwargs['prefix'] = event_type.name

        self.logger.info(kwargs['msg'], extra={'tag': kwargs['prefix'].strip()})

    def add_stream_handler(self):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

    def add_file_handler(self):
        log_file = self.file_resolve()
        file_handler = RotatingFileHandler(log_file, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding='utf-8')
        file_handler.setLevel(self.level)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

    def file_resolve(self):
        p = Path(self.log_dir)
        if p.exists():
            if p.is_dir():
                if p.name != 'logs':
                    # print("log path is an existing directory, but dir name is not logs")
                    p = p.joinpath('logs')
                    p.mkdir(exist_ok=True, parents=True)

                return p.joinpath('{}_train.log'.format(datetime.now().strftime('%Y-%m-%d %H:%M').replace(":", "").replace("-", '').replace(
            " ", '')))

            elif p.is_file() and p.suffix == '.log':
                return p

        else:
            raise ValueError("log_dir must be a directory or a .log file")