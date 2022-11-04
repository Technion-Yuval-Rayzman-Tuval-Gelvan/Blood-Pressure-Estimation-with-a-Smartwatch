import os
import sys
import datetime
import Config as cfg
import logging
import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class Logger:
    def __init__(self, path):
        self.path = path
        self.old_stdout = sys.stdout
        self.log = None

    def redirect_output(self):
        self.log = open(self.path, "w")
        sys.stdout = self.log

    def close_log_file(self):
        sys.stdout = self.old_stdout
        self.log.close()
