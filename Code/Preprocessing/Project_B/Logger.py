import io
import os
import sys
import datetime
import Config as cfg
import logging
import tqdm


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
