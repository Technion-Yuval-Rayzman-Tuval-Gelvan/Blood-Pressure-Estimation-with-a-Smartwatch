import copy
import pickle
from multiprocessing import Pool, Lock
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import heartpy as hp
import Utils as utils
import Config as cfg
import Plot as plot
import Trainer


class ClassifyPlatform:

    def __init__(self):
        self.mah_model = None
        self.lda_model = None
        self.qda_model = None
        self.svm_model = None

    def load_models(self):
        self.mah_model = utils.load_model(cfg.PPG_MODELS_LOAD_DIR, 'good_bad_mahanlobis')
        self.svm_model = utils.load_model(cfg.PPG_MODELS_LOAD_DIR, 'good_bad_svm')
        self.qda_model = utils.load_model(cfg.PPG_MODELS_LOAD_DIR, 'good_bad_qda')
        self.lda_model = utils.load_model(cfg.PPG_MODELS_LOAD_DIR, 'good_bad_lda')

    def signal_valid(self, x):

        y_mah = self.mah_model.predict(x)
        y_lda = self.lda_model.predict(x)
        y_qda = self.qda_model.predict(x)
        y_svm = self.svm_model.predict(x)

        # True label is 1 and Bad is -1
        score = y_mah + y_lda + y_qda + y_svm

        if score >= 0:
            return True

        return False

    def valid_win(self, win):
        ppg_sqi = win.ppg_sqi
        x_ppg = ppg_sqi.get_ski_list()
        bp_sqi = win.bp_sqi
        x_bp = bp_sqi.get_ski_list()

        if self.signal_valid(x_ppg) and self.signal_valid(x_bp):
            return True

        return False


