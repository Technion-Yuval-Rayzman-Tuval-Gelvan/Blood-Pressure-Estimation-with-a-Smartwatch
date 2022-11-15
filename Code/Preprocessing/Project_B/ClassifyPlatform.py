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
        self.ppg_mah_model = None
        self.ppg_lda_model = None
        self.ppg_qda_model = None
        self.ppg_svm_model = None
        self.bp_mah_model = None
        self.bp_lda_model = None
        self.bp_qda_model = None
        self.bp_svm_model = None

    def load_models(self):

        self.ppg_mah_model = utils.load_model(cfg.PPG_MODELS_LOAD_DIR, 'good_bad_mahanlobis')
        self.ppg_svm_model = utils.load_model(cfg.PPG_MODELS_LOAD_DIR, 'good_bad_svm')
        self.ppg_qda_model = utils.load_model(cfg.PPG_MODELS_LOAD_DIR, 'good_bad_qda')
        self.ppg_lda_model = utils.load_model(cfg.PPG_MODELS_LOAD_DIR, 'good_bad_lda')
        self.bp_mah_model = utils.load_model(cfg.BP_MODELS_LOAD_DIR, 'good_bad_mahanlobis')
        self.bp_svm_model = utils.load_model(cfg.BP_MODELS_LOAD_DIR, 'good_bad_svm')
        self.bp_qda_model = utils.load_model(cfg.BP_MODELS_LOAD_DIR, 'good_bad_qda')
        self.bp_lda_model = utils.load_model(cfg.BP_MODELS_LOAD_DIR, 'good_bad_lda')

    def signal_valid(self, x, is_ppg):

        if is_ppg:
            mah_model = self.ppg_mah_model
            svm_model = self.ppg_svm_model
            lda_model = self.ppg_lda_model
            qda_model = self.ppg_qda_model
        else:
            mah_model = self.bp_mah_model
            svm_model = self.bp_svm_model
            lda_model = self.bp_lda_model
            qda_model = self.bp_qda_model

        x = np.array(x)
        y_mah = mah_model.predict_class(x, [1, -1])

        x = x.reshape(1, -1)
        y_lda = lda_model.predict(x)
        y_qda = qda_model.predict(x)
        y_svm = svm_model.predict(x)

        # True label is 1 and Bad is -1
        score = y_mah + y_lda + y_qda + y_svm

        if is_ppg:
            true_score = cfg.TRUE_PPG_SCORE
        else:
            true_score = cfg.TRUE_BP_SCORE

        if score >= true_score:
            return True

        return False

    def valid_win(self, win):
        x_ppg = win.ppg_sqi
        x_bp = win.bp_sqi
        bp_valid = False
        ppg_valid = False

        if self.signal_valid(x_ppg, is_ppg=True):
            ppg_valid = True

        if self.signal_valid(x_bp, is_ppg=False):
            bp_valid = True

        return bp_valid, ppg_valid

    def test_platform(self, win_list):

        valid_windows = 0
        results_array = {'PPG True': [], 'PPG False': [], 'BP True': [], 'BP False': []}

        for win in win_list:
            bp_valid, ppg_valid = self.valid_win(win)
            if bp_valid is True:
                results_array['BP True'].append(win)
            else:
                results_array['BP False'].append(win)
            if ppg_valid is True:
                results_array['PPG True'].append(win)
            else:
                results_array['PPG False'].append(win)
            if ppg_valid and bp_valid:
                valid_windows += 1

        for i in range(10):

            """Plot 10 True window"""
            utils.plot_signal(results_array['PPG True'][i].ppg_signal, f'true_signal_{i}', is_ppg=True)
            utils.plot_signal(results_array['BP True'][i].bp_signal, f'true_signal_{i}', is_ppg=False)
            """Plot 10 False window"""
            utils.plot_signal(results_array['PPG False'][i].ppg_signal, f'false_signal_{i}', is_ppg=True)
            utils.plot_signal(results_array['BP False'][i].bp_signal, f'false_signal_{i}', is_ppg=False)

        print(f"Results -\nBP True signals: {len(results_array['BP True'])}\n"
              f"BP False signals: {len(results_array['BP False'])}\n"
              f"PPG True signals: {len(results_array['PPG True'])}\n"
              f"PPG False signals: {len(results_array['PPG False'])}\n"
              f"Valid Windows: {valid_windows}\n")


def main():
    clf = ClassifyPlatform()
    clf.load_models()

    """ Test Platform """
    # win_list = utils.load_list()
    # clf.test_platform(win_list)


if __name__ == "__main__":
    cfg.CLASSIFICATION_LOG.redirect_output()
    main()
    cfg.CLASSIFICATION_LOG.redirect_output()