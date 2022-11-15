
""" Compare between MIMIC dataset SQI model to Heartpy """

import logging
import warnings
import time
from tqdm import tqdm
import heartpy as hp
import Utils as utils
import Config as cfg
import Plot as plot
import Trainer
from Code.Preprocessing.Project_B.CardiacPreProcess import load_files, preprocess_data
from Code.Preprocessing.Project_B.ClassifyPlatform import ClassifyPlatform
from Code.Preprocessing.Project_B.CreateDataset import DatasetCreator
from Code.Preprocessing.Project_B.MimicPreProcess import classify_target
from Code.Training.LoadData import arrange_folders
from SQI import SQI
import copy
import pickle
from multiprocessing import Pool, Lock
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt


class CompareWin:

    def __init__(self, win):
        self.ppg_cardiac_target = win.ppg_target
        self.ppg_heartpy_target = classify_target(win.ppg_signal, cfg.FREQUENCY)
        self.ppg_our_model_target = None
        self.bp_cardiac_target = win.bp_target
        self.bp_heartpy_target = classify_target(win.bp_signal, cfg.FREQUENCY)
        self.bp_our_model_target = None
        self.win = win

    def get_our_model_classification(self, clf):
        bp_valid, ppg_valid = clf.valid_win(self.win)

        if bp_valid:
            self.bp_our_model_target = utils.Label.good
        else:
            self.bp_our_model_target = utils.Label.bad

        if ppg_valid:
            self.ppg_our_model_target = utils.Label.good
        else:
            self.ppg_our_model_target = utils.Label.bad


def save_compare_plots(target_1, target_2, target_1_name, target_2_name, win, is_ppg=True, plot_results=True):

    if is_ppg:
        signal = win.ppg_signal
    else:
        signal = win.bp_signal

    # good 1 - bad 2
    if target_1 == utils.Label.good and target_2 == utils.Label.bad:
        if plot_results:
            utils.plot_win(signal, f'good_{target_1_name}_bad_{target_2_name}/{win.win_name}')

    # bad 1 - good 2
    if target_1 == utils.Label.bad and target_2 == utils.Label.good:
        if plot_results:
            utils.plot_win(signal, f'bad_{target_1_name}_good_{target_2_name}/{win.win_name}')

    # good 1 - good 2
    if target_1 == utils.Label.good and target_2 == utils.Label.good:
        if plot_results:
            utils.plot_win(signal, f'good_{target_1_name}_good_{target_2_name}/{win.win_name}')

    # bad 1 - bad 2
    if target_1 == utils.Label.bad and target_2 == utils.Label.bad:
        if plot_results:
            utils.plot_win(signal, f'bad_{target_1_name}_bad_{target_2_name}/{win.win_name}')


def compare_accuracy_to_cardiac(win_list):
    clf = ClassifyPlatform()
    clf.load_models()

    for win in tqdm(win_list):
        """ Get all targets """
        new_win = CompareWin(win)

        new_win.get_our_model_classification(clf)

        """ Compare Results"""
        save_compare_plots(new_win.ppg_cardiac_target, new_win.ppg_our_model_target, "cardiac", "our",
                           new_win.win, is_ppg=True)
        save_compare_plots(new_win.ppg_cardiac_target, new_win.ppg_heartpy_target, "cardiac", "heartpy",
                           new_win.win, is_ppg=True)
        save_compare_plots(new_win.ppg_our_model_target, new_win.ppg_heartpy_target, "our", "heartpy",
                           new_win.win, is_ppg=True)
        save_compare_plots(new_win.bp_cardiac_target, new_win.bp_our_model_target, "cardiac", "our",
                           new_win.win, is_ppg=False)
        save_compare_plots(new_win.bp_cardiac_target, new_win.bp_heartpy_target, "cardiac", "heartpy",
                           new_win.win, is_ppg=False)
        save_compare_plots(new_win.bp_our_model_target, new_win.bp_heartpy_target, "our", "heartpy",
                           new_win.win, is_ppg=False)


def main():
    assert cfg.WORK_MODE == cfg.Mode.compare_results

    """ Compare our model to Heartpy on Mimic Dataset"""
    # assert cfg.DATASET == cfg.Dataset.mimic
    # dc = DatasetCreator()
    # dc.create_dataset()

    """ Check Cardiac Accuracy on Our Model/Heartpy (Change DATASET config)"""
    assert cfg.DATASET == cfg.Dataset.cardiac
    data = load_files()
    win_list = preprocess_data(data)
    compare_accuracy_to_cardiac(win_list)


if __name__ == "__main__":
    # cfg.DATASET_LOG.redirect_output()
    warnings.simplefilter(action='ignore')
    main()
    # cfg.DATASET_LOG.close_log_file()
