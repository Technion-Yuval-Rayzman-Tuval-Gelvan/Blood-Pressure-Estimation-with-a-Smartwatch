
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
            self.bp_our_model_target = utils.Label.good




def get_our_model_classification(signal, is_ppg=True):
    hp_target_list = []

    for win in win_list:
        if is_ppg:
            signal = win.ppg_signal
        else:
            signal = win.bp_signal

        target = classify_target(signal, cfg.FREQUENCY)


def compare_accuracy_to_cardiac(win_list):
    compare_win_list = []
    clf = ClassifyPlatform()
    clf.load_models()

    for win in win_list:
        new_win = CompareWin(win, clf)

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
    utils.save_list(win_list)
    win_list = utils.load_list()





if __name__ == "__main__":
    # cfg.DATASET_LOG.redirect_output()
    warnings.simplefilter(action='ignore')
    main()
    # cfg.DATASET_LOG.close_log_file()
