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


def save_compare_plots(target_1, target_2, target_1_name, target_2_name, win, is_ppg=True):
    if is_ppg:
        signal = win.ppg_signal
    else:
        signal = win.bp_signal

    # good 1 - bad 2
    if target_1 == utils.Label.good and target_2 == utils.Label.bad:
        utils.plot_win(signal, f'good_{target_1_name}_bad_{target_2_name}/{win.win_name}')

    # bad 1 - good 2
    if target_1 == utils.Label.bad and target_2 == utils.Label.good:
        utils.plot_win(signal, f'bad_{target_1_name}_good_{target_2_name}/{win.win_name}')

    # good 1 - good 2
    if target_1 == utils.Label.good and target_2 == utils.Label.good:
        utils.plot_win(signal, f'good_{target_1_name}_good_{target_2_name}/{win.win_name}')

    # bad 1 - bad 2
    if target_1 == utils.Label.bad and target_2 == utils.Label.bad:
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


def calculate_histogram(win_hist, name):

    total_windows = 0
    hist_list = {}
    for key in win_hist.keys():
        win_list = win_hist[key]
        key_list = key.split(", ")
        hist_key = f"{key_list[0][0]}, {key_list[1][0]}, {key_list[2][0]}"
        hist_list[hist_key] = len(win_list)

        total_windows += len(win_list)

        for compare_win in win_list:
            if cfg.PLOT:
                if name == 'ppg':
                    signal = compare_win.win.ppg_signal
                else:
                    signal = compare_win.win.bp_signal

                name_list = key.split(", ")
                utils.plot_win(signal, f'cardiac_{name_list[0]}_heartpy_{name_list[1]}_ours_{name_list[2]}/{compare_win.win.win_name}')

    for key, value in hist_list.items():
        print(f"{key}: {value}, {value*100/total_windows}%")

    plt.bar(x=list(hist_list.keys()), height=list(hist_list.values()))
    plt.grid(axis='y')
    plt.xlabel('Cardiac Label, Heartpy Label, Our Label')
    plt.ylabel('# windows')
    # plt.show()
    plt.savefig(f'{cfg.COMPARE_DIR}/Histogram_Models_{name}_Classification.png')
    plt.close()


def plot_histogram_results(win_list):
    clf = ClassifyPlatform()
    clf.load_models()
    none_counter = 0
    ppg_win_hist = {'good, good, good': [], 'good, good, bad': [], 'good, bad, good': [],
                            'bad, good, good': [], 'good, bad, bad': [], 'bad, bad, good': [],
                            'bad, good, bad': [], 'bad, bad, bad': []}
    bp_win_hist = {'good, good, good': [], 'good, good, bad': [], 'good, bad, good': [],
                    'bad, good, good': [], 'good, bad, bad': [], 'bad, bad, good': [],
                    'bad, good, bad': [], 'bad, bad, bad': []}

    for win in tqdm(win_list):
        """ Get all targets """
        new_win = CompareWin(win)
        new_win.get_our_model_classification(clf)

        if new_win.ppg_heartpy_target is None or \
                new_win.bp_heartpy_target is None or \
                new_win.bp_cardiac_target is None or \
                new_win.ppg_cardiac_target is None or \
                new_win.ppg_our_model_target is None or \
                new_win.bp_our_model_target is None:
            none_counter += 1
            continue

        # Classify Mid as Bad windows
        new_win.ppg_heartpy_target = utils.Label.bad if new_win.ppg_heartpy_target == utils.Label.mid else new_win.ppg_heartpy_target
        new_win.bp_heartpy_target = utils.Label.bad if new_win.bp_heartpy_target == utils.Label.mid else new_win.bp_heartpy_target
        new_win.bp_cardiac_target = utils.Label.bad if new_win.bp_cardiac_target == utils.Label.mid else new_win.bp_cardiac_target
        new_win.ppg_cardiac_target = utils.Label.bad if new_win.ppg_cardiac_target == utils.Label.mid else new_win.ppg_cardiac_target
        new_win.ppg_our_model_target = utils.Label.bad if new_win.ppg_our_model_target == utils.Label.mid else new_win.ppg_our_model_target
        new_win.bp_our_model_target = utils.Label.bad if new_win.bp_our_model_target == utils.Label.mid else new_win.bp_our_model_target

        ppg_win_hist[f'{new_win.ppg_cardiac_target.name}, {new_win.ppg_heartpy_target.name}, {new_win.ppg_our_model_target.name}'].append(new_win)
        bp_win_hist[f'{new_win.bp_cardiac_target.name}, {new_win.bp_heartpy_target.name}, {new_win.bp_our_model_target.name}'].append(new_win)

    calculate_histogram(ppg_win_hist, 'ppg')
    calculate_histogram(bp_win_hist, 'bp')

    print(f"None labels: {none_counter}")


def main():
    assert cfg.WORK_MODE == cfg.Mode.compare_models

    """ Compare our model to Heartpy on Mimic Dataset"""
    # assert cfg.DATASET == cfg.Dataset.mimic
    # dc = DatasetCreator()
    # dc.create_dataset()

    """ Check Cardiac Accuracy on Our Model/Heartpy (Change DATASET config)"""
    assert cfg.DATASET == cfg.Dataset.cardiac
    data = load_files()
    win_list = preprocess_data(data)
    # compare_accuracy_to_cardiac(win_list)

    """ Plot Histogram Results """
    plot_histogram_results(win_list)


if __name__ == "__main__":
    # cfg.DATASET_LOG.redirect_output()
    warnings.simplefilter(action='ignore')
    main()
    # cfg.DATASET_LOG.close_log_file()
