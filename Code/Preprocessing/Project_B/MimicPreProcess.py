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
import SQI as sqi
from Code.Preprocessing.Project_B.ClassifyPlatform import ClassifyPlatform


class DatasetCreator:

    def __init__(self):
        self.windows_list = []

    # load database with ABP and PLETH signals
    def create_records_dataset(self):

        # load list
        if cfg.MIN_RECORDS_PER_PATIENT > 0:
            file_name = f'{cfg.MIN_RECORDS_PER_PATIENT}_records_per_patient_list'
        else:
            file_name = 'records_list'

        with open(file_name, 'rb') as file:
            records_list = pickle.load(file)

        if not cfg.ALL_PATIENTS:
            records_list = records_list[:cfg.NUM_PATIENTS]

        sampled_records_list = []
        for patient_records in records_list:
            np.random.seed(5)
            sampled_records = np.random.choice(np.array(patient_records), cfg.TRAIN_RECORDS_PER_PATIENT)
            sampled_records_list.append(sampled_records)

        sampled_records_list = np.concatenate(np.array(sampled_records_list))

        print("loading records..")
        pool = Pool()
        for valid_windows in tqdm(pool.imap(func=self.create_record_dataset, iterable=sampled_records_list),
                                   total=len(sampled_records_list)):
            if cfg.TRAIN_MODELS:
                self.windows_list += valid_windows

    def create_record_dataset(self, record_path):
        with open(record_path, 'rb') as file:
            record = pickle.load(file)

        valid_windows = []
        if record.record_name[-1] == 'n':
            return valid_windows

        record_windows = record_to_windows(record)

        if len(record_windows) != 0:
            bp_index = record_windows[0].sig_name.index('ABP')
            ppg_index = record_windows[0].sig_name.index('PLETH')
            for i, win in enumerate(record_windows):
                bp_signal = win.p_signal[:, bp_index]
                ppg_signal = win.p_signal[:, ppg_index]
                if not np.count_nonzero(np.isnan(ppg_signal)) and not np.count_nonzero(np.isnan(bp_signal)):
                    new_win = self.create_window(win, ppg_index, bp_index, record, i)
                    if cfg.TRAIN_MODELS:
                        if new_win is not None:
                            valid_windows.append(new_win)
                    # else:
                    #     # Classify window with the trained models


        return valid_windows

    def create_window(self, win, ppg_index, bp_index, record, i):

        win_ppg_signal = win.p_signal[:, ppg_index]
        win_bp_signal = win.p_signal[:, bp_index]

        name = record.record_name
        win_ppg_target = classify_target(win_ppg_signal, record.fs)
        win_bp_target = classify_target(win_bp_signal, record.fs)
        sys_bp, dias_bp = utils.bp_detection(win_bp_signal)
        if win_ppg_target is None or win_bp_target is None or utils.bp_valid(sys_bp, dias_bp) is False:
            return

        win_ppg_sqi = sqi.SQI()
        win_ppg_sqi.calculate_sqi(win_ppg_signal)
        win_bp_sqi = sqi.SQI()
        win_bp_sqi.calculate_sqi(win_bp_signal)

        new_win = utils.Window(win_ppg_signal, win_bp_signal, win_ppg_target,
                               win_bp_target, win_bp_sqi, win_ppg_sqi,
                               win_name=f'{name}_{i}', sys_bp=sys_bp, dias_bp=dias_bp)

        return new_win

    def save_windows_list(self):
        utils.save_list(self.windows_list)
        # utils.save_win(new_win, win_name=f'{name}_{i}')

    def create_dataset(self):
        self.create_records_dataset()
        self.save_windows_list()


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def record_to_windows(record):
    start_point = 0
    fs = cfg.FREQUENCY
    window_in_sec = 30
    end_point = start_point + (window_in_sec * fs)
    window_overlap = window_in_sec / 2
    new_samples_per_step = int(window_overlap * fs)
    record_windows = []

    while end_point < record.sig_len:
        p_signal = record.p_signal[start_point:end_point]
        window = copy.copy(record)
        window.p_signal = p_signal
        window.sig_len = end_point - start_point
        start_point += new_samples_per_step
        end_point += new_samples_per_step
        record_windows.append(window)

    return record_windows


def classify_target(signal, fs):
    try:
        wd, m = hp.process(signal, fs)
    except:
        return

    peaks_len = len(wd['peaklist'])
    num_bad_peaks = np.count_nonzero(wd['RR_masklist'])
    quality_percent = (1 - (num_bad_peaks / peaks_len)) * 100
    high_tresh = int(cfg.HIGH_THRESH)
    low_tresh = int(cfg.LOW_THRESH)

    if quality_percent >= high_tresh:
        target = utils.Label.good
    elif quality_percent >= low_tresh:
        target = utils.Label.mid
    else:
        target = utils.Label.bad

    return target


def save_records_list():
    records_list = [os.path.join(path, name) for path, subdirs, files in os.walk(cfg.MIMIC_LOAD_DIR) for name in files]
    # save list
    with open('records_list', 'wb') as file:
        pickle.dump(records_list, file)


def save_good_records_list():
    records_list = []
    for path, subdirs, files in os.walk(cfg.MIMIC_LOAD_DIR):
        if len(files) >= cfg.MIN_RECORDS_PER_PATIENT:
            files_list = []
            for name in files:
                files_list.append(os.path.join(path, name))
            records_list.append(files_list)
    # save list
    with open(f'{cfg.MIN_RECORDS_PER_PATIENT}_records_per_patient_list', 'wb') as file:
        pickle.dump(records_list, file)


def main():
    assert cfg.DATASET == cfg.Dataset.mimic

    """save list of records"""
    # if cfg.MIN_RECORDS_PER_PATIENT > 0:
    #     save_good_records_list()
    # else:
    #     save_records_list()

    """save records as windows"""
    # dataset_creator = DatasetCreator()
    # dataset_creator.create_dataset()

    # """load_windows_dictionary"""
    # win_list = utils.load_list()

    """  Train Models  """
    # win_dict = utils.convert_list_to_dict(win_list)

    """plot windows"""
    # if cfg.PLOT:
    #     utils.plot_windows(win_dict)

    """histogram of labels"""
    # plot.label_histogram(win_dict)
    # plot.features_histogram(win_dict)

    # with pd.ExcelWriter(f'{cfg.DATA_DIR}/{cfg.TIME_DIR}/accuracy.xlsx') as excel_writer:
    #
    #     """good/mid"""
    #     print("************************* Good / Mid **************************************")
    #     trainer = Trainer.Trainer(true_label=utils.Label.good, false_label=utils.Label.mid,
    #                               win_dict=win_dict, excel_writer=excel_writer)
    #     trainer.run()
    #
    #     """good/bad"""
    #     print("************************* Good / Bad **************************************")
    #     trainer = Trainer.Trainer(true_label=utils.Label.good, false_label=utils.Label.bad,
    #                               win_dict=win_dict, excel_writer=excel_writer)
    #     trainer.run()
    #
    #     """mid/bad"""
    #     print("************************* Mid / Bad **************************************")
    #     trainer = Trainer.Trainer(true_label=utils.Label.mid, false_label=utils.Label.bad,
    #                               win_dict=win_dict, excel_writer=excel_writer)
    #     trainer.run()


if __name__ == "__main__":
    cfg.LOG.redirect_output()
    main()
    cfg.LOG.close_log_file()
