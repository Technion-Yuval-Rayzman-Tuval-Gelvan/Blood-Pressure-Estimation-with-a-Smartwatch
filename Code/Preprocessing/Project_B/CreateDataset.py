import logging
import warnings

from tqdm import tqdm
import heartpy as hp
import Utils as utils
import Config as cfg
import Plot as plot
import Trainer
from Code.Training.LoadData import arrange_folders
from SQI import SQI
import copy
import pickle
from multiprocessing import Pool, Lock
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from Code.Preprocessing.Project_B.ClassifyPlatform import ClassifyPlatform

if cfg.TRAIN_MODELS is False:
    from Code.Preprocessing.Project_B.MimicPreProcess import classify_target


class DatasetCreator:

    def __init__(self):
        self.windows_list = []
        self.classify_platform = ClassifyPlatform()

    # load database with ABP and PLETH signals
    def create_records_dataset(self):

        # load relevant list
        if cfg.MIN_RECORDS_PER_PATIENT > 0:
            file_name = f'{cfg.MIN_RECORDS_PER_PATIENT}_records_per_patient_list'
        else:
            file_name = 'records_list'

        print("Load Records List")
        with open(file_name, 'rb') as file:
            records_list = pickle.load(file)
        print(f"Load Done. Num Patients: {len(records_list)}")

        sampled_records_list = []
        if not cfg.ALL_PATIENTS:
            records_list = records_list[:cfg.NUM_PATIENTS]

            for patient_records in records_list:
                np.random.seed(5)
                sampled_records = np.random.choice(np.array(patient_records), cfg.TRAIN_RECORDS_PER_PATIENT)
                sampled_records_list.append(sampled_records)

            sampled_records_list = np.concatenate(np.array(sampled_records_list))
        else:
            sampled_records_list = np.concatenate(np.array(records_list))

        print("Loading Records..")
        pool = Pool()
        for valid_windows in tqdm(pool.imap(func=self.create_record_dataset, iterable=sampled_records_list),
                                  total=len(sampled_records_list), mininterval=30):
            if cfg.TRAIN_MODELS:
                self.windows_list += valid_windows

    def create_record_dataset(self, record_path):
        with open(record_path, 'rb') as file:
            record = pickle.load(file)

        valid_windows = []
        if record.record_name[-1] == 'n':
            return valid_windows

        valid_windows = self.record_to_windows(record)

        return valid_windows

    def record_to_windows(self, record):
        start_point = 0
        fs = cfg.FREQUENCY
        window_in_sec = 30
        end_point = start_point + (window_in_sec * fs)
        window_overlap = window_in_sec / 2
        new_samples_per_step = int(window_overlap * fs)
        bp_index = record.sig_name.index('ABP')
        ppg_index = record.sig_name.index('PLETH')
        bp_signal = record.p_signal[:, bp_index]
        ppg_signal = record.p_signal[:, ppg_index]
        num_win = 0
        valid_windows = []

        dir_path = f"{cfg.DATASET_DIR}/{record.record_name}"
        if os.path.exists(dir_path):
            return

        while end_point < record.sig_len:
            win_ppg_signal = ppg_signal[start_point:end_point]
            win_bp_signal = bp_signal[start_point:end_point]
            win_ppg_sqi = SQI()
            win_ppg_sqi.calculate_sqi(win_ppg_signal)
            win_ppg_sqi_list = win_ppg_sqi.get_ski_list()
            win_bp_sqi = SQI()
            win_bp_sqi.calculate_sqi(win_bp_signal)
            win_bp_sqi_list = win_bp_sqi.get_ski_list()

            if not is_nan_value(win_bp_signal, win_ppg_signal, win_bp_sqi_list, win_ppg_sqi_list):
                new_win = create_window(win_ppg_signal, win_bp_signal, win_ppg_sqi_list,
                                        win_bp_sqi_list, record.record_name, num_win)

                if cfg.TRAIN_MODELS:
                    if new_win.ppg_target is not None and new_win.bp_target is not None:
                        valid_windows.append(new_win)
                else:
                    if utils.bp_valid(new_win.sys_bp, new_win.dias_bp):

                        # Classify window with the trained models
                        bp_platform_valid, ppg_platform_valid = self.classify_platform.valid_win(new_win)
                        if bp_platform_valid and ppg_platform_valid and new_win.ppg_target == utils.Label.good:
                            # save spectogram of the valid window
                            window_to_spectogram(new_win)

            start_point += new_samples_per_step
            end_point += new_samples_per_step
            num_win += 1

        return valid_windows

    def save_windows_list(self):
        utils.save_list(self.windows_list)
        # utils.save_win(new_win, win_name=f'{name}_{i}')

    def create_dataset(self):
        self.classify_platform.load_models()
        self.create_records_dataset()
        if cfg.TRAIN_MODELS:
            self.save_windows_list()


def is_nan_value(bp_signal, ppg_signal, bp_ski, ppg_ski):
    if np.count_nonzero(np.isnan(ppg_signal)) or np.count_nonzero(np.isnan(bp_signal)) \
            or np.count_nonzero(np.isnan(ppg_ski)) or np.count_nonzero(np.isnan(bp_ski)):
        return True

    return False


def window_to_spectogram(win):
    ppg_signal = win.ppg_signal
    fs = cfg.FREQUENCY
    noverlap = 0.96 * cfg.STFT_WIN_SIZE
    plt.specgram(ppg_signal, Fs=fs, scale='dB', NFFT=cfg.STFT_WIN_SIZE, noverlap=noverlap)
    plt.axis(ymin=cfg.FREQUENCY_START, ymax=cfg.FREQUENCY_END)
    record_name = win.win_name.split('_')[0]
    dir_path = f"{cfg.DATASET_DIR}/{record_name}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(f"{dir_path}/{record_name}_{win.sys_bp}_{win.dias_bp}.png"):
        plt.savefig(f"{dir_path}/{record_name}_{win.sys_bp}_{win.dias_bp}.png")


def create_window(win_ppg_signal, win_bp_signal, win_ppg_sqi, win_bp_sqi, record_name, num_win):
    win_ppg_target = classify_target(win_ppg_signal, cfg.FREQUENCY)
    win_bp_target = classify_target(win_bp_signal, cfg.FREQUENCY)
    sys_bp, dias_bp = utils.bp_detection(win_bp_signal)

    new_win = utils.Window(win_ppg_signal, win_bp_signal, win_ppg_target,
                           win_bp_target, win_bp_sqi, win_ppg_sqi,
                           win_name=f'{record_name}_{num_win}', sys_bp=sys_bp, dias_bp=dias_bp)

    return new_win


def main():
    assert cfg.TRAIN_MODELS == False and cfg.DATASET == cfg.Dataset.mimic

    """save list of records"""
    # if cfg.MIN_RECORDS_PER_PATIENT > 0:
    #     save_good_records_list()
    # else:
    #     save_records_list()

    """ Save spectograms """
    dc = DatasetCreator()
    # dc.create_dataset()

    """Split images to Test/Val/Test folders
       Activate only if all patients in the same directory"""
    data_path = cfg.DATASET_DIR
    arrange_folders(data_path)


if __name__ == "__main__":
    # cfg.DATASET_LOG.redirect_output()
    warnings.simplefilter(action='ignore')
    main()
    # cfg.DATASET_LOG.close_log_file()
