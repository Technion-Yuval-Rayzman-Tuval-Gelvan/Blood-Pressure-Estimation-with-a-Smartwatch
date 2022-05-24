import glob
import pickle
from multiprocessing import Pool
# import vital_sqi as vs
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath
import pandas as pd
import requests
import wfdb
from tqdm import tqdm
from wfdb.io import download
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.stats import entropy
# You can also use Pandas if you so desire
import pandas as pd
import heartpy as hp
import PyQt5
from scipy.stats import kurtosis, skew, entropy

from Code.Preprocessing.SQICalc import calculate_win_sqi
from PreProcessing import remove_unrelevant_records, record_to_windows, window_valid
import os

DB_DIR = 'mimic3wdb'
BASE_DIR = '/host/media/tuvalgelvan@staff.technion.ac.il/HD34/Estimated-Blood-Pressure-Project/'
LOAD_DIR = f'{BASE_DIR}/mimic3wdb/1.0'
TEST_DIR = f'{BASE_DIR}/test_data'
WINDOWS_DIR = f'{BASE_DIR}/test_data/windows'
PLOT_DIR = f'{BASE_DIR}/test_data/plots'
PLOT = False


class Window:

    def __init__(self, window, bp_index, ppg_index, wd, m):
        self.record = window
        self.bp_index = bp_index
        self.ppg_index = ppg_index
        self.working_data = wd
        self.measures = m
        self.s_sqi = None
        self.p_sqi = None
        self.m_sqi = None
        self.e_sqi = None
        self.z_sqi = None
        self.snr_sqi = None
        self.k_sqi = None


# load database with ABP and PLETH signals
def load_filtered_records(max_len=None):
    # load list
    with open('records_list', 'rb') as file:
        records_list = pickle.load(file)

    if max_len:
        records_list = records_list[:max_len]

    print("loading records..")
    pool = Pool()
    for _ in tqdm(pool.imap(func=create_record_dataset, iterable=records_list), total=len(records_list)):
        pass
    # for record_path in records_list:
    #     load_record(record_path)


def plot_win(wd, m, name):
    hp.config.colorblind = False
    hp.config.color_style = 'default'

    try:
        hp.plotter(wd, m, show=False)
    except:
        print(f"Bad plot: {name}")
        return

    plt.savefig(f"{PLOT_DIR}/{name}.png")


def save_win(win, win_name):
    with open(f"{WINDOWS_DIR}/{win_name}", 'wb') as file:
        pickle.dump(win, file)


def save_valid_windows(win, ppg_index, bp_index, record, i):
    ppg_signal = win.p_signal[:, ppg_index]

    try:
        wd, m = hp.process(ppg_signal, record.fs)
    except:
        print(f"Bad record: {record.record_name} Win: {i}")
        return

    peaks_len = len(wd['peaklist'])
    num_bad_peaks = np.count_nonzero(wd['RR_masklist'])
    bad_peak_precent = (num_bad_peaks / peaks_len) * 100
    win = Window(win, bp_index, ppg_index, wd, m)
    if bad_peak_precent < 2:
        win_name = f'good/{record.record_name}_{i}'
        if PLOT:
            plot_win(wd, m, name=win_name)
        calculate_win_sqi(win)
        save_win(win, win_name)
    if bad_peak_precent > 80:
        win_name = f'/bad/{record.record_name}_{i}'
        if PLOT:
            plot_win(wd, m, name=win_name)
        calculate_win_sqi(win)
        save_win(win, win_name)
    if 48 < bad_peak_precent < 52:
        win_name = f'/mid/{record.record_name}_{i}'
        if PLOT:
            plot_win(wd, m, name=win_name)
        calculate_win_sqi(win)
        save_win(win, win_name)


def create_record_dataset(record_path):
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
            if window_valid(win, bp_index, ppg_index):
                save_valid_windows(win, ppg_index, bp_index, record, i)


def save_records_list():
    records_list = [os.path.join(path, name) for path, subdirs, files in os.walk(LOAD_DIR) for name in files]
    # save list
    with open('records_list', 'wb') as file:
        pickle.dump(records_list, file)


def main():
    max_len = 50
    # save_records_list()
    load_filtered_records(max_len)


if __name__ == "__main__":
    main()
