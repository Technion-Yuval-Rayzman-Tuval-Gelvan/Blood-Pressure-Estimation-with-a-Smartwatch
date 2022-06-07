import enum
import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from vital_sqi.sqi.standard_sqi import (
    perfusion_sqi,
    kurtosis_sqi,
    skewness_sqi,
    entropy_sqi,
    signal_to_noise_sqi,
    zero_crossings_rate_sqi,
    mean_crossing_rate_sqi
)
import heartpy as hp


FREQUENCY = 256
BASE_DIR = '/host/media/tuvalgelvan@staff.technion.ac.il/HD34/Estimated-Blood-Pressure-Project'
LOAD_DIR = f'{BASE_DIR}/mimic3wdb/1.0'
CARDIAC_DIR = f'{BASE_DIR}/cardiac_data'
WINDOWS_DIR = f'{BASE_DIR}/cardiac_data/windows'
PLOT_DIR = f'{BASE_DIR}/cardiac_data/plots'
PLOT = True


class Label(enum.Enum):
    very_good = 0
    good = 1
    mid = 2
    bad = 3
    very_bad = 4


class SQI:

    def __init__(self):
        self.s_sqi = None
        self.p_sqi = None
        self.m_sqi = None
        self.e_sqi = None
        self.z_sqi = None
        self.snr_sqi = None
        self.k_sqi = None

    def calculate_sqi(self, signal):
        self.s_sqi = round(skewness_sqi(signal), 4)
        self.p_sqi = round(perfusion_sqi(signal, signal), 4)
        self.m_sqi = round(mean_crossing_rate_sqi(signal), 5)
        self.e_sqi = round(entropy_sqi(signal), 4)
        self.z_sqi = round(zero_crossings_rate_sqi(signal), 4)
        self.snr_sqi = round(float(signal_to_noise_sqi(signal)), 4)
        self.k_sqi = round(kurtosis_sqi(signal), 4)


class Window:

    def __init__(self, record, ppg_signal, bp_signal, ppg_target, bp_target, bp_sqi, ppg_sqi):
        self.record = record
        self.bp_signal = bp_signal
        self.ppg_signal = ppg_signal
        self.ppg_target = ppg_target
        self.bp_target = bp_target
        self.bp_sqi = bp_sqi
        self.ppg_sqi = ppg_sqi


def classify_target(signal_flags):
    quality_percent = (np.count_nonzero(signal_flags) / len(signal_flags)) * 100

    if quality_percent > 90:
        target = Label.very_good
    elif quality_percent > 70:
        target = Label.good
    elif quality_percent > 30:
        target = Label.mid
    elif quality_percent > 10:
        target = Label.bad
    else:
        target = Label.very_bad

    return target


def save_win(win, win_name):
    with open(f"{WINDOWS_DIR}/{win_name}", 'wb') as file:
        pickle.dump(win, file)


def load_win(win_path):
    with open(win_path, 'rb') as file:
        window = pickle.load(file)

    return window


def load_windows(path_dir):
    windows_list = [os.path.join(path, name) for path, subdirs, files in os.walk(path_dir) for name in files]

    windows = []
    for win_path in tqdm(windows_list):
        windows.append(load_win(win_path))

    return windows


def preprocess_data(data):

    for name, record in tqdm(data.items()):
        ppg_signal = np.array(record['IR'])
        bp_signal = np.array(record['Aline'])
        ppg_flags = np.array(record['IrFlag'])
        bp_flags = np.array(record['AlineFlag'])
        sig_len = len(ppg_signal)
        window_in_sec = 30
        win_counter = 0

        start_point = 0
        end_point = start_point + (window_in_sec * FREQUENCY)
        window_overlap = window_in_sec / 2
        new_samples_per_step = int(window_overlap * FREQUENCY)

        while end_point < sig_len:
            win_ppg_signal = ppg_signal[start_point:end_point]
            win_bp_signal = bp_signal[start_point:end_point]

            if np.count_nonzero(np.isnan(win_ppg_signal)) or np.count_nonzero(np.isnan(win_bp_signal)):
                start_point += new_samples_per_step
                end_point += new_samples_per_step
                print(f"invalid window {name}_{win_counter}")
                continue

            win_ppg_flags = ppg_flags[start_point:end_point]
            win_bp_flags = bp_flags[start_point:end_point]
            win_ppg_target = classify_target(win_ppg_flags)
            win_bp_target = classify_target(win_bp_flags)
            win_ppg_sqi = SQI()
            win_ppg_sqi.calculate_sqi(win_ppg_signal)
            win_bp_sqi = SQI()
            win_bp_sqi.calculate_sqi(win_bp_signal)

            start_point += new_samples_per_step
            end_point += new_samples_per_step

            new_win = Window(record, win_ppg_signal, win_bp_signal, win_ppg_target,
                             win_bp_target, win_bp_sqi, win_ppg_sqi)
            save_win(new_win, win_name=f'{name}_{win_counter}')

            if PLOT:
                plot_win(win_ppg_signal, f'ppg_{name}_{win_counter}')

            win_counter += 1


def plot_win(win, name):
    hp.config.colorblind = False
    hp.config.color_style = 'default'

    try:
        wd, m = hp.process(win, FREQUENCY)
        hp.plotter(wd, m, show=False)
    except:
        print(f"Bad record: {name}")
        return

    plt.savefig(f"{PLOT_DIR}/{name}.png")


def load_files():
    data = {}
    file_names = ["Subject #1", "Subject #2", "Subject #3", "Subject #4", "Subject #5"]
    for name in file_names:
        data[name] = pd.read_csv(f"{CARDIAC_DIR}/Technion_Synched_Data/{name}.csv")

    return data


def plot_signals(data):
    for record in data.values():
        ppg_signal = record['IR']
        window_in_sec = len(ppg_signal) / 256
        window_in_min = window_in_sec / 60
        x = np.linspace(0, window_in_min, len(ppg_signal))
        plt.plot(x, ppg_signal)
        plt.show()


def main():
    data = load_files()
    # plot_signals(data)
    preprocess_data(data)
    windows = load_windows(WINDOWS_DIR)
    print(len(windows))
    labels = []
    sqi_data = []
    for win in windows:
        sqi_data.append(win.ppg_sqi.s_sqi)
        labels.append(win.ppg_target)

    hist, bin_edges = np.histogram(labels)
    print(hist)

    print(labels)
    print(sqi_data)



if __name__ == "__main__":
    main()
