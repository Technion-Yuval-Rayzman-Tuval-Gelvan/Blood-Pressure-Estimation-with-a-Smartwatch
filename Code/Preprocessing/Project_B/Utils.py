import enum
import os
import pickle
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from vital_sqi.sqi.standard_sqi import (perfusion_sqi, kurtosis_sqi, skewness_sqi,
                                        entropy_sqi, signal_to_noise_sqi, zero_crossings_rate_sqi,
                                        mean_crossing_rate_sqi)
import heartpy as hp
import Config as cfg


class Label(enum.Enum):
    good = 0
    mid = 1
    bad = 2


class SQI:

    def __init__(self):
        self.s_sqi = None
        self.p_sqi = None
        self.m_sqi = None
        self.e_sqi = None
        self.z_sqi = None
        self.snr_sqi = None
        self.k_sqi = None
        self.corr_sqi = None

    def calculate_sqi(self, signal):
        self.s_sqi = round(skewness_sqi(signal), 4)
        self.p_sqi = round(perfusion_sqi(signal, signal), 4)
        self.m_sqi = round(mean_crossing_rate_sqi(signal), 5)
        self.e_sqi = round(entropy_sqi(signal), 4)
        self.z_sqi = round(zero_crossings_rate_sqi(signal), 4)
        self.snr_sqi = round(float(signal_to_noise_sqi(signal)), 4)
        self.k_sqi = round(kurtosis_sqi(signal), 4)
        self.corr_sqi = round(calculate_corr_sqi(signal), 4)


class Window:

    def __init__(self, ppg_signal, bp_signal, ppg_target, bp_target, bp_sqi, ppg_sqi, win_name):
        self.bp_signal = bp_signal
        self.ppg_signal = ppg_signal
        self.ppg_target = ppg_target
        self.bp_target = bp_target
        self.bp_sqi = bp_sqi
        self.ppg_sqi = ppg_sqi
        self.win_name = win_name


def calculate_corr_sqi(signal):
    signal_centered = signal - np.mean(signal)
    signal_corr = np.correlate(signal_centered, signal_centered, 'full')
    if signal_corr[3750] != 0:
        corr_norm = signal_corr / signal_corr[3750]
        corr_norm = corr_norm[len(signal):]
        squared_magnitude = np.sum(np.power(corr_norm, 2))
    else:
        squared_magnitude = 0

    return squared_magnitude


def save_model(model, model_name):
    with open(f"{cfg.MODELS_DIR}/{model_name}.pkl", 'wb') as file:
        pickle.dump(model, file)


def load_model(dir_path, model_name):
    with open(f"{dir_path}/{model_name}.pkl", 'rb') as file:
        model = pickle.load(file)
    return model


def save_win(win, win_name):
    with open(f"{cfg.WINDOWS_DIR}/{win_name}", 'wb') as file:
        pickle.dump(win, file)


def save_list(list):
    with open(f"{cfg.DATA_DIR}/windows_list", 'wb') as file:
        pickle.dump(list, file)


def load_list():
    with open(f"{cfg.DATA_DIR}/windows_list", 'rb') as file:
        win_list = pickle.load(file)
    return win_list


def load_win(win_path):
    try:
        with open(win_path, 'rb') as file:
            window = pickle.load(file)
    except:
        return None

    return window


def load_windows():
    windows_list = [os.path.join(path, name) for path, subdirs, files in os.walk(cfg.WINDOWS_DIR) for name in files]

    windows = []
    print("loading windows..")
    pool = Pool()
    for window in tqdm(pool.imap(func=load_win, iterable=windows_list), total=len(windows_list)):
        if window is not None:
            windows.append(window)

    return windows


def convert_list_to_dict(windows_list):
    win_dict = {'s_sqi': [], 'p_sqi': [], 'm_sqi': [], 'e_sqi': [],
                'z_sqi': [], 'snr_sqi': [], 'k_sqi': [], 'corr': [], 'label': [], 'signal': []}

    for window in tqdm(windows_list):
        if window is not None:
            add_window_to_dict(window, win_dict)

    return win_dict


def plot_windows(win_dict):
    plot_counters = {0: 0, 1: 0, 2: 0}
    signals = win_dict['signal']
    labels = win_dict['label']
    for i in range(len(labels)):
        if plot_counters[labels[i]] > cfg.MAX_PLOT_PER_LABEL:
            continue
        else:
            plot_counters[labels[i]] += 1
            plot_win(signals[i], f'{Label(labels[i])}_ppg_{i}')


def plot_win(win, name):
    hp.config.colorblind = False
    hp.config.color_style = 'default'

    try:
        wd, m = hp.process(win, cfg.FREQUENCY)
        hp.plotter(wd, m, show=False)
    except:
        print(f"Bad record: {name}")
        return

    plt.savefig(f"{cfg.PLOT_DIR}/{name}.png")


def show_histogram(windows):
    histogram = {}
    for win in windows:
        key = win.ppg_target.name
        if key in histogram:
            histogram[key] += 1
        else:
            histogram[key] = 1

    print(f"histogram: {histogram}")


def add_window_to_dict(window, win_dict):
    if cfg.SIGNAL_TYPE == 'ppg':
        win_sqi = window.ppg_sqi
        win_label = window.ppg_target
        win_signal = window.ppg_signal
    else:
        win_sqi = window.bp_sqi
        win_label = window.bp_target
        win_signal = window.bp_signal

    win_dict['s_sqi'].append(win_sqi.s_sqi)
    win_dict['p_sqi'].append(win_sqi.p_sqi)
    win_dict['e_sqi'].append(win_sqi.e_sqi)
    win_dict['m_sqi'].append(win_sqi.m_sqi)
    win_dict['z_sqi'].append(win_sqi.z_sqi)
    win_dict['snr_sqi'].append(win_sqi.snr_sqi)
    win_dict['k_sqi'].append(win_sqi.k_sqi)
    win_dict['corr'].append(win_sqi.corr_sqi)
    win_dict['label'].append(win_label.value)
    win_dict['signal'].append(win_signal)


def bp_detection(signal):
    max_peaks, _ = find_peaks(signal)
    min_peaks, _ = find_peaks(-signal)
    systolic_bp, diastolic_bp = 0, 0

    if len(max_peaks) != 0:
        sys_peaks = max_peaks[signal[max_peaks] >= np.mean(signal[max_peaks])]
        if len(sys_peaks) != 0:
            systolic_bp = np.mean(signal[sys_peaks])

    if len(min_peaks) != 0:
        dias_peaks = min_peaks[signal[min_peaks] <= np.mean(signal[min_peaks])]
        if len(dias_peaks) != 0:
            diastolic_bp = np.mean(signal[dias_peaks])

    return systolic_bp, diastolic_bp


def bp_valid(signal):

    systolic_bp, diastolic_bp = bp_detection(signal)
    # remove windows that not in valid range
    if systolic_bp > 185 or systolic_bp < 55 or diastolic_bp < 30 or diastolic_bp > 120:
        return False

    return True