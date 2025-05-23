import enum
import os
import pickle
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.signal import find_peaks
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from SQI import SQI
import heartpy as hp
from Code.Preprocessing.Project_B import Config as cfg

class Label(enum.Enum):
    good = 0
    mid = 1
    bad = 2


class Window:

    def __init__(self, ppg_signal, bp_signal, ppg_target, bp_target, bp_sqi, ppg_sqi, win_name, sys_bp, dias_bp):
        self.bp_signal = bp_signal
        self.ppg_signal = ppg_signal
        self.ppg_target = ppg_target
        self.bp_target = bp_target
        self.bp_sqi = bp_sqi
        self.ppg_sqi = ppg_sqi
        self.win_name = win_name
        self.sys_bp = sys_bp
        self.dias_bp = dias_bp


def remove_bp_bounds(y, y_pred, model_name):
    y = np.array(y.cpu())
    y_pred = np.array(y_pred.cpu())
    if model_name == 'dias_model':
        indices = np.where((y < 85) & (y > 45))
    else:
        indices = np.where((y > 95) & (y < 150))

    if len(indices) == 0:
        new_y = []
        new_y_pred = []
    else:
        new_y = y[indices]
        new_y_pred = y_pred[indices]

    return torch.Tensor(new_y), torch.Tensor(new_y_pred)


def filter_bp_bounds(y, y_pred, model_name):
    assert len(y) == len(y_pred)

    for i in range(len(y)):
        if model_name == 'dias_model':
            if y[i] > 85:
                y[i] = 85
            if y[i] < 45:
                y[i] = 45
            if y_pred[i] > 85:
                y_pred[i] = 85
            if y_pred[i] < 45:
                y_pred[i] = 45
        if model_name == 'sys_model':
            if y[i] > 150:
                y[i] = 150
            if y[i] < 95:
                y[i] = 95
            if y_pred[i] > 150:
                y_pred[i] = 150
            if y_pred[i] < 95:
                y_pred[i] = 95

    return y, y_pred


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
    if cfg.WORK_MODE == cfg.Mode.compare_models:
        save_dir = cfg.COMPARE_DIR
    else:
        save_dir = cfg.DATA_DIR

    with open(f"{save_dir}/windows_list", 'wb') as file:
        pickle.dump(list, file)


def load_list():
    if cfg.WORK_MODE == cfg.Mode.compare_models:
        save_dir = cfg.COMPARE_DIR
    else:
        save_dir = cfg.DATA_DIR

    with open(f"{save_dir}/windows_list", 'rb') as file:
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

    save_dir = cfg.COMPARE_DIR if cfg.WORK_MODE == cfg.Mode.compare_models else cfg.PLOT_DIR

    if not os.path.exists(os.path.dirname(f"{save_dir}/{name}.png")):
        os.makedirs(os.path.dirname(f"{save_dir}/{name}.png"))

    plt.savefig(f"{save_dir}/{name}.png")
    plt.close()


def plot_signal(signal, name, is_ppg):
    hp.config.colorblind = False
    hp.config.color_style = 'default'

    try:
        wd, m = hp.process(signal, cfg.FREQUENCY)
        hp.plotter(wd, m, show=False)
    except:
        print(f"Bad record: {name}")
        return

    if is_ppg:
        dir = f"{cfg.CLASSIFIED_PLOTS}/ppg"
    else:
        dir = f"{cfg.CLASSIFIED_PLOTS}/bp"

    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(f"{dir}/{name}.png")


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


def bp_valid(systolic_bp, diastolic_bp):

    # remove windows that not in valid range
    if systolic_bp > 185 or systolic_bp < 55 or diastolic_bp < 30 or diastolic_bp > 120:
        return False

    return True


def compare_heartpy_sqi_model(win, ppg_platform, ppg_heartpy):

    # good model - bad heartpy
    if ppg_platform and not ppg_heartpy == Label.good:
        plot_win(win.ppg_signal, f'good_model_bad_heartpy/{win.win_name}')

    # bad model - good heartpy
    if not ppg_platform and ppg_heartpy == Label.good:
        plot_win(win.ppg_signal, f'bad_model_good_heartpy/{win.win_name}')

    # good model - good heartpy
    if ppg_platform and ppg_heartpy == Label.good:
        plot_win(win.ppg_signal, f'good_model_good_heartpy/{win.win_name}')

    # bad model - bad heartpy
    if not ppg_platform and not ppg_heartpy == Label.good:
        plot_win(win.ppg_signal, f'bad_model_bad_heartpy/{win.win_name}')
