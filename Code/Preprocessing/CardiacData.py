import enum
import os
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
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
import seaborn as sns


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

    if quality_percent > 95:
        target = Label.very_good
    elif quality_percent > 80:
        target = Label.good
    elif quality_percent > 60:
        target = Label.mid
    elif quality_percent > 40:
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
                plot_win(win_ppg_signal, f'{win_ppg_target}_ppg_{name}_{win_counter}')

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


def show_histogram(windows):
    histogram = {}
    for win in windows:
        key = win.ppg_target.name
        if key in histogram:
            histogram[key] += 1
        else:
            histogram[key] = 1

    print(f"histogram: {histogram}")


def create_dataset(windows):

    data = []
    labels = []
    for win in windows:
        data.append([win.ppg_sqi.s_sqi, win.ppg_sqi.p_sqi])
        labels.append(win.ppg_target.value)

    data = np.array(data)
    labels = np.array(labels)
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                        random_state=109)  # 70% training and 30% test

    print(X_test[0])
    print(y_test[0])
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def main():
    """get dictionary of records"""
    # data = load_files()

    """plot full signals"""
    # plot_signals(data)

    """save records as windows"""
    # preprocess_data(data)

    """save records as windows"""
    windows = load_windows(WINDOWS_DIR)

    """histogram of labels"""
    # show_histogram(windows)

    """create data set for training"""
    create_dataset(windows)


if __name__ == "__main__":
    main()
