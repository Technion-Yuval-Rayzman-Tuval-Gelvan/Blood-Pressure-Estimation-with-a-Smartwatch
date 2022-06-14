import enum
import os
import pickle
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from vital_sqi.sqi.standard_sqi import (perfusion_sqi, kurtosis_sqi, skewness_sqi,
    entropy_sqi, signal_to_noise_sqi, zero_crossings_rate_sqi, mean_crossing_rate_sqi)
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


def save_win(win, win_name):
    with open(f"{cfg.WINDOWS_DIR}/{win_name}", 'wb') as file:
        pickle.dump(win, file)


def load_win(win_path):
    with open(win_path, 'rb') as file:
        window = pickle.load(file)

    return window


def load_windows():
    windows_list = [os.path.join(path, name) for path, subdirs, files in os.walk(cfg.WINDOWS_DIR) for name in files]

    windows = []
    print("loading windows..")
    pool = Pool()
    for window in tqdm(pool.imap(func=load_win, iterable=windows_list), total=len(windows_list)):
        windows.append(window)

    # for win_path in tqdm(windows_list):
    #     windows.append(load_win(win_path))

    return windows


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


def create_dataset(windows):

    data = []
    labels = []
    for win in windows:
        if win.ppg_target.value != 1:
            data.append([win.ppg_sqi.s_sqi, win.ppg_sqi.p_sqi])
            labels.append(win.ppg_target.value)

    data = np.array(data)
    labels = np.array(labels)
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                        random_state=109)  # 80% training and 20% test

    #Create a svm Classifier
    clf = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print(y_pred, y_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # create a mesh to plot in
    h = .02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.legend()
    plt.show()
