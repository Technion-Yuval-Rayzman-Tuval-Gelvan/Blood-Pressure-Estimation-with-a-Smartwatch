import pickle
import posixpath

import numpy as np
import pandas as pd
import os
import wfdb
import copy
import os
import shutil
import posixpath
import matplotlib
import pandas as pd
import requests
import wfdb
from wfdb.io import download
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.stats import entropy
import tqdm
import time
from scipy import signal
from scipy.fft import fftshift
from multiprocessing import Pool

squered_magnitude_threshold = 200
window_in_sec = 30
frequency_end = 12
frequency_start = 0
stft_window_size = 750


class Sample:

    def __init__(self, window, systolic_bp, diastolic_bp, bp_index, ppg_index):
        self.window = window
        self.systolic_bp = systolic_bp
        self.diastolic_bp = diastolic_bp
        self.bp_index = bp_index
        self.ppg_index = ppg_index
        self.squared_magnitude_ppg = None
        self.window_number = None


def load_records(records_path):
    records = []
    # traverse root directory, and list directories as dirs and files as files
    print("loading records...")
    for root, dirs, files in tqdm.tqdm(os.walk(records_path)):
        path = root.split(os.sep)
        # print(root)
        # print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            # print(len(path) * '---', file)
            with open(f'{posixpath.join(root, file)}', 'rb') as f:
                record = pickle.load(f)
                records.append(record)
    print("loading records done")
    return records


def print_records(records):
    print("The records are:")
    for record in records:
        print(record.record_name)

    print(f"Number of records: {len(records)}")


# remove files without relevant PPG and ABP
def remove_unrelevant_records(records):
    filtered_records = []

    # remove file names without relevant data
    for record in records:
        if record.record_name[-1] != 'n':
            filtered_records.append(record)

    return filtered_records


def record_to_windows(record):
    start_point = 0
    fs = record.fs
    end_point = start_point + (window_in_sec * fs)
    window_overlap = window_in_sec / 2
    new_samples_per_step = np.int(window_overlap * fs)
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


def records_to_windows(records):
    windows = []
    for record in records:
        record_windows = record_to_windows(record)
        windows.append(record_windows)

    windows = np.concatenate(windows)
    return windows


def print_window(window):
    window_in_sec = window.sig_len / window.fs
    x = np.linspace(0, window_in_sec, window.sig_len)

    fig, axs = plt.subplots(2)
    fig.suptitle('30 Seconds Window signal')
    axs[0].plot(x, window.p_signal[:, 0])
    axs[1].plot(x, window.p_signal[:, 1])
    axs[1].set(xlabel='Time [sec]', ylabel='ABP [mmHg]')
    axs[0].set(xlabel='Time [sec]', ylabel='PLETH [NU]')
    plt.show()


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


def sample_bp_valid(sample):
    # remove windows that not in valid range
    if sample.systolic_bp > 185 or sample.systolic_bp < 55 or sample.diastolic_bp < 30 or sample.diastolic_bp > 120:
        return False

    return True


def bp_filter(windows):
    samples = []
    bp_index = windows[0].sig_name.index('ABP')
    ppg_index = windows[0].sig_name.index('PLETH')
    for window in windows:
        if window_valid(window, bp_index, ppg_index):
            bp_signal = window.p_signal[:, bp_index]
            bp_signal = np.array(bp_signal)
            systolic_bp, diastolic_bp = bp_detection(bp_signal)
            sample = Sample(window, systolic_bp, diastolic_bp, bp_index, ppg_index)
            if sample_bp_valid(sample):
                samples.append(sample)

    return samples


def window_valid(window, bp_index, ppg_index):
    bp_signal = window.p_signal[:, bp_index]
    ppg_signal = window.p_signal[:, ppg_index]
    # remove windows with nan values
    if np.count_nonzero(np.isnan(ppg_signal)) or np.count_nonzero(np.isnan(bp_signal)):
        return False

    return True


def squared_magnitude_ppg_detection(ppg_signal):
    ppg_centered = ppg_signal - np.mean(ppg_signal)
    ppg_corr = np.correlate(ppg_centered, ppg_centered, 'full')
    if ppg_corr[3750] != 0:
        ppg_corr_norm = ppg_corr / ppg_corr[3750]
        ppg_corr_norm = ppg_corr_norm[len(ppg_signal):]
        squared_magnitude_ppg = np.sum(np.power(ppg_corr_norm, 2))
    else:
        squared_magnitude_ppg = 0

    return squared_magnitude_ppg


def ppg_filter(samples):
    filtered_samples = []
    for sample in samples:
        ppg_signal = sample.window.p_signal[:, sample.ppg_index]
        sample.squared_magnitude_ppg = squared_magnitude_ppg_detection(ppg_signal)
        if sample.squared_magnitude_ppg > squered_magnitude_threshold:
            filtered_samples.append(sample)

    return filtered_samples


def save_sample(sample):
    ppg_signal = sample.window.p_signal[:, sample.ppg_index]
    fs = sample.window.fs
    noverlap = 0.96 * stft_window_size
    plt.specgram(ppg_signal, Fs=fs, scale='dB', NFFT=stft_window_size, noverlap=noverlap)
    plt.axis(ymin=frequency_start, ymax=frequency_end)
    dir_path = f"../../Data/{sample.window.record_name}"
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    plt.savefig(f"../../Data/{sample.window.record_name}/"
                f"{sample.window.record_name}_{sample.window_number}_{sample.systolic_bp}_{sample.diastolic_bp}.png")
    plt.close()


def samples_to_spectograms(samples):
    print("saving images..")
    pool = Pool()
    for _ in tqdm.tqdm(pool.imap(func=save_sample, iterable=samples), total=len(samples)):
        pass

    print("images saving done")


def get_window_numbers(samples):
    window_number = 0
    last_record_name = samples[0].window.record_name
    for sample in samples:
        sample.window_number = window_number
        window_number += 1
        if last_record_name != sample.window.record_name:
            window_number = 0
        last_record_name = sample.window.record_name

    return samples


def main():
    # records_path = '../../../mimic3wdb'
    records_path = '../../../Test_data'
    records = load_records(records_path)
    # print_records(records)
    filtered_records = remove_unrelevant_records(records)
    # print_records(filtered_records)
    windows = records_to_windows(filtered_records)
    # print_window(windows[200])
    filtered_bp_samples = bp_filter(windows)
    # print_window(filtered_bp_samples[2].window)
    filtered_ppg_samples = ppg_filter(filtered_bp_samples)
    # print_window(filtered_ppg_samples[5].window)
    samples = get_window_numbers(filtered_ppg_samples)
    samples_to_spectograms(samples)
    print(
        f"Num Windows (30 sec): {len(windows)}, After bp filter: {len(filtered_bp_samples)}, After ppg filter: {len(filtered_ppg_samples)}")


if __name__ == "__main__":
    main()
