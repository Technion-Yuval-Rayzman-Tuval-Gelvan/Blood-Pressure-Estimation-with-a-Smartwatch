import copy
import pickle
from multiprocessing import Pool
import numpy as np
import os
from tqdm import tqdm
import heartpy as hp
import Utils as utils
import Config as cfg
import Plot as plot

# load database with ABP and PLETH signals
def create_records_dataset(num_patients = 0):
    # load list
    if cfg.MIN_RECORDS_PER_PATIENT > 0:
        file_name = f'{cfg.MIN_RECORDS_PER_PATIENT}_records_per_patient_list'
    else:
        file_name = 'records_list'

    with open(file_name, 'rb') as file:
        records_list = pickle.load(file)

    records_list = records_list[:num_patients]

    sampled_records_list = []
    for patient_records in records_list:
        sampled_records = np.random.choice(np.array(patient_records), cfg.TRAIN_RECORDS_PER_PATIENT)
        sampled_records_list.append(sampled_records)

    sampled_records_list = np.concatenate(np.array(sampled_records_list))

    print("loading records..")
    pool = Pool()
    for _ in tqdm(pool.imap(func=create_record_dataset, iterable=sampled_records_list), total=len(sampled_records_list)):
        pass
    # for record_path in records_list:
    #     load_record(record_path)


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

    if quality_percent > 90:
        target = utils.Label.good
    elif quality_percent > 60:
        target = utils.Label.mid
    else:
        target = utils.Label.bad

    return target


def save_valid_windows(win, ppg_index, bp_index, record, i):
    win_ppg_signal = win.p_signal[:, ppg_index]
    win_bp_signal = win.p_signal[:, bp_index]

    name = record.record_name
    win_ppg_target = classify_target(win_ppg_signal, record.fs)
    win_bp_target = classify_target(win_bp_signal, record.fs)
    if win_ppg_target is None or win_bp_target is None:
        return

    win_ppg_sqi = utils.SQI()
    win_ppg_sqi.calculate_sqi(win_ppg_signal)
    win_bp_sqi = utils.SQI()
    win_bp_sqi.calculate_sqi(win_bp_signal)

    new_win = utils.Window(record, win_ppg_signal, win_bp_signal, win_ppg_target,
                     win_bp_target, win_bp_sqi, win_ppg_sqi)

    utils.save_win(new_win, win_name=f'{name}_{i}')

    if cfg.PLOT:
        utils.plot_win(win_ppg_signal, f'{win_ppg_target}_ppg_{name}_{i}')


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
            bp_signal = win.p_signal[:, bp_index]
            ppg_signal = win.p_signal[:, ppg_index]
            if not np.count_nonzero(np.isnan(ppg_signal)) and not np.count_nonzero(np.isnan(bp_signal)):
                save_valid_windows(win, ppg_index, bp_index, record, i)


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
    # create_records_dataset(num_patients=20)

    """load windows"""
    windows = utils.load_windows()

    """histogram of labels"""
    utils.show_histogram(windows)
    dataset = utils.windows_to_dict(windows)
    plot.label_histogram(dataset)
    plot.features_histogram(dataset)

    """create data set for training"""
    utils.create_dataset(windows)


if __name__ == "__main__":
    main()
