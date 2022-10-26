import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import Utils as utils
import Config as cfg
import Plot as plot
import Trainer

def classify_target(signal_flags):
    quality_percent = (np.count_nonzero(signal_flags) / len(signal_flags)) * 100
    high_tresh = int(cfg.HIGH_THRESH)
    low_tresh = int(cfg.LOW_THRESH)

    if quality_percent >= high_tresh:
        target = utils.Label.good
    elif quality_percent >= low_tresh:
        target = utils.Label.mid
    else:
        target = utils.Label.bad

    return target


def preprocess_data(data):

    win_list = []

    for name, record in tqdm(data.items()):
        ppg_signal = np.array(record['IR'])
        bp_signal = np.array(record['Aline'])
        ppg_flags = np.array(record['IrFlag'])
        bp_flags = np.array(record['AlineFlag'])
        sig_len = len(ppg_signal)
        window_in_sec = 30
        i = 0

        start_point = 0
        end_point = start_point + (window_in_sec * cfg.FREQUENCY)
        window_overlap = window_in_sec / 2
        new_samples_per_step = int(window_overlap * cfg.FREQUENCY)

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
            if win_ppg_target is None or win_bp_target is None:
                return

            win_ppg_sqi = utils.SQI()
            win_ppg_sqi.calculate_sqi(win_ppg_signal)
            win_bp_sqi = utils.SQI()
            win_bp_sqi.calculate_sqi(win_bp_signal)

            start_point += new_samples_per_step
            end_point += new_samples_per_step

            new_win = utils.Window(win_ppg_signal, win_bp_signal, win_ppg_target,
                             win_bp_target, win_bp_sqi, win_ppg_sqi, win_name = f'{name}_{i}')
            i += 1

            win_list += [new_win]

    return win_list

def load_files():
    data = {}
    file_names = ["Subject #1", "Subject #2", "Subject #3", "Subject #4", "Subject #5"]
    for name in file_names:
        data[name] = pd.read_csv(f"{cfg.CARDIAC_LOAD_DIR}/{name}.csv")

    return data


def plot_signals(data):
    for record in data.values():
        ppg_signal = record['IR']
        window_in_sec = len(ppg_signal) / 256
        window_in_min = window_in_sec / 60
        x = np.linspace(0, window_in_min, len(ppg_signal))
        plt.plot(x, ppg_signal)
        plt.show()


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

    assert cfg.DATASET == cfg.Dataset.cardiac

    """get dictionary of records"""
    data = load_files()

    """plot full signals"""
    # plot_signals(data)

    # TODO: change win dict to win list in the following methods:
    """save records as windows"""
    win_list = preprocess_data(data)

    """load windows"""
    utils.save_list(win_list)

    """load_windows_dictionary"""
    win_list = utils.load_list()
    win_dict = utils.convert_list_to_dict(win_list)

    """plot windows"""
    if cfg.PLOT:
        utils.plot_windows(win_dict)

    """histogram of labels"""
    # plot.label_histogram(win_dict)
    # plot.features_histogram(win_dict)

    with pd.ExcelWriter(f'{cfg.DATA_DIR}/{cfg.TIME_DIR}/accuracy.xlsx') as excel_writer:

        """good/mid"""
        print("************************* Good / Mid **************************************")
        trainer = Trainer.Trainer(true_label=utils.Label.good, false_label=utils.Label.mid,
                                  win_dict=win_dict, excel_writer=excel_writer)
        trainer.run()

        """good/bad"""
        print("************************* Good / Bad **************************************")
        trainer = Trainer.Trainer(true_label=utils.Label.good, false_label=utils.Label.bad,
                                  win_dict=win_dict, excel_writer=excel_writer)
        trainer.run()

        """mid/bad"""
        print("************************* Mid / Bad **************************************")
        trainer = Trainer.Trainer(true_label=utils.Label.mid, false_label=utils.Label.bad,
                                  win_dict=win_dict, excel_writer=excel_writer)
        trainer.run()


if __name__ == "__main__":
    # cfg.LOG.redirect_output()
    main()
    # cfg.LOG.close_log_file()
