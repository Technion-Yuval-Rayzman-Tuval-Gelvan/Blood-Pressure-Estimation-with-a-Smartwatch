import os
import pickle
from multiprocessing import Pool
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


def load_and_calc_win(win_path):
    with open(win_path, 'rb') as file:
        window = pickle.load(file)

    calculate_win_sqi(window)


def calculate_win_sqi(window):

    ppg_signal = window.record.p_signal[:, window.ppg_index]
    window.s_sqi = round(skewness_sqi(ppg_signal), 4)
    window.p_sqi = round(perfusion_sqi(ppg_signal, ppg_signal), 4)
    window.m_sqi = round(mean_crossing_rate_sqi(ppg_signal), 5)
    window.e_sqi = round(entropy_sqi(ppg_signal), 4)
    window.z_sqi = round(zero_crossings_rate_sqi(ppg_signal), 4)
    window.snr_sqi = round(float(signal_to_noise_sqi(ppg_signal)), 4)
    window.k_sqi = round(kurtosis_sqi(ppg_signal), 4)


def calc_windows_sqi():
    windows_list = [os.path.join(path, name) for path, subdirs, files in os.walk(WINDOWS_DIR) for name in files]

    for win_path in tqdm(windows_list):
        load_and_calc_win(win_path)


def main():
    calc_windows_sqi()


if __name__ == "__main__":
    main()
