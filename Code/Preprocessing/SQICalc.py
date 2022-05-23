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
from Code.Preprocessing.SignalQualityPlatform import TEST_DIR, Window


WINDOWS_DIR = '/host/media/tuvalgelvan@staff.technion.ac.il/HD34/Estimated-Blood-Pressure-Project/test_data/windows'


def load_and_calculate_win_sqi(win_path):
    with open(win_path, 'rb') as file:
        window = pickle.load(file)

    ppg_signal = window.record.p_signal[:, window.ppg_index]
    window.s_sqi = round(skewness_sqi(ppg_signal), 2)
    window.p_sqi = round(perfusion_sqi(ppg_signal, ppg_signal), 2)
    window.m_sqi = round(mean_crossing_rate_sqi(ppg_signal), 2)
    window.e_sqi = round(entropy_sqi(ppg_signal), 2)
    window.z_sqi = round(zero_crossings_rate_sqi(ppg_signal), 2)
    window.snr_sqi = round(float(signal_to_noise_sqi(ppg_signal)), 2)
    window.k_sqi = round(kurtosis_sqi(ppg_signal), 2)

    print(window.s_sqi, window.p_sqi, window.m_sqi, window.e_sqi, window.z_sqi, window.snr_sqi, window.k_sqi)


def calc_windows_sqi():
    windows_list = [os.path.join(path, name) for path, subdirs, files in os.walk(WINDOWS_DIR) for name in files]

    for win_path in windows_list:
        load_and_calculate_win_sqi(win_path)


def main():
    calc_windows_sqi()


if __name__ == "__main__":
    main()
