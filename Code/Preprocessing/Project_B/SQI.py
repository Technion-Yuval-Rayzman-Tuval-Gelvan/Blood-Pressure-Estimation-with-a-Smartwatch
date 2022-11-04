import numpy as np
from vital_sqi.sqi.standard_sqi import (perfusion_sqi, kurtosis_sqi, skewness_sqi,
                                        entropy_sqi, signal_to_noise_sqi, zero_crossings_rate_sqi,
                                        mean_crossing_rate_sqi)


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

    def get_ski_list(self):

        return [self.s_sqi, self.p_sqi, self.m_sqi, self.e_sqi, self.snr_sqi, self.k_sqi, self.corr_sqi]


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