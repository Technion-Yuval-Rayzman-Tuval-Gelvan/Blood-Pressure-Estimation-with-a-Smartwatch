## Setting some nice matplotlib defaults
import os
import matplotlib.pyplot as plt
import pandas as pd

import Config as cfg
import Utils as utils

plt.rcParams['figure.figsize'] = (4.5, 4.5)  # Set default plot's sizes
plt.rcParams['figure.dpi'] = 120  # Set default plot's dpi (increase fonts' size)
plt.rcParams['axes.grid'] = True  # Show grid by default in figures


# get dict data set at the following format using (utils.windows_to_dict):
# win_dict = {'s_sqi': [],
#                 'p_sqi': [],
#                 'm_sqi': [],
#                 'e_sqi': [],
#                 'z_sqi': [],
#                 'snr_sqi': [],
#                 'k_sqi': [],
#                 'corr': [],
#                 'label': [],
#                 }
def features_histogram(dataset):

    output_dir = cfg.HIST_DIR
    feature_list = ['s_sqi', 'p_sqi', 'm_sqi', 'e_sqi', 'z_sqi', 'snr_sqi', 'k_sqi', 'corr']
    dataset = pd.DataFrame(dataset)

    ## Plotting the histograms
    fig, ax_list = plt.subplots(2, 4, figsize=(10, 8))
    for i, feature in enumerate(feature_list):
        ax = ax_list.flat[i]
        len_bad = len(dataset.query('label == 2')[feature].values)
        len_mid = len(dataset.query('label == 1')[feature].values)
        len_good = len(dataset.query('label == 0')[feature].values)
        good_weight = 1 - (len_good / (len_mid + len_bad + len_good))
        mid_weight = 1 - (len_mid / (len_mid + len_bad + len_good))
        bad_weight = 1 - (len_bad / (len_mid + len_bad + len_good))
        good_weights = [good_weight for i in range(len_good)]
        mid_weights = [mid_weight for i in range(len_mid)]
        bad_weights = [bad_weight for i in range(len_bad)]

        ax.hist(dataset.query('label == 2')[feature].values, bins=20, alpha=0.6, label='bad', weights=bad_weights)
        ax.hist(dataset.query('label == 1')[feature].values, bins=20, alpha=0.6, label='mid', weights=mid_weights)
        ax.hist(dataset.query('label == 0')[feature].values, bins=20, alpha=0.6, label='good', weights=good_weights)
        ax.set_title(feature)

    for ax_list2 in ax_list:
        ax_list2[0].set_ylabel('Number of samples')

    labels_list = ['bad', 'mid', 'good']
    plt.legend(labels_list)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/features_hist.png', dpi=240)


def label_histogram(dataset):
    fig, ax = plt.subplots()
    dataset = pd.DataFrame(dataset)
    dataset.groupby('label').size().plot.bar(ax=ax)
    ax.set_title('Label')
    ax.set_xlabel('Label')
    ax.set_ylabel('Number of samples')
    plt.tight_layout()

    fig.savefig(f'{cfg.HIST_DIR}/labels_hist.png', dpi=240)

