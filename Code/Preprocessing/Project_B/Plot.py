## Setting some nice matplotlib defaults
import os
import matplotlib.pyplot as plt
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

    ## Plotting the histograms
    fig, ax_list = plt.subplots(4, 2, figsize=(10, 8))
    for i, feature in enumerate(feature_list):
        ax = ax_list.flat[i]
        ax.hist(dataset.query('label == "Label.bad"')[feature].values, bins=20, alpha=0.5, label='bad')
        ax.hist(dataset.query('label == "Label.mid"')[feature].values, bins=20, alpha=0.5, label='mid')
        ax.hist(dataset.query('label == "Label.good"')[feature].values, bins=20, alpha=0.5, label='good')
        ax.set_title(feature)

    for ax_list2 in ax_list:
        ax_list2[0].set_ylabel('Number of samples')

    ax_list.flat[-1].legend()
    plt.tight_layout()
    fig.savefig(f'{output_dir}/features_hist.png', dpi=240)


def label_histogram(dataset):
    fig, ax = plt.subplots()
    dataset.groupby('label').size().plot.bar(ax=ax)
    ax.set_title('Label')
    ax.set_xlabel('Label')
    ax.set_ylabel('Number of samples')
    plt.tight_layout()

    fig.savefig(f'{cfg.HIST_DIR}/labels_hist.png', dpi=240)

