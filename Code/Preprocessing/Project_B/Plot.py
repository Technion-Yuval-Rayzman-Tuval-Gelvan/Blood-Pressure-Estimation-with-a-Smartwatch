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
def histogram(dataset):

    if not os.path.exists(f'{cfg.DATA_DIR}/histogram_plots'):
        os.mkdir(f'{cfg.DATA_DIR}/histogram_plots')

    feature_list = ['s_sqi', 'p_sqi', 'm_sqi', 'e_sqi', 'z_sqi', 'snr_sqi', 'k_sqi', 'corr']

    ## Plotting the histograms
    fig, ax_list = plt.subplots(4, 5, figsize=(10, 8))
    for i, feature in enumerate(feature_list):
        ax = ax_list.flat[i]
        ax.hist(dataset.query('label == "male"')[feature].values, bins=20, alpha=0.5, label='Male')
        ax.hist(dataset.query('label == "female"')[feature].values, bins=20, alpha=0.5, label='Female')
        ax.set_title(feature)

    for ax_list2 in ax_list:
        ax_list2[0].set_ylabel('Number of samples')

    ax_list.flat[-1].legend()
    plt.tight_layout()
    fig.savefig('./output/voices_distributions.png', dpi=240)




