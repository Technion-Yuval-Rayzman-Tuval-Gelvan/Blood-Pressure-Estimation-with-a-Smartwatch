import datetime

import torch
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np

# from Code.Training import HDF5DataLoader
# from Code.Training.BPHistogram import calculate_histogram
# from Code.Training.Trainer import Trainer
from Code.Preprocessing.Project_B.Utils import filter_bp_bounds


def plot_confusion(all_categories, confusion, directory='../../Results', model_name='dias_model'):
    n_categories = len(all_categories)

    confusion_txt = np.asarray(confusion)

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        if confusion[i].sum() > 0:
            confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy(), vmin=0, vmax=0.15)
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories[::5], rotation=90)
    ax.set_yticklabels([''] + all_categories[::5])
    ax.set_ylabel('Target Class')
    ax.set_xlabel('Output Class')
    ax.set_title('Confusion Matrix')

    majorLocator = ticker.MultipleLocator(5)
    majorFormatter = ticker.FormatStrFormatter('%d')
    minorLocator = ticker.MultipleLocator(1)

    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)

    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)

    # plt.show()
    print("Save confusion matrix at:", directory)
    plt.savefig(f"{directory}/Confusion_Matrix_{model_name}.png")


def confusion_matrix(epoch_result, save_dir='../../Results', model_name='dias_model'):

    y_pred = epoch_result.pred_labels
    y = epoch_result.target_labels
    new_y, new_y_pred = filter_bp_bounds(y, y_pred, model_name)
    if model_name == 'dias_model':
        all_categories = list(range(40, 86, 1))
    else:
        all_categories = list(range(100, 151, 1))
    confusion = torch.zeros(len(all_categories), len(all_categories))
    for k in range(len(new_y)):
        confusion[all_categories.index(int(new_y[k]))][all_categories.index(int(new_y_pred[k]))] += 1
    plot_confusion(all_categories, confusion, save_dir, model_name=model_name)


def main():
    save_dir = '../../Results'
    model_path = '../../Models/Densenet_Models/29_12_2021_dias_model'
    model_name = 'dias_model'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.L1Loss()
    loss_fn = loss_fn.to(device)
    model_path = '../../Models/Densenet_Models/29_12_2021_dias_model'
    model = torch.load(model_path).to(device)
    print(f"\n*** Load checkpoint {model_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trainer = Trainer(model, loss_fn, optimizer, device)
    dl_train = HDF5DataLoader.get_hdf5_dataset('../../Data', model_name, 'Train', batch_size=64, max_chuncks=8)
    epoch_res = trainer.test_epoch(dl_train, verbose=True, max_batches=12500, plot_confusion=True)
    confusion_matrix(epoch_res)
    calculate_histogram(epoch_res.pred_labels, model_name)
    calculate_histogram(epoch_res.target_labels, model_name)


if __name__ == "__main__":
    main()
