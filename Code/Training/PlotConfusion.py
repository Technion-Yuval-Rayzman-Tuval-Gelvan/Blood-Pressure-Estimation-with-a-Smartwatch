import datetime

import torch
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np


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

    plt.show()
    save_path = f"{directory}/confusion_matrix"
    now = datetime.now()
    plt.savefig(f"{save_path}/Confusion_Matrix_{now}_{model_name}.png")


def confusion_matrix(epoch_result, save_dir='../../Results', model_name='dias_model'):

    y_pred = np.array(epoch_result.pred_labels)
    y = np.array(epoch_result.target_labels)
    print(y_pred, y)
    all_categories = list(range(40, 101, 1))
    confusion = torch.zeros(len(all_categories), len(all_categories))
    for k in range(len(y)):
        if y_pred[k] < 40 or y[k] < 40:
            y[k] = 40
            y_pred[k] = 40
        if y_pred[k] > 100 or y[k] > 100:
            y[k] = 100
            y_pred[k] = 100
        confusion[all_categories.index(int(y[k]))][all_categories.index(int(y_pred[k]))] += 1
    plot_confusion(all_categories, confusion, save_dir, model_name=model_name)


def main():
    save_dir = '../../Results'
    model_path = '../../Models/Densenet_Models/29_12_2021_dias_model'
    model_name = 'dias_model'

    y = np.array([47, 82, 53])
    y_pred = np.array([50, 75, 60])
    confusion_matrix(y, y_pred, save_dir)


if __name__ == "__main__":
    main()
