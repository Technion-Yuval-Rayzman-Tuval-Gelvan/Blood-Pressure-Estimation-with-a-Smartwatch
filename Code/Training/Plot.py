import glob
import math
import os
import re
import sys

import numpy as np
import itertools
import matplotlib.pyplot as plt
from Code.Preprocessing.Project_B import Config as cfg
from Code.Training.Experiments import load_experiment
from Code.Training.TrainResult import FitResult


def plot_fit(
    fit_res: FitResult,
    fig=None,
    log_loss=False,
    legend=None,
    train_test_overlay: bool = False,
    window_size=1000
):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :param train_test_overlay: Whether to overlay train/test plots on the same axis.
    :return: The figure.
    """
    if fig is None:
        nrows = 1 if train_test_overlay else 2
        ncols = 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8 * ncols, 5 * nrows),
            sharex="col",
            sharey=False,
            squeeze=False,
        )
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(enumerate(["train", "test"]), enumerate(["loss", "acc"]))
    for (i, traintest), (j, lossacc) in p:

        ax = axes[j if train_test_overlay else i * 2 + j]

        attr = f"{traintest}_{lossacc}"
        data = getattr(fit_res, attr)
        label = traintest if train_test_overlay else legend
        if attr == 'train_loss' or attr == 'test_loss':
            data = np.mean(np.array(data).reshape(-1, window_size), axis=1)
        h = ax.plot(np.arange(1, len(data)*window_size + 1, window_size), data, label=label)
        ax.set_title(attr)

        if lossacc == "loss":
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Loss")
            if log_loss:
                ax.set_yscale("log")
                ax.set_ylabel("Loss (log)")
        else:
            ax.set_xlabel("Iteration #")
            ax.set_ylabel("Accuracy (%)")

        if legend or train_test_overlay:
            ax.legend()
        ax.grid(True)

    return fig, axes


def plot_exp_results(filename_pattern, window_size=1000, results_dir='../../Results/experiments/resnet18/'):
    fig = None
    result_files = glob.glob(os.path.join(results_dir, filename_pattern))
    result_files.sort()
    if len(result_files) == 0:
        print(f'No results found for pattern {filename_pattern}.', file=sys.stderr)
        return
    for filepath in result_files:
        m = re.match('exp_tr_(\d_)?(.*)\.json', os.path.basename(filepath))
        print(m[0], m)
        cfg, fit_res = load_experiment(filepath)
        fig, axes = plot_fit(fit_res, fig, legend=m[0], log_loss=False, window_size=window_size)

    print('common config: ', cfg)
    plt.show()


def main():
    # plot_exp_results('exp1_1*.json')
    # plot_exp_results('tutorial_64*_dias_model*.json', results_dir=cfg.RESNET_MODELS, window_size=1)
    # plot_exp_results('exp_tr_64*_dias_model*.json', results_dir=cfg.RESNET_RESULTS, window_size=50)
    # plot_exp_results('exp_tr_64*_sys_model*.json', results_dir=cfg.RESNET_RESULTS, window_size=50)
    # plot_exp_results('exp_tr_16*09*_dias_model*.json', results_dir=cfg.DENSENET_RESULTS, window_size=50)
    plot_exp_results('exp_tr_16*10*_sys_model*.json', results_dir=cfg.DENSENET_RESULTS, window_size=50)

if __name__ == "__main__":
    main()