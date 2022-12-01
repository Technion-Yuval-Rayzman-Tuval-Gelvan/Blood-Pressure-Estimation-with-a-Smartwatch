import abc
import os
import sys
import tqdm.auto
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import torch
import torch.autograd as autograd
from torch import Tensor, nn
from torch.utils.data import DataLoader

from Code.Preprocessing.Project_B.Utils import filter_bp_bounds

sys.path.append('../../Code/Training')
import Code.Training.ResNet
from typing import Any, Tuple, Callable, Optional, cast
sys.path.append('../../Code/Training/TrainResult.py')
from Code.Training.TrainResult import FitResult, EpochResult, BatchResult


class Trainer(abc.ABC):

    def __init__(
            self,
            model,
            loss_fn,
            optimizer,
            device: Optional[torch.device] = None,
            scheduler=None,
            model_name = 'dias_model'
    ):
        """
        Initialize the trainer.
        :param model: Instance of the classifier model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.model_name = model_name

        if self.device:
            model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs: int,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every: int = 1,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """

        actual_num_epochs = 0
        epochs_without_improvement = 0

        train_loss, train_acc, test_loss, test_acc = [], [], [], []
        target_labels, pred_labels = [], []
        best_loss = None

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if print_every > 0 and (
                epoch % print_every == 0 or epoch == num_epochs - 1
            ):
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            actual_num_epochs += 1
            train_result = self.train_epoch(dl_train, verbose=verbose, **kw)
            test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            train_loss.extend(train_result.losses)
            train_acc.append(train_result.accuracy)
            test_loss.extend(test_result.losses)
            test_acc.append(test_result.accuracy)

            if self.scheduler is True:
                self.scheduler.step(test_loss)

            if best_loss is None or np.mean(test_result.losses) < best_loss:
                best_loss = np.mean(test_result.losses)
                epochs_without_improvement = 0
                if checkpoints is not None:
                    self.save_checkpoint(checkpoints)
            else:
                epochs_without_improvement += 1
                if early_stopping == epochs_without_improvement:
                    final_model_path = f"{checkpoints[:-3]}_final.pt"
                    self.save_checkpoint(final_model_path)
                    return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

        final_model_path = f"{checkpoints[:-3]}_final.pt"
        self.save_checkpoint(final_model_path)
        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def save_checkpoint(self, checkpoint_filename: str):
        """
        Saves the model in it's current state to a file with the given name (treated
        as a relative path).
        :param checkpoint_filename: File name or relative path to save to.
        """
        torch.save(self.model, checkpoint_filename)
        print(f"\n*** Saved checkpoint {checkpoint_filename}")

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and updates weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """

        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        batch_loss: float
        num_correct: int

        y_pred = np.squeeze(self.model(X))
        # new_y, new_y_pred = filter_bp_bounds(y, y_pred, self.model_name)
        new_y, new_y_pred = y, y_pred
        loss = self.loss_fn(new_y_pred, new_y)
        self.optimizer.zero_grad()  # Zero gradients of all parameters
        loss.backward()  # Run backprop algorithms to calculate gradients
        self.optimizer.step()  # Use gradients to update model parameters
        batch_loss = loss.item()
        num_correct = int(torch.sum((new_y_pred - new_y) < 3))

        return BatchResult(batch_loss, num_correct, pred_labels=new_y_pred, target_labels=new_y)

    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        batch_loss: float
        num_correct: int

        with torch.no_grad():
            y_pred = np.squeeze(self.model(X))
            # new_y, new_y_pred = filter_bp_bounds(y, y_pred, self.model_name)
            new_y, new_y_pred = y, y_pred
            loss = self.loss_fn(new_y_pred, new_y)
            batch_loss = loss.item()
            num_correct = int(torch.sum((new_y_pred - new_y) < 3))

        return BatchResult(batch_loss, num_correct, pred_labels=new_y_pred, target_labels=new_y)

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
        plot_confusion=False,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        target_labels = []
        pred_labels = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_fn = tqdm.auto.tqdm
            pbar_file = sys.stdout
        else:
            pbar_fn = tqdm.tqdm
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with pbar_fn(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct
                if plot_confusion is True:
                    pred_labels.extend(batch_res.pred_labels)
                    target_labels.extend(batch_res.target_labels)

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        if not verbose:
            pbar_file.close()

        return EpochResult(losses=losses, accuracy=accuracy, pred_labels=pred_labels, target_labels=target_labels)


def main():
    torch.manual_seed(42)

    model = ResNet.create_resnet_model()
    print(model)

    loss_fn = torch.nn.L1Loss()

    # End-to-end optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001, eps=1e-6)

    # Decay learning rate each epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


if __name__ == "__main__":
    main()
