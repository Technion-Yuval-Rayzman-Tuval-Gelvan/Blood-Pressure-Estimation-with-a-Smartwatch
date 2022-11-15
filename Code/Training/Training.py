import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import tqdm
import time

from torch.autograd import Variable
from torchvision import datasets, models, transforms
from datetime import datetime
import sys, os

from Code.Training import ResNet, HDF5DataLoader

sys.path.append(os.path.abspath(os.path.join('..', 'LoadData')))
sys.path.append(os.path.abspath(os.path.join('..', 'ResNet')))
sys.path.append(os.path.abspath(os.path.join('..', 'HDF5DataLoader')))
# Now do your import

# Set some default values of the the matplotlib plots
plt.rcParams['figure.figsize'] = (8.0, 8.0)  # Set default plot's sizes
plt.rcParams['axes.grid'] = True  # Show grid by default in figures
print_every = 1
max_epochs_stop = 2
HDF5 = True


# Reference - 'https://towardsdatascience.com/end-to-end-pipeline-for-setting-up-multiclass-image-classification-for
# -data-scientists-2e051081d41c'
def train(model, learning_rate, n_epochs, train_loader, val_loader, model_name, save_file_name):

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_loss_path = f"{save_file_name}/{model_name}_valid_loss"
    if os.path.exists(valid_loss_path):
        print("Load loss:", valid_loss_path)
        valid_loss_min, best_epoch = load_data(valid_loss_path)

    # Prepare lists to store intermediate objectives
    train_objective_list = [np.inf]
    val_objective_list = [np.inf]

    lists_path = f"{save_file_name}/{model_name}_objective_lists"
    if os.path.exists(lists_path):
        print("Load lists:", lists_path)
        val_objective_list, train_objective_list = load_data(lists_path)

    # Initial Parameters
    overall_start = time.time()

    # criterion (PyTorch loss): objective to minimize
    # optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss()
    device = get_device()

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    # Run for n_epochs
    for epoch in range(n_epochs):

        # Set to training
        model.train()
        start = time.time()

        # keep track of training and validation loss each epoch
        train_loss = []
        valid_loss = []
        train_loss_items = 0
        valid_loss_items = 0

        # Training loop
        for i, (data, target) in enumerate(train_loader):

            if data is None or target is None:
                continue

            # Tensors to gpu
            if torch.cuda.is_available():
                data, target = Variable(data.cuda()), Variable(target.cuda())
            else:
                data, target = Variable(data), Variable(target)

            # Clear gradients
            optimizer.zero_grad()
            # Predicted output
            output = np.squeeze(model(data))

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            train_objective_list += [loss.item()]

            train_loss += [loss.item()]
            train_loss_items += 1

            # Track training progress
            print(f"\rEpoch: {epoch}\t{100 * (i + 1) / len(train_loader):.2f}% complete. {(time.time() - start):.2f} seconds elapsed in epoch.", end='')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in val_loader:

                    if data is None or target is None:
                        continue

                    # Tensors to gpu
                    if torch.cuda.is_available():
                        data, target = Variable(data.cuda()), Variable(target.cuda())
                    else:
                        data, target = Variable(data), Variable(target)

                    # Forward pass
                    output = np.squeeze(model(data))

                    # Validation loss
                    loss = criterion(output, target)

                    val_objective_list += [loss.item()]

                    valid_loss += [loss.item()]
                    valid_loss_items += 1

                # Calculate average losses
                average_train_loss = np.sum(train_loss) / len(train_loss)
                average_valid_loss = np.sum(valid_loss) / len(valid_loss)
                
                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(f'\nEpoch: {epoch} \tTraining Loss: {np.min(train_loss):.4f} \tValidation Loss: {np.min(valid_loss):.4f}')

                # Save the model if validation loss decreases
                if np.min(valid_loss) < valid_loss_min:
                    # Save model
                    print(f"Save better model. last valid loss: {valid_loss_min}. new valid loss: {np.min(valid_loss)}")
                    torch.save(model.state_dict(), f'{save_file_name}/model_min_loss_{np.min(valid_loss)}')
                    save_data((val_objective_list, train_objective_list), lists_path)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = np.min(valid_loss)
                    best_epoch = epoch
                    save_data((valid_loss_min, best_epoch), f'{save_file_name}/valid_min_loss_{np.min(valid_loss)}')

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f}')
                        total_time = time.time() - overall_start
                        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.')

                        return train_objective_list, val_objective_list

    # Record overall time and print out stats
    total_time = time.time() - overall_start
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f}')
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.')

    return train_objective_list, val_objective_list


def fine_tuning(model, train_loader, val_loader, model_name, save_file_name):
    n_epochs = 20
    etas_list = [5e-3, 1e-3, 1e-2, 1e-1]

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for i_eta, eta in enumerate(etas_list):
        model_copy = copy.deepcopy(model)
        print("Learning rate:", eta)
        train_objective_list, val_objective_list = train(model_copy, eta, n_epochs, train_loader,
                                                         val_loader, model_name, save_file_name)

        # Plot
        ax = axes.flat[i_eta]
        ax.plot(np.arange(len(train_objective_list)), train_objective_list, label='Train')
        ax.plot(np.arange(len(train_objective_list)), val_objective_list, label='Validation')
        ax.set_title(r'$\eta={' + f'{eta:g}' + r'}$')
        ax.set_xlabel('Step')
        ax.set_ylabel('Objective')
    axes[1, 1].legend()
    fig.tight_layout()

    # dd/mm/YY H:M:S
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
    plt.savefig(f'../../Results/fine_tuning_{now}_{model_name}.png')


def train_model(model, train_loader, val_loader, model_name, save_file_name):
    learning_rate = 0.005
    # n_epochs = 200
    n_epochs = 20

    train_objective_list, val_objective_list = train(model, learning_rate, n_epochs, train_loader, val_loader, model_name, save_file_name)

    plot_results(train_objective_list, val_objective_list, model_name, save_file_name=save_file_name)


def plot_results(train_objective_list, val_objective_list, model_name, save_file_name=None):
    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(np.arange(len(train_objective_list)), train_objective_list, label='Train')
    ax.plot(np.arange(len(val_objective_list)), val_objective_list, label='Validation')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Objective')

    # Project A
    # # dd/mm/YY H:M:S
    # now = datetime.now()
    # dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
    # plt.savefig(f'../../Results/{model_name}/training_{now}.png')

    plt.savefig(f'{save_file_name}/training_results.png')


def calculate_test_score(model, test_loader, model_name):
    total_mse = 0
    total_mae = 0
    num_samples = 0
    device = get_device()

    # Evaluate the score on the test set
    with torch.no_grad():
        errors = []
        # Test loop
        for x, y in test_loader:
            # Tensors to gpu
            if torch.cuda.is_available():
                x, y = Variable(x.cuda()), Variable(y.cuda())
            else:
                x, y = Variable(x), Variable(y)

            y_hat = np.squeeze(model(x))
            error = ((y_hat - y).abs()).sum().cpu()
            errors.append(error)

            total_mse += ((y_hat - y) ** 2).sum()
            total_mae += error
            num_samples += len(y_hat)

    mse_score = total_mse / num_samples
    mae_score = total_mae / num_samples
    std_score = np.std(errors)

    print(f'The MSE score of {model_name} is: {mse_score:.3}')
    print(f'The MAE score of {model_name} is: {mae_score:.3}')
    print(f'The STD score of {model_name} is: {std_score:.3}')


def save_data(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_data(path):
    with open(path, 'rb') as file:
        data_set = pickle.load(file)

    return data_set


def get_device():
    print("torch version:", torch.__version__)
    if torch.cuda.is_available():
        print("cuda version:", torch.version.cuda)
        device = torch.device('cuda:0')
        print("run on GPU.")
    else:
        device = torch.device('cpu')
        print("no cuda GPU available")
        print("run on CPU")

    return device


def main():
    """Paths"""
    # data_path = '/media/tuvalgelvan@staff.technion.ac.il/hd-21/Estimated-Blood-Presure-Project/Blood-Pressure' \
    #             '-Estimation-with-a-Smartwatch/Data'
    # model_path ='/media/tuvalgelvan@staff.technion.ac.il/hd-21/Estimated-Blood-Presure-Project/Blood-Pressure' \
    #             '-Estimation-with-a-Smartwatch/Models'
    # data_path = '../../Test_Data'
    data_path = '../../Data'
    model_path = '../../Models'

    date = datetime.now()

    """Create Model"""
    # model = ResNet.create_resnet_model()
    # save_model(f'{model_path}/dias_model', model)
    # save_model(f'{model_path}/sys_model', model)

    """Get Device"""
    device = get_device()
    # model.to(device)

    """Check Model"""
    # n_epochs = 200
    # learning_rate = 0.01
    # model_name = 'dias_model'
    # save_file_name = f'../../Models/{model_name}_batch_{LoadData.batch_size}_samples_{LoadData.total_samples}.pt'
    # train_loader = LoadData.get_dataset(data_path, 'dias_model', 'Train')
    # val_loader = LoadData.get_dataset(data_path, 'dias_model', 'Validation')
    # print("Check Dias Model")
    # dias_model, _, _ = train_model(model, learning_rate, train_loader, val_loader, n_epochs, 'dias_model', save_file_name)
    #
    # model = ResNet.create_resnet_model().to(device)
    # model_name = 'sys_model'
    # save_file_name = f'../../Models/{model_name}_batch_{LoadData.batch_size}_samples_{LoadData.total_samples}.pt'
    # train_loader = LoadData.get_dataset(data_path, 'sys_model', 'Train')
    # val_loader = LoadData.get_dataset(data_path, 'sys_model', 'Validation')
    # print("Check Sys Model")
    # sys_model, _, _= train_model(model, learning_rate, train_loader, val_loader, n_epochs, 'sys_model', save_file_name)

    # print("****** Fine Tuning Dias Model ******")
    # model = ResNet.create_resnet_model().to(device)
    # model_name = 'dias_model'
    # save_file_name = f'../../Models/HDF5_Models/fine_tuning_{date}_{model_name}.pt'
    # if os.path.exists(save_file_name):
    #     # Load the best state dict
    #     model.load_state_dict(torch.load(save_file_name))
    #
    # train_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train', 6)
    # val_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Validation', 2)
    # fine_tuning(model, train_loader, val_loader, model_name, save_file_name)

    # model = ResNet.create_resnet_model().to(device)
    # model_name = 'dias_model'
    # save_file_name = f'../../Models/{model_name}_batch_{LoadData.batch_size}_samples_{LoadData.total_samples}.pt'
    # train_loader = LoadData.get_dataset(data_path, 'dias_model', 'Train')
    # val_loader = LoadData.get_dataset(data_path, 'dias_model', 'Validation')
    # fine_tuning(model, train_loader, val_loader, 'dias_model', save_file_name)
    #
    # model = ResNet.create_resnet_model().to(device)
    # model_name = 'sys_model'
    # save_file_name = f'../../Models/{model_name}_batch_{LoadData.batch_size}_samples_{LoadData.total_samples}.pt'
    # train_loader = LoadData.get_dataset(data_path, 'sys_model', 'Train')
    # val_loader = LoadData.get_dataset(data_path, 'sys_model', 'Validation')
    # fine_tuning(model, train_loader, val_loader, 'sys_model', save_file_name)

    print("****** Train Dias Model ******")
    model = ResNet.create_resnet_model().to(device)
    model_name = 'dias_model'
    dias_save_file_name = f'../../Models/HDF5_Models/{date}_{model_name}.pt'
    # dias_save_file_name = f'../../Models/HDF5_Models/2021-12-21_dias_model.pt'
    if os.path.exists(dias_save_file_name):
        # Load the best state dict
        print("Load model:", dias_save_file_name)
        model.load_state_dict(torch.load(dias_save_file_name))

    train_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train')
    val_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Validation')
    train_model(model, train_loader, val_loader, model_name, dias_save_file_name)

    # print("****** Train Sys Model ******")
    # model = ResNet.create_resnet_model().to(device)
    model_name = 'sys_model'
    sys_save_file_name = f'../../Models/HDF5_Models/2021-12-21_sys_model.pt'
    # if os.path.exists(sys_save_file_name):
    #     # Load the best state dict
    #     model.load_state_dict(torch.load(sys_save_file_name))
    #
    # train_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train')
    # val_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Validation')
    # train_model(model, train_loader, val_loader, model_name, sys_save_file_name)

    # print("****** Check Test Score *******")
    # """Load Dias Model"""
    # model = ResNet.create_resnet_model().to(device)
    # model_name = 'dias_model'
    # if os.path.exists(dias_save_file_name):
    #     # Load the best state dict
    #     print(f"Load Model: {dias_save_file_name}")
    #     model.load_state_dict(torch.load(dias_save_file_name))
    #
    # test_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Test')
    # calculate_test_score(model, test_loader, model_name)
    #
    # """Load Sys Model"""
    # model = ResNet.create_resnet_model().to(device)
    # model_name = 'sys_model'
    # if os.path.exists(sys_save_file_name):
    #     # Load the best state dict
    #     print(f"Load Model: {sys_save_file_name}")
    #     model.load_state_dict(torch.load(sys_save_file_name))
    #
    # test_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Test')
    # calculate_test_score(model, test_loader, model_name)

    print(""" Print Results""")
    model_name = 'dias_model'
    lists_path = f"../../Variables/{model_name}_objective_lists"
    if os.path.exists(lists_path):
        print("Load lists:", lists_path)
        val_objective_list, train_objective_list = load_data(lists_path)
    plot_results(train_objective_list[:100], val_objective_list[:100], model_name)

    model_name = 'sys_model'
    lists_path = f"../../Variables/{model_name}_objective_lists"
    if os.path.exists(lists_path):
        print("Load lists:", lists_path)
        val_objective_list, train_objective_list = load_data(lists_path)
    plot_results(train_objective_list[:100], val_objective_list[:100], model_name)


if __name__ == "__main__":
    main()
