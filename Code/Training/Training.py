import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import tqdm
from torchvision import datasets, models, transforms
from datetime import datetime
from pytictoc import TicToc

# Set some default values of the the matplotlib plots
from Code.Training import LoadData
from Code.Training import ResNet

plt.rcParams['figure.figsize'] = (8.0, 8.0)  # Set default plot's sizes
plt.rcParams['axes.grid'] = True  # Show grid by default in figures

def train(model, learning_rate, n_epochs, train_loader, val_loader):

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    # Prepare lists to store intermediate obejectives
    train_objective_list = [np.inf]
    val_objective_list = [np.inf]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss()
    device = get_device()

    # Run for n_epochs
    for epoch in tqdm.tqdm(range(n_epochs)):

        # Set to training
        model.train()

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        # Training loop
        for i, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            data, target = data.to(device), target.to(device)

            # Clear gradients
            optimizer.zero_grad()
            # Predicted output
            output = np.squeeze(model(data))

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            train_loss += loss.item()

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in val_loader:
                    # Tensors to gpu
                    data, target = data.to(device), target.to(device)

                    # Forward pass
                    output = np.squeeze(model(data))

                    # Validation loss
                    loss = criterion(output, target)

                    valid_loss += loss.item()

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(val_loader.dataset)

                # Save validation loss
                val_objective_list.append(valid_loss)
                train_objective_list.append(train_loss)

                print(train_loss, valid_loss)

    return train_objective_list, val_objective_list


def fine_tuning(model, train_loader, val_loader, model_name):
    n_epochs = 200
    etas_list = [3e-3, 1e-3, 3e-2, 1e-2]

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for i_eta, eta in enumerate(etas_list):
        model_copy = copy.deepcopy(model)
        train_objective_list, val_objective_list = train(model_copy, eta, n_epochs, train_loader,
                                                         val_loader)

        # Plot
        ax = axes.flat[i_eta]
        ax.plot(np.arange(len(train_objective_list)), train_objective_list, label='Train')
        ax.plot(np.arange(len(val_objective_list)), val_objective_list, label='Validation')
        ax.set_title(r'$\eta={' + f'{eta:g}' + r'}$')
        ax.set_xlabel('Step')
        ax.set_ylabel('Objective')
    axes[1, 1].legend()
    fig.tight_layout()

    # dd/mm/YY H:M:S
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
    plt.savefig(f'../../Results/fine_tuning_{now}_{model_name}.png')


def train_model(model, learning_rate, train_loader, val_loader, n_epochs, model_name):
    train_objective_list, val_objective_list = train(model, learning_rate, n_epochs, train_loader, val_loader)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(np.arange(len(train_objective_list)), train_objective_list, label='Train')
    ax.plot(np.arange(len(val_objective_list)), val_objective_list, label='Validation')
    ax.set_title(r'$learning_rate={' + f'{learning_rate:g}' + r'}$')
    ax.set_xlabel('Step')
    ax.set_ylabel('Objective')

    # dd/mm/YY H:M:S
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
    plt.savefig(f'../../Results/{model_name}/training_{now}.png')

    return model, train_objective_list, val_objective_list


def calculate_test_score(model, test_images, test_labels, model_name):
    test_loader, _ = prepare_data(test_images, test_labels)
    total_mse = 0
    total_mae = 0
    num_samples = 0

    # Evaluate the score on the test set
    with torch.no_grad():
        errors = []
        for x, y in test_loader:
            x = x.cuda()
            y = y.cuda()

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


def load_data_set(path):
    with open(path, 'rb') as file:
        data_set = pickle.load(file)

    return data_set


def save_model(model_name, trained_model):
    torch.save(trained_model, f'{model_name}.pth')


def train_and_save_model(model, train_loader, val_loader, model_name, model_path):
    best_learning_rate = 0.001
    n_epochs = 200

    test_model, train_objective_list, val_objective_list = train_model(model, best_learning_rate,
                                                                                 train_loader, val_loader, n_epochs, model_name)
    optimal_number_of_steps = np.argmin(val_objective_list)
    n_epochs = optimal_number_of_steps
    trained_model, train_objective_list, val_objective_list = train_model(model_copy, best_learning_rate,
                                                                                 train_loader, val_loader, n_epochs, model_name)
    save_model(f'{model_path}/{model_name}', trained_model)


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
    data_path = '../../Test_Data'
    # data_path = '../../Data'
    save_path = '../../Dataset'
    model_path = '../../Models'

    """Name"""
    # data_set_name = 'test_data_set'
    data_set_name = 'data_set'

    """Create Model"""
    model = ResNet.create_resnet_model()
    # save_model(f'{model_path}/dias_model', model)
    # save_model(f'{model_path}/sys_model', model)

    """Get Device"""
    device = get_device()
    model.to(device)

    """Get Data"""
    train_loader = LoadData.get_dataset(data_path, 'dias_model', 'Train')
    val_loader = LoadData.get_dataset(data_path, 'dias_model', 'Validation')

    """Check Model"""
    # n_epochs = 200
    # learning_rate = 0.01
    # print("Check Dias Model")
    # dias_model, _, _ = train_model(model, learning_rate, train_loader, val_loader, n_epochs, 'dias_model')
    #
    # model = ResNet.create_resnet_model().to(device)
    # train_loader = LoadData.get_dataset(data_path, 'sys_model', 'Train')
    # val_loader = LoadData.get_dataset(data_path, 'sys_model', 'Validation')
    # print("Check Sys Model")
    # sys_model, _, _= train_model(model, learning_rate, train_loader, val_loader, n_epochs, 'sys_model')

    """Fine Tuning"""
    model = ResNet.create_resnet_model().to(device)
    fine_tuning(model, train_loader, val_loader, 'dias_model')

    model = ResNet.create_resnet_model().to(device)
    train_loader = LoadData.get_dataset(data_path, 'sys_model', 'Train')
    val_loader = LoadData.get_dataset(data_path, 'sys_model', 'Validation')
    fine_tuning(model, train_loader, val_loader, 'sys_model')

    # print("****** Train Dias Model ******")
    # model = torch.load('dias_model.pth')
    # model.to(device)
    # train_and_save_model(model, train_loader, val_loader, 'dias_model', model_path)
    # 
    # print("****** Train Sys Model ******")
    # model = torch.load('sys_model.pth')
    # model.to(device)
    # train_and_save_model(model, train_loader, val_loader, 'sys_model', model_path)

    # """Load Model"""
    # dias_model = torch.load('dias_model.pth')
    # dias_model.to(device)
    # dias_model.eval()
    # calculate_test_score(dias_model, data.images_test, data.dias_bp_test, 'dias_model')
    #
    # sys_model = torch.load('sys_model.pth')
    # sys_model.to(device)
    # sys_model.eval()
    # calculate_test_score(sys_model, data.images_test, data.sys_bp_test, 'sys_model')


if __name__ == "__main__":
    main()