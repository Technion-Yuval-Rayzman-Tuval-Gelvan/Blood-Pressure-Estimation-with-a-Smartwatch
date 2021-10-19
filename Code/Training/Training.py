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

# Set some default values of the the matplotlib plots
from Code.Training import LoadData
from Code.Training import ResNet

plt.rcParams['figure.figsize'] = (8.0, 8.0)  # Set default plot's sizes
plt.rcParams['axes.grid'] = True  # Show grid by default in figures


def train(model, learning_rate, n_epochs, train_loader, x_val, y_val):
    # Move validation set to the GPU
    x_val = x_val.cuda()
    y_val = y_val.cuda()

    # Prepare lists to store intermediate obejectives
    train_objective_list = [np.inf]
    val_objective_list = [np.inf]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Run for n_epochs
    for epoch in tqdm.tqdm(range(n_epochs)):
        # Run over the batches
        for x, y in train_loader:  # forward

            # Move batch to GPU
            x = x.cuda()
            y = y.cuda()

            # start training
            optimizer.zero_grad()
            # forward
            py_hat = np.squeeze(model(x))
            loss = torch.nn.functional.l1_loss(py_hat, y)
            # backward
            loss.backward()
            optimizer.step()

        # Evaluate the objective on the validation set
        with torch.no_grad():  # This tell PyTorch not to calculate the gradients to save time
            train_objective_list.append(loss.item())
            # forward
            py_hat = np.squeeze(model(x_val))
            # loss
            loss = torch.nn.functional.l1_loss(py_hat, y_val)
            val_objective_list.append(loss.item())

    return train_objective_list, val_objective_list


def fine_tuning(model, train_loader, x_val, y_val, model_name):
    n_epochs = 200
    etas_list = [3e-3, 1e-3, 3e-2, 1e-2]

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for i_eta, eta in enumerate(etas_list):
        model_copy = copy.deepcopy(model)
        train_objective_list, val_objective_list = train(model_copy, eta, n_epochs, train_loader,
                                                         torch.tensor(x_val).float(),
                                                         torch.tensor(y_val).float())

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


def train_resnet_model(model, learning_rate, train_loader, x_val, y_val, n_epochs):
    train_objective_list, val_objective_list = train(model, learning_rate, n_epochs, train_loader,
                                                     torch.tensor(x_val).float(),
                                                     torch.tensor(y_val).float())

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
    plt.savefig(f'../../Results/training_{now}.png')

    return model, train_objective_list, val_objective_list


def prepare_data(x, y):
    batch_size = 64
    data_set = torch.utils.data.TensorDataset(torch.tensor(x).float(), torch.tensor(y).float())
    loader = torch.utils.data.DataLoader(dataset=data_set, shuffle=True, batch_size=batch_size, num_workers=2,
                                         drop_last=True, pin_memory=False)

    return loader, data_set


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
    # torch.save(trained_model, f'../Models/{model_name}')
    if model_name == 'dias_model':
        torch.save(trained_model, 'dias_model')
    else:
        torch.save(trained_model, 'sys_model')


def train_and_save_model(model, data, train_loader, model_name):
    best_learning_rate = 0.001
    n_epochs = 200

    if model_name == 'dias_model':
        y_val = data.dias_bp_val
    else:
        y_val = data.sys_bp_val

    trained_model, train_objective_list, val_objective_list = train_resnet_model(model, best_learning_rate,
                                                                                 train_loader, data.images_val,
                                                                                 y_val, n_epochs)
    optimal_number_of_steps = np.argmin(val_objective_list)
    n_epochs = optimal_number_of_steps
    trained_model, train_objective_list, val_objective_list = train_resnet_model(model, best_learning_rate,
                                                                                 train_loader, data.images_val,
                                                                                 y_val, n_epochs)
    save_model(model_name, trained_model)


def main():
    """Paths"""
    # data_path = '../../Test_Data'
    data_path = '../../Data'
    save_path = '../../Dataset'
    # data_set_name = 'test_data_set'
    data_set_name = 'data_set'
    chunks_list = LoadData.get_data_chunks(data_path)
    torch.save(chunks_list, 'chunks_list')

    model = ResNet.create_resnet_model()

    save_model('dias_model', model)
    save_model('sys_model', model)

    for i, chunk in enumerate(chunks_list):
        print(f"****** Start Train Chunk {i} ******")

        """Save Data"""
        print("****** Get Chunk Data ******")
        data = LoadData.get_data(data_path, chunk)
        path = f'{save_path}/{data_set_name}_chunk_{i}'
        save_data(data, path)

    # chunks_list = torch.load('chunks_list')
    # for i, chunk in enumerate(chunks_list):
    #     """Load Data"""
    #     # data = load_data_set(path)
    #     dias_train_loader, _ = prepare_data(data.images_train, data.dias_bp_train)
    #     sys_train_loader, _ = prepare_data(data.images_train, data.sys_bp_train)
    #
    #     """check train model"""
    #     # model = ResNet.create_resnet_model()
    #     # n_epochs = 20
    #     # learning_rate = 0.01
    #     # dias_model, _, _ = train_resnet_model(model, learning_rate, train_loader, data.images_val, data.dias_bp_val, n_epochs)
    #     # sys_model, _, _= train_resnet_model(model, learning_rate, train_loader, data.images_val, data.sys_bp_val, n_epochs)
    #
    #     # if i == 0:
    #     #     print("****** Fine Tuning ******")
    #     #     """look for best learning rate"""
    #     #     model = ResNet.create_resnet_model()
    #     #     fine_tuning(model, dias_train_loader, data.images_val, data.dias_bp_val, 'dias_model')
    #     #     fine_tuning(model, sys_train_loader, data.images_val, data.sys_bp_val, 'sys_model')
    #
    #     print("****** Train Dias Model ******")
    #     dias_model = torch.load('dias_model')
    #     train_and_save_model(dias_model, data, dias_train_loader, 'dias_model')
    #
    #     print("****** Train Sys Model ******")
    #     sys_model = torch.load('sys_model')
    #     train_and_save_model(sys_model, data, sys_train_loader, 'sys_model')

    # model = ResNet.create_resnet_model()
    # model_name = dias_model

    # """Load Model"""
    # dias_model = torch.load('dias_model')
    # dias_model.eval()
    # calculate_test_score(dias_model, data.images_test, data.dias_bp_test, 'dias_model')
    #
    # sys_model = torch.load('sys_model')
    # sys_model.eval()
    # calculate_test_score(sys_model, data.images_test, data.sys_bp_test, 'sys_model')


if __name__ == "__main__":
    main()
