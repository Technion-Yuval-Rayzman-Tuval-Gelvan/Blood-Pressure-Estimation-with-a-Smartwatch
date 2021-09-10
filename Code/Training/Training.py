import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import tqdm
from torchvision import datasets, models, transforms

# Set some default values of the the matplotlib plots
from Code.Training import LoadData
from Code.Training import ResNet

plt.rcParams['figure.figsize'] = (8.0, 8.0)  # Set default plot's sizes
plt.rcParams['axes.grid'] = True  # Show grid by default in figures

learning_rate = 0.01


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

            objective = torch.nn.L1Loss()

            # start training
            optimizer.zero_grad()
            # forward
            py_hat = np.squeeze(model(x), axis=0)
            loss = torch.nn.functional.l1_loss(py_hat, y)
            # backward
            loss.backward()
            optimizer.step()

        # Evaluate the objective on the validation set
        with torch.no_grad():  # This tell PyTorch not to calculate the gradients to save time
            train_objective_list.append(loss.item())
            # forward
            py_hat = np.squeeze(model(x), axis=0)
            # loss
            loss = torch.nn.functional.l1_loss(py_hat, y)
            val_objective_list.append(loss.item())

    return train_objective_list, val_objective_list


def fine_tuning(model, train_loader, x_val, y_val):
    n_epochs = 200
    etas_list = [1e-3, 3e-2, 1e-2, 3e-3]

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for i_eta, eta in enumerate(etas_list):
        train_objective_list, val_objective_list = train(model, eta, n_epochs, train_loader,
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


def train_resnet_module(model, learning_rate, train_loader, x_val, y_val):
    n_epochs = 40

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

    return model


def prepare_data(x, y):
    batch_size = 1
    data_set = torch.utils.data.TensorDataset(torch.tensor(x).float(), torch.tensor(y).float())
    loader = torch.utils.data.DataLoader(dataset=data_set, shuffle=True, batch_size=batch_size)

    return loader


def train_model(data, model):
    train_loader = prepare_data(data.images_train, data.dias_bp_train)
    dias_model = train_resnet_module(model, learning_rate, train_loader, data.images_val, data.dias_bp_val)

    return dias_model, train_loader


def main():
    data_path = '../../Data'
    data = LoadData.get_data(data_path)
    model = ResNet.create_resnet_model(data.images_train)

    dias_model, train_loader = train_model(data, model)
    fine_tuning(dias_model, train_loader, data.images_val, data.dias_bp_val)


if __name__ == "__main__":
    main()
