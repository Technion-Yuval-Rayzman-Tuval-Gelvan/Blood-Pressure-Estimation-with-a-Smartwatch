import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

## Set some default values of the the matplotlib plots
plt.rcParams['figure.figsize'] = (8.0, 8.0)  # Set default plot's sizes
plt.rcParams['axes.grid'] = True  # Show grid by default in figures

'''
Example for CNN module:

class CNN(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(CNN, self).__init__()

        ## Defining the convolutional and fully connected layers with their parameters
        ## ===========================================================================
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.fc3 = torch.nn.Linear(in_features=256 * 2 * 2, out_features=256, bias=True)
        self.fc4 = torch.nn.Linear(in_features=256, out_features=10, bias=True)

    def forward(self, x):
        x = x[:, None, :, :]

        z = self.conv1(x)
        z = torch.nn.functional.relu(z)
        z = self.conv2(z)
        z = torch.nn.functional.relu(z)
        z = z.view(z.shape[0], -1)
        z = self.fc3(z)
        z = torch.nn.functional.relu(z)
        z = self.fc4(z)
        y = torch.nn.functional.log_softmax(z, dim=1)

        return y
'''


def train(model, learning_rate, n_epochs, train_loader, x_val, y_val, optimizer, criterion):
    ## Move validation set to the GPU
    x_val = x_val.cuda()
    y_val = y_val.cuda()

    ## Prepare lists to store intermediate obejectives
    train_objective_list = [np.inf]
    val_objective_list = [np.inf]

    ## Run for n_epochs
    for epoch in tqdm.tqdm(range(n_epochs)):
        ## Run over the batches
        for x, y in train_loader:  # forward
            outputs = torch.squeeze(model(x))
            # loss
            loss = criterion(outputs, y)
            ## Move batch to GPU
            x = x.cuda()
            y = y.cuda()

            # start training
            optimizer.zero_grad()
            # forward
            outputs = torch.squeeze(model(x))
            # loss
            loss = criterion(outputs, y)
            # backward
            loss.backward()
            optimizer.step()

        ## Evaluate the objective on the validation set
        with torch.no_grad():  ## This tell PyTorch not to calculate the gradients to save time
            train_objective_list.append(loss.item())
            # forward
            outputs = torch.squeeze(model(x))
            # loss
            loss = criterion(outputs, y)
            val_objective_list.append(loss.item())

    return train_objective_list, val_objective_list


def fine_tuning(model):
    n_epochs = 20
    etas_list = (1e-1, 3e-2, 1e-2, 3e-3)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for i_eta, eta in enumerate(etas_list):
        train_objective_list, val_objective_list = train(model, eta, n_epochs, train_loader,
                                                         torch.tensor(x_val).float(),
                                                         torch.tensor(y_val).long())

        ## Plot
        ax = axes.flat[i_eta]
        ax.plot(np.arange(len(train_objective_list)), train_objective_list, label='Train')
        ax.plot(np.arange(len(val_objective_list)), val_objective_list, label='Validation')
        ax.set_title(r'$\eta={' + f'{eta:g}' + r'}$')
        ax.set_xlabel('Step')
        ax.set_ylabel('Objective')
    axes[1, 1].legend()
    fig.tight_layout()


def train_resnet_module():
    n_epochs = 10
    model = models.resnet18(pretrained=True)
    model = model.cuda()

    ## set optimizer and criterion
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    train_objective_list, val_objective_list = train(model, learning_rate, n_epochs, train_loader,
                                                     torch.tensor(x_val).float(),
                                                     torch.tensor(y_val).long())

    ## Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(np.arange(len(train_objective_list)), train_objective_list, label='Train')
    ax.plot(np.arange(len(val_objective_list)), val_objective_list, label='Validation')
    ax.set_title(r'$\eta={' + f'{eta:g}' + r'}$')
    ax.set_xlabel('Step')
    ax.set_ylabel('Objective')
