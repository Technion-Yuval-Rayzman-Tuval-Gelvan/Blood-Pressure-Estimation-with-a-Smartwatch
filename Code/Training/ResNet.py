import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms


def create_resnet_model():
    model = models.resnet18(pretrained=False)
    out_channels = model.conv1.out_channels
    kernel_size = model.conv1.kernel_size
    padding = model.conv1.padding
    model.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)

    return model
