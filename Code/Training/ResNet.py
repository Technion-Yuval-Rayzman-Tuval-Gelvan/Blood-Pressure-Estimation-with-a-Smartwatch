import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms


def create_resnet_model():
    model = models.resnet18(pretrained=False)
    model.conv1.in_channels = 1
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024, bias=True),
        nn.ReLU(),
        nn.Linear(1024, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, 1, bias=True)
    )
    return model
