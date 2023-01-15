import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms


def create_densenet_model():
    model = models.densenet121(pretrained=False)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 1024, bias=True),
        nn.ReLU(),
        nn.Linear(1024, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, 1, bias=True)
    )

    return model