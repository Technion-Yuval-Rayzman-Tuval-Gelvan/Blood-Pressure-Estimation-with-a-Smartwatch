import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms


def create_resnet_model():
    model = models.resnet18(pretrained=False, num_classes=1)
    model.conv1.in_channels = 1

    return model
