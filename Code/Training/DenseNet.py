import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms


def create_densenet_model():
    model = models.densenet121(pretrained=False, num_classes=1)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
    return model