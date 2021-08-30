import os
import posixpath

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import tqdm
from torch.autograd import Variable
from torchvision import datasets, models, transforms


class Data:

    def __init__(self, images_train, images_val, images_test, dias_bp_list_train, dias_bp_list_val,
               dias_bp_list_test,sys_bp_list_train, sys_bp_list_val, sys_bp_list_test):
        self.images_train = images_train
        self.images_val = images_val
        self.images_test = images_test
        self.dias_bp_train = dias_bp_list_train
        self.dias_bp_val = dias_bp_list_val
        self.dias_bp_test = dias_bp_list_test
        self.sys_bp_train = sys_bp_list_train
        self.sys_bp_val = sys_bp_list_val
        self.sys_bp_test = sys_bp_list_test


def load_data(data_path):
    # traverse root directory, and list directories as dirs and files as files
    print("loading images...")
    # data = Data()
    images_list = []
    sys_bp_list = []
    dias_bp_list = []
    for root, dirs, images in tqdm.tqdm(os.walk(data_path)):
        path = root.split(os.sep)
        for image_name in images:
            image = cv2.imread(posixpath.join(root, image_name))
            image = cv2.resize(image, (110, 110))
            # normalized_image = (image - np.mean(image)) / np.std(image)
            # normalized_image = torch.tensor(normalized_image.T).view(1, 4, 640, 480)
            # normalized_image = normalized_image.to(device=device, dtype=torch.float)
            # normalized_image = Variable(normalized_image.cuda(device))
            name = image_name.split('_')
            # name example: 3000063_0025_1_124.32324334151441_60.60244138052165
            systolic_bp = name[3]
            diastolic_bp = name[4]
            # data.add_image(image, image_name, systolic_bp, diastolic_bp)
            images_list.append(image)
            sys_bp_list.append(systolic_bp)
            dias_bp_list.append(diastolic_bp)

    print("loading images done")
    images_list = np.asarray(images_list)
    sys_bp_list = np.asarray(sys_bp_list)
    dias_bp_list = np.asarray(dias_bp_list)
    return images_list, sys_bp_list, dias_bp_list


def split_data(images_list, sys_bp_list, dias_bp_list):


    n_samples = images_list.shape[0]  # The total number of samples in the dataset

    # Generate a random generator with a fixed seed
    rand_gen = np.random.RandomState(0)

    # Generating a shuffled vector of indices
    indices = np.arange(n_samples)
    rand_gen.shuffle(indices)

    # Split the indices into 60% train / 20% validation / 20% test
    n_samples_train = int(n_samples * 0.6)
    n_samples_val = n_samples_train + int(n_samples * 0.2)
    train_indices = indices[:n_samples_train]
    val_indices = indices[n_samples_train:n_samples_val]
    test_indices = indices[n_samples_val:]

    # Extract the sub datasets from the full dataset using the calculated indices

    images_train = images_list[train_indices]
    dias_bp_list_train = dias_bp_list[train_indices]
    sys_bp_list_train = sys_bp_list[train_indices]
    images_val = images_list[val_indices]
    dias_bp_list_val = dias_bp_list[val_indices]
    sys_bp_list_val = sys_bp_list[val_indices]
    images_test = images_list[test_indices]
    dias_bp_list_test = dias_bp_list[test_indices]
    sys_bp_list_test = sys_bp_list[test_indices]

    data = Data(images_train, images_val, images_test, dias_bp_list_train,dias_bp_list_val,
               dias_bp_list_test,sys_bp_list_train, sys_bp_list_val, sys_bp_list_test)
    return data


def print_some_images(images):
    print('Number of images in the dataset: {}'.format(len(images)))
    print('Each images size is: {}'.format(images.shape[1:]))
    print('These are the first 4 images:')
    fig, ax_array = plt.subplots(4, 4)
    for i, ax in enumerate(ax_array.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_yticks([])
        ax.set_xticks([])
    plt.show()

def get_device():
    # load torch
    print("torch version:", torch.__version__)
    if torch.cuda.is_available():
        print("cuda version:", torch.version.cuda)
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        print("run on GPU.")
    else:
        device = torch.device('cpu')
        print("no cuda GPU available")
        print("run on CPU")
    return device


def main():
    data_path = '../../Data'
    # device = get_device()
    images_list, sys_bp_list, dias_bp_list = load_data(data_path)
    data = split_data(images_list, sys_bp_list, dias_bp_list)
    print_some_images(data.images_train)


if __name__ == "__main__":
    main()
