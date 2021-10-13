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
                 dias_bp_list_test, sys_bp_list_train, sys_bp_list_val, sys_bp_list_test):
        self.images_train = images_train
        self.images_val = images_val
        self.images_test = images_test
        self.dias_bp_train = dias_bp_list_train
        self.dias_bp_val = dias_bp_list_val
        self.dias_bp_test = dias_bp_list_test
        self.sys_bp_train = sys_bp_list_train
        self.sys_bp_val = sys_bp_list_val
        self.sys_bp_test = sys_bp_list_test


class Patient:

    def __init__(self, patient_name):
        self.patient_name = patient_name
        self.images = []
        self.dias_bp = []
        self.sys_bp = []

    def add_data(self, image, dias_bp, sys_bp):
        self.images.append(image)
        self.dias_bp.append(dias_bp)
        self.sys_bp.append(sys_bp)

    def get_data(self):
        return self.images, self.dias_bp, self.sys_bp


def load_data(data_path):
    # traverse root directory, and list directories as dirs and files as files
    print("loading images...")
    first_time = True
    patients = []
    for root, dirs, images in tqdm.tqdm(os.walk(data_path)):
        path = root.split(os.sep)
        if len(images) != 0:
            patient_name = images[0].split('_')[0]
            new_patient = Patient(patient_name)
            print(patient_name)
            for image_name in tqdm.tqdm(images):

                image = cv2.imread(posixpath.join(root, image_name))
                image = cv2.resize(image, (110, 110))

                # name example: 3000063_124.32324334151441_60.60244138052165
                name = image_name.split('_')
                systolic_bp = np.float(name[1])
                diastolic_bp = np.float(name[2][:-4])

                new_patient.add_data(image, diastolic_bp, systolic_bp)

    patients.append(new_patient)
    patients = np.array(patients)
    print("loading images done")

    return patients


def split_data(patients_data):

    n_patients = len(patients_data)

    # Generate a random generator with a fixed seed
    rand_gen = np.random.RandomState(0)

    # Generating a shuffled vector of indices
    indices = np.arange(n_patients)
    rand_gen.shuffle(indices)

    # Split the indices into 60% train / 20% validation / 20% test
    n_patients_train = int(n_patients * 0.6)
    n_patients_val = n_patients_train + int(n_patients * 0.2)
    train_indices = indices[:n_patients_train]
    val_indices = indices[n_patients_train:n_patients_val]
    test_indices = indices[n_patients_val:]

    train_patients = patients_data[train_indices]
    val_patients = patients_data[val_indices]
    test_patients = patients_data[test_indices]
    print(train_indices, val_indices, test_indices)



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

    data = Data(images_train, images_val, images_test, dias_bp_list_train, dias_bp_list_val,
                dias_bp_list_test, sys_bp_list_train, sys_bp_list_val, sys_bp_list_test)
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


def get_data(data_path):
    patients_data = load_data(data_path)
    data = split_data(patients_data)

    return data


def main():
    data_path = '../../Test_Data'
    patients_data = load_data(data_path)
    # data = split_data(patients_data)
    # print_some_images(data.images_train)


if __name__ == "__main__":
    main()
