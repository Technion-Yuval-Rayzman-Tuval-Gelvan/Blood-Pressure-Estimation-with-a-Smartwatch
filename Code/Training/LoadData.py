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
from multiprocessing import Pool, Lock

lock = Lock()

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
        self.names = []

    def add_data(self, image, dias_bp, sys_bp, name):
        self.images.append(image)
        self.dias_bp.append(dias_bp)
        self.sys_bp.append(sys_bp)
        self.names.append(name)

    def get_data(self):
        return self.images, self.dias_bp, self.sys_bp


def get_dir_list(data_path):
    return os.listdir(data_path)


def read_patient_data(image_name):
    patient_name = image_name.split('_')[0]
    # root = '../../Test_Data/'
    root = '../../Data/'
    image = cv2.imread(posixpath.join(root, patient_name, image_name))
    image = cv2.resize(image, (110, 110))

    # name example: 3000063_124.32324334151441_60.60244138052165
    name = image_name.split('_')
    systolic_bp = np.float(name[1])
    diastolic_bp = np.float(name[2][:-4])

    data = (image, diastolic_bp, systolic_bp, image_name)

    return data


def load_data(data_path, list_dir):
    # traverse root directory, and list directories as dirs and files as files
    print("loading images...")
    first_time = True
    patients = []
    for patient in list_dir:
        images = os.listdir(f"{data_path}/{patient}")
        if len(images) != 0:
            patient_name = images[0].split('_')[0]
            new_patient = Patient(patient_name)
            print(f"loading patient {patient_name} data.")
            pool = Pool()
            for image, diastolic_bp, systolic_bp, name in tqdm.tqdm(pool.imap(func=read_patient_data, iterable=images), total=len(images)):
                new_patient.add_data(image, diastolic_bp, systolic_bp, name)
                patients.append(new_patient)

            pool.close()
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

    images_train = []
    dias_bp_list_train = []
    sys_bp_list_train = []
    images_val = []
    dias_bp_list_val = []
    sys_bp_list_val = []
    images_test = []
    dias_bp_list_test = []
    sys_bp_list_test = []

    for patient in train_patients:
        images, dias_bp, sys_bp = patient.get_data()
        images_train.append(images)
        dias_bp_list_train.append(dias_bp)
        sys_bp_list_train.append(sys_bp)

    for patient in val_patients:
        images, dias_bp, sys_bp = patient.get_data()
        images_val.append(images)
        dias_bp_list_val.append(dias_bp)
        sys_bp_list_val.append(sys_bp)

    for patient in test_patients:
        images, dias_bp, sys_bp = patient.get_data()
        images_test.append(images)
        dias_bp_list_test.append(dias_bp)
        sys_bp_list_test.append(sys_bp)


    # Extract the sub datasets from the full dataset using the calculated indices
    images_train = np.concatenate(images_train)
    dias_bp_list_train = np.concatenate(dias_bp_list_train)
    sys_bp_list_train = np.concatenate(sys_bp_list_train)
    images_val = np.concatenate(images_val)
    dias_bp_list_val = np.concatenate(dias_bp_list_val)
    sys_bp_list_val = np.concatenate(sys_bp_list_val)
    images_test = np.concatenate(images_test)
    dias_bp_list_test = np.concatenate(dias_bp_list_test)
    sys_bp_list_test = np.concatenate(sys_bp_list_test)

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


def get_data(data_path, list_dir):
    patients_data = load_data(data_path, list_dir)
    data = split_data(patients_data)

    return data


def get_data_chunks(data_path):
    dir_list = get_dir_list(data_path)

    n_patients = len(dir_list)

    # Generate a random generator with a fixed seed
    rand_gen = np.random.RandomState(0)

    # Generating a shuffled vector of indices
    indices = np.arange(n_patients)
    rand_gen.shuffle(indices)

    n_patients_per_chunk = int(n_patients / 10)
    start = 0
    end = n_patients_per_chunk
    chunks_list = []
    while end < n_patients:
        chunks_list.append(dir_list[start:end])
        start = end + 1
        end += n_patients_per_chunk

    chunks_list.append(dir_list[start:])
    print(f"Total Patients: {n_patients}")

    return chunks_list


def main():
    # data_path = '../../Test_Data'
    data_path = '../../Data'
    chunks_list = get_data_chunks(data_path)
    print (chunks_list)
    # dir_list = get_dir_list(data_path)
    # patients_data = load_data(data_path, dir_list)
    # data = split_data(patients_data)
    # print_some_images(data.images_train)


if __name__ == "__main__":
    main()
