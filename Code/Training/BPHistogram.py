import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import tqdm
import os
import sys
from torchvision import datasets, models, transforms
from datetime import datetime
# Set some default values of the the matplotlib plots
from Code.Training import LoadData
sys.path.append(os.path.abspath(os.path.join('..', 'HDF5DataLoader')))
import HDF5DataLoader


def load_bp_data(data_path, list_dir):
    dias_bp_list = []
    sys_bp_list = []
    print("loading blood pressure data")
    total_patients = len(list_dir)
    for patient in tqdm.tqdm(list_dir):
        images = os.listdir(f"{data_path}/{patient}")
        if images:
            patient_name = images[0].split('_')[0]
            for image_name in images:
                # name example: 3000063_124.32324334151441_60.60244138052165
                name = image_name.split('_')
                systolic_bp = np.float(name[1])
                diastolic_bp = np.float(name[2][:-4])
                dias_bp_list.append(diastolic_bp)
                sys_bp_list.append(systolic_bp)

    return dias_bp_list, sys_bp_list


def load_bp_data_loader(data_loader):

    bp_list = []

    for data, label in data_loader:
        bp_list += label

    print("done. len:", len(bp_list))

    return bp_list


def calculate_histogram(bp_list, bp_name):
    n, bins, patches = plt.hist(x=bp_list, bins='auto', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Blood Pressure')
    plt.ylabel('# windows')
    plt.title(f'{bp_name} Blood Pressure Histogram')
    # dd/mm/YY H:M:S
    now = datetime.now()
    plt.savefig(f'../../Results/Histogram/Histogram_{now}_{bp_name}_Blood_Pressure.png')
    plt.show()


def main():
    # data_path = '../../Test_Data'
    data_path = '../../Data'
    # dir_list = LoadData.get_dir_list(data_path)
    # dias_bp_list, sys_bp_list = load_bp_data(data_path, dir_list)
    # calculate_histogram(dias_bp_list, "Diastolic")
    # calculate_histogram(sys_bp_list, "Systolic")
    train_loader = HDF5DataLoader.get_hdf5_dataset(data_path, 'dias_model', 'Train', max_chuncks=8)
    train_dias_bp_list = load_bp_data_loader(train_loader)
    calculate_histogram(train_dias_bp_list, "Train Diastolic")
    val_loader = HDF5DataLoader.get_hdf5_dataset(data_path, 'dias_model', 'Validation', max_chuncks=2)
    valid_dias_bp_list = load_bp_data_loader(val_loader)
    calculate_histogram(valid_dias_bp_list, "Valid Diastolic")

    train_loader = HDF5DataLoader.get_hdf5_dataset(data_path, 'sys_model', 'Train', max_chuncks=8)
    train_sys_bp_list = load_bp_data_loader(train_loader)
    calculate_histogram(train_sys_bp_list, "Train Systolic")
    val_loader = HDF5DataLoader.get_hdf5_dataset(data_path, 'sys_model', 'Validation', max_chuncks=2)
    valid_sys_bp_list = load_bp_data_loader(val_loader)
    calculate_histogram(valid_sys_bp_list, "Valid Systolic")


if __name__ == "__main__":
    main()
