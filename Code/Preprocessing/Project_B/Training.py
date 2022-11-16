import copy
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import tqdm
import time
from datetime import datetime
import sys, os
import Config as cfg
from Code.Training import ResNet, HDF5DataLoader
from Code.Training.Training import get_device, train_model, plot_results, load_data, fine_tuning, calculate_test_score


def main():
    assert cfg.WORK_MODE == cfg.Mode.nn_training

    """Paths"""
    data_path = cfg.DATASET_DIR
    model_path = cfg.NN_MODELS
    device = get_device()

    # print("****** Fine Tuning Dias Model ******")
    # model = ResNet.create_resnet_model().to(device)
    # model_name = 'dias_model'
    # save_file_name = f'{cfg.DIAS_BP_MODEL_DIR}/fine_tuning_{model_name}.pt'
    # if os.path.exists(save_file_name):
    #     # Load the best state dict
    #     model.load_state_dict(torch.load(save_file_name))
    #
    # train_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train', 6)
    # val_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Validation', 2)
    # fine_tuning(model, train_loader, val_loader, model_name, save_file_name)

    print("****** Train Dias Model ******")
    model = ResNet.create_resnet_model().to(device)
    model_name = 'dias_model'
    dias_save_file_name = cfg.DIAS_BP_MODEL_DIR
    dias_model_path = f"{dias_save_file_name}/resnet_model.pt"
    if os.path.exists(dias_model_path):
        # Load the best state dict
        print("Load model:", dias_model_path)
        model.load_state_dict(torch.load(model_path))

    train_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train')
    val_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Validation')
    train_model(model, train_loader, val_loader, model_name, dias_save_file_name)

    print("****** Train Sys Model ******")
    model = ResNet.create_resnet_model().to(device)
    model_name = 'sys_model'
    sys_save_file_name = cfg.SYS_BP_MODEL_DIR
    sys_model_path = f"{dias_save_file_name}/resnet_model.pt"
    if os.path.exists(sys_model_path):
        # Load the best state dict
        model.load_state_dict(torch.load(sys_model_path))

    train_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train')
    val_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Validation')
    train_model(model, train_loader, val_loader, model_name, sys_save_file_name)

    print(""" Print Results""")
    model_name = 'dias_model'
    dias_model_load_dir = f'{cfg.LOAD_DIAS_BP_MODEL_DIR}'
    lists_path = f"{dias_model_load_dir}/{model_name}_objective_lists"
    if os.path.exists(lists_path):
        print("Load lists:", lists_path)
        val_objective_list, train_objective_list = load_data(lists_path)
        print(len(train_objective_list))
        plot_results(train_objective_list[:100], val_objective_list[:100], model_name, dias_model_load_dir)

    model_name = 'sys_model'
    lists_path = f"{sys_save_file_name}/{model_name}_objective_lists"
    if os.path.exists(lists_path):
        print("Load lists:", lists_path)
        val_objective_list, train_objective_list = load_data(lists_path)
    plot_results(train_objective_list[:100], val_objective_list[:100], model_name)


if __name__ == "__main__":
    # cfg.NN_LOG.redirect_output()
    warnings.simplefilter(action='ignore')
    main()
    # cfg.NN_LOG.close_log_file()
