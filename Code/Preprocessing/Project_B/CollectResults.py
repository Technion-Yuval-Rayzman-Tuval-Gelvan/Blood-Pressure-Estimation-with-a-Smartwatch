import logging
import os
import warnings
import time
import torch
from tqdm import tqdm
import heartpy as hp
import Utils as utils
import Config as cfg
from Code.Training import ResNet, HDF5DataLoader
from Code.Training.Training import get_device, calculate_test_score


def main():
    assert cfg.WORK_MODE == cfg.Mode.nn_results
    device = get_device()
    data_path = cfg.DATASET_DIR
    model_path = cfg.NN_MODELS

    print("****** Check Test Score *******")
    """Load Dias Model"""
    model = ResNet.create_resnet_model().to(device)
    model_name = 'dias_model'
    assert os.path.exists(cfg.LOAD_DIAS_BP_MODEL_DIR)
    # Load the best state dict
    print(f"Load Model: {cfg.LOAD_DIAS_BP_MODEL_DIR}/dias_model")
    model.load_state_dict(torch.load(f"{cfg.LOAD_DIAS_BP_MODEL_DIR}/dias_model"))

    test_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Test')
    calculate_test_score(model, test_loader, model_name)

    # """Load Sys Model"""
    # model = ResNet.create_resnet_model().to(device)
    # model_name = 'sys_model'
    # if os.path.exists(cfg.LOAD_SYS_BP_MODEL_DIR):
    #     # Load the best state dict
    #     print(f"Load Model: {cfg.LOAD_SYS_BP_MODEL_DIR}/sys_model")
    #     model.load_state_dict(torch.load(f"{cfg.LOAD_SYS_BP_MODEL_DIR}/dias_model"))
    #
    # test_loader = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Test')
    # calculate_test_score(model, test_loader, model_name)


if __name__ == "__main__":
    cfg.NN_LOG.redirect_output()
    warnings.simplefilter(action='ignore')
    main()
    cfg.NN_LOG.close_log_file()