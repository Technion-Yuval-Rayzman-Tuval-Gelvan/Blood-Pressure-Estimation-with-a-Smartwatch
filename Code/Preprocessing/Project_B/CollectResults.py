import logging
import os
import pickle
import warnings
import time
import torch
from tqdm import tqdm
import heartpy as hp
import Utils as utils
import Config as cfg
from Code.Training import ResNet, HDF5DataLoader
from Code.Training.BPHistogram import calculate_histogram
from Code.Training.PlotConfusion import confusion_matrix
from Code.Training.Trainer import Trainer
from Code.Training.Training import get_device, calculate_test_score


def main():
    assert cfg.WORK_MODE == cfg.Mode.nn_results
    device = get_device()
    data_path = cfg.DATASET_DIR
    model_path = cfg.NN_MODELS
    loss_fn = torch.nn.L1Loss().to(device)

    # """Load Dias Model"""
    # cfg.NN_LOG_DIAS.redirect_output()
    # model_name = 'dias_model'
    # assert os.path.exists(cfg.LOAD_DIAS_BP_MODEL_DIR)
    # # Load the best state dict
    # print(f"Load Model: {cfg.LOAD_DIAS_BP_MODEL_DIR}")
    # model = torch.load(cfg.LOAD_DIAS_BP_MODEL_DIR).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    #
    # print("****** Check Dias Test Score *******")
    # trainer = Trainer(model, loss_fn, optimizer, device)
    # dl_test = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Test', batch_size=16)
    # epoch_res = trainer.test_epoch(dl_test, verbose=True, plot_confusion=True, max_batches=3750)
    # pred_labels = [pred.cpu().numpy() for pred in epoch_res.pred_labels]
    # target_labels = [pred.cpu().numpy() for pred in epoch_res.target_labels]
    # confusion_matrix(epoch_res, save_dir=cfg.BEST_DIAS_RESULTS, model_name=model_name)
    # calculate_histogram(pred_labels, model_name)
    # calculate_histogram(target_labels, model_name)
    # calculate_test_score(model, dl_test, model_name)
    # cfg.NN_LOG_DIAS.close_log_file()

    """Load Sys Model"""
    cfg.NN_LOG_SYS.redirect_output()
    model = ResNet.create_resnet_model().to(device)
    model_name = 'sys_model'
    print(cfg.LOAD_SYS_BP_MODEL_DIR)
    assert os.path.exists(cfg.LOAD_SYS_BP_MODEL_DIR)
    # Load the best state dict
    print(f"Load Model: {cfg.LOAD_SYS_BP_MODEL_DIR}")
    model = torch.load(cfg.LOAD_SYS_BP_MODEL_DIR).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print("****** Check Dias Test Score *******")
    trainer = Trainer(model, loss_fn, optimizer, device, model_name = model_name)
    dl_test = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Test', batch_size=16)
    epoch_res = trainer.test_epoch(dl_test, verbose=True, plot_confusion=True, max_batches=3750)
    pred_labels = [pred.cpu().numpy() for pred in epoch_res.pred_labels]
    target_labels = [pred.cpu().numpy() for pred in epoch_res.target_labels]
    confusion_matrix(epoch_res, save_dir=cfg.BEST_SYS_RESULTS, model_name=model_name)
    calculate_histogram(pred_labels, model_name)
    calculate_histogram(target_labels, model_name)
    calculate_test_score(model, dl_test, model_name)
    cfg.NN_LOG_SYS.close_log_file()


if __name__ == "__main__":
    warnings.simplefilter(action='ignore')
    main()
