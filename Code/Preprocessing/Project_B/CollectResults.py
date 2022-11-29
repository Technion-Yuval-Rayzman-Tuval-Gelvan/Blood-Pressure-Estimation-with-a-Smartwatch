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

    """Load Dias Model"""
    model = ResNet.create_resnet_model().to(device)
    model_name = 'dias_model'
    assert os.path.exists(cfg.LOAD_DIAS_BP_MODEL_DIR)
    # Load the best state dict
    print(f"Load Model: {cfg.LOAD_DIAS_BP_MODEL_DIR}/dias_model")
    model.load_state_dict(torch.load(f"{cfg.LOAD_DIAS_BP_MODEL_DIR}/dias_model"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # print("****** Check Train Histogram *******")
    # trainer = Trainer(model, loss_fn, optimizer, device)
    # dl_train = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train')
    # epoch_res = trainer.test_epoch(dl_train, verbose=True, plot_confusion=True)
    # pred_labels = [pred.cpu().numpy() for pred in epoch_res.pred_labels]
    # target_labels = [pred.cpu().numpy() for pred in epoch_res.target_labels]
    # save_path = f"{cfg.NN_RESULTS_DIR}/dias train_list"
    # with open(save_path, 'wb') as file:
    #     pickle.dump(epoch_res, file)
    # # confusion_matrix(epoch_res, save_dir=cfg.NN_RESULTS_DIR, model_name=model_name)
    # calculate_histogram(pred_labels, 'dias pred labels')
    # calculate_histogram(target_labels, 'dias target labels')

    print("****** Check Test Score *******")
    trainer = Trainer(model, loss_fn, optimizer, device)
    dl_test = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Test')
    epoch_res = trainer.test_epoch(dl_test, verbose=True, plot_confusion=True)
    pred_labels = [pred.cpu().numpy() for pred in epoch_res.pred_labels]
    target_labels = [pred.cpu().numpy() for pred in epoch_res.target_labels]
    confusion_matrix(epoch_res, save_dir=cfg.NN_RESULTS_DIR, model_name=model_name)
    calculate_histogram(pred_labels, model_name)
    calculate_histogram(target_labels, model_name)
    calculate_test_score(model, dl_test, model_name)

    """Load Sys Model"""
    model = ResNet.create_resnet_model().to(device)
    model_name = 'sys_model'
    if os.path.exists(cfg.LOAD_SYS_BP_MODEL_DIR):
        # Load the best state dict
        print(f"Load Model: {cfg.LOAD_SYS_BP_MODEL_DIR}/sys_model")
        model.load_state_dict(torch.load(f"{cfg.LOAD_SYS_BP_MODEL_DIR}/sys_model"))

    print("****** Check Train Histogram *******")
    trainer = Trainer(model, loss_fn, optimizer, device)
    dl_train = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Train')
    epoch_res = trainer.test_epoch(dl_train, verbose=True, plot_confusion=True)
    pred_labels = [pred.cpu().numpy() for pred in epoch_res.pred_labels]
    target_labels = [pred.cpu().numpy() for pred in epoch_res.target_labels]
    save_path = f"{cfg.NN_RESULTS_DIR}/sys_train_list"
    with open(save_path, 'wb') as file:
        pickle.dump(epoch_res, file)
    # confusion_matrix(epoch_res, save_dir=cfg.NN_RESULTS_DIR, model_name=model_name)
    calculate_histogram(pred_labels, 'sys pred labels')
    calculate_histogram(target_labels, 'sys target labels')

    trainer = Trainer(model, loss_fn, optimizer, device)
    dl_test = HDF5DataLoader.get_hdf5_dataset(data_path, model_name, 'Test')
    epoch_res = trainer.test_epoch(dl_test, verbose=True, plot_confusion=True)
    pred_labels = [pred.cpu().numpy() for pred in epoch_res.pred_labels]
    target_labels = [pred.cpu().numpy() for pred in epoch_res.target_labels]
    confusion_matrix(epoch_res, save_dir=cfg.NN_RESULTS_DIR, model_name=model_name)
    calculate_histogram(pred_labels, model_name)
    calculate_histogram(target_labels, model_name)
    calculate_test_score(model, dl_test, model_name)


if __name__ == "__main__":
    cfg.NN_LOG.redirect_output()
    warnings.simplefilter(action='ignore')
    main()
    cfg.NN_LOG.close_log_file()