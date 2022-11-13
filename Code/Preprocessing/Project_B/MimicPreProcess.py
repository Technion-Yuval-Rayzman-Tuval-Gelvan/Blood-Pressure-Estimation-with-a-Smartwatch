import copy
import pickle
from multiprocessing import Pool, Lock
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import heartpy as hp
import Utils as utils
import Config as cfg
import Plot as plot
import Trainer
if cfg.TRAIN_MODELS is True:
    from Code.Preprocessing.Project_B.CreateDataset import DatasetCreator
from SQI import SQI
import warnings


def classify_target(signal, fs):
    try:
        wd, m = hp.process(signal, fs)
    except:
        return

    peaks_len = len(wd['peaklist'])
    num_bad_peaks = np.count_nonzero(wd['RR_masklist'])
    quality_percent = (1 - (num_bad_peaks / peaks_len)) * 100
    high_tresh = int(cfg.HIGH_THRESH)
    low_tresh = int(cfg.LOW_THRESH)

    if quality_percent >= high_tresh:
        target = utils.Label.good
    elif quality_percent >= low_tresh:
        target = utils.Label.mid
    else:
        target = utils.Label.bad

    return target


def save_records_list():
    records_list = [os.path.join(path, name) for path, subdirs, files in os.walk(cfg.MIMIC_LOAD_DIR) for name in files]
    # save list
    with open('records_list', 'wb') as file:
        pickle.dump(records_list, file)


def save_good_records_list():
    records_list = []
    for path, subdirs, files in os.walk(cfg.MIMIC_LOAD_DIR):
        if len(files) >= cfg.MIN_RECORDS_PER_PATIENT:
            files_list = []
            for name in files:
                files_list.append(os.path.join(path, name))
            records_list.append(files_list)
    # save list
    with open(f'{cfg.MIN_RECORDS_PER_PATIENT}_records_per_patient_list', 'wb') as file:
        pickle.dump(records_list, file)


def main():
    assert cfg.DATASET == cfg.Dataset.mimic and cfg.TRAIN_MODELS is True
    assert cfg.WORK_MODE == cfg.Mode.train_sqi_models

    """save list of records"""
    # if cfg.MIN_RECORDS_PER_PATIENT > 0:
    #     save_good_records_list()
    # else:
    #     save_records_list()

    """save records as windows"""
    # dataset_creator = DatasetCreator()
    # dataset_creator.create_dataset()

    # """load_windows_dictionary"""
    win_list = utils.load_list()

    """  Train Models  """
    win_dict = utils.convert_list_to_dict(win_list)

    """plot windows"""
    # if cfg.PLOT:
    #     utils.plot_windows(win_dict)

    """histogram of labels"""
    plot.label_histogram(win_dict)
    plot.features_histogram(win_dict)

    with pd.ExcelWriter(f'{cfg.DATA_DIR}/{cfg.TIME_DIR}/accuracy.xlsx') as excel_writer:

        """good/mid"""
        print("************************* Good / Mid **************************************")
        trainer = Trainer.Trainer(true_label=utils.Label.good, false_label=utils.Label.mid,
                                  win_dict=win_dict, excel_writer=excel_writer)
        trainer.run()

        """good/bad"""
        print("************************* Good / Bad **************************************")
        trainer = Trainer.Trainer(true_label=utils.Label.good, false_label=utils.Label.bad,
                                  win_dict=win_dict, excel_writer=excel_writer)
        trainer.run()

        """mid/bad"""
        print("************************* Mid / Bad **************************************")
        trainer = Trainer.Trainer(true_label=utils.Label.mid, false_label=utils.Label.bad,
                                  win_dict=win_dict, excel_writer=excel_writer)
        trainer.run()


if __name__ == "__main__":
    cfg.LOG.redirect_output()
    warnings.simplefilter(action='ignore', category=FutureWarning)
    main()
    cfg.LOG.close_log_file()
