
""" Compare between MIMIC dataset SQI model to Heartpy """

import logging
import warnings
import time
from tqdm import tqdm
import heartpy as hp
import Utils as utils
import Config as cfg
import Plot as plot
import Trainer
from Code.Preprocessing.Project_B.CreateDataset import DatasetCreator
from Code.Training.LoadData import arrange_folders
from SQI import SQI
import copy
import pickle
from multiprocessing import Pool, Lock
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt


def main():
    assert cfg.WORK_MODE == cfg.Mode.compare_results

    """ Save spectograms """
    dc = DatasetCreator()
    dc.create_dataset()


if __name__ == "__main__":
    # cfg.DATASET_LOG.redirect_output()
    warnings.simplefilter(action='ignore')
    main()
    # cfg.DATASET_LOG.close_log_file()
