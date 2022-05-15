import glob
from multiprocessing import Pool
# import vital_sqi as vs
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath
import pandas as pd
import requests
import wfdb
from tqdm import tqdm
from wfdb.io import download
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,welch
from scipy.stats import entropy
#You can also use Pandas if you so desire
import pandas as pd
import heartpy as hp
from scipy.stats import kurtosis, skew, entropy
from PreProcessing import remove_unrelevant_records
import os

DB_DIR = 'mimic3wdb'
LOAD_DIR = '../mimic3wdb/1.0/'

# load database with ABP and PLETH signals
def load_filtered_records():
    for root, dirs, files in os.walk(LOAD_DIR):
        records = []
        path = root.split(os.sep)
        print(f"loading {os.path.basename(root)}")
    print(os.path.exists(LOAD_DIR))
    records_list = glob.glob(LOAD_DIR, recursive=True)
    print(records_list)


    # print("loading records..")
    # pool = Pool()
    # for _ in tqdm.tqdm(pool.imap(func=load_record(), iterable=records_path_list), total=len(records_path_list)):
    #     pass


def load_record(records_path):
    records = []
    total_windows = total_bp_filtered = total_ppg_filtered = 0
    # traverse root directory, and list directories as dirs and files as files
    print("loading records...")
    for root, dirs, files in os.walk(records_path):
        records = []
        path = root.split(os.sep)
        print(f"loading {os.path.basename(root)}")
        # print(root)
        # print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            # print(len(path) * '---', file)
            with open(f'{posixpath.join(root, file)}', 'rb') as f:
                record = pickle.load(f)
                records.append(record)

        num_windows, num_filtered_bp_samples, num_filtered_ppg_samples = filter_and_save_data(records)
        total_windows += num_windows
        total_bp_filtered += num_filtered_bp_samples
        total_ppg_filtered += num_filtered_ppg_samples
        print(
            f"Num Windows (30 sec): {total_windows}, After bp filter: {total_bp_filtered}, After ppg filter: {total_ppg_filtered}")

    print("loading records done")


def main():

    load_filtered_records()


if __name__ == "__main__":
    main()


