from IPython.display import display
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import os
import shutil
import posixpath
import pandas as pd
import wfdb
import requests
import math
import functools
import struct
import pdb
import datetime
import multiprocessing
import posixpath
import pickle
import re
import pytictoc

from pytictoc import TicToc
from wfdb.io import _header
from wfdb.io import _signal
from wfdb.io import download
from wfdb.io import annotation


def save_class(db_dir, dir_name, record):
    # make directory to download the class if not exists already
    os.makedirs(posixpath.join(db_dir, dir_name), exist_ok=True)

    print(f'Save record: {record.record_name}')

    # save class as record name
    with open(f'{posixpath.join(db_dir, dir_name, record.record_name)}', 'wb') as file:
        pickle.dump(record, file)


def load_class(db_dir, dir_name, record_name):
    # load class called record name
    with open(f'{posixpath.join(db_dir, dir_name, record_name)}', 'rb') as file:
        record = pickle.load(file)

    return record


def save_nested_files_list(db_dir):

    # full URL from Physionet
    if '/' in db_dir:
        dir_list = db_dir.split('/')
        db_dir = posixpath.join(dir_list[0], wfdb.io.record.get_version(dir_list[0]), *dir_list[1:])
    else:
        db_dir = posixpath.join(db_dir, wfdb.io.record.get_version(db_dir))
    db_url = posixpath.join(download.PN_CONTENT_URL, db_dir) + '/'
    # Check if the database is valid
    r = requests.get(db_url)
    r.raise_for_status()
    # Get the list of records
    record_list = download.get_record_list(db_dir)

    print(f'Record list directories to download: {record_list}')

    all_files = []
    nested_records = []

    for rec in record_list:
        # Check out whether each record is in MIT or EDF format
        if rec.endswith('.edf'):
            all_files.append(rec)
        else:
            # May be pointing to directory
            if rec.endswith(os.sep):
                nested_records += [posixpath.join(rec, sr) for sr in
                                   download.get_record_list(posixpath.join(db_dir, rec))]
            else:
                nested_records.append(rec)

    print(f'Record list is: {nested_records}')

    # save list
    with open('nested_list', 'wb') as file:
        pickle.dump(nested_records, file)


def cut_list(files_list, file_name):
    for i, file in enumerate(files_list):
        if file == file_name:
            return files_list[i:]


# download only database with ABP and PLETH signals
# based on the function wfdb.dl_database()
def dl_filtered_database(db_dir):
    with open('nested_list', 'rb') as file:
        nested_records = pickle.load(file)

    nested_records = cut_list(nested_records, '33/3307857/3307857n')

    t = TicToc()
    t.tic()
    rec_counter = 0
    total_precetage = 0

    filtered_records_list = []
    for rec in nested_records:
        t.toc()
        rec_counter += 1
        dir_name, base_rec_name = os.path.split(rec)
        rec_header = wfdb.rdheader(base_rec_name, pn_dir=posixpath.join(db_dir, dir_name))

        # download only records with PPG and ABP
        if rec_header.sig_name and 'PLETH' in rec_header.sig_name and 'ABP' in rec_header.sig_name:
            filtered_record = wfdb.rdrecord(rec_header.record_name, pn_dir=posixpath.join(db_dir, dir_name),
                                            channel_names=['PLETH', 'ABP'])
            save_class(db_dir, dir_name, filtered_record)

        # if file name ends with 'n' it contains numeric data
        if rec_header.record_name[-1] == 'n':
            filtered_record = wfdb.rdrecord(rec_header.record_name, pn_dir=posixpath.join(db_dir, dir_name),
                                            channel_names=['ABP SYS', 'ABP DIAS', 'ABP MEAN'])
            save_class(db_dir, dir_name, filtered_record)

        # print each 5% of download
        precentage = (rec_counter * 100) / len(nested_records)
        if precentage % 5 == 0:
            total_precetage += 5
            print(f'****** Download progress: {total_precetage}% *********')


def main():
    # database from Physionet to download
    # use get_db() to find other database
    db_dir = 'mimic3wdb'

    # download data
    # save_nested_files_list(db_dir)
    dl_filtered_database(db_dir)


if __name__ == "__main__":
    main()
