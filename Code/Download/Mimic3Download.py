# from IPython.display import display
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
import tqdm
from multiprocessing import Pool, Lock
from pytictoc import TicToc
from wfdb.io import _header
from wfdb.io import _signal
from wfdb.io import download
from wfdb.io import annotation

DB_DIR = 'mimic3wdb'
SAVE_DIR = '/media/administrator/HD34/Estimated-Blood-Pressure-Project/mimic3wdb/1.0'


def save_class(dir_name, record):
    # make directory to download the class if not exists already
    os.makedirs(posixpath.join(SAVE_DIR, dir_name), exist_ok=True)

    # print(f'Save record: {record.record_name}')

    # save class as record name
    with open(f'{posixpath.join(SAVE_DIR, dir_name, record.record_name)}', 'wb') as file:
        pickle.dump(record, file)


def load_class(dir_name, record_name):
    # load class called record name
    with open(f'{posixpath.join(DB_DIR, dir_name, record_name)}', 'rb') as file:
        record = pickle.load(file)

    return record


def save_nested_files_list():
    # full URL from Physionet
    if '/' in DB_DIR:
        dir_list = DB_DIR.split('/')
        DB_DIR = posixpath.join(dir_list[0], wfdb.io.record.get_version(dir_list[0]), *dir_list[1:])
    else:
        DB_DIR = posixpath.join(DB_DIR, wfdb.io.record.get_version(DB_DIR))
    db_url = posixpath.join(download.PN_CONTENT_URL, DB_DIR) + '/'
    # Check if the database is valid
    r = requests.get(db_url)
    r.raise_for_status()
    # Get the list of records
    record_list = download.get_record_list(DB_DIR)

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
                                   download.get_record_list(posixpath.join(DB_DIR, rec))]
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


def get_record(rec):
    dir_name, base_rec_name = os.path.split(rec)
    try:
        rec_header = wfdb.rdheader(base_rec_name, pn_dir=posixpath.join(DB_DIR, dir_name))

        # download only records with PPG and ABP
        if rec_header.sig_name and 'PLETH' in rec_header.sig_name and 'ABP' in rec_header.sig_name:
            filtered_record = wfdb.rdrecord(rec_header.record_name, pn_dir=posixpath.join(DB_DIR, dir_name),
                                            channel_names=['PLETH', 'ABP'])
            save_class(dir_name, filtered_record)

        # if file name ends with 'n' it contains numeric data
        if rec_header.record_name[-1] == 'n':
            filtered_record = wfdb.rdrecord(rec_header.record_name, pn_dir=posixpath.join(DB_DIR, dir_name),
                                            channel_names=['ABP SYS', 'ABP DIAS', 'ABP MEAN'])
            save_class(dir_name, filtered_record)
    except:
        print(f"Error at record: {rec}")


# download only database with ABP and PLETH signals
# based on the function wfdb.dl_database()
def dl_filtered_database():
    with open('nested_list', 'rb') as file:
        nested_records = pickle.load(file)

    # nested_records = cut_list(nested_records, '32/3294477/3294477_0053')
    print("download records..")
    pool = Pool()
    for _ in tqdm.tqdm(pool.imap(func=get_record, iterable=nested_records), total=len(nested_records)):
        pass


def main():
    # database from Physionet to download
    # use get_db() to find other database to change DB_DIR

    # download data
    # save_nested_files_list()
    dl_filtered_database()


if __name__ == "__main__":
    main()
