import enum
import os
from sklearn import svm
from datetime import datetime


class Dataset(enum.Enum):
    mimic = 0
    cardiac = 1

# EXPERIMENT = ''

MODELS = {'svm': svm,

          }

# ------------------------------------------------
#                   CONFIG
# ------------------------------------------------
DATASET = Dataset.mimic
# DATASET = Dataset.cardiac
PLOT = False
MAX_PLOT_PER_LABEL = 10
MIN_RECORDS_PER_PATIENT = 1000  # take only patients with more records
TRAIN_RECORDS_PER_PATIENT = 10  # how many records to take from each patient for training
HIGH_THRESH = 100
LOW_THRESH = 70
WINDOWS_PER_LABEL = 8000
SIGNAL_TYPE = 'ppg'
EXP_DIR = f'{SIGNAL_TYPE}_thresh_{HIGH_THRESH}_{LOW_THRESH}'

# ------------------------------------------------
#                   Directories
# ------------------------------------------------
BASE_DIR = '/host/media/tuvalgelvan@staff.technion.ac.il/HD34/Estimated-Blood-Pressure-Project'
MIMIC_LOAD_DIR = f'{BASE_DIR}/mimic3wdb/1.0'
CARDIAC_LOAD_DIR = f'{BASE_DIR}/cardiac_data/Technion_Synched_Data'

# datetime object containing current date and time
now = datetime.now()
TIME = now.strftime("%d_%m_%Y_%H_%M_%S")

if DATASET == Dataset.mimic:
    DATA_DIR = f'{BASE_DIR}/mimic_data/{EXP_DIR}'
    FREQUENCY = 125
else:
    DATA_DIR = f'{BASE_DIR}/cardiac_data/{EXP_DIR}'
    FREQUENCY = 256

WINDOWS_DIR = f'{DATA_DIR}/windows'
PLOT_DIR = f'{DATA_DIR}/windows_plots'
HIST_DIR = f'{DATA_DIR}/histogram_plots'
SVM_DIR = f'{DATA_DIR}/svm_plots/{TIME}'

DIRS_LIST = [DATA_DIR, WINDOWS_DIR, PLOT_DIR, HIST_DIR, SVM_DIR]

for output_dir in DIRS_LIST:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None
# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None
