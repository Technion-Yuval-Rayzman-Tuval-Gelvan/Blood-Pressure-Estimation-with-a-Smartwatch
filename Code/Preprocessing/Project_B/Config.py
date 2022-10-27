import enum
import os
from sklearn import svm
from datetime import datetime
from multiprocessing import Lock

from Code.Preprocessing.Project_B.Logger import Logger


class Dataset(enum.Enum):
    mimic = 0
    cardiac = 1

# EXPERIMENT = ''

# ------------------------------------------------
#                   CONFIG
# ------------------------------------------------
DATASET = Dataset.mimic
# DATASET = Dataset.cardiac
PLOT = False
MAX_PLOT_PER_LABEL = 10
MIN_RECORDS_PER_PATIENT = 1000  # take only patients with more records
TRAIN_RECORDS_PER_PATIENT = 50  # how many records to take from each patient for training
NUM_PATIENTS = 5
ALL_PATIENTS = False
HIGH_THRESH = 100
LOW_THRESH = 70
WINDOWS_PER_LABEL = 8000
TRAIN_MODELS = False
SIGNAL_TYPE = 'bp'  # ppg or bp
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
TIME_DIR = f'Experiments/{TIME}'

if DATASET == Dataset.mimic:
    DATA_DIR = f'{BASE_DIR}/mimic_data/{EXP_DIR}'
    FREQUENCY = 125
else:
    DATA_DIR = f'{BASE_DIR}/cardiac_data/{EXP_DIR}'
    FREQUENCY = 256

WINDOWS_DIR = f'{DATA_DIR}/windows'
PLOT_DIR = f'{DATA_DIR}/windows_plots'
HIST_DIR = f'{DATA_DIR}/histogram_plots'
SVM_DIR = f'{DATA_DIR}/{TIME_DIR}/svm_plots'
LDA_DIR = f'{DATA_DIR}/{TIME_DIR}/lda_plots'
QDA_DIR = f'{DATA_DIR}/{TIME_DIR}/qda_plots'
MAH_DIR = f'{DATA_DIR}/{TIME_DIR}/mah_plots'
MODELS_DIR = f'{DATA_DIR}/{TIME_DIR}/models'
PPG_MODELS_LOAD_DIR = '/media/tuvalgelvan@staff.technion.ac.il/HD34/Estimated-Blood-Pressure-Project/mimic_data/ppg_thresh_100_70/Final_results/Final_Results_26_10_2022_16_23_19/models'
BP_MODELS_LOAD_DIR = '/media/tuvalgelvan@staff.technion.ac.il/HD34/Estimated-Blood-Pressure-Project/mimic_data/bp_thresh_100_70/Final_results/Final_result_27_10_2022_15_21_19/models'

DIRS_LIST = [DATA_DIR, WINDOWS_DIR, PLOT_DIR, HIST_DIR, SVM_DIR, LDA_DIR, QDA_DIR, MAH_DIR, MODELS_DIR]

for output_dir in DIRS_LIST:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None
LOG = Logger(f'{DATA_DIR}/{TIME_DIR}/output.log')
# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None
