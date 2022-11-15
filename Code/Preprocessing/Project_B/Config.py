import enum
import os
from sklearn import svm
from datetime import datetime
from multiprocessing import Lock

from Code.Preprocessing.Project_B.Logger import Logger


class Dataset(enum.Enum):
    mimic = 0
    cardiac = 1


class BPType(enum.Enum):
    diastolic = 0
    systolic = 1

class Mode(enum.Enum):
    train_sqi_models = 0
    save_valid_data = 1  # After training use models to classify and save valid data to NN module
    compare_results = 2
    nn_training = 3


# ------------------------------------------------
#                   CONFIG
# ------------------------------------------------
DATASET = Dataset.mimic
WORK_MODE = Mode.nn_training

# DATASET = Dataset.cardiac
PLOT = False
MAX_PLOT_PER_LABEL = 10
MIN_RECORDS_PER_PATIENT = 1000  # take only patients with more records
TRAIN_RECORDS_PER_PATIENT = 3  # how many records to take from each patient for training
NUM_PATIENTS = 5
ALL_PATIENTS = True
TRAIN_SEED = 5
TEST_SEED = 10
HIGH_THRESH = 100
LOW_THRESH = 70
WINDOWS_PER_LABEL = 8000
TRAIN_MODELS = False
SIGNAL_TYPE = 'ppg'  # ppg or bp
EXP_DIR = f'{SIGNAL_TYPE}_thresh_{HIGH_THRESH}_{LOW_THRESH}'
TRUE_PPG_SCORE = 2 # Mah + LDA + QDA + SVM ( score can be - [-4, -2, 0, 2, 4])
TRUE_BP_SCORE = 0 # Mah + LDA + QDA + SVM ( score can be - [-4, -2, 0, 2, 4])
FREQUENCY_END = 12
FREQUENCY_START = 0
STFT_WIN_SIZE = 750
NUM_PER_SHARD = 20000

if TRAIN_MODELS:
    SEED = TRAIN_SEED
else:
    SEED = TEST_SEED

# datetime object containing current date and time
now = datetime.now()
TIME = now.strftime("%d_%m_%Y_%H_%M_%S")

# ------------------------------------------------
#                   Directories
# ------------------------------------------------
BASE_DIR = '/host/media/tuvalgelvan@staff.technion.ac.il/HD34/Estimated-Blood-Pressure-Project'
MIMIC_LOAD_DIR = f'{BASE_DIR}/mimic3wdb/1.0'
CARDIAC_LOAD_DIR = f'{BASE_DIR}/cardiac_data/Technion_Synched_Data'
TIME_DIR = f'Experiments/{TIME}'

if DATASET == Dataset.mimic:
    DATA_DIR = f'{BASE_DIR}/mimic_data/{EXP_DIR}'
    FREQUENCY = 125
else:
    DATA_DIR = f'{BASE_DIR}/cardiac_data/{EXP_DIR}'
    FREQUENCY = 256

WINDOWS_DIR = f'{DATA_DIR}/windows'
PLOT_DIR = f'{DATA_DIR}/windows_plots'
CLASSIFY_PLATFORM_DIR = f'{BASE_DIR}/platform_results/{TIME_DIR}_PPG_Score_{TRUE_PPG_SCORE}_BP Score_{TRUE_BP_SCORE}'
CLASSIFIED_PLOTS = f'{CLASSIFY_PLATFORM_DIR}/plots'
HIST_DIR = f'{DATA_DIR}/histogram_plots'
SVM_DIR = f'{DATA_DIR}/{TIME_DIR}/svm_plots'
LDA_DIR = f'{DATA_DIR}/{TIME_DIR}/lda_plots'
QDA_DIR = f'{DATA_DIR}/{TIME_DIR}/qda_plots'
MAH_DIR = f'{DATA_DIR}/{TIME_DIR}/mah_plots'
MODELS_DIR = f'{DATA_DIR}/{TIME_DIR}/models'
PPG_MODELS_LOAD_DIR = f'{BASE_DIR}/mimic_data/ppg_thresh_100_70/Final_results/Final_Results_26_10/models'
BP_MODELS_LOAD_DIR = f'{BASE_DIR}/mimic_data/bp_thresh_100_70/Final_results/Final_result_27_10_2022_15_21_19/models'
DATASET_DIR = f'{BASE_DIR}/NN_Data'
NN_MODELS = f'{BASE_DIR}/nn_models'
DIAS_BP_MODEL_DIR = f'{NN_MODELS}/{TIME_DIR}/Dias'
SYS_BP_MODEL_DIR = f'{NN_MODELS}/{TIME_DIR}/Sys'
LOAD_DIAS_BP_MODEL_DIR = f'{NN_MODELS}/Experiments/10_11_2022_17_52_06/Dias'
LOAD_SYS_BP_MODEL_DIR = f'{NN_MODELS}//Sys'
COMPARE_DIR = f'{BASE_DIR}/Results/CompareModels/{TIME}'

CLASSIFY_DIRS = [CLASSIFY_PLATFORM_DIR, CLASSIFIED_PLOTS]
TRAINING_DIRS = [DATA_DIR, WINDOWS_DIR, PLOT_DIR, HIST_DIR, SVM_DIR, LDA_DIR, QDA_DIR, MAH_DIR, MODELS_DIR]
SAVE_DATA_DIRS = [DATASET_DIR]
NN_TRAIN_DIRS = [DIAS_BP_MODEL_DIR, SYS_BP_MODEL_DIR]
COMPARE_DIRS = [COMPARE_DIR]

match WORK_MODE:
    case Mode.save_valid_data:
        dir_list = SAVE_DATA_DIRS
    case Mode.train_sqi_models:
        if TRAIN_MODELS is True:
            dir_list = TRAINING_DIRS
        else:
            dir_list = CLASSIFY_DIRS
    case Mode.compare_results:
            dir_list = COMPARE_DIRS
    case Mode.nn_training:
            dir_list = NN_TRAIN_DIRS

for output_dir in dir_list:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None
LOG = Logger(f'{DATA_DIR}/{TIME_DIR}/output.log')
CLASSIFICATION_LOG = Logger(f'{CLASSIFIED_PLOTS}/output.log')
DATASET_LOG = Logger(f'{DATASET_DIR}/output.log')
NN_LOG = Logger(f'{NN_MODELS}/{TIME_DIR}/output.log')
# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None
