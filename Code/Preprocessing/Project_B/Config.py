import enum
import os
from sklearn import svm


class Dataset(enum.Enum):
    mimic = 0
    cardiac = 1

# EXPERIMENT = ''

MODELS = {'svm': svm,

          }

# ------------------------------------------------
#                   CONFIG
# ------------------------------------------------
DATASET = Dataset.cardiac
PLOT = False
MIN_RECORDS_PER_PATIENT = 1000
TRAIN_RECORDS_PER_PATIENT = 100

# ------------------------------------------------
#                   Directories
# ------------------------------------------------
BASE_DIR = '/host/media/tuvalgelvan@staff.technion.ac.il/HD34/Estimated-Blood-Pressure-Project'
MIMIC_LOAD_DIR = f'{BASE_DIR}/mimic3wdb/1.0'
CARDIAC_LOAD_DIR = f'{BASE_DIR}/cardiac_data/Technion_Synched_Data'

if DATASET == Dataset.mimic:
    DATA_DIR = f'{BASE_DIR}/test_data'
    FREQUENCY = 125
else:
    DATA_DIR = f'{BASE_DIR}/cardiac_data'
    FREQUENCY = 256

WINDOWS_DIR = f'{DATA_DIR}/windows'
PLOT_DIR =  f'{DATA_DIR}/windows_plots'
HIST_DIR = f'{DATA_DIR}/histogram_plots'
DIRS_LIST = [DATA_DIR, WINDOWS_DIR, PLOT_DIR, HIST_DIR]

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

