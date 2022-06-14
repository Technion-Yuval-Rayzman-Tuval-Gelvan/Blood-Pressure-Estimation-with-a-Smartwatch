## Setting some nice matplotlib defaults
import os
import matplotlib.pyplot as plt
import Config as cfg

plt.rcParams['figure.figsize'] = (4.5, 4.5)  # Set default plot's sizes
plt.rcParams['figure.dpi'] = 120  # Set default plot's dpi (increase fonts' size)
plt.rcParams['axes.grid'] = True  # Show grid by default in figures

if not os.path.exists(f'{cfg.DATA_DIR}/plots'):
    os.mkdir(f'{cfg.DATA_DIR}/plots')

