import os
from glob import glob
from pathlib import Path

import mne
import pandas as pd
import numpy as np

# Configuration
BIDS_ROOT = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/Raw Data")
TASK_NAME = "SimonConflict"

TRIAL_TABLE_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Preprocessing/derived_trial_tables")
OUTPUT_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/derived_epochs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TMIN = -0.2
TMAX = 0.8
BASELINE = (-0.2, 0.0)
REJECT_CRITERIA = dict(eeg=150e-6)

KEEP_ONLY_COMPLETE = True
KEEP_ONLY_MAIN_ANALYSIS = True
