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

# Helpers
def get_subject_session_pairs(bids_root):
    subjects = glob(os.path.join(bids_root, "sub-*"))
    pairs = []
    for sub_path in subjects:
        subject_id = os.path.basename(sub_path).split("-")[1]
        sessions = glob(os.path.join(sub_path, "ses-*"))
        for ses_path in sessions:
            session_id = os.path.basename(ses_path).split("-")[1]
            pairs.append((subject_id, session_id))
    return sorted(pairs)


def get_preprocessed_fif_path(subject_id, session_id):
    eeg_dir = BIDS_ROOT / f"sub-{subject_id}" / f"ses-{session_id}" / "eeg"
    return eeg_dir / f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_eeg.fif"


def get_trial_table_path(subject_id, session_id):
    return TRIAL_TABLE_DIR / f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_trial_table.csv"


def get_output_epoch_path(subject_id, session_id):
    return OUTPUT_DIR / f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_stim-epo.fif"


def load_trial_table(subject_id, session_id):
    fpath = get_trial_table_path(subject_id, session_id)
    if not fpath.exists():
        raise FileNotFoundError(f"Trial table not found: {fpath}")

    df = pd.read_csv(fpath)

    if KEEP_ONLY_COMPLETE and "is_complete_trial" in df.columns:
        df = df[df["is_complete_trial"] == 1].copy()

    if KEEP_ONLY_MAIN_ANALYSIS and "exclude_from_main_analysis" in df.columns:
        df = df[df["exclude_from_main_analysis"] == 0].copy()

    df = df[df["stim_onset"].notna()].copy()
    df = df.sort_values("stim_onset").reset_index(drop=True)

    return df


def make_event_id_from_metadata(df):
    event_names = []
    event_codes = []
    event_id = {}
    next_code = 1

    for _, row in df.iterrows():
        group = str(row.get("group", "UNK"))
        med_state = str(row.get("med_state", "NA"))
        is_correct = row.get("is_correct", np.nan)

        if group == "CTL":
            cond = "CTL"
        elif group == "PD":
            cond = f"PD_{med_state}"
        else:
            cond = "UNK"

        if pd.isna(is_correct):
            correctness = "UNK"
        elif int(is_correct) == 1:
            correctness = "correct"
        else:
            correctness = "incorrect"

        name = f"{cond}/{correctness}"

        if name not in event_id:
            event_id[name] = next_code
            next_code += 1

        event_names.append(name)
        event_codes.append(event_id[name])

    return event_names, np.array(event_codes, dtype=int), event_id


def build_events_array(raw, trial_df, event_codes):
    sfreq = raw.info["sfreq"]
    first_samp = raw.first_samp

    samples = np.round(trial_df["stim_onset"].to_numpy(dtype=float) * sfreq).astype(int) + first_samp

    events = np.column_stack([
        samples,
        np.zeros(len(samples), dtype=int),
        event_codes
    ])

    return events
