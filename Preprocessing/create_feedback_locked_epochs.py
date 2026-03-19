import os
from glob import glob
from pathlib import Path

import mne
import pandas as pd
import numpy as np

# Configuration
BIDS_ROOT = Path("/Users/Raw Data") # Raw Data
TASK_NAME = "SimonConflict"

TRIAL_TABLE_DIR = Path("/Users/derived_trial_tables")
OUTPUT_DIR = Path("/Users/derived_epochs_feedback")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TMIN = -0.2
TMAX = 0.8
BASELINE = (-0.2, 0.0)
REJECT_CRITERIA = None

KEEP_ONLY_COMPLETE = True
KEEP_ONLY_MAIN_ANALYSIS = True
KEEP_ONLY_FEEDBACK_PRESENT = True


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
    return OUTPUT_DIR / f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_fb-epo.fif"


def load_trial_table(subject_id, session_id):
    fpath = get_trial_table_path(subject_id, session_id)
    if not fpath.exists():
        raise FileNotFoundError(f"Trial table not found: {fpath}")

    df = pd.read_csv(fpath)

    if KEEP_ONLY_COMPLETE and "is_complete_trial" in df.columns:
        df = df[df["is_complete_trial"] == 1].copy()

    if KEEP_ONLY_MAIN_ANALYSIS and "exclude_from_main_analysis" in df.columns:
        df = df[df["exclude_from_main_analysis"] == 0].copy()

    if KEEP_ONLY_FEEDBACK_PRESENT and "feedback_present" in df.columns:
        df = df[df["feedback_present"] == 1].copy()

    df = df[df["feedback_onset"].notna()].copy()
    df = df.sort_values("feedback_onset").reset_index(drop=True)

    return df


def make_event_id_from_metadata(df):
    event_names = []
    event_codes = []
    event_id = {}
    next_code = 1

    for _, row in df.iterrows():
        group = str(row.get("group", "UNK"))
        med_state = str(row.get("med_state", "NA"))
        feedback_value = str(row.get("feedback_value", "UNK"))

        if group == "CTL":
            cond = "CTL"
        elif group == "PD":
            cond = f"PD_{med_state}"
        else:
            cond = "UNK"

        fb_label = f"FB_{feedback_value}".replace(" ", "")
        name = f"{cond}/{fb_label}"

        if name not in event_id:
            event_id[name] = next_code
            next_code += 1

        event_names.append(name)
        event_codes.append(event_id[name])

    return event_names, np.array(event_codes, dtype=int), event_id


def build_events_array(raw, trial_df, event_codes):
    sfreq = raw.info["sfreq"]
    first_samp = raw.first_samp

    samples = np.round(trial_df["feedback_onset"].to_numpy(dtype=float) * sfreq).astype(int) + first_samp

    events = np.column_stack([
        samples,
        np.zeros(len(samples), dtype=int),
        event_codes
    ])

    return events


def create_feedback_locked_epochs(subject_id, session_id):
    print(f"\n=== Feedback-locked epoching: sub-{subject_id}, ses-{session_id} ===")

    fif_path = get_preprocessed_fif_path(subject_id, session_id)
    trial_path = get_trial_table_path(subject_id, session_id)
    out_path = get_output_epoch_path(subject_id, session_id)

    if not fif_path.exists():
        print(f"  -> Missing preprocessed fif: {fif_path}")
        return

    if not trial_path.exists():
        print(f"  -> Missing trial table: {trial_path}")
        return

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    trial_df = load_trial_table(subject_id, session_id)

    if len(trial_df) == 0:
        print("  -> No valid feedback-locked trials after filtering. Skipping.")
        return

    event_names, event_codes, event_id = make_event_id_from_metadata(trial_df)
    trial_df = trial_df.copy()
    trial_df["event_name"] = event_names
    trial_df["event_code"] = event_codes

    events = build_events_array(raw, trial_df, event_codes)

    metadata_cols = [
        "subject_id",
        "session_id",
        "group",
        "med_state",
        "stim_onset",
        "stim_code_raw",
        "response_onset",
        "response_side",
        "response_correctness",
        "is_correct",
        "rt",
        "feedback_present",
        "feedback_onset",
        "feedback_value",
        "event_name",
        "event_code",
    ]
    metadata_cols = [c for c in metadata_cols if c in trial_df.columns]
    metadata = trial_df[metadata_cols].copy()

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        picks="eeg",
        preload=True,
        reject=REJECT_CRITERIA,
        metadata=metadata,
        verbose=True,
    )

    epochs.save(out_path, overwrite=True)
    print(f"  -> Saved: {out_path}")
    print(f"  -> Epochs kept: {len(epochs)}")
    print(f"  -> Event ID: {event_id}")


def main():
    test_pairs = [("027", "01")]

    for subject_id, session_id in test_pairs:
        try:
            create_feedback_locked_epochs(subject_id, session_id)
        except Exception as e:
            print(f"Failed for sub-{subject_id}, ses-{session_id}: {e}")


if __name__ == "__main__":
    main()
