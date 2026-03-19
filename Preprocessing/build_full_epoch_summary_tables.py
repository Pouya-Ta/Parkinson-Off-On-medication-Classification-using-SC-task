import os
from glob import glob
from pathlib import Path

import mne
import pandas as pd
import numpy as np

# Configuration
BIDS_ROOT = Path("/Users/Raw Data")
TRIAL_TABLE_DIR = Path("/Users/derived_trial_tables")
STIM_EPOCH_DIR = Path("/Users/derived_epochs")
RESP_EPOCH_DIR = Path("/Users/derived_epochs_response")
FB_EPOCH_DIR = Path("/Users/derived_epochs_feedback")
OUTPUT_DIR = Path("/Users/summary_tables")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TASK_NAME = "SimonConflict"


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


def get_trial_table_path(subject_id, session_id):
    return TRIAL_TABLE_DIR / f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_trial_table.csv"

def get_stim_epoch_path(subject_id, session_id):
    return STIM_EPOCH_DIR / f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_stim-epo.fif"

def get_resp_epoch_path(subject_id, session_id):
    return RESP_EPOCH_DIR / f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_resp-epo.fif"

def get_fb_epoch_path(subject_id, session_id):
    return FB_EPOCH_DIR / f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_fb-epo.fif"

def safe_read_epochs_len(fpath):
    if not fpath.exists():
        return np.nan
    epochs = mne.read_epochs(fpath, preload=False, verbose=False)
    return len(epochs)

def load_trial_table(subject_id, session_id):
    fpath = get_trial_table_path(subject_id, session_id)
    if not fpath.exists():
        return None
    return pd.read_csv(fpath)

def derive_condition_label(group, med_state):
    group = str(group)
    med_state = str(med_state)

    if group == "CTL":
        return "CTL"
    if group == "PD":
        return f"PD_{med_state}"
    return "UNK"

# Per-session summary
def summarize_one_subject_session(subject_id, session_id):
    trial_df = load_trial_table(subject_id, session_id)

    if trial_df is None or len(trial_df) == 0:
        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "group": np.nan,
            "med_state": np.nan,
            "condition": np.nan,
            "n_trials_total": 0,
            "n_trials_complete": 0,
            "n_trials_main_analysis": 0,
            "n_response_trials": 0,
            "n_no_response_trials": 0,
            "n_feedback_trials": 0,
            "n_correct_trials": 0,
            "n_incorrect_trials": 0,
            "mean_rt_sec": np.nan,
            "std_rt_sec": np.nan,
            "n_stim_epochs": np.nan,
            "n_resp_epochs": np.nan,
            "n_fb_epochs": np.nan,
        }

    group = trial_df["group"].iloc[0] if "group" in trial_df.columns else np.nan
    med_state = trial_df["med_state"].iloc[0] if "med_state" in trial_df.columns else np.nan
    condition = derive_condition_label(group, med_state)

    n_trials_total = len(trial_df)
    n_trials_complete = int(trial_df["is_complete_trial"].sum()) if "is_complete_trial" in trial_df.columns else np.nan

    if "exclude_from_main_analysis" in trial_df.columns:
        main_df = trial_df[trial_df["exclude_from_main_analysis"] == 0].copy()
    else:
        main_df = trial_df.copy()

    n_trials_main_analysis = len(main_df)
    n_response_trials = int((main_df["response_exists"] == 1).sum()) if "response_exists" in main_df.columns else np.nan
    n_no_response_trials = int((main_df["no_response"] == 1).sum()) if "no_response" in main_df.columns else np.nan
    n_feedback_trials = int((main_df["feedback_present"] == 1).sum()) if "feedback_present" in main_df.columns else np.nan

    if "is_correct" in main_df.columns:
        n_correct_trials = int((main_df["is_correct"] == 1).sum())
        n_incorrect_trials = int((main_df["is_correct"] == 0).sum())
    else:
        n_correct_trials = np.nan
        n_incorrect_trials = np.nan

    if "rt" in main_df.columns:
        rt_series = main_df.loc[main_df["response_exists"] == 1, "rt"].dropna()
        mean_rt_sec = rt_series.mean() if len(rt_series) > 0 else np.nan
        std_rt_sec = rt_series.std() if len(rt_series) > 0 else np.nan
    else:
        mean_rt_sec = np.nan
        std_rt_sec = np.nan

    n_stim_epochs = safe_read_epochs_len(get_stim_epoch_path(subject_id, session_id))
    n_resp_epochs = safe_read_epochs_len(get_resp_epoch_path(subject_id, session_id))
    n_fb_epochs = safe_read_epochs_len(get_fb_epoch_path(subject_id, session_id))

    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "group": group,
        "med_state": med_state,
        "condition": condition,
        "n_trials_total": n_trials_total,
        "n_trials_complete": n_trials_complete,
        "n_trials_main_analysis": n_trials_main_analysis,
        "n_response_trials": n_response_trials,
        "n_no_response_trials": n_no_response_trials,
        "n_feedback_trials": n_feedback_trials,
        "n_correct_trials": n_correct_trials,
        "n_incorrect_trials": n_incorrect_trials,
        "mean_rt_sec": mean_rt_sec,
        "std_rt_sec": std_rt_sec,
        "n_stim_epochs": n_stim_epochs,
        "n_resp_epochs": n_resp_epochs,
        "n_fb_epochs": n_fb_epochs,
    }

# Group-level summary
def build_group_summary(session_summary_df):
    grouped = session_summary_df.groupby("condition", dropna=False)

    rows = []
    for condition, g in grouped:
        row = {
            "condition": condition,
            "n_sessions": len(g),
            "n_unique_subjects": g["subject_id"].astype(str).nunique(),

            "sum_trials_total": g["n_trials_total"].sum(),
            "sum_trials_main_analysis": g["n_trials_main_analysis"].sum(),
            "sum_response_trials": g["n_response_trials"].sum(),
            "sum_no_response_trials": g["n_no_response_trials"].sum(),
            "sum_feedback_trials": g["n_feedback_trials"].sum(),
            "sum_correct_trials": g["n_correct_trials"].sum(),
            "sum_incorrect_trials": g["n_incorrect_trials"].sum(),

            "sum_stim_epochs": g["n_stim_epochs"].sum(),
            "sum_resp_epochs": g["n_resp_epochs"].sum(),
            "sum_fb_epochs": g["n_fb_epochs"].sum(),

            "mean_trials_main_per_session": g["n_trials_main_analysis"].mean(),
            "mean_stim_epochs_per_session": g["n_stim_epochs"].mean(),
            "mean_resp_epochs_per_session": g["n_resp_epochs"].mean(),
            "mean_fb_epochs_per_session": g["n_fb_epochs"].mean(),

            "mean_rt_sec_across_sessions": g["mean_rt_sec"].mean(),
            "std_rt_sec_across_sessions": g["mean_rt_sec"].std(),
        }

        correct = row["sum_correct_trials"]
        incorrect = row["sum_incorrect_trials"]
        total_scored = correct + incorrect

        if total_scored > 0:
            row["pooled_accuracy"] = correct / total_scored
        else:
            row["pooled_accuracy"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


# Main
def main():
    pairs = get_subject_session_pairs(BIDS_ROOT)

    session_rows = []
    for subject_id, session_id in pairs:
        row = summarize_one_subject_session(subject_id, session_id)
        session_rows.append(row)

    session_summary_df = pd.DataFrame(session_rows).sort_values(
        ["condition", "subject_id", "session_id"]
    ).reset_index(drop=True)

    group_summary_df = build_group_summary(session_summary_df)

    session_out = OUTPUT_DIR / "full_session_level_summary.csv"
    group_out = OUTPUT_DIR / "full_group_level_summary.csv"

    session_summary_df.to_csv(session_out, index=False)
    group_summary_df.to_csv(group_out, index=False)

    print(f"Saved full session-level summary to: {session_out}")
    print(f"Saved full group-level summary to: {group_out}")

    print("\nFull session-level summary preview:")
    print(session_summary_df.head().to_string(index=False))

    print("\nFull group-level summary:")
    print(group_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
