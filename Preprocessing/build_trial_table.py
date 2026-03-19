import os
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
SCRIPT_DIR = Path(__file__).parent
BIDS_ROOT = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/Raw Data")
TASK_NAME = "SimonConflict"
PARTICIPANTS_FPATH = BIDS_ROOT / "participants.tsv"
OUTPUT_DIR = SCRIPT_DIR / "derived_trial_tables"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MIN_RT = 0.15   # seconds
MAX_RT = 3.00   # seconds


# ----------------------------
# Helpers
# ----------------------------
def get_subject_session_pairs(bids_root):
    subjects = glob(os.path.join(bids_root, "sub-*"))
    pairs = []
    for sub_path in subjects:
        subject_id = os.path.basename(sub_path).split("-")[1]
        sessions = glob(os.path.join(sub_path, "ses-*"))
        if sessions:
            for ses_path in sessions:
                session_id = os.path.basename(ses_path).split("-")[1]
                pairs.append((subject_id, session_id))
        else:
            pairs.append((subject_id, None))
    return sorted(pairs)


def get_group_and_med_state(subject_id, session_id, participants_df):
    row = participants_df.loc[
        participants_df["participant_id"] == f"sub-{subject_id}"
    ]
    if row.empty:
        raise ValueError(f"Participant sub-{subject_id} not found in participants.tsv")

    row = row.iloc[0]
    group = row["Group"]

    if group == "CTL":
        return "CTL", "NA"

    if group == "PD":
        if session_id == "01":
            med_state = row["sess1_Med"]
        elif session_id == "02":
            med_state = row["sess2_Med"]
        else:
            raise ValueError(f"Unexpected session_id={session_id} for PD subject.")

        return "PD", str(med_state)

    raise ValueError(f"Unknown group '{group}' for sub-{subject_id}")


def read_events_tsv(subject_id, session_id):
    bids_dir = BIDS_ROOT / f"sub-{subject_id}" / f"ses-{session_id}" / "eeg"
    fpath = bids_dir / f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_events.tsv"

    if not fpath.exists():
        raise FileNotFoundError(f"Missing events file: {fpath}")

    df = pd.read_csv(fpath, sep="\t")
    df = df.copy()

    # Keep only rows with a trial_type string
    df = df[df["trial_type"].notna()].reset_index(drop=True)

    # Standardize strings
    df["trial_type"] = df["trial_type"].astype(str).str.strip()
    df["value"] = df["value"].astype(str).str.strip()

    return df, fpath


def is_test_stim(trial_type):
    return trial_type.startswith("Test Stim:")


def is_test_resp(trial_type):
    return trial_type.startswith("Test Resp:")


def is_test_no_response(trial_type):
    return trial_type == "Test No Response"


def is_feedback(trial_type):
    return trial_type.startswith("FB:")


def is_next_test_stim(trial_type):
    return trial_type.startswith("Test Stim:")


def parse_test_stim(trial_type):
    """
    Example:
    'Test Stim: AB'
    """
    code = trial_type.replace("Test Stim:", "").strip()
    return {
        "stim_code_raw": code,
        "trial_type_raw": trial_type,
    }


def parse_test_resp(trial_type):
    """
    Example:
    'Test Resp: left,correct'
    'Test Resp: right,incorrect'
    """
    text = trial_type.replace("Test Resp:", "").strip()
    parts = [p.strip() for p in text.split(",")]

    response_side = np.nan
    response_correctness = np.nan

    if len(parts) >= 1:
        response_side = parts[0]
    if len(parts) >= 2:
        response_correctness = parts[1]

    return {
        "response_side": response_side,
        "response_correctness": response_correctness,
    }


def parse_feedback(trial_type):
    """
    Example:
    'FB: +1'
    'FB: 0'
    """
    text = trial_type.replace("FB:", "").strip()
    return {
        "feedback_value": text
    }


def build_trials_from_events(events_df, subject_id, session_id, group, med_state):
    rows = []

    # Keep test events + feedback only
    keep_mask = events_df["trial_type"].apply(
        lambda x: (
            is_test_stim(x)
            or is_test_resp(x)
            or is_test_no_response(x)
            or is_feedback(x)
        )
    )
    df = events_df.loc[keep_mask].reset_index(drop=True)

    n = len(df)

    for i in range(n):
        current_type = df.loc[i, "trial_type"]

        if not is_test_stim(current_type):
            continue

        stim_onset = float(df.loc[i, "onset"])
        stim_value_raw = df.loc[i, "value"]
        stim_info = parse_test_stim(current_type)

        # Default trial record
        trial = {
            "subject_id": subject_id,
            "session_id": session_id,
            "group": group,
            "med_state": med_state,

            "stim_index": i,
            "stim_onset": stim_onset,
            "stim_value_raw": stim_value_raw,
            "stim_code_raw": stim_info["stim_code_raw"],
            "trial_type_raw": stim_info["trial_type_raw"],

            "response_exists": 0,
            "response_onset": np.nan,
            "response_value_raw": np.nan,
            "response_side": np.nan,
            "response_correctness": np.nan,
            "no_response": 0,
            "rt": np.nan,

            "feedback_present": 0,
            "feedback_onset": np.nan,
            "feedback_value": np.nan,

            "is_complete_trial": 0,
            "has_multiple_outcomes": 0,
            "rt_too_fast": 0,
            "rt_too_slow": 0,
            "rt_invalid": 0,
            "exclude_from_main_analysis": 0,
            "exclude_reason": "",

            # placeholders for future mapping
            "congruency": np.nan,
            "expected_response_side": np.nan,
            "stimulus_identity": np.nan,
            "stimulus_position": np.nan,
        }

        # Search forward until next Test Stim
        j = i + 1
        outcome_found = False
        outcome_count = 0
        first_outcome_idx = None

        while j < n:
            tt = df.loc[j, "trial_type"]

            if is_next_test_stim(tt):
                break

            if is_test_resp(tt) or is_test_no_response(tt):
                outcome_count += 1
                if not outcome_found:
                    outcome_found = True
                    first_outcome_idx = j

            j += 1

        # Handle missing outcome
        if not outcome_found:
            trial["exclude_from_main_analysis"] = 1
            trial["exclude_reason"] = "missing_outcome_before_next_test_stim"
            rows.append(trial)
            continue

        if outcome_count > 1:
            trial["has_multiple_outcomes"] = 1

        # Fill outcome
        outcome_type = df.loc[first_outcome_idx, "trial_type"]
        outcome_onset = float(df.loc[first_outcome_idx, "onset"])
        outcome_value = df.loc[first_outcome_idx, "value"]

        if is_test_resp(outcome_type):
            resp_info = parse_test_resp(outcome_type)
            trial["response_exists"] = 1
            trial["response_onset"] = outcome_onset
            trial["response_value_raw"] = outcome_value
            trial["response_side"] = resp_info["response_side"]
            trial["response_correctness"] = resp_info["response_correctness"]
            trial["no_response"] = 0
            trial["rt"] = outcome_onset - stim_onset

        elif is_test_no_response(outcome_type):
            trial["response_exists"] = 0
            trial["no_response"] = 1

        # Search feedback after outcome until next Test Stim
        k = first_outcome_idx + 1
        while k < n:
            tt = df.loc[k, "trial_type"]

            if is_next_test_stim(tt):
                break

            if is_feedback(tt):
                fb_info = parse_feedback(tt)
                trial["feedback_present"] = 1
                trial["feedback_onset"] = float(df.loc[k, "onset"])
                trial["feedback_value"] = fb_info["feedback_value"]
                break

            k += 1

        # Trial completeness
        trial["is_complete_trial"] = 1

        # RT quality checks
        if trial["response_exists"] == 1:
            if pd.isna(trial["rt"]) or trial["rt"] <= 0:
                trial["rt_invalid"] = 1
                trial["exclude_from_main_analysis"] = 1
                trial["exclude_reason"] = "invalid_rt"
            elif trial["rt"] < MIN_RT:
                trial["rt_too_fast"] = 1
                trial["exclude_from_main_analysis"] = 1
                trial["exclude_reason"] = "rt_too_fast"
            elif trial["rt"] > MAX_RT:
                trial["rt_too_slow"] = 1
                trial["exclude_from_main_analysis"] = 1
                trial["exclude_reason"] = "rt_too_slow"

        # Add binary accuracy column
        if trial["response_correctness"] == "correct":
            trial["is_correct"] = 1
        elif trial["response_correctness"] == "incorrect":
            trial["is_correct"] = 0
        else:
            trial["is_correct"] = np.nan
        rows.append(trial)

    trial_df = pd.DataFrame(rows)
    return trial_df


def make_summary(trial_df):
    total_trials = len(trial_df)
    complete_trials = int(trial_df["is_complete_trial"].sum()) if total_trials > 0 else 0
    responded = int((trial_df["response_exists"] == 1).sum()) if total_trials > 0 else 0
    no_response = int((trial_df["no_response"] == 1).sum()) if total_trials > 0 else 0
    valid_main = int((trial_df["exclude_from_main_analysis"] == 0).sum()) if total_trials > 0 else 0

    correct_trials = int((trial_df["response_correctness"] == "correct").sum()) if total_trials > 0 else 0
    incorrect_trials = int((trial_df["response_correctness"] == "incorrect").sum()) if total_trials > 0 else 0

    rt_series = trial_df.loc[
        (trial_df["exclude_from_main_analysis"] == 0) &
        (trial_df["response_exists"] == 1),
        "rt"
    ]

    summary = {
        "total_test_stim_trials": total_trials,
        "complete_trials": complete_trials,
        "responded_trials": responded,
        "no_response_trials": no_response,
        "valid_for_main_analysis": valid_main,
        "correct_trials": correct_trials,
        "incorrect_trials": incorrect_trials,
        "mean_rt_sec": rt_series.mean() if len(rt_series) > 0 else np.nan,
        "std_rt_sec": rt_series.std() if len(rt_series) > 0 else np.nan,
    }

    return pd.DataFrame([summary])


def process_one_subject_session(subject_id, session_id, participants_df):
    print(f"\n=== Building trial table for sub-{subject_id}, ses-{session_id} ===")

    group, med_state = get_group_and_med_state(subject_id, session_id, participants_df)
    events_df, events_fpath = read_events_tsv(subject_id, session_id)

    trial_df = build_trials_from_events(
        events_df=events_df,
        subject_id=subject_id,
        session_id=session_id,
        group=group,
        med_state=med_state
    )

    summary_df = make_summary(trial_df)

    base_name = f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}"
    trial_out = OUTPUT_DIR / f"{base_name}_trial_table.csv"
    summary_out = OUTPUT_DIR / f"{base_name}_trial_summary.csv"

    trial_df.to_csv(trial_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"  -> Events read from: {events_fpath}")
    print(f"  -> Trial table saved to: {trial_out}")
    print(f"  -> Summary saved to: {summary_out}")
    print(summary_df.to_string(index=False))


def main():
    if not PARTICIPANTS_FPATH.exists():
        raise FileNotFoundError(f"participants.tsv not found at: {PARTICIPANTS_FPATH}")

    participants_df = pd.read_csv(
    PARTICIPANTS_FPATH,
    sep="\t",
    keep_default_na=False,
    dtype=str
    )

    subject_session_pairs = get_subject_session_pairs(BIDS_ROOT)

    for subject_id, session_id in subject_session_pairs:
        if session_id is None:
            continue

        try:
            process_one_subject_session(subject_id, session_id, participants_df)
        except Exception as e:
            print(f"⚠️ Failed for sub-{subject_id}, ses-{session_id}: {e}")


if __name__ == "__main__":
    main()
