import os
from glob import glob
from pathlib import Path

import mne
import mne_bids
import pandas as pd

from mne.preprocessing import ICA
from mne_icalabel import label_components

# Configuration
BIDS_ROOT = Path("/Users/Raw Data")
TASK_NAME = "SimonConflict"

N_JOBS = 8
DOWNSAMPLE_FREQ = 250

# Filtering
APPLY_NOTCH = True
NOTCH_FREQ = 60
L_FREQ = 0.5
H_FREQ = 40.0

# ICA
ICA_L_FREQ = 1.0
ICA_METHOD = "infomax"
ICA_RANDOM_STATE = 42
ICA_COMPONENTS = 20
ICA_LABEL_THRESHOLD = 0.80

# Artifact labels to exclude if probability is high enough
ARTIFACT_LABELS = {
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
}

# Output QC
QC_DIR = Path(__file__).parent / "preprocessing_qc"
QC_DIR.mkdir(exist_ok=True, parents=True)


# Helpers
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


def set_channel_types_for_this_dataset(raw):
    """
    For ds003509, many channels are read as 'misc' because channels.tsv uses n/a.
    We explicitly restore scalp channels to EEG, set VEOG to EOG,
    and leave X/Y/Z to be dropped later.
    """
    ch_type_map = {}

    aux_names = {"X", "Y", "Z"}

    for ch in raw.ch_names:
        ch_upper = ch.upper()

        if ch_upper == "VEOG" or "VEOG" in ch_upper:
            ch_type_map[ch] = "eog"
        elif ch in aux_names:
            # accelerometers will be dropped later; keep as misc for now
            continue
        else:
            ch_type_map[ch] = "eeg"

    raw.set_channel_types(ch_type_map)
    return raw


def drop_non_eeg_aux_channels(raw):
    """
    Remove accelerometer channels if present.
    Keep peripheral EEG channels unless there is a strong reason to drop them.
    """
    accel_channels = ["X", "Y", "Z"]
    to_drop = [ch for ch in accel_channels if ch in raw.ch_names]

    if to_drop:
        raw.drop_channels(to_drop)

    return raw, to_drop


def run_ica_and_label(raw):
    """
    Fit ICA on a 1-100 Hz filtered copy for better ICLabel compatibility.
    If ICLabel dependencies are missing, continue without auto-exclusion.
    """
    print("  -> Preparing ICA copy...")
    ica_raw = raw.copy().filter(
        l_freq=1.0,
        h_freq=100.0,
        picks="eeg",
        n_jobs=N_JOBS,
        verbose=False
    )

    print("  -> Running ICA...")
    ica = ICA(
        n_components=ICA_COMPONENTS,
        method="infomax",
        fit_params=dict(extended=True),
        max_iter="auto",
        random_state=ICA_RANDOM_STATE
    )
    ica.fit(ica_raw, picks="eeg", decim=3)

    print("  -> Running ICLabel...")
    exclude_idx = []
    exclude_info = []

    try:
        ic_labels = label_components(ica_raw, ica, method="iclabel")
        labels = ic_labels["labels"]

        probs = None
        for key in ["y_pred_proba", "probs", "probabilities"]:
            if key in ic_labels:
                probs = ic_labels[key]
                break

        for idx, label in enumerate(labels):
            prob = None

            if probs is not None:
                try:
                    if hasattr(probs[idx], "__len__"):
                        prob = float(max(probs[idx]))
                    else:
                        prob = float(probs[idx])
                except Exception:
                    prob = None

            should_exclude = False
            if label in ARTIFACT_LABELS:
                if prob is None:
                    should_exclude = True
                elif prob >= ICA_LABEL_THRESHOLD:
                    should_exclude = True

            if should_exclude:
                exclude_idx.append(idx)
                exclude_info.append(
                    {
                        "component": idx,
                        "label": label,
                        "probability": prob,
                    }
                )

        ica.exclude = exclude_idx

        if exclude_idx:
            print(f"  -> Excluding {len(exclude_idx)} ICs: {exclude_idx}")
            ica.apply(raw)
        else:
            print("  -> No ICA components met exclusion criteria.")

    except Exception as e:
        print(f"  -> ICLabel unavailable or failed: {e}")
        print("  -> Continuing without automatic ICA component rejection.")
        ica.exclude = []

    return raw, ica, exclude_info
"""
def run_ica_and_label(raw):
=    #Fit ICA on a 1 Hz high-pass filtered copy,
    #then apply excluded components to the less aggressively filtered raw.
    print("  -> Preparing ICA copy...")
    ica_raw = raw.copy().filter(
        l_freq=ICA_L_FREQ,
        h_freq=None,
        picks="eeg",
        n_jobs=N_JOBS,
        verbose=False
    )

    print("  -> Running ICA...")
    ica = ICA(
        n_components=ICA_COMPONENTS,
        method=ICA_METHOD,
        max_iter="auto",
        random_state=ICA_RANDOM_STATE
    )
    ica.fit(ica_raw, picks="eeg", decim=3)

    print("  -> Running ICLabel...")
    ic_labels = label_components(ica_raw, ica, method="iclabel")

    labels = ic_labels["labels"]

    # Some versions expose probabilities with different keys
    probs = None
    for key in ["y_pred_proba", "probs", "probabilities"]:
        if key in ic_labels:
            probs = ic_labels[key]
            break

    exclude_idx = []
    exclude_info = []

    for idx, label in enumerate(labels):
        prob = None

        if probs is not None:
            try:
                # Usually probs is per-component confidence for predicted label
                if hasattr(probs[idx], "__len__"):
                    # If array-like, take max probability
                    prob = float(max(probs[idx]))
                else:
                    prob = float(probs[idx])
            except Exception:
                prob = None

        should_exclude = False

        if label in ARTIFACT_LABELS:
            if prob is None:
                should_exclude = True
            elif prob >= ICA_LABEL_THRESHOLD:
                should_exclude = True

        if should_exclude:
            exclude_idx.append(idx)
            exclude_info.append(
                {
                    "component": idx,
                    "label": label,
                    "probability": prob,
                }
            )

    ica.exclude = exclude_idx

    if exclude_idx:
        print(f"  -> Excluding {len(exclude_idx)} ICs: {exclude_idx}")
        ica.apply(raw)
    else:
        print("  -> No ICA components met exclusion criteria.")

    return raw, ica, exclude_info
"""

def save_qc_row(qc_row):
    qc_path = QC_DIR / "preprocessing_qc_summary.csv"
    qc_df = pd.DataFrame([qc_row])

    if qc_path.exists():
        old = pd.read_csv(qc_path)
        qc_df = pd.concat([old, qc_df], ignore_index=True)

        # Keep latest unique row per subject/session if rerun
        qc_df = qc_df.drop_duplicates(
            subset=["subject_id", "session_id"],
            keep="last"
        )

    qc_df.to_csv(qc_path, index=False)


# Main preprocessing
def preprocess_continuous(subject_id, session_id):
    ses_text = session_id if session_id is not None else "N/A"
    print(f"\n=== Preprocessing sub-{subject_id}, ses-{ses_text} ===")

    bids_path = mne_bids.BIDSPath(
        subject=subject_id,
        session=session_id,
        task=TASK_NAME,
        suffix="eeg",
        datatype="eeg",
        root=BIDS_ROOT
    )

    try:
        raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)
    except FileNotFoundError:
        print("  -> Original BIDS file not found. Skipping.")
        return

    raw.load_data()
    print(f"  -> Loaded: {bids_path.fpath}")
    original_sfreq = raw.info["sfreq"]
    original_n_channels = len(raw.ch_names)

    # Drop accelerometer channels only
    raw = set_channel_types_for_this_dataset(raw)
    
    #XYZ
    raw, dropped_aux = drop_non_eeg_aux_channels(raw)
    if dropped_aux:
        print(f"  -> Dropped auxiliary channels: {dropped_aux}")
    else:
        print("  -> No auxiliary accelerometer channels found.")

    # Channel typing
    # raw, dropped_aux = drop_non_eeg_aux_channels(raw)

    # Standard montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="raise")
    print("  -> Standard 10-20 montage applied.")

    # Filtering
    if APPLY_NOTCH:
        print(f"  -> Applying notch filter at {NOTCH_FREQ} Hz...")
        raw.notch_filter(
            freqs=NOTCH_FREQ,
            picks=["eeg", "eog"],
            n_jobs=N_JOBS,
            verbose=False
        )
    else:
        print("  -> Notch filter skipped.")

    print(f"  -> Applying band-pass filter: {L_FREQ}-{H_FREQ} Hz...")
    raw.filter(
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        picks=["eeg", "eog"],
        n_jobs=N_JOBS,
        verbose=False
    )

    # Average reference applied directly
    raw.set_eeg_reference("average", projection=False)
    print("  -> Average reference applied.")

    # ICA + ICLabel
    raw, ica, exclude_info = run_ica_and_label(raw)

    # Downsample
    print(f"  -> Resampling to {DOWNSAMPLE_FREQ} Hz...")
    raw.resample(DOWNSAMPLE_FREQ, n_jobs=N_JOBS)
    final_sfreq = raw.info["sfreq"]

    # Save preprocessed FIF
    output_dir = Path(bids_path.directory)
    output_fname = output_dir / f"{bids_path.basename}.fif"
    raw.save(output_fname, overwrite=True)
    print(f"  -> Saved preprocessed FIF: {output_fname}")

    # Save per-subject IC details
    ic_detail_path = QC_DIR / f"sub-{subject_id}_ses-{session_id}_ica_excluded_components.csv"
    pd.DataFrame(exclude_info).to_csv(ic_detail_path, index=False)

    # Save QC summary row
    qc_row = {
        "subject_id": subject_id,
        "session_id": session_id,
        "input_file": str(bids_path.fpath),
        "output_fif": str(output_fname),
        "original_sfreq": original_sfreq,
        "final_sfreq": final_sfreq,
        "original_n_channels": original_n_channels,
        "final_n_channels": len(raw.ch_names),
        "dropped_aux_channels": ",".join(dropped_aux) if dropped_aux else "",
        "n_ica_excluded": len(exclude_info),
        "ica_excluded_labels": ",".join([x["label"] for x in exclude_info]) if exclude_info else "",
    }
    save_qc_row(qc_row)


def main():
    """
    test_pairs = [("027", "01")]

    for subject_id, session_id in test_pairs:
        try:
            preprocess_continuous(subject_id, session_id)
        except Exception as e:
            print(f"Failed for sub-{subject_id}, ses-{session_id}: {e}")    
    """
    subject_session_pairs = get_subject_session_pairs(BIDS_ROOT)

    print(f"Found {len(subject_session_pairs)} subject/session pairs.")

    for subject_id, session_id in subject_session_pairs:
        try:
            preprocess_continuous(subject_id, session_id)
        except Exception as e:
            print(f"Skipped sub-{subject_id}, ses-{session_id or 'N/A'} Error:")
            print(e)
    

if __name__ == "__main__":
    main()
