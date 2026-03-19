import os
from glob import glob
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# Configuration
BIDS_ROOT = Path("/Users/Raw Data")
RESP_EPOCH_DIR = Path("/Users/derived_epochs_response")
OUTPUT_DIR = Path("/Users/classification_datasets_response")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TASK_NAME = "SimonConflict"
KEEP_ONLY_CORRECT = True

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

def get_resp_epoch_path(subject_id, session_id):
    return RESP_EPOCH_DIR / f"sub-{subject_id}_ses-{session_id}_task-{TASK_NAME}_resp-epo.fif"

def derive_condition(group, med_state):
    group = str(group)
    med_state = str(med_state)

    if group == "CTL":
        return "CTL"
    if group == "PD":
        return f"PD_{med_state}"
    return "UNK"

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

# Load all response-locked epochs
def load_all_response_epochs():
    pairs = get_subject_session_pairs(BIDS_ROOT)

    X_list = []
    meta_rows = []
    common_ch_names = None
    times = None
    sfreq = None

    for subject_id, session_id in pairs:
        fpath = get_resp_epoch_path(subject_id, session_id)

        if not fpath.exists():
            print(f"Missing response epochs: {fpath}")
            continue

        print(f"Loading: {fpath}")
        epochs = mne.read_epochs(fpath, preload=True, verbose=False)

        if len(epochs) == 0:
            print("  -> Empty epochs, skipping.")
            continue

        if common_ch_names is None:
            common_ch_names = epochs.ch_names
            times = epochs.times
            sfreq = epochs.info["sfreq"]
        else:
            if epochs.ch_names != common_ch_names:
                raise ValueError(f"Channel mismatch in {fpath}")

            if len(epochs.times) != len(times) or not np.allclose(epochs.times, times):
                raise ValueError(f"Time axis mismatch in {fpath}")

        metadata = epochs.metadata.copy() if epochs.metadata is not None else pd.DataFrame(index=np.arange(len(epochs)))

        # Ensure required columns exist
        for col in ["subject_id", "session_id", "group", "med_state", "is_correct", "rt"]:
            if col not in metadata.columns:
                metadata[col] = np.nan

        # Optional filter: only correct trials
        if KEEP_ONLY_CORRECT:
            keep_mask = metadata["is_correct"] == 1
            epochs = epochs[keep_mask.to_numpy()]
            metadata = metadata.loc[keep_mask].reset_index(drop=True)

        if len(epochs) == 0:
            print("  -> No epochs left after filtering.")
            continue

        data = epochs.get_data()  # shape: [n_epochs, n_channels, n_times]

        # Build session-level metadata rows
        for i in range(len(epochs)):
            row = metadata.iloc[i].to_dict()

            row["subject_id"] = safe_str(row.get("subject_id", subject_id)).zfill(3)
            row["session_id"] = safe_str(row.get("session_id", session_id)).zfill(2)
            row["group"] = safe_str(row.get("group", ""))
            row["med_state"] = safe_str(row.get("med_state", ""))
            row["condition"] = derive_condition(row["group"], row["med_state"])

            meta_rows.append(row)

        X_list.append(data)

    if len(X_list) == 0:
        raise RuntimeError("No response-locked epochs found after filtering.")

    X = np.concatenate(X_list, axis=0)
    metadata_df = pd.DataFrame(meta_rows).reset_index(drop=True)

    return X, metadata_df, common_ch_names, times, sfreq

# Build binary comparison datasets
def build_binary_dataset(X, metadata_df, class_a, class_b, output_name):
    subset_mask = metadata_df["condition"].isin([class_a, class_b])
    sub_meta = metadata_df.loc[subset_mask].reset_index(drop=True)
    sub_X = X[subset_mask.to_numpy()]

    if len(sub_meta) == 0:
        print(f"No data found for {output_name}")
        return

    y = np.where(sub_meta["condition"] == class_a, 0, 1)

    subject_ids = sub_meta["subject_id"].astype(str).to_numpy()
    session_ids = sub_meta["session_id"].astype(str).to_numpy()
    rt = pd.to_numeric(sub_meta["rt"], errors="coerce").to_numpy()

    out_path = OUTPUT_DIR / f"{output_name}.npz"

    np.savez_compressed(
        out_path,
        X=sub_X.astype(np.float32),
        y=y.astype(np.int64),
        subject_ids=subject_ids,
        session_ids=session_ids,
        condition=sub_meta["condition"].astype(str).to_numpy(),
        rt=rt.astype(np.float32),
        ch_names=np.array(sub_X.shape[1] * [""]),  # placeholder, filled later if needed
    )

    summary = {
        "dataset": output_name,
        "class_0": class_a,
        "class_1": class_b,
        "n_epochs_total": len(sub_meta),
        "n_epochs_class_0": int((y == 0).sum()),
        "n_epochs_class_1": int((y == 1).sum()),
        "n_subjects_class_0": sub_meta.loc[sub_meta["condition"] == class_a, "subject_id"].nunique(),
        "n_subjects_class_1": sub_meta.loc[sub_meta["condition"] == class_b, "subject_id"].nunique(),
        "output_path": str(out_path),
    }

    return summary

def build_combined_pd_dataset(X, metadata_df):
    sub_meta = metadata_df.loc[
        metadata_df["condition"].isin(["CTL", "PD_ON", "PD_OFF"])
    ].copy().reset_index(drop=True)

    if len(sub_meta) == 0:
        print("No data found for CTL vs PD")
        return

    sub_X = X[sub_meta.index.to_numpy()]
    y = np.where(sub_meta["condition"] == "CTL", 0, 1)

    subject_ids = sub_meta["subject_id"].astype(str).to_numpy()
    session_ids = sub_meta["session_id"].astype(str).to_numpy()
    rt = pd.to_numeric(sub_meta["rt"], errors="coerce").to_numpy()

    out_path = OUTPUT_DIR / "CTL_vs_PD.npz"

    np.savez_compressed(
        out_path,
        X=sub_X.astype(np.float32),
        y=y.astype(np.int64),
        subject_ids=subject_ids,
        session_ids=session_ids,
        condition=sub_meta["condition"].astype(str).to_numpy(),
        rt=rt.astype(np.float32),
        ch_names=np.array(sub_X.shape[1] * [""]),
    )

    summary = {
        "dataset": "CTL_vs_PD",
        "class_0": "CTL",
        "class_1": "PD_combined",
        "n_epochs_total": len(sub_meta),
        "n_epochs_class_0": int((y == 0).sum()),
        "n_epochs_class_1": int((y == 1).sum()),
        "n_subjects_class_0": sub_meta.loc[sub_meta["condition"] == "CTL", "subject_id"].nunique(),
        "n_subjects_class_1": sub_meta.loc[sub_meta["condition"].isin(["PD_ON", "PD_OFF"]), "subject_id"].nunique(),
        "output_path": str(out_path),
    }

    return summary


# Main
def main():
    X, metadata_df, ch_names, times, sfreq = load_all_response_epochs()

    print(f"\nLoaded all response-locked data:")
    print(f"X shape = {X.shape}")
    print(f"Metadata rows = {len(metadata_df)}")
    print(f"Unique conditions = {metadata_df['condition'].value_counts().to_dict()}")

    # Save master metadata
    metadata_out = OUTPUT_DIR / "response_locked_master_metadata.csv"
    metadata_df.to_csv(metadata_out, index=False)

    # Save shared arrays
    np.save(OUTPUT_DIR / "response_locked_times.npy", times)
    np.save(OUTPUT_DIR / "response_locked_ch_names.npy", np.array(ch_names))
    with open(OUTPUT_DIR / "response_locked_sfreq.txt", "w") as f:
        f.write(str(sfreq))

    summaries = []

    s1 = build_binary_dataset(X, metadata_df, "CTL", "PD_ON", "CTL_vs_PD_ON")
    if s1 is not None:
        summaries.append(s1)

    s2 = build_binary_dataset(X, metadata_df, "CTL", "PD_OFF", "CTL_vs_PD_OFF")
    if s2 is not None:
        summaries.append(s2)

    s3 = build_binary_dataset(X, metadata_df, "PD_ON", "PD_OFF", "PD_ON_vs_PD_OFF")
    if s3 is not None:
        summaries.append(s3)

    s4 = build_combined_pd_dataset(X, metadata_df)
    if s4 is not None:
        summaries.append(s4)

    summary_df = pd.DataFrame(summaries)
    summary_out = OUTPUT_DIR / "dataset_build_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    print(f"\nSaved master metadata to: {metadata_out}")
    print(f"Saved dataset summary to: {summary_out}")
    print("\nDataset summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
