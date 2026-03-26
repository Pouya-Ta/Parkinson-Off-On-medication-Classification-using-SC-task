from pathlib import Path
import numpy as np
import pandas as pd

# Configuration
SOURCE_DATA_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Preprocessing/classification_datasets_response")
OUTPUT_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Revised/revised_channel_subset_datasets")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DATASET_NAME = "PD_ON_vs_PD_OFF"

CHANNEL_SUBSETS = {
    "all_channels": None,
    "motor_related": ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4"],
    "fronto_central": ["Fz", "FC1", "FC2", "FCz", "Cz", "C1", "C2", "C3", "C4"],
    "midline": ["Fz", "FCz", "Cz", "Pz"],
}


def main():
    npz_path = SOURCE_DATA_DIR / f"{DATASET_NAME}.npz"
    ch_names_path = SOURCE_DATA_DIR / "response_locked_ch_names.npy"
    times_path = SOURCE_DATA_DIR / "response_locked_times.npy"
    sfreq_path = SOURCE_DATA_DIR / "response_locked_sfreq.txt"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing dataset: {npz_path}")
    if not ch_names_path.exists():
        raise FileNotFoundError(f"Missing ch_names: {ch_names_path}")
    if not times_path.exists():
        raise FileNotFoundError(f"Missing times: {times_path}")
    if not sfreq_path.exists():
        raise FileNotFoundError(f"Missing sfreq: {sfreq_path}")

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]  # [n_epochs, n_channels, n_times]
    y = data["y"]
    subject_ids = data["subject_ids"]
    session_ids = data["session_ids"]
    condition = data["condition"]
    rt = data["rt"]

    ch_names = np.load(ch_names_path, allow_pickle=True).tolist()
    times = np.load(times_path)
    with open(sfreq_path, "r") as f:
        sfreq = float(f.read().strip())

    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded {len(ch_names)} channel names.")

    summary_rows = []

    for subset_name, subset_channels in CHANNEL_SUBSETS.items():
        if subset_channels is None:
            selected_indices = list(range(len(ch_names)))
            selected_ch_names = ch_names.copy()
        else:
            missing = [ch for ch in subset_channels if ch not in ch_names]
            if missing:
                raise ValueError(f"Missing channels for subset '{subset_name}': {missing}")

            selected_indices = [ch_names.index(ch) for ch in subset_channels]
            selected_ch_names = [ch_names[i] for i in selected_indices]

        X_subset = X[:, selected_indices, :]

        out_name = f"{DATASET_NAME}_subset_{subset_name}"
        out_npz = OUTPUT_DIR / f"{out_name}.npz"
        out_ch = OUTPUT_DIR / f"{out_name}_ch_names.npy"

        np.savez_compressed(
            out_npz,
            X=X_subset.astype(np.float32),
            y=y.astype(np.int64),
            subject_ids=subject_ids,
            session_ids=session_ids,
            condition=condition,
            rt=rt.astype(np.float32),
        )
        np.save(out_ch, np.array(selected_ch_names, dtype=object))

        summary_rows.append({
            "dataset_name": out_name,
            "subset_name": subset_name,
            "n_epochs": X_subset.shape[0],
            "n_channels": X_subset.shape[1],
            "n_times": X_subset.shape[2],
            "channels": ", ".join(selected_ch_names),
            "class_0_epochs": int((y == 0).sum()),
            "class_1_epochs": int((y == 1).sum()),
            "n_unique_subjects": len(np.unique(subject_ids.astype(str))),
            "output_npz": str(out_npz),
        })

        print(f"Saved {out_name} | shape={X_subset.shape}")

    summary_df = pd.DataFrame(summary_rows)
    summary_out = OUTPUT_DIR / "revised_channel_subset_dataset_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    # Shared metadata
    np.save(OUTPUT_DIR / "response_locked_times.npy", times)
    with open(OUTPUT_DIR / "response_locked_sfreq.txt", "w") as f:
        f.write(str(sfreq))

    print(f"\nSaved summary to: {summary_out}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()