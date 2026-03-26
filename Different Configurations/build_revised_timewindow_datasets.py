from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# Configuration
# =========================
SOURCE_DATA_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Preprocessing/classification_datasets_response")
OUTPUT_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Revised/revised_timewindow_datasets")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DATASET_NAME = "PD_ON_vs_PD_OFF"

# windows in seconds
TIME_WINDOWS = [
    (-0.4, 0.2),
    (-0.3, 0.1),
    (-0.2, 0.2),
    (-0.5, 0.0),
]


# =========================
# Helpers
# =========================
def sanitize_window_name(tmin, tmax):
    def fmt(x):
        sign = "m" if x < 0 else "p"
        val = str(abs(x)).replace(".", "")
        return f"{sign}{val}"
    return f"{fmt(tmin)}_to_{fmt(tmax)}"


def main():
    npz_path = SOURCE_DATA_DIR / f"{DATASET_NAME}.npz"
    times_path = SOURCE_DATA_DIR / "response_locked_times.npy"
    ch_names_path = SOURCE_DATA_DIR / "response_locked_ch_names.npy"
    sfreq_path = SOURCE_DATA_DIR / "response_locked_sfreq.txt"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing source dataset: {npz_path}")
    if not times_path.exists():
        raise FileNotFoundError(f"Missing times array: {times_path}")
    if not ch_names_path.exists():
        raise FileNotFoundError(f"Missing ch_names array: {ch_names_path}")
    if not sfreq_path.exists():
        raise FileNotFoundError(f"Missing sfreq file: {sfreq_path}")

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]                       # [n_epochs, n_channels, n_times]
    y = data["y"]
    subject_ids = data["subject_ids"]
    session_ids = data["session_ids"]
    condition = data["condition"]
    rt = data["rt"]

    times = np.load(times_path)
    ch_names = np.load(ch_names_path, allow_pickle=True)
    with open(sfreq_path, "r") as f:
        sfreq = float(f.read().strip())

    summary_rows = []

    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded times shape: {times.shape}")
    print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} sec")

    for tmin, tmax in TIME_WINDOWS:
        mask = (times >= tmin) & (times <= tmax)

        if mask.sum() == 0:
            print(f"Skipping window [{tmin}, {tmax}] because no samples were found.")
            continue

        X_window = X[:, :, mask]
        times_window = times[mask]

        window_tag = sanitize_window_name(tmin, tmax)
        out_name = f"{DATASET_NAME}_window_{window_tag}"
        out_npz = OUTPUT_DIR / f"{out_name}.npz"
        out_times = OUTPUT_DIR / f"{out_name}_times.npy"

        np.savez_compressed(
            out_npz,
            X=X_window.astype(np.float32),
            y=y.astype(np.int64),
            subject_ids=subject_ids,
            session_ids=session_ids,
            condition=condition,
            rt=rt.astype(np.float32),
        )
        np.save(out_times, times_window)

        summary_rows.append({
            "dataset_name": out_name,
            "tmin": tmin,
            "tmax": tmax,
            "n_epochs": X_window.shape[0],
            "n_channels": X_window.shape[1],
            "n_times": X_window.shape[2],
            "class_0_epochs": int((y == 0).sum()),
            "class_1_epochs": int((y == 1).sum()),
            "n_unique_subjects": len(np.unique(subject_ids.astype(str))),
            "output_npz": str(out_npz),
        })

        print(
            f"Saved {out_name} | "
            f"shape={X_window.shape} | "
            f"time range={times_window[0]:.3f} to {times_window[-1]:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_out = OUTPUT_DIR / "revised_timewindow_dataset_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    np.save(OUTPUT_DIR / "response_locked_ch_names.npy", ch_names)
    with open(OUTPUT_DIR / "response_locked_sfreq.txt", "w") as f:
        f.write(str(sfreq))

    print(f"\nSaved summary to: {summary_out}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()