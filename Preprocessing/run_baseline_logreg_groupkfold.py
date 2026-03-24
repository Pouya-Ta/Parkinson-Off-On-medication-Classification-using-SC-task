from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# =========================
# Configuration
# =========================
DATA_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Preprocessing/classification_datasets_response")
OUTPUT_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Preprocessing/baseline_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Choose one dataset to run
# DATASET_NAME = "PD_ON_vs_PD_OFF"
# Other options:
# DATASET_NAME = "CTL_vs_PD_ON"
# DATASET_NAME = "CTL_vs_PD_OFF"
# DATASET_NAME = "PD_ON_vs_PD_OFF"
DATASET_NAME = "CTL_vs_PD"

N_SPLITS = 5
RANDOM_STATE = 42


# =========================
# Helpers
# =========================
def load_dataset(dataset_name: str):
    fpath = DATA_DIR / f"{dataset_name}.npz"
    if not fpath.exists():
        raise FileNotFoundError(f"Dataset not found: {fpath}")

    data = np.load(fpath, allow_pickle=True)

    X = data["X"]                       # [n_epochs, n_channels, n_times]
    y = data["y"]                       # [n_epochs]
    subject_ids = data["subject_ids"]   # [n_epochs]
    session_ids = data["session_ids"]   # [n_epochs]
    condition = data["condition"]       # [n_epochs]
    rt = data["rt"]                     # [n_epochs]

    return X, y, subject_ids, session_ids, condition, rt


def flatten_epochs(X):
    n_epochs, n_channels, n_times = X.shape
    return X.reshape(n_epochs, n_channels * n_times)


def summarize_dataset(X, y, subject_ids, condition):
    df = pd.DataFrame({
        "y": y,
        "subject_id": subject_ids.astype(str),
        "condition": condition.astype(str),
    })

    summary = {
        "n_epochs": len(df),
        "n_features_per_epoch": X.shape[1] * X.shape[2],
        "n_unique_subjects": df["subject_id"].nunique(),
        "n_class_0_epochs": int((df["y"] == 0).sum()),
        "n_class_1_epochs": int((df["y"] == 1).sum()),
        "n_class_0_subjects": df.loc[df["y"] == 0, "subject_id"].nunique(),
        "n_class_1_subjects": df.loc[df["y"] == 1, "subject_id"].nunique(),
        "class_0_conditions": sorted(df.loc[df["y"] == 0, "condition"].unique().tolist()),
        "class_1_conditions": sorted(df.loc[df["y"] == 1, "condition"].unique().tolist()),
    }
    return summary


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE
        ))
    ])


# =========================
# Main training/eval
# =========================
def main():
    X, y, subject_ids, session_ids, condition, rt = load_dataset(DATASET_NAME)

    dataset_summary = summarize_dataset(X, y, subject_ids, condition)
    print("Dataset summary:")
    for k, v in dataset_summary.items():
        print(f"  {k}: {v}")

    X_flat = flatten_epochs(X)
    groups = subject_ids.astype(str)

    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_flat, y, groups=groups), start=1):
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])

        # Sanity check: no overlap in subjects
        overlap = set(train_subjects).intersection(set(test_subjects))
        if overlap:
            raise RuntimeError(f"Subject leakage detected in fold {fold_idx}: {overlap}")

        pipe = build_pipeline()
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fold_rows.append({
            "dataset": DATASET_NAME,
            "fold": fold_idx,
            "n_train_epochs": len(train_idx),
            "n_test_epochs": len(test_idx),
            "n_train_subjects": len(train_subjects),
            "n_test_subjects": len(test_subjects),
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1": f1,
            "auc": auc,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

        print(
            f"Fold {fold_idx}: "
            f"acc={acc:.4f}, "
            f"bacc={bacc:.4f}, "
            f"f1={f1:.4f}, "
            f"auc={auc:.4f}, "
            f"train_subj={len(train_subjects)}, "
            f"test_subj={len(test_subjects)}"
        )

    fold_df = pd.DataFrame(fold_rows)

    summary_row = {
        "dataset": DATASET_NAME,
        "n_folds": len(fold_df),
        "mean_accuracy": fold_df["accuracy"].mean(),
        "std_accuracy": fold_df["accuracy"].std(),
        "mean_balanced_accuracy": fold_df["balanced_accuracy"].mean(),
        "std_balanced_accuracy": fold_df["balanced_accuracy"].std(),
        "mean_f1": fold_df["f1"].mean(),
        "std_f1": fold_df["f1"].std(),
        "mean_auc": fold_df["auc"].mean(),
        "std_auc": fold_df["auc"].std(),
    }
    summary_df = pd.DataFrame([summary_row])

    fold_out = OUTPUT_DIR / f"{DATASET_NAME}_fold_results.csv"
    summary_out = OUTPUT_DIR / f"{DATASET_NAME}_summary.csv"

    fold_df.to_csv(fold_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"\nSaved fold results to: {fold_out}")
    print(f"Saved summary to: {summary_out}")
    print("\nFinal summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()