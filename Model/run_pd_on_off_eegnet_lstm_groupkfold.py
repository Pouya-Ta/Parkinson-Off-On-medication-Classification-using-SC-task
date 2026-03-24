from pathlib import Path
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Configuration
# =========================
"""
DATA_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Preprocessing/classification_datasets_response")
OUTPUT_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Preprocessing/eegnet_lstm_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
"""

DATA_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Preprocessing/classification_datasets_response")
OUTPUT_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Revised/NewSeed77/eegnet_lstm_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


DATASET_NAME = "PD_ON_vs_PD_OFF"
#DATASET_NAME = "CTL_vs_PD_OFF"
#DATASET_NAME = "CTL_vs_PD_ON"
# DATASET_NAME = "CTL_vs_PD"

N_SPLITS = 5
RANDOM_STATE = 42

BATCH_SIZE = 64
N_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 77):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        """
        X shape: [n_samples, 1, n_channels, n_times]
        y shape: [n_samples]
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# Hybrid EEGNet-LSTM
# =========================
class EEGNetLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
    ):
        super().__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # infer LSTM input size and sequence length
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            x = self.firstconv(dummy)
            x = self.depthwise(x)
            x = self.separable(x)
            # x shape: [1, F2, 1, T_reduced]
            _, feat_dim, _, seq_len = x.shape

        self.lstm_input_dim = feat_dim
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0 if lstm_layers == 1 else dropout,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        # [B, F2, 1, T] -> [B, T, F2]
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]   # last time step
        x = self.classifier(x)
        return x.squeeze(1)


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


def summarize_dataset(X, y, subject_ids, condition):
    df = pd.DataFrame({
        "y": y,
        "subject_id": subject_ids.astype(str),
        "condition": condition.astype(str),
    })

    summary = {
        "n_epochs": len(df),
        "shape_per_epoch": X.shape[1:],
        "n_unique_subjects": df["subject_id"].nunique(),
        "n_class_0_epochs": int((df["y"] == 0).sum()),
        "n_class_1_epochs": int((df["y"] == 1).sum()),
        "n_class_0_subjects": df.loc[df["y"] == 0, "subject_id"].nunique(),
        "n_class_1_subjects": df.loc[df["y"] == 1, "subject_id"].nunique(),
        "class_0_conditions": sorted(df.loc[df["y"] == 0, "condition"].unique().tolist()),
        "class_1_conditions": sorted(df.loc[df["y"] == 1, "condition"].unique().tolist()),
    }
    return summary


def normalize_train_test(X_train, X_test):
    """
    Train-fold-only normalization.
    X shape: [n_samples, n_channels, n_times]
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1.0

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm


def train_one_fold(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_probs = []
    all_preds = []
    all_true = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_probs.append(probs)
        all_preds.append(preds)
        all_true.append(y_batch.numpy().astype(int))

    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1": f1,
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# =========================
# Main
# =========================
def main():
    set_seed(RANDOM_STATE)

    X, y, subject_ids, session_ids, condition, rt = load_dataset(DATASET_NAME)
    dataset_summary = summarize_dataset(X, y, subject_ids, condition)

    print("Dataset summary:")
    for k, v in dataset_summary.items():
        print(f"  {k}: {v}")

    groups = subject_ids.astype(str)
    gkf = GroupKFold(n_splits=N_SPLITS)

    n_epochs, n_channels, n_times = X.shape
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"\n===== Fold {fold_idx} =====")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_subjects = np.unique(groups[train_idx])
        test_subjects = np.unique(groups[test_idx])

        overlap = set(train_subjects).intersection(set(test_subjects))
        if overlap:
            raise RuntimeError(f"Subject leakage detected in fold {fold_idx}: {overlap}")

        X_train, X_test = normalize_train_test(X_train, X_test)

        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]

        train_ds = EEGDataset(X_train, y_train)
        test_ds = EEGDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = EEGNetLSTM(n_channels=n_channels, n_times=n_times).to(DEVICE)

        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_bacc = -np.inf
        best_metrics = None
        best_epoch = None

        for epoch in range(1, N_EPOCHS + 1):
            train_loss = train_one_fold(model, train_loader, optimizer, criterion)
            metrics = evaluate(model, test_loader)

            if metrics["balanced_accuracy"] > best_bacc:
                best_bacc = metrics["balanced_accuracy"]
                best_metrics = metrics.copy()
                best_epoch = epoch

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | "
                    f"loss={train_loss:.4f} | "
                    f"acc={metrics['accuracy']:.4f} | "
                    f"bacc={metrics['balanced_accuracy']:.4f} | "
                    f"f1={metrics['f1']:.4f} | "
                    f"auc={metrics['auc']:.4f}"
                )

        fold_rows.append({
            "dataset": DATASET_NAME,
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "n_train_epochs": len(train_idx),
            "n_test_epochs": len(test_idx),
            "n_train_subjects": len(train_subjects),
            "n_test_subjects": len(test_subjects),
            **best_metrics,
        })

        print(
            f"Best fold result | epoch={best_epoch} | "
            f"acc={best_metrics['accuracy']:.4f}, "
            f"bacc={best_metrics['balanced_accuracy']:.4f}, "
            f"f1={best_metrics['f1']:.4f}, "
            f"auc={best_metrics['auc']:.4f}"
        )

    fold_df = pd.DataFrame(fold_rows)

    summary_row = {
        "dataset": DATASET_NAME,
        "n_folds": len(fold_df),
        "mean_best_epoch": fold_df["best_epoch"].mean(),
        "std_best_epoch": fold_df["best_epoch"].std(),
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

    fold_out = OUTPUT_DIR / f"{DATASET_NAME}_eegnet_lstm_fold_results.csv"
    summary_out = OUTPUT_DIR / f"{DATASET_NAME}_eegnet_lstm_summary.csv"

    fold_df.to_csv(fold_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"\nSaved fold results to: {fold_out}")
    print(f"Saved summary to: {summary_out}")
    print("\nFinal summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()