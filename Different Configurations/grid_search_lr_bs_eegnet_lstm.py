"""
Grid search over batch sizes and learning rates for the EEGNet-LSTM model.

Search grid
-----------
  Batch sizes   : 64, 128
  Learning rates: 1e-3, 1e-4, 1e-5

Early stopping (patience = 7 epochs on validation loss) is applied to every
configuration so training automatically stops once a config has converged.
"""

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
DATA_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Preprocessing/classification_datasets_response")
OUTPUT_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Revised/grid_search_lr_bs_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DATASET_NAME = "PD_ON_vs_PD_OFF"
N_SPLITS = 5
RANDOM_STATE = 42

N_EPOCHS = 200                  # high cap — early stopping is the primary terminator
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7    # stop if val loss does not improve for 7 epochs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grid search axes
BATCH_SIZES = [64, 128]
LEARNING_RATES = [1e-3, 1e-4, 1e-5]

# Fixed architecture hyper-parameters (best from prior search)
LSTM_HIDDEN = 64
DROPOUT = 0.4


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Dataset
# =========================
class EEGDataset(Dataset):
    def __init__(self, X, y):
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

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            x = self.firstconv(dummy)
            x = self.depthwise(x)
            x = self.separable(x)
            _, feat_dim, _, seq_len = x.shape

        self.lstm_input_dim = feat_dim
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0 if lstm_layers == 1 else dropout,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.squeeze(2).permute(0, 2, 1)   # [B, T, F]
        x, _ = self.lstm(x)
        x = x[:, -1, :]                      # last time step
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
    X = data["X"]
    y = data["y"]
    subject_ids = data["subject_ids"].astype(str)
    session_ids = data["session_ids"].astype(str)
    condition = data["condition"].astype(str)
    rt = data["rt"]

    return X, y, subject_ids, session_ids, condition, rt


def normalize_train_test(X_train, X_test):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def compute_val_loss(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    all_probs, all_preds, all_true = [], [], []

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

    return {"accuracy": acc, "balanced_accuracy": bacc, "f1": f1, "auc": auc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def make_config_name(batch_size, lr):
    lr_txt = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"bs{batch_size}_lr{lr_txt}"


# =========================
# Run one grid cell
# =========================
def run_one_config(X, y, subject_ids, batch_size, lr):
    set_seed(RANDOM_STATE)

    config_name = make_config_name(batch_size, lr)
    print(f"\n{'='*50}")
    print(f"Config: {config_name}  (bs={batch_size}, lr={lr:.0e})")
    print(f"{'='*50}")

    groups = subject_ids.copy()
    gkf = GroupKFold(n_splits=N_SPLITS)
    _, n_channels, n_times = X.shape

    fold_best_rows = []
    epoch_history_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"\n----- Fold {fold_idx} | {config_name} -----")

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

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = EEGNetLSTM(
            n_channels=n_channels,
            n_times=n_times,
            dropout=DROPOUT,
            lstm_hidden=LSTM_HIDDEN,
        ).to(DEVICE)

        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        best_bacc = -np.inf
        best_epoch_row = None

        # Early stopping state
        best_val_loss = np.inf
        no_improve_count = 0
        stopped_epoch = None

        for epoch in range(1, N_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss = compute_val_loss(model, test_loader, criterion)
            metrics = evaluate(model, test_loader)

            epoch_row = {
                "config_name": config_name,
                "batch_size": batch_size,
                "learning_rate": lr,
                "fold": fold_idx,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
            epoch_history_rows.append(epoch_row)

            if metrics["balanced_accuracy"] > best_bacc:
                best_bacc = metrics["balanced_accuracy"]
                best_epoch_row = epoch_row.copy()

            # Early stopping on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train={train_loss:.4f} | "
                    f"val={val_loss:.4f} | "
                    f"acc={metrics['accuracy']:.4f} | "
                    f"bacc={metrics['balanced_accuracy']:.4f} | "
                    f"auc={metrics['auc']:.4f}"
                )

            if no_improve_count >= EARLY_STOPPING_PATIENCE:
                stopped_epoch = epoch
                print(f"Early stopping at epoch {epoch} (no val-loss improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

        best_epoch_row["stopped_epoch"] = stopped_epoch if stopped_epoch is not None else epoch
        best_epoch_row["early_stopped"] = stopped_epoch is not None
        fold_best_rows.append(best_epoch_row)

        print(
            f"Fold {fold_idx} best | epoch={best_epoch_row['epoch']} | "
            f"acc={best_epoch_row['accuracy']:.4f} | "
            f"bacc={best_epoch_row['balanced_accuracy']:.4f} | "
            f"auc={best_epoch_row['auc']:.4f}"
        )

    fold_best_df = pd.DataFrame(fold_best_rows)
    epoch_history_df = pd.DataFrame(epoch_history_rows)

    summary_row = {
        "config_name": config_name,
        "batch_size": batch_size,
        "learning_rate": lr,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "n_early_stopped_folds": int(fold_best_df["early_stopped"].sum()),
        "mean_stopped_epoch": fold_best_df["stopped_epoch"].mean(),
        "mean_best_epoch": fold_best_df["epoch"].mean(),
        "mean_accuracy": fold_best_df["accuracy"].mean(),
        "std_accuracy": fold_best_df["accuracy"].std(),
        "mean_balanced_accuracy": fold_best_df["balanced_accuracy"].mean(),
        "std_balanced_accuracy": fold_best_df["balanced_accuracy"].std(),
        "mean_f1": fold_best_df["f1"].mean(),
        "std_f1": fold_best_df["f1"].std(),
        "mean_auc": fold_best_df["auc"].mean(),
        "std_auc": fold_best_df["auc"].std(),
        "best_fold_balanced_accuracy": fold_best_df["balanced_accuracy"].max(),
        "best_fold_auc": fold_best_df["auc"].max(),
    }

    return fold_best_df, epoch_history_df, summary_row


# =========================
# Main
# =========================
def main():
    X, y, subject_ids, _, _, _ = load_dataset(DATASET_NAME)

    all_summary_rows = []

    for batch_size in BATCH_SIZES:
        for lr in LEARNING_RATES:
            config_name = make_config_name(batch_size, lr)

            fold_best_df, epoch_history_df, summary_row = run_one_config(
                X, y, subject_ids, batch_size, lr
            )
            all_summary_rows.append(summary_row)

            fold_best_path = OUTPUT_DIR / f"{DATASET_NAME}_{config_name}_fold_best.csv"
            epoch_history_path = OUTPUT_DIR / f"{DATASET_NAME}_{config_name}_epoch_history.csv"

            fold_best_df.to_csv(fold_best_path, index=False)
            epoch_history_df.to_csv(epoch_history_path, index=False)

            print(f"\nSaved: {fold_best_path.name}")
            print(f"Saved: {epoch_history_path.name}")

    summary_df = (
        pd.DataFrame(all_summary_rows)
        .sort_values(
            by=["mean_balanced_accuracy", "mean_auc"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary_path = OUTPUT_DIR / f"{DATASET_NAME}_grid_search_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved global summary to: {summary_path}")
    print("\nGrid search results (ranked by mean balanced accuracy):")
    print(summary_df[[
        "config_name", "batch_size", "learning_rate",
        "mean_balanced_accuracy", "std_balanced_accuracy",
        "mean_auc", "std_auc",
        "mean_stopped_epoch", "n_early_stopped_folds",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
