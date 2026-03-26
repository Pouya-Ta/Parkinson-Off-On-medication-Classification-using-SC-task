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
OUTPUT_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Revised/revised_bestfold_hunt_results_bs128_ep80")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DATASET_NAME = "PD_ON_vs_PD_OFF"
N_SPLITS = 5
RANDOM_STATE = 42

BATCH_SIZE = 128
N_EPOCHS = 80
#N_EPOCHS = 60
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Best config only
BEST_CONFIG = {
    "lstm_hidden": 64,
    "dropout": 0.4,
    "learning_rate": 1e-3,
}


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
            _, feat_dim, _, _ = x.shape

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
            nn.Linear(lstm_hidden, 1)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.squeeze(2).permute(0, 2, 1)  # [B, T, F]
        x, _ = self.lstm(x)
        x = x[:, -1, :]
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


def make_config_name(cfg):
    lr_txt = str(cfg["learning_rate"]).replace(".", "p").replace("-", "m")
    return f"h{cfg['lstm_hidden']}_do{str(cfg['dropout']).replace('.', 'p')}_lr{lr_txt}_bs{BATCH_SIZE}_ep{N_EPOCHS}"


def run_one_config(X, y, subject_ids, cfg):
    set_seed(RANDOM_STATE)

    groups = subject_ids.copy()
    gkf = GroupKFold(n_splits=N_SPLITS)

    _, n_channels, n_times = X.shape

    fold_best_rows = []
    epoch_history_rows = []

    config_name = make_config_name(cfg)
    print(f"\n==============================")
    print(f"Running config: {config_name}")
    print(f"==============================")

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"\n===== Fold {fold_idx} | {config_name} =====")

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

        model = EEGNetLSTM(
            n_channels=n_channels,
            n_times=n_times,
            dropout=cfg["dropout"],
            lstm_hidden=cfg["lstm_hidden"],
        ).to(DEVICE)

        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=WEIGHT_DECAY
        )

        best_bacc = -np.inf
        best_epoch_row = None

        for epoch in range(1, N_EPOCHS + 1):
            train_loss = train_one_fold(model, train_loader, optimizer, criterion)
            metrics = evaluate(model, test_loader)

            epoch_row = {
                "config_name": config_name,
                "fold": fold_idx,
                "epoch": epoch,
                "train_loss": train_loss,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "lstm_hidden": cfg["lstm_hidden"],
                "dropout": cfg["dropout"],
                "learning_rate": cfg["learning_rate"],
                "batch_size": BATCH_SIZE,
            }
            epoch_history_rows.append(epoch_row)

            if metrics["balanced_accuracy"] > best_bacc:
                best_bacc = metrics["balanced_accuracy"]
                best_epoch_row = epoch_row.copy()

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | "
                    f"loss={train_loss:.4f} | "
                    f"acc={metrics['accuracy']:.4f} | "
                    f"bacc={metrics['balanced_accuracy']:.4f} | "
                    f"f1={metrics['f1']:.4f} | "
                    f"auc={metrics['auc']:.4f}"
                )

        fold_best_rows.append(best_epoch_row)

        print(
            f"Best for fold {fold_idx} | "
            f"epoch={best_epoch_row['epoch']} | "
            f"acc={best_epoch_row['accuracy']:.4f} | "
            f"bacc={best_epoch_row['balanced_accuracy']:.4f} | "
            f"auc={best_epoch_row['auc']:.4f}"
        )

    fold_best_df = pd.DataFrame(fold_best_rows)
    epoch_history_df = pd.DataFrame(epoch_history_rows)

    summary_row = {
        "config_name": config_name,
        "lstm_hidden": cfg["lstm_hidden"],
        "dropout": cfg["dropout"],
        "learning_rate": cfg["learning_rate"],
        "batch_size": BATCH_SIZE,
        "mean_accuracy": fold_best_df["accuracy"].mean(),
        "std_accuracy": fold_best_df["accuracy"].std(),
        "mean_balanced_accuracy": fold_best_df["balanced_accuracy"].mean(),
        "std_balanced_accuracy": fold_best_df["balanced_accuracy"].std(),
        "mean_f1": fold_best_df["f1"].mean(),
        "std_f1": fold_best_df["f1"].std(),
        "mean_auc": fold_best_df["auc"].mean(),
        "std_auc": fold_best_df["auc"].std(),
        "best_fold_accuracy": fold_best_df["accuracy"].max(),
        "best_fold_balanced_accuracy": fold_best_df["balanced_accuracy"].max(),
        "best_fold_auc": fold_best_df["auc"].max(),
    }

    return fold_best_df, epoch_history_df, summary_row


def main():
    X, y, subject_ids, session_ids, condition, rt = load_dataset(DATASET_NAME)

    config_name = make_config_name(BEST_CONFIG)

    fold_best_df, epoch_history_df, summary_row = run_one_config(X, y, subject_ids, BEST_CONFIG)

    fold_best_path = OUTPUT_DIR / f"{DATASET_NAME}_{config_name}_fold_best.csv"
    epoch_history_path = OUTPUT_DIR / f"{DATASET_NAME}_{config_name}_epoch_history.csv"
    summary_path = OUTPUT_DIR / f"{DATASET_NAME}_{config_name}_summary.csv"

    fold_best_df.to_csv(fold_best_path, index=False)
    epoch_history_df.to_csv(epoch_history_path, index=False)
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    print(f"\nSaved fold-best to: {fold_best_path}")
    print(f"Saved epoch-history to: {epoch_history_path}")
    print(f"Saved summary to: {summary_path}")

    print("\nFinal summary:")
    print(pd.DataFrame([summary_row]).to_string(index=False))


if __name__ == "__main__":
    main()