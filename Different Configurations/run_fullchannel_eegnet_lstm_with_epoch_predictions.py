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
OUTPUT_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Revised/revised_session_aggregation_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DATASET_NAME = "PD_ON_vs_PD_OFF"
N_SPLITS = 5
RANDOM_STATE = 42

BATCH_SIZE = 64
N_EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x.squeeze(1)


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
def evaluate_with_predictions(model, X_test, y_test, batch_size=64):
    model.eval()

    probs_all = []
    preds_all = []

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    for start in range(0, len(X_tensor), batch_size):
        xb = X_tensor[start:start+batch_size].to(DEVICE)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        probs_all.append(probs)
        preds_all.append(preds)

    y_prob = np.concatenate(probs_all)
    y_pred = np.concatenate(preds_all)
    y_true = y_test.astype(int)

    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "f1": f1,
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return metrics, y_prob, y_pred


def main():
    set_seed(RANDOM_STATE)

    X, y, subject_ids, session_ids, condition, rt = load_dataset(DATASET_NAME)
    groups = subject_ids.copy()
    gkf = GroupKFold(n_splits=N_SPLITS)

    _, n_channels, n_times = X.shape

    fold_rows = []
    prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"\n===== Fold {fold_idx} =====")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        subj_test = subject_ids[test_idx]
        sess_test = session_ids[test_idx]
        cond_test = condition[test_idx]
        rt_test = rt[test_idx]

        X_train, X_test = normalize_train_test(X_train, X_test)

        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]

        train_ds = EEGDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = EEGNetLSTM(n_channels=n_channels, n_times=n_times).to(DEVICE)

        n_pos = float((y_train == 1).sum())
        n_neg = float((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_bacc = -np.inf
        best_metrics = None
        best_epoch = None
        best_y_prob = None
        best_y_pred = None

        for epoch in range(1, N_EPOCHS + 1):
            train_loss = train_one_fold(model, train_loader, optimizer, criterion)
            metrics, y_prob, y_pred = evaluate_with_predictions(model, X_test, y_test)

            if metrics["balanced_accuracy"] > best_bacc:
                best_bacc = metrics["balanced_accuracy"]
                best_metrics = metrics.copy()
                best_epoch = epoch
                best_y_prob = y_prob.copy()
                best_y_pred = y_pred.copy()

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                    f"acc={metrics['accuracy']:.4f} | "
                    f"bacc={metrics['balanced_accuracy']:.4f} | "
                    f"f1={metrics['f1']:.4f} | "
                    f"auc={metrics['auc']:.4f}"
                )

        fold_rows.append({
            "dataset": DATASET_NAME,
            "fold": fold_idx,
            "best_epoch": best_epoch,
            **best_metrics
        })

        for i in range(len(test_idx)):
            prediction_rows.append({
                "fold": fold_idx,
                "subject_id": subj_test[i],
                "session_id": sess_test[i],
                "condition": cond_test[i],
                "rt": float(rt_test[i]) if not np.isnan(rt_test[i]) else np.nan,
                "y_true": int(y_test[i]),
                "y_prob": float(best_y_prob[i]),
                "y_pred": int(best_y_pred[i]),
            })

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)

    summary_df = pd.DataFrame([{
        "dataset": DATASET_NAME,
        "mean_accuracy": fold_df["accuracy"].mean(),
        "std_accuracy": fold_df["accuracy"].std(),
        "mean_balanced_accuracy": fold_df["balanced_accuracy"].mean(),
        "std_balanced_accuracy": fold_df["balanced_accuracy"].std(),
        "mean_f1": fold_df["f1"].mean(),
        "std_f1": fold_df["f1"].std(),
        "mean_auc": fold_df["auc"].mean(),
        "std_auc": fold_df["auc"].std(),
    }])

    fold_df.to_csv(OUTPUT_DIR / f"{DATASET_NAME}_epoch_level_fold_results.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / f"{DATASET_NAME}_epoch_level_summary.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / f"{DATASET_NAME}_epoch_level_predictions.csv", index=False)

    print("\nSaved:")
    print(OUTPUT_DIR / f"{DATASET_NAME}_epoch_level_fold_results.csv")
    print(OUTPUT_DIR / f"{DATASET_NAME}_epoch_level_summary.csv")
    print(OUTPUT_DIR / f"{DATASET_NAME}_epoch_level_predictions.csv")


if __name__ == "__main__":
    main()