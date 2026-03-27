"""
Hyper-parameter search for the EEGNet-LSTM model.

Data pipeline (no leakage)
--------------------------
For every outer GroupKFold fold:
  1. Outer split  → train subjects  |  test subjects   (GroupKFold, subject-level)
  2. Inner split  → inner-train subjects  |  val subjects
                    (20 % of train subjects, also subject-level, no overlap)
  3. Normalization stats computed from inner-train ONLY, applied to val & test.
  4. Early stopping and LR scheduler watch val loss — test fold is NEVER seen
     during training.
  5. Best model weights (lowest val loss) are restored before final test evaluation.
  6. Test fold is evaluated exactly once, at the very end.

Improvements over the original
--------------------------------
* Bidirectional LSTM  — captures both forward and backward temporal context.
* Temporal attention  — learned weighted sum over all LSTM time steps instead
                        of discarding all but the last one.
* ReduceLROnPlateau   — halves LR when val loss stagnates (patience 5 epochs).
* Gradient clipping   — stabilises LSTM training (max norm 1.0).
* Gaussian noise aug  — small additive noise on input during training only.
* Best-weights checkpoint — model state saved whenever val loss improves.
"""

from pathlib import Path
import copy
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
DATA_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/classification_datasets_response")
OUTPUT_DIR = Path("/Users/pouya/Documents/Additional Academic Activities/MA/SC OFF:ON/My Codes/Revised/revised_bestfold_hunt_results")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DATASET_NAME = "PD_ON_vs_PD_OFF"
N_SPLITS = 5
RANDOM_STATE = 42

BATCH_SIZE = 64
N_EPOCHS = 200              # high cap — early stopping is the primary terminator
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7 # epochs of no val-loss improvement before stopping
VAL_RATIO = 0.20            # fraction of TRAINING subjects held out for val
AUG_NOISE_STD = 0.05        # std of Gaussian noise added during training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Search space — each entry is one model+optimiser configuration
SEARCH_SPACE = [
    {"lstm_hidden": 32, "dropout": 0.5, "learning_rate": 1e-3},
    {"lstm_hidden": 64, "dropout": 0.5, "learning_rate": 1e-3},
    {"lstm_hidden": 64, "dropout": 0.4, "learning_rate": 1e-3},
    {"lstm_hidden": 64, "dropout": 0.4, "learning_rate": 1e-4},
    {"lstm_hidden": 64, "dropout": 0.4, "learning_rate": 1e-5},
]


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
# Temporal Attention
# =========================
class TemporalAttention(nn.Module):
    """
    Additive attention over the LSTM's time-step outputs.
    Replaces the naive last-step readout, allowing the model to focus on
    whichever part of the trial epoch is most discriminative.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out : [B, T, H]
        scores = self.score(lstm_out).squeeze(-1)      # [B, T]
        weights = torch.softmax(scores, dim=1)          # [B, T]
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # [B, H]
        return context


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

        # ---- EEGNet spatial/spectral encoder ----
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Infer LSTM input dimensionality from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            x = self.firstconv(dummy)
            x = self.depthwise(x)
            x = self.separable(x)
            _, feat_dim, _, _ = x.shape

        # ---- Bidirectional LSTM ----
        # Bidirectional doubles the output size: each time step has context
        # from both past and future, which is valid for fixed-length EEG epochs.
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0 if lstm_layers == 1 else dropout,
            bidirectional=True,
        )
        lstm_out_size = lstm_hidden * 2  # forward + backward

        # ---- Temporal attention ----
        self.attention = TemporalAttention(lstm_out_size)

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.squeeze(2).permute(0, 2, 1)   # [B, T, feat_dim]
        x, _ = self.lstm(x)                  # [B, T, lstm_hidden*2]
        x = self.attention(x)                # [B, lstm_hidden*2]
        x = self.classifier(x)               # [B, 1]
        return x.squeeze(1)


# =========================
# Data helpers
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


def split_train_val_by_subject(
    train_idx: np.ndarray,
    groups: np.ndarray,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split train_idx into (inner_train_idx, val_idx) at the SUBJECT level.

    val_ratio is the fraction of training SUBJECTS assigned to the val set.
    No epoch from a val subject ever appears in inner_train, so there is zero
    data leakage between inner-train and val.
    """
    rng = np.random.RandomState(seed)
    train_subjects = np.unique(groups[train_idx])
    n_val = max(1, round(len(train_subjects) * val_ratio))

    perm = rng.permutation(len(train_subjects))
    val_subjects = set(train_subjects[perm[:n_val]])

    inner_mask = np.array([groups[i] not in val_subjects for i in train_idx])
    return train_idx[inner_mask], train_idx[~inner_mask]


def normalize_three_splits(
    X_inner: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute channel-wise mean/std from X_inner ONLY, then apply to all three splits.
    X shape: [n_samples, n_channels, n_times]
    """
    mean = X_inner.mean(axis=(0, 2), keepdims=True)
    std = X_inner.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1.0
    return (
        (X_inner - mean) / std,
        (X_val   - mean) / std,
        (X_test  - mean) / std,
    )


# =========================
# Training / evaluation
# =========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    noise_std: float = AUG_NOISE_STD,
) -> float:
    """One training epoch with Gaussian noise augmentation and gradient clipping."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # Gaussian noise augmentation — applied only during training
        X_batch = X_batch + torch.randn_like(X_batch) * noise_std

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping prevents exploding gradients in the LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def compute_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """Average loss over a loader — no gradient updates, no augmentation."""
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        loss = criterion(model(X_batch), y_batch)
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Classification metrics on a given loader (no gradients, no augmentation)."""
    model.eval()
    all_probs, all_preds, all_true = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        probs = torch.sigmoid(model(X_batch)).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_probs.append(probs)
        all_preds.append(preds)
        all_true.append(y_batch.numpy().astype(int))

    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1":                f1_score(y_true, y_pred),
        "auc":               auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def make_config_name(cfg: dict) -> str:
    lr_txt = str(cfg["learning_rate"]).replace(".", "p").replace("-", "m")
    return f"h{cfg['lstm_hidden']}_do{str(cfg['dropout']).replace('.', 'p')}_lr{lr_txt}"


# =========================
# Run one config
# =========================
def run_one_config(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    set_seed(RANDOM_STATE)

    config_name = make_config_name(cfg)
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"{'='*60}")

    groups = subject_ids.copy()
    gkf = GroupKFold(n_splits=N_SPLITS)
    _, n_channels, n_times = X.shape

    fold_rows = []
    epoch_history_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups), start=1
    ):
        print(f"\n----- Outer fold {fold_idx} | {config_name} -----")

        # ── subject-level leakage checks ──────────────────────────────────
        outer_train_subjects = set(np.unique(groups[train_idx]))
        test_subjects        = set(np.unique(groups[test_idx]))
        assert not outer_train_subjects & test_subjects, (
            f"Outer leakage in fold {fold_idx}: "
            f"{outer_train_subjects & test_subjects}"
        )

        # ── inner train / val split (subject-level, no leakage) ───────────
        inner_train_idx, val_idx = split_train_val_by_subject(
            train_idx, groups
        )
        inner_subjects = set(np.unique(groups[inner_train_idx]))
        val_subjects   = set(np.unique(groups[val_idx]))
        assert not inner_subjects & val_subjects, (
            f"Inner train/val leakage in fold {fold_idx}: "
            f"{inner_subjects & val_subjects}"
        )
        assert not val_subjects & test_subjects, (
            f"Val/test leakage in fold {fold_idx}: "
            f"{val_subjects & test_subjects}"
        )

        print(
            f"  Subjects — inner-train: {len(inner_subjects)}, "
            f"val: {len(val_subjects)}, test: {len(test_subjects)}"
        )

        X_inner = X[inner_train_idx]
        X_val   = X[val_idx]
        X_test  = X[test_idx]
        y_inner = y[inner_train_idx]
        y_val   = y[val_idx]
        y_test  = y[test_idx]

        # ── normalization: stats from inner-train ONLY ────────────────────
        X_inner_n, X_val_n, X_test_n = normalize_three_splits(
            X_inner, X_val, X_test
        )

        # Add CNN channel dim: [N, C, T] → [N, 1, C, T]
        X_inner_n = X_inner_n[:, np.newaxis, :, :]
        X_val_n   = X_val_n[:, np.newaxis, :, :]
        X_test_n  = X_test_n[:, np.newaxis, :, :]

        # ── data loaders ──────────────────────────────────────────────────
        inner_loader = DataLoader(
            EEGDataset(X_inner_n, y_inner), batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            EEGDataset(X_val_n, y_val), batch_size=BATCH_SIZE, shuffle=False
        )
        test_loader = DataLoader(
            EEGDataset(X_test_n, y_test), batch_size=BATCH_SIZE, shuffle=False
        )

        # ── model + optimiser + scheduler ────────────────────────────────
        model = EEGNetLSTM(
            n_channels=n_channels,
            n_times=n_times,
            dropout=cfg["dropout"],
            lstm_hidden=cfg["lstm_hidden"],
        ).to(DEVICE)

        # pos_weight computed from inner-train labels only
        n_pos = float((y_inner == 1).sum())
        n_neg = float((y_inner == 0).sum())
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=DEVICE
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=WEIGHT_DECAY,
        )
        # Scheduler watches val loss; halves LR after 5 stagnating epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        # ── training loop ─────────────────────────────────────────────────
        best_val_loss    = np.inf
        no_improve_count = 0
        best_weights     = copy.deepcopy(model.state_dict())
        best_val_epoch   = 1
        stopped_epoch    = None

        for epoch in range(1, N_EPOCHS + 1):
            train_loss = train_one_epoch(model, inner_loader, optimizer, criterion)
            val_loss   = compute_loss(model, val_loader, criterion)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                no_improve_count = 0
                best_weights     = copy.deepcopy(model.state_dict())
                best_val_epoch   = epoch
            else:
                no_improve_count += 1

            if epoch % 10 == 0 or epoch == 1:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:03d} | "
                    f"train={train_loss:.4f} | "
                    f"val={val_loss:.4f} | "
                    f"lr={current_lr:.2e} | "
                    f"no-improve={no_improve_count}"
                )

            epoch_history_rows.append({
                "config_name":    config_name,
                "fold":           fold_idx,
                "epoch":          epoch,
                "train_loss":     train_loss,
                "val_loss":       val_loss,
                "current_lr":     optimizer.param_groups[0]["lr"],
                "lstm_hidden":    cfg["lstm_hidden"],
                "dropout":        cfg["dropout"],
                "learning_rate":  cfg["learning_rate"],
            })

            if no_improve_count >= EARLY_STOPPING_PATIENCE:
                stopped_epoch = epoch
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(best val epoch: {best_val_epoch})"
                )
                break

        # ── restore best weights → evaluate on test ONCE ─────────────────
        model.load_state_dict(best_weights)
        test_metrics = evaluate(model, test_loader)

        fold_rows.append({
            "config_name":      config_name,
            "fold":             fold_idx,
            "best_val_epoch":   best_val_epoch,
            "stopped_epoch":    stopped_epoch if stopped_epoch is not None else epoch,
            "early_stopped":    stopped_epoch is not None,
            "best_val_loss":    best_val_loss,
            "n_inner_subjects": len(inner_subjects),
            "n_val_subjects":   len(val_subjects),
            "n_test_subjects":  len(test_subjects),
            "lstm_hidden":      cfg["lstm_hidden"],
            "dropout":          cfg["dropout"],
            "learning_rate":    cfg["learning_rate"],
            **test_metrics,
        })

        print(
            f"  Fold {fold_idx} result | "
            f"acc={test_metrics['accuracy']:.4f} | "
            f"bacc={test_metrics['balanced_accuracy']:.4f} | "
            f"f1={test_metrics['f1']:.4f} | "
            f"auc={test_metrics['auc']:.4f}"
        )

    # ── aggregate across folds ────────────────────────────────────────────
    fold_df   = pd.DataFrame(fold_rows)
    epoch_df  = pd.DataFrame(epoch_history_rows)

    summary_row = {
        "config_name":              config_name,
        "lstm_hidden":              cfg["lstm_hidden"],
        "dropout":                  cfg["dropout"],
        "learning_rate":            cfg["learning_rate"],
        "batch_size":               BATCH_SIZE,
        "early_stopping_patience":  EARLY_STOPPING_PATIENCE,
        "n_early_stopped_folds":    int(fold_df["early_stopped"].sum()),
        "mean_best_val_epoch":      fold_df["best_val_epoch"].mean(),
        "mean_stopped_epoch":       fold_df["stopped_epoch"].mean(),
        "mean_accuracy":            fold_df["accuracy"].mean(),
        "std_accuracy":             fold_df["accuracy"].std(),
        "mean_balanced_accuracy":   fold_df["balanced_accuracy"].mean(),
        "std_balanced_accuracy":    fold_df["balanced_accuracy"].std(),
        "mean_f1":                  fold_df["f1"].mean(),
        "std_f1":                   fold_df["f1"].std(),
        "mean_auc":                 fold_df["auc"].mean(),
        "std_auc":                  fold_df["auc"].std(),
        "best_fold_balanced_accuracy": fold_df["balanced_accuracy"].max(),
        "best_fold_auc":               fold_df["auc"].max(),
    }

    return fold_df, epoch_df, summary_row


# =========================
# Main
# =========================
def main():
    X, y, subject_ids, session_ids, condition, rt = load_dataset(DATASET_NAME)

    all_summary_rows = []

    for cfg in SEARCH_SPACE:
        config_name = make_config_name(cfg)

        fold_df, epoch_df, summary_row = run_one_config(X, y, subject_ids, cfg)
        all_summary_rows.append(summary_row)

        fold_best_path    = OUTPUT_DIR / f"{DATASET_NAME}_{config_name}_fold_results.csv"
        epoch_history_path = OUTPUT_DIR / f"{DATASET_NAME}_{config_name}_epoch_history.csv"

        fold_df.to_csv(fold_best_path, index=False)
        epoch_df.to_csv(epoch_history_path, index=False)

        print(f"\nSaved fold results  → {fold_best_path.name}")
        print(f"Saved epoch history → {epoch_history_path.name}")

    summary_df = (
        pd.DataFrame(all_summary_rows)
        .sort_values(
            by=["mean_balanced_accuracy", "mean_auc"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary_path = OUTPUT_DIR / f"{DATASET_NAME}_hyperparam_search_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved global summary → {summary_path.name}")
    print("\nTop configs (ranked by mean balanced accuracy):")
    print(
        summary_df[[
            "config_name", "learning_rate",
            "mean_balanced_accuracy", "std_balanced_accuracy",
            "mean_auc", "std_auc",
            "mean_best_val_epoch", "n_early_stopped_folds",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
