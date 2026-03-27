# Latest Modifications by Ali

This document describes all changes introduced in the `alis-version` branch.
The original files on `main` are untouched — every modification lives exclusively in this branch.

---

## Files Changed

| File | Type |
|---|---|
| `Model/run_pd_on_off_eegnet_lstm_groupkfold.py` | Modified |
| `Model/run_pd_on_off_eegnet_groupkfold.py` | Modified |
| `Different Configurations/Batch128_Insteadof64.py` | Rewritten |
| `Different Configurations/run_revised_bestfold_hunt_eegnet_lstm.py` | Rewritten |
| `Different Configurations/grid_search_lr_bs_eegnet_lstm.py` | **New file** |

---

## 1. Early Stopping

**Problem in the original code:**
Both model files trained for a fixed number of epochs (60 in the main models, 80 in the batch-128 file) with no mechanism to stop early. This wastes compute when the model has already converged and risks overfitting in later epochs.

**What was changed:**
- The hard epoch limit was raised to **200** (a soft ceiling — early stopping is the primary terminator).
- A patience counter monitors the **validation loss** after every epoch.
- If the validation loss does not improve for **7 consecutive epochs**, training stops immediately.
- A message is printed showing the epoch at which stopping occurred and the best validation epoch.
- The saved CSVs now include `stopped_epoch`, `early_stopped` (bool), and `best_val_epoch` columns so the stopping behaviour is fully traceable.

```python
EARLY_STOPPING_PATIENCE = 7

if val_loss < best_val_loss:
    best_val_loss = val_loss
    no_improve_count = 0
else:
    no_improve_count += 1

if no_improve_count >= EARLY_STOPPING_PATIENCE:
    print(f"Early stopping at epoch {epoch}")
    break
```

---

## 2. Learning Rate Adjustment

**Problem in the original code:**
All configurations used a learning rate of `1e-3`, which is relatively high and can cause the optimiser to overshoot the loss minimum, especially with the larger batch size of 128.

**What was changed:**
- The default learning rate in all files was reduced from `1e-3` → **`1e-4`**.
- The `BEST_CONFIG` in `Batch128_Insteadof64.py` was updated accordingly.
- The search space in `run_revised_bestfold_hunt_eegnet_lstm.py` was expanded to also include `1e-4` and `1e-5` so all three values can be compared directly.

---

## 3. Learning Rate Scheduler (`ReduceLROnPlateau`)

**Problem in the original code:**
The learning rate was fixed for the entire training run. Once the model reaches a good basin, a fixed LR can prevent fine-grained convergence.

**What was changed:**
A `ReduceLROnPlateau` scheduler was added to all rewritten files. It monitors the **validation loss** and halves the learning rate whenever no improvement is seen for 5 consecutive epochs, down to a minimum of `1e-6`.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
)
scheduler.step(val_loss)  # called after every epoch
```

The current learning rate is logged each epoch so its decay is visible in the printed output and saved epoch-history CSVs.

---

## 4. Best Model Checkpoint (Restore Best Weights)

**Problem in the original code:**
The code tracked the epoch with the highest balanced accuracy but never saved the actual model weights at that point. At the end of training the model in memory was the *last epoch's* model, not the best one. Final metrics were therefore reported on a potentially worse model.

**What was changed:**
Every time the validation loss improves, a deep copy of the model's state dictionary is saved in memory. After the training loop ends (either by early stopping or reaching the epoch cap), the **best weights are restored** before the test set is evaluated.

```python
best_weights = copy.deepcopy(model.state_dict())

if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_weights  = copy.deepcopy(model.state_dict())

# After training loop:
model.load_state_dict(best_weights)
test_metrics = evaluate(model, test_loader)
```

---

## 5. Leakage-Free Train / Validation / Test Split

**Problem in the original code (subtle but important):**
Early stopping and the LR scheduler were monitoring the **test fold's** loss. This means every decision about when to stop training and when to reduce the LR was influenced by test-set performance. Even though the model weights were never directly optimised on the test set, this still constitutes indirect data leakage and inflates reported metrics.

**What was changed:**
A strict three-way, **subject-level** split is now enforced:

```
All data
  │
  └─ GroupKFold (outer, by subject)
        │
        ├── TEST subjects ──────────────────────────────► evaluated ONCE,
        │                                                  with best weights,
        │                                                  after training ends
        └── TRAIN subjects
              │
              └─ Inner subject-level split (20 % of train subjects → VAL)
                    │
                    ├── VAL subjects ──► early stopping, ReduceLROnPlateau,
                    │                    best-weights checkpoint
                    └── INNER-TRAIN subjects
                            │
                            ├── normalization mean/std computed here ONLY
                            ├── applied to VAL and TEST without re-fitting
                            └── model trained here (+ augmentation)
```

Key guarantees:
- **No subject appears in more than one split** — enforced by three explicit `assert` checks.
- **Normalisation statistics** are computed from inner-train only, so val and test are normalised using only information available at training time.
- **The test fold is evaluated exactly once**, at the very end, with no influence on any training decision.

```python
def split_train_val_by_subject(train_idx, groups, val_ratio=0.20):
    rng = np.random.RandomState(RANDOM_STATE)
    train_subjects = np.unique(groups[train_idx])
    n_val = max(1, round(len(train_subjects) * val_ratio))
    perm = rng.permutation(len(train_subjects))
    val_subjects = set(train_subjects[perm[:n_val]])
    inner_mask = np.array([groups[i] not in val_subjects for i in train_idx])
    return train_idx[inner_mask], train_idx[~inner_mask]

# Three leakage assertions
assert not outer_train_subjects & test_subjects
assert not inner_subjects & val_subjects
assert not val_subjects & test_subjects
```

---

## 6. Bidirectional LSTM

**Problem in the original code:**
The LSTM processed the EEG time series in one direction only (forward). For a fixed-length epoch, the network at any given time step has no information about what comes later in the trial.

**What was changed:**
`bidirectional=True` was set in `EEGNetLSTM`. This runs two LSTM passes — one forward, one backward — and concatenates their outputs. Every time step now has context from both its past and its future within the epoch, which is appropriate for offline classification of fixed-length segments.

The classifier's input size was updated from `lstm_hidden` → `lstm_hidden * 2` to match the doubled output dimension.

```python
self.lstm = nn.LSTM(
    input_size=feat_dim,
    hidden_size=lstm_hidden,
    bidirectional=True,   # changed from False
    ...
)
lstm_out_size = lstm_hidden * 2
```

---

## 7. Temporal Attention

**Problem in the original code:**
After the LSTM, only the final time step's hidden state was used (`x[:, -1, :]`). All other time steps were discarded. For EEG classification, the most discriminative moment in a trial is not necessarily the last one.

**What was changed:**
A lightweight **additive attention** module was added. It learns a scalar score for each LSTM time step and produces a weighted sum (context vector) as the representation passed to the classifier. The model can therefore focus on whichever part of the trial is most informative.

```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, lstm_out):
        # lstm_out: [B, T, H]
        scores  = self.score(lstm_out).squeeze(-1)               # [B, T]
        weights = torch.softmax(scores, dim=1)                   # [B, T]
        context = (lstm_out * weights.unsqueeze(-1)).sum(dim=1)  # [B, H]
        return context
```

In the model's `forward` method, the last-step readout is replaced:

```python
# Before:
x = x[:, -1, :]

# After:
x = self.attention(x)   # attends over all T time steps
```

---

## 8. Gradient Clipping

**Problem in the original code:**
LSTMs are susceptible to exploding gradients, which can cause sudden large weight updates and destabilise training — especially at lower learning rates where training runs longer.

**What was changed:**
`clip_grad_norm_` with `max_norm=1.0` was added inside `train_one_epoch`, applied after `loss.backward()` and before `optimizer.step()`.

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 9. Gaussian Noise Augmentation

**Problem in the original code:**
No data augmentation was used. With a limited number of subjects, the model can memorise training examples rather than learning generalisable EEG patterns.

**What was changed:**
Small Gaussian noise is added to every training batch. The noise standard deviation (`AUG_NOISE_STD = 0.05`) is small enough not to corrupt the signal but large enough to act as a regulariser. Noise is **only applied during training** — validation and test loaders receive clean data.

```python
# Inside train_one_epoch — never in evaluate() or compute_loss()
X_batch = X_batch + torch.randn_like(X_batch) * noise_std
```

---

## 10. New File — Grid Search (`grid_search_lr_bs_eegnet_lstm.py`)

A new script was added to `Different Configurations/` that systematically searches over:

| Axis | Values |
|---|---|
| Batch size | 64, 128 |
| Learning rate | `1e-3`, `1e-4`, `1e-5` |

This gives **6 configurations** in total. Every configuration runs the full 5-fold GroupKFold cross-validation with all of the above improvements (early stopping, leakage-free split, bidirectional LSTM, temporal attention, scheduler, gradient clipping, augmentation).

Results are saved per configuration (fold-best CSV + epoch-history CSV) and a final ranked summary CSV is written, sorted by mean balanced accuracy across folds.

---

## Summary Table

| Change | Files affected |
|---|---|
| Early stopping (patience = 7, on val loss) | All 4 existing files |
| Learning rate lowered `1e-3` → `1e-4` | All 4 existing files |
| `ReduceLROnPlateau` scheduler | `Batch128`, `hunt` (rewritten files) |
| Best-weights checkpoint + restore | `Batch128`, `hunt` (rewritten files) |
| Leakage-free 3-way subject split | `Batch128`, `hunt` (rewritten files) |
| Bidirectional LSTM | `Batch128`, `hunt` (rewritten files) |
| Temporal attention | `Batch128`, `hunt` (rewritten files) |
| Gradient clipping | `Batch128`, `hunt` (rewritten files) |
| Gaussian noise augmentation | `Batch128`, `hunt` (rewritten files) |
| Grid search script | New file |
