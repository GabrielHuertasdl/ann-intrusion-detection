"""
Shared utilities: ANN model, training loop, evaluation, alert system,
streaming simulation, and output I/O helpers.

Model variants:
  - build_model_baseline() — Part A human-guided design
  - build_model_improved() — Part B GenAI-assisted design
"""

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.common.preprocessing import CLASSES, N_CLS, SEED

torch.manual_seed(SEED)
np.random.seed(SEED)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ── ANN architecture ─────────────────────────────────────────────────────────

class ANN(nn.Module):
    """Fully-connected ANN with optional BatchNorm and variable dropout."""

    def __init__(self, input_dim: int, hidden_dims: list[int],
                 dropouts: list[float], use_bn: bool = False):
        super().__init__()
        layers = []
        in_d = input_dim
        for h, d in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(in_d, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if d > 0:
                layers.append(nn.Dropout(d))
            in_d = h
        layers.append(nn.Linear(in_d, N_CLS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # raw logits — softmax applied at inference


def build_model_baseline(input_dim: int) -> ANN:
    """
    Part A — Human-guided baseline.
    Architecture: 128 → 64 → 32
    Dropout: uniform 0.3 on first two layers
    Rationale: compact, interpretable; standard dropout regularisation;
               sufficient capacity for well-represented classes.
    """
    return ANN(input_dim,
               hidden_dims=[128, 64, 32],
               dropouts=[0.3, 0.3, 0.0],
               use_bn=False)


def build_model_improved(input_dim: int) -> ANN:
    """
    Part B — GenAI-assisted improved model.
    Architecture: 256 → 128 → 64
    Additions: BatchNorm1d, graduated dropout (0.4/0.3/0.2), L2 via weight_decay
    GenAI suggestions adopted:
      - Wider first layer (256) for better minority-class representation
      - BatchNorm for training stability on mixed-scale features
      - Graduated dropout (stronger near input, gentler near output)
      - Lower LR (5e-4) + smaller batch (256) to complement L2
    See prompts/prompts_used_part_b.txt for the full prompt log.
    """
    return ANN(input_dim,
               hidden_dims=[256, 128, 64],
               dropouts=[0.4, 0.3, 0.2],
               use_bn=True)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model: ANN, X_tr: np.ndarray, y_tr: np.ndarray,
          batch_size: int = 512, max_epochs: int = 50,
          lr: float = 1e-3, weight_decay: float = 0.0,
          patience: int = 5, log_path: Path | None = None):
    """
    Adam + ReduceLROnPlateau + early stopping on 15% val split.
    Returns trained model and list of (train_loss, val_loss) per epoch.
    """
    X_t = torch.tensor(X_tr)
    y_t = torch.tensor(y_tr)
    ds  = TensorDataset(X_t, y_t)

    val_n = int(0.15 * len(ds))
    tr_n  = len(ds) - val_n
    tr_ds, val_ds = random_split(
        ds, [tr_n, val_n],
        generator=torch.Generator().manual_seed(SEED))

    tr_ld  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=2048, shuffle=False)

    opt       = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=2, factor=0.5)

    best_val, best_state, no_imp = float("inf"), None, 0
    history = []

    log_lines = [f"epoch,train_loss,val_loss"]

    for ep in range(1, max_epochs + 1):
        model.train()
        tr_loss = sum(
            _batch_loss(model, xb, yb, opt, criterion)
            for xb, yb in tr_ld
        ) / tr_n

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= val_n
        scheduler.step(val_loss)

        history.append((tr_loss, val_loss))
        log_lines.append(f"{ep},{tr_loss:.6f},{val_loss:.6f}")
        print(f"  Epoch {ep:>3}/{max_epochs}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stopping at epoch {ep} (patience={patience})")
                break

    model.load_state_dict(best_state)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_lines) + "\n")

    return model, history


def _batch_loss(model, xb, yb, opt, criterion):
    opt.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()
    opt.step()
    return loss.item() * len(xb)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model: ANN, X: np.ndarray, batch_size: int = 2048):
    """Return (probs, y_pred, conf) arrays."""
    model.eval()
    ld = DataLoader(TensorDataset(torch.tensor(X)),
                    batch_size=batch_size, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for (xb,) in ld:
            probs_list.append(torch.softmax(model(xb), dim=1).numpy())
    probs  = np.vstack(probs_list)
    y_pred = np.argmax(probs, axis=1)
    conf   = np.max(probs, axis=1)
    return probs, y_pred, conf


# ── Multiclass + binary evaluation ───────────────────────────────────────────

def eval_metrics(model: ANN, X_te: np.ndarray, y_te: np.ndarray,
                 label: str = "Model"):
    """
    Compute and print multiclass + binary metrics.
    Returns (probs, y_pred, conf, bin_stats dict, report_dict).
    """
    probs, y_pred, conf = predict(model, X_te)

    sep = "=" * 64
    print(f"\n{sep}\n  MULTICLASS METRICS — {label}\n{sep}")

    acc       = accuracy_score(y_te, y_pred)
    macro_f1  = f1_score(y_te, y_pred, average="macro",   zero_division=0)
    weighted_f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}\n")
    print(classification_report(y_te, y_pred,
                                 labels=list(range(N_CLS)),
                                 target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(y_te, y_pred, labels=list(range(N_CLS)))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    # Binary view
    y_bin_t = (y_te   != 0).astype(int)
    y_bin_p = (y_pred != 0).astype(int)
    TP = int(((y_bin_t == 1) & (y_bin_p == 1)).sum())
    TN = int(((y_bin_t == 0) & (y_bin_p == 0)).sum())
    FP = int(((y_bin_t == 0) & (y_bin_p == 1)).sum())
    FN = int(((y_bin_t == 1) & (y_bin_p == 0)).sum())
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec  = TP / (TP + FN) if TP + FN else 0.0
    f1b  = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr  = FP / (FP + TN) if FP + TN else 0.0

    print(f"\n--- Binary View ---")
    print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"  Precision={prec:.4f}  Recall/TPR={rec:.4f}  "
          f"F1={f1b:.4f}  FPR={fpr:.4f}")

    # Confidence summary
    correct = (y_pred == y_te)
    print(f"\n--- Confidence Distribution ---")
    print(f"  Correct   mean={conf[correct].mean():.4f}  "
          f"median={np.median(conf[correct]):.4f}")
    print(f"  Incorrect mean={conf[~correct].mean():.4f}  "
          f"median={np.median(conf[~correct]):.4f}")

    bin_stats = dict(acc=acc, macro_f1=macro_f1, weighted_f1=weighted_f1,
                     prec=prec, rec=rec, f1=f1b, fpr=fpr,
                     TP=TP, TN=TN, FP=FP, FN=FN)
    return probs, y_pred, conf, bin_stats, cm


# ── Alert system ──────────────────────────────────────────────────────────────

def alert_level(pred_class: int, confidence: float) -> str:
    """
    Four-tier confidence-gated alerting.
    Thresholds (0.6, 0.8) chosen as balanced operating point:
      - 0.8 → CRITICAL: high-confidence attack → auto-block
      - 0.6 → SUSPICIOUS: medium confidence → human review
      - <0.6 → LOW_CONF_ATTACK: ambiguous → log and review
    """
    if pred_class == 0:
        return "NO_ALERT"
    if confidence >= 0.8:
        return "CRITICAL_ALERT"
    if confidence >= 0.6:
        return "SUSPICIOUS"
    return "LOW_CONF_ATTACK"


def eval_alerts(y_te: np.ndarray, y_pred: np.ndarray,
                conf: np.ndarray, label: str = "Model"):
    """Compute alert tier distribution and FP/FN alert metrics."""
    alerts = np.array([alert_level(int(p), float(c))
                       for p, c in zip(y_pred, conf)])
    levels = ["NO_ALERT", "LOW_CONF_ATTACK", "SUSPICIOUS", "CRITICAL_ALERT"]
    n = len(alerts)

    sep = "=" * 64
    print(f"\n{sep}\n  ALERT SYSTEM METRICS — {label}\n{sep}")
    for lv in levels:
        cnt = (alerts == lv).sum()
        print(f"  {lv:<22}: {cnt:>7}  ({100*cnt/n:.2f}%)")

    is_normal = (y_te == 0)
    is_attack = ~is_normal
    FP_a = int(((is_normal) & (alerts != "NO_ALERT")).sum())
    FN_h = int(((is_attack) & (alerts == "NO_ALERT")).sum())
    FN_s = int(((is_attack) & (alerts == "LOW_CONF_ATTACK")).sum())

    print(f"\n  FP  (Normal→alert)         : {FP_a:>7} "
          f"({100*FP_a/is_normal.sum():.2f}% of normals)")
    print(f"  FN hard (Attack→NO_ALERT)  : {FN_h:>7} "
          f"({100*FN_h/is_attack.sum():.2f}% of attacks)")
    print(f"  FN soft (Attack→LOW_CONF)  : {FN_s:>7} "
          f"({100*FN_s/is_attack.sum():.2f}% of attacks)")

    alert_stats = {
        "FP_alerts": FP_a, "FN_hard": FN_h, "FN_soft": FN_s,
        **{lv: int((alerts == lv).sum()) for lv in levels},
    }
    return alerts, alert_stats


# ── Streaming simulation ──────────────────────────────────────────────────────

def simulate_stream(model: ANN, X_te: np.ndarray, y_te: np.ndarray,
                    n_samples: int = 1000, label: str = "Model"):
    """Random 1000-sample stream simulation, returns event log DataFrame."""
    rng  = np.random.default_rng(SEED)
    idxs = rng.choice(len(X_te), size=n_samples, replace=False)
    _, preds, confs = predict(model, X_te[idxs])

    log = [
        {
            "stream_pos":  i,
            "orig_idx":    int(orig),
            "true_label":  CLASSES[y_te[orig]],
            "pred_label":  CLASSES[pred],
            "confidence":  round(float(c), 4),
            "alert":       alert_level(int(pred), float(c)),
        }
        for i, (orig, pred, c) in enumerate(zip(idxs, preds, confs))
    ]
    df = pd.DataFrame(log)

    sep = "=" * 64
    print(f"\n{sep}\n  STREAM SIMULATION ({n_samples} samples) — {label}\n{sep}")
    print(df.head(20).to_string(index=False))

    print("\n  Stream alert summary:")
    for lv in ["NO_ALERT", "LOW_CONF_ATTACK", "SUSPICIOUS", "CRITICAL_ALERT"]:
        cnt = (df["alert"] == lv).sum()
        print(f"    {lv:<22}: {cnt:>4}  ({100*cnt/n_samples:.1f}%)")

    s_FN = ((df["true_label"] != "Normal") & (df["alert"] == "NO_ALERT")).sum()
    s_FP = ((df["true_label"] == "Normal") & (df["alert"] != "NO_ALERT")).sum()
    print(f"  Stream FP: {s_FP}  |  Stream FN (Attack→NO_ALERT): {s_FN}")
    return df


# ── Output I/O helpers ────────────────────────────────────────────────────────

def save_metrics_csv(bin_stats: dict, alert_stats: dict,
                     label: str, path: Path):
    """Export combined scalar metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"model": label, **bin_stats, **alert_stats}
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"  ✓ metrics saved → {path}")


def save_training_curves(history: list, label: str, path: Path):
    """Plot and save training/validation loss curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs  = range(1, len(history) + 1)
    tr_loss = [h[0] for h in history]
    vl_loss = [h[1] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, tr_loss, label="Train loss", linewidth=2)
    ax.plot(epochs, vl_loss, label="Val loss",   linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Training curves — {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ training curves → {path}")


def save_confusion_matrix(cm: np.ndarray, label: str, path: Path):
    """Plot and save confusion matrix heatmap (normalised by true class)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalised) — {label}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ confusion matrix → {path}")


def save_confidence_plot(y_te: np.ndarray, y_pred: np.ndarray,
                         conf: np.ndarray, label: str, path: Path):
    """Histogram of prediction confidence — correct vs incorrect."""
    path.parent.mkdir(parents=True, exist_ok=True)
    correct = conf[y_pred == y_te]
    wrong   = conf[y_pred != y_te]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correct, bins=40, alpha=0.7, label=f"Correct (n={len(correct):,})",
            color="steelblue", density=True)
    ax.hist(wrong,   bins=40, alpha=0.7, label=f"Incorrect (n={len(wrong):,})",
            color="tomato", density=True)
    ax.axvline(0.6, linestyle="--", color="orange", label="Threshold 0.6")
    ax.axvline(0.8, linestyle="--", color="red",    label="Threshold 0.8")
    ax.set_xlabel("Softmax confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence distribution — {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ confidence plot → {path}")


def save_alert_distribution(alert_stats: dict, label: str, path: Path):
    """Bar chart of alert tier counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    levels = ["NO_ALERT", "LOW_CONF_ATTACK", "SUSPICIOUS", "CRITICAL_ALERT"]
    counts = [alert_stats[lv] for lv in levels]
    colours = ["#4caf50", "#ff9800", "#ff5722", "#f44336"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(levels, counts, color=colours)
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)
    ax.set_ylabel("Sample count")
    ax.set_title(f"Alert tier distribution — {label}")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ alert distribution → {path}")


def save_comparison_chart(stats_a: dict, stats_b: dict,
                          label_a: str, label_b: str, path: Path):
    """Side-by-side bar comparison of key metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["acc", "macro_f1", "prec", "rec", "f1"]
    names   = ["Accuracy", "Macro-F1", "Precision", "Recall", "Binary-F1"]
    vals_a  = [stats_a[m] for m in metrics]
    vals_b  = [stats_b[m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, vals_a, w, label=label_a, color="steelblue")
    b2 = ax.bar(x + w/2, vals_b, w, label=label_b, color="coral")
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Model comparison: {label_a} vs {label_b}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ comparison chart → {path}")
