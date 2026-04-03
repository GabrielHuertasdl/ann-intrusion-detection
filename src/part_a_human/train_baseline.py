"""
Part A — Human-Guided ANN Design
=================================
Course  : 6009CMD Artificial Neural Networks, Coventry University
Dataset : UNSW-NB15 (Moustafa & Slay, 2015)
Task    : 10-class network intrusion detection

This script was designed independently by the student without any
Generative AI assistance.  Every architectural decision, preprocessing
choice, training configuration, and evaluation methodology was reasoned
from first principles:

  * Architecture  : 128→64→32 — small, interpretable baseline sufficient
                    for a well-studied tabular benchmark
  * Dropout 0.3   : standard regularisation; prevents co-adaptation without
                    overly restricting capacity on the 175k training set
  * Adam lr=1e-3  : default Adam starting point for tabular data
  * Batch 512     : large enough for stable CPU gradients; fits training set
  * EarlyStopping : patience=5 on val_loss eliminates manual epoch tuning

Run
---
  cd /path/to/ANN
  source venv/bin/activate
  python -m src.part_a_human.train_baseline

Outputs (all saved to outputs/)
--------------------------------
  figures/eda_class_distribution.png
  figures/eda_imbalance_ratio.png
  figures/training_curves_baseline.png
  figures/confusion_matrix_baseline.png
  figures/confidence_distribution_baseline.png
  figures/alert_distribution_baseline.png
  tables/metrics_baseline.csv
  tables/stream_log_baseline.csv
  models/baseline_model.pt
  models/scaler.pkl
  logs/training_baseline.csv
"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# Allow running as `python -m src.part_a_human.train_baseline`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.preprocessing import (
    CLASSES, N_CLS, load_data, preprocess,
)
from src.common.utils import (
    build_model_baseline,
    eval_alerts,
    eval_metrics,
    predict,
    save_alert_distribution,
    save_confidence_plot,
    save_confusion_matrix,
    save_metrics_csv,
    save_training_curves,
    simulate_stream,
    train,
)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

OUT = PROJECT_ROOT / "outputs"
FIGS   = OUT / "figures"
TABLES = OUT / "tables"
MODELS = OUT / "models"
LOGS   = OUT / "logs"
for d in [FIGS, TABLES, MODELS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)


# ── EDA ────────────────────────────────────────────────────────────────────

def run_eda(tr_raw: pd.DataFrame, te_raw: pd.DataFrame,
            feature_names: list[str]) -> None:
    """
    Exploratory Data Analysis — performed before any modelling.
    Analyses: class distribution, imbalance ratios, feature types,
              missing values.  All figures saved to outputs/figures/.
    """
    from src.common.preprocessing import TARGET, CAT_COLS

    print("\n" + "=" * 64)
    print("  EDA — UNSW-NB15 DATASET")
    print("=" * 64)

    # ── 1. Basic shape ──────────────────────────────────────────
    print(f"\n  Training set : {tr_raw.shape[0]:>7,} rows × {tr_raw.shape[1]} cols")
    print(f"  Test set     : {te_raw.shape[0]:>7,} rows × {te_raw.shape[1]} cols")

    # ── 2. Class distribution ───────────────────────────────────
    tr_dist = tr_raw[TARGET].value_counts().reindex(CLASSES, fill_value=0)
    te_dist = te_raw[TARGET].value_counts().reindex(CLASSES, fill_value=0)

    print("\n  Class distribution (train | test):")
    for cls in CLASSES:
        print(f"    {cls:<20}: {tr_dist[cls]:>7,}  |  {te_dist[cls]:>7,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, dist, split in zip(axes, [tr_dist, te_dist], ["Train", "Test"]):
        bars = ax.bar(dist.index, dist.values,
                      color=sns.color_palette("muted", len(CLASSES)))
        ax.bar_label(bars, fmt="{:,.0f}", padding=2, fontsize=8)
        ax.set_title(f"Class distribution — {split} ({dist.sum():,} samples)")
        ax.set_ylabel("Sample count")
        ax.set_xlabel("Traffic class")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "eda_class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"\n  ✓ class distribution plot → {FIGS / 'eda_class_distribution.png'}")

    # ── 3. Imbalance analysis ────────────────────────────────────
    total_tr = tr_dist.sum()
    dominant = tr_dist.max()
    minority = tr_dist[tr_dist > 0].min()
    imbalance_ratio = dominant / minority
    print(f"\n  Imbalance ratio (max/min count): {imbalance_ratio:.1f}×")
    print(f"  Dominant class : {tr_dist.idxmax()} ({tr_dist.max():,})")
    print(f"  Minority class : {tr_dist[tr_dist > 0].idxmin()} "
          f"({tr_dist[tr_dist > 0].min():,})")

    pct = (tr_dist / total_tr * 100).round(2)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4caf50" if cls == "Normal" else "#f44336" for cls in CLASSES]
    bars = ax.barh(CLASSES[::-1], pct.values[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.set_xlabel("Percentage of training set")
    ax.set_title("Class imbalance — training set")
    fig.tight_layout()
    fig.savefig(FIGS / "eda_imbalance_ratio.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ imbalance chart → {FIGS / 'eda_imbalance_ratio.png'}")

    # ── 4. Missing values ────────────────────────────────────────
    missing = tr_raw.isnull().sum()
    n_missing = (missing > 0).sum()
    print(f"\n  Missing values in training set: {n_missing} columns affected")
    if n_missing:
        print(f"  {missing[missing > 0].to_string()}")
    else:
        print("  (No missing values detected — clean dataset)")

    # ── 5. Feature type breakdown ────────────────────────────────
    numeric_cols = [c for c in tr_raw.columns
                    if c not in CAT_COLS + [TARGET, "id", "label"]]
    print(f"\n  Feature types:")
    print(f"    Numeric (continuous/discrete) : {len(numeric_cols)}")
    print(f"    Categorical (one-hot encoded) : {len(CAT_COLS)}  → {CAT_COLS}")
    print(f"    Post-encoding feature dim     : {len(feature_names)}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── Load ────────────────────────────────────────────────────
    print("\nLoading UNSW-NB15 data...")
    tr_raw, te_raw = load_data()
    print(f"  Train: {tr_raw.shape}  |  Test: {te_raw.shape}")

    # ── Preprocess ──────────────────────────────────────────────
    print("\nPreprocessing...")
    X_tr, X_te, y_tr, y_te, scaler, feature_names = preprocess(tr_raw, te_raw)
    n_feat = X_tr.shape[1]
    print(f"  Features after encoding: {n_feat}")
    print(f"  Train: {X_tr.shape[0]:,}  |  Test: {X_te.shape[0]:,}")
    test_dist = {CLASSES[i]: int((y_te == i).sum()) for i in range(N_CLS)}
    print(f"  Test class distribution: {test_dist}")

    # ── EDA ─────────────────────────────────────────────────────
    run_eda(tr_raw, te_raw, feature_names)

    # ── Save scaler ─────────────────────────────────────────────
    scaler_path = MODELS / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"\n  ✓ scaler saved → {scaler_path}")

    # ── Build & train baseline ───────────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}")
    print("  PART A — TRAINING BASELINE  (128→64→32, Dropout 0.3, lr=1e-3, bs=512)")
    print(sep)

    baseline = build_model_baseline(n_feat)
    baseline, history = train(
        baseline, X_tr, y_tr,
        batch_size=512, lr=1e-3, weight_decay=0.0, patience=5,
        log_path=LOGS / "training_baseline.csv",
    )

    # ── Save model ───────────────────────────────────────────────
    model_path = MODELS / "baseline_model.pt"
    torch.save(baseline.state_dict(), model_path)
    print(f"  ✓ model saved → {model_path}")

    # ── Evaluate ─────────────────────────────────────────────────
    probs, y_pred, conf, bin_stats, cm = eval_metrics(
        baseline, X_te, y_te, "Baseline (Part A)")
    alerts, alert_stats = eval_alerts(
        y_te, y_pred, conf, "Baseline (Part A)")
    stream_df = simulate_stream(
        baseline, X_te, y_te, n_samples=1000, label="Baseline (Part A)")

    # ── Save outputs ─────────────────────────────────────────────
    save_training_curves(history,  "Baseline",
                         FIGS / "training_curves_baseline.png")
    save_confusion_matrix(cm,       "Baseline (Part A)",
                          FIGS / "confusion_matrix_baseline.png")
    save_confidence_plot(y_te, y_pred, conf, "Baseline (Part A)",
                         FIGS / "confidence_distribution_baseline.png")
    save_alert_distribution(alert_stats, "Baseline (Part A)",
                             FIGS / "alert_distribution_baseline.png")
    save_metrics_csv(bin_stats, alert_stats, "Baseline",
                     TABLES / "metrics_baseline.csv")

    stream_df.to_csv(TABLES / "stream_log_baseline.csv", index=False)
    print(f"  ✓ stream log → {TABLES / 'stream_log_baseline.csv'}")

    print(f"\n  Total runtime: {time.time() - t0:.1f}s")
    print("\n  Part A complete. Run src/part_b_genai/train_improved.py next.\n")


if __name__ == "__main__":
    main()
