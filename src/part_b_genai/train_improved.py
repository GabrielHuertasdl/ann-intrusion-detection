"""
Part B — GenAI-Assisted ANN Design
=====================================
Course  : 6009CMD Artificial Neural Networks, Coventry University
Dataset : UNSW-NB15 (Moustafa & Slay, 2015)
Task    : 10-class network intrusion detection

GenAI tools used
----------------
  Claude 3.5 Sonnet (Anthropic) was used during the design phase to:
    1. Suggest architectural improvements over the Part A baseline
    2. Justify the combination of BatchNorm + graduated dropout
    3. Recommend learning-rate and batch-size adjustments to match L2
    4. Cross-check threshold selection for the confidence-based alert system

Every suggestion was critically evaluated before adoption.
Rejected and partially accepted suggestions are documented in
  prompts/prompts_rejected_or_partially_used.txt

Design changes from Part A baseline (all GenAI-informed)
---------------------------------------------------------
  | Component          | Baseline (Part A)    | Improved (Part B)      |
  |--------------------|----------------------|------------------------|
  | Hidden layers      | 128 → 64 → 32        | 256 → 128 → 64         |
  | BatchNorm          | No                   | Yes (after each Linear)|
  | Dropout            | 0.3 / 0.3 / 0.0     | 0.4 / 0.3 / 0.2        |
  | L2 regularisation  | None                 | weight_decay = 1e-4    |
  | Learning rate      | 1e-3                 | 5e-4                   |
  | Batch size         | 512                  | 256                    |

Rationale (GenAI-suggested, student-verified)
- Wider (256) first layer captures richer feature combinations from the
  190+ post-encoding dimensions; important for minority classes.
- BatchNorm stabilises gradients on mixed-scale features (continuous
  flow stats + one-hot binary columns) even after StandardScaler.
- Graduated dropout (0.4 → 0.3 → 0.2): stronger near input encourages
  complementary representations; gentler near output preserves class
  boundaries for softmax calibration.
- Lower LR + smaller batch: synergises with L2; tighter generalisation
  especially on under-represented Worms, Shellcode, Backdoor classes.

Run
---
  cd /path/to/ANN
  source venv/bin/activate
  python -m src.part_b_genai.train_improved

  NOTE: Run Part A first so baseline metrics exist for comparison.

Outputs (all saved to outputs/)
--------------------------------
  figures/training_curves_improved.png
  figures/confusion_matrix_improved.png
  figures/confidence_distribution_improved.png
  figures/alert_distribution_improved.png
  figures/comparison_baseline_vs_improved.png
  tables/metrics_improved.csv
  tables/comparison_summary.csv
  tables/stream_log_improved.csv
  models/improved_model.pt
  logs/training_improved.csv
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.preprocessing import CLASSES, N_CLS, load_data, preprocess
from src.common.utils import (
    build_model_improved,
    eval_alerts,
    eval_metrics,
    save_alert_distribution,
    save_comparison_chart,
    save_confidence_plot,
    save_confusion_matrix,
    save_metrics_csv,
    save_training_curves,
    simulate_stream,
    train,
)

OUT    = PROJECT_ROOT / "outputs"
FIGS   = OUT / "figures"
TABLES = OUT / "tables"
MODELS = OUT / "models"
LOGS   = OUT / "logs"
for d in [FIGS, TABLES, MODELS, LOGS]:
    d.mkdir(parents=True, exist_ok=True)


def _load_baseline_stats() -> dict | None:
    """Load Part A metrics for side-by-side comparison."""
    path = TABLES / "metrics_baseline.csv"
    if not path.exists():
        print("  ⚠  metrics_baseline.csv not found — "
              "run Part A first for comparison table.")
        return None
    df = pd.read_csv(path)
    return df.iloc[0].to_dict()


def main():
    t0 = time.time()

    # ── Load & preprocess (identical pipeline to Part A) ─────────
    print("\nLoading UNSW-NB15 data...")
    tr_raw, te_raw = load_data()

    print("\nPreprocessing (same pipeline as Part A)...")
    X_tr, X_te, y_tr, y_te, scaler, feature_names = preprocess(tr_raw, te_raw)
    n_feat = X_tr.shape[1]
    print(f"  Features: {n_feat}  |  Train: {X_tr.shape[0]:,}  |  "
          f"Test: {X_te.shape[0]:,}")

    # ── Build & train improved model ──────────────────────────────
    sep = "=" * 64
    print(f"\n{sep}")
    print("  PART B — TRAINING IMPROVED  "
          "(256→128→64, BN, L2=1e-4, lr=5e-4, bs=256)")
    print(sep)

    improved = build_model_improved(n_feat)
    improved, history = train(
        improved, X_tr, y_tr,
        batch_size=256, lr=5e-4, weight_decay=1e-4, patience=5,
        log_path=LOGS / "training_improved.csv",
    )

    # ── Save model ────────────────────────────────────────────────
    model_path = MODELS / "improved_model.pt"
    torch.save(improved.state_dict(), model_path)
    print(f"  ✓ model saved → {model_path}")

    # ── Evaluate ──────────────────────────────────────────────────
    probs, y_pred, conf, bin_stats, cm = eval_metrics(
        improved, X_te, y_te, "Improved (Part B)")
    alerts, alert_stats = eval_alerts(
        y_te, y_pred, conf, "Improved (Part B)")
    stream_df = simulate_stream(
        improved, X_te, y_te, n_samples=1000, label="Improved (Part B)")

    # ── Save outputs ──────────────────────────────────────────────
    save_training_curves(history,   "Improved",
                         FIGS / "training_curves_improved.png")
    save_confusion_matrix(cm,        "Improved (Part B)",
                          FIGS / "confusion_matrix_improved.png")
    save_confidence_plot(y_te, y_pred, conf, "Improved (Part B)",
                         FIGS / "confidence_distribution_improved.png")
    save_alert_distribution(alert_stats, "Improved (Part B)",
                             FIGS / "alert_distribution_improved.png")
    save_metrics_csv(bin_stats, alert_stats, "Improved",
                     TABLES / "metrics_improved.csv")

    stream_df.to_csv(TABLES / "stream_log_improved.csv", index=False)
    print(f"  ✓ stream log → {TABLES / 'stream_log_improved.csv'}")

    # ── Side-by-side comparison ───────────────────────────────────
    baseline_stats = _load_baseline_stats()
    if baseline_stats:
        print(f"\n{sep}")
        print("  BASELINE vs IMPROVED — SUMMARY")
        print(sep)
        print(f"  {'Metric':<14}  {'Baseline':>10}  {'Improved':>10}  {'Δ':>8}")
        print(f"  {'-'*48}")
        metrics = [
            ("Accuracy",    "acc"),
            ("Macro-F1",    "macro_f1"),
            ("Binary-F1",   "f1"),
            ("Precision",   "prec"),
            ("Recall",      "rec"),
            ("FPR",         "fpr"),
        ]
        for name, key in metrics:
            bv = float(baseline_stats.get(key, 0))
            iv = bin_stats.get(key, 0)
            print(f"  {name:<14}  {bv:>10.4f}  {iv:>10.4f}  {iv-bv:>+8.4f}")

        print(f"\n  {'Alert metric':<22}  {'Baseline':>10}  {'Improved':>10}")
        print(f"  {'-'*46}")
        for k in ["FP_alerts", "FN_hard", "FN_soft"]:
            bv = int(baseline_stats.get(k, 0))
            iv = alert_stats.get(k, 0)
            print(f"  {k:<22}  {bv:>10}  {iv:>10}")

        # Save comparison CSV
        comp = {
            "metric": [m[0] for m in metrics],
            "baseline": [float(baseline_stats.get(m[1], 0)) for m in metrics],
            "improved": [bin_stats.get(m[1], 0) for m in metrics],
        }
        comp_df = pd.DataFrame(comp)
        comp_df["delta"] = comp_df["improved"] - comp_df["baseline"]
        comp_path = TABLES / "comparison_summary.csv"
        comp_df.to_csv(comp_path, index=False)
        print(f"\n  ✓ comparison summary → {comp_path}")

        # Save comparison chart
        # Filter to numeric metrics for chart
        chart_stats_b = {k: float(baseline_stats.get(k, 0)) for k in bin_stats}
        save_comparison_chart(
            chart_stats_b, bin_stats,
            "Baseline (Part A)", "Improved (Part B)",
            FIGS / "comparison_baseline_vs_improved.png",
        )

    print(f"\n  Total runtime: {time.time() - t0:.1f}s")
    print("\n  Part B complete.\n")


if __name__ == "__main__":
    main()
