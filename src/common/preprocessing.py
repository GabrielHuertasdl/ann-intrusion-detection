"""
Shared data loading and preprocessing pipeline for UNSW-NB15.
Used by both Part A (human-guided) and Part B (GenAI-assisted) scripts.

Design decisions (human-guided, Part A):
  - One-hot encode {proto, service, state}; align test to train schema
  - StandardScaler fit on train only (no leakage)
  - attack_cat mapped to fixed 10-class integer index
  - binary 'label' column and row-id 'id' column dropped before training
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Project paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR  = PROJECT_ROOT / "Dataset"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

# ── Constants ────────────────────────────────────────────────────────────────
SEED      = 42
TRAIN_CSV = DATASET_DIR / "UNSW_NB15_training-set.csv"
TEST_CSV  = DATASET_DIR / "UNSW_NB15_testing-set.csv"

CAT_COLS  = ["proto", "service", "state"]
DROP_COLS = ["id", "label"]   # row-id carries no signal; binary label unused
TARGET    = "attack_cat"

CLASSES = [
    "Normal", "Fuzzers", "Analysis", "Backdoors", "DoS",
    "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms",
]
N_CLS = len(CLASSES)


def load_data(train_path: Path = TRAIN_CSV,
              test_path:  Path = TEST_CSV):
    """Load train/test CSVs; normalise attack_cat labels."""
    tr = pd.read_csv(train_path, low_memory=False)
    te = pd.read_csv(test_path,  low_memory=False)
    for df in (tr, te):
        df[TARGET] = (df[TARGET]
                      .fillna("Normal")
                      .astype(str)
                      .str.strip()
                      .str.title()
                      .replace("", "Normal"))
    return tr, te


def preprocess(tr: pd.DataFrame, te: pd.DataFrame):
    """
    Full preprocessing pipeline:
      1. Drop id + label columns
      2. Separate target (attack_cat)
      3. One-hot encode categorical columns
      4. Align test schema to train schema (fill missing → 0)
      5. Coerce to float32; fill residual NaNs
      6. StandardScaler (fit on train only)
      7. Encode labels as integer indices

    Returns
    -------
    X_tr, X_te : np.ndarray float32
    y_tr, y_te : np.ndarray int64
    scaler     : fitted StandardScaler (save this for production inference)
    feature_names : list[str]
    """
    tr = tr.drop(columns=[c for c in DROP_COLS if c in tr.columns]).copy()
    te = te.drop(columns=[c for c in DROP_COLS if c in te.columns]).copy()

    y_tr_raw = tr.pop(TARGET)
    y_te_raw = te.pop(TARGET)

    tr = pd.get_dummies(tr, columns=[c for c in CAT_COLS if c in tr.columns])
    te = pd.get_dummies(te, columns=[c for c in CAT_COLS if c in te.columns])

    # Keep test aligned to train feature space
    tr, te = tr.align(te, join="left", axis=1, fill_value=0)

    feature_names = list(tr.columns)

    tr = tr.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    te = te.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr.values).astype(np.float32)
    X_te = scaler.transform(te.values).astype(np.float32)

    def _encode(y_raw):
        return np.array(
            [CLASSES.index(c) if c in CLASSES else 0 for c in y_raw],
            dtype=np.int64,
        )

    return X_tr, X_te, _encode(y_tr_raw), _encode(y_te_raw), scaler, feature_names
