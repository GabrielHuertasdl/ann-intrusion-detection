# ============================================================
# ANN-based Intrusion Detection System — UNSW-NB15
# Course: Final-Year Neural Networks
# Backend: PyTorch (CPU) — TensorFlow not available on Python 3.14
# ============================================================
import os, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Constants ──────────────────────────────────────────────
SEED      = 42
DATASET   = 'Dataset'
TRAIN_CSV = os.path.join(DATASET, 'UNSW_NB15_training-set.csv')
TEST_CSV  = os.path.join(DATASET, 'UNSW_NB15_testing-set.csv')
CAT_COLS  = ['proto', 'service', 'state']
DROP_COLS = ['id', 'label']          # 'id'=row id; 'label'=binary (unused)
TARGET    = 'attack_cat'
CLASSES   = ['Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS',
             'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
N_CLS     = len(CLASSES)
DEVICE    = torch.device('cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── 1. Load ────────────────────────────────────────────────
def load_data(train_path=TRAIN_CSV, test_path=TEST_CSV):
    tr = pd.read_csv(train_path, low_memory=False)
    te = pd.read_csv(test_path,  low_memory=False)
    for df in (tr, te):
        df[TARGET] = (df[TARGET].fillna('Normal')
                                .astype(str).str.strip().str.title())
        df[TARGET] = df[TARGET].replace('', 'Normal')
    return tr, te


# ── 2. Preprocess ──────────────────────────────────────────
def preprocess(tr, te):
    """Drop IDs → one-hot categoricals → StandardScaler → align columns."""
    tr = tr.drop(columns=[c for c in DROP_COLS if c in tr.columns])
    te = te.drop(columns=[c for c in DROP_COLS if c in te.columns])

    y_tr_raw = tr.pop(TARGET)
    y_te_raw = te.pop(TARGET)

    tr = pd.get_dummies(tr, columns=[c for c in CAT_COLS if c in tr.columns])
    te = pd.get_dummies(te, columns=[c for c in CAT_COLS if c in te.columns])

    # Align: train is the reference schema
    tr, te = tr.align(te, join='left', axis=1, fill_value=0)

    tr = tr.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    te = te.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(tr.values).astype(np.float32)
    X_te = scaler.transform(te.values).astype(np.float32)

    def encode(y_raw):
        idx = np.array([CLASSES.index(c) if c in CLASSES else 0
                        for c in y_raw], dtype=np.int64)
        return idx

    return X_tr, X_te, encode(y_tr_raw), encode(y_te_raw)


# ── 3. ANN Module ──────────────────────────────────────────
class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropouts, use_bn=False):
        super().__init__()
        layers = []
        in_d = input_dim
        for h, d in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(in_d, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(d))
            in_d = h
        layers.append(nn.Linear(in_d, N_CLS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)          # raw logits; softmax applied at inference


def build_model_baseline(input_dim):
    """128→64→32, Dropout(0.3). Adam lr=1e-3, batch=512."""
    return ANN(input_dim,
               hidden_dims=[128, 64, 32],
               dropouts=[0.3, 0.3, 0.0],
               use_bn=False)


def build_model_improved(input_dim):
    """
    256→128→64, BatchNorm, L2(1e-4), graduated Dropout(0.4/0.3/0.2).
    Adam lr=5e-4, batch=256.
    Rationale: wider capacity improves minority-class recall; BN stabilises
    training and acts as implicit regulariser; L2 penalises large weights
    on the imbalanced dataset; smaller LR + batch improves generalisation.
    """
    return ANN(input_dim,
               hidden_dims=[256, 128, 64],
               dropouts=[0.4, 0.3, 0.2],
               use_bn=True)


# ── 4. Train ───────────────────────────────────────────────
def train(model, X_tr, y_tr, batch_size=512, max_epochs=50,
          lr=1e-3, weight_decay=0.0, patience=5):
    """
    batch_size=512 (baseline): large for CPU speed, stable gradients on 175k rows.
    EarlyStopping patience=5 on val_loss avoids overfitting without manual tuning.
    weight_decay used for improved model's L2 regularisation via Adam.
    """
    X_t = torch.tensor(X_tr)
    y_t = torch.tensor(y_tr)
    ds  = TensorDataset(X_t, y_t)

    val_n   = int(0.15 * len(ds))
    tr_n    = len(ds) - val_n
    tr_ds, val_ds = random_split(ds, [tr_n, val_n],
                                 generator=torch.Generator().manual_seed(SEED))

    tr_ld  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=2048, shuffle=False)

    opt      = torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=2, factor=0.5)

    best_val, best_state, no_imp = float('inf'), None, 0

    for ep in range(1, max_epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_ld:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= tr_n

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= val_n
        scheduler.step(val_loss)

        print(f'  Epoch {ep:>3}/{max_epochs}  '
              f'train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}', flush=True)

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'  Early stopping at epoch {ep} (patience={patience})')
                break

    model.load_state_dict(best_state)
    return model


# ── 5. Predict helper ──────────────────────────────────────
def predict(model, X, batch_size=2048):
    model.eval()
    X_t   = torch.tensor(X)
    ld    = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    probs_list = []
    with torch.no_grad():
        for (xb,) in ld:
            logits = model(xb)
            probs_list.append(torch.softmax(logits, dim=1).numpy())
    probs  = np.vstack(probs_list)
    y_pred = np.argmax(probs, axis=1)
    conf   = np.max(probs, axis=1)
    return probs, y_pred, conf


# ── 6. Multiclass + Binary + Confidence Evaluation ─────────
def eval_metrics(model, X_te, y_te_idx, label='Model'):
    probs, y_pred, conf = predict(model, X_te)

    sep = '=' * 64
    print(f'\n{sep}\n  MULTICLASS METRICS — {label}\n{sep}')
    acc = accuracy_score(y_te_idx, y_pred)
    print(f'Accuracy: {acc:.4f}\n')
    print(classification_report(y_te_idx, y_pred,
                                 labels=list(range(N_CLS)),
                                 target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(y_te_idx, y_pred, labels=list(range(N_CLS)))
    print('Confusion Matrix (rows=true, cols=pred):')
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    # Binary: Normal(0) vs Attack(1+)
    print(f'\n--- Binary View ---')
    y_bin_t = (y_te_idx != 0).astype(int)
    y_bin_p = (y_pred   != 0).astype(int)
    TP = int(((y_bin_t == 1) & (y_bin_p == 1)).sum())
    TN = int(((y_bin_t == 0) & (y_bin_p == 0)).sum())
    FP = int(((y_bin_t == 0) & (y_bin_p == 1)).sum())
    FN = int(((y_bin_t == 1) & (y_bin_p == 0)).sum())
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec  = TP / (TP + FN) if TP + FN else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr  = FP / (FP + TN) if FP + TN else 0.0
    print(f'  TP={TP}  TN={TN}  FP={FP}  FN={FN}')
    print(f'  Precision={prec:.4f}  Recall/TPR={rec:.4f}  F1={f1:.4f}  FPR={fpr:.4f}')

    # Confidence distributions
    print(f'\n--- Confidence Distribution ---')
    correct = (y_pred == y_te_idx)
    print(f'  Correct   — mean={conf[correct].mean():.4f}  '
          f'median={np.median(conf[correct]):.4f}  '
          f'<0.6: {(conf[correct]<0.6).sum()}')
    print(f'  Incorrect — mean={conf[~correct].mean():.4f}  '
          f'median={np.median(conf[~correct]):.4f}  '
          f'<0.6: {(conf[~correct]<0.6).sum()}')
    nm = (y_te_idx == 0)
    print(f'  Normal traffic  — mean conf={conf[nm].mean():.4f}')
    print(f'  Attack traffic  — mean conf={conf[~nm].mean():.4f}')

    bin_stats = dict(acc=acc, prec=prec, rec=rec, f1=f1, fpr=fpr,
                     TP=TP, TN=TN, FP=FP, FN=FN)
    return probs, y_pred, conf, bin_stats


# ── 7. Alert System ────────────────────────────────────────
def alert_level(pred_class: int, confidence: float) -> str:
    if pred_class == 0:      return 'NO_ALERT'
    if confidence >= 0.8:    return 'CRITICAL_ALERT'
    if confidence >= 0.6:    return 'SUSPICIOUS'
    return 'LOW_CONF_ATTACK'


def eval_alerts(y_te_idx, y_pred, conf, label='Model'):
    """
    Policy: FN-hard = Attack→NO_ALERT (missed entirely).
            FN-soft = Attack→LOW_CONF_ATTACK (low-priority flag; strict policy
                      escalates to human review).
            FP      = Normal→any alert (drives analyst fatigue).
    Trade-off: raising CRITICAL threshold (0.8→0.9) cuts alert volume but
    pushes more true attacks into SUSPICIOUS; lowering it amplifies FP rate.
    """
    alerts  = np.array([alert_level(int(p), float(c))
                        for p, c in zip(y_pred, conf)])
    levels  = ['NO_ALERT', 'LOW_CONF_ATTACK', 'SUSPICIOUS', 'CRITICAL_ALERT']
    n       = len(alerts)
    sep     = '=' * 64
    print(f'\n{sep}\n  ALERT SYSTEM METRICS — {label}\n{sep}')
    for lv in levels:
        cnt = (alerts == lv).sum()
        print(f'  {lv:<22}: {cnt:>7}  ({100*cnt/n:.2f}%)')

    is_normal = (y_te_idx == 0)
    is_attack = ~is_normal
    FP_a  = int(((is_normal) & (alerts != 'NO_ALERT')).sum())
    FN_h  = int(((is_attack) & (alerts == 'NO_ALERT')).sum())
    FN_s  = int(((is_attack) & (alerts == 'LOW_CONF_ATTACK')).sum())

    print(f'\n  FP  (Normal→alert)          : {FP_a:>7} '
          f'({100*FP_a/is_normal.sum():.2f}% of normals)')
    print(f'  FN hard (Attack→NO_ALERT)   : {FN_h:>7} '
          f'({100*FN_h/is_attack.sum():.2f}% of attacks)')
    print(f'  FN soft (Attack→LOW_CONF)   : {FN_s:>7} '
          f'({100*FN_s/is_attack.sum():.2f}% of attacks)')
    print(f'  → SUSPICIOUS + LOW_CONF_ATTACK: recommend human review.')
    print(f'  → CRITICAL_ALERT: automated block/quarantine.')

    return alerts, dict(FP_alerts=FP_a, FN_hard=FN_h, FN_soft=FN_s)


# ── 8. Streaming Simulation ────────────────────────────────
def simulate_stream(model, X_te, y_te_idx, n_samples=1000, label='Model'):
    """Sample n_samples rows in random order; log each event as a stream."""
    rng   = np.random.default_rng(SEED)
    idxs  = rng.choice(len(X_te), size=n_samples, replace=False)
    probs, preds, confs = predict(model, X_te[idxs])

    log = [{'stream_pos' : i,
            'orig_idx'   : int(orig_idx),
            'true_label' : CLASSES[y_te_idx[orig_idx]],
            'pred_label' : CLASSES[pred],
            'confidence' : round(float(c), 4),
            'alert'      : alert_level(int(pred), float(c))}
           for i, (orig_idx, pred, c) in enumerate(zip(idxs, preds, confs))]

    df = pd.DataFrame(log)
    sep = '=' * 64
    print(f'\n{sep}\n  STREAM SIMULATION ({n_samples} samples) — {label}\n{sep}')
    print(df.head(25).to_string(index=False))
    print(f'  ... ({n_samples} rows total)\n')

    print('  Stream alert summary:')
    for lv in ['NO_ALERT', 'LOW_CONF_ATTACK', 'SUSPICIOUS', 'CRITICAL_ALERT']:
        cnt = (df['alert'] == lv).sum()
        print(f'    {lv:<22}: {cnt:>4}  ({100*cnt/n_samples:.1f}%)')

    is_atk = df['true_label'] != 'Normal'
    s_FN   = ((is_atk)  & (df['alert'] == 'NO_ALERT')).sum()
    s_FP   = ((~is_atk) & (df['alert'] != 'NO_ALERT')).sum()
    print(f'  Stream FP: {s_FP}  |  Stream FN (Attack→NO_ALERT): {s_FN}')
    return df


# ── 9. Main ────────────────────────────────────────────────
def main():
    t0 = time.time()
    print('Loading UNSW-NB15 data...')
    tr_raw, te_raw = load_data()
    print(f'  Train: {tr_raw.shape} | Test: {te_raw.shape}')

    print('Preprocessing...')
    X_tr, X_te, y_tr, y_te = preprocess(tr_raw, te_raw)
    n_feat = X_tr.shape[1]
    print(f'  Features: {n_feat} | Train: {X_tr.shape[0]} | Test: {X_te.shape[0]}')
    dist = {CLASSES[i]: int((y_te == i).sum()) for i in range(N_CLS)}
    print(f'  Test class dist: {dist}')

    # ── BASELINE ──────────────────────────────────────────
    sep = '=' * 64
    print(f'\n{sep}\n  TRAINING BASELINE  (128→64→32, Dropout 0.3, lr=1e-3, bs=512)\n{sep}')
    baseline = build_model_baseline(n_feat)
    train(baseline, X_tr, y_tr, batch_size=512, lr=1e-3)

    probs_b, pred_b, conf_b, bin_b = eval_metrics(baseline, X_te, y_te, 'Baseline')
    alerts_b, astats_b = eval_alerts(y_te, pred_b, conf_b, 'Baseline')
    stream_b = simulate_stream(baseline, X_te, y_te, label='Baseline')

    # ── IMPROVED ──────────────────────────────────────────
    print(f'\n{sep}\n  TRAINING IMPROVED  (256→128→64, BN, L2=1e-4, lr=5e-4, bs=256)\n{sep}')
    improved = build_model_improved(n_feat)
    train(improved, X_tr, y_tr, batch_size=256, lr=5e-4, weight_decay=1e-4)

    probs_i, pred_i, conf_i, bin_i = eval_metrics(improved, X_te, y_te, 'Improved')
    alerts_i, astats_i = eval_alerts(y_te, pred_i, conf_i, 'Improved')
    stream_i = simulate_stream(improved, X_te, y_te, label='Improved')

    # ── COMPARISON ────────────────────────────────────────
    print(f'\n{sep}\n  BASELINE vs IMPROVED — SUMMARY\n{sep}')
    print(f'  {"Metric":<12}  {"Baseline":>10}  {"Improved":>10}  {"Δ":>8}')
    print(f'  {"-"*46}')
    for m in ['acc', 'prec', 'rec', 'f1', 'fpr']:
        bv, iv = bin_b[m], bin_i[m]
        print(f'  {m.upper():<12}  {bv:>10.4f}  {iv:>10.4f}  {iv-bv:>+8.4f}')

    print(f'\n  {"Alert metric":<22}  {"Baseline":>10}  {"Improved":>10}')
    print(f'  {"-"*46}')
    for k in ['FP_alerts', 'FN_hard', 'FN_soft']:
        print(f'  {k:<22}  {astats_b[k]:>10}  {astats_i[k]:>10}')

    print(f'\n  Total runtime: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
