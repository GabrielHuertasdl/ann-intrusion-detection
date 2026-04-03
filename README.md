# 6009CMD — ANN-Based Network Intrusion Detection System

**Module:** 6009CMD Artificial Neural Networks  
**University:** Coventry University  
**Dataset:** UNSW-NB15 (Moustafa & Slay, 2015)  
**Task:** 10-class network intrusion detection using PyTorch ANNs

---

## Project Overview

This coursework compares two ANN development approaches for network intrusion
detection on the UNSW-NB15 benchmark:

- **Part A** — A baseline ANN (128→64→32) designed entirely by the student
  without any Generative AI assistance
- **Part B** — An improved ANN (256→128→64 + BatchNorm + L2) developed with
  Generative AI (Claude 3.5 Sonnet) assistance for architecture decisions
- **Part C** — Critical comparative reflection (in `report/report-final.md`)

Both models classify network flows into 10 categories (Normal + 9 attack types)
and feed into a four-tier confidence-gated alert system.

---

## Dataset

**UNSW-NB15** — Australian Centre for Cyber Security, 2015  
- Training set: 82,332 records · 45 features  
- Test set: 175,341 records · 45 features  
- 10 classes: Normal, Fuzzers, Analysis, Backdoors, DoS, Exploits,
  Generic, Reconnaissance, Shellcode, Worms  
- Severe class imbalance: 840× ratio (Normal vs Worms in training)

**Download:** https://research.unsw.edu.au/projects/unsw-nb15-dataset  
Place CSV files in `Dataset/`:
```
Dataset/
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv
```

---

## Repository Structure

```
.
├── src/
│   ├── common/
│   │   ├── preprocessing.py   # Shared data loading + preprocessing pipeline
│   │   └── utils.py           # Shared ANN, training, evaluation, plotting
│   ├── part_a_human/
│   │   └── train_baseline.py  # Part A: human-guided baseline
│   └── part_b_genai/
│       └── train_improved.py  # Part B: GenAI-assisted improved model
├── outputs/
│   ├── figures/               # PNG plots (EDA, training curves, CM, alerts)
│   ├── tables/                # CSV metric exports + stream logs
│   ├── models/                # Saved .pt model weights + scaler
│   └── logs/                  # Training loss CSV logs per epoch
├── report/
│   └── report-final.md        # Technical report (≤2500 words)
├── prompts/
│   ├── prompts_used_part_b.txt              # Adopted GenAI suggestions
│   ├── prompts_rejected_or_partially_used.txt  # Rejected suggestions
│   └── genai_reflection_notes.md           # Critical GenAI reflection
├── docs/
│   ├── ai_acknowledgment.md               # AI usage declaration table
│   ├── video_script_5min.md               # Video walkthrough script
│   ├── github_submission_checklist.md     # Pre-submission checklist
│   └── project_todo_before_submission.md  # Outstanding tasks
├── Dataset/                   # UNSW-NB15 CSV files (not committed)
├── requirements.txt
└── .gitignore
```

---

## How to Run

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Part A (human-guided baseline)
python -m src.part_a_human.train_baseline

# 4. Run Part B (GenAI-assisted improved model)
python -m src.part_b_genai.train_improved
```

Both scripts must be run from the project root directory.  
Part A should be run first; Part B loads Part A metrics for comparison.

---

## Key Results Summary

| Metric | Part A Baseline | Part B Improved |
|---|---|---|
| Accuracy | 0.7609 | 0.7525 |
| Macro-F1 | 0.4642 | 0.4492 |
| Weighted-F1 | 0.7290 | 0.7146 |
| Binary F1 | 0.8155 | 0.7802 |
| Binary Precision | 0.8934 | **0.9540** |
| Binary Recall | 0.7500 | 0.6600 |
| FPR | 0.1346 | **0.0479** |
| FP alerts | 9,426 | **3,355 (−64%)** |
| FN hard | 26,328 (25.0%) | 35,812 (34.0%) |

See `outputs/tables/comparison_summary.csv` after running both parts.

---

## Outputs Generated

After running both scripts, the following files are available:

**Figures:**
- `eda_class_distribution.png` — Train/test class counts
- `eda_imbalance_ratio.png` — Imbalance severity chart
- `training_curves_{baseline,improved}.png` — Loss curves per epoch
- `confusion_matrix_{baseline,improved}.png` — Row-normalised heatmap
- `confidence_distribution_{baseline,improved}.png` — Correct vs incorrect
- `alert_distribution_{baseline,improved}.png` — Alert tier breakdown
- `comparison_baseline_vs_improved.png` — Side-by-side metric bars

**Tables:**
- `metrics_{baseline,improved}.csv` — All scalar metrics
- `comparison_summary.csv` — Δ between models
- `stream_log_{baseline,improved}.csv` — 1000-sample stream event logs

---

## AI Acknowledgment Summary

| Part | GenAI used? | Purpose |
|---|---|---|
| Part A | No | Entirely human-guided design |
| Part B | Yes — Claude 3.5 Sonnet | Architecture suggestions, hyperparameter guidance |
| Part C | No | Entirely student reflection |

Full disclosure: `docs/ai_acknowledgment.md`
