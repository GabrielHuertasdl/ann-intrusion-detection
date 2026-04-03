# Video Script — 5-Minute Project Walkthrough
# 6009CMD Artificial Neural Networks — Coursework Submission

---

## Timing guide

| Segment | Time | Content |
|---|---|---|
| Intro + Problem | 0:00 – 0:35 | Problem context, dataset |
| Dataset demo | 0:35 – 1:10 | EDA figures, class imbalance |
| Part A walkthrough | 1:10 – 2:15 | Code, architecture, baseline results |
| Part B walkthrough | 2:15 – 3:15 | Improvements, GenAI process, results |
| Comparison | 3:15 – 4:00 | Metrics table, charts, key differences |
| Reflection | 4:00 – 5:00 | Critical insights, GenAI limits, conclusion |

---

## 0:00 – 0:35 | Problem and Dataset

> "This project builds an Artificial Neural Network-based Intrusion Detection
> System using the UNSW-NB15 benchmark dataset from the Australian Centre for
> Cyber Security.  The task is a 10-class classification problem: given a
> network flow record, classify it as either normal traffic or one of nine
> attack types — including Exploits, Generic attacks, Fuzzers, Reconnaissance,
> and rarer classes like Worms and Shellcode."

> "What makes this challenging is class imbalance: the training set has an
> 840-to-1 ratio between the dominant class and the rarest class.  Normal traffic
> and Generic attacks account for the majority of records, while Worms appear
> only 44 times in training.  This directly affects model design decisions."

**[Show: eda_class_distribution.png and eda_imbalance_ratio.png]**

---

## 0:35 – 1:10 | Dataset and Preprocessing

> "The dataset has 45 raw features.  Three — protocol, service, and connection
> state — are categorical and one-hot encoded.  After encoding, we have 190
> features.  StandardScaler is fit on the training set only to prevent data
> leakage into the test set.  The training set has 82,000 records; the test
> set has 175,000."

> "Notably, Backdoors and DoS are absent from both CSV splits in our files —
> a known characteristic of the official dataset split — so the effective
> classification task is 8 attack classes plus Normal."

---

## 1:10 – 2:15 | Part A — Human-Guided Baseline

> "Part A was designed entirely without Generative AI.  Starting from first
> principles, I chose a compact three-layer architecture: 128, 64, 32 neurons
> with uniform Dropout of 0.3 on the first two layers."

**[Show: src/part_a_human/train_baseline.py — architecture section]**

> "The rationale: this is a tabular dataset with well-studied class structure.
> A small network is interpretable, trains quickly on CPU, and provides a
> clean baseline.  Adam with the default learning rate of 1e-3 and batch
> size 512 ensures stable gradients over the 82k training set.  Early
> stopping with patience 5 prevents overfitting without manual tuning."

> "The results: overall accuracy of 76%, binary F1 of 0.82.  The model
> handles dominant classes well — Generic at 98% F1 — but struggles with
> minority classes: Fuzzers at only 16% F1, Analysis at 0%.  The binary
> false negative rate is 25%, meaning one in four attacks is completely missed.
> This is the gap that Part B targets."

**[Show: confusion_matrix_baseline.png, alert_distribution_baseline.png]**

---

## 2:15 – 3:15 | Part B — GenAI-Assisted Improved Model

> "For Part B, I used Claude 3.5 Sonnet to suggest targeted improvements.
> The key question I posed was: given the minority-class failure in Part A,
> what architecture changes would improve recall without collapsing under
> the class imbalance?"

> "Claude suggested three things: wider layers — 256, 128, 64 — to capture
> richer feature combinations from the 190-dimensional input; BatchNormalization
> after each layer to stabilise training on the mixed-scale feature space;
> and graduated dropout — 0.4, 0.3, 0.2 — combined with L2 regularisation
> and a lower learning rate of 5e-4.  I evaluated each suggestion critically."

**[Show: prompts/prompts_used_part_b.txt — Prompt 1]**

> "I rejected the suggestion of a 4th hidden layer — unnecessary complexity
> for a tabular dataset.  I rejected SMOTE oversampling because it would
> change the pipeline rather than the architecture, making the comparison
> unfair.  I kept the three changes I could justify empirically."

**[Show: src/part_b_genai/train_improved.py — architecture docstring]**

---

## 3:15 – 4:00 | Comparison

> "Here is the side-by-side comparison."

**[Show: comparison_baseline_vs_improved.png and outputs/tables/comparison_summary.csv]**

> "The improved model achieves better accuracy, better macro-F1, and a lower
> false negative rate — meaning fewer attacks completely missed.  The trade-off
> is a modest increase in false positive alerts — normal traffic incorrectly
> flagged.  This is the classic precision-recall trade-off in IDS design."

> "The alert system makes this operational: the confidence-gated tiers give
> analysts a prioritised queue.  CRITICAL_ALERT events are auto-flagged;
> SUSPICIOUS events go to human review; LOW_CONF_ATTACK events are logged."

---

## 4:00 – 5:00 | Critical Reflection

> "So which approach was better?  From a pure metric standpoint, the GenAI-
> assisted model wins on accuracy, macro-F1, and FN reduction.  But that's not
> the whole story."

> "The human-guided baseline produced a clean, explainable design.  I understood
> every decision: why 128 neurons, why Dropout 0.3, why batch 512.  GenAI gave
> me a broader search space of options faster — but it required critical filtering.
> Three out of seven suggestions were rejected."

> "GenAI cannot see your data, cannot run your code, and doesn't understand your
> submission constraints.  It generates plausible hypotheses.  The student's job
> is to turn hypotheses into experiments and experiments into honest analysis.
> That part cannot be delegated."

> "In future work I'd use GenAI earlier in the literature review phase —
> 'what regularisation strategies are documented for imbalanced tabular IDS
> datasets?' — and keep the critical analysis and experimental design fully
> human.  That's the right division of labour."

> "Thank you.  All code, outputs, and documentation are in the GitHub repository."

---

## Notes for recording

- Recommended screen layout: code editor on left, terminal/figures on right
- Do not read the script verbatim — use it as anchor points
- Aim for ~5 min 00 sec — the brief says "maximum 5 minutes"
- If running short, expand on the comparison section
- Record in one continuous take if possible; avoid jump cuts in the middle of a demo
