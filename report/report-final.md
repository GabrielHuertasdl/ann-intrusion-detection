# ANN-Based Network Intrusion Detection: Human-Guided vs GenAI-Assisted Design

**Module:** 6009CMD Artificial Neural Networks — Coventry University  
**Dataset:** UNSW-NB15 · **Framework:** PyTorch 2.x (CPU) · **SEED:** 42

---

## Abstract

This report presents a controlled comparison of two Artificial Neural Network (ANN) designs for network intrusion detection on the UNSW-NB15 benchmark. **Part A** implements a compact baseline (128→64→32) designed exclusively through independent reasoning; **Part B** implements an improved variant (256→128→64 with BatchNorm, L2 regularisation, and graduated dropout) whose architecture was informed by Claude 3.5 Sonnet. Both models perform 10-class traffic classification and feed into a confidence-gated four-tier alert system. Empirical results reveal a genuine precision-recall trade-off: the GenAI-assisted model reduces false positive alerts by 64% and lowers the false positive rate from 13.5% to 4.8%, while the human-guided baseline achieves higher recall (0.750 vs 0.660), binary F1 (0.816 vs 0.780), and misses 9,484 fewer attacks. The analysis demonstrates that GenAI accelerated architectural search but required critical human filtering, and that classifier accuracy alone is insufficient for operational IDS evaluation.

---

## 1. Introduction

Network intrusion detection systems (NIDS) must classify high-speed traffic in real time, distinguishing legitimate flows from a heterogeneous attack taxonomy. Rule-based systems such as Snort and Suricata require continuous signature updates and fail against novel variants; machine learning approaches offer adaptive, data-driven classification that generalises across attack families [1].

ANNs are well-suited for tabular NIDS tasks: they learn non-linear decision boundaries over high-dimensional network flow features, produce calibrated softmax probabilities compatible with confidence-gated alerting, and support inference speeds feasible for stream processing. The softmax confidence score is particularly valuable in SOC environments, where distinguishing high-certainty detections from ambiguous cases directly reduces analyst alert fatigue.

UNSW-NB15 (Australian Centre for Cyber Security, 2015) provides 49 engineered network flow features and ground-truth labels across 10 traffic classes [2]. Its realistic feature set, documented class imbalance, and pre-defined train/test splits have established it as a standard IDS evaluation benchmark. The 840× imbalance ratio between the dominant class (Normal, 37,000 samples) and the rarest (Worms, 44 samples) represents a central design challenge addressed—with different consequences—in each part.

---

## 2. Part A: Human-Guided ANN Design

### 2.1 Dataset and Exploratory Data Analysis

The UNSW-NB15 pre-defined splits allocate 82,332 records to training and 175,341 to testing. Raw features include 39 numeric flow statistics (durations, byte counts, packet rates, jitter, TTL values) and three nominal columns: `proto`, `service`, and `state`. After one-hot encoding these three categorical features and aligning train/test schemas, the input dimensionality expands to **190 features**.

EDA (Figure 1: `eda_class_distribution.png`) reveals severe imbalance: Normal (44.9%), Generic (22.9%), and Exploits (13.5%) together constitute 81.4% of training samples. At the other extreme, Worms (44 records, 0.05%) and Shellcode (378 records, 0.46%) are critically under-represented. The imbalance ratio reaches 840× (Figure 2: `eda_imbalance_ratio.png`). Importantly, Backdoors and DoS are absent from both pre-defined CSV splits—an artefact of the UNSW-NB15 official data distribution—yielding an effective eight-class attack taxonomy plus Normal. No missing values were detected in either split. This EDA directly motivated the regularisation and capacity decisions described below.

### 2.2 Preprocessing Pipeline

The preprocessing pipeline was designed entirely without GenAI assistance, following principled decisions to prevent data leakage:

- **One-hot encoding:** Applied independently to each split. Test columns are aligned to the training schema via `DataFrame.align(join='left')`, filling unseen categorical levels with zero. This mirrors production conditions where novel protocol types may appear at inference time.
- **StandardScaler:** Fitted exclusively on the training set. Using test-set statistics would constitute data leakage and produce artificially optimistic evaluation results.
- **Label encoding:** The `attack_cat` string column is mapped to integer indices 0–9 using a fixed deterministic class list, ensuring consistency across runs.
- **Dropped features:** Row identifier `id` carries no predictive signal and is discarded. The binary `label` column is unused in the multiclass setup.

### 2.3 Baseline Architecture and Training

The baseline ANN was designed from first principles as a compact, interpretable starting point appropriate for a well-studied tabular benchmark:

| Component | Configuration | Rationale |
|---|---|---|
| Hidden layers | 128 → 64 → 32 | Sufficient capacity for dominant classes; trainable on CPU |
| Activations | ReLU | Avoids vanishing gradients; standard for tabular data |
| Dropout | 0.3 / 0.3 / 0.0 | Prevents co-adaptation without over-constraining capacity |
| BatchNorm | No | Baseline intentionally minimal |
| Optimiser | Adam, lr = 1×10⁻³ | Default starting point; adapts per-parameter LR |
| Batch size | 512 | Stable CPU gradients over 82k training samples |
| Early stopping | patience = 5, val\_loss | Eliminates manual epoch tuning (15% validation split) |

Training ran to the full 50-epoch budget without triggering early stopping, converging from train\_loss = 1.049 (epoch 1) to 0.340 (epoch 50) with val\_loss = 0.333.

### 2.4 Evaluation Results

| Metric | Part A Baseline |
|---|---|
| Multiclass Accuracy | 76.09% |
| Macro-F1 | 0.464 |
| Weighted-F1 | 0.729 |
| Binary Precision | 0.893 |
| Binary Recall (TPR) | 0.750 |
| Binary F1 | 0.815 |
| False Positive Rate | 0.135 |
| FP alerts | 9,426 (13.5% of normals) |
| FN hard (missed attacks) | 26,328 (25.0% of attacks) |

Per-class breakdown illustrates the imbalance effect clearly: Generic achieves F1 = 0.98, Exploits F1 = 0.69, Reconnaissance F1 = 0.75, but Fuzzers only F1 = 0.16 and both Analysis and Worms F1 = 0.00. The gap between macro-F1 (0.464) and weighted-F1 (0.729) quantifies the minority-class collapse: the unweighted average is dragged down by classes with fewer than 700 training samples. The confidence distribution (Figure 5) shows correctly classified samples cluster near confidence 1.0 (mean = 0.903), while misclassified samples are broadly distributed (mean = 0.640), validating the use of confidence as an alert discriminator.

---

## 3. Part B: GenAI-Assisted ANN Design

### 3.1 GenAI Design Process

Claude 3.5 Sonnet was consulted across two sessions via claude.ai. Four structured prompts were submitted covering: (1) architecture expansion for minority-class recall, (2) learning-rate and batch-size adjustment alongside L2 regularisation, (3) alert threshold validation, and (4) interpretation of the accuracy–macro-F1 gap in imbalanced settings. Full prompt texts and Claude's responses are documented in `prompts/prompts_used_part_b.txt`.

Critical filtering was applied throughout. Of seven distinct suggestions, three were rejected with explicit reasoning: a fourth hidden layer (unnecessary depth for tabular data; established three-layer baselines in [3] suffice); SMOTE oversampling (alters the preprocessing pipeline rather than the architecture, preventing a clean controlled comparison with Part A); and post-training temperature scaling (requires a third calibration split and is disproportionate in scope to coursework constraints). Documentation of rejected suggestions is in `prompts/prompts_rejected_or_partially_used.txt`.

### 3.2 Architecture and Training Changes

| Component | Baseline (Part A) | Improved (Part B) | GenAI-informed? |
|---|---|---|---|
| Hidden layers | 128→64→32 | **256→128→64** | Yes |
| BatchNorm | No | **Yes (after each layer)** | Yes |
| Dropout | 0.3 / 0.3 / 0.0 | **0.4 / 0.3 / 0.2** | Yes |
| L2 (weight\_decay) | 0 | **1×10⁻⁴** | Yes |
| Learning rate | 1×10⁻³ | **5×10⁻⁴** | Yes |
| Batch size | 512 | **256** | Yes |
| Preprocessing | Identical | Identical | — |

GenAI's rationale for graduated dropout (0.4 → 0.3 → 0.2): stronger regularisation near the 190-dimensional input forces complementary feature representations; lighter dropout near the output preserves class-boundary information for softmax calibration. The lower learning rate (5×10⁻⁴) was recommended to stabilise training under simultaneous L2 and BatchNorm, avoiding overshooting narrow loss minima.

### 3.3 Evaluation Results

| Metric | Part B Improved |
|---|---|
| Multiclass Accuracy | 75.25% |
| Macro-F1 | 0.449 |
| Weighted-F1 | 0.715 |
| Binary Precision | **0.954** |
| Binary Recall (TPR) | 0.660 |
| Binary F1 | 0.780 |
| False Positive Rate | **0.048** |
| FP alerts | **3,355 (−64% vs baseline)** |
| FN hard (missed attacks) | 35,812 (+36% vs baseline) |

The improved model dramatically reduces false positive alerts (9,426 → 3,355) and cuts the false positive rate by a factor of 2.8×. However, hard false negatives increase from 26,328 to 35,812, indicating that the model became more conservative overall. The confidence distribution for incorrect predictions tightens slightly (mean = 0.630 vs 0.640), suggesting marginally better calibration but not the minority-class recall improvement that GenAI suggested.

---

## 4. Comparative Reflection

### Which model performed better, and why?

The answer is operationally context-dependent. On precision-oriented metrics, the GenAI-assisted improved model is unambiguously stronger: false positive rate 4.8% versus 13.5%, and 6,071 fewer unnecessary alerts per evaluation run. For a SOC analyst processing thousands of events daily, this 64% reduction in false alarms translates directly into reduced cognitive load and faster triage of genuine threats. The combination of wider layers (256), BatchNorm, and L2 collectively pushed the model toward higher-confidence predictions—with the 0.8 CRITICAL threshold in place, more Normal traffic clears the threshold confidently rather than being borderline-flagged.

On recall-oriented metrics, the baseline is strictly better: binary recall 0.750 versus 0.660, binary F1 0.816 versus 0.780, and 9,484 fewer hard false negatives. For a security-first deployment where missing any attack is unacceptable, the baseline is the stronger operational choice. This contrast directly exposes the fundamental NIDS design tension: precision and recall cannot simultaneously improve without additional mechanisms—class weighting, targeted oversampling, or threshold calibration—each of which was deliberately excluded to maintain a clean architectural comparison between the two parts.

The macro-F1 comparison (0.464 vs 0.449) confirms neither model adequately handles minority attack classes. Worms, Analysis, and Shellcode remain poorly detected by both variants, because the training representations for these classes—44, 677, and 378 samples respectively—are insufficient for any unconstrained ANN to generalise from. This is an honest finding: the GenAI-suggested architectural changes did not resolve the root cause of minority-class failure, which is a data problem, not an architecture problem.

### Independent thinking versus GenAI assistance

Designing Part A independently required engagement with every decision at a mechanical level: why 128 neurons and not 256, why uniform Dropout(0.3) and not graduated, why batch size 512. This process, while slower, produced a design that is fully explainable and easy to reason about. The baseline's relative strength on recall is likely a direct consequence of its simpler regularisation: without L2 pulling weights toward zero and BatchNorm stabilising the gradient landscape, the model remained more aggressive in predicting attack classes rather than defaulting to Normal.

GenAI assistance in Part B compressed the hypothesis space substantially. What might have required consulting four or five papers on regularisation strategies for imbalanced tabular datasets was reduced to a structured four-prompt dialogue. The suggestions for graduated dropout and the LR-batch coupling with L2 were technically sound and improved the system's precision performance in a measurable way.

However, GenAI assistance required non-trivial critical oversight. Three of seven suggestions were rejected. More importantly, GenAI predicted that the wider architecture with BatchNorm would improve minority-class recall—the stated objective of the prompt. The actual empirical result showed the opposite: the improved model missed more attacks and flagged fewer false positives. GenAI produced a plausible hypothesis; it could not validate it. The experimental execution, result interpretation, and operational framing of the precision-recall trade-off were entirely human responsibilities. This is the central insight: GenAI functions as a hypothesis generator, not an experimenter.

### Balancing GenAI and independent reasoning in future work

In future projects, GenAI would be most productively deployed during the literature review and hypothesis-generation phases—prompting for known regularisation strategies on imbalanced network datasets, or cross-checking threshold-selection reasoning against published IDS evaluation frameworks. Architecture and hyperparameter decisions should remain grounded in dataset-specific empirical testing, which GenAI cannot perform. Evaluation of results, particularly the interpretation of nuanced operational trade-offs as observed here, must remain fully human-driven. The most dangerous misuse pattern would be accepting GenAI's predicted performance improvements without running experiments: in this project, the direction of improvement was wrong on three of six key metrics.

---

## 5. GitHub and Reproducibility

The repository follows the required brief structure. `src/part_a_human/` and `src/part_b_genai/` are physically separated with no cross-imports between parts. All experiments use `SEED=42` throughout (NumPy, PyTorch generator, and dataset split). Both scripts are executable from the project root: `python -m src.part_a_human.train_baseline` and `python -m src.part_b_genai.train_improved`. The `prompts/` directory contains complete GenAI interaction logs. All generated artefacts (11 figures, 5 metric CSVs, 2 trained model files, 2 epoch-level training logs) are committed under `outputs/`. The `Dataset/` directory is excluded via `.gitignore`; download instructions with the official UNSW-NB15 source are in `README.md`.

---

## 6. Conclusion

This project demonstrates that the framing of "improved model" is operationally context-dependent. The GenAI-assisted variant (256→128→64, BatchNorm, L2) achieves a 64% reduction in false positive alerts and a 2.8× lower false positive rate—a decisive advantage for alert-fatigue-constrained SOC environments. The human-guided baseline (128→64→32) achieves 9,484 fewer missed attacks and a 9-point higher binary F1—the decisive advantage for security-first deployments. Neither model resolves the minority-class detection problem (Worms, Analysis, Shellcode), identifying inverse-frequency class weighting and targeted oversampling as the highest-priority directions for future work. The comparison also demonstrates that GenAI accelerates architectural hypothesis generation while requiring rigorous experimental validation, and that classifier accuracy alone is an insufficient evaluation metric for operational IDS.

---

## References

[1] Buczak, A. L. & Guven, E. (2016). A survey of data mining and machine learning methods for cyber security intrusion detection. *IEEE Communications Surveys & Tutorials*, 18(2), 1153–1176.

[2] Moustafa, N. & Slay, J. (2016). The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set. *Information Security Journal: A Global Perspective*, 25(1–3), 18–31.

[3] Faker, O. & Dogru, E. (2019). Intrusion detection using big data and deep learning techniques. In *Proceedings of the 2019 ACM SE Conference*. ACM.

[4] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15, 1929–1958.

[5] Ioffe, S. & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of ICML 2015*. PMLR.

---

## AI Acknowledgment Table

| AI Tool | Version / Platform | Task | Part | Suggestions Adopted | Suggestions Rejected |
|---|---|---|---|---|---|
| Claude 3.5 Sonnet | Anthropic, claude.ai | Architecture expansion | Part B | Wider layers 256→128→64; BatchNorm; graduated Dropout 0.4/0.3/0.2 | 4th hidden layer |
| Claude 3.5 Sonnet | Anthropic, claude.ai | Hyperparameter tuning | Part B | lr = 5×10⁻⁴; batch = 256; weight\_decay = 1×10⁻⁴ | Class-weighted CrossEntropyLoss |
| Claude 3.5 Sonnet | Anthropic, claude.ai | Alert threshold validation | Part B | FN-soft metric explicitly reported and analysed | Post-training temperature scaling |
| Claude 3.5 Sonnet | Anthropic, claude.ai | Metric interpretation | Part B | Macro-F1 and weighted-F1 both reported with explicit discussion | — |

**Part A (all design, code, experiments), Part C (Comparative Reflection), and all evaluation analysis** were completed entirely without GenAI assistance. Full prompt logs and rejection reasoning are documented in `prompts/`.
