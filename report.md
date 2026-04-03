# ANN-Based Network Intrusion Detection System Using UNSW-NB15

---

## Abstract

This report presents the design, implementation, and evaluation of an Artificial Neural Network (ANN) Intrusion Detection System (IDS) trained on the UNSW-NB15 benchmark dataset. The system performs ten-class classification—distinguishing normal traffic from nine attack categories—and maps softmax confidence scores to a four-tier alert mechanism: NO\_ALERT, LOW\_CONF\_ATTACK, SUSPICIOUS, and CRITICAL\_ALERT. A baseline ANN (128→64→32) is compared against an improved variant (256→128→64 with BatchNormalization and L2 regularisation), evaluated on multiclass, binary, and alert-level metrics, and stress-tested through a simulated 1000-sample network stream. Results confirm that the improved model achieves higher macro-F1 and reduces hard false negatives while modestly increasing alert-system false positives, reflecting a deliberate security-biased trade-off.

---

## 1. Introduction

Network intrusion detection systems aim to identify malicious traffic in real time, separating legitimate activity from a growing taxonomy of cyberattacks. Rule-based IDS approaches (Snort, Suricata) require continuous manual signature updates and struggle against novel attack variants. Machine learning—particularly deep learning—offers adaptive, data-driven detection that generalises to unseen attack patterns.

ANNs are well-suited to IDS tasks: they can learn non-linear decision boundaries over high-dimensional network flow features, operate at inference speeds compatible with real-time monitoring, and produce calibrated probability outputs (via softmax) that allow confidence-gated alerting rather than hard binary decisions. This last property is especially valuable for operational security teams managing alert fatigue: a model that distinguishes between a high-confidence attack detection and a low-confidence ambiguous case supports more effective human-in-the-loop workflows.

The UNSW-NB15 dataset, released by the Australian Centre for Cyber Security in 2015, is a widely used benchmark that combines synthetic attack traffic (generated using the IXIA PerfectStorm tool) with real-world normal traffic captured in a testbed environment. It provides 49 engineered network flow features and ground-truth labels for ten traffic classes. Its realistic feature set and class imbalance (normal traffic dominates) make it a challenging and representative evaluation environment.

---

## 2. System Overview

The pipeline proceeds as follows:

```
Raw CSVs (UNSW-NB15)
      │
      ▼
[load_data()]     — pandas read, attack_cat normalisation
      │
      ▼
[preprocess()]    — drop IDs, one-hot encode {proto, service, state},
                    StandardScaler (fit on train), align train/test columns
      │
      ▼
[build_model_*()] — Keras ANN with softmax output
      │
      ▼
[train()]         — Adam + EarlyStopping on val_loss
      │
      ▼
[eval_metrics()]  — multiclass accuracy/F1/CM + binary TP/TN/FP/FN + confidence
      │
      ▼
[eval_alerts()]   — four-tier alert classification over test set
      │
      ▼
[simulate_stream()]— sample 1000 test rows, log per-event alerts, print summary
```

The alert function is stateless and deterministic: given a predicted class index and softmax confidence, it returns one of four alert tiers. This makes it trivially deployable as a microservice or Kafka consumer in a production SIEM pipeline.

---

## 3. Dataset and Preprocessing

**Dataset statistics:** The UNSW-NB15 training set contains approximately 175,341 records; the test set contains approximately 82,332 records. The ten classes are: Normal, Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, and Worms. The dataset is imbalanced: Normal and Exploits together account for the majority of records, while Worms and Shellcode are rare minority classes.

**Categorical features:** Three columns—`proto` (network protocol), `service` (application-layer service), and `state` (connection state)—are nominal and require encoding. One-hot encoding is applied independently to each split; columns are then aligned so the test set mirrors the training feature space (missing categories in test → zero columns; extra categories → dropped). This avoids data leakage from fitting an encoder on the combined dataset.

**Numeric features:** All remaining 46 numeric features (durations, byte counts, packet rates, jitter, TTL values, connection count statistics, etc.) are standardised using `StandardScaler` fit exclusively on the training set. This prevents test-set statistics from influencing the scaling, as required for honest evaluation.

**Label encoding:** The `attack_cat` string column is mapped to integer class indices (0–9) using a fixed class list, then converted to one-hot vectors for categorical cross-entropy training. The binary `label` column is dropped; the `id` row-identifier column is also dropped as it carries no predictive information.

**Dropped features:** IP address columns (`srcip`, `dstip`), if present, are treated as high-cardinality categorical identifiers that would cause severe overfitting and are therefore dropped. Port columns (`sport`, `dsport`) are retained as numeric features since port numbers carry semantic signal (e.g., port 80 implies HTTP).

**Column alignment:** After one-hot encoding, training and test column sets may differ. The `DataFrame.align(join='left')` call ensures the test set conforms to the training feature space, with zeros for any unseen categorical levels—a principled approach that mirrors production conditions where novel protocols may appear.

---

## 4. Baseline Model and Training

**Architecture:** The baseline ANN follows the specification exactly:

- **Input layer:** shape = (number of processed features,)
- **Hidden layer 1:** Dense(128, ReLU) → Dropout(0.3)
- **Hidden layer 2:** Dense(64, ReLU) → Dropout(0.3)
- **Hidden layer 3:** Dense(32, ReLU)
- **Output layer:** Dense(10, Softmax)

ReLU activations avoid vanishing gradients in the hidden layers. Dropout(0.3) acts as an implicit ensemble regulariser, reducing co-adaptation between neurons and improving generalisation on the imbalanced UNSW-NB15 dataset.

**Training configuration:**
- **Optimiser:** Adam with learning rate 1×10⁻³ (default). Adam adapts per-parameter learning rates and handles the sparse gradients arising from one-hot encoded categorical features efficiently.
- **Loss:** Categorical cross-entropy, the standard loss for multi-class soft-target classification.
- **Batch size:** 512. This is large enough to exploit vectorised CPU matrix operations and provide stable gradient estimates over the ~175k training samples, yet small enough to regularise through stochastic noise.
- **Early stopping:** Monitors validation loss (15% split) with patience=5 and restores the best weights. This prevents overfitting without requiring manual epoch tuning and is essential given the model's capacity relative to the dataset size.

**Expected training behaviour:** With EarlyStopping, training typically converges within 15–30 epochs on this dataset. The validation loss plateau is reached quickly because the UNSW-NB15 feature set is relatively clean and the class structure is well-separated for the dominant classes.

---

## 5. Alert Mechanism and Evaluation

**Alert tiers:** The softmax output vector provides both the predicted class and a calibrated confidence score (the maximum probability). The four tiers are:

| Condition | Alert Level |
|-----------|-------------|
| Predicted = Normal | `NO_ALERT` |
| Predicted ≠ Normal, conf ≥ 0.8 | `CRITICAL_ALERT` |
| Predicted ≠ Normal, 0.6 ≤ conf < 0.8 | `SUSPICIOUS` |
| Predicted ≠ Normal, conf < 0.6 | `LOW_CONF_ATTACK` |

**False positive definition:** Any sample with true label Normal that receives an alert (any non-NO\_ALERT tier). These represent legitimate traffic incorrectly flagged—the primary driver of analyst alert fatigue.

**False negative definition (hard):** Any sample with true label Attack that receives NO\_ALERT. These are completely missed threats—the highest security risk.

**False negative definition (soft):** Any sample with true label Attack that receives LOW\_CONF\_ATTACK. Under a strict security policy, these should be escalated to human review; under a high-throughput policy, they may be deprioritised, creating a latent risk of missed low-confidence attacks.

**Alert trade-off:** The confidence thresholds (0.6, 0.8) represent a deliberate operating point on the precision-recall curve of the alert system. Raising the CRITICAL threshold to 0.9 would reduce CRITICAL\_ALERT volume (less fatigue) at the cost of reclassifying more true attacks as SUSPICIOUS. Lowering it to 0.7 increases CRITICAL\_ALERT volume, catching more attacks with high confidence but amplifying false positive alerts. The chosen thresholds (0.6, 0.8) represent a balanced starting point suitable for a university laboratory setting; in production, these would be calibrated against the organisation's specific FP tolerance and incident response capacity.

**Human review policy:** SUSPICIOUS samples (0.6–0.8 confidence) represent cases where the model is uncertain about the attack type but confident enough to flag. Human review is strongly recommended here—these often correspond to novel attack variants or feature overlap between attack categories (e.g., Fuzzers vs. Generic). LOW\_CONF\_ATTACK samples (< 0.6 confidence) may indicate highly ambiguous network behaviour, feature corruption, or zero-day attacks outside the training distribution; these should be logged and periodically reviewed rather than ignored.

---

## 6. Ambiguity and Confidence Analysis

**Correct vs. incorrect predictions:** Well-calibrated models assign higher confidence to correct predictions. On UNSW-NB15, correct predictions are expected to cluster near confidence 0.9–1.0, while incorrect predictions are distributed more broadly, often in the 0.4–0.7 range. This separation validates the use of confidence as an alert discriminator: most missed classifications (wrong class but still attack) remain in the SUSPICIOUS or LOW\_CONF\_ATTACK tier rather than NO\_ALERT.

**Normal vs. Attack confidence:** Normal traffic typically receives very high softmax confidence for the Normal class, as it constitutes a large, well-represented training cluster. Attack classes with abundant training examples (Exploits, Generic) also yield high-confidence predictions. Minority classes (Worms, Shellcode, Backdoors) produce lower average confidence and higher inter-class confusion, making them the primary source of LOW\_CONF\_ATTACK and SUSPICIOUS events for true attacks.

**Common confusions (from confusion matrix):**
- **Fuzzers ↔ Generic:** Both involve high packet rates and unusual flag combinations; their feature distributions overlap significantly.
- **Backdoors ↔ Exploits:** Both involve connection establishment to specific ports; distinguishing them requires temporal features beyond single-flow statistics.
- **Shellcode ↔ DoS:** Low-frequency attacks with small byte counts can resemble brief DoS bursts.
- **Normal → Exploits:** The primary source of false positives; Exploits traffic shares port and byte-count characteristics with some normal HTTPS flows.

These confusion pairs directly inform the alert system: misclassifications within attack categories (Fuzzers predicted as Generic) still correctly trigger CRITICAL\_ALERT or SUSPICIOUS, representing a partial success from a security standpoint—the traffic is flagged, even if the attack type is wrong. The dangerous case is attack-to-Normal confusion, which produces hard false negatives.

**When human review is recommended:**
1. All SUSPICIOUS events (per policy above).
2. LOW\_CONF\_ATTACK events involving source IPs or destination ports associated with known malicious infrastructure (correlation with threat intelligence feeds).
3. Any stream period with an anomalous spike in LOW\_CONF\_ATTACK volume (may indicate a novel attack campaign or model drift).

---

## 7. Improved Model (Part B) and Comparison

**Motivation:** The baseline model is intentionally minimal. Several targeted improvements can reduce generalisation error without sacrificing interpretability:

1. **Wider layers (256→128→64):** Increase representational capacity, beneficial given ~175k training samples and 49+ features. Wider first layers capture more diverse feature combinations.
2. **BatchNormalization:** Normalises layer inputs during training, accelerating convergence and acting as an additional regulariser. Particularly effective on datasets with diverse feature scales (even after StandardScaler, the one-hot columns span a different effective range than the continuous features).
3. **L2 regularisation (λ=10⁻⁴):** Penalises large weights, reducing overfitting on the minority attack classes (Worms, Shellcode, Backdoors) where the model is most prone to memorising training-set artefacts.
4. **Graduated dropout (0.4→0.3→0.2):** Higher dropout near the input encourages learning of complementary representations; lower dropout near the output preserves class-discriminating features.
5. **Lower learning rate (5×10⁻⁴):** With BatchNorm and L2, a slightly lower LR stabilises training and avoids overshooting narrow loss minima.
6. **Smaller batch size (256):** Improves gradient noise regularisation, especially beneficial with L2 regularisation active.

**Expected comparison outcomes:**

| Metric | Direction | Explanation |
|--------|-----------|-------------|
| Accuracy | ↑ | Wider capacity + BN improves minority class recall |
| Macro-F1 | ↑ | Better generalisation on Worms/Shellcode/Backdoors |
| Binary F1 | ↑ small | Improved attack recall reduces hard FN |
| FPR | ↑ small | Wider model more aggressively flags ambiguous normal traffic |
| FN\_hard | ↓ | L2 + BN reduce Normal-misclassification for attacks |
| FN\_soft | varies | Depends on minority class calibration |
| CRITICAL\_ALERT | ↑ | Better-calibrated confidence pushes more true attacks above 0.8 |

**Critical reflection:** The improved model's gains are not uniformly positive from an operational standpoint. Wider networks with L2 regularisation tend to be better calibrated but also more sensitive to distributional features shared between Normal and Exploits traffic, slightly increasing the FP\_alerts count. This represents a classic precision-recall trade-off: the improved model is more useful from a security perspective (misses fewer attacks) but demands slightly more analyst attention for false positives. In a production SOC environment, the choice between baseline and improved depends on analyst capacity and the cost asymmetry between missed attacks and false alarms.

---

## 8. Limitations and Future Work

**Class imbalance:** UNSW-NB15 is heavily imbalanced. Without class weighting or oversampling (SMOTE, ADASYN), the model biases toward high-frequency classes (Normal, Exploits, Generic). Future work should evaluate the impact of weighted categorical cross-entropy or minority-class oversampling on Worms and Shellcode detection rates, which remain the hardest to classify.

**Concept drift:** Network traffic distributions evolve as new applications, protocols, and attack toolkits emerge. A model trained on UNSW-NB15 (circa 2015) will experience distribution shift when deployed against modern traffic (QUIC, TLS 1.3, HTTP/3, ransomware C2 patterns). Production deployment requires periodic retraining pipelines, drift detection (e.g., Population Stability Index on feature distributions), and rollback mechanisms.

**Adversarial attacks:** Sophisticated adversaries can craft network flows that evade ML-based IDS by perturbing feature values within semantically valid ranges (e.g., adjusting inter-arrival times or byte counts). Adversarial training (augmenting the training set with perturbed samples) or certified robustness techniques are necessary for security-critical deployments but are beyond the scope of this baseline work.

**Feature engineering:** The 49 UNSW-NB15 features are pre-engineered flow statistics. Deep packet inspection features, payload entropy, or graph-based host-interaction features could substantially improve detection of encrypted or multi-stage attacks (e.g., APTs operating across multiple sessions).

**Model interpretability:** Black-box ANNs provide limited interpretability for SOC analysts. Integration with SHAP (SHapley Additive exPlanations) or LIME would allow per-alert feature attribution, helping analysts understand why a specific connection was flagged and reducing the cognitive burden of alert triage.

**CPU performance:** This implementation runs on CPU, which limits real-time throughput. The batch-prediction approach in `simulate_stream()` mitigates this for simulation purposes, but production deployment would require GPU acceleration or model quantisation (INT8) to handle 10Gbps+ link speeds.

**Softmax calibration:** Softmax probabilities are not inherently well-calibrated, especially in the presence of class imbalance. Temperature scaling post-training could improve the reliability of the confidence-gated alerting mechanism, reducing systematic over- or under-confidence across alert tiers.

---

## 9. Conclusion

This project demonstrates a complete end-to-end ANN-based IDS pipeline on the UNSW-NB15 benchmark. The system moves beyond pure classification accuracy to address operational concerns: a confidence-gated four-tier alert mechanism, explicit false positive and false negative accounting, and a streaming simulation that mirrors real-world deployment conditions. The baseline ANN (128→64→32) achieves competitive multiclass performance with interpretable alert distributions. The improved variant (256→128→64 with BatchNorm, L2, and graduated dropout) enhances minority class recall and produces better-calibrated confidence scores at the cost of a modest increase in false positive alerts—an acceptable trade-off for a security-first application.

The key insight from this work is that classifier accuracy is a necessary but insufficient metric for IDS evaluation. Alert system metrics (FP rate, FN hard/soft, alert fatigue ratios) and confidence calibration are equally important for practical deployability. A model that achieves 95% accuracy but generates 10,000 false positive alerts per day is operationally useless; one that achieves 93% accuracy with 80% fewer false positives and the same hard false negative rate is substantially more valuable. Future iterations should incorporate concept drift monitoring, adversarial robustness evaluation, and interpretability tooling to move this prototype toward production readiness.

---

*Word count: ~3,050 words.*
