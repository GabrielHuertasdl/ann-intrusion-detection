# GenAI Reflection Notes — Part B Design Process

**Course:** 6009CMD Artificial Neural Networks, Coventry University  
**Tool used:** Claude 3.5 Sonnet (Anthropic), via claude.ai  
**Purpose:** Document the student's critical engagement with GenAI during Part B

---

## What GenAI did well

**Architecture reasoning.** Claude provided well-grounded justifications for
wider layers and BatchNorm in the context of mixed-scale tabular features.
The explanation of why graduated dropout (stronger near input) works better
than uniform dropout on high-dimensional one-hot + continuous feature spaces
was technically sound and aligned with published regularisation theory.

**Metric interpretation.** Claude correctly identified the accuracy–macro-F1
gap and its cause in class imbalance without being prompted about the
specific numbers.  The suggestion to report both weighted-F1 and macro-F1
as complementary lenses was a useful addition to the evaluation framework.

**Alert threshold analysis.** Claude's framing of threshold selection as a
precision-recall operating point on the alert system was appropriate and
operationally relevant.  The suggestion to track FN-soft separately from
FN-hard was genuinely useful for the alert system evaluation.

---

## Where GenAI required critical oversight

**Over-engineering tendency.** Claude consistently suggested more complex
solutions than the task required: a 4th hidden layer, self-attention, SMOTE
preprocessing, temperature scaling.  Without critical filtering, these
suggestions would have made the project scope balloon and obscured the
clear Part A vs Part B comparison the brief requires.

**Lack of scope awareness.** GenAI does not understand the constraints of an
academic coursework: word limits, fair comparison requirements, CPU
constraints, submission timelines.  Every suggestion had to be evaluated
against these constraints, not just on technical merit.

**No awareness of actual results.** Claude could not run the code or see real
metrics.  Its suggestions about expected improvements were directionally
correct but could not be validated against actual training behaviour.  The
student had to run experiments to verify whether BatchNorm + wider layers
actually improved macro-F1 meaningfully on *this* dataset split.

---

## Revised view after seeing real results

After running both models:

- The improved model's gains on **macro-F1 and weighted-F1** confirm that
  GenAI's architecture suggestions were directionally correct.
- The **binary recall** improvement and **FN_hard reduction** validate the
  claim that wider capacity + BN helps minority-class detection.
- However, FPR and FP_alerts may increase slightly — consistent with
  GenAI's warning about the precision-recall trade-off.

The critical insight: GenAI provided *hypotheses* about what should improve.
The student's role was to run experiments, interpret results against real
data, and synthesise a technically honest comparison.  GenAI did not — and
cannot — replace that experimental and analytical process.

---

## Summary: appropriate use of GenAI in ML coursework

| Task | GenAI appropriate? | Student action required |
|------|-------------------|------------------------|
| Architecture suggestion | Yes — generates options | Evaluate against scope, data, constraints |
| Hyperparameter guidance | Partially — direction only | Verify empirically |
| Code generation | Yes — boilerplate only | Review, test, adapt |
| Metric interpretation | Yes — useful framing | Validate against actual results |
| Result analysis | No — cannot see data | Student must do this entirely |
| Critical reflection | No — cannot be delegated | Student must do this entirely |
