# AI Tool Acknowledgment

**Module:** 6009CMD Artificial Neural Networks  
**Coventry University Academic Integrity Declaration**

---

## Declaration

In accordance with Coventry University's policy on the use of Generative AI
tools, this document discloses all AI tool usage within this coursework.
Generative AI was used exclusively in **Part B** of this submission.
**Part A was completed entirely without GenAI assistance.**

---

## AI Usage Table

| AI Tool | Version / Platform | Task | Part | What was adopted | What was rejected |
|---|---|---|---|---|---|
| Claude 3.5 Sonnet | Anthropic, claude.ai | Architecture design suggestions | Part B | Wider layers (256→128→64), BatchNorm, graduated dropout (0.4/0.3/0.2) | 4th hidden layer suggestion |
| Claude 3.5 Sonnet | Anthropic, claude.ai | Hyperparameter tuning guidance | Part B | lr=5e-4, batch=256, weight_decay=1e-4 | Class-weighted loss (scope reasons) |
| Claude 3.5 Sonnet | Anthropic, claude.ai | Alert threshold validation | Part B | FN-soft metric explicitly reported | Temperature scaling post-training |
| Claude 3.5 Sonnet | Anthropic, claude.ai | Metric interpretation (accuracy vs macro-F1) | Part B | Both metrics reported with explicit discussion | — |

---

## Scope of GenAI Use

GenAI was **not** used for:

- Any code in `src/part_a_human/` — written entirely by the student
- The EDA analysis and observations
- The streaming simulation design
- The alert tier system design and thresholds
- Any content in the Comparative Reflection (Part C)
- Data collection, preprocessing decisions, or evaluation framework
- Running experiments — all training was executed locally by the student

---

## Verification

All adopted GenAI suggestions were:
1. Critically evaluated by the student before implementation
2. Verified by empirical training runs on the actual UNSW-NB15 dataset
3. Documented with reasoning in `prompts/prompts_used_part_b.txt`
4. Contrasted against rejected suggestions in `prompts/prompts_rejected_or_partially_used.txt`

Full prompt logs are available in the `prompts/` directory.
