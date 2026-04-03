# Project TODO — Before Submission
# 6009CMD Artificial Neural Networks

---

## MUST DO (submission blockers)

- [x] **Run Part B** — COMPLETE. All outputs generated.
      ```bash
      source venv/bin/activate
      python -m src.part_b_genai.train_improved
      ```
      Expected outputs:
        - `outputs/figures/training_curves_improved.png`
        - `outputs/figures/confusion_matrix_improved.png`
        - `outputs/figures/confidence_distribution_improved.png`
        - `outputs/figures/alert_distribution_improved.png`
        - `outputs/figures/comparison_baseline_vs_improved.png`
        - `outputs/tables/metrics_improved.csv`
        - `outputs/tables/comparison_summary.csv`
        - `outputs/models/improved_model.pt`

- [x] **Report contains real Part B metrics** — all values are actual results
      from training runs, not placeholders.

- [ ] **Create GitHub repository** (private) and push all files

- [ ] **Make at least 10 meaningful commits** spanning the development timeline
      (see `docs/github_submission_checklist.md` for suggested commit messages)

- [ ] **Record the 5-minute video** using `docs/video_script_5min.md` as guide

---

## SHOULD DO (quality improvements)

- [ ] Check final word count of report: `wc -w report/report-final.md`
      (must be ≤ 2500; soft limit starts at 2275)

- [ ] Verify all figure references in report match actual filenames in
      `outputs/figures/`

- [ ] Add video link/file reference to README.md once recorded

- [ ] Proofread Comparative Reflection section (500–700 word target)

---

## NICE TO HAVE (First Class polish)

- [ ] Add per-class F1 comparison bar chart between Baseline and Improved
      (highlights minority-class improvements specifically)

- [ ] Add a short Limitations subsection to each Part (one paragraph each)

- [ ] Add dataset download instructions to README.md with link to the
      official UNSW-NB15 page

---

## Already done ✓

- [x] Project structure created (`src/`, `outputs/`, `prompts/`, `docs/`)
- [x] `src/common/preprocessing.py` — shared pipeline
- [x] `src/common/utils.py` — shared model, training, evaluation, plotting
- [x] `src/part_a_human/train_baseline.py` — Part A complete
- [x] `src/part_b_genai/train_improved.py` — Part B complete (needs run)
- [x] `requirements.txt` and `.gitignore`
- [x] Part A training run — real metrics obtained
- [x] All Part A outputs generated (figures, CSVs, model, scaler)
- [x] `prompts/prompts_used_part_b.txt`
- [x] `prompts/prompts_rejected_or_partially_used.txt`
- [x] `prompts/genai_reflection_notes.md`
- [x] `docs/ai_acknowledgment.md`
- [x] `docs/video_script_5min.md`
- [x] `docs/github_submission_checklist.md`
- [x] `report/report-final.md` (Part B metrics need updating after run)
- [x] `README.md`
