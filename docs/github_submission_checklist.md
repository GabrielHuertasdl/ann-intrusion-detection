# GitHub Submission Checklist
# 6009CMD ANN — Pre-submission Verification

---

## Repository setup

- [ ] Repository is **private** (brief requires private repo)
- [ ] Collaborator access granted to marker (check brief for their GitHub username)
- [ ] Repository name is clear and professional (e.g., `6009CMD-ANN-IDS`)
- [ ] Repository has at minimum **10 commits** — brief expects "regular commits"
- [ ] Commits span the development period (not all pushed in one batch)
- [ ] Commit messages are descriptive (not "update", "fix", "stuff")

Suggested commit history:
```
feat: initial project structure and dataset loading
feat: preprocessing pipeline with one-hot encoding and StandardScaler
feat: Part A baseline ANN 128-64-32 with training loop
feat: Part A evaluation — multiclass, binary, confidence metrics
feat: Part A alert system and streaming simulation
feat: Part A EDA visualisations and CSV exports
feat: Part B improved model 256-128-64 with BatchNorm and L2
feat: Part B GenAI prompt documentation
feat: comparison outputs and summary CSV
docs: README, requirements, AI acknowledgment
docs: final report and video script
```

---

## Structure check

- [ ] `src/part_a_human/train_baseline.py` exists and runs cleanly
- [ ] `src/part_b_genai/train_improved.py` exists and runs cleanly
- [ ] `src/common/preprocessing.py` and `utils.py` exist
- [ ] `outputs/figures/` contains PNG plots (or instructions to generate them)
- [ ] `outputs/tables/` contains CSV files with metrics
- [ ] `outputs/models/` contains saved `.pt` model files (or note about size)
- [ ] `prompts/prompts_used_part_b.txt` is present and substantive
- [ ] `prompts/prompts_rejected_or_partially_used.txt` is present
- [ ] `prompts/genai_reflection_notes.md` is present
- [ ] `docs/ai_acknowledgment.md` is present with the AI usage table
- [ ] `README.md` is present and informative
- [ ] `requirements.txt` is present and accurate
- [ ] `.gitignore` excludes `venv/`, `__pycache__/`, large CSVs

---

## Report check

- [ ] `report/report-final.md` is present
- [ ] Word count is ≤ 2500 words (run `wc -w report/report-final.md`)
- [ ] Report has sections: Abstract, Introduction, Part A, Part B,
      Comparative Reflection, GitHub, Conclusion, References, AI Table
- [ ] Comparative Reflection is 500–700 words
- [ ] References use consistent citation style (IEEE or Harvard)
- [ ] AI Acknowledgment table matches `docs/ai_acknowledgment.md`
- [ ] No mention of Part A using any AI tool

---

## Code quality check

- [ ] Both scripts run end-to-end from a clean environment:
      ```bash
      pip install -r requirements.txt
      python -m src.part_a_human.train_baseline
      python -m src.part_b_genai.train_improved
      ```
- [ ] No hardcoded absolute paths (use `Path(__file__).resolve().parents`)
- [ ] Random seeds set for reproducibility (SEED=42 in preprocessing.py)
- [ ] No data leakage: scaler fit on train only, test columns aligned to train

---

## Dataset note

The UNSW-NB15 CSVs are not committed to the repository due to file size.
Add a note in README.md with download instructions:
  - Official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
  - Place files in `Dataset/` directory
  - Files required: `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`

---

## Final checks

- [ ] Video is ≤ 5 minutes (re-record if over)
- [ ] Video link or file reference included in README or report
- [ ] Submission portal upload complete (if required separately from GitHub)
- [ ] Module submission deadline confirmed
