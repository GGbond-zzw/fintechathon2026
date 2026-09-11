# Final reproducibility guide

## Environment

- Python 3.12
- Install: `.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`
- All configured data, caches, models, reports and submissions are on `D:\codex\fintechathon2026`.

## Immutable inputs

- Train SHA-256: `E9D23E87F7E1F9439F28F7DD951D129EA49EAACDA5AD650DA8576A830A6DEAC9`
- Test SHA-256: `DCEFFBF68BBAB07A254BC6CAA4D8A1E33DE8DE9C15489D936C5D15020DBFA90F`
- Official evaluator SHA-256: `72DC59987E92BA8AE6B506E114DFE8591B5608C9BECCD0769D5D1F778F9189C2`

## Run order

Run numbered scripts in `scripts/` from 01 through 19. Expensive existing outputs reject overwrite by default; use explicit overwrite flags only when intentionally rebuilding that phase. Phase 15 supports `--resume` for complete fold artifacts.

The final production sequence is:

```powershell
.\.venv\Scripts\python.exe scripts\16_finalize_model.py
.\.venv\Scripts\python.exe scripts\17_predict_test.py
.\.venv\Scripts\python.exe scripts\18_build_submission.py
.\.venv\Scripts\python.exe scripts\19_build_final_report.py
```

## Final outputs

- Model manifest: `models/phase16_final/manifest.json`
- Prediction Parquet: `models/phase17_prediction/test_predictions.parquet`
- Submission: `submissions/submission_p16_locked_ensemble_v1.csv`
- Report PDF: `output/pdf/fintechathon2026_task5_report.pdf`
- Experiment registry: `experiments/experiment_registry.csv`

No test label is created or inferred. Structural submission validation is not a hidden-test performance score.
