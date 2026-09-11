# Project architecture and phase contracts

## Data layers

1. `data/raw`: immutable official CSV inputs. No code writes here.
2. `data/cache`: dtype-controlled Parquet mirrors of raw CSV files, guarded by raw file identity metadata.
3. `data/processed/features_v1_time.parquet`: stock-major causal time-series features, useful for per-stock generation and warm-up checks.
4. `data/processed/features_v1_model_base_by_year/year=YYYY/features.parquet`: the predeclared 36-feature modeling core partitioned by year for bounded-memory cross-sectional transforms and walk-forward CV. The complete 147-column store remains available for later ablation without being duplicated.
5. `data/processed/features_v1_market.parquet`: one row per trade date; joined by `trade_date` only when assembling model views.
6. Future cross-sectional artifacts contain only keys plus rank/z-score columns. Raw features remain in the Phase 3 store rather than being duplicated.
7. `data/processed/model_view_v1_by_year/year=YYYY/data.parquet` is the training-only joined view: keys, real label, limit-up flag, and an explicit 119-feature allowlist. Downstream models must read `feature_names` from its manifest and must never infer predictors by excluding a few columns.
8. `models/phase8_baselines/fold_YYYY/*_oof.parquet` stores full validation universes, including rows with missing labels, because official turnover does not depend on label availability. Predictions are finite float32, and metrics are computed at that exact persisted precision.
9. `experiments/experiment_registry.csv` is append-only by `(experiment_id, record_type, fold)`. Immutable config, source-tree and artifact manifests live under `experiments/configs` and `experiments/manifests`; generated-model paths alone are never considered sufficient provenance.
10. `models/phase10_turnover/fold_YYYY/lightgbm_turnover_oof.parquet` stores the selected causal post-processing result. The full 18-candidate trade-off table is retained as Parquet and CSV under `reports/tables`; only the selected OOF is duplicated.
11. `models/phase11_ensemble/fold_YYYY/ensemble_oof.parquet` stores the selected LightGBM/Ridge daily-rank blend after the locked Phase 10 post-process. The complete 21-weight OOF grid remains a small table under `reports/tables`; source OOF files are referenced by hash rather than copied.
12. `models/phase12_lambdarank/fold_YYYY` stores one LambdaRank model plus raw and locked-turnover OOF files. Training matrices are temporary D-drive memmaps and are deleted after each fold; complete model-view partitions remain the immutable training input.
13. `data/processed/advanced_features_v1_{train,test}_by_year/year=YYYY/features.parquet` stores the 32 Phase 13 causal candidates directly by year. Test partitions are computed with the configured 250-date real training tail but retain test keys only; no monolithic advanced-feature copy is created.
14. Phase 14 screening artifacts remain compact under `reports/tables`: daily feature IC is Parquet; summaries, yearly/regime diagnostics, rank-correlation redundancy and family ablation are CSV; the locked 16-feature allowlist and all rejection reasons are JSON.
15. `data/processed/model_view_v2_atr_rank_by_year` contains only the 47 Phase 12 ranker inputs plus two Phase 14-selected ATR ranks and training metadata. `models/phase15_stability/{full_expanding,trailing_4y,trailing_2y}` stores the three window variants without retaining training memmaps.
16. `models/phase16_final` contains the locked 60% base-ranker, 20% trailing-two-year ATR-ranker and 20% regression-sleeve configuration, three reproducible final-ensemble OOF files, and four full-training production models. The regression sleeve remains 65% LightGBM and 35% Ridge. The 21-row simplex summary plus fold rows are retained under `reports/tables`.
17. `models/phase17_prediction/test_predictions.parquet` is the label-free locked-model prediction artifact. It contains only `ts_code`, `trade_date`, and finite float32 `pred`; the adjacent manifest and daily diagnostics record the 250-date warm-up, component schemas, state sizes, raw-test key hash and prediction fingerprint. Submission CSV creation remains isolated to Phase 18.
18. `submissions/submission_p16_locked_ensemble_v1.csv` is the atomically promoted competition-delivery file. It is built only from the Phase 17 Parquet after an in-memory contract check, then reloaded from a temporary CSV and checked again before promotion. The Phase 18 audit records both validation passes and both source/output hashes.
19. `output/pdf/fintechathon2026_task5_report.pdf` is the fixed eight-page Phase 19 report. Its five figures are regenerated only from prior audit tables; `FINAL_DELIVERY_MANIFEST.json` and `phase19_delivery_audit.json` record report, submission and reproducibility fingerprints. `PROJECT_GUIDE.md` maps the complete flow, file responsibilities and verification commands.

All large intermediate artifacts remain on D:. Generated artifacts are ignored by Git; code, configuration, manifests and documented hashes are the reproducibility contract.

## Leakage boundaries

- Production feature code may use row `t` and earlier rows of the same stock. It may not use `shift(-1)`.
- Same-day market and cross-sectional aggregates may use the full date-`t` competition universe, but never another date.
- Negative shifts are confined to `src/data/label_audit.py` and are never exported as features.
- `y_ret_1d` is joined only after feature construction, for training/validation rows where it is non-null.
- Phase 13 advanced features accept only the explicit raw feature inputs. Adjusted-price VWAP proxy uses rolling `close*vol` weights because raw `amount/vol` is not on a stable adjusted-price scale; amount is otherwise used only in scale-free ratios.
- Phase 14 selection uses labels from 2018–2021 only. The 2022–2024 holdout statistics, extreme-market diagnostics and family ablation may validate stability but may not change the allowlist or feature directions.
- Phase 15 training windows are derived from each already-purged fold calendar and end before validation. Regime thresholds remain the Phase 14 screen-period thresholds; OOF regime analysis never refits thresholds on validation years.
- Preprocessors are fit inside each training fold. No random split is permitted.
- Prediction smoothing first forms a same-date rank, then applies the recurrence within each stock in ascending date order. Hysteresis may use the current eligible ranking and the immediately preceding eligible Top set only; neither transform accepts labels as an input.

## Locked downstream contracts

- Phase 4 transforms only the predeclared `cross_sectional.source_features`; each year is processed independently by date.
- Phase 6 uses the expanding folds in `config.yaml` with a one-trading-day purge before validation because labels refer to the next row.
- Phase 7 must reproduce the vendored official evaluator byte-for-byte in behavior. Constants marked locked in `config.yaml` are documentation, not tuning parameters.
- Phase 8 and later read year partitions instead of loading the 5.23 GB stock-major feature file repeatedly.
- Fold summaries are equal-weighted. Concatenating OOF years and scoring once is diagnostic only because it would add artificial turnover comparisons across fold boundaries.
- Submission generation must join back to the untouched test keys and validate exact key-set equality.
- Reusing an experiment key with changed content is an error. Identical registration is idempotent, and a new parameter/data/source variant must receive a new experiment ID.
- Phase 10 parameters are selected by equal-weight mean official score across the three isolated folds. EMA and membership state reset at each validation boundary; the production API accepts explicit prior EMA and Top-set state for causal train-to-test continuation.
- Hysteresis maintains the official exact daily Top size. It only permutes unique ordinal scores among non-limit-up rows, leaving the daily score multiset and limit-up positions unchanged; sub-100 eligible days clear membership exactly as the official evaluator clears turnover state.
- Phase 11 aligns complete OOF universes by exact keys and evaluator metadata before blending. Every source is independently ranked inside the current date; weights are nonnegative and sum to one, so raw model scale cannot dominate the ensemble.
- Ensemble candidates share the Phase 10-selected smoothing and hysteresis parameters. The LightGBM-only endpoint must exactly reproduce Phase 10 before the grid can be accepted; this prevents an unnoticed post-processing or evaluation drift from being misreported as ensemble gain.
- LambdaRank relevance is derived only from non-null training-fold labels within the same trade date. Parquet batches must be coalesced at date boundaries, group dates must exactly equal the ordered training calendar, and group sizes must sum to the labeled training rows before fitting.
- Phase 12 turnover parameters remain the Phase 10 values; the ranking model is compared with regression using the same feature allowlist, fold dates and boosting rounds. Ranking is accepted by official OOF score, not by training NDCG alone.
- The post-Phase-12 incumbent is `p12_lambdarank_turnover_v1`. Phase 13–15 candidates must compare mean, worst fold, fold standard deviation and turnover against it; Phase 16 must revisit the ensemble with ranking OOF included.
- Phase 13 is construction-only: no label screening or model fit is permitted. Phase 14 owns within-date IC, temporal stability, coverage/cost diagnostics and family ablation before any expensive ranker retrain.
- Phase 14 fills missing same-date percentile ranks with the fixed neutral value `0.5`, reports coverage separately and selects at most 16 candidates after threshold, family-cap and mean daily rank-correlation checks. Its equal-rank family composites are screening proxies, not substitutes for fold-isolated LambdaRank results.
- Phase 15 changes only the two ATR ranks and training-window history; relevance, model parameters, validation universes and turnover controls match Phase 12. The two-year ATR result is a robustness candidate for Phase 16, while Phase 12 remains incumbent because all ATR-window mean scores are lower.
- A resumed Phase 15 fold is accepted only when its model and both OOF files exist and the OOF contract and official metrics replay successfully. Identical train/validation signatures may reuse byte-identical artifacts and must be recorded explicitly.
- Phase 16 independently ranks each raw component by current date, blends nonnegative weights summing to one, and applies the frozen Phase 10 post-process exactly once. Its three simplex endpoints must reproduce the Phase 12 ranker, Phase 15 trailing-two-year ATR model and Phase 11 regression ensemble metrics within `1e-12` before a candidate can be locked.
- The production component schemas are explicit and unequal by design: base ranker and regression LightGBM use 47 rank/market features, ATR ranker uses the compact 49-feature view, and Ridge uses 36 cross-sectional z-scores. Phase 17 must validate each saved feature list rather than apply one inferred schema to every model.
- Production post-processing may not reset at the test boundary. The final model must predict at least the configured 250-date training tail and pass its EMA and eligible Top-set state into the first test date.
- Phase 17 reads existing test cross-sectional, market and advanced-feature partitions without rebuilding them. ATR inputs are ranked independently within each current test date with the same neutral-missing rule as training; model weights and parameters remain immutable.
- The Phase 17 Parquet preserves the untouched raw-test row order and exact key set. It is an intermediate prediction artifact, not an authorized competition submission; Phase 18 owns CSV formatting, round-trip validation and non-overwrite naming.
- Phase 18 requires exactly `ts_code, trade_date, pred`, one row per untouched test key, finite numeric predictions, valid stock/date formats, unchanged inner-merge cardinality and float32-exact CSV round-trip. Existing submission paths are rejected unless the caller explicitly passes `--overwrite`.
- Hidden test metrics are never synthesized. The official evaluator can only be run on test data when authentic official test labels are available; structural submission validation is recorded separately from performance evaluation.
- Phase 19 is documentation-only: it consumes locked audits and delivery artifacts, does not retrain, does not alter prediction/submission values, and labels every score as walk-forward OOF rather than hidden-test performance.

## Artifact safety

- Stable outputs are written through temporary files/directories and promoted only after success.
- Existing partition directories are never overwritten unless the caller passes an explicit `--overwrite` flag.
- Feature manifests record row counts, schemas, key hashes, missing/finite coverage and source fingerprints where practical.
