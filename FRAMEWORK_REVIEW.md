# Framework review before Phase 4

Review date: 2026-09-09

## Outcome

The project phase sequence, leakage boundaries and raw/cache/processed separation are sound. Four adjustments were required before continuing.

1. The complete 147-column time-feature file is stock-major and too expensive to scan repeatedly for same-date transforms on a 16 GB machine. A predeclared 36-feature modeling core is now partitioned by year; the full store remains available for later ablation.
2. `market_up_ratio` and `market_down_ratio` previously counted missing returns as zero-direction observations. They now exclude missing returns from the denominator, and the market artifact was rebuilt.
3. Expanding CV dates, a one-trading-day label purge, locked evaluator constants and the official evaluator hash are now explicit in configuration.
4. The official evaluator is copied byte-for-byte under `official/`, and `requirements-lock.txt` records the working environment.

## Decisions for later phases

- Phase 4 transforms exactly the 36 predeclared factors; it does not rank every V1 feature or select factors based on observed outcomes.
- Phase 5 must generate test time features from training-tail warm-up plus test rows, then partition by year using the same schema.
- Phase 6 splitters must purge the final training trade date before each validation period because `y_ret_1d` uses the next row.
- Phase 7 wraps but does not edit `official/evaluate.py`; parity fixtures must cover all threshold and filter branches.
- Baseline training joins labels only after features are materialized and reads one year partition at a time.
- Model/preprocessing state is fit only inside each fold. Test data may participate in unsupervised same-date rank calculations, but never in fitting supervised state or selecting parameters.

# Framework review after Phase 8

Review date: 2026-09-09

## Outcome

The raw/cache/feature/model separation, causal boundaries, exact trading-calendar CV and official evaluator contract remain sound. Phase 8 exposed two reproducibility gaps; both were fixed before the phase was marked complete.

1. Metrics were initially computed from in-memory float64 predictions while OOF files stored float32. Evaluation now casts to float32 first, so saved OOF files reproduce every recorded metric exactly.
2. OOF artifacts initially lacked explicit key and finite-value evidence. Every model/fold now records row count, key hash, uniqueness, prediction dtype and finite status; all models share the same validation key hash within each fold.

## Evidence and downstream decisions

- The reusable training-only model view has 7,900,350 rows, 119 allowlisted features, no Inf and key hash `89f1e0cc9d675b51`, identical to the Phase 3 feature key hash. Real labels are present as targets but excluded from the manifest feature list.
- LightGBM has stable fold Rank IC (0.07910–0.07967) but turnover rises from 0.23261 to 0.75559. Phase 10 must treat turnover stability as the primary optimization risk and must choose causal smoothing/hysteresis parameters from per-fold OOF only.
- Ridge has useful positive IC but mean turnover 0.87616; it is retained for later ensemble diversity, not treated as the leading standalone baseline.
- The predeclared positive 20-day momentum factor is negative on all three validation years. It remains a plumbing baseline; its direction will not be silently flipped and reported as if predeclared.
- CatBoost remains deferred because it was optional in the plan and is not justified before experiment tracking and turnover diagnostics on the 16 GB machine.
- Phase 9 should register these immutable Phase 8 runs, including config, feature-view manifest, OOF paths and all fold metrics. Phase 10–11 must preserve fold boundaries when evaluating turnover and ensembles. The next scheduled architecture review is after Phase 12.

# Framework review after Phase 12

Review date: 2026-09-10

## Outcome

The project structure, immutable raw-data boundary, causal feature rules, expanding-window CV, official evaluator parity and append-only experiment registry remain sound. The Phase 9–12 sequence produced reproducible evidence without using test labels. The Phase 13–16 order does not need to change, but four downstream contracts are tightened.

1. The fixed-turnover LambdaRank result (`p12_lambdarank_turnover_v1`) replaces the Phase 11 regression ensemble as the incumbent. Its mean official score is 0.37798 and worst-fold score is 0.31573.
2. Ranking training is materially more expensive than regression (about 1,628 seconds for this full run). Phase 13/14 therefore screen feature families causally before a small number of full three-fold ranker retrains. Proxy screening is diagnostic and may not be reported as final model performance.
3. Ranker performance declines from 0.44522 to 0.31573 across the three validation years while turnover rises from 0.08281 to 0.32056. Phase 15 must treat temporal stability and recent-window sensitivity as primary questions, not assume that the higher mean removes regime risk.
4. Validation folds intentionally reset post-processing state. Production inference must instead generate predictions on at least the last 250 training dates and pass per-stock EMA plus the previous eligible Top set into the test period.

## Evidence

- All six Phase 12 OOF artifacts have 1,125,300 rows, unique keys, finite float32 predictions and exact metric replay (`max_abs_delta=0.0`).
- Training group counts are 972/1,214/1,456 and group-size sums exactly match 3,568,878/4,597,785/5,659,954 real non-null training labels.
- Raw LambdaRank improves mean official score by 0.09083 versus the like-for-like Phase 8 regression LightGBM. Frozen turnover post-processing lifts the ranker to 0.37798, 0.03716 above the Phase 11 ensemble.
- The Phase 12 source snapshot is `228A0329CC8D8488BB5B03822F617B770174FAC1F9ACA2622AA47D02CC385FDE`; its artifact manifest is `A32B48A98B711CE7B77426DF56BEA9F3ADB6A3B4CB843F6F7DADA39F7F4EBF19`.
- The official evaluator, raw train and raw test hashes remain unchanged. The registry has 28 unique experiment keys after adding both Phase 12 variants.

## Decisions for later phases

- Phase 13 stores advanced feature families incrementally by year and does not rebuild or duplicate the complete V1 feature store before screening.
- Phase 14 combines coverage, daily factor IC, temporal stability and family-level ablation; only shortlisted families receive full LambdaRank CV.
- Phase 15 compares full history and predeclared recent training windows with explicit mean/worst/std/turnover reporting. It must investigate the observed year-on-year degradation.
- Phase 16 reopens ensemble selection using LambdaRank, regression LightGBM and Ridge OOF. Phase 11 weights are historical evidence, not final weights.
- Phase 17 builds and verifies train-tail post-processing state before test prediction. Test data remains unavailable for supervised fitting or parameter choice.
- The next scheduled framework review is after Phase 16.

# Framework review after Phase 16

Review date: 2026-09-11

## Outcome

The phase sequence, immutable raw-data boundary, causal feature rules, expanding-window validation and official evaluator contract remain appropriate. Phase 16 completed the planned model-selection cycle and the project can proceed to Phase 17 without adding another modeling phase. Three adjustments were made before continuing.

1. The downstream incumbent now points to `p16_locked_ensemble_v1`, not the superseded Phase 12 ranker. The locked blend improves mean official OOF score from 0.37798 to 0.38179, worst fold from 0.31573 to 0.32395, and lowers mean turnover from 0.21230 to 0.20343.
2. Phase 17 must assemble and validate component-specific feature schemas. The base ranker and regression LightGBM require 47 features, the ATR ranker requires 49, and Ridge requires 36 z-score features; using one blanket test matrix would be incorrect.
3. The environment lock was corrected from the non-working SciPy entry to the actually verified `scipy==1.16.3`. The project requirement range is unchanged, and the full 60-test suite passes in this environment.

## Evidence

- The predeclared 0.20-step three-component simplex contains 21 candidates. The selected weights are 0.60 base LambdaRank, 0.20 trailing-two-year ATR LambdaRank and 0.20 regression sleeve; the sleeve contains 0.65 LightGBM and 0.35 Ridge.
- All eight metrics at all three endpoints reproduce their historical audits with zero delta. The selected fold scores are 0.44822, 0.37320 and 0.32395.
- Final models use only real non-null training labels: 6,758,531 rows for the full-history base ranker and both regression models, and 2,156,466 rows for the 2023–2024 ATR ranker. No test data was used for fitting or selection.
- The three selected OOF artifacts each cover 1,125,300 unique keys with finite predictions. Raw train and test hashes remain unchanged.
- Phase 16 source snapshot is `591F5023B0521E80AC701FFE97A124538CE535F4BC2FDDF4F7331CB03DA4CCA0`; artifact manifest is `D348C8027D74250C2D5AB1BB228917AFA660CCF6CB6B006FE24175C5D4E04532`.
- That source snapshot predates the environment-lock correction. The executable environment used for the complete Phase 16 run was NumPy 2.2.6 and SciPy 1.16.3; the corrected lock hash and reason are recorded explicitly in the Phase 16 framework-review JSON rather than rewriting the immutable experiment snapshot.

## Decisions for Phase 17–19

- Phase 17 must generate every component's raw score on at least the last 250 real training dates plus the full test period, independently rank components within each date, blend using the locked nested weights, then run one continuous EMA and hysteresis state across the train/test boundary.
- Test rows may be used for same-date unsupervised ranks and causal test-period recurrences. They may not be used to fit model parameters, choose weights or infer labels.
- Before writing predictions, Phase 17 must compare test keys exactly with the untouched raw test file, reject duplicate/non-finite predictions, and retain a Parquet prediction artifact before any CSV submission formatting.
- Phase 18 remains a strict submission/evaluator-contract phase; Phase 19 remains documentation and reproducibility packaging. No further feature or model search is authorized unless new training-only evidence justifies reopening the frozen Phase 16 configuration.
- The next scheduled four-phase review point is Phase 20 if the project extends beyond the current Phase 19 plan.
