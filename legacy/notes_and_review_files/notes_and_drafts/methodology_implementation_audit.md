# Methodology Implementation Audit

Audit basis: executed notebook cells, saved CSV/parquet artifacts, saved metadata, and source utilities in the honours-project folder. No neural-network training was rerun. The final controlled implementation is treated as `weekly_controlled_hmm_lstm_comparison.ipynb` plus `outputs/weekly_controlled_comparison/*`, with upstream weekly data from `LSTM_input_weekly.ipynb` and HMM features from `hmm_regime_weekly.ipynb`.

## Executive Finding

The implemented primary research question is:

> Does an HMM-derived probability provide incremental out-of-sample value for weekly LSTM stock-direction forecasting beyond stock-specific technical features?

The final controlled ablation compares a technical-only LSTM against a technical-plus-HMM LSTM under the same tickers, splits, architecture, lookback grid, seeds, validation model-selection rule, and threshold-selection rule. The only input difference is one additional scaled HMM feature: `HMM_Up_Prob_Scaled`.

The final HMM feature is not `HMM_Bullish_Prob_Next`. It is derived upstream as `HMM_Positive_Return_Prob_Next`, renamed in the controlled notebook to `HMM_Up_Prob_Raw`, and then scaled per ticker to `HMM_Up_Prob_Scaled`.

## Implementation Table

| Methodological item | Verified implementation | Evidence location | Status | Report implication |
|---|---|---|---|---|
| Tickers | AAPL, IBM, MSFT for the fixed-split controlled comparison. | `LSTM_input_weekly.ipynb` cell 1; `outputs/weekly_controlled_comparison/controlled_run_metadata.json`; `controlled_model_summary.csv`. | Fully verified | State all primary fixed-split results are ticker-specific for AAPL, IBM, MSFT. |
| Original data source | `yfinance.download` with `auto_adjust=True`, `start="2010-01-01"`, `end="2026-01-01"`. | `LSTM_input_weekly.ipynb` cells 1 and 3. | Fully verified | Describe prices as Yahoo Finance/yfinance auto-adjusted OHLCV; do not call them unadjusted. |
| Daily range | Raw weekly-input download produced daily rows from 2010-01-04 to 2025-12-31, inferred from saved weekly labels; older daily modelling file has usable daily rows from 2010-03-31 to 2025-12-30. | `LSTM_input_weekly.ipynb` cell 3 output; `preprocessed_stock_data.csv`. | Partly verified | For the final weekly pipeline, cite the download start/end and saved weekly ranges rather than old daily usable ranges. |
| Weekly OHLCV range | 835 weekly bars per ticker, labelled 2010-01-08 to 2026-01-02 before feature warm-up and target drop. | `outputs/weekly/weekly_ohlcv_df.csv`. | Fully verified | Explain that the final 2026-01-02 label comes from Friday resampling of data downloaded through 2025-12-31. |
| Weekly modelling range | 808 usable rows per ticker, feature dates 2010-07-09 to 2025-12-26; target dates 2010-07-16 to 2026-01-02. | `outputs/weekly/model_df_weekly.csv`. | Fully verified | The report values 808 total rows and 565/121/122 split rows are supported for the row-level dataset. |
| Weekly resampling rule | `resample("W-FRI")`; `Open=first`, `High=max`, `Low=min`, `Close=last`, `Volume=sum`; rows with missing OHLC dropped. | `LSTM_input_weekly.ipynb` cell 5. | Fully verified | Use this exact aggregation. |
| Target definition | `Target_Date = Date.shift(-1)`, `Next_Week_Log_Return = Weekly_Log_Return.shift(-1)`, `Target_State_Binary = 1` if next-week log return is strictly positive, else `0`; final missing target rows dropped. | `LSTM_input_weekly.ipynb` cell 9. | Fully verified | Zero return is assigned to the bearish/down class. |
| Split basis | Rows are sorted by `Target_Date` per ticker and split 70%/15%/15%; train 565, validation 121, test 122 rows per ticker. | `LSTM_input_weekly.ipynb` cell 11; `outputs/weekly/model_df_weekly.csv`; `controlled_class_balance.csv`. | Fully verified | Say split is target-date chronological, per ticker. |
| Test period | Test feature dates 2023-09-01 to 2025-12-26; test target dates 2023-09-08 to 2026-01-02. | `outputs/weekly/model_df_weekly.csv`; `selected_test_predictions.csv`. | Fully verified | If the report says only 2023-09-01 to 2025-12-26, clarify these are feature-week dates. |
| Per-ticker processing | Weekly features, splits, scalers, HMMs, sequences, and models are constructed per ticker. | `LSTM_input_weekly.ipynb` cells 7, 11, 13, 15; `hmm_regime_weekly.ipynb` cell 8; controlled notebook cells 16, 23. | Fully verified | Do not describe a pooled primary model. |
| Technical feature set | Exactly 15 features: `Weekly_Log_Return`, `Weekly_Open_Close_Log_Return`, `Weekly_High_Low_Range`, `Weekly_Volume_Change`, `Rolling_Vol_4`, `Rolling_Vol_12`, `Rolling_Vol_26`, `Momentum_4`, `Momentum_12`, `Momentum_26`, `MA_Gap_4`, `MA_Gap_12`, `MA_Gap_26`, `Drawdown_12`, `Drawdown_26`. | `LSTM_input_weekly.ipynb` cell 1; controlled notebook cell 14; `controlled_run_metadata.json`. | Fully verified | Use this list as the primary feature set. |
| Technical feature formulas | Log close difference; log close/open; `(High-Low)/Close`; log-volume difference; rolling return std, rolling return sum, close/rolling-mean minus 1 for windows 4/12/26; close/rolling max minus 1 for 12/26. | `LSTM_input_weekly.ipynb` cell 7; `outputs/weekly/weekly_features_df.csv`. | Fully verified | Include formulas or table if space permits. |
| QQQ/VIX/VXN/recession/relative features in primary comparison | Explicitly excluded from controlled feature lists by forbidden-substring checks; only appear in context/secondary experiments. | Controlled notebook cells 14 and 33; context output folders. | Fully verified | Treat market-context experiments as exploratory/robustness, not primary. |
| Cross-stock features | No primary feature list includes another stock's data. QQQ/relative-market features appear only in secondary QQQ context experiments. | Controlled notebook cell 14; `controlled_run_metadata.json`; QQQ context outputs. | Fully verified | Do not claim the models directly exploit AAPL/IBM/MSFT cross-stock correlations. |
| Technical scaler | `StandardScaler`, fitted separately per ticker on training rows only, applied to all splits. | `LSTM_input_weekly.ipynb` cell 13; `outputs/weekly/weekly_scaler_metadata.csv`. | Fully verified | State train-only per-ticker scaling. |
| HMM feature scaler | Separate `StandardScaler` for `HMM_Up_Prob_Raw`, fitted per ticker on train rows only. | Controlled notebook cell 10; `controlled_hmm_scaling_diagnostics.csv`; `controlled_run_metadata.json`. | Fully verified | Do not imply HMM probabilities were left unscaled inside the LSTM. |
| Validation/test influence on scalers | Metadata and guards show no validation/test fitting for technical or HMM scalers. | `weekly_scaler_metadata.csv`; controlled notebook cells 10 and 33. | Fully verified | Reasonable to state scaler fitting used training rows only. |
| Sequence history across splits | Split assignment is based on the target row. Validation/test sequences may include earlier historical rows from previous splits because windows are cut by target row, not reset at split boundaries. `Target_Date` is checked to be later than all sequence dates. | `LSTM_input_weekly.ipynb` cell 15; controlled notebook cell 16; `controlled_sequence_diagnostics.csv`. | Fully verified | Describe as causal historical lookback, but do not claim validation/test sequences contain only rows inside their own split. |
| Primary HMM states and emissions | Two-state `GaussianHMM`, univariate observation `Weekly_Log_Return` after train-only standardization, `covariance_type="full"`, `n_iter=500`, `tol=1e-4`, `random_state=42`, `min_covar=1e-4`. | `hmm_regime_weekly.ipynb` cells 2 and 8; `weekly_hmm_model_diagnostics.csv`. | Fully verified | Mathematical HMM should be univariate Gaussian for the primary fixed split. |
| HMM fitting scope | One HMM fitted separately per ticker on training rows only; one fixed HMM per ticker for fixed split. | `hmm_regime_weekly.ipynb` cell 8; sanity checks cell 16; `weekly_hmm_model_diagnostics.csv`. | Fully verified | State ticker-specific train-only HMMs. |
| HMM convergence | All three fixed HMMs converged. Iterations: AAPL 135, IBM 63, MSFT 121. | `weekly_hmm_model_diagnostics.csv`. | Fully verified | Convergence can be reported in appendix/audit, not necessary in main Methodology. |
| Rolling HMM refit | Rolling replication refits HMM and scalers separately for every ticker/fold using fold-training rows only. | `weekly_rolling_hmm_replication_aapl_ibm.ipynb` cells 27 and 30; `rolling_hmm_fold_diagnostics.csv`; `rolling_run_metadata.json`. | Fully verified | Treat rolling as completed secondary replication, but note its variants differ by ticker. |
| State labelling | Higher training-sample mean `Weekly_Log_Return` state labelled Bullish; lower labelled Bearish. Volatility does not determine the label. | `hmm_regime_weekly.ipynb` cell 6 and cell 8; `weekly_hmm_state_summary.csv`. | Fully verified | Say labels are relative, training-sample labels, not true economic states. |
| Fixed-split raw state numbers | State 0 labelled Bullish and state 1 Bearish for AAPL, IBM, MSFT in the saved fixed HMM. | `weekly_hmm_model_diagnostics.csv`; `weekly_hmm_state_summary.csv`. | Fully verified | Avoid assigning intrinsic meaning to state numbers. |
| Causal filtered probabilities | Custom forward algorithm computes `P(S_w | Y_1:w)`; code explicitly distinguishes this from smoothing. | `hmm_regime_weekly.ipynb` markdown cell 5 and code cell 6. | Fully verified | State no smoothed probabilities were used for the primary HMM feature. |
| Next-state probabilities | `next_probs = filtered_probs @ hmm.transmat_`. | `hmm_regime_weekly.ipynb` cell 8. | Fully verified | The feature is one-step-ahead in state probability space. |
| Positive-return transformation | Training rows estimate state positive-return rates as `mean(Weekly_Log_Return > 0)` by hard state; `HMM_Positive_Return_Prob_Next = sum_k next_prob_k * state_positive_rate_k`. | `hmm_regime_weekly.ipynb` cell 8; `weekly_hmm_state_summary.csv`. | Fully verified | Use this exact definition. Do not call it a pure bullish-regime probability. |
| Final HMM feature column | Controlled notebook merges `HMM_Positive_Return_Prob_Next`, renames to `HMM_Up_Prob_Raw`, and uses scaled `HMM_Up_Prob_Scaled` in the LSTM. HMM-only benchmark uses `HMM_Up_Prob_Raw`. | Controlled notebook cells 6, 8, 10, 14, 22; `controlled_run_metadata.json`. | Fully verified | Prominently correct any report text saying final LSTM used `HMM_Bullish_Prob_Next` or two HMM features. |
| Number of HMM columns entering LSTM | Exactly one: `HMM_Up_Prob_Scaled`. Complementary probability omitted. | Controlled notebook cell 14; `controlled_run_metadata.json`. | Fully verified | State the complementary probability is redundant in binary setting and was not used. |
| Paired LSTM design | Technical-only and technical-plus-HMM LSTMs use same tickers, target, splits, technical features, lookback grid, architecture, seeds, optimizer/loss, batch size, epochs, early stopping, threshold rule. Only difference is `HMM_Up_Prob_Scaled`. | Controlled notebook cells 14, 16, 23, 25, 31. | Fully verified | This is the primary controlled ablation. |
| LSTM architecture | Input `(lookback, n_features)`; one LSTM layer with 16 units; one dense ReLU layer with 8 units; dropout 0.2 in LSTM and after dense; recurrent dropout 0; sigmoid output; Adam lr 0.001; binary cross-entropy. | `weekly_lstm_architecture.py`; controlled notebook cell 4; `controlled_run_metadata.json`. | Fully verified | Do not import daily regression architecture values. |
| LSTM training | Max 150 epochs, batch size 32, shuffle true, class_weight none, early stopping monitor `val_auc` mode max patience 12 restore best weights, ReduceLROnPlateau on `val_loss`, ModelCheckpoint on `val_auc`. | `weekly_lstm_architecture.py`; controlled notebook cells 23 and 31. | Fully verified | Mention no class weighting in final controlled LSTMs. |
| Lookback grid | Final controlled grid is `[4, 8, 12]`, not `[4, 8, 12, 26]`. | Controlled notebook cell 3; `controlled_run_metadata.json`; `lstm_validation_lookback_selection.csv`. | Fully verified | Report old 26-week grid only for exploratory weekly notebooks. |
| Lookback selection | Separately by ticker and feature set, using mean validation ROC-AUC across 5 seeds; ties prefer shorter lookback. Technical and HMM models can select different lookbacks. | Controlled notebook cell 25; `lstm_validation_lookback_selection.csv`. | Fully verified | Use selected lookbacks: AAPL technical 4, AAPL HMM 12; IBM both 4; MSFT both 4. |
| Seed handling | Seeds `[7, 19, 42, 91, 123]`; every ticker-feature-lookback candidate trained for all seeds, producing 90 runs. Lookback selected by mean validation ROC-AUC across seeds. Selected test metrics are mean and SD across the five selected-lookback seed runs. | Controlled notebook cells 3, 23, 25, 27; `lstm_all_runs.csv`; `controlled_model_summary.csv`. | Fully verified | Avoid vague "multiple seeds"; specify aggregation. |
| Test probabilities and ensembling | Controlled summary is mean of seed-level metrics. ROC/PR plots aggregate LSTM probabilities by averaging across selected seeds for one curve per model. | Controlled notebook cells 27 and 30. | Fully verified | Do not describe the table as a separately trained ensemble; plots use averaged probabilities. |
| Threshold grid | Final controlled threshold grid is 0.35 to 0.65 in 0.01 increments. | Controlled notebook cell 3; `controlled_run_metadata.json`. | Fully verified | Do not use the HMM-only notebook's wider 0.05 to 0.95 grid for final controlled methods. |
| Threshold rule | Threshold selected on validation balanced accuracy; ties choose closest to 0.50, then lower threshold. Each ticker/model/seed receives a validation-selected threshold; LSTM summary reports mean threshold across seeds. | Controlled notebook cell 18; `selected_test_predictions.csv`; `controlled_model_summary.csv`. | Fully verified | State `tau* = argmax BA_val(tau)` with tie rule. |
| Train-majority baseline | Outputs constant probability equal to training up-share and uses threshold 0.50. Since all train up-shares exceed 0.5, it predicts up for all rows. | Controlled notebook cell 18; `controlled_model_results_long.csv`; `controlled_model_summary.csv`. | Fully verified | Explain high F1 can be majority-class behavior, not discrimination. |
| Persistence baseline | Probability is 1 when current week return is positive and 0 otherwise; intended as next week same direction as current week; threshold selected on validation BA. | Controlled notebook cell 18; `controlled_model_summary.csv`. | Fully verified | State it uses current-week direction only. |
| HMM-only benchmark | Uses raw unscaled `HMM_Up_Prob_Raw`; threshold selected on validation BA; HMM not retrained in controlled notebook. | Controlled notebook cell 22; `controlled_model_summary.csv`. | Fully verified | Keep separate from HMM-LSTM input, which uses scaled column. |
| Logistic regression | Deterministic `LogisticRegression(C=C, class_weight=None, max_iter=5000)` on one current weekly feature row with the 15 scaled technical features, not a flattened sequence. `C` grid `[0.01, 0.1, 1.0, 10.0]`; selected by validation ROC-AUC, tie lower C; threshold by validation BA. | Controlled notebook cell 20; `logistic_validation_selection.csv`. | Fully verified | Call it a technical current-row benchmark. |
| Metrics | Accuracy, balanced accuracy, precision, recall, F1, Cohen kappa, ROC-AUC, PR-AUC, Brier score, log-loss, predicted/actual up share, and confusion-matrix counts are saved. | Controlled notebook cell 18; `controlled_model_results_long.csv`; `controlled_model_summary.csv`. | Fully verified | In Methodology, distinguish threshold-independent ROC-AUC from thresholded BA/F1. |
| Calibration | Brier score and log-loss are saved; probability-bin calibration diagnostics are in target-overlap outputs. No formal ECE-style calibration test found. | `controlled_model_summary.csv`; `weekly_hmm_probability_bins.csv`. | Fully verified | Do not overstate calibration analysis. |
| Uncertainty/significance | LSTM seed-level SDs and rolling fold SDs exist; no bootstrap CI, paired block bootstrap, DeLong, McNemar, or formal hypothesis test was found. | `controlled_model_summary.csv`; `rolling_paired_hmm_summary.csv`; project-wide search. | Fully verified | Treat tiny differences, e.g. IBM 0.585 vs 0.583 ROC-AUC, as practically indistinguishable without formal uncertainty. |
| Rolling replication | Completed for AAPL and IBM with 9 folds each, expanding training windows, 52-week validation, 52-week test, 52-week step. Final fold results, paired deltas, summaries, predictions, and figures are saved. | `weekly_rolling_hmm_replication_aapl_ibm.ipynb` cells 23, 25, 27, 30; `rolling_*` CSVs. | Fully verified | Correct current report text that says rolling final cells were not executed. Treat as secondary, not primary. |
| Rolling variants | AAPL base uses technical plus VIX/VXN volatility context; IBM base uses technical only. HMM variants add `HMM_Up_Prob_Scaled`. | `rolling_run_metadata.json`; `rolling_feature_audit.csv`. | Fully verified | Do not present rolling as identical to the primary all-ticker technical-only vs technical-plus-HMM design. |
| Integrated Gradients | Only a helper function exists in old `lstm_architecture.py`; no executed attribution outputs, aggregation tables, figures, baseline/reference choices, or interpretations found. | `lstm_architecture.py`; project-wide search. | Missing as completed analysis | Remove or revise any report claim that Integrated Gradients was used empirically. |
| Primary vs exploratory | Primary is controlled technical-only vs technical-plus-HMM LSTM plus required baselines. Volatility, QQQ/relative/recession, HMM robustness, target-overlap/calibration diagnostics, and rolling replication are secondary exploratory/robustness analyses. | Controlled and secondary output folders. | Fully verified | Do not use best isolated exploratory test result as confirmatory evidence. |

## Final Architecture Table

| Component | Verified final controlled setting |
|---|---|
| LSTM layers | One LSTM layer |
| Hidden units | 16 |
| Dense layers | One dense ReLU layer with 8 units |
| Dropout | 0.20 in LSTM layer and after dense layer |
| Recurrent dropout | 0.00 |
| Output layer | Dense(1), sigmoid, named `bullish_probability` |
| Loss | Binary cross-entropy |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Maximum epochs | 150 |
| Early stopping | Monitor `val_auc`, mode max, patience 12, restore best weights |
| Other callbacks | ReduceLROnPlateau on `val_loss`; ModelCheckpoint on `val_auc` |
| Seeds | 7, 19, 42, 91, 123 |
| Class weights | None |

## Primary HMM Mathematical Specification

For ticker \(i\) and week \(w\), the final fixed-split HMM uses the train-standardized weekly log return \(x_{i,w}\) as the only observed variable:

\[
S_{i,w} \in \{0,1\}, \qquad
\Pr(S_{i,w}=b \mid S_{i,w-1}=a) = A_{i,ab},
\]

\[
x_{i,w} \mid S_{i,w}=k \sim \mathcal{N}(\mu_{i,k}, \sigma^2_{i,k}).
\]

The observation scaler and HMM parameters are fitted using fixed-split training rows only. Filtered probabilities are computed by the forward recursion:

\[
\alpha_{i,w,k} = \Pr(S_{i,w}=k \mid x_{i,1:w}),
\]

and one-step state probabilities are:

\[
\pi^{S}_{i,w+1,k} = \sum_a \alpha_{i,w,a} A_{i,ak}.
\]

The implemented next-week positive-return probability is:

\[
q_{i,w+1} = \sum_k \pi^{S}_{i,w+1,k} \hat{u}_{i,k},
\]

where \(\hat{u}_{i,k}\) is the training-row empirical probability that `Weekly_Log_Return > 0` among rows assigned to state \(k\). This value is saved upstream as `HMM_Positive_Return_Prob_Next`, renamed to `HMM_Up_Prob_Raw` in the controlled notebook, then train-standardized per ticker as `HMM_Up_Prob_Scaled` for the HMM-LSTM input.

## Contradictions and Unresolved Implementation Details

| Issue | Conflicting evidence | Most authoritative source | Recommended wording |
|---|---|---|---|
| HMM feature name and meaning | Earlier report text and exploratory notebooks refer to `HMM_Bullish_Prob_Next`, `HMM_Positive_Return_Prob_Next`, and two HMM features. | Controlled notebook cells 8, 10, 14, 22; `controlled_run_metadata.json`. | "The controlled HMM-LSTM used one scaled HMM-implied next-week up-probability feature, `HMM_Up_Prob_Scaled`, derived from `HMM_Positive_Return_Prob_Next`; it did not use `HMM_Bullish_Prob_Next`." |
| Filtered vs smoothed probabilities | Report wording may say regime probabilities generally. | `hmm_regime_weekly.ipynb` cell 6 and markdown cell 5. | "Filtered causal probabilities were used; smoothed probabilities were not used for prediction." |
| HMM observation variables | Daily HMM used `Log_Return` and `Rolling_Vol_20`; weekly robustness tested richer HMMs. | Primary fixed HMM: `hmm_regime_weekly.ipynb` cells 2 and 8; controlled metadata points to weekly HMM output. | "The primary controlled HMM is univariate in weekly log return; richer and three-state HMMs are robustness analyses." |
| Lookback grid | Weekly preprocessing and early LSTM work include `[4,8,12,26]`; controlled notebook uses `[4,8,12]`. | Controlled notebook cell 3; `controlled_run_metadata.json`. | "The final controlled comparison used lookbacks 4, 8, and 12 weeks." |
| LSTM architecture | Daily regression used stacked/larger LSTMs and Huber loss. | `weekly_lstm_architecture.py`; controlled notebook cell 4. | "The final weekly controlled classifier used one 16-unit LSTM and binary cross-entropy; daily architectures are historical only." |
| Seed handling | Earlier notebooks sometimes use single seed or selected runs. | Controlled notebook cells 3, 23, 25, 27. | "Each controlled LSTM candidate was trained for five seeds; lookback was selected by mean validation ROC-AUC, and test metrics are mean/SD over selected-lookback seed runs." |
| Test-period dates | Current report says test ran 2023-09-01 to 2025-12-26. | `outputs/weekly/model_df_weekly.csv`. | "Feature-week test dates are 2023-09-01 to 2025-12-26; target weeks are 2023-09-08 to 2026-01-02." |
| Class weights | Current report line 202 says training rows were used to estimate class weights. | Controlled notebook cell 23 and metadata: `class_weight=None`. | Remove class-weight claim for final controlled models; class weights were not used. |
| Market-context variables in final controlled model | Context notebooks use VIX/VXN/QQQ/recession features. | Controlled notebook cells 14 and 33. | "Market-context variables were excluded from the primary controlled comparison and treated as secondary experiments." |
| Rolling replication completion | Current report says final rolling result cells were not executed. | `rolling_fold_model_results.csv`, `rolling_paired_hmm_auc_deltas.csv`, `rolling_paired_hmm_summary.csv`, `rolling_oos_predictions.csv`; rolling notebook cells 23, 25, 30. | "Rolling replication was completed and saved, but remains secondary because it uses AAPL volatility context and IBM technical features rather than the full primary design." |
| Integrated Gradients | `lstm_architecture.py` contains an `integrated_gradients` helper, but no completed outputs found. | Project-wide search. | "Integrated Gradients was not executed as part of the completed empirical analysis." |
| Yahoo adjusted vs unadjusted prices | Report may simply say OHLCV or adjusted prices inconsistently. | `LSTM_input_weekly.ipynb` cell 3 uses `auto_adjust=True`. | "Daily OHLCV were downloaded through yfinance with auto-adjusted price fields." |
| Cross-stock correlations | Reference paper discusses exploiting cross-correlations. Primary feature list is per-ticker only. | Controlled notebook cell 14. | Do not claim the implemented primary models exploit cross-stock correlations. |

## Final Output Summary

- Final verified primary HMM specification: ticker-specific two-state Gaussian HMM, univariate train-standardized `Weekly_Log_Return`, full covariance, random seed 42, 500 iterations, tolerance `1e-4`, `min_covar=1e-4`, fitted on training rows only.
- Exact HMM feature used by the controlled LSTM: `HMM_Up_Prob_Scaled`, train-standardized from `HMM_Up_Prob_Raw`, which is renamed from upstream `HMM_Positive_Return_Prob_Next`.
- Final verified LSTM architecture: one 16-unit LSTM, one 8-unit ReLU dense layer, dropout 0.20, sigmoid output, Adam lr 0.001, binary cross-entropy, batch size 32, maximum 150 epochs, early stopping on validation AUC with patience 12.
- Lookback grid: `[4, 8, 12]` in the final controlled comparison.
- Seed procedure: five seeds `[7, 19, 42, 91, 123]` per candidate; lookback selected by mean validation ROC-AUC; selected test metrics reported as mean/SD across seed-level runs.
- Threshold procedure: validation balanced-accuracy maximization over thresholds 0.35 to 0.65 by 0.01, with ties closest to 0.50.
- Final baseline set: train-majority probability baseline, persistence baseline, HMM-only probability benchmark, technical logistic regression, technical-only LSTM, technical-plus-HMM LSTM.
- Leakage controls: target-date chronological splits, per-ticker train-only scaling, train-only HMM fitting and state/up-rate estimation, validation-only lookback/C/threshold selection, and no test-set model selection in the controlled notebook. Validation/test sequences can use earlier historical rows from previous splits because sequence assignment uses the target row.
- Unresolved questions: exact raw yfinance daily min/max should be cited from notebook output if needed; no formal statistical significance test was implemented; Integrated Gradients was not completed.
- Files created by this audit: `methodology_implementation_audit.md`, `methodology_rewrite.tex`, and `methodology_integration_notes.md`.
