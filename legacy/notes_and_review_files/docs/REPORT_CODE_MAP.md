# Report To Code Map

This map aligns the final report structure with the notebooks, outputs, figures, and helper modules in the repository.

## Report Overview

Final report: `report/final_report.pdf`  
Report title: `Hidden market regimes in weekly stock-market direction classification: comparing standard and HMM-informed LSTMs`

Unless a path starts with `report/`, `docs/`, `figures/`, `section_6_7_used_figures/`, or `report_materials/`, paths in the table are relative to `report_notebooks_outputs/`.

## Section Mapping

| Report section | Purpose in report | Main notebook or code | Main outputs and figures |
| --- | --- | --- | --- |
| 1 Introduction | Frames the research question and move from return regression to direction classification. | No primary code. Background evidence from `notebooks/daily_return_regression_window_sensitivity.ipynb` and `notebooks/preprocessing_baseline_models.ipynb`. | `outputs/daily_return_regression_window_sensitivity/`, `outputs/metrics/`, `report_materials/key_result_tables.md` |
| 2 Literature Review | Connects EMH, HMMs, LSTMs, and HMM-LSTM hybrid forecasting. | No primary code. | `report_materials/reference_paper_extract.txt` |
| 3 Data and Feature Engineering | Builds weekly OHLCV data, target labels, technical predictors, context variables, HMM features, chronological splits, scaling, and sequences. | `notebooks/LSTM_input_weekly.ipynb`, `notebooks/weekly_external_volatility_features.ipynb`, `notebooks/hmm_regime_weekly.ipynb`, `report_materials/build_stock_crisis_visualizations.py` | `outputs/weekly/`, `outputs/weekly_context/`, `outputs/weekly_hmm/`, `report_materials/figures/normalized_stock_trajectories_crisis_periods.png` |
| 4 Methodology | Defines controlled comparison, HMM construction, LSTM architecture, validation-only selection, leakage controls, metrics, and robustness design. | `notebooks/weekly_controlled_hmm_lstm_comparison.ipynb`, `notebooks/weekly_rolling_hmm_replication_aapl_ibm.ipynb`, `weekly_lstm_architecture.py` | `outputs/weekly_controlled_comparison/controlled_run_metadata.json`, `outputs/weekly_rolling_hmm_replication_aapl_ibm/rolling_run_metadata.json` |
| 5 HMM Regime Diagnostics | Shows whether fitted HMM regimes are economically interpretable. | `notebooks/hmm_regime_weekly.ipynb`, `notebooks/weekly_hmm_robustness.ipynb` | `outputs/weekly_hmm/weekly_hmm_state_summary.csv`, `outputs/weekly_hmm/weekly_hmm_transition_matrices.csv`, `outputs/weekly_hmm/plots/` |
| 6 Primary Results: Controlled HMM Feature Comparison | Main fixed-split test of whether adding HMM probability improves LSTM performance. | `notebooks/weekly_controlled_hmm_lstm_comparison.ipynb` | `outputs/weekly_controlled_comparison/controlled_model_summary.csv`, `outputs/weekly_controlled_comparison/controlled_model_results_long.csv`, `outputs/weekly_controlled_comparison/figures/final_test_roc_curves_by_ticker.png`, `outputs/weekly_controlled_comparison/figures/final_test_balanced_accuracy_comparison.png` |
| 6.1 Exploratory Best-Observed Market-Context Models | Reports broader context-feature grids as exploratory evidence. | `notebooks/weekly_context_lstm_training.ipynb`, `notebooks/weekly_context_lstm_training_qqq.ipynb` | `outputs/weekly_context_lstm/`, `outputs/weekly_context_lstm_qqq/`, `outputs/weekly_context_lstm_qqq_final/`, `report_materials/roc_auc_audit.md` |
| 7 Rolling Paired Replication Results | Refits HMMs and scalers inside rolling folds to test robustness over time. | `notebooks/weekly_rolling_hmm_replication_aapl_ibm.ipynb` | `outputs/weekly_rolling_hmm_replication_aapl_ibm/rolling_paired_hmm_summary.csv`, `outputs/weekly_rolling_hmm_replication_aapl_ibm/rolling_paired_hmm_auc_deltas.csv`, `outputs/weekly_rolling_hmm_replication_aapl_ibm/figures/paired_hmm_auc_delta_by_fold.png`, `outputs/weekly_rolling_hmm_replication_aapl_ibm/figures/pooled_oos_roc_aapl.png`, `outputs/weekly_rolling_hmm_replication_aapl_ibm/figures/pooled_oos_roc_ibm.png` |
| 8 Discussion | Interprets why HMMs describe regimes better than they improve forecasts. | No separate code. Supported by Sections 5-7 outputs. | `report_materials/roc_auc_audit.md`, `report_materials/key_result_tables.md` |
| 9 Limitations | Notes data size, ticker scope, noisy financial targets, selection risk, and model limitations. | No separate code. Supported by diagnostics. | `outputs/weekly_controlled_comparison/controlled_sequence_diagnostics.csv`, `outputs/weekly_rolling_hmm_replication_aapl_ibm/rolling_sequence_diagnostics.csv` |
| 10 Conclusion | Summarizes the evidence against a robust isolated HMM feature advantage. | No separate code. | Final results from Sections 6 and 7. |
| Appendix A | Additional weekly HMM diagnostics. | `notebooks/hmm_regime_weekly.ipynb`, `notebooks/weekly_hmm_target_overlap_diagnostic.ipynb`, `notebooks/weekly_hmm_robustness.ipynb` | `outputs/weekly_hmm/plots/`, `outputs/weekly_hmm_target_overlap_diagnostic/`, `outputs/weekly_hmm_robustness/` |
| Appendix B | Controlled weekly LSTM architecture. | `weekly_lstm_architecture.py`, `notebooks/weekly_controlled_hmm_lstm_comparison.ipynb` | `outputs/weekly_controlled_comparison/controlled_run_metadata.json`, `report_materials/figures/weekly_hmm_lstm_model_hierarchy.png` |

## Consistency Audit

The report-code consistency check is saved in `report_materials/report_code_consistency_audit.md`. It verifies Table 4, Table 5, Section 6 deltas, Table 6 bootstrap intervals, Section 6.1 exploratory ROC-AUC values, Table 7, Section 7 sign-flip p-values, and key supervisor-facing paths. The audit script is `report_materials/audit_report_code_consistency.py`.

## Notebook Run Order

Use this order when reproducing the main weekly pipeline from inside `report_notebooks_outputs/`:

1. `notebooks/LSTM_input_weekly.ipynb`
2. `notebooks/hmm_regime_weekly.ipynb`
3. `notebooks/weekly_external_volatility_features.ipynb`
4. `notebooks/weekly_lstm_training.ipynb`
5. `notebooks/weekly_hmm_informed_lstm.ipynb`
6. `notebooks/weekly_controlled_hmm_lstm_comparison.ipynb`
7. `notebooks/weekly_context_lstm_training.ipynb`
8. `notebooks/weekly_context_lstm_training_qqq.ipynb`
9. `notebooks/weekly_hmm_target_overlap_diagnostic.ipynb`
10. `notebooks/weekly_hmm_robustness.ipynb`
11. `notebooks/weekly_rolling_hmm_replication_aapl_ibm.ipynb`

Daily background notebooks can be reviewed separately:

- `notebooks/preprocessing_baseline_models.ipynb`
- `notebooks/hmm_regime.ipynb`
- `notebooks/hmm_informed_lstm.ipynb`
- `notebooks/daily_return_regression_window_sensitivity.ipynb`

## Helper Modules

The report bundle now contains the helper modules expected by the notebooks:

- `weekly_lstm_architecture.py` - final weekly binary LSTM classifier, callbacks, seeding, and binary metrics.
- `lstm_architecture.py` - earlier daily return-regression LSTM helpers.
- `forecast_metrics.py` - directional evaluation metrics for return forecasts.
- `plot_helpers.py` - plotting helper for earlier prediction visualizations.
- `load_saved_results.py` - utility for reading saved early-stage result manifests.

The original copies remain in `legacy/research_code/` for traceability.
