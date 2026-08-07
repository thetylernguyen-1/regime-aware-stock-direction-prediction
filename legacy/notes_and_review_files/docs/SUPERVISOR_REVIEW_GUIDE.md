# Supervisor Review Guide

This guide explains how to review the code and outputs alongside the final report without reverse-engineering the whole workspace.

## Review Path

1. Open `report/final_report.pdf`.
2. Use `docs/REPORT_CODE_MAP.md` to connect each report section to the relevant notebook and output files.
3. Review the final evidence first:
   - `report_notebooks_outputs/notebooks/weekly_controlled_hmm_lstm_comparison.ipynb`
   - `report_notebooks_outputs/outputs/weekly_controlled_comparison/controlled_model_summary.csv`
   - `report_notebooks_outputs/notebooks/weekly_rolling_hmm_replication_aapl_ibm.ipynb`
   - `report_notebooks_outputs/outputs/weekly_rolling_hmm_replication_aapl_ibm/rolling_paired_hmm_summary.csv`
4. Use `report_materials/report_code_consistency_audit.md` to confirm the final report values match the saved outputs.
5. Use `report_materials/roc_auc_audit.md` to see why exploratory high-AUC runs are separated from the final claim.

## Final Claim To Check Against Code

The report does not claim that HMM features robustly improve LSTM stock-direction forecasting. The final controlled comparison shows:

| Ticker | Technical-only LSTM ROC-AUC | Technical-plus-HMM LSTM ROC-AUC | Difference |
| --- | ---: | ---: | ---: |
| AAPL | 0.531 | 0.470 | -0.061 |
| IBM | 0.583 | 0.585 | +0.002 |
| MSFT | 0.476 | 0.474 | -0.002 |

The rolling paired replication also weakens a strong improvement claim:

| Ticker | Mean base AUC | Mean HMM-LSTM AUC | Mean delta | Pooled base OOS AUC | Pooled HMM OOS AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| AAPL | 0.479 | 0.490 | +0.010 | 0.437 | 0.434 |
| IBM | 0.545 | 0.527 | -0.018 | 0.512 | 0.509 |

## What Is Final Evidence

- Final data formulation: weekly next-week binary direction classification.
- Final tickers: AAPL, IBM, MSFT.
- Final model comparison: technical-only LSTM versus technical-plus-HMM LSTM, with matched seeds and validation-only lookback/threshold selection.
- Final robustness check: rolling paired replication for AAPL and IBM.

## What Is Background Or Exploratory

- Daily return-regression notebooks explain why the project moved away from daily next-day regression.
- Earlier HMM-informed daily/weekly notebooks show development history.
- Market-context and QQQ grids produced some higher ROC-AUC values, especially for IBM, but they were retrospectively selected and are used only as exploratory evidence.

## Reproducibility Notes

The final notebooks expect the working directory to contain `outputs/` and the helper Python modules. For that reason, start Jupyter from:

```powershell
cd "d:\BA_year_3\honour project\report_notebooks_outputs"
jupyter lab
```

Then open notebooks from the `notebooks/` folder. Running from the repository root may require changing notebook paths.
