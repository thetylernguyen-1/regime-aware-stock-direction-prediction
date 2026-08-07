# Honour Project

Final project layout for:

**Hidden market regimes in weekly stock-market direction classification: comparing standard and HMM-informed LSTMs**

The top level contains only the core material needed to read the report and inspect the code/results behind it. Older experiments, full output bundles, draft notes, and unused files are kept in `legacy/`.

## Folder Guide

```text
honour-project/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- notebooks/
|-- figures/
|-- outputs/
|-- report/
`-- legacy/
```

## Main Report

- `report/final_report.pdf` - final submitted report.

## Core Notebooks

Read the notebooks in this order:

1. `notebooks/01_weekly_preprocessing.ipynb`
   - Builds weekly data, features, target labels, chronological splits, scaling, and LSTM sequences.
   - Supports report Section 3.

2. `notebooks/02_weekly_hmm.ipynb`
   - Fits ticker-specific weekly two-state HMMs and creates regime probabilities.
   - Supports report Sections 3.3 and 5.

3. `notebooks/03_controlled_hmm_lstm_comparison.ipynb`
   - Main controlled comparison between technical-only LSTM and technical-plus-HMM LSTM.
   - Supports report Sections 4 and 6.

4. `notebooks/04_market_context_experiments.ipynb`
   - Exploratory market, volatility, QQQ, and regime-context experiments.
   - Supports report Section 6.1.

5. `notebooks/05_rolling_replication.ipynb`
   - Rolling paired replication for AAPL and IBM.
   - Supports report Section 7.

## Final Outputs

`outputs/` contains selected final summary files only:

- `weekly_hmm_state_summary.csv`
- `weekly_hmm_expected_durations.csv`
- `weekly_hmm_transition_matrices.csv`
- `controlled_model_summary.csv`
- `table6_bootstrap_recomputed.csv`
- `rolling_paired_hmm_summary.csv`
- `rolling_paired_hmm_auc_deltas.csv`
- `rolling_sign_flip_pvalues.csv`
- `best_found_roc_auc_summary.csv`
- `report_code_consistency_checks.csv`

These files are the direct result tables behind the final report's main numerical claims.

## Final Figures

`figures/` contains the final figures used in the report. Older or unused figures are in `legacy/old_figures/`.

## Legacy

`legacy/` contains everything that is not part of the clean final review layout:

- `legacy/daily_regression/`
- `legacy/earlier_hmm_experiments/`
- `legacy/exploratory_notebooks/`
- `legacy/old_figures/`
- `legacy/report_materials/`
- `legacy/notes_and_review_files/`

The full original notebook/output bundle is archived at:

- `legacy/report_materials/full_notebook_output_bundle/`

Use that archived bundle if you need every generated intermediate file, helper Python module, old notebook, or full reproducibility context.

## Consistency Note

The final report values were checked against the saved outputs. The selected consistency results are in:

- `outputs/report_code_consistency_checks.csv`

The full audit material is archived in `legacy/report_materials/`.
