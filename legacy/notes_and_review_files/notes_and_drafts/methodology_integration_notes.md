# Methodology Integration Notes

## 1. Where to Insert the New Methodology

Insert `methodology_rewrite.tex` at `overleaf_report/main.tex` line 206, replacing the current `\section{Methodology}` block from lines 206-460. The replacement should run from `\section{Methodology}` through the end of the new `Secondary and Robustness Analyses` subsection.

The current lines 206-460 mix historical daily regression, exploratory weekly LSTM results, HMM result tables, and early HMM-LSTM results into Methodology. The new section keeps methods separate from results and uses the final controlled implementation as the authoritative source.

## 2. Sections to Rename, Move, or Delete

| Current section | Current location | Recommendation |
|---|---:|---|
| `Data and Feature Engineering` | lines 138-205 | Keep before Methodology, but revise line 202 to remove "estimate class weights" for final controlled LSTMs. Add the exact yfinance `auto_adjust=True` wording if data source detail is needed here. |
| `Methodology` | lines 206-460 | Replace with `methodology_rewrite.tex`. |
| `Daily LSTM Regression` under Methodology | lines 208-212 | Move to a historical development or appendix section; do not include in final Methodology except as motivation. |
| Weekly LSTM result tables/figures | lines 226-276 | Move to Results or Appendix as exploratory development evidence. |
| Weekly HMM result tables/figures | lines 282-374 | Move tables/figures to Results/EDA/Appendix. Keep only HMM specification in Methodology. |
| `HMM-LSTM Feature Fusion` early results | lines 376-448 | Move to Exploratory Results. It describes the initial HMM-LSTM with two HMM features, which is not the final controlled method. |
| `Evaluation Metrics` | lines 450-458 | Replace with the fuller metrics subsection in `methodology_rewrite.tex`. |
| `Extended Experiments` | lines 461-621 | Keep after Methodology, but rename to `Secondary and Robustness Analyses` or move after the final controlled Results section. |
| `Final Controlled Comparison` | lines 622-741 | Keep as the primary Results section. Consider renaming to `Primary Results: Controlled HMM Ablation`. |
| `Rolling Paired Replication Design` | lines 743-767 | Rename to `Rolling Paired Replication` and revise because final rolling result tables and figures now exist. |

## 3. Current Paragraphs That Duplicate the New Methodology

- Lines 173-179 duplicate target definition and row counts. Keep in Data, but avoid repeating detailed split controls in Methodology.
- Lines 200-204 duplicate leakage-control language. Keep a short overview in Data and let the new Methodology carry exact controls.
- Lines 216-224 duplicate LSTM input and threshold notation, but line 216 incorrectly uses the earlier 26-week grid for the final methodology.
- Lines 280-281 duplicate HMM feature construction, but need to be replaced by the exact `HMM_Positive_Return_Prob_Next` to `HMM_Up_Prob_Raw` to `HMM_Up_Prob_Scaled` chain.
- Lines 378-379 describe the initial HMM-LSTM using two HMM features; this conflicts with the final controlled HMM-LSTM and should be moved to exploratory narrative only.
- Lines 624-630 summarize the final controlled design. Keep this in Results as a short reminder, but avoid duplicating full methodology details.

## 4. Introduction Claims to Revise

- Line 66 says the HMM-LSTM receives "regime or positive-return probabilities." Revise to distinguish exploratory versus primary: the final controlled HMM-LSTM used one scaled HMM-implied next-week up-probability feature.
- Line 77 asks whether HMM-derived regime probabilities improve forecasts. Revise the primary question to: "Does an HMM-derived probability provide incremental out-of-sample value for weekly LSTM stock-direction forecasting beyond stock-specific technical features?"
- Lines 61-67 should avoid implying the final primary HMM feature is a pure regime probability. It is an HMM-derived positive-return probability.
- Objective wording at lines 82-89 is broadly fine, but objective 4 should say "technical-plus-HMM LSTM" rather than generic "HMM-informed LSTM" if referring to the controlled comparison.

## 5. Integrated Gradients References

No completed Integrated Gradients analysis was found. The only project hit was a helper function in `lstm_architecture.py`; no executed attribution tables, feature/lag aggregation, figures, or interpretation artifacts were found.

If any draft text outside `overleaf_report/main.tex` claims Integrated Gradients was used, remove it or rewrite it as unexecuted future work. No Integrated Gradients paragraph should appear in Methodology, Results, or Discussion unless the analysis is actually executed and saved.

## 6. Should "Extended Experiments" Remain After Methodology?

Yes, but not as part of the primary methodology. Recommended placement:

1. Keep `Methodology` as the clean controlled design section.
2. Rename current `Extended Experiments` to `Secondary and Robustness Analyses`.
3. Put it after the primary controlled Results section if the report structure is Results-driven, or keep it immediately after Methodology if the report structure is chronology-driven.

The section should make clear that volatility context, QQQ/relative features, recession/expansion variables, richer HMMs, three-state HMMs, target-overlap diagnostics, and rolling replication are secondary or robustness analyses.

## 7. Where HMM Regime Visualisations Should Appear

- `weekly_hmm_returns_by_regime.png`: Results or EDA subsection showing descriptive regime interpretation.
- `weekly_hmm_transition_matrices.png`: Results or Appendix; useful if expected durations are discussed.
- `weekly_hmm_positive_prob_time.png`: Appendix or secondary diagnostic unless the main Results section discusses the HMM-only benchmark.
- `weekly_hmm_only_roc.png`: Results if HMM-only benchmark is discussed; otherwise Appendix.
- Regime visualisations should not sit inside Methodology except perhaps as a small schematic. Methodology should define how the HMM was fitted, not present empirical regime findings.

## 8. Tables and Plots by Report Location

| Artifact | Best location |
|---|---|
| Feature list and LSTM architecture table | Methodology or Appendix |
| Split/class balance table | Data or Methodology |
| HMM state summary and expected duration tables | Results or Appendix |
| Weekly LSTM exploratory selected-model tables | Exploratory Results or Appendix |
| Initial HMM-LSTM same-lookback deltas | Exploratory Results |
| Controlled model summary table | Primary Results |
| Controlled ROC/PR curves and balanced accuracy plot | Primary Results |
| Controlled validation lookback/logistic selection plots | Appendix or brief Methodology support |
| Volatility/QQQ/context grid tables | Secondary Results or Appendix |
| HMM robustness variant table and rolling ROC diagnostics | Secondary/Robustness Results |
| Rolling fold definitions | Appendix |
| Rolling paired AUC deltas and summary | Secondary/Robustness Results |
| Calibration bins and target-overlap diagnostics | Secondary/Robustness Results or Appendix |

## 9. Equations and Label References to Update

- Keep the target equation currently around lines 175-177, but ensure notation aligns with the new Methodology: \(y_{i,w+1}=\mathbf{1}\{r_{i,w+1}>0\}\).
- Add or update HMM equations for \(S_{i,w}\), \(A_{i,ab}\), \(x_{i,w}\mid S_{i,w}=k\), filtered probabilities, one-step state probabilities, and HMM-implied up probability.
- Keep the balanced-accuracy equation, but move it into the new Methodology metrics subsection.
- Update any references to `tab:weekly-lstm-selected`, `tab:hmm-lstm-selected`, or `tab:hmm-lstm-delta` if those tables move to Appendix or Exploratory Results.
- Add a new label for the architecture table, e.g. `tab:method-lstm-architecture`.
- If rolling results are added, replace `tab:rolling-summary` or add a new rolling results table label such as `tab:rolling-paired-results`.

## 10. Rolling Replication Correction

Current `overleaf_report/main.tex` lines 765-767 state that final rolling result cells were not executed and that rolling should be treated only as a design. This is now contradicted by saved artifacts:

- `outputs/weekly_rolling_hmm_replication_aapl_ibm/rolling_fold_model_results.csv`
- `outputs/weekly_rolling_hmm_replication_aapl_ibm/rolling_paired_hmm_auc_deltas.csv`
- `outputs/weekly_rolling_hmm_replication_aapl_ibm/rolling_paired_hmm_summary.csv`
- `outputs/weekly_rolling_hmm_replication_aapl_ibm/rolling_oos_predictions.csv`
- rolling figures in `outputs/weekly_rolling_hmm_replication_aapl_ibm/figures/`

Revise the rolling section to say it was completed as a secondary paired replication for AAPL and IBM. Also revise line 804 and line 816, which describe rolling as incomplete future work. A defensible wording is:

> The rolling replication was completed for AAPL and IBM and provides secondary robustness evidence. It does not replace the primary fixed-split controlled comparison because AAPL used volatility-context base features whereas IBM used technical base features.

## 11. Proposed Revised Table of Contents

1. Introduction
2. Research Questions and Contributions
3. Literature and Conceptual Background
4. Data and Feature Engineering
5. Methodology
6. Primary Results: Controlled HMM Ablation
7. Secondary and Robustness Analyses
8. Rolling Paired Replication
9. Discussion
10. Limitations
11. Conclusion
12. Appendix A: Notebook Map
13. Appendix B: Feature Definitions and Architecture Details
14. Appendix C: Exploratory Model Grids
15. Appendix D: Reproducibility Notes

## 12. Highest-Priority Edits

1. Replace the current Methodology block with `methodology_rewrite.tex`.
2. Correct the final HMM feature wording everywhere: final controlled LSTM uses `HMM_Up_Prob_Scaled`, not two HMM features and not `HMM_Bullish_Prob_Next`.
3. Correct the final lookback grid to `{4, 8, 12}` for the controlled comparison.
4. Remove final-method claims about class weights.
5. Remove or revise any Integrated Gradients claims.
6. Update the rolling section from incomplete design to completed secondary replication.
7. Reframe IBM 0.585 versus 0.583 as a tiny difference without formal uncertainty evidence.
