# Old Code Validation Reference

This file captures the current stored outputs from `main_preprocessing.ipynb` together with the saved result summaries already on disk. Use it as a comparison snapshot when you change the modelling framework later.

## Scope

- Source notebook: `main_preprocessing.ipynb`
- Saved summaries: `saved_results/regression_summary.csv`, `saved_results/directional_summary.csv`, `saved_results/result_snapshot.json`
- Purpose: preserve the observable outputs of the old code path so later changes can be validated against a stable reference

## Notebook Cell 2

**Code starts with:** `import sys`

Validation note: this locks down the interpreter path used for the recorded notebook run.

### Output 1 (stream)

```text
D:\keras_env\.venv\Scripts\python.exe
```

## Notebook Cell 6

**Code starts with:** `raw_data = yf.download(`

Validation note: the old run downloaded 3 tickers and produced a multi-index price table. If this shape or ticker layout changes later, downstream comparisons are not like-for-like.

### Output 1 (stream)

```text


[                       0%                       ]
```

### Output 2 (stream)

```text


[**********************67%*******                ]  2 of 3 completed
```

### Output 3 (stream)

```text


[*********************100%***********************]  3 of 3 completed
```

### Output 4 (stream)

```text

```

### Output 5 (execute_result)

```text
Ticker          AAPL                                                MSFT                                                   IBM             \
Price           Open      High       Low     Close     Volume       Open       High        Low      Close    Volume       Open       High   
Date                                                                                                                                        
2010-01-04  6.389119  6.421150  6.357687  6.406482  493729600  22.781978  23.139109  22.759657  23.027506  38409100  71.249085  72.221306   
2010-01-05  6.424144  6.453780  6.383730  6.417558  601904800  22.953107  23.139113  22.796861  23.034948  49749600  71.520651  71.612988   
2010-01-06  6.417557  6.443002  6.308891  6.315477  552160000  22.975429  23.124234  22.707581  22.893587  58182400  70.977511  71.417457   
2010-01-07  6.338825  6.346309  6.257999  6.303800  477131200  22.789418  22.841500  22.462049  22.655495  50559700  70.537541  70.743935   
2010-01-08  6.295420  6.346311  6.258301  6.345712  447610800  22.529011  22.975423  22.499249  22.811739  51197400  70.103058  71.107865   

Ticker                                     
Price             Low      Close   Volume  
Date                                       
2010-01-04  71.069852  71.938873  6438444  
2010-01-05  70.662492  71.069847  7156104  
2010-01-06  70.504983  70.608177  5863144  
2010-01-07  70.016130  70.363739  6109268  
2010-01-08  70.092196  71.069847  4390271
```

## Notebook Cell 7

**Code starts with:** `raw_data.info()`

Validation note: raw download covered 4,024 trading dates with 15 columns (3 tickers x 5 OHLCV fields).

### Output 1 (stream)

```text
<class 'pandas.DataFrame'>
DatetimeIndex: 4024 entries, 2010-01-04 to 2025-12-31
Data columns (total 15 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   (AAPL, Open)    4024 non-null   float64
 1   (AAPL, High)    4024 non-null   float64
 2   (AAPL, Low)     4024 non-null   float64
 3   (AAPL, Close)   4024 non-null   float64
 4   (AAPL, Volume)  4024 non-null   int64  
 5   (MSFT, Open)    4024 non-null   float64
 6   (MSFT, High)    4024 non-null   float64
 7   (MSFT, Low)     4024 non-null   float64
 8   (MSFT, Close)   4024 non-null   float64
 9   (MSFT, Volume)  4024 non-null   int64  
 10  (IBM, Open)     4024 non-null   float64
 11  (IBM, High)     4024 non-null   float64
 12  (IBM, Low)      4024 non-null   float64
 13  (IBM, Close)    4024 non-null   float64
 14  (IBM, Volume)   4024 non-null   int64  
dtypes: float64(12), int64(3)
memory usage: 503.0 KB
```

## Notebook Cell 10

**Code starts with:** `def yfinance_to_long(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:`

Validation note: long-format conversion produced 12,072 rows and 7 columns. This is a key preprocessing checkpoint.

### Output 1 (stream)

```text
(12072, 7)
```

### Output 2 (execute_result)

```text
Price       Date Ticker      Open      High       Low     Close     Volume
0     2010-01-04   AAPL  6.389119  6.421150  6.357687  6.406482  493729600
1     2010-01-05   AAPL  6.424144  6.453780  6.383730  6.417558  601904800
2     2010-01-06   AAPL  6.417557  6.443002  6.308891  6.315477  552160000
3     2010-01-07   AAPL  6.338825  6.346309  6.257999  6.303800  477131200
4     2010-01-08   AAPL  6.295420  6.346311  6.258301  6.345712  447610800
5     2010-01-11   AAPL  6.370259  6.376246  6.240040  6.289733  462229600
6     2010-01-12   AAPL  6.262192  6.279556  6.179272  6.218187  594459600
7     2010-01-13   AAPL  6.222677  6.314279  6.109820  6.305897  605892000
8     2010-01-14   AAPL  6.289732  6.300209  6.257102  6.269375  432894000
9     2010-01-15   AAPL  6.314279  6.334336  6.162806  6.164602  594067600
```

## Notebook Cell 11

**Code starts with:** `import sys`

Validation note: package versions matter for reproducibility, especially pandas / sklearn / yfinance behavior.

### Output 1 (stream)

```text
D:\keras_env\.venv\Scripts\python.exe
scikit-learn: 1.9.0
pandas: 3.0.3
yfinance: 1.4.1
matplotlib: 3.10.9
```

## Notebook Cell 15

**Code starts with:** `features_raw = (`

Validation note: this shows the engineered feature table immediately before modelling-table filtering. In the stored output, Target_Date / Target_State columns were not displayed yet, so this snapshot reflects the earlier visible state.

### Output 1 (stream)

```text
prices columns: ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
prices index names: [None]
features_raw columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker', 'Log_Close', 'Log_Return', 'Simple_Return', 'Next_Day_Log_Return', 'Direction_Target', 'High_Low_Range', 'Open_Close_Log_Return', 'Log_Volume', 'Volume_Change', 'Rolling_Vol_5', 'Momentum_5', 'MA_Gap_5', 'Rolling_Vol_10', 'Momentum_10', 'MA_Gap_10', 'Rolling_Vol_20', 'Momentum_20', 'MA_Gap_20', 'Rolling_Vol_60', 'Momentum_60', 'MA_Gap_60', 'Drawdown_20']
features_raw index names: [None]
```

### Output 2 (execute_result)

```text
Price       Date      Open      High       Low     Close     Volume Ticker  Log_Close  Log_Return  Simple_Return  Next_Day_Log_Return  \
0     2010-01-04  6.389119  6.421150  6.357687  6.406482  493729600   AAPL   1.857310         NaN            NaN             0.001727   
1     2010-01-05  6.424144  6.453780  6.383730  6.417558  601904800   AAPL   1.859038    0.001727       0.001729            -0.016034   
2     2010-01-06  6.417557  6.443002  6.308891  6.315477  552160000   AAPL   1.843003   -0.016034      -0.015906            -0.001851   
3     2010-01-07  6.338825  6.346309  6.257999  6.303800  477131200   AAPL   1.841153   -0.001851      -0.001849             0.006627   
4     2010-01-08  6.295420  6.346311  6.258301  6.345712  447610800   AAPL   1.847779    0.006627       0.006649            -0.008861   

Price  Direction_Target  High_Low_Range  Open_Close_Log_Return  Log_Volume  Volume_Change  Rolling_Vol_5  Momentum_5  MA_Gap_5  \
0                   1.0        0.009906               0.002714   20.017499            NaN            NaN         NaN       NaN   
1                   0.0        0.010915              -0.001026   20.215610       0.198111            NaN         NaN       NaN   
2                   0.0        0.021235              -0.016034   20.129348      -0.086261            NaN         NaN       NaN   
3                   1.0        0.014009              -0.005541   19.983302      -0.146046            NaN         NaN       NaN   
4                   0.0        0.013869               0.007957   19.919435      -0.063867            NaN         NaN -0.001902   

Price  Rolling_Vol_10  Momentum_10  MA_Gap_10  Rolling_Vol_20  Momentum_20  MA_Gap_20  Rolling_Vol_60  Momentum_60  MA_Gap_60  Drawdown_20  
0                 NaN          NaN        NaN             NaN          NaN        NaN             NaN          NaN        NaN          NaN  
1                 NaN          NaN        NaN             NaN          NaN        NaN             NaN          NaN        NaN          NaN  
2                 NaN          NaN        NaN             NaN          NaN        NaN             NaN          NaN        NaN          NaN  
3                 NaN          NaN        NaN             NaN          NaN        NaN             NaN          NaN        NaN          NaN  
4                 NaN          NaN        NaN             NaN          NaN        NaN             NaN          NaN        NaN          NaN
```

## Notebook Cell 16

**Code starts with:** `target_col = "Next_Day_Log_Return"`

Validation note: 11,889 rows survived the modelling-table filter in the old run. This row count is a strong preprocessing regression check.

### Output 1 (stream)

```text
Rows after preprocessing: 11889
Price Ticker       Date     Close  Log_Return  Next_Day_Log_Return
0       AAPL 2010-03-31  7.034826   -0.003610             0.004119
1       AAPL 2010-04-01  7.063863    0.004119             0.010622
2       AAPL 2010-04-05  7.139298    0.010622             0.004393
3       AAPL 2010-04-06  7.170733    0.004393             0.004415
4       AAPL 2010-04-07  7.202463    0.004415            -0.002705
```

## Notebook Cell 18

**Code starts with:** `for ticker in tickers:`

Validation note: these per-ticker head rows are useful for sanity-checking target alignment and sign labels after later edits.

### Output 1 (stream)

```text

AAPL
```

### Output 2 (display_data)

```text
Price       Date     Close  Log_Return  Next_Day_Log_Return  Direction_Target
0     2010-03-31  7.034826   -0.003610             0.004119               1.0
1     2010-04-01  7.063863    0.004119             0.010622               1.0
2     2010-04-05  7.139298    0.010622             0.004393               1.0
3     2010-04-06  7.170733    0.004393             0.004415               1.0
4     2010-04-07  7.202463    0.004415            -0.002705               0.0
```

### Output 3 (stream)

```text

MSFT
```

### Output 4 (display_data)

```text
Price       Date      Close  Log_Return  Next_Day_Log_Return  Direction_Target
7926  2010-03-31  21.894331   -0.016255            -0.004448               0.0
7927  2010-04-01  21.797163   -0.004448             0.003765               1.0
7928  2010-04-05  21.879385    0.003765             0.001707               1.0
7929  2010-04-06  21.916759    0.001707             0.001023               1.0
7930  2010-04-07  21.939182    0.001023             0.019235               1.0
```

### Output 5 (stream)

```text

IBM
```

### Output 6 (display_data)

```text
Price       Date      Close  Log_Return  Next_Day_Log_Return  Direction_Target
3963  2010-03-31  69.969185   -0.004047             0.000000               0.0
3964  2010-04-01  69.969185    0.000000             0.008540               1.0
3965  2010-04-05  70.569305    0.008540            -0.003252               0.0
3966  2010-04-06  70.340195   -0.003252            -0.003497               0.0
3967  2010-04-07  70.094658   -0.003497            -0.006794               0.0
```

## Notebook Cell 19

**Code starts with:** `summary = (`

Validation note: mean and std by ticker give a compact reference for whether return scaling/drift changed unexpectedly.

### Output 1 (execute_result)

```text
            start        end  rows  mean_return  return_std  target_mean  target_std
Ticker                                                                              
AAPL   2010-03-31 2025-12-30  3963     0.000922    0.017746     0.000922    0.017746
IBM    2010-03-31 2025-12-30  3963     0.000365    0.014434     0.000361    0.014437
MSFT   2010-03-31 2025-12-30  3963     0.000778    0.016108     0.000780    0.016106
```

## Notebook Cell 21

**Code starts with:** `train_end = pd.Timestamp("2020-12-31")`

Validation note: split counts per ticker should remain stable unless you intentionally change the date logic.

### Output 1 (execute_result)

```text
                       start        end  rows
Ticker Split                                 
AAPL   test       2023-01-03 2025-12-30   751
       train      2010-03-31 2020-12-31  2709
       validation 2021-01-04 2022-12-30   503
IBM    test       2023-01-03 2025-12-30   751
       train      2010-03-31 2020-12-31  2709
       validation 2021-01-04 2022-12-30   503
MSFT   test       2023-01-03 2025-12-30   751
       train      2010-03-31 2020-12-31  2709
       validation 2021-01-04 2022-12-30   503
```

## Notebook Cell 23

**Code starts with:** `# MinMaxScaler maps features to [0, 1], which is common for LSTM inputs.`

Validation note: this output comes from the legacy globally scaled dataframe. Keep it only as an old-reference checkpoint; the active framework now scales per ticker.

### Output 1 (execute_result)

```text
Price Ticker       Date  Split  Log_Return_scaled  High_Low_Range_scaled  Open_Close_Log_Return_scaled  Volume_Change_scaled  \
0       AAPL 2010-03-31  train           0.533012               0.023438                      0.461556              0.377315   
1       AAPL 2010-04-01  train           0.559447               0.092060                      0.436351              0.517372   
2       AAPL 2010-04-05  train           0.581690               0.051123                      0.568087              0.462758   
3       AAPL 2010-04-06  train           0.560386               0.041986                      0.510020              0.319238   
4       AAPL 2010-04-07  train           0.560460               0.042086                      0.502233              0.518386   

Price  Rolling_Vol_5_scaled  Next_Day_Log_Return  
0                  0.098709             0.004119  
1                  0.066850             0.010622  
2                  0.051607             0.004393  
3                  0.052119             0.004415  
4                  0.035956            -0.002705
```

## Notebook Cell 24

**Code starts with:** `# Check scaling range on the training set.`

Validation note: training-set feature ranges hit [0, 1] in the stored old run. This specifically reflects the earlier global scaling setup.

### Output 1 (execute_result)

```text
                              min  max
Price                                 
Log_Return_scaled             0.0  1.0
High_Low_Range_scaled         0.0  1.0
Open_Close_Log_Return_scaled  0.0  1.0
Volume_Change_scaled          0.0  1.0
Rolling_Vol_5_scaled          0.0  1.0
Rolling_Vol_10_scaled         0.0  1.0
Rolling_Vol_20_scaled         0.0  1.0
Rolling_Vol_60_scaled         0.0  1.0
Momentum_5_scaled             0.0  1.0
Momentum_10_scaled            0.0  1.0
Momentum_20_scaled            0.0  1.0
Momentum_60_scaled            0.0  1.0
MA_Gap_5_scaled               0.0  1.0
MA_Gap_10_scaled              0.0  1.0
MA_Gap_20_scaled              0.0  1.0
MA_Gap_60_scaled              0.0  1.0
Drawdown_20_scaled            0.0  1.0
```

## Notebook Cell 26

**Code starts with:** `def make_lstm_sequences_by_target_split(`

Validation note: pooled sequence counts were train=7,950 / val=1,332 / test=2,076. Later per-ticker counts will differ but the pooled counts are still useful as an old baseline reference.

### Output 1 (stream)

```text
X_train: (7950, 60, 17) y_train: (7950,)
X_val:   (1332, 60, 17) y_val:   (1332,)
X_test:  (2076, 60, 17) y_test:  (2076,)
```

## Notebook Cell 29

**Code starts with:** `def make_yearly_walk_forward_folds(`

Validation note: walk-forward fold row counts are a stable data-splitting checkpoint.

### Output 1 (execute_result)

```text
   fold train_start  train_end  val_start    val_end test_start   test_end  train_rows  val_rows  test_rows
0     1  2010-03-31 2016-12-30 2017-01-03 2017-12-29 2018-01-02 2018-12-31        5106       753        753
1     2  2010-03-31 2017-12-29 2018-01-02 2018-12-31 2019-01-02 2019-12-31        5859       753        756
2     3  2010-03-31 2018-12-31 2019-01-02 2019-12-31 2020-01-02 2020-12-31        6612       756        759
3     4  2010-03-31 2019-12-31 2020-01-02 2020-12-31 2021-01-04 2021-12-31        7368       759        756
4     5  2010-03-31 2020-12-31 2021-01-04 2021-12-31 2022-01-03 2022-12-30        8127       756        753
5     6  2010-03-31 2021-12-31 2022-01-03 2022-12-30 2023-01-03 2023-12-29        8883       753        750
6     7  2010-03-31 2022-12-30 2023-01-03 2023-12-29 2024-01-02 2024-12-31        9636       750        756
7     8  2010-03-31 2023-12-29 2024-01-02 2024-12-31 2025-01-02 2025-12-30       10386       756        747
```

## Notebook Cell 31

**Code starts with:** `output_path = "preprocessed_stock_data.csv"`

Validation note: the old run wrote the unscaled and globally scaled modelling tables to CSV.

### Output 1 (stream)

```text
Saved unscaled modelling table to: preprocessed_stock_data.csv
Saved scaled modelling table to:   preprocessed_stock_data_scaled.csv
```

## Notebook Cell 33

**Code starts with:** `plt.style.use('seaborn-v0_8-whitegrid')`

Validation note: notebook stored a figure for the per-ticker closing-price plot.

### Output 1 (display_data)

```text
<Figure size 1600x1000 with 3 Axes>
```

## Notebook Cell 34

**Code starts with:** `plt.figure(figsize=(15, 10))`

Validation note: notebook stored a figure for the per-ticker volume plot.

### Output 1 (display_data)

```text
<Figure size 1500x1000 with 3 Axes>
```

## Notebook Cell 35

**Code starts with:** `ma_days = [5, 10, 20, 60]`

Validation note: notebook stored a figure for the moving-average plot.

### Output 1 (display_data)

```text
<Figure size 1600x1000 with 3 Axes>
```

## Notebook Cell 62

**Code starts with:** `from plot_helpers import plot_actual_vs_predicted_returns`

Validation note: notebook stored 3 actual-vs-predicted plots for the old single-stock route.

### Output 1 (display_data)

```text
<Figure size 1200x500 with 1 Axes>
```

### Output 2 (display_data)

```text
<Figure size 1200x500 with 1 Axes>
```

### Output 3 (display_data)

```text
<Figure size 1200x500 with 1 Axes>
```

## Notebook Cell 63

**Code starts with:** `ticker = "AAPL"`

Validation note: this debug block is useful for checking whether later models still collapse in variance on AAPL.

### Output 1 (stream)

```text
Small true std: 0.013836404
Small pred std: 0.0043151113
Small pred min/max: -0.010125365 0.0075109843
```

## Notebook Cell 64

**Code starts with:** `plt.figure(figsize=(12, 4))`

Validation note: notebook stored an overfit-debug plot on raw return scale.

### Output 1 (display_data)

```text
<Figure size 1200x400 with 1 Axes>
```

## Notebook Cell 65

**Code starts with:** `print(f"DEBUG CHECK FOR: {ticker}")`

Validation note: confirms there were no NaN/inf issues in the AAPL debug subset used by the old route.

### Output 1 (stream)

```text
DEBUG CHECK FOR: AAPL

Shapes
X_small shape: (200, 60, 17)
y_small_true shape: (200,)
y_small_scaled shape: (200,)

X_small checks
X_small contains NaN: False
X_small contains inf: False
X_small min/max: 0.0022545466 1.0
X_small mean: 0.47138372
X_small std: 0.24696384

y_small_true checks: real return scale
y_small_true contains NaN: False
y_small_true contains inf: False
y_small_true mean: 0.0010979921
y_small_true std: 0.013836392
y_small_true min/max: -0.0462646 0.04029398
First 10 y_small_true: [-0.00858736  0.00598159 -0.0462646  -0.01827876 -0.0122004  -0.00621614
  0.00682028  0.03958704 -0.00224439  0.00591072]

y_small_scaled checks: target scale used by model
y_small_scaled contains NaN: False
y_small_scaled contains inf: False
y_small_scaled mean: 0.0034305633
y_small_scaled std: 0.7790811
y_small_scaled min/max: -2.663399 2.2104259
First 10 y_small_scaled: [-0.5419195   0.27840957 -2.663399   -1.0876095  -0.7453573  -0.40840378
  0.32563332  2.1706207  -0.18476796  0.274419  ]
```

## Notebook Cell 66

**Code starts with:** `print("DEBUG TRAINING RESULT")`

Validation note: the old debug model reduced loss but still produced materially lower variance than the target. This is a good benchmark for later improvements.

### Output 1 (stream)

```text

DEBUG TRAINING RESULT

First loss: 0.6641892194747925
Last loss : 0.34972336888313293
Min loss  : 0.3065141439437866
```

### Output 2 (stream)

```text

Scaled-space prediction
y_small_scaled std: 0.7790811
y_small_pred_scaled std: 0.42449927
y_small_pred_scaled min/max: -1.3055321 1.1603932

Real-return-space prediction
y_small_true std: 0.013836392
y_small_pred std: 0.007539059
y_small_pred min/max: -0.022149038 0.021645518
```

## Notebook Cell 67

**Code starts with:** `plt.figure(figsize=(12, 4))`

Validation note: notebook stored an overfit-debug plot on scaled-target space.

### Output 1 (display_data)

```text
<Figure size 1200x400 with 1 Axes>
```

## Notebook Cell 68

**Code starts with:** `plt.figure(figsize=(12, 4))`

Validation note: notebook stored an overfit-debug plot on real-return space.

### Output 1 (display_data)

```text
<Figure size 1200x400 with 1 Axes>
```

## Saved Regression Summary

Validation note: these are the old persisted regression metrics and are the most important benchmark rows to compare after model changes.

```text
       scope ticker              model      mse     rmse      mae        r2  direction_accuracy  pred_mean     pred_std      pred_min      pred_max  share_predicted_up
      pooled    ALL        always_down 0.000230 0.015162 0.010351 -0.004175            0.446532  -0.000001 1.136868e-13 -1.000000e-06 -1.000000e-06            0.000000
      pooled    ALL          always_up 0.000230 0.015162 0.010351 -0.004158            0.552023   0.000001 1.136868e-13  1.000000e-06  1.000000e-06            1.000000
      pooled    ALL      baseline_lstm 0.000229 0.015141 0.010315 -0.001297            0.548651   0.000325 2.985383e-04 -6.375122e-04  1.629920e-03            0.892582
      pooled    ALL        bigger_lstm 0.000249 0.015789 0.010974 -0.088919            0.552023   0.005477 1.784949e-04  5.087459e-03  6.723322e-03            1.000000
      pooled    ALL predict_train_mean 0.000229 0.015134 0.010299 -0.000369            0.552023   0.000686 0.000000e+00  6.862028e-04  6.862028e-04            1.000000
      pooled    ALL       predict_zero 0.000230 0.015162 0.010351 -0.004167            0.001445   0.000000 0.000000e+00  0.000000e+00  0.000000e+00            0.000000
      pooled    ALL      scaled_y_lstm 0.000229 0.015126 0.010281  0.000680            0.552023   0.000782 3.031316e-04  3.355526e-04  2.659860e-03            1.000000
single_stock   AAPL  lstm_single_stock 0.000261 0.016147 0.010824 -0.003793            0.539017   0.001547 8.113128e-04 -5.120713e-03  5.242031e-03            0.969653
single_stock    IBM  lstm_single_stock 0.000231 0.015212 0.010151 -0.009867            0.459538  -0.000238 2.109986e-04 -5.246049e-04  5.760784e-04            0.127168
single_stock   MSFT  lstm_single_stock 0.000197 0.014032 0.010033  0.004357            0.534682   0.000998 1.355043e-03 -5.738523e-03  5.922810e-03            0.836705
```

## Saved Directional Summary

Validation note: use these rows to compare directional behavior later, especially `direction_accuracy`, `balanced_accuracy`, `hm_stat`, and `share_predicted_up`.

```text
       scope ticker              model      mse     rmse      mae  direction_accuracy  balanced_accuracy  share_actual_up  share_predicted_up  hm_stat  mean_directional_value  cumulative_directional_value  mean_always_up_value  cumulative_always_up_value  tn  fp   fn   tp
      pooled    ALL        always_down 0.000230 0.015162 0.010351            0.447977           0.500000         0.552023            0.000000 1.000000               -0.000977                     -2.027628              0.000977                    2.027628 930   0 1146    0
      pooled    ALL          always_up 0.000230 0.015162 0.010351            0.552023           0.500000         0.552023            1.000000 1.000000                0.000977                      2.027628              0.000977                    2.027628   0 930    0 1146
      pooled    ALL      baseline_lstm 0.000229 0.015141 0.010315            0.548651           0.507890         0.552023            0.892582 1.015780                0.001129                      2.344302              0.000977                    2.027628 108 822  115 1031
      pooled    ALL        bigger_lstm 0.000249 0.015789 0.010974            0.552023           0.500000         0.552023            1.000000 1.000000                0.000977                      2.027628              0.000977                    2.027628   0 930    0 1146
      pooled    ALL predict_train_mean 0.000229 0.015134 0.010299            0.552023           0.500000         0.552023            1.000000 1.000000                0.000977                      2.027628              0.000977                    2.027628   0 930    0 1146
      pooled    ALL       predict_zero 0.000230 0.015162 0.010351            0.447977           0.500000         0.552023            0.000000 1.000000               -0.000977                     -2.027628              0.000977                    2.027628 930   0 1146    0
      pooled    ALL      scaled_y_lstm 0.000229 0.015126 0.010281            0.552023           0.500000         0.552023            1.000000 1.000000                0.000977                      2.027628              0.000977                    2.027628   0 930    0 1146
single_stock   AAPL  lstm_single_stock 0.000261 0.016147 0.010824            0.539017           0.498283         0.543353            0.969653 0.996566                0.000705                      0.487524              0.000779                    0.538760   9 307   12  364
single_stock    IBM  lstm_single_stock 0.000231 0.015212 0.010151            0.460983           0.511846         0.567919            0.127168 1.023692               -0.000876                     -0.605992              0.001334                    0.923051 265  34  339   54
single_stock   MSFT  lstm_single_stock 0.000197 0.014032 0.010033            0.534682           0.504551         0.544798            0.836705 1.009103                0.000675                      0.467086              0.000818                    0.565816  53 262   60  317
```

## Saved Result Manifest

Validation note: this records where the old saved artifacts were written.

```json
{
  "run_name": "main_preprocessing_saved_run_2026-06-03",
  "manifest_path": "D:\\BA_year_3\\honour project\\saved_run_manifest.json",
  "regression_summary_csv": "D:\\BA_year_3\\honour project\\saved_results\\regression_summary.csv",
  "directional_summary_csv": "D:\\BA_year_3\\honour project\\saved_results\\directional_summary.csv",
  "predictions_csv": "D:\\BA_year_3\\honour project\\saved_results\\predictions.csv"
}
```

## How To Use This Later

1. Re-run the notebook after your modelling changes.
2. Compare row counts, split counts, and sequence counts first.
3. Compare the saved regression and directional summary rows second.
4. If metrics change, check whether preprocessing changed too; if preprocessing checkpoints are unchanged, the difference is more likely to come from modelling logic.
5. For the AAPL debug block, compare prediction standard deviation and loss reduction to see whether the new model still collapses toward near-constant forecasts.
