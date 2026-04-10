# uqtools

`uqtools` is a Python package for **uncertainty quantification (UQ) and plotting** in both **regression and classification** tasks.  
It supports Monte Carlo or ensemble predictions and provides tools for metrics, visualizations, and automated health reports.


## Modules

## `getClass.py`

**Binary Classification UQ Utilities**  

`getClass.py` provides functions to quantify and analyze **uncertainty in binary classification** using Monte Carlo predictions or ensembles. It includes methods to compute predictive statistics, spread vs skill, discard tests, reliability curves, ROC curves, and performance diagrams.

### Key Functions

1. **`get_mean_std(mc_probs)`**  
   Computes the **predictive mean** and **predictive standard deviation** (uncertainty) for each sample from MC probabilities.  
   - Input: `(N, Q)` MC predictions  
   - Output: `(N,)` mean probabilities and predictive standard deviations

2. **`get_spread_vs_skill(prediction_matrix, target_values, bin_edges_std)`**  
   Evaluates **spread vs skill**: how predicted uncertainty compares to actual error across bins.  
   - Returns weighted RMSE, mean predictive std per bin, example counts, and spread–skill reliability.

3. **`get_discard_test(prediction_matrix, target_values, discard_fractions)`**  
   Computes the **discard test**, which removes the most uncertain predictions progressively and evaluates the **error reduction**.  
   - Outputs cross-entropy error, fraction of remaining examples, and monotonicity fraction.

4. **`get_reliability_curve_points(observed_labels, forecast_probabilities, num_bins=20, climatology=None)`**  
   Computes **reliability curves** and **Brier Skill Score** for predicted probabilities.  
   - Returns per-bin mean forecast, observed frequency, and Brier score components (uncertainty, reliability, resolution).

5. **`get_roc_with_uq(prediction_matrix, target_values, uncertainty_split=True)`**  
   Computes **ROC curve and AUC**, optionally split by **low/high uncertainty** examples.  
   - Returns FPR, TPR, AUC, and per-sample mean predictions and uncertainty.

6. **`get_perf_diagram_with_uq(prediction_matrix, target_values, uncertainty_split=True, num_thresholds=1001)`**  
   Generates a **performance diagram** (POD vs Success Ratio) and optionally splits by predictive uncertainty.  
   - Returns POD and SR curves for all, low, and high uncertainty examples.

---

**Purpose:**  
`getClass.py` is useful for **evaluating the quality of predictive probabilities**, measuring **uncertainty calibration**, and identifying when **uncertainty is informative** for discarding high-risk predictions.

## `getRegression.py`

**Regression UQ Utilities**  

`getRegression.py` provides functions to quantify and analyze **uncertainty in regression tasks** using Monte Carlo predictions or ensembles. It includes methods to compute predictive statistics, spread vs skill, discard tests, and PIT calibration for continuous outputs.

### Key Functions

1. **`get_mean_std_regression(mc_preds)`**  
   Computes the **predictive mean** and **predictive standard deviation** for each sample from Monte Carlo predictions.  
   - Input: `(N, Q)` MC predictions  
   - Output: `(N,)` mean predictions and predictive standard deviations

2. **`get_reliability_curve_points_regression(y_true, y_pred, y_train=None, n_bins=10)`**  
   Prepares data for an **Attributes Diagram**, summarizing model predictions versus observed values per bin.  
   - Returns per-bin mean predictions, observed values, counts, and a baseline climatology mean.

3. **`get_spread_vs_skill_regression(y_true, y_pred_matrix, n_bins=12)`**  
   Computes **spread vs skill** for regression: compares predictive uncertainty to RMSE across bins.  
   - Outputs mean predictive standard deviations, RMSE per bin, bias, bin counts, and max axis value.

4. **`get_discard_test_regression(y_true, y_pred_matrix, discard_bins=None)`**  
   Performs a **discard test** for regression: removes the most uncertain predictions and evaluates RMSE reduction.  
   - Returns RMSE per discard fraction, example fractions, mean uncertainty, and total sample count.

5. **`get_regression_pit_data(y_true, y_pred_matrix, n_bins=10)`**  
   Computes **Probability Integral Transform (PIT)** for regression calibration.  
   - Returns PIT values, histogram frequencies, D-value (calibration deviation), and E-value (expected sampling noise).

6. **Helper Functions**  
   - `rmse(y_true, y_pred)` — computes root mean squared error.  
   - `create_contours(min_val, max_val, n_bins)` — generates linearly spaced bin centers.  
   - `get_edges(bin_centers)` — calculates bin edges from centers for plotting.  

---

**Purpose:**  
`getRegression.py` is designed to evaluate **uncertainty calibration for continuous predictions**, analyze **spread vs actual errors**, and measure how predictive uncertainty can guide **risk-aware decision-making** in regression.

## `plotClass.py`

**Plotting Utilities for Binary Classification UQ**  

`plotClass.py` provides functions to **visualize uncertainty and model performance** in binary classification tasks. It supports spread vs skill, discard tests, ROC curves, performance diagrams, reliability curves, and attributes diagrams.

### Key Functions

1. **`plot_spread_vs_skill_curve(result_dicts, model_names=None, ...)`**  
   Plots **spread vs skill curves** (predictive standard deviation vs actual RMSE) for one or more models.  
   - Can include reference line (`spread = skill`) and over/underconfident shading.  
   - Supports color coding, legends, and sample count annotations.

2. **`plot_example_histogram(result_dict, ...)`**  
   Shows **histogram of example counts per uncertainty bin**.  
   - Highlights the distribution of samples across predictive uncertainty.  
   - Optionally labels percentages and shows average spread.

3. **`plot_mean_pred_vs_target(result_dict, ...)`**  
   Plots **mean predicted probability vs mean observed target** per bin.  
   - Useful to check **class-specific bias** or calibration trends.

4. **`plot_spread_vs_rmse_bar(result_dict, ...)`**  
   Displays **RMSE per uncertainty bin** as a bar chart.  
   - Supports gradient coloring and optional bar labels.  
   - Highlights the relationship between predictive spread and actual error.

5. **`plot_discard_test(result_dict, ...)`**  
   Visualizes the **discard test**: error vs fraction of most uncertain examples removed.  
   - Optionally overlays fraction of examples remaining on a secondary y-axis.  

6. **`plot_reliability_curve(result_dicts, ...)`**  
   Plots **reliability curves** for one or more models.  
   - Compares forecast probability to observed frequency.  
   - Useful for calibration assessment.

7. **`plot_attributes_diagram(rel_dict, ...)`**  
   Creates a **full Attributes Diagram** with inset histogram (sharpness).  
   - Combines reliability, no-skill line, and data distribution in one figure.

8. **`plot_roc_with_uq(roc_result_dict, ...)`**  
   Plots **ROC curves with uncertainty**: overall, low, and high uncertainty subsets.  
   - Displays AUC for each subset.  

9. **`plot_perf_diagram_with_uq(perf_result_dict, ...)`**  
   Draws **performance diagrams** with Probability of Detection (POD) vs Success Ratio (Precision).  
   - Supports splitting by predictive uncertainty.  

---

**Purpose:**  
`plotClass.py` is designed to provide **comprehensive visualization tools for uncertainty in binary classification**, enabling analysis of model calibration, confidence, spread-skill relationships, and error reduction strategies.

## `plotRegression.py`

**Plotting Utilities for Regression Uncertainty Quantification (UQ)**  

`plotRegression.py` provides functions to **visualize model predictions, uncertainty, calibration, and skill** for regression tasks. It supports attributes diagrams, reliability curves, spread-skill analysis, discard tests, bias checks, and PIT histograms.

### Key Functions

1. **`plot_attributes_diagram_regression(data_dict, ...)`**  
   Plots an **Attributes Diagram** for regression.  
   - Visualizes conditional mean observations vs mean predictions.  
   - Includes perfect calibration, climatology, no-skill lines, and positive skill shading.  
   - Adds inset histograms for predicted and observed distributions.

2. **`plot_reliability_curve_regression(attr_dicts_list, model_names, ...)`**  
   Compares **multiple regression models** on one reliability plot.  
   - Shows calibration of each model against the 1:1 line.  
   - Supports automatic global axis scaling.

3. **`plot_spread_vs_skill_curve_regression(ss_dicts_list, model_names, ...)`**  
   Plots **predicted spread vs actual RMSE** to evaluate uncertainty calibration.  
   - Highlights overconfident and underconfident regions.  
   - Supports multiple models with different colors.

4. **`plot_example_histogram_regression(ss_dict, model_label, ...)`**  
   Displays the **distribution of samples across uncertainty bins**.  
   - Shows percentage of samples per bin and mean spread reference line.

5. **`plot_uncertainty_bias_check(ss_dict, ...)`**  
   Checks for **systematic bias** in model predictions.  
   - Plots mean error (prediction minus observation) against predicted spread.  
   - Zero line represents ideal unbiased behavior.

6. **`plot_discard_test_regression(discard_dict, ...)`**  
   Visualizes **RMSE vs discard fraction**: the error when removing the most uncertain examples.  
   - Dual y-axis for fraction of examples remaining.  
   - Matches classification-style discard visuals.

7. **`plot_pit_histogram_regression(pit_dict, ...)`**  
   Plots the **Probability Integral Transform (PIT) histogram**.  
   - Shows predictive calibration for a single regression model.  
   - Includes ideal uniform line and D/E calibration metrics.

8. **`plot_multi_model_pit_regression(pit_dicts_map, ...)`**  
   Plots **PIT histograms for multiple regression models** as grouped bars.  
   - Useful for comparing calibration across models.  
   - Includes ideal uniform line and legend annotations.

---

**Purpose:**  
`plotRegression.py` provides comprehensive visualization for regression UQ. It enables assessment of model **calibration, spread-skill alignment, bias, discard performance, and predictive distribution**, supporting both single-model and multi-model analysis.

## `getReport.py`

**Uncertainty Health & Diagnostic Reports**  

`getReport.py` provides **textual and numerical diagnostics** for both classification and regression models.  
These functions evaluate the **quality of predictive uncertainty**, calibration, spread-skill alignment, discard performance, and PIT calibration.

### Classification Health Checks

1. **`spread_vs_skill_health(result_dict, generate_report=False)`**  
   Evaluates the **spread vs. skill** relationship for classification models.  
   - Computes **SSRAT** (Spread/Skill ratio) and **SSREL** (Spread-Skill Reliability).  
   - Assesses global calibration, bin-wise correlation, and high-uncertainty risk.  
   - `generate_report=True` prints a full textual assessment with emojis.

2. **`discard_health(discard_results, model_name="Model", generate_report=False)`**  
   Evaluates the **Discard Test**.  
   - Computes **Monotonicity Fraction (MF)** and **Discard Improvement (DI)**.  
   - Assigns an overall health grade: EXCELLENT, FAIR, UNSTABLE, POOR.  
   - `generate_report=True` prints detailed textual diagnosis.

3. **`reliability_health(rel_dict, model_name="Model", generate_report=False)`**  
   Evaluates **probability calibration**.  
   - Calculates **Brier Score (CRPS)**, **Brier Skill Score (BSS)**, and **Expected Calibration Error (ECE)**.  
   - Provides qualitative interpretation: EXCELLENT, CAUTION, WEAK, FAIL.

---

### Regression Health Checks

1. **`reliability_health_regression(attr_dict)`**  
   Extracts **scalar diagnostics** from regression attributes diagrams.  
   - Computes **Mean Squared Error Skill Score (MSESS)**, climatology mean, and binned RMSE.  

2. **`spread_vs_skill_health_regression(ss_dict)`**  
   Evaluates **Spread vs Skill** for regression models.  
   - Computes **SSRAT** and **SSREL** using weighted bins.  
   - Helps assess uncertainty calibration in continuous tasks.

3. **`discard_health_regression(discard_dict)`**  
   Evaluates regression discard tests.  
   - Computes **monotonicity**, **RMSE drop (%)**, and **AUC sparsification**.  
   - Measures how well predicted uncertainty identifies high-error samples.

4. **`pit_health_regression(pit_dict)`**  
   Evaluates **PIT (Probability Integral Transform) calibration** for regression.  
   - Computes calibration deviation (`D`), expected deviation (`E`), and `D/E` ratio.  
   - Boolean flag `Is_Well_Calibrated` indicates whether the model is statistically well-calibrated.

---

**Purpose:**  
`getReport.py` allows for **quantitative and qualitative evaluation** of model uncertainty.  
It bridges **numeric metrics** and **human-readable assessments** for both classification and regression tasks, providing actionable insights for model calibration, discard tests, and overall predictive reliability.

## Installation
### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/uq-plotting.git
cd uq-plotting
```
### 2. Create a Python Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows

### 3. Install Required Dependencies
pip install numpy matplotlib scipy OR pip install -r requirements.txt

### 4. Verify Installation
import plotClass
import plotRegression
import getReport

## Minimal Example

Here's how to use `uq_plotting` for visualizing and evaluating uncertainty quantification.

### 1. Binary Classification

```python
import numpy as np
from uqtools import plotClass, getReport

# --- Fake data for illustration ---
result_dict = {
    'mean_prediction_stds': np.array([0.1, 0.2, 0.3, 0.4]),
    'rmse_values': np.array([0.12, 0.18, 0.28, 0.38]),
    'example_counts': np.array([50, 40, 30, 20]),
    'spread_skill_reliability': 0.05
}

# Plot Spread vs Skill
fig, ax = plotClass.plot_spread_vs_skill_curve(result_dict, show_samples=True)

# Print health report
health = getReport.spread_vs_skill_health(result_dict, generate_report=True)
```
### 2. Regression
```python
import numpy as np
from uq_plotting import plotRegression, getReport

# --- Fake regression spread-skill data ---
ss_dict = {
    'ss_spread_vals': np.array([0.2, 0.4, 0.6, 0.8]),
    'ss_error_vals': np.array([0.25, 0.45, 0.55, 0.78]),
    'ss_bin_counts': np.array([50, 40, 30, 20]),
    'ss_max': 0.8
}

# Plot Spread vs Skill for regression
fig, ax = plotRegression.plot_spread_vs_skill_curve_regression([ss_dict], ["NN Model"])

# Get health report
report = getReport.spread_vs_skill_health_regression(ss_dict)
print(report)




