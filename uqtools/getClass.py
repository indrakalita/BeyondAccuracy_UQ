# getClass.py
"""
Binary Classification UQ Utilities

Contains functions to compute:
- Predictive mean & std of MC probabilities
- Spread vs Skill metrics
- Discard test
- ROC and Performance diagram with uncertainty
- Reliability curve and Brier Skill Score
"""

import numpy as np
from sklearn.metrics import roc_curve, auc

# --------------------------------------------
# Function 1: Mean & std of MC predictions
# --------------------------------------------
def get_mean_std(mc_probs):
    """
    Summarize Monte Carlo predictive distributions (mean and std) for binary classification.

    Parameters
    ----------
    mc_probs : np.ndarray, shape (N, Q)
        Monte Carlo predicted probabilities for the positive class.
        N = number of samples
        Q = number of MC forward passes

    Returns
    -------
    mean_probs : np.ndarray, shape (N,)
        Predictive mean probability for each sample.

    predictive_stds : np.ndarray, shape (N,)
        Predictive standard deviation (epistemic uncertainty proxy).
    """
    mc_probs = np.asarray(mc_probs)
    assert mc_probs.ndim == 2, "mc_probs must have shape (N, Q)"
    assert np.all((mc_probs >= 0) & (mc_probs <= 1)), "Probabilities must be in [0, 1]"

    mean_probs = np.mean(mc_probs, axis=1)
    predictive_stds = np.std(mc_probs, axis=1, ddof=1)
    return mean_probs, predictive_stds

# --------------------------------------------
# Function 2: Spread vs Skill
# --------------------------------------------
def get_spread_vs_skill(prediction_matrix, target_values, bin_edges_std):
    """
    Computes spread vs. skill for a set of MC predictive distributions.

    Parameters
    ----------
    prediction_matrix : np.ndarray, shape (N, Q)
        Monte Carlo predicted probabilities for the positive class. 
        N = number of samples, Q = number of MC passes.

    target_values : np.ndarray, shape (N,)
        True binary labels (0 or 1).

    bin_edges_std : np.ndarray, shape (B-1,)
        Bin edges for predictive standard deviations. 
        0 and +inf are automatically added.

    Returns
    -------
    result_dict : dict
        Keys:
        - 'mean_prediction_stds': mean predictive std per bin
        - 'bin_edges_std': bin edges including 0 and +inf
        - 'rmse_values': RMSE of mean predictions per bin
        - 'spread_skill_reliability': weighted average |spread - skill|
        - 'example_counts': number of examples in each bin
        - 'mean_central_predictions': mean of mean predictions in each bin
        - 'mean_target_values': mean target value per bin
    """

    # Check input
    bin_edges_std = np.asarray(bin_edges_std)
    assert np.all(bin_edges_std > 0.) and np.all(bin_edges_std < 1.)
    assert np.all(np.diff(bin_edges_std) > 0.)

    # Add 0 and inf to bin edges
    bin_edges_std = np.concatenate(([0.], bin_edges_std, [np.inf]))
    num_bins = len(bin_edges_std) - 1
    assert num_bins >= 2

    # Compute mean and std per sample
    mean_preds, pred_stds = get_mean_std(prediction_matrix)
    squared_errors = (mean_preds - target_values) ** 2 #This is probabilistic error

    # Initialize arrays. For each bin compute the following 
    mean_pred_stds_bin = np.full(num_bins, np.nan) #Mean predictive std (Average uncertainty)
    rmse_bin = np.full(num_bins, np.nan) #RMSE (Actual error))
    example_counts_bin = np.zeros(num_bins, dtype=int) # Example count (Reliability weight)
    mean_central_preds_bin = np.full(num_bins, np.nan) # Mean Prediction (Bias check)
    mean_target_vals_bin = np.full(num_bins, np.nan) # Mean target (Base rate)

    # Bin-by-bin computation
    for k in range(num_bins):
        idx = np.where((pred_stds >= bin_edges_std[k]) & (pred_stds < bin_edges_std[k + 1]))[0] # get all the index where condition meets
        
        if len(idx) > 0:
            mean_pred_stds_bin[k] = np.mean(pred_stds[idx])
            rmse_bin[k] = np.sqrt(np.mean(squared_errors[idx]))
            example_counts_bin[k] = len(idx)
            mean_central_preds_bin[k] = np.mean(mean_preds[idx])
            mean_target_vals_bin[k] = np.mean(target_values[idx])

    # Compute spread–skill reliability (SSREL).Value = {0: perfect uncertainty, small: good uncertainty, large: poor/misleading uncertainty}
    diffs = np.abs(mean_pred_stds_bin - rmse_bin)
    diffs[np.isnan(diffs)] = 0.
    spread_skill_reliability = np.average(diffs, weights=example_counts_bin) #weighted avg. Multiply each diff by its weight(example_counts_bin)

    # Prepare output
    result_dict = {
        'mean_prediction_stds': mean_pred_stds_bin,
        'bin_edges_std': bin_edges_std,
        'rmse_values': rmse_bin,
        'spread_skill_reliability': spread_skill_reliability,
        'example_counts': example_counts_bin,
        'mean_central_predictions': mean_central_preds_bin,
        'mean_target_values': mean_target_vals_bin
    }

    return result_dict

# --------------------------------------------
# Function 3: Discard Test
# --------------------------------------------
def get_discard_test(prediction_matrix, target_values, discard_fractions):
    """
    Computes the discard test: error vs discard fraction.

    The discard test removes the most uncertain predictions (highest predictive std)
    progressively and evaluates error on remaining examples.

    Parameters
    ----------
    prediction_matrix : np.ndarray, shape (E, Q)
        Monte Carlo predictions for E examples and Q samples.
    target_values : np.ndarray, shape (E,)
        True labels (0 or 1 for binary classification)
    discard_fractions : np.ndarray, shape (F-1,)
        Fractions of examples to discard, values in (0,1) means discard top 10%, 20%, 30% uncertain

    Returns
    -------
    result_dict : dict
        Contains the following keys:
        - 'discard_fractions': fractions including 0
        - 'error_values': mean cross-entropy of remaining examples
        - 'example_fractions': fraction of examples left after discard
        - 'mean_central_predictions': mean prediction of remaining examples
        - 'mean_target_values': mean target of remaining examples
        - 'monotonicity_fraction': fraction of times error decreases as discard fraction increases
    """

    # Ensure discard fractions are valid and include 0 (baseline, no examples removed).
    discard_fractions = np.concatenate(([0.], np.sort(discard_fractions)))
    num_fractions = len(discard_fractions)
    assert num_fractions >= 2

    # Compute mean prediction and predictive std per example
    mean_predictions, predictive_stds = get_mean_std(prediction_matrix)
    # Compute per-sample cross-entropy (error)
    epsilon = 1e-8  # numerical stability to prevents invalid inputs. Negative numbers are also invalid for log and log2(0) = -inf
    cross_entropy = -(target_values * np.log2(mean_predictions + epsilon) +
                      (1 - target_values) * np.log2(1 - mean_predictions + epsilon))

    # Prepare arrays
    error_values = np.full(num_fractions, np.nan)
    example_fractions = np.full(num_fractions, np.nan)
    mean_central_predictions = np.full(num_fractions, np.nan)
    mean_target_values = np.full(num_fractions, np.nan)
    use_examples = np.ones_like(mean_predictions, dtype=bool)

    # Loop over discard fractions
    for k, frac in enumerate(discard_fractions):
        # Compute threshold percentile to discard top uncertain examples (Compute percentile of uncertainty)
        percentile_level = 100 * (1 - frac) # convert to %
        threshold = np.percentile(predictive_stds, percentile_level)
        remaining = predictive_stds <= threshold
        #discard_mask = predictive_stds > np.percentile(predictive_stds, percentile_level)
        #use_examples[discard_mask] = False

        # Compute metrics for remaining examples
        #remaining = use_examples
        example_fractions[k] = np.mean(remaining)
        error_values[k] = np.mean(cross_entropy[remaining])
        mean_central_predictions[k] = np.mean(mean_predictions[remaining])
        mean_target_values[k] = np.mean(target_values[remaining])

    # Monotonicity fraction: Checks if the error decreases as more uncertain examples are removed.
    monotonicity_fraction = np.mean(np.diff(error_values) < 0) #Ideal UQ: monotonicity fraction → 1 (always decreasing error).

    # Return results
    return {
        'discard_fractions': discard_fractions, #fractions of discarded examples
        'error_values': error_values, #error (cross-entropy) after discarding
        'example_fractions': example_fractions, #fraction of examples left
        'mean_central_predictions': mean_central_predictions, #mean prediction of remaining examples
        'mean_target_values': mean_target_values, #mean target of remaining examples
        'monotonicity_fraction': monotonicity_fraction #fraction of times error decreases as discard increases
    }
# --------------------------------------------
# Function 4: Reliability curve + Brier skill
# --------------------------------------------
def get_reliability_curve_points(observed_labels, forecast_probabilities, num_bins=20, climatology=None):
    """
    Compute points for the reliability curve and Brier skill score. Returns a dictionary for easier handling and storage.
    Args:
        observed_labels: (E,) array of 0/1 labels
        forecast_probabilities: (E,) array of predicted probabilities for class 1
        num_bins: number of bins for forecast probability
        climatology: float, event frequency (if None, computed from observed_labels)
    
    Returns: Dict with all the required details
        
    """
    observed_labels = np.asarray(observed_labels).astype(int)
    forecast_probabilities = np.asarray(forecast_probabilities)
    
    if climatology is None:
        climatology = np.mean(observed_labels)
    
    # Assign forecasts to bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    inputs_to_bins = np.minimum(np.digitize(forecast_probabilities, bin_edges) - 1, num_bins - 1)
    
    mean_forecast_probs = np.full(num_bins, np.nan)
    mean_event_frequencies = np.full(num_bins, np.nan)
    num_examples_by_bin = np.zeros(num_bins, dtype=int)
    
    for k in range(num_bins):
        indices = np.where(inputs_to_bins == k)[0]
        num_examples_by_bin[k] = len(indices)
        if len(indices) > 0:
            mean_forecast_probs[k] = np.mean(forecast_probabilities[indices])  #mean predicted probability for all samples in bin k
            mean_event_frequencies[k] = np.mean(observed_labels[indices]) #mean of the actual observed labels for the same samples in bin k
    
    # Brier Score Decomposition
    # 1. Uncertainty (UNC): Inherent variance of the data
    uncertainty = climatology * (1 - climatology)
    
    # 2. Reliability (REL): Weighted squared distance from the diagonal
    mask = num_examples_by_bin > 0
    reliability = np.nansum(num_examples_by_bin * (mean_forecast_probs - mean_event_frequencies)**2) / np.sum(num_examples_by_bin)
    
    # 3. Resolution (RES): Ability to distinguish different outcomes
    sample_climatology = np.average(mean_event_frequencies[mask], weights=num_examples_by_bin[mask])
    resolution = np.nansum(num_examples_by_bin * (mean_event_frequencies - sample_climatology)**2) / np.sum(num_examples_by_bin)
    
    # Total Brier Score
    brier_score = uncertainty + reliability - resolution
    brier_skill_score = 1 - brier_score / uncertainty if uncertainty > 0 else np.nan
    
    return {
        'mean_forecast_probs': mean_forecast_probs,
        'mean_event_frequencies': mean_event_frequencies,
        'num_examples_by_bin': num_examples_by_bin,
        'bss': brier_skill_score,
        'brier_score': brier_score,
        'reliability_comp': reliability,
        'resolution_comp': resolution,
        'uncertainty_comp': uncertainty,
        'climatology': climatology
    }
# --------------------------------------------
# Function 5: ROC with UQ
# --------------------------------------------
def get_roc_with_uq(prediction_matrix, target_values, uncertainty_split=True):
    """
    Computes ROC curve using MC predictions and optionally splits by predictive uncertainty.

    Parameters
    ----------
    prediction_matrix : np.ndarray, shape (E, Q)
        Monte Carlo predictions for E examples and Q samples
    target_values : np.ndarray, shape (E,)
        True binary labels (0 or 1)
    uncertainty_split : bool
        If True, also returns ROC for low and high uncertainty examples separately

    Returns
    -------
    result_dict : dict
        Contains:
        - 'fpr_all', 'tpr_all', 'auc_all' : ROC and AUC using all examples
        - 'fpr_low', 'tpr_low', 'auc_low' : ROC/AUC for low uncertainty examples (optional)
        - 'fpr_high', 'tpr_high', 'auc_high' : ROC/AUC for high uncertainty examples (optional)
        - 'mean_predictions', 'predictive_stds' : per-example mean and std
    """
    # Compute mean prediction and predictive std
    mean_preds, pred_stds = get_mean_std(prediction_matrix)
    # Overall ROC
    fpr_all, tpr_all, _ = roc_curve(target_values, mean_preds)
    auc_all = auc(fpr_all, tpr_all)

    result_dict = {
        'fpr_all': fpr_all,
        'tpr_all': tpr_all,
        'auc_all': auc_all,
        'mean_predictions': mean_preds,
        'predictive_stds': pred_stds
    }

    if uncertainty_split:
        # Median split by predictive std
        median_std = np.median(pred_stds)
        low_mask = pred_stds <= median_std
        high_mask = pred_stds > median_std

        # Low uncertainty ROC
        if np.any(low_mask):
            fpr_low, tpr_low, _ = roc_curve(target_values[low_mask], mean_preds[low_mask])
            auc_low = auc(fpr_low, tpr_low)
        else:
            fpr_low, tpr_low, auc_low = None, None, None

        # High uncertainty ROC
        if np.any(high_mask):
            fpr_high, tpr_high, _ = roc_curve(target_values[high_mask], mean_preds[high_mask])
            auc_high = auc(fpr_high, tpr_high)
        else:
            fpr_high, tpr_high, auc_high = None, None, None

        # Add to result dict
        result_dict.update({
            'fpr_low': fpr_low,
            'tpr_low': tpr_low,
            'auc_low': auc_low,
            'fpr_high': fpr_high,
            'tpr_high': tpr_high,
            'auc_high': auc_high
        })

    return result_dict

# --------------------------------------------
# Function 6: Performance diagram with UQ
# --------------------------------------------
def get_perf_diagram_with_uq(prediction_matrix, target_values, uncertainty_split=True, num_thresholds=1001):
    """
    Computes points for a performance diagram using MC predictions and optionally splits by uncertainty.

    Parameters
    ----------
    prediction_matrix : np.ndarray, shape (E, Q)
        Monte Carlo predictions for E examples and Q samples
    target_values : np.ndarray, shape (E,)
        True binary labels (0 or 1)
    uncertainty_split : bool
        If True, also returns POD/Success Ratio for low/high uncertainty examples
    num_thresholds : int
        Number of thresholds to evaluate

    Returns
    -------
    result_dict : dict
        Contains:
        - 'pod_all', 'sr_all' : POD and Success Ratio for all examples
        - 'pod_low', 'sr_low' : POD/SR for low uncertainty (optional)
        - 'pod_high', 'sr_high' : POD/SR for high uncertainty (optional)
        - 'mean_predictions', 'predictive_stds' : per-example mean and std
    """
    # Compute mean prediction and predictive std
    mean_preds, pred_stds = get_mean_std(prediction_matrix)
    
    thresholds = np.linspace(0, 1, num_thresholds)
    
    def compute_points(preds, targets):
        pod = np.full(num_thresholds, np.nan)
        sr = np.full(num_thresholds, np.nan)
        
        for k, th in enumerate(thresholds):
            forecast_labels = (preds >= th).astype(int)
            hits = np.sum((forecast_labels == 1) & (targets == 1))
            false_alarms = np.sum((forecast_labels == 1) & (targets == 0))
            misses = np.sum((forecast_labels == 0) & (targets == 1))
            
            if hits + misses > 0:
                pod[k] = hits / (hits + misses)
            if hits + false_alarms > 0:
                sr[k] = hits / (hits + false_alarms)
        
        # Add endpoints for plotting
        pod = np.array([1.] + pod.tolist() + [0.])
        sr = np.array([0.] + sr.tolist() + [1.])
        return pod, sr
    
    # Overall
    pod_all, sr_all = compute_points(mean_preds, target_values)
    
    result_dict = {
        'pod_all': pod_all,
        'sr_all': sr_all,
        'mean_predictions': mean_preds,
        'predictive_stds': pred_stds
    }
    
    if uncertainty_split:
        median_std = np.median(pred_stds)
        low_mask = pred_stds <= median_std
        high_mask = pred_stds > median_std
        
        if np.any(low_mask):
            pod_low, sr_low = compute_points(mean_preds[low_mask], target_values[low_mask])
        else:
            pod_low, sr_low = None, None
        
        if np.any(high_mask):
            pod_high, sr_high = compute_points(mean_preds[high_mask], target_values[high_mask])
        else:
            pod_high, sr_high = None, None
        
        result_dict.update({
            'pod_low': pod_low,
            'sr_low': sr_low,
            'pod_high': pod_high,
            'sr_high': sr_high
        })
    
    return result_dict
