import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def get_mean_std_regression(mc_preds):
    """
    Summarize Monte Carlo predictive distributions for regression.
    
    Parameters
    ----------
    mc_preds : np.ndarray, shape (N, Q)
        Monte Carlo predicted values for N samples across Q forward passes.

    Returns
    -------
    mean_preds : np.ndarray, shape (N,)
        The central prediction (mean of the ensemble).
    predictive_stds : np.ndarray, shape (N,)
        The predictive standard deviation (measure of spread/uncertainty).
    """
    mc_preds = np.asarray(mc_preds)
    
    # Check dimensionality
    assert mc_preds.ndim == 2, "mc_preds must have shape (N, Q)"
    
    # Calculate mean and sample standard deviation (ddof=1 is often preferred for small ensembles)
    mean_preds = np.mean(mc_preds, axis=1)
    predictive_stds = np.std(mc_preds, axis=1, ddof=1)
    
    return mean_preds, predictive_stds

def create_contours(min_val, max_val, n_bins):
    """Linearly spaced bin centers."""
    return np.linspace(min_val, max_val, n_bins)

def get_edges(bin_centers):
    """
    Derives bin edges from centers. 
    Each edge is placed exactly halfway between centers.
    """
    # Calculate midpoints
    cmid = [bin_centers[j] - (bin_centers[j + 1] - bin_centers[j]) * 0.5 
            for j in range(len(bin_centers) - 1)]
    
    # Add the outer start and end edges
    # (distance from last center to last edge is half a bin width)
    c1 = [bin_centers[-1] - (bin_centers[-1] - bin_centers[-2]) * 0.5]
    c2 = [bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) * 0.5]
    
    return np.array(cmid + c1 + c2)

def get_reliability_curve_points_regression(y_true, y_pred, y_train=None, n_bins=10):
    """
    Single-target logic for Attributes Diagram data preparation.
    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground truth continuous values for the test set.
    y_pred : np.ndarray, shape (N,Q)
        The model's ensemble predictions (e.g., the MC ensemble).
    y_train : np.ndarray, shape (M,), optional
        Training data labels used to calculate the climatology mean. 
        If None, the mean of y_true is used as the baseline.
    n_bins : int, default=10
        Number of bins to divide the prediction range into.

    Returns
    -------
    dict
        A dictionary containing:
        - 'attr_min_val': Overall minimum value for axis scaling.
        - 'attr_max_val': Overall maximum value for axis scaling.
        - 'attr_mean_val': The climatology baseline (scalar).
        - 'attr_obs_vals': Mean observations per bin, shape (n_bins,).
        - 'attr_pred_vals': Mean predictions per bin, shape (n_bins,).
        - 'attr_bin_centers': Center coordinate for each bin on the x-axis, shape (n_bins,).
        - 'attr_bin_counts': Number of samples in each bin, shape (n_bins,).
        - 'attr_tick_vals': Recommended integer tick marks for the plot.
    """
    #get the mean
    y_pred, _ = get_mean_std_regression(y_pred)
    # 1. Handle Dynamic Range (Min/Max Detection)
    # This ensures the plot axes fit the data perfectly
    min_val = np.min([y_pred.min(), y_true.min()])
    max_val = np.max([y_pred.max(), y_true.max()])
    
    # 2. Climatology Baseline (Skill anchor)
    # Vital for the 'No-Skill' line in the Attributes Diagram
    if y_train is not None:
        mean_val = np.mean(y_train)
    else:
        mean_val = np.mean(y_true) # Fallback if training data is missing
        
    # 3. Bin Geometry
    # Logic: Centers -> Edges ensures points are plotted in the middle of buckets
    bin_centers = create_contours(min_val, max_val, n_bins)
    bin_edges = get_edges(bin_centers)
    
    # 4. Statistical Binning
    pred_vals = np.empty(n_bins)
    obs_vals = np.empty(n_bins)
    bin_counts = np.empty(n_bins)
    
    for i in range(n_bins):
        # Identify indices falling within the current bin boundaries
        # Logical range: [edge_low, edge_high)
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        
        if np.any(mask):
            pred_vals[i] = np.mean(y_pred[mask])
            obs_vals[i] = np.mean(y_true[mask])
            bin_counts[i] = np.sum(mask)
        else:
            pred_vals[i] = np.nan
            obs_vals[i] = np.nan
            bin_counts[i] = 0
            
    # Package into a dictionary matching the dictionary structure
    result_dict = {
        'attr_min_val': min_val,
        'attr_max_val': max_val,
        'attr_mean_val': mean_val,
        'attr_obs_vals': obs_vals,
        'attr_pred_vals': pred_vals,
        'attr_bin_centers': bin_centers,
        'attr_bin_counts': bin_counts,
        'attr_tick_vals': np.linspace(min_val, max_val, 6).astype(int)
    }
    return result_dict

def get_spread_vs_skill_regression(y_true, y_pred_matrix, n_bins=12):
    """
    Consolidated Single-Target Spread-Skill logic.
    
    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_pred_matrix : np.ndarray, shape (N, Q) - raw ensemble/MC passes
    n_bins : int
    """
    # 1. Convert ensemble to stats
    # mean_preds: (N,), std_preds: (N,)
    mean_preds, std_preds = get_mean_std_regression(y_pred_matrix)
    
    # 2. Define Bins based on uncertainty (Spread)
    bin_edges = np.linspace(std_preds.min(), std_preds.max(), n_bins + 1)
    
    # Initialize containers
    spread_vals = np.full(n_bins, np.nan)
    rmse_vals = np.full(n_bins, np.nan)
    bias_vals = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (std_preds >= bin_edges[i]) & (std_preds < bin_edges[i + 1])
        
        if np.any(mask):
            counts[i] = np.sum(mask)
            spread_vals[i] = np.mean(std_preds[mask])
            # RMSE
            rmse_vals[i] = np.sqrt(np.mean((y_true[mask] - mean_preds[mask])**2))
            # BIAS
            bias_vals[i] = np.mean(mean_preds[mask] - y_true[mask])

    # For a square plot, we need the max of both axes
    valid_vals = np.concatenate([spread_vals[~np.isnan(spread_vals)], 
                                 rmse_vals[~np.isnan(rmse_vals)]])
    ss_max = np.max(valid_vals) if len(valid_vals) > 0 else 1.0

    return {
        'ss_spread_vals': spread_vals,
        'ss_error_vals': rmse_vals,
        'ss_bias_vals': bias_vals,
        'ss_bin_counts': counts,
        'ss_bin_edges': bin_edges,
        'ss_max': ss_max
    }

def rmse(y_true, y_pred):
    """Computes Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred)**2))
def get_discard_test_regression(y_true, y_pred_matrix, discard_bins=None):
    """
    Computes discard points by internally calculating mean/std from an ensemble matrix.
    
    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        True values.
    y_pred_matrix : np.ndarray, shape (N, Q)
        Raw predictions from ensemble members or MC passes.
    discard_bins : np.ndarray, optional
        Fractions to discard. Defaults to np.linspace(0., 0.9, 10).
    """
    # Internal call to get mean and std from the prediction matrix
    # Assumes get_mean_std_regression is defined in your environment
    ymean, ystd = get_mean_std_regression(y_pred_matrix)
    
    if discard_bins is None:
        nbins = 10
        discard_bins = np.linspace(0., 0.9, nbins)
    else:
        nbins = len(discard_bins)

    # Flattening for safety (Single Target)
    ytrue1d = y_true.reshape(-1)
    ymean1d = ymean.reshape(-1)
    ystd1d = ystd.reshape(-1)
    nsamples = ystd1d.shape[0]

    # Sort indices based on uncertainty (ystd)
    yrefs = np.argsort(ystd1d)
    ytrueSorted = ytrue1d[yrefs]
    ymeanSorted = ymean1d[yrefs]

    rmse_out = np.empty(nbins)
    example_fractions = np.empty(nbins) # For consistency with classification

    # 3. Compute error reduction loop
    for i in range(nbins):
        # iCutoff is the number of LEAST uncertain samples to keep
        iCutoff = nsamples - int(nsamples * discard_bins[i])
        
        if iCutoff > 0:
            ytrueH = ytrueSorted[:iCutoff]
            ymeanH = ymeanSorted[:iCutoff]
            # Assumes rmse(y_true, y_pred) helper is defined
            rmse_out[i] = rmse(ytrueH, ymeanH)
        else:
            rmse_out[i] = np.nan
            
        # Matching classification naming: fraction of examples left (1.0 down to 0.1)
        example_fractions[i] = iCutoff / nsamples

    return {
        'discard_vals': rmse_out,
        'discard_bins': discard_bins,
        'example_fractions': example_fractions,
        'pdp_norm_mean': np.mean(ystd1d),
        'total_samples': nsamples
    }

def get_regression_pit_data(y_true, y_pred_matrix, n_bins=10):
    """
    Step 1: Calculate PIT values and calibration metrics (D and E).
    """
    # 1. Internal stats from matrix
    ymean, ystd = get_mean_std_regression(y_pred_matrix)
    
    # 2. Compute PIT values: Where does y_true sit in the predicted Gaussian?
    # Values will be between 0 and 1
    pit_values = scipy.stats.norm.cdf(x=y_true, loc=ymean, scale=ystd)
    
    # 3. Create histogram counts
    pit_bins = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(pit_values, bins=pit_bins)
    
    # Normalize counts to probability (frequencies)
    pit_freqs = counts / np.sum(counts)
    bin_centers = (pit_bins[:-1] + pit_bins[1:]) / 2
    
    # 4. Calculate D-value (Calibration Deviation)
    ideal_freq = 1.0 / n_bins
    d_value = np.sqrt(np.mean((pit_freqs - ideal_freq)**2))
    
    # 5. Calculate E-value (Expected sampling noise)
    # n_samples = tsamples
    n_samples = len(y_true)
    e_value = np.sqrt((1 - 1/n_bins) / (n_samples * n_bins))
    
    return {
        'pit_values': pit_values,
        'pit_freqs': pit_freqs,
        'bin_centers': bin_centers,
        'pit_dvalue': d_value,
        'pit_evalue': e_value,
        'n_bins': n_bins,
        'ideal_hline': ideal_freq
    }