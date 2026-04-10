import numpy as np

def spread_vs_skill_health(result_dict, generate_report = False):
    """
    Analyzes the result_dict from get_spread_vs_skill to provide 
    a text-based health report of the model's uncertainty calibration.
    """
    counts = result_dict['example_counts']
    spreads = result_dict['mean_prediction_stds']
    skills = result_dict['rmse_values']
    ssrel = result_dict['spread_skill_reliability']

    # Filter for valid bins (where we actually have data)
    valid = counts > 0
    v_spreads = spreads[valid]
    v_skills = skills[valid]
    v_counts = counts[valid]

    # Calculate Global SSRAT
    global_spread = np.average(v_spreads, weights=v_counts)
    global_skill = np.average(v_skills, weights=v_counts)
    ssrat = global_spread / global_skill if global_skill > 0 else np.nan
    if len(v_skills) > 1: correlation = np.corrcoef(v_spreads, v_skills)[0, 1]
    
    if generate_report == True:
        print("--- UQ Health Diagnosis ---")
        print(f"SSREL (Reliability): {ssrel:.4f} (Lower is better)")
        print(f"SSRAT (Ratio):       {ssrat:.4f} (Ideal is 1.0)")
        print("-" * 27)
    
        # 1. Global Assessment
        if 0.9 <= ssrat <= 1.1:
            print("✅ OVERALL: Well-Calibrated. The model's confidence matches its error.")
        elif ssrat < 0.9:
            print(f"⚠️ OVERALL: Overconfident. Model error is {((1/ssrat)-1)*100:.1f}% higher than its spread.")
        else:
            print(f"ℹ️ OVERALL: Underconfident. Model is {((ssrat)-1)*100:.1f}% more 'scared' than it needs to be.")
    
        # 2. Bin-wise Assessment (Identifying non-linear issues)
        # Check if uncertainty increases as error increases (The Correlation check)
        if correlation > 0.8:
            print(f"✅ TREND: High Correlation ({correlation:.2f}). Uncertainty is a great ranking metric.")
        elif correlation > 0.4:
            print(f"⚠️ TREND: Moderate Correlation ({correlation:.2f}). Uncertainty is helpful but noisy.")
        else:
            print(f"❌ TREND: Poor Correlation ({correlation:.2f}). High spread does NOT mean high error.")
    
        # 3. Specific Risk Check: High-Uncertainty Bins
        if v_skills[-1] > v_spreads[-1] * 1.5:
            print("🚨 RISK: Critical Overconfidence in high-uncertainty samples. Errors are exploding.")
    
    return {"ssrat": ssrat, "ssrel": ssrel, "correlation": correlation if len(v_skills)>1 else None}

def discard_health(discard_results, model_name="Model", generate_report = False):
    """
    Computes MF (Monotonicity Fraction) and DI (Discard Improvement) 
    and provides a health assessment of the uncertainty-error relationship.
    """
    errors = discard_results['error_values']
    mf = discard_results['monotonicity_fraction']
    
    # 1. Calculate Discard Improvement (DI)
    # DI = % reduction in error from 0% discard to the maximum discard point
    initial_error = errors[0]
    final_error = errors[-1]
    
    # Handle edge case where initial error is 0
    if initial_error > 0:
        di = (initial_error - final_error) / initial_error * 100
    else:
        di = 0.0
    # --- Average incremental DI (reference-style) ---
    # DI_avg = mean(E_{k-1} - E_k)
    di_avg = np.mean(errors[:-1] - errors[1:])

    if generate_report == True:
        # 2. Assessment Logic
        # We want MF > 0.9 and DI > 15% for a "Healthy" model
        is_monotonic = mf >= 0.9
        is_impactful = di >= 15.0
        
        if is_monotonic and is_impactful:
            grade = "EXCELLENT"
            comment = "Uncertainty perfectly tracks error with significant error reduction."
        elif is_monotonic and not is_impactful:
            grade = "FAIR"
            comment = "Uncertainty tracks error directionally, but the error drop is minimal."
        elif not is_monotonic and is_impactful:
            grade = "UNSTABLE"
            comment = "Significant error reduction possible, but uncertainty ranking is inconsistent."
        else:
            grade = "POOR"
            comment = "Uncertainty does not reliably identify high-error samples."
    
        # 3. Print Report
        print(f"--- Discard Health Report: {model_name} ---")
        print(f"Monotonicity Fraction (MF):      {mf:.2f}")
        print(f"Discard Improvement (DI_end):    {di:.1f}%")
        print(f"Avg. Incremental DI (DI_avg):    {di_avg:.4f}")
        print(f"Overall Health Grade:            {grade}")
        print(f"Diagnosis: {comment}\n")

    return {'mf': mf, 'di': di, 'di_avg': di_avg, 'initial_error': initial_error, 'final_error': final_error}

def reliability_health(rel_dict, model_name="Model", generate_report = False):
    """
    Analyzes the reliability dictionary to evaluate probability calibration.
    """
    bss = rel_dict['bss']
    rel_comp = rel_dict['reliability_comp']
    res_comp = rel_dict['resolution_comp']
    brier_score = rel_dict['brier_score'] # This IS your CRPS
    
    # 1. Expected Calibration Error (ECE) Approximation
    # Weight the absolute difference between forecast and frequency by bin counts
    forecasts = rel_dict['mean_forecast_probs']
    events = rel_dict['mean_event_frequencies']
    counts = rel_dict['num_examples_by_bin']
    
    valid = ~np.isnan(forecasts) & ~np.isnan(events)
    total_n = np.sum(counts[valid])
    ece = np.sum(np.abs(forecasts[valid] - events[valid]) * counts[valid]) / total_n

    if generate_report == True:
        # 2. Skill assessment
        # BSS > 0 means the model is better than climatology
        is_skillful = bss > 0
        
        # 3. Calibration assessment
        # ECE < 0.05 is usually considered well-calibrated
        is_calibrated = ece < 0.05
    
        print(f"--- Reliability Health Report: {model_name} ---")
        print(f"Brier Score (CRPS):       {brier_score:.4f} (Raw error - lower is better)")
        print(f"Brier Skill Score (BSS):   {bss:.3f} ({'SKILLFUL' if is_skillful else 'NO SKILL'})")
        print(f"Calibration Error (ECE):  {ece:.3f} ({'GOOD' if is_calibrated else 'POOR'})")
        print(f"Resolution Component:     {res_comp:.3f} (Ability to separate classes)")
        print(f"Reliability Component:    {rel_comp:.3f} (Lower is better)")
    
        # Qualitative Diagnosis
        if is_skillful and is_calibrated:
            diagnosis = "EXCELLENT: The probabilities are honest and useful."
        elif is_skillful and not is_calibrated:
            diagnosis = "CAUTION: The model has predictive power, but the probabilities are biased/miscalibrated."
        elif not is_skillful and is_calibrated:
            diagnosis = "WEAK: The model is honest about its uncertainty, but it's not very smart (low resolution)."
        else:
            diagnosis = "FAIL: The model is worse than predicting the dataset average."
    
        print(f"Diagnosis: {diagnosis}\n")

    return {'crps_bs': brier_score, 'bss': bss, 'ece': ece}

#====================================== Reports for Regression Problems=======================================================================#
def reliability_health_regression(attr_dict):
    """
    Extracts scalar diagnostic metrics from the Attributes Diagram dictionary.
    
    Parameters:
    -----------
    attr_dict : dict
        The output from get_regression_attributes_data()
        
    Returns:
    --------
    diag_report : dict
        A dictionary containing the MSESS and Reliability metrics.
    """
    # 1. Calculate MSE of the model using the binned values
    # We use the binned counts to weight the errors correctly
    mask = ~np.isnan(attr_dict['attr_pred_vals'])
    p = attr_dict['attr_pred_vals'][mask]
    o = attr_dict['attr_obs_vals'][mask]
    w = attr_dict['attr_bin_counts'][mask]
    
    # Model Mean Squared Error
    mse_model = np.sum(w * (p - o)**2) / np.sum(w)
    
    # 2. Calculate MSE of the Climatology (Baseline)
    # This represents the variance of the observations
    mse_climo = np.sum(w * (o - attr_dict['attr_mean_val'])**2) / np.sum(w)
    
    # 3. MSESS (Mean Squared Error Skill Score)
    # 1.0 is perfect, 0.0 means no better than the average, negative is worse than average
    msess = 1.0 - (mse_model / mse_climo)
    
    return {
        "MSESS": msess,
        "Climatology_Mean": attr_dict['attr_mean_val'],
        "Binned_RMSE": np.sqrt(mse_model)
    }

def spread_vs_skill_health_regression(ss_dict):
    """
    Computes SSRAT and SSREL from binned spread-skill data.
    """
    spreads = ss_dict['ss_spread_vals']
    rmses = ss_dict['ss_error_vals']
    counts = ss_dict['ss_bin_counts']
    
    # Filter out empty bins
    mask = ~np.isnan(spreads) & ~np.isnan(rmses)
    s = spreads[mask]
    e = rmses[mask]
    w = counts[mask]
    
    # 1. SSRAT (Spread-Skill Ratio): Perfect = 1.0
    # Average Spread / Global RMSE
    ssrat = np.average(s, weights=w) / np.sqrt(np.average(e**2, weights=w))
    
    # 2. SSREL (Spread-Skill Reliability): Perfect = 0.0
    # Weighted average of the distance from the 1:1 line
    ssrel = np.average(np.abs(s - e), weights=w)
    
    return {"SSRAT": ssrat, "SSREL": ssrel}

def discard_health_regression(discard_dict):
    """
    Computes Monotonicity and improvement percentage for regression discard test.
    """
    rmse_vals = discard_dict['discard_vals']
    
    # 1. Handle potential NaNs at the end of the array
    valid_mask = ~np.isnan(rmse_vals)
    valid_rmse = rmse_vals[valid_mask]
    
    if len(valid_rmse) < 2:
        return {"Error": "Not enough valid bins to compute health metrics"}

    # 2. Monotonicity: Is error dropping?
    # Perfect UQ = 1.0. Random UQ ~ 0.5.
    monotonicity = np.mean(np.diff(valid_rmse) < 0)
    
    # 3. Improvement: % drop from 0% discarded to the furthest valid discard point.
    total_drop_pct = (valid_rmse[0] - valid_rmse[-1]) / valid_rmse[0] * 100
    
    # 4. AUC-Sparsification (Advanced Health Metric)
    # This measures the area under the curve. 
    # A lower area means the model is "smarter" at discarding error quickly.
    auc_sparsification = np.trapz(valid_rmse, discard_dict['discard_bins'][valid_mask])

    return {
        'Monotonicity': monotonicity,
        'RMSE_Drop_Pct': total_drop_pct,
        'Baseline_RMSE': valid_rmse[0],
        'Final_RMSE': valid_rmse[-1],
        'AUC_Sparsification': auc_sparsification
    }

def pit_health_regression(pit_dict):
    """
    Step 2: Interpret PIT calibration.
    """
    d = pit_dict['pit_dvalue']
    e = pit_dict['pit_evalue']
    
    # A common rule of thumb: If D < E, the calibration is excellent.
    is_calibrated = d <= e
    
    return {
        'Calibration_Deviation_D': d,
        'Expected_Deviation_E': e,
        'Is_Well_Calibrated': is_calibrated,
        'D_to_E_Ratio': d / e
    }