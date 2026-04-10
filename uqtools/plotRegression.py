import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats

def plot_attributes_diagram_regression(data_dict, figsize=(8, 8), model_label="NN", color="green"):
    """
    Plots the Attributes Diagram for a regression task.
    
    Parameters
    ----------
    data_dict : dict
        The dictionary returned by get_reliability_curve_points_regression().
        Contains keys: 'attr_min_val', 'attr_max_val', 'attr_mean_val', 
        'attr_obs_vals', 'attr_pred_vals', 'attr_bin_centers', 'attr_bin_counts'.
    model_label : str, default="NN"
        Name of the model to display in the legend and insets.
    color : str, default="green"
        The color of the reliability curve and the predicted distribution inset.
    Returns
    -------
    fig, ax
    """
    res = data_dict
    fig, ax = plt.subplots(figsize=figsize)
    
    # --- 1. Reference Lines ---> 1-to-1 Line (Perfect Calibration): If points fall here, Pred == Obs
    ax.plot([res['attr_min_val'], res['attr_max_val']], [res['attr_min_val'], res['attr_max_val']], 
            color='gray', linestyle='--', label="1-to-1")
    
    # Climatology Lines (Horizontal and Vertical)
    ax.axhline(res['attr_mean_val'], color='gray', linestyle='--', alpha=0.6, label="Climatology")
    ax.axvline(res['attr_mean_val'], color='gray', linestyle='--', alpha=0.6)
    
    # No-Resolution / No-Skill Line (Halfway between 1:1 and Climatology)
    # y = 0.5 * (Climatology + x)
    x_range = np.array([res['attr_min_val'], res['attr_max_val']])
    no_skill_line = 0.5 * (res['attr_mean_val'] + x_range)
    ax.plot(x_range, no_skill_line, color='gray', linestyle=':', alpha=0.8, label="No-Skill Line")
    
    # --- 2. Positive Skill Area ---
    # Shaded polygon: Where the model is better than just guessing the mean
    ax.fill_between(x_range, x_range, no_skill_line, color='blue', alpha=0.1, label="Positive Skill Area")

    # 3. Plot Reliability Curve
    # Filter out NaNs (bins with no data)
    mask = ~np.isnan(res['attr_pred_vals'])
    ax.plot(res['attr_pred_vals'][mask], res['attr_obs_vals'][mask], 
            color=color, marker='o', linewidth=2, label=f"{model_label}")

    # --- 4. Inset Histograms (The 'Full Distribution' plots) ---
    # Inset Top-Left: Distribution of Ground Truth (y_true)
    ax_true = fig.add_axes([0.15, 0.65, 0.2, 0.2]) 
    ax_true.bar(res['attr_bin_centers'], res['attr_bin_counts'], 
                width=(res['attr_max_val']-res['attr_min_val'])/len(res['attr_bin_centers']),
                color='gray', alpha=0.5)
    ax_true.set_title("$y_{true}$ Dist", fontsize=10)
    ax_true.axis('off')

    # Inset Bottom-Right: Distribution of Model Mean Predictions (y_pred)
    ax_pred = fig.add_axes([0.65, 0.15, 0.2, 0.2])
    ax_pred.bar(res['attr_bin_centers'][mask], res['attr_bin_counts'][mask], 
                width=(res['attr_max_val']-res['attr_min_val'])/len(res['attr_bin_centers']),
                color=color, alpha=0.5)
    ax_pred.set_title(r"$\bar{y}_{pred}$ Dist", fontsize=10)
    ax_pred.axis('off')
    # --- 5. Legend Placement (Below Figure) ---
    # loc='upper center' combined with bbox_to_anchor outside [0,1] range puts it below.
    # ncol=2 spreads the legend items into two columns.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=5, fontsize=10, frameon=True)# Labels and Formatting
    ax.set_xlabel("Mean Predicted Value")
    ax.set_ylabel("Conditional Observed Mean")
    ax.set_xlim(res['attr_min_val'], res['attr_max_val'])
    ax.set_ylim(res['attr_min_val'], res['attr_max_val'])
    ax.set_title(f"Attributes Diagram: {model_label}")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_reliability_curve_regression(attr_dicts_list, model_names, figsize=(8, 8), colors=None):
    """
    Plots multiple reliability curves using pre-calculated attribute dictionaries.

    Parameters
    ----------
    attr_dicts_list : list of dict
        A list of dictionaries returned by get_regression_attributes_data().
    model_names : list of str
        List of names corresponding to each dictionary for the legend.
    colors : list of str, optional
        List of colors for each model line.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Determine Global Min/Max for Axis Scaling
    # We look at all dictionaries to find the absolute min and max
    global_min = min([d['attr_min_val'] for d in attr_dicts_list])
    global_max = max([d['attr_max_val'] for d in attr_dicts_list])
    
    # Perfect Calibration Line
    ax.plot([global_min, global_max], [global_min, global_max], 'k--', alpha=0.7, label="Perfectly Calibrated")

    # 2. Set up colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(attr_dicts_list)))

    # 3. Plot each model's curve from its dictionary
    for i, res in enumerate(attr_dicts_list):
        mask = ~np.isnan(res['attr_pred_vals'])
        ax.plot(res['attr_pred_vals'][mask], res['attr_obs_vals'][mask], 
                marker='o', markersize=5, linewidth=2,
                label=model_names[i], color=colors[i])

    # 4. Formatting to match your paper's style
    ax.set_xlabel("Mean Predicted Value")
    ax.set_ylabel("Conditional Observed Mean")
    ax.set_title("Multi-Model Reliability Comparison")
    ax.set_xlim(global_min, global_max)
    ax.set_ylim(global_min, global_max)
    ax.grid(True, alpha=0.2)
    
    # Legend below the figure in rows/columns
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    plt.show()
    return fig, ax

def plot_spread_vs_skill_curve_regression(ss_dicts_list, model_names, figsize=(8, 8), colors=None):
    """
    Plots Spread vs Skill (RMSE) for multiple models to compare uncertainty calibration.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine global max for 1:1 scaling
    global_max = max([d['ss_max'] for d in ss_dicts_list])
    limit = global_max * 1.1

    # Ideal calibration line
    ax.plot([0, limit], [0, limit], 'k--', alpha=0.6, label="Ideal (Spread=RMSE)")
    
    # Shaded regions for interpretation
    ax.fill_between([0, limit], [0, limit], limit, color='red', alpha=0.03, label='Overconfident')
    ax.fill_between([0, limit], 0, [0, limit], color='blue', alpha=0.03, label='Underconfident')

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(ss_dicts_list)))

    for i, res in enumerate(ss_dicts_list):
        mask = ~np.isnan(res['ss_spread_vals'])
        ax.plot(res['ss_spread_vals'][mask], res['ss_error_vals'][mask], 
                'o-', color=colors[i], linewidth=2, markersize=6, label=model_names[i])

    ax.set_xlabel("Predicted Spread (Uncertainty $\sigma$)", fontsize=12)
    ax.set_ylabel("Actual Error (RMSE)", fontsize=12)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    return fig, ax

def plot_example_histogram_regression(ss_dict, model_label="NN", figsize=(8, 5), color="green"):
    """
    Plots the density of samples across uncertainty bins.
    """
    res = ss_dict
    mask = ~np.isnan(res['ss_spread_vals'])
    bin_centers = res['ss_spread_vals'][mask]
    freqs = (res['ss_bin_counts'][mask] / np.sum(res['ss_bin_counts'])) * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(bin_centers, freqs, width=(res['ss_max']/len(bin_centers)), 
           color=color, alpha=0.6, edgecolor='black')
    
    # Add mean spread reference
    avg_spread = np.average(bin_centers, weights=res['ss_bin_counts'][mask])
    ax.axvline(avg_spread, color='red', linestyle='--', label=f'Mean Spread: {avg_spread:.2f}')
    
    ax.set_xlabel("Predicted Spread ($\sigma$)")
    ax.set_ylabel("% of Total Samples")
    ax.set_title(f"Uncertainty Distribution: {model_label}")
    ax.legend()
    plt.show()
    return fig, ax

def plot_uncertainty_bias_check(ss_dict, model_label="NN", figsize=(8, 5), color="orange"):
    """
    Checks if the model has a systematic bias using the pre-calculated dictionary.
    
    Parameters
    ----------
    ss_dict : dict
        The output from get_regression_spread_skill_data().
    """
    res = ss_dict
    mask = ~np.isnan(res['ss_bias_vals'])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # The "Zero Bias" line is the ideal target
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7) 
    
    # Plot Mean Error vs. Spread
    ax.plot(res['ss_spread_vals'][mask], res['ss_bias_vals'][mask], 
            's-', color=color, linewidth=2, label=f'{model_label} Bias')
    
    ax.set_xlabel("Predicted Spread (Uncertainty $\sigma$)")
    ax.set_ylabel("Mean Error ($Y_{pred} - Y_{true}$)")
    ax.set_title(f"Bias Stability: {model_label}")
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend()
    plt.show()
    return fig, ax

def plot_discard_test_regression(discard_dict, model_label="NN", figsize=(8, 6), color="seagreen"):
    """
    Step 3: Plot the Discard Test (RMSE vs. Discard Fraction).
    Matches classification style with dual axis for example fractions.
    """
    res = discard_dict
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 1. Primary Axis: Remaining RMSE (Left Y-axis)
    line1, = ax1.plot(res['discard_bins'], res['discard_vals'], 'o-', 
                      color=color, linewidth=2.5, markersize=8, label=f'RMSE ({model_label})')
    
    ax1.set_xlabel('Discard fraction (most uncertain examples removed)', fontsize=11)
    ax1.set_ylabel('Remaining RMSE', fontsize=11)
    ax1.set_title(f'Discard Test: {model_label}', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Secondary Axis: Examples Left (Right Y-axis)
    # Using the fraction (1.0 to 0.1) to match classification visuals
    ax2 = ax1.twinx()
    line2, = ax2.plot(res['discard_bins'], res['example_fractions'], 's--', 
                      color='#d62728', alpha=0.5, label='Examples left')
    ax2.set_ylabel('Fraction of examples left', color='#d62728', fontsize=11)
    ax2.set_ylim(0, 1.05) # Fixed scale for fractions
    ax2.tick_params(axis='y', labelcolor='#d62728')

    # 3. Legend: Combined at the bottom for a clean look
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=2, frameon=True)

    plt.tight_layout()
    # Ensure there is room for the legend at the bottom
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    return fig, ax1

def plot_pit_histogram_regression(pit_dict, model_label="NN", figsize=(8, 6), color="seagreen"):
    """
    Step 3: Visualize the PIT Histogram.
    """
    res = pit_dict
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars
    ax.bar(res['bin_centers'], res['pit_freqs'], 
           width=1.0/res['n_bins'], color=color, alpha=0.7, 
           edgecolor='black', label=f"{model_label} (D: {res['pit_dvalue']:.3f})")
    
    # Plot Ideal Uniform Line
    ax.axhline(res['ideal_hline'], color='black', linestyle='--', 
               linewidth=2, label=f"Ideal (E: {res['pit_evalue']:.3f})")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(res['pit_freqs']) * 1.2) # Give some head room
    ax.set_xlabel("PIT (Quantile)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_title(f"PIT Calibration: {model_label}", fontsize=14, fontweight='bold')
    
    ax.legend(loc='lower right', frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_multi_model_pit_regression(pit_dicts_map, figsize=(8, 6), bar_colors=None):
    """
    Plots PIT Histograms for multiple models using grouped bars.
    Bars are clustered at bin centers (0.05, 0.15, etc.)
    """
    model_names = list(pit_dicts_map.keys())
    n_models = len(model_names)
    
    # Extract shared properties from the first model
    first_model = pit_dicts_map[model_names[0]]
    n_bins = first_model['n_bins']
    bin_centers = first_model['bin_centers'] # e.g., [0.05, 0.15, ...]
    ideal_hline = first_model['ideal_hline']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if bar_colors is None:
        # Generate distinct colors for each model
        bar_colors = plt.colormaps['tab10'](np.linspace(0, 1, n_models))
    
    # Logic for grouping
    total_bin_width = 1.0 / n_bins  # e.g., 0.1
    gap_between_groups = 0.2 * total_bin_width # 20% space between clusters
    available_width = total_bin_width - gap_between_groups
    bar_width = available_width / n_models

    for i, name in enumerate(model_names):
        res = pit_dicts_map[name]
        
        # Calculate the offset for this specific model's bar within the cluster
        # This centers the group of bars exactly on the bin_center
        offset = (i - (n_models - 1) / 2) * bar_width
        
        ax.bar(bin_centers + offset, res['pit_freqs'], 
               width=bar_width, 
               color=bar_colors[i], 
               edgecolor='black', 
               alpha=0.8,
               label=f"{name} (D: {res['pit_dvalue']:.3f})")

    # Plot the Ideal Line (dashed horizontal)
    ax.axhline(ideal_hline, color='black', linestyle='--', linewidth=1.5, label="Ideal")
    
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xlabel("PIT (Quantile)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.set_title("PIT Calibration: Multi-Model Comparison", fontsize=13, fontweight='bold')
    
    # Legend at bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    return fig, ax