# plotClass.py
"""
Plotting Utilities for Binary Classification UQ

Includes:
- Spread vs Skill
- Example Histogram
- Mean Prediction vs Target
- Spread vs RMSE Bar
- Discard Test
- ROC Curve with UQ
- Performance Diagram with UQ
- Reliability Curve
- Attributes Diagram
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. Spread vs Skill Curve
# --------------------------------------------
def plot_spread_vs_skill_curve(result_dicts, model_names=None, figsize= (7, 7), reference_line=True, colors=None, show_title=True, show_samples=False):
    """
    Plots Spread vs Skill curve: predictive std (spread) vs RMSE (skill).

    Parameters
    ----------
    result_dict : dict
        Output from compute_spread_vs_skill
    reference_line : bool
        If True, plots dashed diagonal: spread = skill
    line_color : str
        Color of main line

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # Ensure inputs are lists even if a single model is passed
    if isinstance(result_dicts, dict):
        result_dicts = [result_dicts]
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(result_dicts))]
    elif isinstance(model_names, str):
        model_names = [model_names]
        
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Determine Global Plot Limits (to keep 1:1 ratio consistent for all models)
    global_max = 0
    for rd in result_dicts:
        m_max = np.nanmax(rd['mean_prediction_stds'])
        r_max = np.nanmax(rd['rmse_values'])
        global_max = max(global_max, m_max, r_max)
    
    limit_val = global_max * 1.1 if global_max > 0 else 0.5
    
    # 2. Draw Reference Background
    if reference_line:
        ax.plot([0, limit_val], [0, limit_val], '--', color='black', alpha=0.6, label='Ideal (Spread=Skill)')
        ax.fill_between([0, limit_val], [0, limit_val], limit_val, color='red', alpha=0.03, label='Overconfident')
        ax.fill_between([0, limit_val], 0, [0, limit_val], color='blue', alpha=0.03, label='Underconfident')

    # 3. Setup Colors
    if colors is None:
        # Use a standard colormap for multiple models
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i) for i in range(len(result_dicts))]
    elif isinstance(colors, str):
        colors = [colors]

    # 4. Plot each model
    for rd, name, color in zip(result_dicts, model_names, colors):
        mean_stds = rd['mean_prediction_stds']
        rmse = rd['rmse_values']
        
        valid = ~np.isnan(mean_stds) & ~np.isnan(rmse)
        
        ax.plot(mean_stds[valid], rmse[valid], 'o-', color=color, markersize=7, 
                linewidth=2, label=name, zorder=5)
    
    # 5. Optional Statistical Annotation (Summary of all N)
    if show_samples:
        total_n = sum([np.sum(rd['example_counts']) for rd in result_dicts])
        ax.text(0.05, 0.95, f'Total samples (all): {total_n:,}', transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Formatting
    ax.set_xlabel('Spread (Predictive Standard Deviation)', fontsize=12)
    ax.set_ylabel('Skill (Actual RMSE)', fontsize=12)
    
    if show_title:
        title = 'Spread-Skill Comparison' if len(result_dicts) > 1 else 'Spread-Skill Reliability'
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, limit_val-0.1)
    ax.set_ylim(0, limit_val)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Move legend smartly
    n_models = len(result_dicts)
    if n_models <= 7: # change the number to get legend at the bottom
        ax.legend(loc='lower right', frameon=True)
    else:
        ncols = min(n_models, 4)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=ncols, frameon=True)
    
    plt.tight_layout()
    return fig, ax
# --------------------------------------------
# 2. Histogram of Examples per Bin
# --------------------------------------------
def plot_example_histogram(result_dict, figsize=(8, 5), color='skyblue', show_title=True, show_bar_labels=False):
    """
    Plots the distribution of samples across bins with optional percentage labels.
    Parameters
    ----------
    result_dict : dict
        Output from compute_spread_vs_skill
    color : str
        Bar color

    Returns
    -------
    fig, ax
    """
    counts = result_dict['example_counts']
    bin_edges = result_dict['bin_edges_std'].copy()
    
    if np.isinf(bin_edges[-1]):
        bin_edges[-1] = bin_edges[-2] + (bin_edges[-2] - bin_edges[-3])
    
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    freqs = (counts / np.sum(counts)) * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(bin_centers, freqs, width=widths, color=color, 
                  edgecolor='black', alpha=0.6, label='Sample Density')
    
    # Optional labels on top of each bar
    if show_bar_labels:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Vertical line for the average spread
    avg_spread = np.average(bin_centers, weights=counts)
    ax.axvline(avg_spread, color='red', linestyle='--', linewidth=2, 
               label=f'Mean Spread ({avg_spread:.2f})')

    ax.set_xlabel('Spread (Predictive Standard Deviation)', fontsize=11)
    ax.set_ylabel('% of Total Examples', fontsize=11)
    
    if show_title:
        ax.set_title('Data Distribution across Uncertainty Bins', fontsize=13, fontweight='bold')
    
    ax.set_xlim(0, 0.55)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    return fig, ax
# --------------------------------------------
# 3. Mean Prediction vs Mean Target
# --------------------------------------------
def plot_mean_pred_vs_target(result_dict, figsize = (8, 5), color_pred='blue', color_target='red', show_title=True):
    """
    Plots mean prediction vs mean target per bin to identify class-specific bias
    Parameters
    ----------
    result_dict : dict
    color_pred : str
        Line color for mean prediction
    color_target : str
        Line color for mean target

    Returns
    -------
    fig, ax
    """
    mean_preds = result_dict['mean_central_predictions']
    mean_targets = result_dict['mean_target_values']
    bin_edges = result_dict['bin_edges_std'].copy()
    
    # Handle the infinite bin for visualization
    if np.isinf(bin_edges[-1]):
        # Use the width of the second-to-last bin to extend the last one
        bin_edges[-1] = bin_edges[-2] + (bin_edges[-2] - bin_edges[-3])
    
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Reference line at 0.5 (Maximum uncertainty for binary classification)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Max Uncertainty (0.5)')
    
    # 2. Plot Mean Targets (Actual Outcomes)
    ax.plot(bin_centers, mean_targets, 's--', color=color_target, 
            markersize=6, alpha=0.8, label='Mean Target (Actual)')
    
    # 3. Plot Mean Predictions (Model Output)
    ax.plot(bin_centers, mean_preds, 'o-', color=color_pred, 
            markersize=8, linewidth=2, label='Mean Prediction (Model)')
    
    # Formatting
    ax.set_xlabel('Spread (Predictive Standard Deviation)', fontsize=11)
    ax.set_ylabel('Probability Value', fontsize=11)
    
    if show_title:
        ax.set_title('Class Bias Check: Prediction vs. Target', fontsize=13, fontweight='bold')
    
    # Set y-limits to probability range [0, 1]
    ax.set_ylim(-0.05, 1.05)
    # Match x-limits to the actual spread range
    actual_max_spread = np.nanmax(bin_centers)
    x_limit = min(0.55, actual_max_spread * 1.1) # Buffer, but don't let it fly beyond 0.5
    ax.set_xlim(0, x_limit) #bin_edges[-1]
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', frameon=True)
    
    plt.tight_layout()
    return fig, ax

# --------------------------------------------
# 4. Spread vs RMSE Bar
# --------------------------------------------
def plot_spread_vs_rmse_bar(result_dict, figsize = (8, 5), show_title=True, show_bar_labels=False):
    """
    Plots a bar chart of RMSE vs Spread with an optional color gradient and optional value labels.
    Parameters
    ----------
    result_dict : dict
    color : str
        Bar color

    Returns
    -------
    fig, ax
    """
    rmse = result_dict['rmse_values']
    bin_edges = result_dict['bin_edges_std'].copy()
    
    # Handle inf and bin centers
    if np.isinf(bin_edges[-1]):
        bin_edges[-1] = bin_edges[-2] + (bin_edges[-2] - bin_edges[-3])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)

    fig, ax = plt.subplots(figsize=figsize)
    
    # Color gradient from Green (Low Spread) to Red (High Spread)
    import matplotlib.cm as cm
    colors = cm.RdYlGn_r(np.linspace(0, 1, len(rmse)))
    
    valid = ~np.isnan(rmse)
    bars = ax.bar(bin_centers[valid], rmse[valid], width=widths[valid], 
                  color=colors[valid], edgecolor='black', alpha=0.8)

    # Optional labels on top of bars
    if show_bar_labels:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                    f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Spread (Predictive Standard Deviation)')
    ax.set_ylabel('Actual Skill (RMSE)')
    ax.set_xlim(0, 0.55) # Capping at the logical max for binary classification
    
    if show_title:
        ax.set_title('Error Growth by Uncertainty Level', fontweight='bold')
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig, ax
# --------------------------------------------
# 5. Discard Test
# --------------------------------------------
def plot_discard_test(result_dict, figsize = (7, 5), line_color='#1f77b4', show_example_fraction=True):
    """
    Plots the discard test: error vs discard fraction. Refined visuals with combined legend and no diagnostic text.
    Parameters
    ----------
    result_dict : dict
        Output from `get_discard_test`
    line_color : str
        Color of the error curve
    show_example_fraction : bool
        If True, overlay fraction of examples remaining on secondary y-axis

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    discard_frac = result_dict['discard_fractions']
    error_vals = result_dict['error_values']

    fig, ax1 = plt.subplots(figsize=figsize)
    
    # 1. Primary Plot: Error Curve (Left Y-axis)
    line1, = ax1.plot(discard_frac, error_vals, 'o-', color=line_color, 
                      linewidth=2, markersize=6, label='Error', zorder=5)
    
    ax1.set_xlabel('Discard fraction (most uncertain examples removed)', fontsize=11)
    ax1.set_ylabel('Error (Cross-entropy)', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_title('Discard Test', fontsize=13, fontweight='bold')

    # 2. Secondary Axis: Examples Remaining (Right Y-axis)
    if show_example_fraction:
        example_frac = result_dict['example_fractions']
        ax2 = ax1.twinx()
        # Red dashed line for coverage
        line2, = ax2.plot(discard_frac, example_frac, 's--', color='#d62728', 
                          alpha=0.6, label='Examples left')
        ax2.set_ylabel('Fraction of examples left', color='#d62728', fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis='y', labelcolor='#d62728')
        
        # Merge legends into one box
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', frameon=True)
    else:
        ax1.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    return fig, ax1
# --------------------------------------------
# 6. Reliability Curve
# --------------------------------------------
def plot_reliability_curve(result_dicts, figsize = (8, 8), model_names=None, colors=None):
    """
    Compiles multiple models on a single reliability plot without numerical scores.
    """
    # Ensure result_dicts is a list even if a single dict is passed
    if isinstance(result_dicts, dict):
        result_dicts = [result_dicts]
        
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')

    if colors is None:
        colors = plt.cm.tab10.colors

    for i, rd in enumerate(result_dicts):
        name = model_names[i] if model_names else f"Model {i+1}"
        valid = ~np.isnan(rd['mean_forecast_probs'])
        
        # Removed BSS from the label string
        ax.plot(rd['mean_forecast_probs'][valid], 
                rd['mean_event_frequencies'][valid], 
                'o-', color=colors[i % len(colors)], label=name)

    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('Reliability Comparison')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig, ax

# --------------------------------------------
# 7. Attributes Diagram
# --------------------------------------------
def plot_attributes_diagram(rel_dict, figsize = (9, 9), model_name="Model", figure=None, axes=None):
    """
    Plots a full Attributes Diagram with a clearly labeled Sharpness (Histogram) inset.
    """
    # Extract values from dictionary
    mean_forecast_probs = rel_dict['mean_forecast_probs']
    mean_event_frequencies = rel_dict['mean_event_frequencies']
    num_examples_by_bin = rel_dict['num_examples_by_bin']
    bss = rel_dict['bss']
    climatology = rel_dict['climatology']
        
    if figure is None or axes is None:
        figure, axes = plt.subplots(1, 1, figsize=figsize)
    
    # 1. Reference Lines & Skill Shading (The Background)
    axes.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Reliability')
    axes.axhline(climatology, color='gray', linestyle=':', label=f'Climatology') # avoid the number:  ({climatology:.2f})
    
    no_skill_x = np.linspace(0, 1, 100)
    no_skill_y = 0.5 * (no_skill_x + climatology)
    axes.plot(no_skill_x, no_skill_y, color='green', linestyle='--', alpha=0.3, label='No-Skill Limit')
    axes.fill_between(no_skill_x, no_skill_x, no_skill_y, color='green', alpha=0.05)

    # 2. Main Reliability Line
    valid = ~np.isnan(mean_forecast_probs)
    axes.plot(mean_forecast_probs[valid], mean_event_frequencies[valid], 'r-o', 
              linewidth=2.5, markersize=8, label=f'{model_name}') # avoid the BSS number (BSS: {bss:.3f})
    
    # 3. IMPROVED INSET: Sharpness Histogram
    # Positioned to not block the main diagonal
    ax_hist = axes.inset_axes([0.62, 0.08, 0.33, 0.22]) 
    
    bin_edges = np.linspace(0, 1, len(num_examples_by_bin) + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Use percentage instead of raw counts for better interpretability
    bin_percentages = (num_examples_by_bin / np.sum(num_examples_by_bin)) * 100
    
    ax_hist.bar(bin_centers, bin_percentages, width=(1/len(bin_centers)), 
                color='gray', edgecolor='white', alpha=0.6)
    
    # Adding specific inset labels
    ax_hist.set_title('Sharpness (Distribution)', fontsize=10, fontweight='bold')
    ax_hist.set_ylabel('% Data', fontsize=8)
    ax_hist.set_xlabel('Confidence', fontsize=8)
    ax_hist.tick_params(labelsize=7)

    axes.set_xlabel('Forecast Probability (Model Confidence)', fontsize=11)
    axes.set_ylabel('Observed Frequency', fontsize=11)
    axes.legend(loc='upper left', frameon=True)
    axes.grid(True, alpha=0.2)
    
    return figure, axes

# --------------------------------------------
# 8. ROC Curve with UQ
# --------------------------------------------
def plot_roc_with_uq(roc_result_dict, figsize=(6,6), title='ROC Curve with UQ', showTitle=True):
    """
    Plots ROC curve for all examples and optionally for low/high uncertainty subsets.

    Parameters
    ----------
    roc_result_dict : dict
        Output from `get_roc_with_uq`
    figsize : tuple
        Size of the figure
    title : str
        Title of the plot

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(roc_result_dict['fpr_all'], roc_result_dict['tpr_all'], color='blue', linewidth=2,
            label=f'All (AUC={roc_result_dict["auc_all"]:.3f})')

    if roc_result_dict.get('fpr_low') is not None:
        ax.plot(roc_result_dict['fpr_low'], roc_result_dict['tpr_low'], color='green', linestyle='--', linewidth=2,
                label=f'Low uncertainty (AUC={roc_result_dict["auc_low"]:.3f})')
    if roc_result_dict.get('fpr_high') is not None:
        ax.plot(roc_result_dict['fpr_high'], roc_result_dict['tpr_high'], color='red', linestyle='--', linewidth=2,
                label=f'High uncertainty (AUC={roc_result_dict["auc_high"]:.3f})')

    ax.plot([0,1], [0,1], color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('False Positive Rate (POFD)')
    ax.set_ylabel('True Positive Rate (POD)')
    ax.grid(True)
    ax.legend(loc='lower right')
    if showTitle:
        ax.set_title(title)
    return fig, ax

# --------------------------------------------
# 9. Performance Diagram with UQ
# --------------------------------------------
def plot_perf_diagram_with_uq(perf_result_dict, figsize=(6,6), title='Performance Diagram with UQ', showTitle=True):
    """
    Plots a performance diagram with optional low/high uncertainty subsets.

    Parameters
    ----------
    perf_result_dict : dict
        Output from `get_perf_diagram_with_uq`
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(perf_result_dict['sr_all'], perf_result_dict['pod_all'], color='blue', linewidth=2, label='All examples')
    if perf_result_dict.get('sr_low') is not None:
        ax.plot(perf_result_dict['sr_low'], perf_result_dict['pod_low'], color='green', linestyle='--', linewidth=2, label='Low uncertainty')
    if perf_result_dict.get('sr_high') is not None:
        ax.plot(perf_result_dict['sr_high'], perf_result_dict['pod_high'], color='red', linestyle='--', linewidth=2, label='High uncertainty')

    ax.plot([0,1], [0,1], color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Success Ratio (Precision)')
    ax.set_ylabel('POD (Recall / Probability of Detection)')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.grid(True)
    ax.legend(loc='lower left')
    if showTitle:
        ax.set_title(title)
    return fig, ax
